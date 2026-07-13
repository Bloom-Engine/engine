//! Jolt-backed physics world for Bloom Engine.
//!
//! Wraps `jolt_sys` behind f64 handle registries so Perry FFI can speak to
//! Jolt using only scalar values.
//!
//! Handle conventions:
//!   - bloom handles are 1-based f64 indices into HandleRegistry
//!   - bj_world, bj_shape, bj_body, bj_constraint are native Jolt handles
//!   - Body/constraint handles are scoped to a single world internally
//!
//! Thread-safety: Jolt's contact listener runs on job threads; the queue it
//! fills is already mutex-protected inside `bloom_jolt.cpp`. Everything in
//! this file is single-threaded (matches Bloom's engine-main-thread model).

use crate::handles::HandleRegistry;
use crate::jolt_sys::*;

use std::sync::Once;

static GLOBAL_INIT: Once = Once::new();

/// Ensures `bj_global_init` is called exactly once per process.
pub fn ensure_jolt_initialised() {
    GLOBAL_INIT.call_once(|| {
        // SAFETY: called exactly once; bj_global_init is refcounted and idempotent.
        let r = unsafe { bj_global_init() };
        assert_eq!(r, BjResult::Ok, "Jolt global init failed");
    });
}

// ============================================================
//  Cached query results (multi-call read-back pattern)
// ============================================================

#[derive(Default)]
struct RayHitCache {
    hits: Vec<BjRayHit>,
}

#[derive(Default)]
struct OverlapCache {
    bodies: Vec<bj_body>,
}

#[derive(Default)]
struct ContactCache {
    /// Drained from Jolt's queue at the start of each read cycle.
    events: Vec<BjContact>,
}

// ============================================================
//  JoltPhysics — owned by EngineState, indexed by f64 handles
// ============================================================

pub struct JoltPhysics {
    /// Usually a single world; supporting a registry keeps the API symmetric.
    worlds: HandleRegistry<bj_world>,
    /// Shapes live across worlds (refcounted in Jolt). Stored as raw pointer values.
    shapes: HandleRegistry<bj_shape>,
    /// Bodies are world-scoped. We remember which world they belong to.
    bodies: HandleRegistry<(bj_world, bj_body)>,
    /// Constraints similarly scoped.
    constraints: HandleRegistry<(bj_world, bj_constraint)>,
    /// Character controllers — world-scoped like bodies/constraints.
    characters: HandleRegistry<(bj_world, bj_character)>,
    /// Wheeled vehicles — world-scoped; stores bj_vehicle + the chassis body
    /// handle it mapped to (for TS-side getChassis()).
    vehicles: HandleRegistry<(bj_world, bj_vehicle, f64 /*chassis body handle*/)>,

    ray_hit_cache: RayHitCache,
    overlap_cache: OverlapCache,
    contact_cache: ContactCache,

    /// Scratch streams for variable-size shape inputs (convex hull points,
    /// mesh vertices/indices, heightfield samples). The TS layer pushes
    /// values via `bloom_physics_scratch_push_*` then calls the shape factory.
    scratch_f32: Vec<f32>,
    scratch_u32: Vec<u32>,
    /// Compound-shape builder state — cleared by compound_begin, extended
    /// by compound_add_child, consumed by compound_end.
    compound_children: Vec<(bj_shape, BjTransform)>,

    /// Per-world fixed-timestep state, keyed by the raw bj_world handle.
    /// Created lazily on the first step_fixed / set_fixed_timestep call;
    /// worlds stepped only via the variable-rate `step()` never allocate
    /// one.
    step_states: std::collections::HashMap<u64, WorldStepState>,
}

/// Fixed-timestep accumulator + interpolation state for one world.
///
/// Variable-dt stepping feeds frame hitches straight into the solver
/// (tunneling, constraint explosions); the accumulator decouples
/// simulation rate from render rate the standard way: clamp the incoming
/// frame dt, simulate N whole fixed steps, carry the remainder, and
/// expose `alpha = remainder / fixed_dt` so rendering can interpolate
/// between the last two physics states.
struct WorldStepState {
    fixed_dt: f32,
    /// Spiral-of-death guard: at most this many fixed steps per frame.
    /// When the cap hits, the surplus backlog is dropped — the simulation
    /// slows down instead of feeding ever-longer frames back into itself.
    max_steps_per_frame: u32,
    accumulator: f32,
    /// remainder / fixed_dt after the latest step_fixed, in [0, 1).
    alpha: f32,
    /// When true, body position/rotation getters blend between the state
    /// snapshot taken before the latest step batch and the current state.
    /// Physics queries (raycast/overlap) always see the real simulation
    /// state regardless.
    interpolate: bool,
    /// bj_body → (position, rotation) captured before the latest batch.
    prev: std::collections::HashMap<u64, (BjVec3, BjQuat)>,
}

impl Default for WorldStepState {
    fn default() -> Self {
        Self {
            fixed_dt: 1.0 / 60.0,
            max_steps_per_frame: 4,
            accumulator: 0.0,
            alpha: 1.0,
            interpolate: false,
            prev: std::collections::HashMap::new(),
        }
    }
}

/// Hard ceiling on a single frame's contribution to the accumulator.
/// A debugger pause or OS hitch produces one slowed-down frame instead of
/// minutes of simulated catch-up.
const MAX_FRAME_DT: f32 = 0.25;

impl Default for JoltPhysics {
    fn default() -> Self { Self::new() }
}

impl JoltPhysics {
    pub fn new() -> Self {
        ensure_jolt_initialised();
        Self {
            worlds: HandleRegistry::new(),
            shapes: HandleRegistry::new(),
            bodies: HandleRegistry::new(),
            constraints: HandleRegistry::new(),
            characters: HandleRegistry::new(),
            vehicles: HandleRegistry::new(),
            ray_hit_cache: RayHitCache::default(),
            overlap_cache: OverlapCache::default(),
            contact_cache: ContactCache::default(),
            scratch_f32: Vec::with_capacity(1024),
            scratch_u32: Vec::with_capacity(512),
            compound_children: Vec::with_capacity(16),
            step_states: std::collections::HashMap::new(),
        }
    }

    // -------------------------------------------------------- Scratch buffer

    pub fn scratch_reset(&mut self) {
        self.scratch_f32.clear();
        self.scratch_u32.clear();
    }
    pub fn scratch_push_f32(&mut self, v: f32) { self.scratch_f32.push(v); }
    pub fn scratch_push_u32(&mut self, v: u32) { self.scratch_u32.push(v); }

    pub fn shape_convex_hull_from_scratch(&mut self, num_points: u32, convex_radius: f32) -> f64 {
        let need = (num_points * 3) as usize;
        if self.scratch_f32.len() < need || num_points < 3 { return 0.0; }
        let points: Vec<BjVec3> = (0..num_points as usize).map(|i| BjVec3 {
            x: self.scratch_f32[i * 3],
            y: self.scratch_f32[i * 3 + 1],
            z: self.scratch_f32[i * 3 + 2],
        }).collect();
        self.create_convex_hull_shape(&points, convex_radius)
    }

    pub fn shape_mesh_from_scratch(&mut self, vertex_count: u32, triangle_count: u32) -> f64 {
        let need_f = (vertex_count * 3) as usize;
        let need_u = (triangle_count * 3) as usize;
        if self.scratch_f32.len() < need_f || self.scratch_u32.len() < need_u { return 0.0; }
        if vertex_count == 0 || triangle_count == 0 { return 0.0; }
        let vertices: Vec<BjVec3> = (0..vertex_count as usize).map(|i| BjVec3 {
            x: self.scratch_f32[i * 3],
            y: self.scratch_f32[i * 3 + 1],
            z: self.scratch_f32[i * 3 + 2],
        }).collect();
        let indices: Vec<u32> = self.scratch_u32[..need_u].to_vec();
        self.create_mesh_shape(&vertices, &indices)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn shape_heightfield_from_scratch(
        &mut self, sample_count: u32,
        ox: f32, oy: f32, oz: f32, sx: f32, sy: f32, sz: f32, block_size: u32,
    ) -> f64 {
        let need = (sample_count * sample_count) as usize;
        if self.scratch_f32.len() < need || sample_count < 2 { return 0.0; }
        let samples = self.scratch_f32[..need].to_vec();
        self.create_heightfield_shape(
            &samples, sample_count,
            BjVec3 { x: ox, y: oy, z: oz },
            BjVec3 { x: sx, y: sy, z: sz },
            block_size,
        )
    }

    pub fn compound_begin(&mut self) {
        self.compound_children.clear();
    }
    #[allow(clippy::too_many_arguments)]
    pub fn compound_add_child(&mut self,
        shape_h: f64,
        px: f32, py: f32, pz: f32,
        rx: f32, ry: f32, rz: f32, rw: f32,
    ) {
        if let Some(&shape) = self.shapes.get(shape_h) {
            self.compound_children.push((shape, BjTransform {
                position: BjVec3 { x: px, y: py, z: pz },
                rotation: BjQuat { x: rx, y: ry, z: rz, w: rw },
            }));
        }
    }
    pub fn compound_end(&mut self) -> f64 {
        if self.compound_children.is_empty() { return 0.0; }
        let shapes: Vec<bj_shape> = self.compound_children.iter().map(|(s, _)| *s).collect();
        let xforms: Vec<BjTransform> = self.compound_children.iter().map(|(_, x)| *x).collect();
        let s = unsafe {
            bj_shape_compound_static(shapes.as_ptr(), xforms.as_ptr(), shapes.len() as u32)
        };
        self.compound_children.clear();
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    // -------------------------------------------------------- World

    pub fn create_world(
        &mut self,
        gx: f32, gy: f32, gz: f32,
        max_bodies: u32, num_threads: u32,
    ) -> f64 {
        let desc = BjWorldDesc {
            gravity: BjVec3 { x: gx, y: gy, z: gz },
            max_bodies: if max_bodies > 0 { max_bodies } else { 65536 },
            num_threads,
            ..Default::default()
        };
        let world = unsafe { bj_world_create(&desc) };
        if world == BJ_INVALID { return 0.0; }
        self.worlds.alloc(world)
    }

    pub fn destroy_world(&mut self, handle: f64) {
        if let Some(world) = self.worlds.free(handle) {
            // Destroy bodies and constraints tied to this world first.
            // (Iterate and collect to avoid borrow conflicts.)
            let body_handles: Vec<f64> = self.bodies.iter()
                .filter(|(_, (w, _))| *w == world)
                .map(|(h, _)| h)
                .collect();
            for h in body_handles {
                if let Some((w, b)) = self.bodies.free(h) {
                    unsafe { bj_body_destroy(w, b); }
                }
            }
            let c_handles: Vec<f64> = self.constraints.iter()
                .filter(|(_, (w, _))| *w == world)
                .map(|(h, _)| h)
                .collect();
            for h in c_handles {
                if let Some((w, c)) = self.constraints.free(h) {
                    unsafe { bj_constraint_destroy(w, c); }
                }
            }
            self.step_states.remove(&world);
            unsafe { bj_world_destroy(world); }
        }
    }

    pub fn set_gravity(&self, world_h: f64, x: f32, y: f32, z: f32) {
        if let Some(&world) = self.worlds.get(world_h) {
            unsafe { bj_world_set_gravity(world, BjVec3 { x, y, z }); }
        }
    }

    pub fn get_gravity_axis(&self, world_h: f64, axis: u32) -> f64 {
        if let Some(&world) = self.worlds.get(world_h) {
            let mut g = BjVec3::default();
            unsafe { bj_world_get_gravity(world, &mut g); }
            return match axis { 0 => g.x, 1 => g.y, 2 => g.z, _ => 0.0 } as f64;
        }
        0.0
    }

    pub fn optimize_broadphase(&self, world_h: f64) {
        if let Some(&world) = self.worlds.get(world_h) {
            unsafe { bj_world_optimize_broadphase(world); }
        }
    }

    pub fn step(&mut self, world_h: f64, dt: f32, collision_steps: u32) {
        if let Some(&world) = self.worlds.get(world_h) {
            unsafe { bj_world_step(world, dt, collision_steps.max(1)); }
            self.drain_contacts(world);
        }
    }

    /// Drain the shim's queued contact events into our cache so they
    /// survive across queries. The shim queue accumulates across multiple
    /// bj_world_step calls, so one drain after a fixed-step batch captures
    /// the whole batch's events.
    fn drain_contacts(&mut self, world: bj_world) {
        let count = unsafe { bj_world_contact_count(world) };
        if count > 0 {
            self.contact_cache.events.resize(count as usize, unsafe { std::mem::zeroed() });
            let drained = unsafe {
                bj_world_pop_contacts(world, self.contact_cache.events.as_mut_ptr(), count)
            };
            self.contact_cache.events.truncate(drained as usize);
        } else {
            self.contact_cache.events.clear();
        }
    }

    /// Advance the world by whole fixed steps, carrying the remainder.
    /// Returns the interpolation alpha in [0, 1): how far the carried
    /// remainder sits between the last simulated state and the next.
    ///
    /// This is the default stepping mode for the TS `physics.step()` API;
    /// the variable-rate `step()` above remains as an explicit opt-out.
    pub fn step_fixed(&mut self, world_h: f64, frame_dt: f32, collision_steps: u32) -> f32 {
        let Some(&world) = self.worlds.get(world_h) else { return 1.0 };
        let dt = if frame_dt.is_finite() { frame_dt.clamp(0.0, MAX_FRAME_DT) } else { 0.0 };

        let st = self.step_states.entry(world).or_default();
        st.accumulator += dt;
        let fixed = st.fixed_dt;
        let mut steps = (st.accumulator / fixed) as u32;
        if steps > st.max_steps_per_frame {
            steps = st.max_steps_per_frame;
            // Drop the surplus backlog (keep at most one extra step's worth
            // so alpha stays meaningful) — slow down, don't spiral.
            st.accumulator = st.accumulator.min(fixed * (steps as f32 + 1.0));
        }
        let interpolate = st.interpolate;

        if steps > 0 {
            // Interpolation blends between the two most recent simulated
            // states, so the snapshot is taken before the LAST step of the
            // batch — snapshotting before the whole batch would render a
            // backward jump on multi-step catch-up frames.
            let mut snap = None;
            for i in 0..steps {
                if interpolate && i + 1 == steps {
                    snap = Some(self.snapshot_world_bodies(world));
                }
                unsafe { bj_world_step(world, fixed, collision_steps.max(1)); }
            }
            if let Some(s) = snap {
                self.step_states.get_mut(&world).unwrap().prev = s;
            }
            self.drain_contacts(world);
        }

        let st = self.step_states.get_mut(&world).unwrap();
        st.accumulator -= steps as f32 * fixed;
        st.alpha = (st.accumulator / fixed).clamp(0.0, 1.0);
        st.alpha
    }

    /// Configure the fixed step rate (`hz`, e.g. 60.0) and the per-frame
    /// catch-up cap for a world. Values <= 0 keep the current setting.
    pub fn set_fixed_timestep(&mut self, world_h: f64, hz: f32, max_steps: u32) {
        let Some(&world) = self.worlds.get(world_h) else { return };
        let st = self.step_states.entry(world).or_default();
        if hz > 0.0 && hz.is_finite() {
            st.fixed_dt = 1.0 / hz;
        }
        if max_steps > 0 {
            st.max_steps_per_frame = max_steps;
        }
    }

    /// Enable/disable render interpolation for a world's body transform
    /// getters. Off by default (getters return the raw simulation state).
    pub fn set_interpolation(&mut self, world_h: f64, on: bool) {
        let Some(&world) = self.worlds.get(world_h) else { return };
        let st = self.step_states.entry(world).or_default();
        st.interpolate = on;
        if !on {
            st.prev.clear();
            st.alpha = 1.0;
        }
    }

    /// Interpolation alpha from the most recent step_fixed (1.0 when the
    /// world has never been fixed-stepped).
    pub fn step_alpha(&self, world_h: f64) -> f32 {
        self.worlds
            .get(world_h)
            .and_then(|w| self.step_states.get(w))
            .map(|st| st.alpha)
            .unwrap_or(1.0)
    }

    /// Capture (position, rotation) of every body in `world`.
    fn snapshot_world_bodies(&self, world: bj_world) -> std::collections::HashMap<u64, (BjVec3, BjQuat)> {
        self.bodies
            .iter()
            .filter(|(_, &(w, _))| w == world)
            .map(|(_, &(w, b))| {
                let mut v = BjVec3 { x: 0.0, y: 0.0, z: 0.0 };
                let mut q = BjQuat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
                unsafe {
                    bj_body_get_position(w, b, &mut v);
                    bj_body_get_rotation(w, b, &mut q);
                }
                (b, (v, q))
            })
            .collect()
    }

    /// Blend factor + previous state for a body, when its world has
    /// interpolation enabled and a fresh snapshot. None → return raw state.
    fn interp_prev(&self, world: bj_world, body: bj_body) -> Option<(f32, &(BjVec3, BjQuat))> {
        let st = self.step_states.get(&world)?;
        if !st.interpolate || st.alpha >= 1.0 {
            return None;
        }
        st.prev.get(&body).map(|p| (st.alpha, p))
    }

    pub fn set_layer_collides(&self, world_h: f64, a: u32, b: u32, collides: bool) {
        if let Some(&world) = self.worlds.get(world_h) {
            unsafe { bj_world_set_layer_collides(world, a, b, collides as u8); }
        }
    }

    pub fn get_layer_collides(&self, world_h: f64, a: u32, b: u32) -> bool {
        if let Some(&world) = self.worlds.get(world_h) {
            return unsafe { bj_world_get_layer_collides(world, a, b) } != 0;
        }
        false
    }

    pub fn body_count(&self, world_h: f64) -> u32 {
        if let Some(&world) = self.worlds.get(world_h) {
            return unsafe { bj_world_body_count(world) };
        }
        0
    }

    pub fn active_body_count(&self, world_h: f64) -> u32 {
        if let Some(&world) = self.worlds.get(world_h) {
            return unsafe { bj_world_active_body_count(world) };
        }
        0
    }

    // -------------------------------------------------------- Shapes

    pub fn create_box_shape(&mut self, hx: f32, hy: f32, hz: f32, convex_radius: f32) -> f64 {
        let s = unsafe { bj_shape_box(BjVec3 { x: hx, y: hy, z: hz }, convex_radius) };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_sphere_shape(&mut self, radius: f32) -> f64 {
        let s = unsafe { bj_shape_sphere(radius) };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_capsule_shape(&mut self, half_height: f32, radius: f32) -> f64 {
        let s = unsafe { bj_shape_capsule(half_height, radius) };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_cylinder_shape(&mut self, half_height: f32, radius: f32, convex_radius: f32) -> f64 {
        let s = unsafe { bj_shape_cylinder(half_height, radius, convex_radius) };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_convex_hull_shape(&mut self, points: &[BjVec3], convex_radius: f32) -> f64 {
        let s = unsafe {
            bj_shape_convex_hull(points.as_ptr(), points.len() as u32, convex_radius)
        };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_mesh_shape(&mut self, vertices: &[BjVec3], indices: &[u32]) -> f64 {
        debug_assert!(indices.len() % 3 == 0);
        let s = unsafe {
            bj_shape_mesh(
                vertices.as_ptr(), vertices.len() as u32,
                indices.as_ptr(), (indices.len() / 3) as u32,
            )
        };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_heightfield_shape(
        &mut self,
        samples: &[f32], sample_count: u32,
        offset: BjVec3, scale: BjVec3, block_size: u32,
    ) -> f64 {
        let s = unsafe {
            bj_shape_heightfield(
                samples.as_ptr(), sample_count, offset, scale, block_size,
            )
        };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_scaled_shape(&mut self, base_h: f64, sx: f32, sy: f32, sz: f32) -> f64 {
        let base = match self.shapes.get(base_h) { Some(&s) => s, None => return 0.0 };
        let s = unsafe { bj_shape_scaled(base, BjVec3 { x: sx, y: sy, z: sz }) };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn create_offset_com_shape(&mut self, base_h: f64, ox: f32, oy: f32, oz: f32) -> f64 {
        let base = match self.shapes.get(base_h) { Some(&s) => s, None => return 0.0 };
        let s = unsafe { bj_shape_offset_com(base, BjVec3 { x: ox, y: oy, z: oz }) };
        if s == BJ_INVALID { 0.0 } else { self.shapes.alloc(s) }
    }

    pub fn release_shape(&mut self, handle: f64) {
        if let Some(s) = self.shapes.free(handle) {
            unsafe { bj_shape_release(s); }
        }
    }

    pub fn shape_bounds_axis(&self, handle: f64, axis: u32) -> f64 {
        if let Some(&s) = self.shapes.get(handle) {
            let mut mn = BjVec3::default();
            let mut mx = BjVec3::default();
            unsafe { bj_shape_get_local_bounds(s, &mut mn, &mut mx); }
            return match axis {
                0 => mn.x, 1 => mn.y, 2 => mn.z,
                3 => mx.x, 4 => mx.y, 5 => mx.z,
                _ => 0.0,
            } as f64;
        }
        0.0
    }

    pub fn shape_volume(&self, handle: f64) -> f32 {
        self.shapes.get(handle).map(|&s| unsafe { bj_shape_get_volume(s) }).unwrap_or(0.0)
    }

    // -------------------------------------------------------- Bodies

    /// Full body descriptor — passed as flat scalars by the FFI wrapper.
    #[allow(clippy::too_many_arguments)]
    pub fn create_body(
        &mut self,
        world_h: f64, shape_h: f64,
        motion_type: u32,
        px: f32, py: f32, pz: f32,
        rx: f32, ry: f32, rz: f32, rw: f32,
        vx: f32, vy: f32, vz: f32,
        wx: f32, wy: f32, wz: f32,
        object_layer: u32,
        is_sensor: bool, allow_sleeping: bool, use_ccd: bool, start_awake: bool,
        friction: f32, restitution: f32,
        lin_damp: f32, ang_damp: f32, gravity_factor: f32,
        mass_override: f32,
        ix: f32, iy: f32, iz: f32,
        user_data: u64,
    ) -> f64 {
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return 0.0 };
        let shape = match self.shapes.get(shape_h) { Some(&s) => s, None => return 0.0 };

        let desc = BjBodyDesc {
            motion_type: match motion_type {
                0 => BjMotionType::Static,
                1 => BjMotionType::Kinematic,
                _ => BjMotionType::Dynamic,
            },
            position: BjVec3 { x: px, y: py, z: pz },
            rotation: BjQuat { x: rx, y: ry, z: rz, w: rw },
            linear_velocity:  BjVec3 { x: vx, y: vy, z: vz },
            angular_velocity: BjVec3 { x: wx, y: wy, z: wz },
            gravity_factor,
            linear_damping:   lin_damp,
            angular_damping:  ang_damp,
            friction,
            restitution,
            mass_override,
            inertia_diag_override: BjVec3 { x: ix, y: iy, z: iz },
            object_layer,
            is_sensor:       is_sensor as u8,
            allow_sleeping:  allow_sleeping as u8,
            use_ccd:         use_ccd as u8,
            start_awake:     start_awake as u8,
            user_data,
        };
        let body = unsafe { bj_body_create(world, shape, &desc) };
        if body == BJ_INVALID { 0.0 } else { self.bodies.alloc((world, body)) }
    }

    pub fn destroy_body(&mut self, handle: f64) {
        if let Some((world, body)) = self.bodies.free(handle) {
            unsafe { bj_body_destroy(world, body); }
        }
    }

    fn resolve_body(&self, h: f64) -> Option<(bj_world, bj_body)> {
        self.bodies.get(h).copied()
    }

    pub fn body_activate(&self, h: f64) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_activate(w, b); } }
    }
    pub fn body_deactivate(&self, h: f64) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_deactivate(w, b); } }
    }
    pub fn body_is_active(&self, h: f64) -> bool {
        self.resolve_body(h).map(|(w, b)| unsafe { bj_body_is_active(w, b) } != 0).unwrap_or(false)
    }
    pub fn body_is_valid(&self, h: f64) -> bool {
        self.resolve_body(h).map(|(w, b)| unsafe { bj_body_is_valid(w, b) } != 0).unwrap_or(false)
    }

    /// Position for rendering. When the world has interpolation enabled
    /// (set_interpolation), blends between the last two simulated states
    /// using the step_fixed remainder; otherwise the raw simulation state.
    pub fn body_get_position_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, b)) = self.resolve_body(h) {
            let mut v = BjVec3::default();
            unsafe { bj_body_get_position(w, b, &mut v); }
            if let Some((a, (pv, _))) = self.interp_prev(w, b) {
                v = BjVec3 {
                    x: pv.x + (v.x - pv.x) * a,
                    y: pv.y + (v.y - pv.y) * a,
                    z: pv.z + (v.z - pv.z) * a,
                };
            }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    /// Rotation for rendering — see body_get_position_axis. Interpolation
    /// is a normalized lerp with shortest-path sign handling (nlerp): for
    /// the sub-17ms rotation deltas a 60 Hz step produces, nlerp is
    /// visually identical to slerp and considerably cheaper.
    pub fn body_get_rotation_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, b)) = self.resolve_body(h) {
            let mut q = BjQuat::default();
            unsafe { bj_body_get_rotation(w, b, &mut q); }
            if let Some((a, (_, pq))) = self.interp_prev(w, b) {
                // shortest path: flip the previous quat if the hemispheres differ
                let dot = pq.x * q.x + pq.y * q.y + pq.z * q.z + pq.w * q.w;
                let s = if dot < 0.0 { -1.0 } else { 1.0 };
                let mut r = BjQuat {
                    x: pq.x * s + (q.x - pq.x * s) * a,
                    y: pq.y * s + (q.y - pq.y * s) * a,
                    z: pq.z * s + (q.z - pq.z * s) * a,
                    w: pq.w * s + (q.w - pq.w * s) * a,
                };
                let len = (r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w).sqrt();
                if len > 1e-6 {
                    r = BjQuat { x: r.x / len, y: r.y / len, z: r.z / len, w: r.w / len };
                }
                q = r;
            }
            return match axis { 0 => q.x, 1 => q.y, 2 => q.z, 3 => q.w, _ => 0.0 } as f64;
        }
        0.0
    }

    pub fn body_set_position(&self, h: f64, x: f32, y: f32, z: f32, activate: bool) {
        if let Some((w, b)) = self.resolve_body(h) {
            unsafe { bj_body_set_position(w, b, BjVec3 { x, y, z }, if activate { BjActivation::Activate } else { BjActivation::DontActivate }); }
        }
    }
    pub fn body_set_rotation(&self, h: f64, x: f32, y: f32, z: f32, ww: f32, activate: bool) {
        if let Some((w, b)) = self.resolve_body(h) {
            unsafe { bj_body_set_rotation(w, b, BjQuat { x, y, z, w: ww }, if activate { BjActivation::Activate } else { BjActivation::DontActivate }); }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn body_set_transform(&self, h: f64, px: f32, py: f32, pz: f32, rx: f32, ry: f32, rz: f32, rw: f32, activate: bool) {
        if let Some((w, b)) = self.resolve_body(h) {
            let x = BjTransform { position: BjVec3 { x: px, y: py, z: pz }, rotation: BjQuat { x: rx, y: ry, z: rz, w: rw } };
            unsafe { bj_body_set_transform(w, b, &x, if activate { BjActivation::Activate } else { BjActivation::DontActivate }); }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn body_move_kinematic(&self, h: f64, px: f32, py: f32, pz: f32, rx: f32, ry: f32, rz: f32, rw: f32, dt: f32) {
        if let Some((w, b)) = self.resolve_body(h) {
            let target = BjTransform { position: BjVec3 { x: px, y: py, z: pz }, rotation: BjQuat { x: rx, y: ry, z: rz, w: rw } };
            unsafe { bj_body_move_kinematic(w, b, &target, dt); }
        }
    }

    pub fn body_get_linear_velocity_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, b)) = self.resolve_body(h) {
            let mut v = BjVec3::default();
            unsafe { bj_body_get_linear_velocity(w, b, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn body_get_angular_velocity_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, b)) = self.resolve_body(h) {
            let mut v = BjVec3::default();
            unsafe { bj_body_get_angular_velocity(w, b, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn body_get_point_velocity_axis(&self, h: f64, px: f32, py: f32, pz: f32, axis: u32) -> f64 {
        if let Some((w, b)) = self.resolve_body(h) {
            let mut v = BjVec3::default();
            unsafe { bj_body_get_point_velocity(w, b, BjVec3 { x: px, y: py, z: pz }, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn body_set_linear_velocity(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_linear_velocity(w, b, BjVec3 { x, y, z }); } }
    }
    pub fn body_set_angular_velocity(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_angular_velocity(w, b, BjVec3 { x, y, z }); } }
    }

    pub fn body_add_force(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_add_force(w, b, BjVec3 { x, y, z }); } }
    }
    pub fn body_add_impulse(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_add_impulse(w, b, BjVec3 { x, y, z }); } }
    }
    pub fn body_add_torque(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_add_torque(w, b, BjVec3 { x, y, z }); } }
    }
    pub fn body_add_angular_impulse(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_add_angular_impulse(w, b, BjVec3 { x, y, z }); } }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn body_add_force_at(&self, h: f64, fx: f32, fy: f32, fz: f32, px: f32, py: f32, pz: f32) {
        if let Some((w, b)) = self.resolve_body(h) {
            unsafe { bj_body_add_force_at(w, b, BjVec3 { x: fx, y: fy, z: fz }, BjVec3 { x: px, y: py, z: pz }); }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn body_add_impulse_at(&self, h: f64, ix: f32, iy: f32, iz: f32, px: f32, py: f32, pz: f32) {
        if let Some((w, b)) = self.resolve_body(h) {
            unsafe { bj_body_add_impulse_at(w, b, BjVec3 { x: ix, y: iy, z: iz }, BjVec3 { x: px, y: py, z: pz }); }
        }
    }

    pub fn body_set_friction(&self, h: f64, v: f32) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_friction(w, b, v); } } }
    pub fn body_set_restitution(&self, h: f64, v: f32) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_restitution(w, b, v); } } }
    pub fn body_set_linear_damping(&self, h: f64, v: f32) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_linear_damping(w, b, v); } } }
    pub fn body_set_angular_damping(&self, h: f64, v: f32) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_angular_damping(w, b, v); } } }
    pub fn body_set_gravity_factor(&self, h: f64, v: f32) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_gravity_factor(w, b, v); } } }
    pub fn body_set_ccd(&self, h: f64, enabled: bool) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_ccd(w, b, enabled as u8); } } }
    pub fn body_set_motion_type(&self, h: f64, t: u32, activate: bool) {
        if let Some((w, b)) = self.resolve_body(h) {
            let mt = match t { 0 => BjMotionType::Static, 1 => BjMotionType::Kinematic, _ => BjMotionType::Dynamic };
            unsafe { bj_body_set_motion_type(w, b, mt, if activate { BjActivation::Activate } else { BjActivation::DontActivate }); }
        }
    }
    pub fn body_set_object_layer(&self, h: f64, layer: u32) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_object_layer(w, b, layer); } } }
    pub fn body_set_is_sensor(&self, h: f64, enabled: bool) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_is_sensor(w, b, enabled as u8); } } }
    pub fn body_set_allow_sleeping(&self, h: f64, enabled: bool) { if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_allow_sleeping(w, b, enabled as u8); } } }
    pub fn body_set_shape(&self, h: f64, shape_h: f64, update_mass: bool, activate: bool) {
        if let (Some((w, b)), Some(&s)) = (self.resolve_body(h), self.shapes.get(shape_h)) {
            unsafe { bj_body_set_shape(w, b, s, update_mass as u8, if activate { BjActivation::Activate } else { BjActivation::DontActivate }); }
        }
    }
    pub fn body_lock_rotation_axes(&self, h: f64, x: bool, y: bool, z: bool) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_lock_rotation_axes(w, b, x as u8, y as u8, z as u8); } }
    }
    pub fn body_lock_translation_axes(&self, h: f64, x: bool, y: bool, z: bool) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_lock_translation_axes(w, b, x as u8, y as u8, z as u8); } }
    }

    pub fn body_get_mass(&self, h: f64) -> f32 { self.resolve_body(h).map(|(w, b)| unsafe { bj_body_get_mass(w, b) }).unwrap_or(0.0) }
    pub fn body_get_friction(&self, h: f64) -> f32 { self.resolve_body(h).map(|(w, b)| unsafe { bj_body_get_friction(w, b) }).unwrap_or(0.0) }
    pub fn body_get_restitution(&self, h: f64) -> f32 { self.resolve_body(h).map(|(w, b)| unsafe { bj_body_get_restitution(w, b) }).unwrap_or(0.0) }
    pub fn body_get_object_layer(&self, h: f64) -> u32 { self.resolve_body(h).map(|(w, b)| unsafe { bj_body_get_object_layer(w, b) }).unwrap_or(0) }

    pub fn body_set_user_data(&self, h: f64, user_data: u64) {
        if let Some((w, b)) = self.resolve_body(h) { unsafe { bj_body_set_user_data(w, b, user_data); } }
    }
    pub fn body_get_user_data(&self, h: f64) -> u64 {
        self.resolve_body(h).map(|(w, b)| unsafe { bj_body_get_user_data(w, b) }).unwrap_or(0)
    }

    // -------------------------------------------------------- Queries

    /// Single-hit raycast. Stores the hit (if any) at cache index 0.
    #[allow(clippy::too_many_arguments)]
    pub fn raycast_closest(
        &mut self, world_h: f64,
        ox: f32, oy: f32, oz: f32,
        dx: f32, dy: f32, dz: f32,
        max_distance: f32, layer_mask: u32,
    ) -> bool {
        self.ray_hit_cache.hits.clear();
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return false };
        let mut hit: BjRayHit = unsafe { std::mem::zeroed() };
        let any = unsafe {
            bj_query_raycast_closest(world,
                BjVec3 { x: ox, y: oy, z: oz },
                BjVec3 { x: dx, y: dy, z: dz },
                max_distance, layer_mask, &mut hit)
        };
        if any != 0 {
            self.ray_hit_cache.hits.push(hit);
            true
        } else { false }
    }

    /// Multi-hit raycast. Stores up to max_hits in the cache.
    #[allow(clippy::too_many_arguments)]
    pub fn raycast_all(
        &mut self, world_h: f64,
        ox: f32, oy: f32, oz: f32,
        dx: f32, dy: f32, dz: f32,
        max_distance: f32, layer_mask: u32, max_hits: u32,
    ) -> u32 {
        self.ray_hit_cache.hits.clear();
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return 0 };
        self.ray_hit_cache.hits.resize(max_hits as usize, unsafe { std::mem::zeroed() });
        let n = unsafe {
            bj_query_raycast_all(world,
                BjVec3 { x: ox, y: oy, z: oz },
                BjVec3 { x: dx, y: dy, z: dz },
                max_distance, layer_mask,
                self.ray_hit_cache.hits.as_mut_ptr(), max_hits)
        };
        self.ray_hit_cache.hits.truncate(n as usize);
        n
    }

    pub fn ray_hit_count(&self) -> u32 { self.ray_hit_cache.hits.len() as u32 }
    pub fn ray_hit_body(&self, i: usize) -> f64 {
        self.ray_hit_cache.hits.get(i)
            .and_then(|h| self.bodies.iter().find(|(_, (_, b))| *b == h.body).map(|(hh, _)| hh))
            .unwrap_or(0.0)
    }
    /// field: 0..5 = point.xyz, normal.xyz
    pub fn ray_hit_axis(&self, i: usize, field: u32) -> f64 {
        let h = match self.ray_hit_cache.hits.get(i) { Some(v) => v, None => return 0.0 };
        (match field {
            0 => h.point.x,  1 => h.point.y,  2 => h.point.z,
            3 => h.normal.x, 4 => h.normal.y, 5 => h.normal.z,
            _ => 0.0,
        }) as f64
    }
    pub fn ray_hit_fraction(&self, i: usize) -> f32 {
        self.ray_hit_cache.hits.get(i).map(|h| h.fraction).unwrap_or(0.0)
    }
    pub fn ray_hit_sub_shape(&self, i: usize) -> u32 {
        self.ray_hit_cache.hits.get(i).map(|h| h.sub_shape_id).unwrap_or(0)
    }

    pub fn overlap_sphere(&mut self, world_h: f64, cx: f32, cy: f32, cz: f32, r: f32, layer_mask: u32, max_results: u32) -> u32 {
        self.overlap_cache.bodies.clear();
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return 0 };
        self.overlap_cache.bodies.resize(max_results as usize, BJ_INVALID);
        let n = unsafe {
            bj_query_overlap_sphere(world,
                BjVec3 { x: cx, y: cy, z: cz }, r, layer_mask,
                self.overlap_cache.bodies.as_mut_ptr(), max_results)
        };
        self.overlap_cache.bodies.truncate(n as usize);
        n
    }

    pub fn overlap_point(&mut self, world_h: f64, px: f32, py: f32, pz: f32, layer_mask: u32, max_results: u32) -> u32 {
        self.overlap_cache.bodies.clear();
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return 0 };
        self.overlap_cache.bodies.resize(max_results as usize, BJ_INVALID);
        let n = unsafe {
            bj_query_overlap_point(world,
                BjVec3 { x: px, y: py, z: pz }, layer_mask,
                self.overlap_cache.bodies.as_mut_ptr(), max_results)
        };
        self.overlap_cache.bodies.truncate(n as usize);
        n
    }

    #[allow(clippy::too_many_arguments)]
    pub fn overlap_box(
        &mut self, world_h: f64,
        px: f32, py: f32, pz: f32, rx: f32, ry: f32, rz: f32, rw: f32,
        hx: f32, hy: f32, hz: f32,
        layer_mask: u32, max_results: u32,
    ) -> u32 {
        self.overlap_cache.bodies.clear();
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return 0 };
        self.overlap_cache.bodies.resize(max_results as usize, BJ_INVALID);
        let xform = BjTransform { position: BjVec3 { x: px, y: py, z: pz }, rotation: BjQuat { x: rx, y: ry, z: rz, w: rw } };
        let n = unsafe {
            bj_query_overlap_box(world, &xform, BjVec3 { x: hx, y: hy, z: hz }, layer_mask,
                self.overlap_cache.bodies.as_mut_ptr(), max_results)
        };
        self.overlap_cache.bodies.truncate(n as usize);
        n
    }

    pub fn overlap_body(&self, i: usize) -> f64 {
        self.overlap_cache.bodies.get(i)
            .and_then(|bj| self.bodies.iter().find(|(_, (_, b))| *b == *bj).map(|(h, _)| h))
            .unwrap_or(0.0)
    }

    // -------------------------------------------------------- Constraints

    fn make_anchors(&self, body_a_h: f64, body_b_h: f64, ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, world_space: bool) -> Option<(bj_world, BjConstraintAnchors)> {
        let (wa, ba) = self.resolve_body(body_a_h)?;
        let bb = if body_b_h == 0.0 {
            BJ_INVALID
        } else {
            let (wb, bb) = self.resolve_body(body_b_h)?;
            if wb != wa { return None; }
            bb
        };
        Some((wa, BjConstraintAnchors {
            body_a: ba, body_b: bb,
            anchor_a: BjVec3 { x: ax, y: ay, z: az },
            anchor_b: BjVec3 { x: bx, y: by, z: bz },
            use_world_space: world_space as u8,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn constraint_fixed(&mut self, body_a: f64, body_b: f64, ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, world_space: bool) -> f64 {
        let (w, anchors) = match self.make_anchors(body_a, body_b, ax, ay, az, bx, by, bz, world_space) { Some(v) => v, None => return 0.0 };
        let c = unsafe { bj_constraint_fixed(w, &anchors) };
        if c == BJ_INVALID { 0.0 } else { self.constraints.alloc((w, c)) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn constraint_point(&mut self, body_a: f64, body_b: f64, ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, world_space: bool) -> f64 {
        let (w, anchors) = match self.make_anchors(body_a, body_b, ax, ay, az, bx, by, bz, world_space) { Some(v) => v, None => return 0.0 };
        let c = unsafe { bj_constraint_point(w, &anchors) };
        if c == BJ_INVALID { 0.0 } else { self.constraints.alloc((w, c)) }
    }

    /// EN-025 — six-DOF, which is what a ragdoll joint actually is: translation
    /// locked (a shoulder does not slide off the torso) and rotation limited (an
    /// elbow does not bend backwards). An unlimited ball joint is cheaper and
    /// gives you spaghetti; the limits are the whole difference between a corpse
    /// and a novelty.
    ///
    /// `rot_limits` is [xmin, xmax, ymin, ymax, zmin, zmax] in radians.
    #[allow(clippy::too_many_arguments)]
    pub fn constraint_six_dof_locked_translation(
        &mut self, body_a: f64, body_b: f64,
        ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32,
        rot_limits: [f32; 6], world_space: bool,
    ) -> f64 {
        let (w, anchors) = match self.make_anchors(body_a, body_b, ax, ay, az, bx, by, bz, world_space) {
            Some(v) => v, None => return 0.0,
        };
        // min >= max means LOCKED (see the shim header), so this pins all three
        // translation axes.
        let trans: [f32; 6] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let c = unsafe {
            bj_constraint_six_dof(w, &anchors, trans.as_ptr(), rot_limits.as_ptr())
        };
        if c == BJ_INVALID { 0.0 } else { self.constraints.alloc((w, c)) }
    }

    /// Body world transform as (position, quaternion) — the ragdoll needs the
    /// full frame, and the existing getters hand back one axis at a time.
    pub fn body_transform(&self, h: f64) -> Option<([f32; 3], [f32; 4])> {
        let p = [
            self.body_get_position_axis(h, 0) as f32,
            self.body_get_position_axis(h, 1) as f32,
            self.body_get_position_axis(h, 2) as f32,
        ];
        let q = [
            self.body_get_rotation_axis(h, 0) as f32,
            self.body_get_rotation_axis(h, 1) as f32,
            self.body_get_rotation_axis(h, 2) as f32,
            self.body_get_rotation_axis(h, 3) as f32,
        ];
        Some((p, q))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn constraint_hinge(
        &mut self, body_a: f64, body_b: f64,
        ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32,
        axis_x: f32, axis_y: f32, axis_z: f32,
        limit_min: f32, limit_max: f32, world_space: bool,
    ) -> f64 {
        let (w, anchors) = match self.make_anchors(body_a, body_b, ax, ay, az, bx, by, bz, world_space) { Some(v) => v, None => return 0.0 };
        let c = unsafe {
            bj_constraint_hinge(w, &anchors, BjVec3 { x: axis_x, y: axis_y, z: axis_z }, limit_min, limit_max)
        };
        if c == BJ_INVALID { 0.0 } else { self.constraints.alloc((w, c)) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn constraint_slider(
        &mut self, body_a: f64, body_b: f64,
        ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32,
        axis_x: f32, axis_y: f32, axis_z: f32,
        limit_min: f32, limit_max: f32, world_space: bool,
    ) -> f64 {
        let (w, anchors) = match self.make_anchors(body_a, body_b, ax, ay, az, bx, by, bz, world_space) { Some(v) => v, None => return 0.0 };
        let c = unsafe {
            bj_constraint_slider(w, &anchors, BjVec3 { x: axis_x, y: axis_y, z: axis_z }, limit_min, limit_max)
        };
        if c == BJ_INVALID { 0.0 } else { self.constraints.alloc((w, c)) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn constraint_distance(
        &mut self, body_a: f64, body_b: f64,
        ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32,
        min_distance: f32, max_distance: f32, world_space: bool,
    ) -> f64 {
        let (w, anchors) = match self.make_anchors(body_a, body_b, ax, ay, az, bx, by, bz, world_space) { Some(v) => v, None => return 0.0 };
        let c = unsafe { bj_constraint_distance(w, &anchors, min_distance, max_distance) };
        if c == BJ_INVALID { 0.0 } else { self.constraints.alloc((w, c)) }
    }

    pub fn constraint_destroy(&mut self, handle: f64) {
        if let Some((w, c)) = self.constraints.free(handle) {
            unsafe { bj_constraint_destroy(w, c); }
        }
    }

    pub fn constraint_set_enabled(&self, handle: f64, enabled: bool) {
        if let Some(&(w, c)) = self.constraints.get(handle) {
            unsafe { bj_constraint_set_enabled(w, c, enabled as u8); }
        }
    }

    // -------------------------------------------------------- Contacts

    pub fn contact_count(&self) -> u32 { self.contact_cache.events.len() as u32 }

    /// field: 0=event, 1=bodyA, 2=bodyB, 3..5=pointA.xyz, 6..8=pointB.xyz,
    ///        9..11=normal.xyz, 12=depth, 13=friction, 14=restitution
    pub fn contact_field(&self, i: usize, field: u32) -> f64 {
        let c = match self.contact_cache.events.get(i) { Some(v) => v, None => return 0.0 };
        let bj_to_bloom = |bj: bj_body| -> f64 {
            self.bodies.iter().find(|(_, (_, b))| *b == bj).map(|(h, _)| h).unwrap_or(0.0)
        };
        match field {
            0  => c.event as u32 as f64,
            1  => bj_to_bloom(c.body_a),
            2  => bj_to_bloom(c.body_b),
            3  => c.point_a.x as f64, 4 => c.point_a.y as f64, 5 => c.point_a.z as f64,
            6  => c.point_b.x as f64, 7 => c.point_b.y as f64, 8 => c.point_b.z as f64,
            9  => c.normal.x  as f64, 10=> c.normal.y  as f64, 11=> c.normal.z  as f64,
            12 => c.penetration_depth    as f64,
            13 => c.combined_friction    as f64,
            14 => c.combined_restitution as f64,
            _  => 0.0,
        }
    }

    pub fn clear_contacts(&mut self) { self.contact_cache.events.clear(); }

    // -------------------------------------------------------- Character controller

    #[allow(clippy::too_many_arguments)]
    pub fn character_create(
        &mut self, world_h: f64, shape_h: f64,
        up_x: f32, up_y: f32, up_z: f32,
        max_slope_angle: f32, character_padding: f32,
        penetration_recovery_speed: f32, predictive_contact_distance: f32,
        max_strength: f32, mass: f32, object_layer: u32,
        px: f32, py: f32, pz: f32,
        rx: f32, ry: f32, rz: f32, rw: f32,
    ) -> f64 {
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return 0.0 };
        let shape = match self.shapes.get(shape_h) { Some(&s) => s, None => return 0.0 };
        let desc = BjCharacterDesc {
            up: BjVec3 { x: up_x, y: up_y, z: up_z },
            max_slope_angle, character_padding, penetration_recovery_speed,
            predictive_contact_distance, max_strength, mass, object_layer,
        };
        let c = unsafe {
            bj_character_create(world, shape, &desc,
                BjVec3 { x: px, y: py, z: pz },
                BjQuat { x: rx, y: ry, z: rz, w: rw })
        };
        if c == BJ_INVALID { 0.0 } else { self.characters.alloc((world, c)) }
    }

    pub fn character_destroy(&mut self, handle: f64) {
        if let Some((w, c)) = self.characters.free(handle) {
            unsafe { bj_character_destroy(w, c); }
        }
    }

    fn resolve_character(&self, h: f64) -> Option<(bj_world, bj_character)> {
        self.characters.get(h).copied()
    }

    pub fn character_update(&self, h: f64, dt: f32, gx: f32, gy: f32, gz: f32) {
        if let Some((w, c)) = self.resolve_character(h) {
            unsafe { bj_character_update(w, c, dt, BjVec3 { x: gx, y: gy, z: gz }); }
        }
    }

    pub fn character_get_position_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, c)) = self.resolve_character(h) {
            let mut v = BjVec3::default();
            unsafe { bj_character_get_position(w, c, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn character_get_rotation_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, c)) = self.resolve_character(h) {
            let mut q = BjQuat::default();
            unsafe { bj_character_get_rotation(w, c, &mut q); }
            return match axis { 0 => q.x, 1 => q.y, 2 => q.z, 3 => q.w, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn character_set_position(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, c)) = self.resolve_character(h) {
            unsafe { bj_character_set_position(w, c, BjVec3 { x, y, z }); }
        }
    }
    pub fn character_set_rotation(&self, h: f64, x: f32, y: f32, z: f32, ww: f32) {
        if let Some((w, c)) = self.resolve_character(h) {
            unsafe { bj_character_set_rotation(w, c, BjQuat { x, y, z, w: ww }); }
        }
    }
    pub fn character_get_linear_velocity_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, c)) = self.resolve_character(h) {
            let mut v = BjVec3::default();
            unsafe { bj_character_get_linear_velocity(w, c, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn character_set_linear_velocity(&self, h: f64, x: f32, y: f32, z: f32) {
        if let Some((w, c)) = self.resolve_character(h) {
            unsafe { bj_character_set_linear_velocity(w, c, BjVec3 { x, y, z }); }
        }
    }

    pub fn character_get_ground_state(&self, h: f64) -> u32 {
        if let Some((w, c)) = self.resolve_character(h) {
            return unsafe { bj_character_get_ground_state(w, c) } as u32;
        }
        BjGroundState::InAir as u32
    }
    pub fn character_get_ground_normal_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, c)) = self.resolve_character(h) {
            let mut v = BjVec3::default();
            unsafe { bj_character_get_ground_normal(w, c, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn character_get_ground_position_axis(&self, h: f64, axis: u32) -> f64 {
        if let Some((w, c)) = self.resolve_character(h) {
            let mut v = BjVec3::default();
            unsafe { bj_character_get_ground_position(w, c, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn character_get_ground_body(&self, h: f64) -> f64 {
        if let Some((w, c)) = self.resolve_character(h) {
            let bj = unsafe { bj_character_get_ground_body(w, c) };
            if bj == BJ_INVALID { return 0.0; }
            return self.bodies.iter()
                .find(|(_, (_, b))| *b == bj)
                .map(|(h, _)| h)
                .unwrap_or(0.0);
        }
        0.0
    }
    pub fn character_set_shape(&self, h: f64, shape_h: f64) {
        if let (Some((w, c)), Some(&s)) = (self.resolve_character(h), self.shapes.get(shape_h)) {
            unsafe { bj_character_set_shape(w, c, s); }
        }
    }

    // -------------------------------------------------------- Soft bodies

    /// Create a soft body from scratch-buffered vertex+index streams.
    /// scratch_f32 layout: vertex_count * 4 floats (x, y, z, invMass per vertex).
    /// scratch_u32 layout: triangle_count * 3 indices.
    /// inv_mass == 0 pins the vertex.
    #[allow(clippy::too_many_arguments)]
    pub fn soft_body_create_from_scratch(
        &mut self, world_h: f64,
        vertex_count: u32, triangle_count: u32,
        px: f32, py: f32, pz: f32, rx: f32, ry: f32, rz: f32, rw: f32,
        object_layer: u32,
        edge_compliance: f32, gravity_factor: f32, linear_damping: f32, pressure: f32,
    ) -> f64 {
        let need_f = (vertex_count * 4) as usize;
        let need_u = (triangle_count * 3) as usize;
        if self.scratch_f32.len() < need_f || self.scratch_u32.len() < need_u {
            return 0.0;
        }
        if vertex_count < 3 || triangle_count == 0 { return 0.0; }
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return 0.0 };

        let body = unsafe {
            bj_soft_body_create(
                world,
                self.scratch_f32.as_ptr(), vertex_count,
                self.scratch_u32.as_ptr(), triangle_count,
                BjVec3 { x: px, y: py, z: pz },
                BjQuat { x: rx, y: ry, z: rz, w: rw },
                object_layer, edge_compliance, gravity_factor, linear_damping, pressure,
            )
        };
        if body == BJ_INVALID { 0.0 } else { self.bodies.alloc((world, body)) }
    }

    pub fn soft_body_vertex_count(&self, body_h: f64) -> u32 {
        self.resolve_body(body_h)
            .map(|(w, b)| unsafe { bj_soft_body_vertex_count(w, b) })
            .unwrap_or(0)
    }
    pub fn soft_body_get_vertex_axis(&self, body_h: f64, idx: u32, axis: u32) -> f64 {
        if let Some((w, b)) = self.resolve_body(body_h) {
            let mut v = BjVec3::default();
            unsafe { bj_soft_body_get_vertex(w, b, idx, &mut v); }
            return match axis { 0 => v.x, 1 => v.y, 2 => v.z, _ => 0.0 } as f64;
        }
        0.0
    }
    pub fn soft_body_set_vertex(&self, body_h: f64, idx: u32, x: f32, y: f32, z: f32) {
        if let Some((w, b)) = self.resolve_body(body_h) {
            unsafe { bj_soft_body_set_vertex(w, b, idx, BjVec3 { x, y, z }); }
        }
    }
    pub fn soft_body_set_vertex_inv_mass(&self, body_h: f64, idx: u32, inv_mass: f32) {
        if let Some((w, b)) = self.resolve_body(body_h) {
            unsafe { bj_soft_body_set_vertex_inv_mass(w, b, idx, inv_mass); }
        }
    }

    // -------------------------------------------------------- Wheeled vehicles

    #[allow(clippy::too_many_arguments)]
    pub fn vehicle_create(
        &mut self, world_h: f64, chassis_shape_h: f64,
        // Axes
        up_x: f32, up_y: f32, up_z: f32,
        fw_x: f32, fw_y: f32, fw_z: f32,
        // 4 wheel positions (flat array of 12 f32s: fl.xyz, fr.xyz, rl.xyz, rr.xyz)
        w0x: f32, w0y: f32, w0z: f32,
        w1x: f32, w1y: f32, w1z: f32,
        w2x: f32, w2y: f32, w2z: f32,
        w3x: f32, w3y: f32, w3z: f32,
        // Shared wheel + engine parameters
        wheel_radius: f32, wheel_width: f32,
        suspension_min: f32, suspension_max: f32,
        max_steer_angle: f32, max_brake_torque: f32, max_handbrake_torque: f32,
        engine_max_torque: f32, max_pitch_roll_angle: f32,
        object_layer: u32,
        // Pose
        px: f32, py: f32, pz: f32, rx: f32, ry: f32, rz: f32, rw: f32,
    ) -> (f64, f64) {
        // Returns (vehicle_handle, chassis_body_handle).
        let world = match self.worlds.get(world_h) { Some(&w) => w, None => return (0.0, 0.0) };
        let shape = match self.shapes.get(chassis_shape_h) { Some(&s) => s, None => return (0.0, 0.0) };

        let desc = BjVehicleDesc {
            up: BjVec3 { x: up_x, y: up_y, z: up_z },
            forward: BjVec3 { x: fw_x, y: fw_y, z: fw_z },
            wheel_positions: [
                BjVec3 { x: w0x, y: w0y, z: w0z },
                BjVec3 { x: w1x, y: w1y, z: w1z },
                BjVec3 { x: w2x, y: w2y, z: w2z },
                BjVec3 { x: w3x, y: w3y, z: w3z },
            ],
            wheel_radius, wheel_width,
            suspension_min_length: suspension_min,
            suspension_max_length: suspension_max,
            max_steer_angle, max_brake_torque, max_handbrake_torque,
            engine_max_torque, max_pitch_roll_angle, object_layer,
        };

        let vehicle = unsafe {
            bj_vehicle_create(world, shape, &desc,
                BjVec3 { x: px, y: py, z: pz },
                BjQuat { x: rx, y: ry, z: rz, w: rw })
        };
        if vehicle == BJ_INVALID { return (0.0, 0.0); }

        // Register the chassis as a regular body so TS can query its transform
        // via normal body APIs.
        let chassis_bj = unsafe { bj_vehicle_get_chassis(world, vehicle) };
        let chassis_h = self.bodies.alloc((world, chassis_bj));
        let vh = self.vehicles.alloc((world, vehicle, chassis_h));
        (vh, chassis_h)
    }

    pub fn vehicle_destroy(&mut self, handle: f64) {
        if let Some((w, v, chassis_h)) = self.vehicles.free(handle) {
            unsafe { bj_vehicle_destroy(w, v); }
            // Chassis was removed by vehicle_destroy; free its Rust handle too.
            let _ = self.bodies.free(chassis_h);
        }
    }

    pub fn vehicle_get_chassis(&self, handle: f64) -> f64 {
        self.vehicles.get(handle).map(|&(_, _, h)| h).unwrap_or(0.0)
    }

    fn resolve_vehicle(&self, h: f64) -> Option<(bj_world, bj_vehicle)> {
        self.vehicles.get(h).map(|&(w, v, _)| (w, v))
    }

    pub fn vehicle_set_input(&self, handle: f64, forward: f32, right: f32, brake: f32, handbrake: f32) {
        if let Some((w, v)) = self.resolve_vehicle(handle) {
            unsafe { bj_vehicle_set_input(w, v, forward, right, brake, handbrake); }
        }
    }

    pub fn vehicle_get_wheel_transform(&self, handle: f64, wheel_idx: u32, axis: u32) -> f32 {
        self.resolve_vehicle(handle)
            .map(|(w, v)| unsafe { bj_vehicle_get_wheel_transform(w, v, wheel_idx, axis) })
            .unwrap_or(0.0)
    }

    pub fn vehicle_get_engine_rpm(&self, handle: f64) -> f32 {
        self.resolve_vehicle(handle)
            .map(|(w, v)| unsafe { bj_vehicle_get_engine_rpm(w, v) })
            .unwrap_or(0.0)
    }

    pub fn vehicle_get_wheel_angular_velocity(&self, handle: f64, wheel_idx: u32) -> f32 {
        self.resolve_vehicle(handle)
            .map(|(w, v)| unsafe { bj_vehicle_get_wheel_angular_velocity(w, v, wheel_idx) })
            .unwrap_or(0.0)
    }
}

// ============================================================================
// FFI macro — generates the bloom_physics_* symbol set in the invoking crate.
//
// Usage (in a platform crate, e.g. native/macos/src/lib.rs):
//
//     #[inline]
//     fn bloom_jolt_ffi_physics() -> &'static mut bloom_shared::physics_jolt::JoltPhysics {
//         &mut engine().jolt
//     }
//     bloom_shared::define_physics_ffi!();
//
// The macro emits #[no_mangle] extern "C" functions, so each platform crate's
// staticlib/cdylib independently re-exports the full Perry FFI surface.
// ============================================================================

#[macro_export]
macro_rules! define_physics_ffi {
    () => {
        // --- World ---

        #[no_mangle] pub extern "C" fn bloom_physics_create_world(gx: f64, gy: f64, gz: f64, max_bodies: f64, num_threads: f64) -> f64 {
            bloom_jolt_ffi_physics().create_world(gx as f32, gy as f32, gz as f32, max_bodies as u32, num_threads as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_destroy_world(world: f64) { bloom_jolt_ffi_physics().destroy_world(world); }
        #[no_mangle] pub extern "C" fn bloom_physics_set_gravity(world: f64, gx: f64, gy: f64, gz: f64) {
            bloom_jolt_ffi_physics().set_gravity(world, gx as f32, gy as f32, gz as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_get_gravity(world: f64, axis: f64) -> f64 { bloom_jolt_ffi_physics().get_gravity_axis(world, axis as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_optimize_broadphase(world: f64) { bloom_jolt_ffi_physics().optimize_broadphase(world); }
        #[no_mangle] pub extern "C" fn bloom_physics_step(world: f64, dt: f64, collision_steps: f64) {
            bloom_jolt_ffi_physics().step(world, dt as f32, collision_steps as u32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_step_fixed(world: f64, dt: f64, collision_steps: f64) -> f64 {
            bloom_jolt_ffi_physics().step_fixed(world, dt as f32, collision_steps as u32) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_set_fixed_timestep(world: f64, hz: f64, max_steps: f64) {
            bloom_jolt_ffi_physics().set_fixed_timestep(world, hz as f32, max_steps as u32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_set_interpolation(world: f64, on: f64) {
            bloom_jolt_ffi_physics().set_interpolation(world, on != 0.0);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_get_step_alpha(world: f64) -> f64 {
            bloom_jolt_ffi_physics().step_alpha(world) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_set_layer_collides(world: f64, a: f64, b: f64, collides: f64) {
            bloom_jolt_ffi_physics().set_layer_collides(world, a as u32, b as u32, collides != 0.0);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_get_layer_collides(world: f64, a: f64, b: f64) -> f64 {
            if bloom_jolt_ffi_physics().get_layer_collides(world, a as u32, b as u32) { 1.0 } else { 0.0 }
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_count(world: f64) -> f64 { bloom_jolt_ffi_physics().body_count(world) as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_active_body_count(world: f64) -> f64 { bloom_jolt_ffi_physics().active_body_count(world) as f64 }

        // --- Shapes ---

        #[no_mangle] pub extern "C" fn bloom_physics_shape_box(hx: f64, hy: f64, hz: f64, convex_radius: f64) -> f64 {
            bloom_jolt_ffi_physics().create_box_shape(hx as f32, hy as f32, hz as f32, convex_radius as f32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_sphere(r: f64) -> f64 { bloom_jolt_ffi_physics().create_sphere_shape(r as f32) }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_capsule(h: f64, r: f64) -> f64 { bloom_jolt_ffi_physics().create_capsule_shape(h as f32, r as f32) }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_cylinder(h: f64, r: f64, cr: f64) -> f64 {
            bloom_jolt_ffi_physics().create_cylinder_shape(h as f32, r as f32, cr as f32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_scaled(base: f64, sx: f64, sy: f64, sz: f64) -> f64 {
            bloom_jolt_ffi_physics().create_scaled_shape(base, sx as f32, sy as f32, sz as f32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_offset_com(base: f64, ox: f64, oy: f64, oz: f64) -> f64 {
            bloom_jolt_ffi_physics().create_offset_com_shape(base, ox as f32, oy as f32, oz as f32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_release(shape: f64) { bloom_jolt_ffi_physics().release_shape(shape); }

        // Scratch streams for variable-size shape inputs.
        #[no_mangle] pub extern "C" fn bloom_physics_scratch_reset() { bloom_jolt_ffi_physics().scratch_reset(); }
        #[no_mangle] pub extern "C" fn bloom_physics_scratch_push_f32(v: f64) { bloom_jolt_ffi_physics().scratch_push_f32(v as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_scratch_push_u32(v: f64) { bloom_jolt_ffi_physics().scratch_push_u32(v as u32); }

        // Complex shape factories — consume scratch streams populated by the caller.
        #[no_mangle] pub extern "C" fn bloom_physics_shape_convex_hull(num_points: f64, convex_radius: f64) -> f64 {
            bloom_jolt_ffi_physics().shape_convex_hull_from_scratch(num_points as u32, convex_radius as f32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_mesh(vertex_count: f64, triangle_count: f64) -> f64 {
            bloom_jolt_ffi_physics().shape_mesh_from_scratch(vertex_count as u32, triangle_count as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_heightfield(
            sample_count: f64, ox: f64, oy: f64, oz: f64, sx: f64, sy: f64, sz: f64, block_size: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().shape_heightfield_from_scratch(
                sample_count as u32, ox as f32, oy as f32, oz as f32,
                sx as f32, sy as f32, sz as f32, block_size as u32,
            )
        }

        // Compound shape builder — begin / add_child (repeat) / end.
        #[no_mangle] pub extern "C" fn bloom_physics_compound_begin() { bloom_jolt_ffi_physics().compound_begin(); }
        #[no_mangle] pub extern "C" fn bloom_physics_compound_add_child(
            shape: f64, px: f64, py: f64, pz: f64, rx: f64, ry: f64, rz: f64, rw: f64,
        ) {
            bloom_jolt_ffi_physics().compound_add_child(
                shape,
                px as f32, py as f32, pz as f32,
                rx as f32, ry as f32, rz as f32, rw as f32,
            );
        }
        #[no_mangle] pub extern "C" fn bloom_physics_compound_end() -> f64 {
            bloom_jolt_ffi_physics().compound_end()
        }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_bounds(shape: f64, axis: f64) -> f64 { bloom_jolt_ffi_physics().shape_bounds_axis(shape, axis as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_shape_volume(shape: f64) -> f64 { bloom_jolt_ffi_physics().shape_volume(shape) as f64 }

        // --- Bodies ---

        #[no_mangle] pub extern "C" fn bloom_physics_body_create(
            world: f64, shape: f64, motion_type: f64,
            px: f64, py: f64, pz: f64,
            rx: f64, ry: f64, rz: f64, rw: f64,
            layer: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().create_body(
                world, shape, motion_type as u32,
                px as f32, py as f32, pz as f32,
                rx as f32, ry as f32, rz as f32, rw as f32,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                layer as u32,
                false, true, false, true,
                0.2, 0.0, 0.05, 0.05, 1.0,
                0.0, 0.0, 0.0, 0.0,
                0,
            )
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_destroy(body: f64) { bloom_jolt_ffi_physics().destroy_body(body); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_activate(body: f64) { bloom_jolt_ffi_physics().body_activate(body); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_deactivate(body: f64) { bloom_jolt_ffi_physics().body_deactivate(body); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_is_active(body: f64) -> f64 { if bloom_jolt_ffi_physics().body_is_active(body) { 1.0 } else { 0.0 } }
        #[no_mangle] pub extern "C" fn bloom_physics_body_is_valid(body: f64) -> f64 { if bloom_jolt_ffi_physics().body_is_valid(body) { 1.0 } else { 0.0 } }

        #[no_mangle] pub extern "C" fn bloom_physics_body_get_position(body: f64, axis: f64) -> f64 { bloom_jolt_ffi_physics().body_get_position_axis(body, axis as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_body_get_rotation(body: f64, axis: f64) -> f64 { bloom_jolt_ffi_physics().body_get_rotation_axis(body, axis as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_position(body: f64, x: f64, y: f64, z: f64, activate: f64) {
            bloom_jolt_ffi_physics().body_set_position(body, x as f32, y as f32, z as f32, activate != 0.0);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_rotation(body: f64, x: f64, y: f64, z: f64, w: f64, activate: f64) {
            bloom_jolt_ffi_physics().body_set_rotation(body, x as f32, y as f32, z as f32, w as f32, activate != 0.0);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_transform(
            body: f64, px: f64, py: f64, pz: f64, rx: f64, ry: f64, rz: f64, rw: f64, activate: f64,
        ) {
            bloom_jolt_ffi_physics().body_set_transform(body, px as f32, py as f32, pz as f32, rx as f32, ry as f32, rz as f32, rw as f32, activate != 0.0);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_move_kinematic(
            body: f64, px: f64, py: f64, pz: f64, rx: f64, ry: f64, rz: f64, rw: f64, dt: f64,
        ) {
            bloom_jolt_ffi_physics().body_move_kinematic(body, px as f32, py as f32, pz as f32, rx as f32, ry as f32, rz as f32, rw as f32, dt as f32);
        }

        #[no_mangle] pub extern "C" fn bloom_physics_body_get_linear_velocity(body: f64, axis: f64) -> f64 { bloom_jolt_ffi_physics().body_get_linear_velocity_axis(body, axis as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_body_get_angular_velocity(body: f64, axis: f64) -> f64 { bloom_jolt_ffi_physics().body_get_angular_velocity_axis(body, axis as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_body_get_point_velocity(body: f64, px: f64, py: f64, pz: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().body_get_point_velocity_axis(body, px as f32, py as f32, pz as f32, axis as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_linear_velocity(body: f64, x: f64, y: f64, z: f64) {
            bloom_jolt_ffi_physics().body_set_linear_velocity(body, x as f32, y as f32, z as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_angular_velocity(body: f64, x: f64, y: f64, z: f64) {
            bloom_jolt_ffi_physics().body_set_angular_velocity(body, x as f32, y as f32, z as f32);
        }

        #[no_mangle] pub extern "C" fn bloom_physics_body_add_force(body: f64, x: f64, y: f64, z: f64) { bloom_jolt_ffi_physics().body_add_force(body, x as f32, y as f32, z as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_add_impulse(body: f64, x: f64, y: f64, z: f64) { bloom_jolt_ffi_physics().body_add_impulse(body, x as f32, y as f32, z as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_add_torque(body: f64, x: f64, y: f64, z: f64) { bloom_jolt_ffi_physics().body_add_torque(body, x as f32, y as f32, z as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_add_angular_impulse(body: f64, x: f64, y: f64, z: f64) { bloom_jolt_ffi_physics().body_add_angular_impulse(body, x as f32, y as f32, z as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_add_force_at(body: f64, fx: f64, fy: f64, fz: f64, px: f64, py: f64, pz: f64) {
            bloom_jolt_ffi_physics().body_add_force_at(body, fx as f32, fy as f32, fz as f32, px as f32, py as f32, pz as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_add_impulse_at(body: f64, ix: f64, iy: f64, iz: f64, px: f64, py: f64, pz: f64) {
            bloom_jolt_ffi_physics().body_add_impulse_at(body, ix as f32, iy as f32, iz as f32, px as f32, py as f32, pz as f32);
        }

        #[no_mangle] pub extern "C" fn bloom_physics_body_set_friction(body: f64, v: f64) { bloom_jolt_ffi_physics().body_set_friction(body, v as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_restitution(body: f64, v: f64) { bloom_jolt_ffi_physics().body_set_restitution(body, v as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_linear_damping(body: f64, v: f64) { bloom_jolt_ffi_physics().body_set_linear_damping(body, v as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_angular_damping(body: f64, v: f64) { bloom_jolt_ffi_physics().body_set_angular_damping(body, v as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_gravity_factor(body: f64, v: f64) { bloom_jolt_ffi_physics().body_set_gravity_factor(body, v as f32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_ccd(body: f64, enabled: f64) { bloom_jolt_ffi_physics().body_set_ccd(body, enabled != 0.0); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_motion_type(body: f64, t: f64, activate: f64) { bloom_jolt_ffi_physics().body_set_motion_type(body, t as u32, activate != 0.0); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_object_layer(body: f64, layer: f64) { bloom_jolt_ffi_physics().body_set_object_layer(body, layer as u32); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_is_sensor(body: f64, enabled: f64) { bloom_jolt_ffi_physics().body_set_is_sensor(body, enabled != 0.0); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_allow_sleeping(body: f64, enabled: f64) { bloom_jolt_ffi_physics().body_set_allow_sleeping(body, enabled != 0.0); }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_shape(body: f64, shape: f64, update_mass: f64, activate: f64) {
            bloom_jolt_ffi_physics().body_set_shape(body, shape, update_mass != 0.0, activate != 0.0);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_lock_rotation_axes(body: f64, x: f64, y: f64, z: f64) {
            bloom_jolt_ffi_physics().body_lock_rotation_axes(body, x != 0.0, y != 0.0, z != 0.0);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_lock_translation_axes(body: f64, x: f64, y: f64, z: f64) {
            bloom_jolt_ffi_physics().body_lock_translation_axes(body, x != 0.0, y != 0.0, z != 0.0);
        }

        #[no_mangle] pub extern "C" fn bloom_physics_body_get_mass(body: f64) -> f64 { bloom_jolt_ffi_physics().body_get_mass(body) as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_body_get_friction(body: f64) -> f64 { bloom_jolt_ffi_physics().body_get_friction(body) as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_body_get_restitution(body: f64) -> f64 { bloom_jolt_ffi_physics().body_get_restitution(body) as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_body_get_object_layer(body: f64) -> f64 { bloom_jolt_ffi_physics().body_get_object_layer(body) as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_body_set_user_data(body: f64, lo: f64, hi: f64) {
            let user = (lo as u64) | ((hi as u64) << 32);
            bloom_jolt_ffi_physics().body_set_user_data(body, user);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_body_get_user_data(body: f64, part: f64) -> f64 {
            let u = bloom_jolt_ffi_physics().body_get_user_data(body);
            (if part as u32 == 1 { (u >> 32) as u32 } else { u as u32 }) as f64
        }

        // --- Queries ---

        #[no_mangle] pub extern "C" fn bloom_physics_raycast(
            world: f64, ox: f64, oy: f64, oz: f64,
            dx: f64, dy: f64, dz: f64, max_dist: f64, layer_mask: f64,
        ) -> f64 {
            if bloom_jolt_ffi_physics().raycast_closest(world, ox as f32, oy as f32, oz as f32, dx as f32, dy as f32, dz as f32, max_dist as f32, layer_mask as u32) { 1.0 } else { 0.0 }
        }
        #[no_mangle] pub extern "C" fn bloom_physics_raycast_all(
            world: f64, ox: f64, oy: f64, oz: f64,
            dx: f64, dy: f64, dz: f64, max_dist: f64, layer_mask: f64, max_hits: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().raycast_all(world, ox as f32, oy as f32, oz as f32, dx as f32, dy as f32, dz as f32, max_dist as f32, layer_mask as u32, max_hits as u32) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_ray_hit_count() -> f64 { bloom_jolt_ffi_physics().ray_hit_count() as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_ray_hit_body(i: f64) -> f64 { bloom_jolt_ffi_physics().ray_hit_body(i as usize) }
        #[no_mangle] pub extern "C" fn bloom_physics_ray_hit_axis(i: f64, field: f64) -> f64 { bloom_jolt_ffi_physics().ray_hit_axis(i as usize, field as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_ray_hit_fraction(i: f64) -> f64 { bloom_jolt_ffi_physics().ray_hit_fraction(i as usize) as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_ray_hit_sub_shape(i: f64) -> f64 { bloom_jolt_ffi_physics().ray_hit_sub_shape(i as usize) as f64 }

        #[no_mangle] pub extern "C" fn bloom_physics_overlap_sphere(world: f64, cx: f64, cy: f64, cz: f64, r: f64, layer_mask: f64, max_results: f64) -> f64 {
            bloom_jolt_ffi_physics().overlap_sphere(world, cx as f32, cy as f32, cz as f32, r as f32, layer_mask as u32, max_results as u32) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_overlap_point(world: f64, px: f64, py: f64, pz: f64, layer_mask: f64, max_results: f64) -> f64 {
            bloom_jolt_ffi_physics().overlap_point(world, px as f32, py as f32, pz as f32, layer_mask as u32, max_results as u32) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_overlap_box(
            world: f64, px: f64, py: f64, pz: f64, rx: f64, ry: f64, rz: f64, rw: f64,
            hx: f64, hy: f64, hz: f64, layer_mask: f64, max_results: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().overlap_box(world, px as f32, py as f32, pz as f32, rx as f32, ry as f32, rz as f32, rw as f32, hx as f32, hy as f32, hz as f32, layer_mask as u32, max_results as u32) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_overlap_body(i: f64) -> f64 { bloom_jolt_ffi_physics().overlap_body(i as usize) }

        // --- Constraints ---

        #[no_mangle] pub extern "C" fn bloom_physics_constraint_fixed(
            body_a: f64, body_b: f64, ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64, world_space: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().constraint_fixed(body_a, body_b, ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32, world_space != 0.0)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_constraint_point(
            body_a: f64, body_b: f64, ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64, world_space: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().constraint_point(body_a, body_b, ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32, world_space != 0.0)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_constraint_hinge(
            body_a: f64, body_b: f64, ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64,
            axx: f64, axy: f64, axz: f64, lmin: f64, lmax: f64, world_space: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().constraint_hinge(
                body_a, body_b,
                ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32,
                axx as f32, axy as f32, axz as f32,
                lmin as f32, lmax as f32, world_space != 0.0,
            )
        }
        #[no_mangle] pub extern "C" fn bloom_physics_constraint_slider(
            body_a: f64, body_b: f64, ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64,
            axx: f64, axy: f64, axz: f64, lmin: f64, lmax: f64, world_space: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().constraint_slider(
                body_a, body_b,
                ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32,
                axx as f32, axy as f32, axz as f32,
                lmin as f32, lmax as f32, world_space != 0.0,
            )
        }
        #[no_mangle] pub extern "C" fn bloom_physics_constraint_distance(
            body_a: f64, body_b: f64, ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64,
            min_d: f64, max_d: f64, world_space: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().constraint_distance(body_a, body_b, ax as f32, ay as f32, az as f32, bx as f32, by as f32, bz as f32, min_d as f32, max_d as f32, world_space != 0.0)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_constraint_destroy(c: f64) { bloom_jolt_ffi_physics().constraint_destroy(c); }
        #[no_mangle] pub extern "C" fn bloom_physics_constraint_set_enabled(c: f64, enabled: f64) { bloom_jolt_ffi_physics().constraint_set_enabled(c, enabled != 0.0); }

        // --- Contact events ---

        #[no_mangle] pub extern "C" fn bloom_physics_contact_count() -> f64 { bloom_jolt_ffi_physics().contact_count() as f64 }
        #[no_mangle] pub extern "C" fn bloom_physics_contact_field(i: f64, field: f64) -> f64 { bloom_jolt_ffi_physics().contact_field(i as usize, field as u32) }
        #[no_mangle] pub extern "C" fn bloom_physics_clear_contacts(world: f64) { let _ = world; bloom_jolt_ffi_physics().clear_contacts(); }

        // --- Character controller (Tier 2) ---

        #[no_mangle] pub extern "C" fn bloom_physics_character_create(
            world: f64, shape: f64,
            up_x: f64, up_y: f64, up_z: f64,
            max_slope_angle: f64, character_padding: f64,
            penetration_recovery_speed: f64, predictive_contact_distance: f64,
            max_strength: f64, mass: f64, object_layer: f64,
            px: f64, py: f64, pz: f64,
            rx: f64, ry: f64, rz: f64, rw: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().character_create(
                world, shape,
                up_x as f32, up_y as f32, up_z as f32,
                max_slope_angle as f32, character_padding as f32,
                penetration_recovery_speed as f32, predictive_contact_distance as f32,
                max_strength as f32, mass as f32, object_layer as u32,
                px as f32, py as f32, pz as f32,
                rx as f32, ry as f32, rz as f32, rw as f32,
            )
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_destroy(c: f64) { bloom_jolt_ffi_physics().character_destroy(c); }
        #[no_mangle] pub extern "C" fn bloom_physics_character_update(c: f64, dt: f64, gx: f64, gy: f64, gz: f64) {
            bloom_jolt_ffi_physics().character_update(c, dt as f32, gx as f32, gy as f32, gz as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_get_position(c: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().character_get_position_axis(c, axis as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_get_rotation(c: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().character_get_rotation_axis(c, axis as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_set_position(c: f64, x: f64, y: f64, z: f64) {
            bloom_jolt_ffi_physics().character_set_position(c, x as f32, y as f32, z as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_set_rotation(c: f64, x: f64, y: f64, z: f64, w: f64) {
            bloom_jolt_ffi_physics().character_set_rotation(c, x as f32, y as f32, z as f32, w as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_get_linear_velocity(c: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().character_get_linear_velocity_axis(c, axis as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_set_linear_velocity(c: f64, x: f64, y: f64, z: f64) {
            bloom_jolt_ffi_physics().character_set_linear_velocity(c, x as f32, y as f32, z as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_get_ground_state(c: f64) -> f64 {
            bloom_jolt_ffi_physics().character_get_ground_state(c) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_get_ground_normal(c: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().character_get_ground_normal_axis(c, axis as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_get_ground_position(c: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().character_get_ground_position_axis(c, axis as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_get_ground_body(c: f64) -> f64 {
            bloom_jolt_ffi_physics().character_get_ground_body(c)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_character_set_shape(c: f64, shape: f64) {
            bloom_jolt_ffi_physics().character_set_shape(c, shape);
        }

        // --- Soft bodies (Tier 2) ---

        #[no_mangle] pub extern "C" fn bloom_physics_soft_body_create(
            world: f64, vertex_count: f64, triangle_count: f64,
            px: f64, py: f64, pz: f64, rx: f64, ry: f64, rz: f64, rw: f64,
            object_layer: f64,
            edge_compliance: f64, gravity_factor: f64, linear_damping: f64, pressure: f64,
        ) -> f64 {
            bloom_jolt_ffi_physics().soft_body_create_from_scratch(
                world, vertex_count as u32, triangle_count as u32,
                px as f32, py as f32, pz as f32,
                rx as f32, ry as f32, rz as f32, rw as f32,
                object_layer as u32,
                edge_compliance as f32, gravity_factor as f32,
                linear_damping as f32, pressure as f32,
            )
        }
        #[no_mangle] pub extern "C" fn bloom_physics_soft_body_vertex_count(body: f64) -> f64 {
            bloom_jolt_ffi_physics().soft_body_vertex_count(body) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_soft_body_get_vertex(body: f64, idx: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().soft_body_get_vertex_axis(body, idx as u32, axis as u32)
        }
        #[no_mangle] pub extern "C" fn bloom_physics_soft_body_set_vertex(body: f64, idx: f64, x: f64, y: f64, z: f64) {
            bloom_jolt_ffi_physics().soft_body_set_vertex(body, idx as u32, x as f32, y as f32, z as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_soft_body_set_vertex_inv_mass(body: f64, idx: f64, inv_mass: f64) {
            bloom_jolt_ffi_physics().soft_body_set_vertex_inv_mass(body, idx as u32, inv_mass as f32);
        }

        // --- Wheeled vehicles (Tier 2) ---
        //
        // vehicle_create returns a packed f64 carrying BOTH the vehicle handle
        // and the chassis body handle: (vehicle << 20) | chassis_body. This keeps
        // the FFI scalar-only while letting TS unpack both. Handles stay <2^20
        // in practice for any realistic scene.

        #[no_mangle] pub extern "C" fn bloom_physics_vehicle_create(
            world: f64, chassis_shape: f64,
            up_x: f64, up_y: f64, up_z: f64,
            fw_x: f64, fw_y: f64, fw_z: f64,
            w0x: f64, w0y: f64, w0z: f64,
            w1x: f64, w1y: f64, w1z: f64,
            w2x: f64, w2y: f64, w2z: f64,
            w3x: f64, w3y: f64, w3z: f64,
            wheel_radius: f64, wheel_width: f64,
            suspension_min: f64, suspension_max: f64,
            max_steer_angle: f64, max_brake_torque: f64, max_handbrake_torque: f64,
            engine_max_torque: f64, max_pitch_roll_angle: f64,
            object_layer: f64,
            px: f64, py: f64, pz: f64, rx: f64, ry: f64, rz: f64, rw: f64,
        ) -> f64 {
            let (vh, _) = bloom_jolt_ffi_physics().vehicle_create(
                world, chassis_shape,
                up_x as f32, up_y as f32, up_z as f32,
                fw_x as f32, fw_y as f32, fw_z as f32,
                w0x as f32, w0y as f32, w0z as f32,
                w1x as f32, w1y as f32, w1z as f32,
                w2x as f32, w2y as f32, w2z as f32,
                w3x as f32, w3y as f32, w3z as f32,
                wheel_radius as f32, wheel_width as f32,
                suspension_min as f32, suspension_max as f32,
                max_steer_angle as f32, max_brake_torque as f32, max_handbrake_torque as f32,
                engine_max_torque as f32, max_pitch_roll_angle as f32,
                object_layer as u32,
                px as f32, py as f32, pz as f32,
                rx as f32, ry as f32, rz as f32, rw as f32,
            );
            vh
        }
        #[no_mangle] pub extern "C" fn bloom_physics_vehicle_destroy(v: f64) { bloom_jolt_ffi_physics().vehicle_destroy(v); }
        #[no_mangle] pub extern "C" fn bloom_physics_vehicle_get_chassis(v: f64) -> f64 { bloom_jolt_ffi_physics().vehicle_get_chassis(v) }
        #[no_mangle] pub extern "C" fn bloom_physics_vehicle_set_input(v: f64, forward: f64, right: f64, brake: f64, handbrake: f64) {
            bloom_jolt_ffi_physics().vehicle_set_input(v, forward as f32, right as f32, brake as f32, handbrake as f32);
        }
        #[no_mangle] pub extern "C" fn bloom_physics_vehicle_get_wheel_transform(v: f64, wheel_index: f64, axis: f64) -> f64 {
            bloom_jolt_ffi_physics().vehicle_get_wheel_transform(v, wheel_index as u32, axis as u32) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_vehicle_get_engine_rpm(v: f64) -> f64 {
            bloom_jolt_ffi_physics().vehicle_get_engine_rpm(v) as f64
        }
        #[no_mangle] pub extern "C" fn bloom_physics_vehicle_get_wheel_angular_velocity(v: f64, wheel_index: f64) -> f64 {
            bloom_jolt_ffi_physics().vehicle_get_wheel_angular_velocity(v, wheel_index as u32) as f64
        }
    };
}

// ---------------------------------------------------------------------------
// Fixed-timestep stepping tests. Run with the same `jolt` feature gate the
// jolt_sys smoke tests use (CI's shared-tests job exercises them on all
// three host OSes).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod step_tests {
    use super::*;

    /// Dynamic sphere falling under default gravity in a fresh world.
    fn world_with_falling_sphere(p: &mut JoltPhysics) -> (f64, f64) {
        let world = p.create_world(0.0, -9.81, 0.0, 1024, 1);
        assert_ne!(world, 0.0);
        let shape = p.create_sphere_shape(0.5);
        assert_ne!(shape, 0.0);
        #[rustfmt::skip]
        let body = p.create_body(
            world, shape, 2 /* dynamic */,
            0.0, 100.0, 0.0,          // position
            0.0, 0.0, 0.0, 1.0,       // rotation
            0.0, 0.0, 0.0,            // linear velocity
            0.0, 0.0, 0.0,            // angular velocity
            1,                        // moving layer
            false, false, false, true, // sensor/sleep/ccd/awake
            0.5, 0.0,                 // friction, restitution
            0.0, 0.0, 1.0,            // damping, gravity factor
            0.0,                      // mass override (0 = from shape)
            0.0, 0.0, 0.0,            // inertia override
            0,                        // user data
        );
        assert_ne!(body, 0.0);
        (world, body)
    }

    /// Irregular frame times through the accumulator land on the same
    /// trajectory as the equivalent number of manual fixed steps.
    #[test]
    fn fixed_accumulator_matches_manual_stepping() {
        let mut p = JoltPhysics::new();
        let (wa, ba) = world_with_falling_sphere(&mut p);
        let (wb, bb) = world_with_falling_sphere(&mut p);

        // A: irregular frames summing to 12.5 fixed steps — the half-step
        // margin keeps the floor() robustly at 12 across f32 accumulation
        // rounding (an exact-boundary sum floors to 11 or 12 depending on
        // rounding direction, which is correct but untestable).
        let frames = [0.04, 0.05, 0.03, 0.06, 12.5 / 60.0 - 0.18];
        let mut alpha = 1.0;
        for dt in frames {
            alpha = p.step_fixed(wa, dt as f32, 1);
        }
        assert!((0.3..=0.7).contains(&alpha), "expected ~half-step remainder, alpha = {alpha}");
        // B: 12 manual steps at exactly 1/60.
        for _ in 0..12 {
            p.step(wb, 1.0 / 60.0, 1);
        }

        let ya = p.body_get_position_axis(ba, 1);
        let yb = p.body_get_position_axis(bb, 1);
        // 0.2 s of free fall from rest ≈ ½·9.81·0.04 ≈ 0.196 m
        assert!(ya < 99.9, "sphere did not fall: y = {ya}");
        assert!(
            (ya - yb).abs() < 1e-4,
            "accumulator diverged from manual stepping: {ya} vs {yb}"
        );

        p.destroy_world(wa);
        p.destroy_world(wb);
    }

    /// A huge hitch frame consumes at most max_steps_per_frame steps and
    /// drops the backlog instead of simulating seconds of catch-up.
    #[test]
    fn hitch_frame_is_clamped_and_capped() {
        let mut p = JoltPhysics::new();
        let (w, b) = world_with_falling_sphere(&mut p);

        p.step_fixed(w, 10.0, 1); // debugger-pause-sized frame
        let y = p.body_get_position_axis(b, 1);
        // 4 steps at 1/60 ≈ 67ms of fall from rest ≈ 2.2cm. If the clamp or
        // cap failed we'd have simulated 0.25–10s (0.3m–490m of fall).
        let fallen = 100.0 - y;
        assert!(fallen > 0.0, "no simulation happened");
        assert!(
            fallen < 0.1,
            "hitch frame simulated too much catch-up: fell {fallen}m"
        );

        // The dropped backlog must not leak into later frames: a zero-dt
        // follow-up frame steps at most once.
        let y1 = p.body_get_position_axis(b, 1);
        p.step_fixed(w, 0.0, 1);
        let y2 = p.body_get_position_axis(b, 1);
        assert!(
            (y1 - y2).abs() < 0.01,
            "backlog leaked into the next frame: {y1} -> {y2}"
        );

        p.destroy_world(w);
    }

    /// With interpolation on, the position getter blends between the last
    /// two simulated states by the carried remainder.
    #[test]
    fn interpolation_blends_between_steps() {
        let mut p = JoltPhysics::new();
        let (w, b) = world_with_falling_sphere(&mut p);
        p.set_interpolation(w, true);

        // One and a half fixed steps: alpha should be 0.5 and the reported
        // position should sit between the step-1 and step-2 states.
        let alpha = p.step_fixed(w, 1.5 / 60.0, 1);
        assert!((alpha - 0.5).abs() < 1e-3, "alpha = {alpha}");

        let y_blended = p.body_get_position_axis(b, 1);
        p.set_interpolation(w, false);
        let y_raw = p.body_get_position_axis(b, 1);
        // The sphere is falling, so the blended (half-step-old) position
        // must be strictly above the raw current position.
        assert!(
            y_blended > y_raw,
            "interpolated y ({y_blended}) not above raw y ({y_raw})"
        );

        p.destroy_world(w);
    }
}
