//! Rapier 3D physics integration for Bloom Engine.
//!
//! Wraps Rapier's physics world, rigid bodies, colliders, joints, and queries
//! behind Bloom's HandleRegistry<f64> system for FFI compatibility.
//!
//! Rapier 0.32 uses glam/glamx types (Vec3, Quat/Rot3, Pose3) instead of nalgebra.

use crate::handles::HandleRegistry;
use crate::scene::SceneGraph;

use rapier3d::prelude::*;
use rapier3d::parry::query::DefaultQueryDispatcher;

// ============================================================
// Collision event
// ============================================================

pub struct CollisionEvent {
    pub body_a: f64,
    pub body_b: f64,
    pub started: bool,
}

// ============================================================
// Ray hit result (cached for multi-call read-back)
// ============================================================

pub struct RayHitResult {
    pub body_handle: f64,
    pub distance: f64,
    pub point: [f64; 3],
    pub normal: [f64; 3],
}

// ============================================================
// Physics World
// ============================================================

pub struct PhysicsWorld {
    // Rapier core state
    gravity: Vector,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,

    // Handle bridging: Bloom f64 handles <-> Rapier opaque handles
    body_handles: HandleRegistry<RigidBodyHandle>,
    collider_handles: HandleRegistry<ColliderHandle>,
    joint_handles: HandleRegistry<ImpulseJointHandle>,

    // Scene node attachment: indexed by (bloom_body_handle - 1)
    body_to_scene_node: Vec<Option<f64>>,
    // Reverse mapping: Rapier RigidBodyHandle -> Bloom f64 handle (for collision events)
    rapier_to_bloom_body: std::collections::HashMap<RigidBodyHandle, f64>,

    // Fixed timestep accumulator
    accumulator: f64,
    pub fixed_dt: f64,
    pub max_substeps: u32,

    // Collision events (populated during step, drained by queries)
    pub collision_events: Vec<CollisionEvent>,

    // Ray hit cache (for multi-call read-back pattern)
    pub last_ray_hit: Option<RayHitResult>,

    // Collision event read cache (for multi-call read-back)
    pub last_collision_read: (f64, f64, bool),
}

impl PhysicsWorld {
    pub fn new(gx: f32, gy: f32, gz: f32) -> Self {
        Self {
            gravity: Vector::new(gx, gy, gz),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            body_handles: HandleRegistry::new(),
            collider_handles: HandleRegistry::new(),
            joint_handles: HandleRegistry::new(),
            body_to_scene_node: Vec::new(),
            rapier_to_bloom_body: std::collections::HashMap::new(),
            accumulator: 0.0,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 4,
            collision_events: Vec::new(),
            last_ray_hit: None,
            last_collision_read: (0.0, 0.0, false),
        }
    }

    pub fn set_gravity(&mut self, gx: f32, gy: f32, gz: f32) {
        self.gravity = Vector::new(gx, gy, gz);
    }

    pub fn set_timestep(&mut self, dt: f64, max_substeps: u32) {
        self.fixed_dt = dt;
        self.max_substeps = max_substeps;
    }

    // ============================================================
    // Rigid bodies
    // ============================================================

    pub fn create_body(
        &mut self,
        body_type: f64,
        px: f64, py: f64, pz: f64,
        rx: f64, ry: f64, rz: f64, rw: f64,
    ) -> f64 {
        let rb_type = match body_type as u32 {
            1 => RigidBodyType::Fixed,
            2 => RigidBodyType::KinematicPositionBased,
            3 => RigidBodyType::KinematicVelocityBased,
            _ => RigidBodyType::Dynamic,
        };

        let rotation = Rotation::from_xyzw(rx as f32, ry as f32, rz as f32, rw as f32);
        let position = Pose::from_parts(
            Vector::new(px as f32, py as f32, pz as f32),
            rotation,
        );

        let mut builder = RigidBodyBuilder::new(rb_type);
        builder.position = position;
        let rb = builder.build();

        let rapier_handle = self.rigid_body_set.insert(rb);
        let bloom_handle = self.body_handles.alloc(rapier_handle);

        // Grow scene node mapping
        let slot = bloom_handle as usize;
        if slot > self.body_to_scene_node.len() {
            self.body_to_scene_node.resize(slot, None);
        }

        self.rapier_to_bloom_body.insert(rapier_handle, bloom_handle);
        bloom_handle
    }

    pub fn destroy_body(&mut self, handle: f64) {
        if let Some(rapier_handle) = self.body_handles.free(handle) {
            self.rapier_to_bloom_body.remove(&rapier_handle);
            self.rigid_body_set.remove(
                rapier_handle,
                &mut self.island_manager,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                true,
            );
            let slot = handle as usize;
            if slot > 0 && slot <= self.body_to_scene_node.len() {
                self.body_to_scene_node[slot - 1] = None;
            }
        }
    }

    pub fn set_body_enabled(&mut self, handle: f64, enabled: bool) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.set_enabled(enabled);
            }
        }
    }

    pub fn set_body_ccd(&mut self, handle: f64, enabled: bool) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.enable_ccd(enabled);
            }
        }
    }

    pub fn set_body_gravity_scale(&mut self, handle: f64, scale: f32) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.set_gravity_scale(scale, true);
            }
        }
    }

    pub fn set_kinematic_target(
        &mut self, handle: f64,
        px: f64, py: f64, pz: f64,
        rx: f64, ry: f64, rz: f64, rw: f64,
    ) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                let rotation = Rotation::from_xyzw(rx as f32, ry as f32, rz as f32, rw as f32);
                let pose = Pose::from_parts(
                    Vector::new(px as f32, py as f32, pz as f32),
                    rotation,
                );
                body.set_next_kinematic_position(pose);
            }
        }
    }

    pub fn lock_rotations(&mut self, handle: f64, lock_x: bool, lock_y: bool, lock_z: bool) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.set_enabled_rotations(!lock_x, !lock_y, !lock_z, true);
            }
        }
    }

    // ============================================================
    // Colliders
    // ============================================================

    fn attach_collider(&mut self, body_handle: f64, collider: Collider) -> f64 {
        let Some(&rapier_body) = self.body_handles.get(body_handle) else {
            return 0.0;
        };
        let rapier_collider = self.collider_set.insert_with_parent(
            collider, rapier_body, &mut self.rigid_body_set,
        );
        self.collider_handles.alloc(rapier_collider)
    }

    pub fn add_box_collider(&mut self, body: f64, hx: f32, hy: f32, hz: f32) -> f64 {
        let collider = ColliderBuilder::cuboid(hx, hy, hz).build();
        self.attach_collider(body, collider)
    }

    pub fn add_sphere_collider(&mut self, body: f64, radius: f32) -> f64 {
        let collider = ColliderBuilder::ball(radius).build();
        self.attach_collider(body, collider)
    }

    pub fn add_capsule_collider(&mut self, body: f64, half_height: f32, radius: f32) -> f64 {
        let collider = ColliderBuilder::capsule_y(half_height, radius).build();
        self.attach_collider(body, collider)
    }

    pub fn add_cylinder_collider(&mut self, body: f64, half_height: f32, radius: f32) -> f64 {
        let collider = ColliderBuilder::cylinder(half_height, radius).build();
        self.attach_collider(body, collider)
    }

    pub fn set_collider_properties(
        &mut self, handle: f64,
        friction: f32, restitution: f32, density: f32,
    ) {
        if let Some(&rapier_handle) = self.collider_handles.get(handle) {
            if let Some(collider) = self.collider_set.get_mut(rapier_handle) {
                collider.set_friction(friction);
                collider.set_restitution(restitution);
                collider.set_density(density);
            }
        }
    }

    // ============================================================
    // Forces / velocities
    // ============================================================

    pub fn apply_force(&mut self, handle: f64, fx: f32, fy: f32, fz: f32) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.add_force(Vector::new(fx, fy, fz), true);
            }
        }
    }

    pub fn apply_impulse(&mut self, handle: f64, ix: f32, iy: f32, iz: f32) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.apply_impulse(Vector::new(ix, iy, iz), true);
            }
        }
    }

    pub fn apply_torque(&mut self, handle: f64, tx: f32, ty: f32, tz: f32) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.add_torque(Vector::new(tx, ty, tz), true);
            }
        }
    }

    pub fn apply_torque_impulse(&mut self, handle: f64, tx: f32, ty: f32, tz: f32) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.apply_torque_impulse(Vector::new(tx, ty, tz), true);
            }
        }
    }

    pub fn set_linear_velocity(&mut self, handle: f64, vx: f32, vy: f32, vz: f32) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.set_linvel(Vector::new(vx, vy, vz), true);
            }
        }
    }

    pub fn set_angular_velocity(&mut self, handle: f64, vx: f32, vy: f32, vz: f32) {
        if let Some(&rapier_handle) = self.body_handles.get(handle) {
            if let Some(body) = self.rigid_body_set.get_mut(rapier_handle) {
                body.set_angvel(Vector::new(vx, vy, vz), true);
            }
        }
    }

    // ============================================================
    // Queries (position / rotation / velocity)
    // ============================================================

    fn get_body(&self, handle: f64) -> Option<&RigidBody> {
        let rapier_handle = self.body_handles.get(handle)?;
        self.rigid_body_set.get(*rapier_handle)
    }

    pub fn get_body_position_x(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.translation().x as f64)
    }
    pub fn get_body_position_y(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.translation().y as f64)
    }
    pub fn get_body_position_z(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.translation().z as f64)
    }

    pub fn get_body_rotation_x(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.rotation().x as f64)
    }
    pub fn get_body_rotation_y(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.rotation().y as f64)
    }
    pub fn get_body_rotation_z(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.rotation().z as f64)
    }
    pub fn get_body_rotation_w(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(1.0, |b| b.rotation().w as f64)
    }

    pub fn get_linear_velocity_x(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.linvel().x as f64)
    }
    pub fn get_linear_velocity_y(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.linvel().y as f64)
    }
    pub fn get_linear_velocity_z(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.linvel().z as f64)
    }

    pub fn get_angular_velocity_x(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.angvel().x as f64)
    }
    pub fn get_angular_velocity_y(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.angvel().y as f64)
    }
    pub fn get_angular_velocity_z(&self, handle: f64) -> f64 {
        self.get_body(handle).map_or(0.0, |b| b.angvel().z as f64)
    }

    // ============================================================
    // Stepping
    // ============================================================

    pub fn step(&mut self, dt: f64) {
        self.collision_events.clear();

        // Fixed-timestep accumulator
        self.accumulator += dt;
        let mut steps = 0u32;

        while self.accumulator >= self.fixed_dt && steps < self.max_substeps {
            self.integration_parameters.dt = self.fixed_dt as f32;

            self.physics_pipeline.step(
                self.gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.rigid_body_set,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                &mut self.ccd_solver,
                &(),
                &(),
            );

            self.accumulator -= self.fixed_dt;
            steps += 1;
        }

        // Cap accumulator to prevent spiral of death
        if self.accumulator > self.fixed_dt {
            self.accumulator = 0.0;
        }

        self.collect_collision_events();
    }

    fn collect_collision_events(&mut self) {
        for pair in self.narrow_phase.contact_pairs() {
            if !pair.has_any_active_contact() {
                continue;
            }
            let body_a_rapier = self.collider_set.get(pair.collider1)
                .and_then(|c| c.parent());
            let body_b_rapier = self.collider_set.get(pair.collider2)
                .and_then(|c| c.parent());

            if let (Some(a), Some(b)) = (body_a_rapier, body_b_rapier) {
                let bloom_a = self.rapier_to_bloom_body.get(&a).copied().unwrap_or(0.0);
                let bloom_b = self.rapier_to_bloom_body.get(&b).copied().unwrap_or(0.0);
                if bloom_a > 0.0 && bloom_b > 0.0 {
                    self.collision_events.push(CollisionEvent {
                        body_a: bloom_a,
                        body_b: bloom_b,
                        started: true,
                    });
                }
            }
        }
    }

    // ============================================================
    // Scene node attachment + transform sync
    // ============================================================

    pub fn attach_scene_node(&mut self, body_handle: f64, scene_node_handle: f64) {
        let slot = body_handle as usize;
        if slot == 0 { return; }
        if slot > self.body_to_scene_node.len() {
            self.body_to_scene_node.resize(slot, None);
        }
        self.body_to_scene_node[slot - 1] = Some(scene_node_handle);
    }

    pub fn sync_transforms(&self, scene: &mut SceneGraph) {
        for (bloom_handle, rapier_handle) in self.body_handles.iter() {
            let slot = bloom_handle as usize;
            if slot == 0 || slot > self.body_to_scene_node.len() {
                continue;
            }
            let scene_node_handle = match self.body_to_scene_node[slot - 1] {
                Some(h) if h > 0.0 => h,
                _ => continue,
            };
            if let Some(body) = self.rigid_body_set.get(*rapier_handle) {
                let mat = pose_to_mat4(body.position());
                scene.set_transform(scene_node_handle, mat);
            }
        }
    }

    // ============================================================
    // Raycasting
    // ============================================================

    pub fn raycast(
        &mut self,
        ox: f64, oy: f64, oz: f64,
        dx: f64, dy: f64, dz: f64,
        max_dist: f64,
    ) -> bool {
        let ray = Ray::new(
            Vector::new(ox as f32, oy as f32, oz as f32),
            Vector::new(dx as f32, dy as f32, dz as f32),
        );

        // Create a temporary query pipeline from the broad phase
        let query_pipeline = self.broad_phase.as_query_pipeline(
            &DefaultQueryDispatcher,
            &self.rigid_body_set,
            &self.collider_set,
            QueryFilter::default(),
        );

        let hit = query_pipeline.cast_ray_and_get_normal(
            &ray,
            max_dist as f32,
            true,
        );

        if let Some((collider_handle, intersection)) = hit {
            let point = ray.point_at(intersection.time_of_impact);
            let body_handle = self.collider_set.get(collider_handle)
                .and_then(|c| c.parent())
                .and_then(|rh| self.rapier_to_bloom_body.get(&rh).copied())
                .unwrap_or(0.0);

            self.last_ray_hit = Some(RayHitResult {
                body_handle,
                distance: intersection.time_of_impact as f64,
                point: [point.x as f64, point.y as f64, point.z as f64],
                normal: [
                    intersection.normal.x as f64,
                    intersection.normal.y as f64,
                    intersection.normal.z as f64,
                ],
            });
            true
        } else {
            self.last_ray_hit = None;
            false
        }
    }

    // ============================================================
    // Joints
    // ============================================================

    pub fn create_fixed_joint(
        &mut self,
        body_a: f64, body_b: f64,
        ax: f32, ay: f32, az: f32,
        bx: f32, by: f32, bz: f32,
    ) -> f64 {
        let Some(&ra) = self.body_handles.get(body_a) else { return 0.0 };
        let Some(&rb) = self.body_handles.get(body_b) else { return 0.0 };

        let joint = FixedJointBuilder::new()
            .local_anchor1(Vector::new(ax, ay, az))
            .local_anchor2(Vector::new(bx, by, bz))
            .build();

        let rapier_handle = self.impulse_joint_set.insert(ra, rb, joint, true);
        self.joint_handles.alloc(rapier_handle)
    }

    pub fn create_revolute_joint(
        &mut self,
        body_a: f64, body_b: f64,
        ax: f32, ay: f32, az: f32,
        axis_x: f32, axis_y: f32, axis_z: f32,
    ) -> f64 {
        let Some(&ra) = self.body_handles.get(body_a) else { return 0.0 };
        let Some(&rb) = self.body_handles.get(body_b) else { return 0.0 };

        let axis = Vector::new(axis_x, axis_y, axis_z).normalize();
        let joint = RevoluteJointBuilder::new(axis)
            .local_anchor1(Vector::new(ax, ay, az))
            .local_anchor2(Vector::ZERO)
            .build();

        let rapier_handle = self.impulse_joint_set.insert(ra, rb, joint, true);
        self.joint_handles.alloc(rapier_handle)
    }

    pub fn create_prismatic_joint(
        &mut self,
        body_a: f64, body_b: f64,
        ax: f32, ay: f32, az: f32,
        axis_x: f32, axis_y: f32, axis_z: f32,
    ) -> f64 {
        let Some(&ra) = self.body_handles.get(body_a) else { return 0.0 };
        let Some(&rb) = self.body_handles.get(body_b) else { return 0.0 };

        let axis = Vector::new(axis_x, axis_y, axis_z).normalize();
        let joint = PrismaticJointBuilder::new(axis)
            .local_anchor1(Vector::new(ax, ay, az))
            .local_anchor2(Vector::ZERO)
            .build();

        let rapier_handle = self.impulse_joint_set.insert(ra, rb, joint, true);
        self.joint_handles.alloc(rapier_handle)
    }

    pub fn destroy_joint(&mut self, handle: f64) {
        if let Some(rapier_handle) = self.joint_handles.free(handle) {
            self.impulse_joint_set.remove(rapier_handle, true);
        }
    }
}

// ============================================================
// Pose (translation + rotation) -> column-major 4x4 matrix
// ============================================================

fn pose_to_mat4(pose: &Pose) -> [[f32; 4]; 4] {
    // Pose3::to_mat4() returns a glam::Mat4 which is column-major
    let m = pose.to_mat4();
    let cols = m.to_cols_array_2d();
    cols
}
