//! Raw FFI bindings to the `bloom_jolt` C++ shim.
//!
//! One-to-one mapping of `native/third_party/bloom_jolt/include/bloom_jolt.h`.
//! Higher-level Rust types and the Perry-facing FFI live in `physics_jolt.rs`.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::c_char;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

pub type bj_world      = u64;
pub type bj_shape      = u64;
pub type bj_body       = u64;
pub type bj_constraint = u64;
pub type bj_character  = u64;
pub type bj_vehicle    = u64;

pub const BJ_INVALID: u64 = 0;
pub const BJ_MAX_OBJECT_LAYERS: u32 = 16;

// ---------------------------------------------------------------------------
// Enums (match C `typedef enum`; layout: C int = 4 bytes on all supported targets)
// ---------------------------------------------------------------------------

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BjMotionType {
    Static    = 0,
    Kinematic = 1,
    Dynamic   = 2,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BjDefaultLayer {
    NonMoving = 0,
    Moving    = 1,
    Sensor    = 2,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BjActivation {
    Activate     = 0,
    DontActivate = 1,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BjContactEvent {
    Added     = 0,
    Persisted = 1,
    Removed   = 2,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BjResult {
    Ok                = 0,
    ErrUninitialized  = 1,
    ErrInvalidHandle  = 2,
    ErrOutOfMemory    = 3,
    ErrInvalidArg     = 4,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BjGroundState {
    OnGround     = 0,
    OnSteep      = 1,
    NotSupported = 2,
    InAir        = 3,
}

// ---------------------------------------------------------------------------
// POD structs
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct BjVec3 { pub x: f32, pub y: f32, pub z: f32 }

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BjQuat { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
impl Default for BjQuat {
    fn default() -> Self { Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 } }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct BjTransform { pub position: BjVec3, pub rotation: BjQuat }

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BjWorldDesc {
    pub max_bodies: u32,
    pub max_body_pairs: u32,
    pub max_contact_constraints: u32,
    pub num_threads: u32,
    pub gravity: BjVec3,
    pub temp_allocator_bytes: u32,
}
impl Default for BjWorldDesc {
    fn default() -> Self {
        Self {
            max_bodies: 65_536,
            max_body_pairs: 65_536,
            max_contact_constraints: 10_240,
            num_threads: 0,
            gravity: BjVec3 { x: 0.0, y: -9.81, z: 0.0 },
            temp_allocator_bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BjBodyDesc {
    pub motion_type: BjMotionType,
    pub position: BjVec3,
    pub rotation: BjQuat,
    pub linear_velocity: BjVec3,
    pub angular_velocity: BjVec3,
    pub gravity_factor: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub friction: f32,
    pub restitution: f32,
    pub mass_override: f32,
    pub inertia_diag_override: BjVec3,
    pub object_layer: u32,
    pub is_sensor: u8,
    pub allow_sleeping: u8,
    pub use_ccd: u8,
    pub start_awake: u8,
    pub user_data: u64,
}
impl Default for BjBodyDesc {
    fn default() -> Self {
        Self {
            motion_type: BjMotionType::Dynamic,
            position: BjVec3::default(),
            rotation: BjQuat::default(),
            linear_velocity: BjVec3::default(),
            angular_velocity: BjVec3::default(),
            gravity_factor: 1.0,
            linear_damping: 0.05,
            angular_damping: 0.05,
            friction: 0.2,
            restitution: 0.0,
            mass_override: 0.0,
            inertia_diag_override: BjVec3::default(),
            object_layer: BjDefaultLayer::Moving as u32,
            is_sensor: 0,
            allow_sleeping: 1,
            use_ccd: 0,
            start_awake: 1,
            user_data: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct BjRayHit {
    pub body: bj_body,
    pub point: BjVec3,
    pub normal: BjVec3,
    pub fraction: f32,
    pub sub_shape_id: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BjContact {
    pub event: BjContactEvent,
    pub body_a: bj_body,
    pub body_b: bj_body,
    pub point_a: BjVec3,
    pub point_b: BjVec3,
    pub normal: BjVec3,
    pub penetration_depth: f32,
    pub combined_friction: f32,
    pub combined_restitution: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BjConstraintAnchors {
    pub body_a: bj_body,
    pub body_b: bj_body,
    pub anchor_a: BjVec3,
    pub anchor_b: BjVec3,
    pub use_world_space: u8,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BjVehicleDesc {
    pub up: BjVec3,
    pub forward: BjVec3,
    pub wheel_positions: [BjVec3; 4],  // front-left, front-right, rear-left, rear-right
    pub wheel_radius: f32,
    pub wheel_width: f32,
    pub suspension_min_length: f32,
    pub suspension_max_length: f32,
    pub max_steer_angle: f32,
    pub max_brake_torque: f32,
    pub max_handbrake_torque: f32,
    pub engine_max_torque: f32,
    pub max_pitch_roll_angle: f32,
    pub object_layer: u32,
}
impl Default for BjVehicleDesc {
    fn default() -> Self {
        // Sensible defaults for a typical compact car. Wheel positions
        // are for a ~3.8m × 1.6m wheelbase/track.
        Self {
            up:      BjVec3 { x: 0.0, y: 1.0, z: 0.0 },
            forward: BjVec3 { x: 0.0, y: 0.0, z: 1.0 },
            wheel_positions: [
                BjVec3 { x: -0.8, y: -0.4, z:  1.3 },  // FL
                BjVec3 { x:  0.8, y: -0.4, z:  1.3 },  // FR
                BjVec3 { x: -0.8, y: -0.4, z: -1.3 },  // RL
                BjVec3 { x:  0.8, y: -0.4, z: -1.3 },  // RR
            ],
            wheel_radius: 0.35,
            wheel_width:  0.2,
            suspension_min_length: 0.3,
            suspension_max_length: 0.5,
            max_steer_angle: 35.0_f32.to_radians(),
            max_brake_torque: 1500.0,
            max_handbrake_torque: 4000.0,
            engine_max_torque: 500.0,
            max_pitch_roll_angle: 60.0_f32.to_radians(),
            object_layer: BjDefaultLayer::Moving as u32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BjCharacterDesc {
    pub up: BjVec3,
    pub max_slope_angle: f32,
    pub character_padding: f32,
    pub penetration_recovery_speed: f32,
    pub predictive_contact_distance: f32,
    pub max_strength: f32,
    pub mass: f32,
    pub object_layer: u32,
}
impl Default for BjCharacterDesc {
    fn default() -> Self {
        Self {
            up: BjVec3 { x: 0.0, y: 1.0, z: 0.0 },
            max_slope_angle: 50.0_f32.to_radians(),
            character_padding: 0.02,
            penetration_recovery_speed: 1.0,
            predictive_contact_distance: 0.1,
            max_strength: 100.0,
            mass: 70.0,
            object_layer: BjDefaultLayer::Moving as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// extern "C" — matches bloom_jolt.h exactly.
// ---------------------------------------------------------------------------

extern "C" {
    // Global
    pub fn bj_global_init() -> BjResult;
    pub fn bj_global_shutdown();
    pub fn bj_version_string() -> *const c_char;

    // World
    pub fn bj_world_create(desc: *const BjWorldDesc) -> bj_world;
    pub fn bj_world_destroy(world: bj_world);
    pub fn bj_world_set_gravity(world: bj_world, gravity: BjVec3);
    pub fn bj_world_get_gravity(world: bj_world, out: *mut BjVec3);
    pub fn bj_world_optimize_broadphase(world: bj_world);
    pub fn bj_world_step(world: bj_world, delta_time: f32, collision_steps: u32) -> BjResult;
    pub fn bj_world_set_layer_collides(world: bj_world, a: u32, b: u32, collides: u8);
    pub fn bj_world_get_layer_collides(world: bj_world, a: u32, b: u32) -> u8;
    pub fn bj_world_body_count(world: bj_world) -> u32;
    pub fn bj_world_active_body_count(world: bj_world) -> u32;

    // Shapes
    pub fn bj_shape_box(half_extents: BjVec3, convex_radius: f32) -> bj_shape;
    pub fn bj_shape_sphere(radius: f32) -> bj_shape;
    pub fn bj_shape_capsule(half_height: f32, radius: f32) -> bj_shape;
    pub fn bj_shape_cylinder(half_height: f32, radius: f32, convex_radius: f32) -> bj_shape;
    pub fn bj_shape_convex_hull(points: *const BjVec3, count: u32, convex_radius: f32) -> bj_shape;
    pub fn bj_shape_mesh(
        vertices: *const BjVec3, vertex_count: u32,
        indices: *const u32, triangle_count: u32,
    ) -> bj_shape;
    pub fn bj_shape_heightfield(
        samples: *const f32, sample_count: u32,
        offset: BjVec3, scale: BjVec3, block_size: u32,
    ) -> bj_shape;
    pub fn bj_shape_compound_static(
        shapes: *const bj_shape, local_transforms: *const BjTransform, count: u32,
    ) -> bj_shape;
    pub fn bj_shape_scaled(base: bj_shape, scale: BjVec3) -> bj_shape;
    pub fn bj_shape_offset_com(base: bj_shape, offset: BjVec3) -> bj_shape;
    pub fn bj_shape_add_ref(shape: bj_shape);
    pub fn bj_shape_release(shape: bj_shape);
    pub fn bj_shape_get_local_bounds(shape: bj_shape, out_min: *mut BjVec3, out_max: *mut BjVec3);
    pub fn bj_shape_get_volume(shape: bj_shape) -> f32;

    // Bodies
    pub fn bj_body_create(world: bj_world, shape: bj_shape, desc: *const BjBodyDesc) -> bj_body;
    pub fn bj_body_destroy(world: bj_world, body: bj_body);
    pub fn bj_body_activate(world: bj_world, body: bj_body);
    pub fn bj_body_deactivate(world: bj_world, body: bj_body);
    pub fn bj_body_is_active(world: bj_world, body: bj_body) -> u8;
    pub fn bj_body_is_valid(world: bj_world, body: bj_body) -> u8;
    pub fn bj_body_get_transform(world: bj_world, body: bj_body, out: *mut BjTransform);
    pub fn bj_body_set_transform(world: bj_world, body: bj_body, xform: *const BjTransform, act: BjActivation);
    pub fn bj_body_get_position(world: bj_world, body: bj_body, out: *mut BjVec3);
    pub fn bj_body_get_rotation(world: bj_world, body: bj_body, out: *mut BjQuat);
    pub fn bj_body_set_position(world: bj_world, body: bj_body, pos: BjVec3, act: BjActivation);
    pub fn bj_body_set_rotation(world: bj_world, body: bj_body, rot: BjQuat, act: BjActivation);
    pub fn bj_body_move_kinematic(world: bj_world, body: bj_body, target: *const BjTransform, delta_time: f32);
    pub fn bj_body_get_linear_velocity(world: bj_world, body: bj_body, out: *mut BjVec3);
    pub fn bj_body_get_angular_velocity(world: bj_world, body: bj_body, out: *mut BjVec3);
    pub fn bj_body_set_linear_velocity(world: bj_world, body: bj_body, v: BjVec3);
    pub fn bj_body_set_angular_velocity(world: bj_world, body: bj_body, v: BjVec3);
    pub fn bj_body_get_point_velocity(world: bj_world, body: bj_body, world_point: BjVec3, out: *mut BjVec3);
    pub fn bj_body_add_force(world: bj_world, body: bj_body, force: BjVec3);
    pub fn bj_body_add_impulse(world: bj_world, body: bj_body, impulse: BjVec3);
    pub fn bj_body_add_torque(world: bj_world, body: bj_body, torque: BjVec3);
    pub fn bj_body_add_angular_impulse(world: bj_world, body: bj_body, impulse: BjVec3);
    pub fn bj_body_add_force_at(world: bj_world, body: bj_body, force: BjVec3, world_point: BjVec3);
    pub fn bj_body_add_impulse_at(world: bj_world, body: bj_body, impulse: BjVec3, world_point: BjVec3);
    pub fn bj_body_set_friction(world: bj_world, body: bj_body, friction: f32);
    pub fn bj_body_set_restitution(world: bj_world, body: bj_body, restitution: f32);
    pub fn bj_body_set_linear_damping(world: bj_world, body: bj_body, damping: f32);
    pub fn bj_body_set_angular_damping(world: bj_world, body: bj_body, damping: f32);
    pub fn bj_body_set_gravity_factor(world: bj_world, body: bj_body, factor: f32);
    pub fn bj_body_set_ccd(world: bj_world, body: bj_body, enabled: u8);
    pub fn bj_body_set_motion_type(world: bj_world, body: bj_body, mtype: BjMotionType, act: BjActivation);
    pub fn bj_body_set_object_layer(world: bj_world, body: bj_body, layer: u32);
    pub fn bj_body_set_is_sensor(world: bj_world, body: bj_body, enabled: u8);
    pub fn bj_body_set_allow_sleeping(world: bj_world, body: bj_body, enabled: u8);
    pub fn bj_body_set_shape(world: bj_world, body: bj_body, shape: bj_shape, update_mass: u8, act: BjActivation);
    pub fn bj_body_lock_rotation_axes(world: bj_world, body: bj_body, lock_x: u8, lock_y: u8, lock_z: u8);
    pub fn bj_body_lock_translation_axes(world: bj_world, body: bj_body, lock_x: u8, lock_y: u8, lock_z: u8);
    pub fn bj_body_get_mass(world: bj_world, body: bj_body) -> f32;
    pub fn bj_body_get_friction(world: bj_world, body: bj_body) -> f32;
    pub fn bj_body_get_restitution(world: bj_world, body: bj_body) -> f32;
    pub fn bj_body_get_object_layer(world: bj_world, body: bj_body) -> u32;
    pub fn bj_body_set_user_data(world: bj_world, body: bj_body, user_data: u64);
    pub fn bj_body_get_user_data(world: bj_world, body: bj_body) -> u64;
    pub fn bj_world_sync_transforms(
        world: bj_world, bodies: *const bj_body, count: u32,
        out_transforms: *mut BjTransform,
    ) -> u32;

    // Queries
    pub fn bj_query_raycast_closest(
        world: bj_world, origin: BjVec3, direction: BjVec3, max_distance: f32,
        layer_mask: u32, out_hit: *mut BjRayHit,
    ) -> u8;
    pub fn bj_query_raycast_all(
        world: bj_world, origin: BjVec3, direction: BjVec3, max_distance: f32,
        layer_mask: u32, out_hits: *mut BjRayHit, max_hits: u32,
    ) -> u32;
    pub fn bj_query_shape_cast_closest(
        world: bj_world, shape: bj_shape, start: *const BjTransform, direction: BjVec3,
        layer_mask: u32, out_hit: *mut BjRayHit,
    ) -> u8;
    pub fn bj_query_overlap_sphere(
        world: bj_world, center: BjVec3, radius: f32,
        layer_mask: u32, out_bodies: *mut bj_body, max_results: u32,
    ) -> u32;
    pub fn bj_query_overlap_box(
        world: bj_world, box_transform: *const BjTransform, half_extents: BjVec3,
        layer_mask: u32, out_bodies: *mut bj_body, max_results: u32,
    ) -> u32;
    pub fn bj_query_overlap_point(
        world: bj_world, point: BjVec3,
        layer_mask: u32, out_bodies: *mut bj_body, max_results: u32,
    ) -> u32;

    // Constraints
    pub fn bj_constraint_fixed(world: bj_world, a: *const BjConstraintAnchors) -> bj_constraint;
    pub fn bj_constraint_point(world: bj_world, a: *const BjConstraintAnchors) -> bj_constraint;
    pub fn bj_constraint_hinge(
        world: bj_world, a: *const BjConstraintAnchors, axis: BjVec3,
        limit_min: f32, limit_max: f32,
    ) -> bj_constraint;
    pub fn bj_constraint_slider(
        world: bj_world, a: *const BjConstraintAnchors, axis: BjVec3,
        limit_min: f32, limit_max: f32,
    ) -> bj_constraint;
    pub fn bj_constraint_distance(
        world: bj_world, a: *const BjConstraintAnchors,
        min_distance: f32, max_distance: f32,
    ) -> bj_constraint;
    pub fn bj_constraint_six_dof(
        world: bj_world, a: *const BjConstraintAnchors,
        translation_limits_xyz_minmax: *const f32,
        rotation_limits_xyz_minmax: *const f32,
    ) -> bj_constraint;
    pub fn bj_constraint_destroy(world: bj_world, c: bj_constraint);
    pub fn bj_constraint_set_enabled(world: bj_world, c: bj_constraint, enabled: u8);

    // Contact events
    pub fn bj_world_contact_count(world: bj_world) -> u32;
    pub fn bj_world_pop_contacts(world: bj_world, out: *mut BjContact, max_out: u32) -> u32;
    pub fn bj_world_clear_contacts(world: bj_world);

    // Character controller
    pub fn bj_character_create(
        world: bj_world, shape: bj_shape, desc: *const BjCharacterDesc,
        position: BjVec3, rotation: BjQuat,
    ) -> bj_character;
    pub fn bj_character_destroy(world: bj_world, character: bj_character);
    pub fn bj_character_update(world: bj_world, character: bj_character,
                               delta_time: f32, gravity: BjVec3);
    pub fn bj_character_get_position(world: bj_world, character: bj_character, out: *mut BjVec3);
    pub fn bj_character_get_rotation(world: bj_world, character: bj_character, out: *mut BjQuat);
    pub fn bj_character_set_position(world: bj_world, character: bj_character, position: BjVec3);
    pub fn bj_character_set_rotation(world: bj_world, character: bj_character, rotation: BjQuat);
    pub fn bj_character_get_linear_velocity(world: bj_world, character: bj_character, out: *mut BjVec3);
    pub fn bj_character_set_linear_velocity(world: bj_world, character: bj_character, velocity: BjVec3);
    pub fn bj_character_get_ground_state(world: bj_world, character: bj_character) -> BjGroundState;
    pub fn bj_character_get_ground_normal  (world: bj_world, character: bj_character, out: *mut BjVec3);
    pub fn bj_character_get_ground_position(world: bj_world, character: bj_character, out: *mut BjVec3);
    pub fn bj_character_get_ground_body    (world: bj_world, character: bj_character) -> bj_body;
    pub fn bj_character_set_shape          (world: bj_world, character: bj_character, shape: bj_shape);

    // Soft bodies
    pub fn bj_soft_body_create(
        world: bj_world,
        vertex_data: *const f32, vertex_count: u32,
        indices:     *const u32, triangle_count: u32,
        position: BjVec3, rotation: BjQuat,
        object_layer: u32,
        edge_compliance: f32, gravity_factor: f32, linear_damping: f32, pressure: f32,
    ) -> bj_body;
    pub fn bj_soft_body_vertex_count(world: bj_world, body: bj_body) -> u32;
    pub fn bj_soft_body_get_vertex(world: bj_world, body: bj_body, idx: u32, out: *mut BjVec3);
    pub fn bj_soft_body_set_vertex(world: bj_world, body: bj_body, idx: u32, position: BjVec3);
    pub fn bj_soft_body_set_vertex_inv_mass(world: bj_world, body: bj_body, idx: u32, inv_mass: f32);

    // Wheeled vehicles
    pub fn bj_vehicle_create(
        world: bj_world, chassis_shape: bj_shape,
        desc: *const BjVehicleDesc,
        position: BjVec3, rotation: BjQuat,
    ) -> bj_vehicle;
    pub fn bj_vehicle_destroy(world: bj_world, vehicle: bj_vehicle);
    pub fn bj_vehicle_get_chassis(world: bj_world, vehicle: bj_vehicle) -> bj_body;
    pub fn bj_vehicle_set_input(
        world: bj_world, vehicle: bj_vehicle,
        forward: f32, right: f32, brake: f32, handbrake: f32,
    );
    pub fn bj_vehicle_get_wheel_transform(
        world: bj_world, vehicle: bj_vehicle, wheel_index: u32, axis: u32,
    ) -> f32;
    pub fn bj_vehicle_get_engine_rpm(world: bj_world, vehicle: bj_vehicle) -> f32;
    pub fn bj_vehicle_get_wheel_angular_velocity(world: bj_world, vehicle: bj_vehicle, wheel_index: u32) -> f32;
}

// ---------------------------------------------------------------------------
// Thin safe wrapper over version string — useful for logs / diagnostics.
// ---------------------------------------------------------------------------

pub fn version_string() -> &'static str {
    // SAFETY: bj_version_string returns a pointer to a static const char[].
    unsafe {
        let ptr = bj_version_string();
        if ptr.is_null() {
            return "unknown";
        }
        std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}

// ---------------------------------------------------------------------------
// Runtime smoke test — verifies that the shim initialises, creates a world,
// steps it, and tears down cleanly. Flipped on by `cargo test --features jolt`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_init_and_world_step() {
        unsafe {
            assert_eq!(bj_global_init(), BjResult::Ok);
            let version = version_string();
            assert!(version.contains("Jolt"), "got version: {}", version);

            let desc = BjWorldDesc::default();
            let world = bj_world_create(&desc);
            assert_ne!(world, BJ_INVALID, "world creation failed");

            let mut g = BjVec3::default();
            bj_world_get_gravity(world, &mut g);
            assert!((g.y - (-9.81)).abs() < 1e-4, "gravity.y = {}", g.y);

            for _ in 0..4 {
                assert_eq!(bj_world_step(world, 1.0 / 60.0, 1), BjResult::Ok);
            }

            assert_eq!(bj_world_body_count(world), 0);
            assert_eq!(bj_world_active_body_count(world), 0);

            bj_world_destroy(world);
            bj_global_shutdown();
        }
    }

    /// End-to-end: sphere at y=10 falls, lands on a static ground box.
    /// Validates shape factories, body lifecycle, gravity, contacts, stepping.
    #[test]
    fn gravity_fall_and_land() {
        unsafe {
            assert_eq!(bj_global_init(), BjResult::Ok);

            let desc = BjWorldDesc::default();
            let world = bj_world_create(&desc);
            assert_ne!(world, BJ_INVALID);

            // Ground: static box, top surface at y=0 (half-height 0.5, centred y=-0.5).
            let ground_shape = bj_shape_box(BjVec3 { x: 50.0, y: 0.5, z: 50.0 }, 0.05);
            assert_ne!(ground_shape, BJ_INVALID);
            let ground_desc = BjBodyDesc {
                motion_type: BjMotionType::Static,
                position: BjVec3 { x: 0.0, y: -0.5, z: 0.0 },
                object_layer: BjDefaultLayer::NonMoving as u32,
                ..Default::default()
            };
            let ground = bj_body_create(world, ground_shape, &ground_desc);
            assert_ne!(ground, BJ_INVALID);

            // Sphere: dynamic, starts at y=10, radius 0.5 → expected rest near y=0.5.
            let sphere_shape = bj_shape_sphere(0.5);
            assert_ne!(sphere_shape, BJ_INVALID);
            let sphere_desc = BjBodyDesc {
                motion_type: BjMotionType::Dynamic,
                position: BjVec3 { x: 0.0, y: 10.0, z: 0.0 },
                object_layer: BjDefaultLayer::Moving as u32,
                restitution: 0.0,
                friction: 0.5,
                ..Default::default()
            };
            let sphere = bj_body_create(world, sphere_shape, &sphere_desc);
            assert_ne!(sphere, BJ_INVALID);

            bj_world_optimize_broadphase(world);

            // Step for 2 seconds at 120 Hz so the body has time to settle.
            for _ in 0..240 {
                assert_eq!(bj_world_step(world, 1.0 / 120.0, 1), BjResult::Ok);
            }

            let mut xform = BjTransform::default();
            bj_body_get_transform(world, sphere, &mut xform);
            assert!(
                xform.position.y >= 0.4 && xform.position.y <= 1.0,
                "sphere landed at y={}, expected near 0.5 (radius atop ground)",
                xform.position.y
            );

            let mut lv = BjVec3::default();
            bj_body_get_linear_velocity(world, sphere, &mut lv);
            assert!(
                lv.y.abs() < 0.5,
                "sphere should be at rest, but v.y={}",
                lv.y
            );

            // Contacts should be populated after collision.
            assert!(
                bj_world_contact_count(world) > 0,
                "expected contact events from sphere/ground collision"
            );

            // Drain at least one contact and verify shape of event.
            let mut contact_buf: [BjContact; 8] = std::mem::zeroed();
            let drained = bj_world_pop_contacts(world, contact_buf.as_mut_ptr(), 8);
            assert!(drained > 0, "expected to drain at least one contact");

            bj_body_destroy(world, sphere);
            bj_body_destroy(world, ground);
            bj_shape_release(sphere_shape);
            bj_shape_release(ground_shape);
            bj_world_destroy(world);
            bj_global_shutdown();
        }
    }

    /// Character falls from y=5 onto a triangle-mesh floor, lands grounded.
    /// Validates mesh shapes (Tier 1 completion) + CharacterVirtual (Tier 2).
    #[test]
    fn character_on_mesh_ground() {
        unsafe {
            assert_eq!(bj_global_init(), BjResult::Ok);

            let world = bj_world_create(&BjWorldDesc::default());
            assert_ne!(world, BJ_INVALID);

            // Triangle-mesh floor: two triangles forming a 20×20 quad at y=0.
            let verts = [
                BjVec3 { x: -10.0, y: 0.0, z: -10.0 },
                BjVec3 { x:  10.0, y: 0.0, z: -10.0 },
                BjVec3 { x:  10.0, y: 0.0, z:  10.0 },
                BjVec3 { x: -10.0, y: 0.0, z:  10.0 },
            ];
            let indices: [u32; 6] = [0, 2, 1,   0, 3, 2];
            let ground_shape = bj_shape_mesh(
                verts.as_ptr(), verts.len() as u32,
                indices.as_ptr(), (indices.len() / 3) as u32,
            );
            assert_ne!(ground_shape, BJ_INVALID, "mesh shape creation failed");

            let ground_desc = BjBodyDesc {
                motion_type: BjMotionType::Static,
                position: BjVec3 { x: 0.0, y: 0.0, z: 0.0 },
                object_layer: BjDefaultLayer::NonMoving as u32,
                ..Default::default()
            };
            let ground = bj_body_create(world, ground_shape, &ground_desc);
            assert_ne!(ground, BJ_INVALID);

            // Character: capsule at y=5 looking down.
            let char_shape = bj_shape_capsule(0.5, 0.3);
            let desc = BjCharacterDesc::default();
            let character = bj_character_create(
                world, char_shape, &desc,
                BjVec3 { x: 0.0, y: 5.0, z: 0.0 },
                BjQuat::default(),
            );
            assert_ne!(character, BJ_INVALID);

            bj_world_optimize_broadphase(world);

            let gravity = BjVec3 { x: 0.0, y: -9.81, z: 0.0 };
            for _ in 0..240 {
                bj_character_update(world, character, 1.0 / 120.0, gravity);
                bj_world_step(world, 1.0 / 120.0, 1);
            }

            let mut pos = BjVec3::default();
            bj_character_get_position(world, character, &mut pos);
            assert!(
                pos.y < 5.0 && pos.y > -0.1,
                "character didn't land correctly: y={}", pos.y
            );

            let state = bj_character_get_ground_state(world, character);
            assert_eq!(
                state, BjGroundState::OnGround,
                "expected OnGround, got {:?}", state
            );

            bj_character_destroy(world, character);
            bj_body_destroy(world, ground);
            bj_shape_release(char_shape);
            bj_shape_release(ground_shape);
            bj_world_destroy(world);
            bj_global_shutdown();
        }
    }

    /// 3x3 cloth patch pinned at 4 corners: lets simulation run and verifies
    /// center vertex drops below starting height (gravity + edge constraints working).
    #[test]
    fn soft_body_cloth_sags_under_gravity() {
        unsafe {
            assert_eq!(bj_global_init(), BjResult::Ok);
            let world = bj_world_create(&BjWorldDesc::default());
            assert_ne!(world, BJ_INVALID);

            // 3x3 vertex grid, 2x2 quads = 8 triangles.
            // Corners pinned (invMass=0); center and edges have mass.
            let step = 1.0_f32;
            let mut vertex_data: Vec<f32> = Vec::with_capacity(9 * 4);
            for gy in 0..3 {
                for gx in 0..3 {
                    let x = (gx as f32 - 1.0) * step;
                    let y = 5.0;
                    let z = (gy as f32 - 1.0) * step;
                    let is_corner = (gx == 0 || gx == 2) && (gy == 0 || gy == 2);
                    let inv_mass = if is_corner { 0.0 } else { 1.0 };
                    vertex_data.extend_from_slice(&[x, y, z, inv_mass]);
                }
            }
            let mut indices: Vec<u32> = Vec::new();
            for gy in 0..2u32 {
                for gx in 0..2u32 {
                    let i0 = gy * 3 + gx;
                    let i1 = gy * 3 + gx + 1;
                    let i2 = (gy + 1) * 3 + gx;
                    let i3 = (gy + 1) * 3 + gx + 1;
                    indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
                }
            }

            // Compliance 1e-3 = realistically soft cloth (stiff cloth = 1e-5).
            let body = bj_soft_body_create(
                world,
                vertex_data.as_ptr(), 9,
                indices.as_ptr(), (indices.len() / 3) as u32,
                BjVec3::default(), BjQuat::default(),
                BjDefaultLayer::Moving as u32,
                1e-3, 1.0, 0.01, 0.0,
            );
            assert_ne!(body, BJ_INVALID, "soft body creation failed");

            bj_world_optimize_broadphase(world);

            // Center vertex is index 4 (middle of 3x3 grid).
            let mut start_center = BjVec3::default();
            bj_soft_body_get_vertex(world, body, 4, &mut start_center);
            let start_y = start_center.y;

            for _ in 0..480 {
                bj_world_step(world, 1.0 / 120.0, 1);
            }

            let count = bj_soft_body_vertex_count(world, body);
            assert_eq!(count, 9, "expected 9 vertices, got {}", count);

            let mut end_center = BjVec3::default();
            bj_soft_body_get_vertex(world, body, 4, &mut end_center);
            assert!(
                end_center.y < start_y - 0.1,
                "cloth center should sag below starting y={}; ended at y={}",
                start_y, end_center.y
            );

            // Corners (index 0, 2, 6, 8) should stay pinned near y=5.
            let mut corner = BjVec3::default();
            bj_soft_body_get_vertex(world, body, 0, &mut corner);
            assert!(
                (corner.y - 5.0).abs() < 0.01,
                "corner should be pinned at y=5, got {}", corner.y
            );

            bj_body_destroy(world, body);
            bj_world_destroy(world);
            bj_global_shutdown();
        }
    }

    /// Minimum-viable vehicle API smoke test: chassis spawns, engine responds
    /// to throttle, wheels spin, API doesn't crash.
    ///
    /// *Does not* assert car-drives-forward — getting a test car to accelerate
    /// reliably against Jolt's default parameters requires tuning the chassis
    /// geometry, suspension spring/damping, and wheel friction curves to match.
    /// That's best done per-game, not in a validation test. The critical signal
    /// this test captures is that the FFI stack (VehicleConstraint + step
    /// listener + controller input) wires up correctly.
    #[test]
    fn vehicle_api_smoke() {
        unsafe {
            assert_eq!(bj_global_init(), BjResult::Ok);
            let world = bj_world_create(&BjWorldDesc::default());
            assert_ne!(world, BJ_INVALID);

            let ground_shape = bj_shape_box(BjVec3 { x: 100.0, y: 0.5, z: 100.0 }, 0.05);
            let ground_desc = BjBodyDesc {
                motion_type: BjMotionType::Static,
                position: BjVec3 { x: 0.0, y: -0.5, z: 0.0 },
                object_layer: BjDefaultLayer::NonMoving as u32,
                friction: 0.8,
                ..Default::default()
            };
            let ground = bj_body_create(world, ground_shape, &ground_desc);

            let inner_box = bj_shape_box(BjVec3 { x: 1.0, y: 0.2, z: 1.9 }, 0.05);
            let chassis_shape = bj_shape_offset_com(inner_box, BjVec3 { x: 0.0, y: -0.6, z: 0.0 });
            let desc = BjVehicleDesc::default();
            let vehicle = bj_vehicle_create(
                world, chassis_shape, &desc,
                BjVec3 { x: 0.0, y: 2.0, z: 0.0 },
                BjQuat::default(),
            );
            assert_ne!(vehicle, BJ_INVALID, "vehicle creation failed");
            assert_ne!(bj_vehicle_get_chassis(world, vehicle), BJ_INVALID);

            bj_world_optimize_broadphase(world);

            // 1 second of throttled simulation — engine should spin up.
            for _ in 0..120 {
                bj_vehicle_set_input(world, vehicle, 1.0, 0.0, 0.0, 0.0);
                bj_world_step(world, 1.0 / 120.0, 1);
            }
            let rpm = bj_vehicle_get_engine_rpm(world, vehicle);
            assert!(rpm > 500.0, "engine RPM should rise under throttle; got {}", rpm);

            // Rear wheel should have non-zero angular velocity.
            let rear_omega = bj_vehicle_get_wheel_angular_velocity(world, vehicle, 2);
            assert!(rear_omega.abs() > 0.5, "rear wheel should be spinning; ω={}", rear_omega);

            // Wheel transform query returns non-zero position.
            let wheel_y = bj_vehicle_get_wheel_transform(world, vehicle, 0, 1);
            assert!(wheel_y.abs() > 0.01, "wheel 0 world Y should be non-zero; got {}", wheel_y);

            bj_vehicle_destroy(world, vehicle);
            bj_body_destroy(world, ground);
            bj_shape_release(chassis_shape);
            bj_shape_release(ground_shape);
            bj_world_destroy(world);
            bj_global_shutdown();
        }
    }
}
