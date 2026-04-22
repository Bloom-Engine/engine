/*
 * bloom_jolt.h — C ABI shim between Rust/Perry FFI and JoltPhysics 5.x
 *
 * This header is internal to the engine: Rust binds to it via extern "C", and
 * the implementation in src/bloom_jolt.cpp calls into Jolt C++. It is NOT the
 * Perry/TypeScript FFI surface — that sits one layer above in physics.rs and
 * only sees f64 / i64 because of Perry's calling convention.
 *
 * Design notes (Tier 1):
 *   - Shape / body separation: shapes are reusable and refcounted.
 *   - POD structs for transforms, descriptors, hit results — packed layouts.
 *   - Batch-friendly queries: raycast_all / sync_transforms take buffers.
 *   - Contact events are polled (drained) after each step, not pushed.
 *   - All handles are uint64_t; 0 == BJ_INVALID.
 *
 * Thread safety: one world per process initially. bj_world_step must run on
 * the thread that owns the world. Jolt internally threads broadphase +
 * integration across num_threads workers.
 */

#ifndef BLOOM_JOLT_H
#define BLOOM_JOLT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Handles                                                            */
/* ------------------------------------------------------------------ */

typedef uint64_t bj_world;
typedef uint64_t bj_shape;
typedef uint64_t bj_body;
typedef uint64_t bj_constraint;
typedef uint64_t bj_character;

#define BJ_INVALID ((uint64_t)0)

/* Max distinct object layers the layer matrix supports. */
#define BJ_MAX_OBJECT_LAYERS 16

/* ------------------------------------------------------------------ */
/*  Enums                                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    BJ_MOTION_STATIC    = 0,
    BJ_MOTION_KINEMATIC = 1,
    BJ_MOTION_DYNAMIC   = 2
} bj_motion_type;

/* Default object layers. Applications may use any value in [0, 15];
 * the layer collision matrix is configured via bj_world_set_layer_collides. */
typedef enum {
    BJ_LAYER_NON_MOVING = 0,
    BJ_LAYER_MOVING     = 1,
    BJ_LAYER_SENSOR     = 2
} bj_default_layer;

typedef enum {
    BJ_ACTIVATE      = 0,
    BJ_DONT_ACTIVATE = 1
} bj_activation;

typedef enum {
    BJ_CONTACT_ADDED     = 0,
    BJ_CONTACT_PERSISTED = 1,
    BJ_CONTACT_REMOVED   = 2
} bj_contact_event;

typedef enum {
    BJ_OK                 = 0,
    BJ_ERR_UNINITIALIZED  = 1,
    BJ_ERR_INVALID_HANDLE = 2,
    BJ_ERR_OUT_OF_MEMORY  = 3,
    BJ_ERR_INVALID_ARG    = 4
} bj_result;

/* ------------------------------------------------------------------ */
/*  POD structs                                                        */
/* ------------------------------------------------------------------ */

typedef struct { float x, y, z;       } bj_vec3;
typedef struct { float x, y, z, w;    } bj_quat;        /* xyzw */
typedef struct { bj_vec3 position;    bj_quat rotation; } bj_transform;

typedef struct {
    uint32_t max_bodies;              /* hard cap on simultaneous bodies */
    uint32_t max_body_pairs;          /* broadphase pair buffer */
    uint32_t max_contact_constraints; /* narrowphase contact buffer */
    uint32_t num_threads;             /* 0 = auto (CPU count - 1) */
    bj_vec3  gravity;
    uint32_t temp_allocator_bytes;    /* per-frame scratch; 0 = default 10MB */
} bj_world_desc;

typedef struct {
    bj_motion_type motion_type;
    bj_vec3   position;
    bj_quat   rotation;
    bj_vec3   linear_velocity;
    bj_vec3   angular_velocity;
    float     gravity_factor;
    float     linear_damping;
    float     angular_damping;
    float     friction;
    float     restitution;
    float     mass_override;          /* 0 = compute from shape density */
    bj_vec3   inertia_diag_override;  /* (0,0,0) = compute from shape */
    uint32_t  object_layer;
    uint8_t   is_sensor;              /* 1 = trigger (events only, no contact response) */
    uint8_t   allow_sleeping;
    uint8_t   use_ccd;                /* continuous collision detection */
    uint8_t   start_awake;
    uint64_t  user_data;              /* opaque; TS side uses for scene-node id */
} bj_body_desc;

typedef struct {
    bj_body  body;
    bj_vec3  point;       /* world-space */
    bj_vec3  normal;      /* world-space, points away from surface */
    float    fraction;    /* 0..1 along ray */
    uint32_t sub_shape_id;/* for compound / mesh shapes */
} bj_ray_hit;

typedef struct {
    bj_contact_event event;
    bj_body  body_a;
    bj_body  body_b;
    bj_vec3  point_a;             /* world-space contact on A */
    bj_vec3  point_b;             /* world-space contact on B */
    bj_vec3  normal;              /* world-space, from A to B */
    float    penetration_depth;
    float    combined_friction;
    float    combined_restitution;
} bj_contact;

/* ------------------------------------------------------------------ */
/*  Global init (call once per process before any world)               */
/* ------------------------------------------------------------------ */

bj_result bj_global_init(void);
void      bj_global_shutdown(void);
const char *bj_version_string(void);   /* "Jolt 5.5.0 / bloom_jolt 0.1" */

/* ------------------------------------------------------------------ */
/*  World                                                              */
/* ------------------------------------------------------------------ */

bj_world  bj_world_create(const bj_world_desc *desc);
void      bj_world_destroy(bj_world world);

void      bj_world_set_gravity(bj_world world, bj_vec3 gravity);
void      bj_world_get_gravity(bj_world world, bj_vec3 *out);

/* Must be called after initial body creation, before first step,
 * for best broadphase performance. Safe to call any time. */
void      bj_world_optimize_broadphase(bj_world world);

/* Step the simulation. collision_steps sub-divides the physics update
 * for stability; use 1 for a 60Hz game, 2+ for high-velocity or low-fps.
 * Returns BJ_OK or error code. */
bj_result bj_world_step(bj_world world, float delta_time, uint32_t collision_steps);

/* Layer collision matrix. Both layers must be in [0, BJ_MAX_OBJECT_LAYERS). */
void      bj_world_set_layer_collides(bj_world world, uint32_t layer_a, uint32_t layer_b, uint8_t collides);
uint8_t   bj_world_get_layer_collides(bj_world world, uint32_t layer_a, uint32_t layer_b);

/* Active-body count (useful for diagnostics / sleep tuning). */
uint32_t  bj_world_body_count(bj_world world);
uint32_t  bj_world_active_body_count(bj_world world);

/* ------------------------------------------------------------------ */
/*  Shapes (refcounted; created in AWAITING_REGISTRATION state)        */
/* ------------------------------------------------------------------ */

/* Primitives — convex_radius smooths sharp edges (Jolt default: 0.05). */
bj_shape bj_shape_box     (bj_vec3 half_extents, float convex_radius);
bj_shape bj_shape_sphere  (float radius);
bj_shape bj_shape_capsule (float half_height, float radius);
bj_shape bj_shape_cylinder(float half_height, float radius, float convex_radius);

/* Convex hull from point cloud. Points are copied. */
bj_shape bj_shape_convex_hull(const bj_vec3 *points, uint32_t count, float convex_radius);

/* Triangle mesh — static bodies only.
 * vertices: array of bj_vec3
 * indices:  triangle_count * 3 uint32 indices
 * Data is copied. */
bj_shape bj_shape_mesh(const bj_vec3 *vertices, uint32_t vertex_count,
                       const uint32_t *indices,  uint32_t triangle_count);

/* Heightfield — static bodies only.
 * samples: sample_count*sample_count row-major float grid (Y values)
 * offset:  world-space offset of the (0,0) sample
 * scale:   per-axis world-space scale of a grid cell (y is Y scale)
 * block_size: power-of-two cell block for BVH (recommended 4 or 8) */
bj_shape bj_shape_heightfield(const float *samples, uint32_t sample_count,
                              bj_vec3 offset, bj_vec3 scale, uint32_t block_size);

/* Compound (static layout, fast query) of other shapes. Child shapes are
 * add-ref'd internally. */
bj_shape bj_shape_compound_static(const bj_shape *shapes,
                                   const bj_transform *local_transforms,
                                   uint32_t count);

/* Wrappers (cheap, no geometry copy) */
bj_shape bj_shape_scaled    (bj_shape base, bj_vec3 scale);
bj_shape bj_shape_offset_com(bj_shape base, bj_vec3 center_of_mass_offset);

void     bj_shape_add_ref   (bj_shape shape);
void     bj_shape_release   (bj_shape shape);

/* Query shape properties */
void     bj_shape_get_local_bounds(bj_shape shape, bj_vec3 *out_min, bj_vec3 *out_max);
float    bj_shape_get_volume      (bj_shape shape);

/* ------------------------------------------------------------------ */
/*  Bodies                                                             */
/* ------------------------------------------------------------------ */

bj_body  bj_body_create (bj_world world, bj_shape shape, const bj_body_desc *desc);
void     bj_body_destroy(bj_world world, bj_body body);

/* Lifecycle */
void     bj_body_activate      (bj_world world, bj_body body);
void     bj_body_deactivate    (bj_world world, bj_body body);
uint8_t  bj_body_is_active     (bj_world world, bj_body body);
uint8_t  bj_body_is_valid      (bj_world world, bj_body body);

/* Transforms */
void     bj_body_get_transform (bj_world world, bj_body body, bj_transform *out);
void     bj_body_set_transform (bj_world world, bj_body body, const bj_transform *xform, bj_activation act);
void     bj_body_get_position  (bj_world world, bj_body body, bj_vec3 *out);
void     bj_body_get_rotation  (bj_world world, bj_body body, bj_quat *out);
void     bj_body_set_position  (bj_world world, bj_body body, bj_vec3 pos,   bj_activation act);
void     bj_body_set_rotation  (bj_world world, bj_body body, bj_quat rot,   bj_activation act);

/* Kinematic interpolation — body moves toward target over delta_time */
void     bj_body_move_kinematic(bj_world world, bj_body body, const bj_transform *target, float delta_time);

/* Velocities */
void     bj_body_get_linear_velocity (bj_world world, bj_body body, bj_vec3 *out);
void     bj_body_get_angular_velocity(bj_world world, bj_body body, bj_vec3 *out);
void     bj_body_set_linear_velocity (bj_world world, bj_body body, bj_vec3 v);
void     bj_body_set_angular_velocity(bj_world world, bj_body body, bj_vec3 v);
/* Velocity at a world-space point (useful for friction / effects) */
void     bj_body_get_point_velocity  (bj_world world, bj_body body, bj_vec3 world_point, bj_vec3 *out);

/* Forces & impulses (center of mass) */
void     bj_body_add_force           (bj_world world, bj_body body, bj_vec3 force);
void     bj_body_add_impulse         (bj_world world, bj_body body, bj_vec3 impulse);
void     bj_body_add_torque          (bj_world world, bj_body body, bj_vec3 torque);
void     bj_body_add_angular_impulse (bj_world world, bj_body body, bj_vec3 impulse);
/* Forces & impulses at a world-space point */
void     bj_body_add_force_at        (bj_world world, bj_body body, bj_vec3 force,   bj_vec3 world_point);
void     bj_body_add_impulse_at      (bj_world world, bj_body body, bj_vec3 impulse, bj_vec3 world_point);

/* Per-body properties (read/write) */
void     bj_body_set_friction         (bj_world world, bj_body body, float friction);
void     bj_body_set_restitution      (bj_world world, bj_body body, float restitution);
void     bj_body_set_linear_damping   (bj_world world, bj_body body, float damping);
void     bj_body_set_angular_damping  (bj_world world, bj_body body, float damping);
void     bj_body_set_gravity_factor   (bj_world world, bj_body body, float factor);
void     bj_body_set_ccd              (bj_world world, bj_body body, uint8_t enabled);
void     bj_body_set_motion_type      (bj_world world, bj_body body, bj_motion_type type, bj_activation act);
void     bj_body_set_object_layer     (bj_world world, bj_body body, uint32_t layer);
void     bj_body_set_is_sensor        (bj_world world, bj_body body, uint8_t enabled);
void     bj_body_set_allow_sleeping   (bj_world world, bj_body body, uint8_t enabled);
void     bj_body_set_shape            (bj_world world, bj_body body, bj_shape shape, uint8_t update_mass, bj_activation act);
void     bj_body_lock_rotation_axes   (bj_world world, bj_body body, uint8_t lock_x, uint8_t lock_y, uint8_t lock_z);
void     bj_body_lock_translation_axes(bj_world world, bj_body body, uint8_t lock_x, uint8_t lock_y, uint8_t lock_z);

float    bj_body_get_mass             (bj_world world, bj_body body);
float    bj_body_get_friction         (bj_world world, bj_body body);
float    bj_body_get_restitution      (bj_world world, bj_body body);
uint32_t bj_body_get_object_layer     (bj_world world, bj_body body);

/* User data pass-through (TS layer stores a scene-node id here). */
void     bj_body_set_user_data(bj_world world, bj_body body, uint64_t user_data);
uint64_t bj_body_get_user_data(bj_world world, bj_body body);

/* Batched transform sync: fills `out_transforms` for the given body handles.
 * Invalid handles receive an identity transform. Returns count written. */
uint32_t bj_world_sync_transforms(bj_world world,
                                  const bj_body *bodies, uint32_t count,
                                  bj_transform *out_transforms);

/* ------------------------------------------------------------------ */
/*  Queries                                                            */
/* ------------------------------------------------------------------ */

/* layer_mask: bitmask of object layers to test against (bit N = layer N).
 * Use 0xFFFFFFFFu for "all layers". */

uint8_t  bj_query_raycast_closest(bj_world world,
                                  bj_vec3 origin, bj_vec3 direction, float max_distance,
                                  uint32_t layer_mask, bj_ray_hit *out_hit);

uint32_t bj_query_raycast_all(bj_world world,
                              bj_vec3 origin, bj_vec3 direction, float max_distance,
                              uint32_t layer_mask,
                              bj_ray_hit *out_hits, uint32_t max_hits);

uint8_t  bj_query_shape_cast_closest(bj_world world, bj_shape shape,
                                     const bj_transform *start, bj_vec3 direction,
                                     uint32_t layer_mask, bj_ray_hit *out_hit);

uint32_t bj_query_overlap_sphere(bj_world world,
                                 bj_vec3 center, float radius,
                                 uint32_t layer_mask,
                                 bj_body *out_bodies, uint32_t max_results);

uint32_t bj_query_overlap_box(bj_world world,
                              const bj_transform *box_transform, bj_vec3 half_extents,
                              uint32_t layer_mask,
                              bj_body *out_bodies, uint32_t max_results);

uint32_t bj_query_overlap_point(bj_world world, bj_vec3 point,
                                uint32_t layer_mask,
                                bj_body *out_bodies, uint32_t max_results);

/* ------------------------------------------------------------------ */
/*  Constraints                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    bj_body body_a;
    bj_body body_b;             /* BJ_INVALID = world-fixed constraint on body_a */
    bj_vec3 anchor_a;           /* local to body_a if use_world_space==0, else world */
    bj_vec3 anchor_b;           /* local to body_b (or world if body_b invalid) */
    uint8_t use_world_space;
} bj_constraint_anchors;

bj_constraint bj_constraint_fixed   (bj_world world, const bj_constraint_anchors *a);
bj_constraint bj_constraint_point   (bj_world world, const bj_constraint_anchors *a);
/* hinge: limit_min >= limit_max disables the limits */
bj_constraint bj_constraint_hinge   (bj_world world, const bj_constraint_anchors *a,
                                      bj_vec3 axis, float limit_min, float limit_max);
bj_constraint bj_constraint_slider  (bj_world world, const bj_constraint_anchors *a,
                                      bj_vec3 axis, float limit_min, float limit_max);
bj_constraint bj_constraint_distance(bj_world world, const bj_constraint_anchors *a,
                                      float min_distance, float max_distance);
/* Six-DOF: per-axis free/limited/locked masks.
 * mask layout: bits 0-2 = translation x/y/z, 3-5 = rotation x/y/z.
 * For each axis, set (min,max) pair; min>=max means locked. Omitted via nullptr uses defaults. */
bj_constraint bj_constraint_six_dof (bj_world world, const bj_constraint_anchors *a,
                                      const float *translation_limits_xyz_minmax, /* 6 floats or NULL */
                                      const float *rotation_limits_xyz_minmax);   /* 6 floats or NULL */

void bj_constraint_destroy    (bj_world world, bj_constraint c);
void bj_constraint_set_enabled(bj_world world, bj_constraint c, uint8_t enabled);

/* ------------------------------------------------------------------ */
/*  Contact events (polled after bj_world_step)                        */
/* ------------------------------------------------------------------ */

uint32_t bj_world_contact_count(bj_world world);

/* Drains up to max_out events into `out`. Returns count written.
 * Call repeatedly until it returns 0 to fully drain. */
uint32_t bj_world_pop_contacts(bj_world world, bj_contact *out, uint32_t max_out);

/* Reset the event queue (e.g. when resetting a level). */
void     bj_world_clear_contacts(bj_world world);

/* ------------------------------------------------------------------ */
/*  Character controller (Tier 2)                                      */
/* ------------------------------------------------------------------ */
/* Kinematic character (Jolt's CharacterVirtual) with slope / step     */
/* handling — what player controllers actually feel like.              */

typedef enum {
    BJ_GROUND_ON_GROUND     = 0,  /* standing on walkable surface */
    BJ_GROUND_ON_STEEP      = 1,  /* slope exceeds mMaxSlopeAngle */
    BJ_GROUND_NOT_SUPPORTED = 2,  /* supported by a body in "NotSupported" state (e.g. sensor) */
    BJ_GROUND_IN_AIR        = 3
} bj_ground_state;

typedef struct {
    bj_vec3  up;                            /* default (0,1,0) */
    float    max_slope_angle;               /* radians; Jolt default ~0.87 (50°) */
    float    character_padding;             /* default 0.02 */
    float    penetration_recovery_speed;    /* default 1.0 */
    float    predictive_contact_distance;   /* default 0.1 */
    float    max_strength;                  /* max force character can apply to bodies, default 100 */
    float    mass;                          /* default 70 kg */
    uint32_t object_layer;                  /* default BJ_LAYER_MOVING */
} bj_character_desc;

bj_character bj_character_create(bj_world world, bj_shape shape,
                                 const bj_character_desc *desc,
                                 bj_vec3 position, bj_quat rotation);
void bj_character_destroy(bj_world world, bj_character character);

/* Drive one frame — handles collision, slope clamping, stair stepping. */
void bj_character_update(bj_world world, bj_character character,
                         float delta_time, bj_vec3 gravity);

void bj_character_get_position(bj_world world, bj_character character, bj_vec3 *out);
void bj_character_get_rotation(bj_world world, bj_character character, bj_quat *out);
void bj_character_set_position(bj_world world, bj_character character, bj_vec3 position);
void bj_character_set_rotation(bj_world world, bj_character character, bj_quat rotation);

void bj_character_get_linear_velocity(bj_world world, bj_character character, bj_vec3 *out);
void bj_character_set_linear_velocity(bj_world world, bj_character character, bj_vec3 velocity);

bj_ground_state bj_character_get_ground_state(bj_world world, bj_character character);
void bj_character_get_ground_normal  (bj_world world, bj_character character, bj_vec3 *out);
void bj_character_get_ground_position(bj_world world, bj_character character, bj_vec3 *out);
bj_body bj_character_get_ground_body (bj_world world, bj_character character);

void bj_character_set_shape(bj_world world, bj_character character, bj_shape shape);

/* ------------------------------------------------------------------ */
/*  Soft bodies (Tier 2)                                               */
/* ------------------------------------------------------------------ */
/* Soft bodies (cloth, rope, jelly) are Jolt SoftBodies — triangle     */
/* meshes with per-vertex mass + edge/bend constraints. Return value   */
/* is a regular bj_body that works with the normal destroy/get-pos     */
/* FFI; per-vertex queries are below.                                  */
/*                                                                     */
/* vertex_data layout: vertex_count * 4 floats per vertex:             */
/*   [x, y, z, inv_mass]                                               */
/* inv_mass == 0 means the vertex is PINNED (won't move under gravity).*/
/* indices: triangle_count * 3 uint32 triangle indices.                */

bj_body bj_soft_body_create(
    bj_world world,
    const float *vertex_data, uint32_t vertex_count,
    const uint32_t *indices,  uint32_t triangle_count,
    bj_vec3 position, bj_quat rotation,
    uint32_t object_layer,
    float edge_compliance,     /* 0 = rigid edges; >0 = soft. Typical cloth: 0.0001 */
    float gravity_factor,
    float linear_damping,
    float pressure             /* 0 = cloth/rope; >0 = inflated volume (balloon/jelly) */
);

uint32_t bj_soft_body_vertex_count(bj_world world, bj_body body);
/* World-space position of vertex `idx` (applies current body transform). */
void     bj_soft_body_get_vertex(bj_world world, bj_body body, uint32_t idx, bj_vec3 *out);
/* Override a vertex position — useful for pinning to a moving anchor. */
void     bj_soft_body_set_vertex(bj_world world, bj_body body, uint32_t idx, bj_vec3 position);
/* Set per-vertex inverse mass; inv_mass == 0 pins the vertex in world space. */
void     bj_soft_body_set_vertex_inv_mass(bj_world world, bj_body body, uint32_t idx, float inv_mass);

/* ------------------------------------------------------------------ */
/*  Wheeled vehicles (Tier 2)                                          */
/* ------------------------------------------------------------------ */
/* A 4-wheel car with a chassis body + VehicleConstraint. Ray collision*/
/* tester, rear-wheel drive + differential, front-wheel steering. For  */
/* tracked vehicles / motorcycles / custom configurations, we'll add   */
/* a lower-level surface later; this is the 90% case.                  */

typedef uint64_t bj_vehicle;

typedef struct {
    /* Up and forward axes (default up=(0,1,0), forward=(0,0,1)). */
    bj_vec3 up;
    bj_vec3 forward;

    /* Wheel world-local positions on the chassis (indices 0..3).
     * Convention: 0=front-left, 1=front-right, 2=rear-left, 3=rear-right.
     * Front wheels (0, 1) steer; rear wheels (2, 3) drive. */
    bj_vec3 wheel_positions[4];

    /* Shared wheel parameters. */
    float   wheel_radius;
    float   wheel_width;
    float   suspension_min_length;
    float   suspension_max_length;
    float   max_steer_angle;        /* radians; applied to front wheels */
    float   max_brake_torque;       /* Nm; applied on brake input */
    float   max_handbrake_torque;   /* Nm; applied on handbrake input */

    /* Engine / chassis — one-dimensional scalar controls. */
    float   engine_max_torque;      /* Nm */
    float   max_pitch_roll_angle;   /* radians; Jolt default ~60° */

    /* Chassis object layer (typically BJ_LAYER_MOVING). */
    uint32_t object_layer;
} bj_vehicle_desc;

/* Creates the chassis body AND the vehicle constraint together — returns
 * the vehicle handle. Call bj_vehicle_get_chassis to get the underlying
 * bj_body if you need to apply forces, set transforms, etc. */
bj_vehicle bj_vehicle_create(bj_world world, bj_shape chassis_shape,
                             const bj_vehicle_desc *desc,
                             bj_vec3 position, bj_quat rotation);
void       bj_vehicle_destroy(bj_world world, bj_vehicle vehicle);

/* Get the chassis body (f64 handle registered on the Rust side). The
 * C shim returns the raw BodyID-encoded bj_body; Rust maps it. */
bj_body    bj_vehicle_get_chassis(bj_world world, bj_vehicle vehicle);

/* Driver input. Forward/right are normalised [-1..1]; brake/handbrake [0..1].
 * Must be called EVERY FRAME before bj_world_step for responsive control. */
void       bj_vehicle_set_input(bj_world world, bj_vehicle vehicle,
                                float forward, float right,
                                float brake, float handbrake);

/* World-space wheel transform (for rendering spinning wheels). axis is
 * 0..2=position.xyz, 3..6=rotation.xyzw. */
float      bj_vehicle_get_wheel_transform(bj_world world, bj_vehicle vehicle,
                                           uint32_t wheel_index, uint32_t axis);

/* Current engine RPM + wheel slip ratios (useful for audio / VFX). */
float      bj_vehicle_get_engine_rpm(bj_world world, bj_vehicle vehicle);
float      bj_vehicle_get_wheel_angular_velocity(bj_world world, bj_vehicle vehicle, uint32_t wheel_index);

#ifdef __cplusplus
}
#endif

#endif /* BLOOM_JOLT_H */
