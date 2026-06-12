// ============================================================================
// Physics v2 — Jolt-backed API
// ============================================================================
//
// Coordinate system: right-handed, Y-up (same as glTF and the renderer).
// Units are SI: meters, seconds, kilograms — default gravity is
// { x: 0, y: -9.81, z: 0 }. Keep dynamic bodies roughly in the 0.1–10 m
// range and within ~1 km of the origin (f32 precision falls off beyond).
// Rotations are quaternions; angular quantities are radians/sec.
//
// Shape/body separation: shapes are reusable geometry; bodies hold motion
// state + reference a shape. Matches Jolt's model.
//
// Handles are 1-based f64 indices (see HandleRegistry on the Rust side).
// 0 = invalid.
//
// Complex shape types (convex hull, triangle mesh, heightfield) use the
// scratch-buffer FFI: push the raw floats/indices via bloom_physics_scratch_*
// then call the shape factory. Compound shapes use a begin/add/end builder.

// ============================================================
// FFI declarations
// ============================================================

// --- World ---
declare function bloom_physics_create_world(gx: number, gy: number, gz: number, maxBodies: number, numThreads: number): number;
declare function bloom_physics_destroy_world(world: number): void;
declare function bloom_physics_set_gravity(world: number, gx: number, gy: number, gz: number): void;
declare function bloom_physics_get_gravity(world: number, axis: number): number;
declare function bloom_physics_optimize_broadphase(world: number): void;
declare function bloom_physics_step(world: number, dt: number, collisionSteps: number): void;
declare function bloom_physics_step_fixed(world: number, dt: number, collisionSteps: number): number;
declare function bloom_physics_set_fixed_timestep(world: number, hz: number, maxSteps: number): void;
declare function bloom_physics_set_interpolation(world: number, on: number): void;
declare function bloom_physics_get_step_alpha(world: number): number;
declare function bloom_physics_set_layer_collides(world: number, a: number, b: number, collides: number): void;
declare function bloom_physics_get_layer_collides(world: number, a: number, b: number): number;
declare function bloom_physics_body_count(world: number): number;
declare function bloom_physics_active_body_count(world: number): number;

// --- Shapes ---
declare function bloom_physics_shape_box(hx: number, hy: number, hz: number, convexRadius: number): number;
declare function bloom_physics_shape_sphere(r: number): number;
declare function bloom_physics_shape_capsule(h: number, r: number): number;
declare function bloom_physics_shape_cylinder(h: number, r: number, cr: number): number;
declare function bloom_physics_shape_scaled(base: number, sx: number, sy: number, sz: number): number;
declare function bloom_physics_shape_offset_com(base: number, ox: number, oy: number, oz: number): number;
declare function bloom_physics_shape_release(shape: number): void;
declare function bloom_physics_shape_bounds(shape: number, axis: number): number;
declare function bloom_physics_shape_volume(shape: number): number;

// --- Bodies ---
declare function bloom_physics_body_create(
  world: number, shape: number, motionType: number,
  px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number,
  layer: number,
): number;
declare function bloom_physics_body_destroy(body: number): void;
declare function bloom_physics_body_activate(body: number): void;
declare function bloom_physics_body_deactivate(body: number): void;
declare function bloom_physics_body_is_active(body: number): number;
declare function bloom_physics_body_is_valid(body: number): number;

declare function bloom_physics_body_get_position(body: number, axis: number): number;
declare function bloom_physics_body_get_rotation(body: number, axis: number): number;
declare function bloom_physics_body_set_position(body: number, x: number, y: number, z: number, activate: number): void;
declare function bloom_physics_body_set_rotation(body: number, x: number, y: number, z: number, w: number, activate: number): void;
declare function bloom_physics_body_set_transform(
  body: number, px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number, activate: number,
): void;
declare function bloom_physics_body_move_kinematic(
  body: number, px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number, dt: number,
): void;

declare function bloom_physics_body_get_linear_velocity(body: number, axis: number): number;
declare function bloom_physics_body_get_angular_velocity(body: number, axis: number): number;
declare function bloom_physics_body_get_point_velocity(body: number, px: number, py: number, pz: number, axis: number): number;
declare function bloom_physics_body_set_linear_velocity(body: number, x: number, y: number, z: number): void;
declare function bloom_physics_body_set_angular_velocity(body: number, x: number, y: number, z: number): void;

declare function bloom_physics_body_add_force(body: number, x: number, y: number, z: number): void;
declare function bloom_physics_body_add_impulse(body: number, x: number, y: number, z: number): void;
declare function bloom_physics_body_add_torque(body: number, x: number, y: number, z: number): void;
declare function bloom_physics_body_add_angular_impulse(body: number, x: number, y: number, z: number): void;
declare function bloom_physics_body_add_force_at(body: number, fx: number, fy: number, fz: number, px: number, py: number, pz: number): void;
declare function bloom_physics_body_add_impulse_at(body: number, ix: number, iy: number, iz: number, px: number, py: number, pz: number): void;

declare function bloom_physics_body_set_friction(body: number, v: number): void;
declare function bloom_physics_body_set_restitution(body: number, v: number): void;
declare function bloom_physics_body_set_linear_damping(body: number, v: number): void;
declare function bloom_physics_body_set_angular_damping(body: number, v: number): void;
declare function bloom_physics_body_set_gravity_factor(body: number, v: number): void;
declare function bloom_physics_body_set_ccd(body: number, enabled: number): void;
declare function bloom_physics_body_set_motion_type(body: number, t: number, activate: number): void;
declare function bloom_physics_body_set_object_layer(body: number, layer: number): void;
declare function bloom_physics_body_set_is_sensor(body: number, enabled: number): void;
declare function bloom_physics_body_set_allow_sleeping(body: number, enabled: number): void;
declare function bloom_physics_body_set_shape(body: number, shape: number, updateMass: number, activate: number): void;
declare function bloom_physics_body_lock_rotation_axes(body: number, x: number, y: number, z: number): void;
declare function bloom_physics_body_lock_translation_axes(body: number, x: number, y: number, z: number): void;

declare function bloom_physics_body_get_mass(body: number): number;
declare function bloom_physics_body_get_friction(body: number): number;
declare function bloom_physics_body_get_restitution(body: number): number;
declare function bloom_physics_body_get_object_layer(body: number): number;
declare function bloom_physics_body_set_user_data(body: number, lo: number, hi: number): void;
declare function bloom_physics_body_get_user_data(body: number, part: number): number;

// --- Queries ---
declare function bloom_physics_raycast(
  world: number, ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number, maxDist: number, layerMask: number,
): number;
declare function bloom_physics_raycast_all(
  world: number, ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number, maxDist: number, layerMask: number, maxHits: number,
): number;
declare function bloom_physics_ray_hit_count(): number;
declare function bloom_physics_ray_hit_body(i: number): number;
declare function bloom_physics_ray_hit_axis(i: number, field: number): number;
declare function bloom_physics_ray_hit_fraction(i: number): number;
declare function bloom_physics_ray_hit_sub_shape(i: number): number;

declare function bloom_physics_overlap_sphere(world: number, cx: number, cy: number, cz: number, r: number, layerMask: number, maxResults: number): number;
declare function bloom_physics_overlap_point(world: number, px: number, py: number, pz: number, layerMask: number, maxResults: number): number;
declare function bloom_physics_overlap_box(
  world: number, px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number,
  hx: number, hy: number, hz: number,
  layerMask: number, maxResults: number,
): number;
declare function bloom_physics_overlap_body(i: number): number;

// --- Constraints ---
declare function bloom_physics_constraint_fixed(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  worldSpace: number,
): number;
declare function bloom_physics_constraint_point(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  worldSpace: number,
): number;
declare function bloom_physics_constraint_hinge(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  axx: number, axy: number, axz: number, lmin: number, lmax: number,
  worldSpace: number,
): number;
declare function bloom_physics_constraint_slider(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  axx: number, axy: number, axz: number, lmin: number, lmax: number,
  worldSpace: number,
): number;
declare function bloom_physics_constraint_distance(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  minD: number, maxD: number, worldSpace: number,
): number;
declare function bloom_physics_constraint_destroy(c: number): void;
declare function bloom_physics_constraint_set_enabled(c: number, enabled: number): void;

// --- Contact events ---
declare function bloom_physics_contact_count(): number;
declare function bloom_physics_contact_field(i: number, field: number): number;
declare function bloom_physics_clear_contacts(world: number): void;

// ============================================================
// Typed handles (nominal types on `number`)
// ============================================================

export type WorldHandle      = number;
export type ShapeHandle      = number;
export type BodyHandle       = number;
export type ConstraintHandle = number;

export const INVALID_HANDLE = 0;

// ============================================================
// Enums
// ============================================================

export const MotionType = {
  STATIC:    0,
  KINEMATIC: 1,
  DYNAMIC:   2,
} as const;

export const ContactEvent = {
  ADDED:     0,
  PERSISTED: 1,
  REMOVED:   2,
} as const;

// Default object layers — applications may define 3..15 for custom purposes.
export const Layer = {
  NON_MOVING: 0,
  MOVING:     1,
  SENSOR:     2,
} as const;

export const MAX_OBJECT_LAYERS = 16;
export const ALL_LAYERS_MASK   = 0xFFFF;

// ============================================================
// POD types (TS-side wrapper objects)
// ============================================================

export interface Vec3 { x: number; y: number; z: number; }
export interface Quat { x: number; y: number; z: number; w: number; }
export interface Transform { position: Vec3; rotation: Quat; }

export interface RayHit {
  body: BodyHandle;
  point: Vec3;
  normal: Vec3;
  fraction: number;
  subShapeId: number;
}

export interface Contact {
  event: number;        // ContactEvent
  bodyA: BodyHandle;
  bodyB: BodyHandle;
  pointA: Vec3;
  pointB: Vec3;
  normal: Vec3;
  penetrationDepth: number;
  combinedFriction: number;
  combinedRestitution: number;
}

// ============================================================
// World
// ============================================================

export interface WorldConfig {
  gravity?: Vec3;
  maxBodies?: number;
  numThreads?: number;
}

export function createWorld(config: WorldConfig = {}): WorldHandle {
  const g = config.gravity ?? { x: 0, y: -9.81, z: 0 };
  return bloom_physics_create_world(
    g.x, g.y, g.z,
    config.maxBodies ?? 0,
    config.numThreads ?? 0,
  );
}

export function destroyWorld(world: WorldHandle): void { bloom_physics_destroy_world(world); }

export function setGravity(world: WorldHandle, g: Vec3): void {
  bloom_physics_set_gravity(world, g.x, g.y, g.z);
}

export function getGravity(world: WorldHandle): Vec3 {
  return {
    x: bloom_physics_get_gravity(world, 0),
    y: bloom_physics_get_gravity(world, 1),
    z: bloom_physics_get_gravity(world, 2),
  };
}

export function optimizeBroadphase(world: WorldHandle): void { bloom_physics_optimize_broadphase(world); }

/// Advance the simulation by `deltaTime` seconds of wall-clock time.
///
/// Steps the world at a fixed rate (default 60 Hz — configure with
/// `setFixedTimestep`) using an accumulator: whole fixed steps are
/// simulated, the remainder carries to the next frame, and a single long
/// frame is clamped (0.25 s) and capped (default 4 steps) so hitches slow
/// the simulation down instead of destabilizing the solver.
///
/// Returns the interpolation alpha in [0, 1): how far the carried
/// remainder sits between the last two physics states. With
/// `setInterpolation(world, true)`, body position/rotation getters apply
/// it automatically for smooth rendering at any display rate.
///
/// Call once per frame with the frame's delta time:
/// ```ts
/// runGame((dt) => {
///   physics.step(world, dt);
///   // draw using body positions...
/// });
/// ```
///
/// Need the old behavior of simulating exactly `deltaTime` in one solver
/// call? Use `stepVariable` — and know that it trades stability for it.
export function step(world: WorldHandle, deltaTime: number, collisionSteps: number = 1): number {
  return bloom_physics_step_fixed(world, deltaTime, collisionSteps);
}

/// Advance the simulation by exactly `deltaTime` in a single solver call
/// (the pre-fixed-timestep behavior). Variable step sizes feed frame
/// hitches straight into the solver — prefer `step` unless you are
/// driving the accumulator yourself.
export function stepVariable(world: WorldHandle, deltaTime: number, collisionSteps: number = 1): void {
  bloom_physics_step(world, deltaTime, collisionSteps);
}

/// Set the fixed simulation rate (`hz`, default 60) and the per-frame
/// catch-up cap (`maxSteps`, default 4) used by `step`.
export function setFixedTimestep(world: WorldHandle, hz: number, maxSteps: number = 4): void {
  bloom_physics_set_fixed_timestep(world, hz, maxSteps);
}

/// Smooth rendering between fixed physics steps: when enabled, body
/// position/rotation getters return states blended by the current step
/// alpha instead of the raw simulation state. Physics queries (raycasts,
/// overlaps) always see the real simulation state. Off by default.
export function setInterpolation(world: WorldHandle, on: boolean): void {
  bloom_physics_set_interpolation(world, on ? 1 : 0);
}

/// Interpolation alpha from the most recent `step` (1.0 if the world has
/// never been fixed-stepped). Useful for interpolating game-side state
/// the same way the engine interpolates body transforms.
export function getStepAlpha(world: WorldHandle): number {
  return bloom_physics_get_step_alpha(world);
}

export function setLayerCollides(world: WorldHandle, a: number, b: number, collides: boolean): void {
  bloom_physics_set_layer_collides(world, a, b, collides ? 1 : 0);
}
export function getLayerCollides(world: WorldHandle, a: number, b: number): boolean {
  return bloom_physics_get_layer_collides(world, a, b) !== 0;
}

export function bodyCount(world: WorldHandle): number { return bloom_physics_body_count(world); }
export function activeBodyCount(world: WorldHandle): number { return bloom_physics_active_body_count(world); }

// ============================================================
// Shapes
// ============================================================

export function boxShape(halfExtents: Vec3, convexRadius: number = 0.05): ShapeHandle {
  return bloom_physics_shape_box(halfExtents.x, halfExtents.y, halfExtents.z, convexRadius);
}
export function sphereShape(radius: number): ShapeHandle {
  return bloom_physics_shape_sphere(radius);
}
export function capsuleShape(halfHeight: number, radius: number): ShapeHandle {
  return bloom_physics_shape_capsule(halfHeight, radius);
}
export function cylinderShape(halfHeight: number, radius: number, convexRadius: number = 0.05): ShapeHandle {
  return bloom_physics_shape_cylinder(halfHeight, radius, convexRadius);
}
export function scaledShape(base: ShapeHandle, scale: Vec3): ShapeHandle {
  return bloom_physics_shape_scaled(base, scale.x, scale.y, scale.z);
}
export function offsetCenterOfMassShape(base: ShapeHandle, offset: Vec3): ShapeHandle {
  return bloom_physics_shape_offset_com(base, offset.x, offset.y, offset.z);
}
export function releaseShape(shape: ShapeHandle): void { bloom_physics_shape_release(shape); }

export function shapeBounds(shape: ShapeHandle): { min: Vec3; max: Vec3 } {
  return {
    min: { x: bloom_physics_shape_bounds(shape, 0), y: bloom_physics_shape_bounds(shape, 1), z: bloom_physics_shape_bounds(shape, 2) },
    max: { x: bloom_physics_shape_bounds(shape, 3), y: bloom_physics_shape_bounds(shape, 4), z: bloom_physics_shape_bounds(shape, 5) },
  };
}
export function shapeVolume(shape: ShapeHandle): number { return bloom_physics_shape_volume(shape); }

// ============================================================
// Bodies
// ============================================================

export interface BodyConfig {
  motionType: number;          // MotionType.*
  position?: Vec3;
  rotation?: Quat;
  objectLayer?: number;        // default Layer.MOVING
  linearVelocity?: Vec3;
  angularVelocity?: Vec3;
  friction?: number;           // applied after creation
  restitution?: number;
  linearDamping?: number;
  angularDamping?: number;
  gravityFactor?: number;
  useCcd?: boolean;
  isSensor?: boolean;
  allowSleeping?: boolean;
  userData?: number;           // low 32 bits
}

export function createBody(world: WorldHandle, shape: ShapeHandle, config: BodyConfig): BodyHandle {
  const p = config.position ?? { x: 0, y: 0, z: 0 };
  const r = config.rotation ?? { x: 0, y: 0, z: 0, w: 1 };
  const layer = config.objectLayer ?? Layer.MOVING;
  const body = bloom_physics_body_create(
    world, shape, config.motionType,
    p.x, p.y, p.z, r.x, r.y, r.z, r.w,
    layer,
  );
  if (body === INVALID_HANDLE) { return INVALID_HANDLE; }

  // Apply any additional properties via setters. Cheap — no physics step happens between.
  if (config.linearVelocity) {
    bloom_physics_body_set_linear_velocity(body, config.linearVelocity.x, config.linearVelocity.y, config.linearVelocity.z);
  }
  if (config.angularVelocity) {
    bloom_physics_body_set_angular_velocity(body, config.angularVelocity.x, config.angularVelocity.y, config.angularVelocity.z);
  }
  if (config.friction !== undefined)      { bloom_physics_body_set_friction(body, config.friction); }
  if (config.restitution !== undefined)   { bloom_physics_body_set_restitution(body, config.restitution); }
  if (config.linearDamping !== undefined) { bloom_physics_body_set_linear_damping(body, config.linearDamping); }
  if (config.angularDamping !== undefined){ bloom_physics_body_set_angular_damping(body, config.angularDamping); }
  if (config.gravityFactor !== undefined) { bloom_physics_body_set_gravity_factor(body, config.gravityFactor); }
  if (config.useCcd)        { bloom_physics_body_set_ccd(body, 1); }
  if (config.isSensor)      { bloom_physics_body_set_is_sensor(body, 1); }
  if (config.allowSleeping === false) { bloom_physics_body_set_allow_sleeping(body, 0); }
  if (config.userData !== undefined) {
    bloom_physics_body_set_user_data(body, config.userData >>> 0, 0);
  }
  return body;
}

export function destroyBody(body: BodyHandle): void { bloom_physics_body_destroy(body); }
export function activateBody(body: BodyHandle): void { bloom_physics_body_activate(body); }
export function deactivateBody(body: BodyHandle): void { bloom_physics_body_deactivate(body); }
export function isBodyActive(body: BodyHandle): boolean { return bloom_physics_body_is_active(body) !== 0; }
export function isBodyValid(body: BodyHandle): boolean { return bloom_physics_body_is_valid(body) !== 0; }

export function getBodyPosition(body: BodyHandle): Vec3 {
  return {
    x: bloom_physics_body_get_position(body, 0),
    y: bloom_physics_body_get_position(body, 1),
    z: bloom_physics_body_get_position(body, 2),
  };
}
export function getBodyRotation(body: BodyHandle): Quat {
  return {
    x: bloom_physics_body_get_rotation(body, 0),
    y: bloom_physics_body_get_rotation(body, 1),
    z: bloom_physics_body_get_rotation(body, 2),
    w: bloom_physics_body_get_rotation(body, 3),
  };
}
export function getBodyTransform(body: BodyHandle): Transform {
  return { position: getBodyPosition(body), rotation: getBodyRotation(body) };
}
export function setBodyPosition(body: BodyHandle, p: Vec3, activate: boolean = true): void {
  bloom_physics_body_set_position(body, p.x, p.y, p.z, activate ? 1 : 0);
}
export function setBodyRotation(body: BodyHandle, r: Quat, activate: boolean = true): void {
  bloom_physics_body_set_rotation(body, r.x, r.y, r.z, r.w, activate ? 1 : 0);
}
export function setBodyTransform(body: BodyHandle, x: Transform, activate: boolean = true): void {
  bloom_physics_body_set_transform(body, x.position.x, x.position.y, x.position.z,
    x.rotation.x, x.rotation.y, x.rotation.z, x.rotation.w, activate ? 1 : 0);
}
export function moveKinematic(body: BodyHandle, target: Transform, deltaTime: number): void {
  bloom_physics_body_move_kinematic(body, target.position.x, target.position.y, target.position.z,
    target.rotation.x, target.rotation.y, target.rotation.z, target.rotation.w, deltaTime);
}

export function getLinearVelocity(body: BodyHandle): Vec3 {
  return {
    x: bloom_physics_body_get_linear_velocity(body, 0),
    y: bloom_physics_body_get_linear_velocity(body, 1),
    z: bloom_physics_body_get_linear_velocity(body, 2),
  };
}
export function getAngularVelocity(body: BodyHandle): Vec3 {
  return {
    x: bloom_physics_body_get_angular_velocity(body, 0),
    y: bloom_physics_body_get_angular_velocity(body, 1),
    z: bloom_physics_body_get_angular_velocity(body, 2),
  };
}
export function getPointVelocity(body: BodyHandle, worldPoint: Vec3): Vec3 {
  return {
    x: bloom_physics_body_get_point_velocity(body, worldPoint.x, worldPoint.y, worldPoint.z, 0),
    y: bloom_physics_body_get_point_velocity(body, worldPoint.x, worldPoint.y, worldPoint.z, 1),
    z: bloom_physics_body_get_point_velocity(body, worldPoint.x, worldPoint.y, worldPoint.z, 2),
  };
}
export function setLinearVelocity(body: BodyHandle, v: Vec3): void {
  bloom_physics_body_set_linear_velocity(body, v.x, v.y, v.z);
}
export function setAngularVelocity(body: BodyHandle, v: Vec3): void {
  bloom_physics_body_set_angular_velocity(body, v.x, v.y, v.z);
}

export function addForce(body: BodyHandle, f: Vec3): void { bloom_physics_body_add_force(body, f.x, f.y, f.z); }
export function addImpulse(body: BodyHandle, i: Vec3): void { bloom_physics_body_add_impulse(body, i.x, i.y, i.z); }
export function addTorque(body: BodyHandle, t: Vec3): void { bloom_physics_body_add_torque(body, t.x, t.y, t.z); }
export function addAngularImpulse(body: BodyHandle, i: Vec3): void { bloom_physics_body_add_angular_impulse(body, i.x, i.y, i.z); }
export function addForceAt(body: BodyHandle, f: Vec3, worldPoint: Vec3): void {
  bloom_physics_body_add_force_at(body, f.x, f.y, f.z, worldPoint.x, worldPoint.y, worldPoint.z);
}
export function addImpulseAt(body: BodyHandle, i: Vec3, worldPoint: Vec3): void {
  bloom_physics_body_add_impulse_at(body, i.x, i.y, i.z, worldPoint.x, worldPoint.y, worldPoint.z);
}

export function setFriction(body: BodyHandle, v: number): void { bloom_physics_body_set_friction(body, v); }
export function setRestitution(body: BodyHandle, v: number): void { bloom_physics_body_set_restitution(body, v); }
export function setLinearDamping(body: BodyHandle, v: number): void { bloom_physics_body_set_linear_damping(body, v); }
export function setAngularDamping(body: BodyHandle, v: number): void { bloom_physics_body_set_angular_damping(body, v); }
export function setGravityFactor(body: BodyHandle, v: number): void { bloom_physics_body_set_gravity_factor(body, v); }
export function setBodyCcd(body: BodyHandle, enabled: boolean): void { bloom_physics_body_set_ccd(body, enabled ? 1 : 0); }
export function setMotionType(body: BodyHandle, motionType: number, activate: boolean = true): void {
  bloom_physics_body_set_motion_type(body, motionType, activate ? 1 : 0);
}
export function setObjectLayer(body: BodyHandle, layer: number): void { bloom_physics_body_set_object_layer(body, layer); }
export function setIsSensor(body: BodyHandle, enabled: boolean): void { bloom_physics_body_set_is_sensor(body, enabled ? 1 : 0); }
export function setAllowSleeping(body: BodyHandle, enabled: boolean): void { bloom_physics_body_set_allow_sleeping(body, enabled ? 1 : 0); }
export function setBodyShape(body: BodyHandle, shape: ShapeHandle, updateMass: boolean = true, activate: boolean = true): void {
  bloom_physics_body_set_shape(body, shape, updateMass ? 1 : 0, activate ? 1 : 0);
}
export function lockRotationAxes(body: BodyHandle, x: boolean, y: boolean, z: boolean): void {
  bloom_physics_body_lock_rotation_axes(body, x ? 1 : 0, y ? 1 : 0, z ? 1 : 0);
}
export function lockTranslationAxes(body: BodyHandle, x: boolean, y: boolean, z: boolean): void {
  bloom_physics_body_lock_translation_axes(body, x ? 1 : 0, y ? 1 : 0, z ? 1 : 0);
}

export function getBodyMass(body: BodyHandle): number { return bloom_physics_body_get_mass(body); }
export function getBodyFriction(body: BodyHandle): number { return bloom_physics_body_get_friction(body); }
export function getBodyRestitution(body: BodyHandle): number { return bloom_physics_body_get_restitution(body); }
export function getBodyObjectLayer(body: BodyHandle): number { return bloom_physics_body_get_object_layer(body); }
export function setBodyUserData(body: BodyHandle, userData: number): void {
  bloom_physics_body_set_user_data(body, userData >>> 0, 0);
}
export function getBodyUserData(body: BodyHandle): number { return bloom_physics_body_get_user_data(body, 0); }

// ============================================================
// Queries
// ============================================================

export function raycast(
  world: WorldHandle, origin: Vec3, direction: Vec3, maxDistance: number, layerMask: number = ALL_LAYERS_MASK,
): RayHit | null {
  const hit = bloom_physics_raycast(world, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, maxDistance, layerMask);
  if (hit === 0) { return null; }
  return readRayHit(0);
}

export function raycastAll(
  world: WorldHandle, origin: Vec3, direction: Vec3, maxDistance: number,
  maxHits: number = 16, layerMask: number = ALL_LAYERS_MASK,
): RayHit[] {
  const n = bloom_physics_raycast_all(world, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, maxDistance, layerMask, maxHits);
  const hits: RayHit[] = new Array(n);
  for (let i = 0; i < n; i = i + 1) { hits[i] = readRayHit(i); }
  return hits;
}

function readRayHit(i: number): RayHit {
  return {
    body: bloom_physics_ray_hit_body(i),
    point: {
      x: bloom_physics_ray_hit_axis(i, 0),
      y: bloom_physics_ray_hit_axis(i, 1),
      z: bloom_physics_ray_hit_axis(i, 2),
    },
    normal: {
      x: bloom_physics_ray_hit_axis(i, 3),
      y: bloom_physics_ray_hit_axis(i, 4),
      z: bloom_physics_ray_hit_axis(i, 5),
    },
    fraction: bloom_physics_ray_hit_fraction(i),
    subShapeId: bloom_physics_ray_hit_sub_shape(i),
  };
}

export function overlapSphere(
  world: WorldHandle, center: Vec3, radius: number,
  maxResults: number = 16, layerMask: number = ALL_LAYERS_MASK,
): BodyHandle[] {
  const n = bloom_physics_overlap_sphere(world, center.x, center.y, center.z, radius, layerMask, maxResults);
  return readOverlapResults(n);
}

export function overlapBox(
  world: WorldHandle, xform: Transform, halfExtents: Vec3,
  maxResults: number = 16, layerMask: number = ALL_LAYERS_MASK,
): BodyHandle[] {
  const n = bloom_physics_overlap_box(world,
    xform.position.x, xform.position.y, xform.position.z,
    xform.rotation.x, xform.rotation.y, xform.rotation.z, xform.rotation.w,
    halfExtents.x, halfExtents.y, halfExtents.z, layerMask, maxResults);
  return readOverlapResults(n);
}

export function overlapPoint(
  world: WorldHandle, point: Vec3,
  maxResults: number = 16, layerMask: number = ALL_LAYERS_MASK,
): BodyHandle[] {
  const n = bloom_physics_overlap_point(world, point.x, point.y, point.z, layerMask, maxResults);
  return readOverlapResults(n);
}

function readOverlapResults(count: number): BodyHandle[] {
  const bodies: BodyHandle[] = new Array(count);
  for (let i = 0; i < count; i = i + 1) { bodies[i] = bloom_physics_overlap_body(i); }
  return bodies;
}

// ============================================================
// Constraints
// ============================================================

export interface ConstraintAnchors {
  bodyA: BodyHandle;
  bodyB?: BodyHandle;
  anchorA: Vec3;
  anchorB: Vec3;
  worldSpace?: boolean;
}

export function fixedConstraint(a: ConstraintAnchors): ConstraintHandle {
  return bloom_physics_constraint_fixed(
    a.bodyA, a.bodyB ?? INVALID_HANDLE,
    a.anchorA.x, a.anchorA.y, a.anchorA.z,
    a.anchorB.x, a.anchorB.y, a.anchorB.z,
    a.worldSpace ? 1 : 0,
  );
}

export function pointConstraint(a: ConstraintAnchors): ConstraintHandle {
  return bloom_physics_constraint_point(
    a.bodyA, a.bodyB ?? INVALID_HANDLE,
    a.anchorA.x, a.anchorA.y, a.anchorA.z,
    a.anchorB.x, a.anchorB.y, a.anchorB.z,
    a.worldSpace ? 1 : 0,
  );
}

export function hingeConstraint(a: ConstraintAnchors, axis: Vec3, limitMin: number = 0, limitMax: number = 0): ConstraintHandle {
  return bloom_physics_constraint_hinge(
    a.bodyA, a.bodyB ?? INVALID_HANDLE,
    a.anchorA.x, a.anchorA.y, a.anchorA.z,
    a.anchorB.x, a.anchorB.y, a.anchorB.z,
    axis.x, axis.y, axis.z, limitMin, limitMax,
    a.worldSpace ? 1 : 0,
  );
}

export function sliderConstraint(a: ConstraintAnchors, axis: Vec3, limitMin: number = 0, limitMax: number = 0): ConstraintHandle {
  return bloom_physics_constraint_slider(
    a.bodyA, a.bodyB ?? INVALID_HANDLE,
    a.anchorA.x, a.anchorA.y, a.anchorA.z,
    a.anchorB.x, a.anchorB.y, a.anchorB.z,
    axis.x, axis.y, axis.z, limitMin, limitMax,
    a.worldSpace ? 1 : 0,
  );
}

export function distanceConstraint(a: ConstraintAnchors, minDistance: number, maxDistance: number): ConstraintHandle {
  return bloom_physics_constraint_distance(
    a.bodyA, a.bodyB ?? INVALID_HANDLE,
    a.anchorA.x, a.anchorA.y, a.anchorA.z,
    a.anchorB.x, a.anchorB.y, a.anchorB.z,
    minDistance, maxDistance, a.worldSpace ? 1 : 0,
  );
}

export function destroyConstraint(c: ConstraintHandle): void { bloom_physics_constraint_destroy(c); }
export function setConstraintEnabled(c: ConstraintHandle, enabled: boolean): void {
  bloom_physics_constraint_set_enabled(c, enabled ? 1 : 0);
}

// ============================================================
// Contact events (drained once per step)
// ============================================================

export function contactCount(): number { return bloom_physics_contact_count(); }

export function popContacts(): Contact[] {
  const n = bloom_physics_contact_count();
  const out: Contact[] = new Array(n);
  for (let i = 0; i < n; i = i + 1) {
    out[i] = {
      event: bloom_physics_contact_field(i, 0),
      bodyA: bloom_physics_contact_field(i, 1),
      bodyB: bloom_physics_contact_field(i, 2),
      pointA: {
        x: bloom_physics_contact_field(i, 3),
        y: bloom_physics_contact_field(i, 4),
        z: bloom_physics_contact_field(i, 5),
      },
      pointB: {
        x: bloom_physics_contact_field(i, 6),
        y: bloom_physics_contact_field(i, 7),
        z: bloom_physics_contact_field(i, 8),
      },
      normal: {
        x: bloom_physics_contact_field(i, 9),
        y: bloom_physics_contact_field(i, 10),
        z: bloom_physics_contact_field(i, 11),
      },
      penetrationDepth:    bloom_physics_contact_field(i, 12),
      combinedFriction:    bloom_physics_contact_field(i, 13),
      combinedRestitution: bloom_physics_contact_field(i, 14),
    };
  }
  return out;
}

export function clearContacts(world: WorldHandle): void { bloom_physics_clear_contacts(world); }

// ============================================================================
// Complex shapes (scratch-buffer variants)
// ============================================================================

declare function bloom_physics_scratch_reset(): void;
declare function bloom_physics_scratch_push_f32(v: number): void;
declare function bloom_physics_scratch_push_u32(v: number): void;
declare function bloom_physics_shape_convex_hull(numPoints: number, convexRadius: number): number;
declare function bloom_physics_shape_mesh(vertexCount: number, triangleCount: number): number;
declare function bloom_physics_shape_heightfield(
  sampleCount: number, ox: number, oy: number, oz: number,
  sx: number, sy: number, sz: number, blockSize: number,
): number;
declare function bloom_physics_compound_begin(): void;
declare function bloom_physics_compound_add_child(
  shape: number, px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number,
): void;
declare function bloom_physics_compound_end(): number;

/** Convex hull from a point cloud. The physics engine computes the hull. */
export function convexHullShape(points: Vec3[], convexRadius: number = 0.05): ShapeHandle {
  if (points.length < 3) { return INVALID_HANDLE; }
  bloom_physics_scratch_reset();
  for (let i = 0; i < points.length; i = i + 1) {
    const p = points[i];
    bloom_physics_scratch_push_f32(p.x);
    bloom_physics_scratch_push_f32(p.y);
    bloom_physics_scratch_push_f32(p.z);
  }
  return bloom_physics_shape_convex_hull(points.length, convexRadius);
}

/** Triangle mesh (static bodies only). Indices flattened (triangleCount * 3). */
export function meshShape(vertices: Vec3[], indices: number[]): ShapeHandle {
  if (vertices.length === 0 || indices.length < 3 || indices.length % 3 !== 0) {
    return INVALID_HANDLE;
  }
  bloom_physics_scratch_reset();
  for (let i = 0; i < vertices.length; i = i + 1) {
    const v = vertices[i];
    bloom_physics_scratch_push_f32(v.x);
    bloom_physics_scratch_push_f32(v.y);
    bloom_physics_scratch_push_f32(v.z);
  }
  for (let i = 0; i < indices.length; i = i + 1) {
    bloom_physics_scratch_push_u32(indices[i]);
  }
  return bloom_physics_shape_mesh(vertices.length, indices.length / 3);
}

/** Heightfield — samples is row-major (sampleCount × sampleCount). */
export function heightfieldShape(
  samples: number[], sampleCount: number,
  offset: Vec3, scale: Vec3, blockSize: number = 4,
): ShapeHandle {
  const need = sampleCount * sampleCount;
  if (samples.length < need || sampleCount < 2) { return INVALID_HANDLE; }
  bloom_physics_scratch_reset();
  for (let i = 0; i < need; i = i + 1) {
    bloom_physics_scratch_push_f32(samples[i]);
  }
  return bloom_physics_shape_heightfield(
    sampleCount, offset.x, offset.y, offset.z, scale.x, scale.y, scale.z, blockSize,
  );
}

/** Static compound built from children (shape + local transform). */
export function compoundShape(children: Array<{ shape: ShapeHandle; local: Transform }>): ShapeHandle {
  if (children.length === 0) { return INVALID_HANDLE; }
  bloom_physics_compound_begin();
  for (let i = 0; i < children.length; i = i + 1) {
    const c = children[i];
    bloom_physics_compound_add_child(
      c.shape,
      c.local.position.x, c.local.position.y, c.local.position.z,
      c.local.rotation.x, c.local.rotation.y, c.local.rotation.z, c.local.rotation.w,
    );
  }
  return bloom_physics_compound_end();
}

// ============================================================================
// Character controller (CharacterVirtual — kinematic, slope-aware)
// ============================================================================

declare function bloom_physics_character_create(
  world: number, shape: number,
  upX: number, upY: number, upZ: number,
  maxSlopeAngle: number, characterPadding: number,
  penetrationRecoverySpeed: number, predictiveContactDistance: number,
  maxStrength: number, mass: number, objectLayer: number,
  px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number,
): number;
declare function bloom_physics_character_destroy(c: number): void;
declare function bloom_physics_character_update(c: number, dt: number, gx: number, gy: number, gz: number): void;
declare function bloom_physics_character_get_position(c: number, axis: number): number;
declare function bloom_physics_character_get_rotation(c: number, axis: number): number;
declare function bloom_physics_character_set_position(c: number, x: number, y: number, z: number): void;
declare function bloom_physics_character_set_rotation(c: number, x: number, y: number, z: number, w: number): void;
declare function bloom_physics_character_get_linear_velocity(c: number, axis: number): number;
declare function bloom_physics_character_set_linear_velocity(c: number, x: number, y: number, z: number): void;
declare function bloom_physics_character_get_ground_state(c: number): number;
declare function bloom_physics_character_get_ground_normal(c: number, axis: number): number;
declare function bloom_physics_character_get_ground_position(c: number, axis: number): number;
declare function bloom_physics_character_get_ground_body(c: number): number;
declare function bloom_physics_character_set_shape(c: number, shape: number): void;

export type CharacterHandle = number;

export const GroundState = {
  ON_GROUND:     0,
  ON_STEEP:      1,
  NOT_SUPPORTED: 2,
  IN_AIR:        3,
} as const;

export interface CharacterConfig {
  up?: Vec3;                          // default (0,1,0)
  maxSlopeAngleRad?: number;          // default ~50° (0.872)
  characterPadding?: number;          // default 0.02
  penetrationRecoverySpeed?: number;  // default 1.0
  predictiveContactDistance?: number; // default 0.1
  maxStrength?: number;               // default 100
  mass?: number;                      // default 70 kg
  objectLayer?: number;               // default Layer.MOVING
  position?: Vec3;
  rotation?: Quat;
}

export function createCharacter(
  world: WorldHandle, shape: ShapeHandle, config: CharacterConfig = {},
): CharacterHandle {
  const up = config.up ?? { x: 0, y: 1, z: 0 };
  const pos = config.position ?? { x: 0, y: 0, z: 0 };
  const rot = config.rotation ?? { x: 0, y: 0, z: 0, w: 1 };
  return bloom_physics_character_create(
    world, shape,
    up.x, up.y, up.z,
    config.maxSlopeAngleRad ?? 0.872,
    config.characterPadding ?? 0.02,
    config.penetrationRecoverySpeed ?? 1.0,
    config.predictiveContactDistance ?? 0.1,
    config.maxStrength ?? 100.0,
    config.mass ?? 70.0,
    config.objectLayer ?? Layer.MOVING,
    pos.x, pos.y, pos.z,
    rot.x, rot.y, rot.z, rot.w,
  );
}

export function destroyCharacter(c: CharacterHandle): void { bloom_physics_character_destroy(c); }

export function updateCharacter(c: CharacterHandle, deltaTime: number, gravity: Vec3): void {
  bloom_physics_character_update(c, deltaTime, gravity.x, gravity.y, gravity.z);
}

export function getCharacterPosition(c: CharacterHandle): Vec3 {
  return {
    x: bloom_physics_character_get_position(c, 0),
    y: bloom_physics_character_get_position(c, 1),
    z: bloom_physics_character_get_position(c, 2),
  };
}
export function getCharacterRotation(c: CharacterHandle): Quat {
  return {
    x: bloom_physics_character_get_rotation(c, 0),
    y: bloom_physics_character_get_rotation(c, 1),
    z: bloom_physics_character_get_rotation(c, 2),
    w: bloom_physics_character_get_rotation(c, 3),
  };
}
export function setCharacterPosition(c: CharacterHandle, p: Vec3): void {
  bloom_physics_character_set_position(c, p.x, p.y, p.z);
}
export function setCharacterRotation(c: CharacterHandle, r: Quat): void {
  bloom_physics_character_set_rotation(c, r.x, r.y, r.z, r.w);
}

export function getCharacterLinearVelocity(c: CharacterHandle): Vec3 {
  return {
    x: bloom_physics_character_get_linear_velocity(c, 0),
    y: bloom_physics_character_get_linear_velocity(c, 1),
    z: bloom_physics_character_get_linear_velocity(c, 2),
  };
}
export function setCharacterLinearVelocity(c: CharacterHandle, v: Vec3): void {
  bloom_physics_character_set_linear_velocity(c, v.x, v.y, v.z);
}

export function getCharacterGroundState(c: CharacterHandle): number {
  return bloom_physics_character_get_ground_state(c);
}
export function isCharacterGrounded(c: CharacterHandle): boolean {
  return bloom_physics_character_get_ground_state(c) === GroundState.ON_GROUND;
}
export function getCharacterGroundNormal(c: CharacterHandle): Vec3 {
  return {
    x: bloom_physics_character_get_ground_normal(c, 0),
    y: bloom_physics_character_get_ground_normal(c, 1),
    z: bloom_physics_character_get_ground_normal(c, 2),
  };
}
export function getCharacterGroundPosition(c: CharacterHandle): Vec3 {
  return {
    x: bloom_physics_character_get_ground_position(c, 0),
    y: bloom_physics_character_get_ground_position(c, 1),
    z: bloom_physics_character_get_ground_position(c, 2),
  };
}
export function getCharacterGroundBody(c: CharacterHandle): BodyHandle {
  return bloom_physics_character_get_ground_body(c);
}

export function setCharacterShape(c: CharacterHandle, shape: ShapeHandle): void {
  bloom_physics_character_set_shape(c, shape);
}

// ============================================================================
// Soft bodies — cloth / rope / jelly with per-vertex simulation
// ============================================================================

declare function bloom_physics_soft_body_create(
  world: number, vertexCount: number, triangleCount: number,
  px: number, py: number, pz: number, rx: number, ry: number, rz: number, rw: number,
  objectLayer: number,
  edgeCompliance: number, gravityFactor: number, linearDamping: number, pressure: number,
): number;
declare function bloom_physics_soft_body_vertex_count(body: number): number;
declare function bloom_physics_soft_body_get_vertex(body: number, idx: number, axis: number): number;
declare function bloom_physics_soft_body_set_vertex(body: number, idx: number, x: number, y: number, z: number): void;
declare function bloom_physics_soft_body_set_vertex_inv_mass(body: number, idx: number, invMass: number): void;

export interface SoftBodyConfig {
  vertices: Vec3[];
  /** Per-vertex inverse mass; length must equal vertices.length. 0 = pinned. */
  inverseMasses: number[];
  /** Triangle indices (length = triangleCount * 3). Each triangle forms edge+bend constraints. */
  indices: number[];
  position?: Vec3;
  rotation?: Quat;
  objectLayer?: number;         // default Layer.MOVING
  edgeCompliance?: number;      // default 0 (rigid edges); cloth: 1e-4
  gravityFactor?: number;       // default 1.0
  linearDamping?: number;       // default 0.05
  pressure?: number;            // 0 = cloth; >0 = inflated volume body
}

export function createSoftBody(world: WorldHandle, config: SoftBodyConfig): BodyHandle {
  const n = config.vertices.length;
  if (n < 3 || config.inverseMasses.length !== n || config.indices.length < 3 || config.indices.length % 3 !== 0) {
    return INVALID_HANDLE;
  }
  bloom_physics_scratch_reset();
  for (let i = 0; i < n; i = i + 1) {
    const v = config.vertices[i];
    bloom_physics_scratch_push_f32(v.x);
    bloom_physics_scratch_push_f32(v.y);
    bloom_physics_scratch_push_f32(v.z);
    bloom_physics_scratch_push_f32(config.inverseMasses[i]);
  }
  for (let i = 0; i < config.indices.length; i = i + 1) {
    bloom_physics_scratch_push_u32(config.indices[i]);
  }
  const pos = config.position ?? { x: 0, y: 0, z: 0 };
  const rot = config.rotation ?? { x: 0, y: 0, z: 0, w: 1 };
  return bloom_physics_soft_body_create(
    world, n, config.indices.length / 3,
    pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w,
    config.objectLayer ?? Layer.MOVING,
    config.edgeCompliance ?? 0,
    config.gravityFactor ?? 1.0,
    config.linearDamping ?? 0.05,
    config.pressure ?? 0,
  );
}

export function softBodyVertexCount(body: BodyHandle): number {
  return bloom_physics_soft_body_vertex_count(body);
}
export function getSoftBodyVertex(body: BodyHandle, idx: number): Vec3 {
  return {
    x: bloom_physics_soft_body_get_vertex(body, idx, 0),
    y: bloom_physics_soft_body_get_vertex(body, idx, 1),
    z: bloom_physics_soft_body_get_vertex(body, idx, 2),
  };
}
export function setSoftBodyVertex(body: BodyHandle, idx: number, position: Vec3): void {
  bloom_physics_soft_body_set_vertex(body, idx, position.x, position.y, position.z);
}
/** Set per-vertex inverse mass. 0 = pinned (won't move under simulation). */
export function setSoftBodyVertexInvMass(body: BodyHandle, idx: number, invMass: number): void {
  bloom_physics_soft_body_set_vertex_inv_mass(body, idx, invMass);
}

// ============================================================================
// Wheeled vehicles — 4-wheel car (FR-drive, front-wheel steering)
// ============================================================================

declare function bloom_physics_vehicle_create(
  world: number, chassisShape: number,
  upX: number, upY: number, upZ: number,
  fwX: number, fwY: number, fwZ: number,
  w0x: number, w0y: number, w0z: number,
  w1x: number, w1y: number, w1z: number,
  w2x: number, w2y: number, w2z: number,
  w3x: number, w3y: number, w3z: number,
  wheelRadius: number, wheelWidth: number,
  suspensionMin: number, suspensionMax: number,
  maxSteerAngle: number, maxBrakeTorque: number, maxHandbrakeTorque: number,
  engineMaxTorque: number, maxPitchRollAngle: number,
  objectLayer: number,
  px: number, py: number, pz: number, rx: number, ry: number, rz: number, rw: number,
): number;
declare function bloom_physics_vehicle_destroy(v: number): void;
declare function bloom_physics_vehicle_get_chassis(v: number): number;
declare function bloom_physics_vehicle_set_input(v: number, forward: number, right: number, brake: number, handbrake: number): void;
declare function bloom_physics_vehicle_get_wheel_transform(v: number, wheelIndex: number, axis: number): number;
declare function bloom_physics_vehicle_get_engine_rpm(v: number): number;
declare function bloom_physics_vehicle_get_wheel_angular_velocity(v: number, wheelIndex: number): number;

export type VehicleHandle = number;

export interface VehicleConfig {
  /** Chassis collision shape. Convention: use `offsetCenterOfMassShape(box, { x: 0, y: -0.6, z: 0 })`
   *  so wheels dangle below the box and the car is stable. */
  chassisShape: ShapeHandle;
  position?: Vec3;
  rotation?: Quat;

  /** World-up axis. Default (0, 1, 0). */
  up?: Vec3;
  /** Chassis forward axis (which way is "front"). Default (0, 0, 1). */
  forward?: Vec3;

  /** Four wheel mount positions in chassis-local space (relative to COM).
   *  Order: [front-left, front-right, rear-left, rear-right].
   *  Default: 1.6m wheelbase/track, wheels ~0.4m below COM. */
  wheelPositions?: [Vec3, Vec3, Vec3, Vec3];

  wheelRadius?: number;           // default 0.35m
  wheelWidth?: number;            // default 0.2m
  suspensionMinLength?: number;   // default 0.3m
  suspensionMaxLength?: number;   // default 0.5m
  maxSteerAngleRad?: number;      // default 35° (≈0.611 rad); applied to front wheels
  maxBrakeTorque?: number;        // default 1500 Nm
  maxHandbrakeTorque?: number;    // default 4000 Nm
  engineMaxTorque?: number;       // default 500 Nm
  maxPitchRollAngleRad?: number;  // default 60° (≈1.047 rad)
  objectLayer?: number;           // default Layer.MOVING
}

/** Returns a vehicle handle. Call getVehicleChassis() for the body handle. */
export function createVehicle(world: WorldHandle, config: VehicleConfig): VehicleHandle {
  const up = config.up ?? { x: 0, y: 1, z: 0 };
  const fw = config.forward ?? { x: 0, y: 0, z: 1 };
  const wp = config.wheelPositions ?? [
    { x: -0.8, y: -0.4, z:  1.3 },
    { x:  0.8, y: -0.4, z:  1.3 },
    { x: -0.8, y: -0.4, z: -1.3 },
    { x:  0.8, y: -0.4, z: -1.3 },
  ];
  const pos = config.position ?? { x: 0, y: 0, z: 0 };
  const rot = config.rotation ?? { x: 0, y: 0, z: 0, w: 1 };
  return bloom_physics_vehicle_create(
    world, config.chassisShape,
    up.x, up.y, up.z, fw.x, fw.y, fw.z,
    wp[0].x, wp[0].y, wp[0].z,
    wp[1].x, wp[1].y, wp[1].z,
    wp[2].x, wp[2].y, wp[2].z,
    wp[3].x, wp[3].y, wp[3].z,
    config.wheelRadius ?? 0.35,
    config.wheelWidth ?? 0.2,
    config.suspensionMinLength ?? 0.3,
    config.suspensionMaxLength ?? 0.5,
    config.maxSteerAngleRad ?? 0.611,
    config.maxBrakeTorque ?? 1500,
    config.maxHandbrakeTorque ?? 4000,
    config.engineMaxTorque ?? 500,
    config.maxPitchRollAngleRad ?? 1.047,
    config.objectLayer ?? Layer.MOVING,
    pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w,
  );
}

export function destroyVehicle(v: VehicleHandle): void { bloom_physics_vehicle_destroy(v); }

/** Returns the underlying chassis body — useful for applying external forces or reading position. */
export function getVehicleChassis(v: VehicleHandle): BodyHandle { return bloom_physics_vehicle_get_chassis(v); }

/**
 * Driver input. Must be called every frame before step() for responsive control.
 * @param forward -1..1 (throttle; negative = reverse)
 * @param right -1..1 (steering)
 * @param brake 0..1
 * @param handbrake 0..1
 */
export function setVehicleInput(
  v: VehicleHandle, forward: number, right: number, brake: number = 0, handbrake: number = 0,
): void {
  bloom_physics_vehicle_set_input(v, forward, right, brake, handbrake);
}

/** World-space transform of a wheel (for rendering spinning wheels). */
export function getWheelTransform(v: VehicleHandle, wheelIndex: number): Transform {
  return {
    position: {
      x: bloom_physics_vehicle_get_wheel_transform(v, wheelIndex, 0),
      y: bloom_physics_vehicle_get_wheel_transform(v, wheelIndex, 1),
      z: bloom_physics_vehicle_get_wheel_transform(v, wheelIndex, 2),
    },
    rotation: {
      x: bloom_physics_vehicle_get_wheel_transform(v, wheelIndex, 3),
      y: bloom_physics_vehicle_get_wheel_transform(v, wheelIndex, 4),
      z: bloom_physics_vehicle_get_wheel_transform(v, wheelIndex, 5),
      w: bloom_physics_vehicle_get_wheel_transform(v, wheelIndex, 6),
    },
  };
}

export function getVehicleEngineRPM(v: VehicleHandle): number { return bloom_physics_vehicle_get_engine_rpm(v); }
export function getWheelAngularVelocity(v: VehicleHandle, wheelIndex: number): number {
  return bloom_physics_vehicle_get_wheel_angular_velocity(v, wheelIndex);
}
