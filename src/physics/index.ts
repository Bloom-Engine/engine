// ============================================================
// Physics (Rapier 3D) — FFI declarations
// ============================================================

declare function bloom_physics_create_world(gx: number, gy: number, gz: number): void;
declare function bloom_physics_set_gravity(gx: number, gy: number, gz: number): void;
declare function bloom_physics_set_timestep(dt: number, maxSubsteps: number): void;

declare function bloom_physics_create_body(
  bodyType: number, px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number,
): number;
declare function bloom_physics_destroy_body(handle: number): void;
declare function bloom_physics_set_body_enabled(handle: number, enabled: number): void;
declare function bloom_physics_set_body_ccd(handle: number, enabled: number): void;
declare function bloom_physics_set_body_gravity_scale(handle: number, scale: number): void;
declare function bloom_physics_set_kinematic_target(
  handle: number, px: number, py: number, pz: number,
  rx: number, ry: number, rz: number, rw: number,
): void;
declare function bloom_physics_lock_rotations(handle: number, lockX: number, lockY: number, lockZ: number): void;

declare function bloom_physics_add_box_collider(body: number, hx: number, hy: number, hz: number): number;
declare function bloom_physics_add_sphere_collider(body: number, radius: number): number;
declare function bloom_physics_add_capsule_collider(body: number, halfHeight: number, radius: number): number;
declare function bloom_physics_add_cylinder_collider(body: number, halfHeight: number, radius: number): number;
declare function bloom_physics_set_collider_properties(
  collider: number, friction: number, restitution: number, density: number,
): void;

declare function bloom_physics_apply_force(body: number, fx: number, fy: number, fz: number): void;
declare function bloom_physics_apply_impulse(body: number, ix: number, iy: number, iz: number): void;
declare function bloom_physics_apply_torque(body: number, tx: number, ty: number, tz: number): void;
declare function bloom_physics_apply_torque_impulse(body: number, tx: number, ty: number, tz: number): void;
declare function bloom_physics_set_linear_velocity(body: number, vx: number, vy: number, vz: number): void;
declare function bloom_physics_set_angular_velocity(body: number, vx: number, vy: number, vz: number): void;

declare function bloom_physics_step(deltaTime: number): void;
declare function bloom_physics_sync_transforms(): void;

declare function bloom_physics_get_body_position_x(body: number): number;
declare function bloom_physics_get_body_position_y(body: number): number;
declare function bloom_physics_get_body_position_z(body: number): number;
declare function bloom_physics_get_body_rotation_x(body: number): number;
declare function bloom_physics_get_body_rotation_y(body: number): number;
declare function bloom_physics_get_body_rotation_z(body: number): number;
declare function bloom_physics_get_body_rotation_w(body: number): number;
declare function bloom_physics_get_linear_velocity_x(body: number): number;
declare function bloom_physics_get_linear_velocity_y(body: number): number;
declare function bloom_physics_get_linear_velocity_z(body: number): number;
declare function bloom_physics_get_angular_velocity_x(body: number): number;
declare function bloom_physics_get_angular_velocity_y(body: number): number;
declare function bloom_physics_get_angular_velocity_z(body: number): number;

declare function bloom_physics_raycast(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  maxDist: number,
): number;
declare function bloom_physics_ray_hit_body(): number;
declare function bloom_physics_ray_hit_distance(): number;
declare function bloom_physics_ray_hit_x(): number;
declare function bloom_physics_ray_hit_y(): number;
declare function bloom_physics_ray_hit_z(): number;

declare function bloom_physics_get_collision_count(): number;
declare function bloom_physics_get_collision_event(index: number): number;
declare function bloom_physics_get_collision_body_b(): number;
declare function bloom_physics_get_collision_started(): number;

declare function bloom_physics_attach_scene_node(body: number, sceneNode: number): void;

declare function bloom_physics_create_fixed_joint(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
): number;
declare function bloom_physics_create_revolute_joint(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number,
  axisX: number, axisY: number, axisZ: number,
): number;
declare function bloom_physics_create_prismatic_joint(
  bodyA: number, bodyB: number,
  ax: number, ay: number, az: number,
  axisX: number, axisY: number, axisZ: number,
): number;
declare function bloom_physics_destroy_joint(handle: number): void;

// ============================================================
// Types
// ============================================================

export type RigidBodyHandle = number;
export type ColliderHandle = number;
export type JointHandle = number;

export const BodyType = {
  DYNAMIC: 0,
  STATIC: 1,
  KINEMATIC_POSITION: 2,
  KINEMATIC_VELOCITY: 3,
} as const;

export interface PhysicsRayHit {
  hit: boolean;
  body: RigidBodyHandle;
  distance: number;
  point: { x: number; y: number; z: number };
}

export interface CollisionInfo {
  bodyA: RigidBodyHandle;
  bodyB: RigidBodyHandle;
  started: boolean;
}

// ============================================================
// World
// ============================================================

export function createPhysicsWorld(
  gravity: { x: number; y: number; z: number } = { x: 0, y: -9.81, z: 0 },
): void {
  bloom_physics_create_world(gravity.x, gravity.y, gravity.z);
}

export function setGravity(gx: number, gy: number, gz: number): void {
  bloom_physics_set_gravity(gx, gy, gz);
}

export function setPhysicsTimestep(fixedDt: number = 1 / 60, maxSubsteps: number = 4): void {
  bloom_physics_set_timestep(fixedDt, maxSubsteps);
}

// ============================================================
// Rigid Bodies
// ============================================================

export function createRigidBody(
  type: number,
  position: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 },
  rotation: { x: number; y: number; z: number; w: number } = { x: 0, y: 0, z: 0, w: 1 },
): RigidBodyHandle {
  return bloom_physics_create_body(
    type, position.x, position.y, position.z,
    rotation.x, rotation.y, rotation.z, rotation.w,
  );
}

export function destroyRigidBody(body: RigidBodyHandle): void {
  bloom_physics_destroy_body(body);
}

export function setBodyEnabled(body: RigidBodyHandle, enabled: boolean): void {
  bloom_physics_set_body_enabled(body, enabled ? 1.0 : 0.0);
}

export function setBodyCcd(body: RigidBodyHandle, enabled: boolean): void {
  bloom_physics_set_body_ccd(body, enabled ? 1.0 : 0.0);
}

export function setBodyGravityScale(body: RigidBodyHandle, scale: number): void {
  bloom_physics_set_body_gravity_scale(body, scale);
}

export function setKinematicTarget(
  body: RigidBodyHandle,
  position: { x: number; y: number; z: number },
  rotation: { x: number; y: number; z: number; w: number } = { x: 0, y: 0, z: 0, w: 1 },
): void {
  bloom_physics_set_kinematic_target(
    body, position.x, position.y, position.z,
    rotation.x, rotation.y, rotation.z, rotation.w,
  );
}

export function lockRotations(
  body: RigidBodyHandle,
  lockX: boolean = true, lockY: boolean = true, lockZ: boolean = true,
): void {
  bloom_physics_lock_rotations(body, lockX ? 1.0 : 0.0, lockY ? 1.0 : 0.0, lockZ ? 1.0 : 0.0);
}

// ============================================================
// Colliders
// ============================================================

export function addBoxCollider(
  body: RigidBodyHandle,
  halfExtents: { x: number; y: number; z: number },
): ColliderHandle {
  return bloom_physics_add_box_collider(body, halfExtents.x, halfExtents.y, halfExtents.z);
}

export function addSphereCollider(body: RigidBodyHandle, radius: number): ColliderHandle {
  return bloom_physics_add_sphere_collider(body, radius);
}

export function addCapsuleCollider(
  body: RigidBodyHandle, halfHeight: number, radius: number,
): ColliderHandle {
  return bloom_physics_add_capsule_collider(body, halfHeight, radius);
}

export function addCylinderCollider(
  body: RigidBodyHandle, halfHeight: number, radius: number,
): ColliderHandle {
  return bloom_physics_add_cylinder_collider(body, halfHeight, radius);
}

export function setColliderProperties(
  collider: ColliderHandle,
  friction: number = 0.5,
  restitution: number = 0.0,
  density: number = 1.0,
): void {
  bloom_physics_set_collider_properties(collider, friction, restitution, density);
}

// ============================================================
// Forces / Velocities
// ============================================================

export function applyForce(body: RigidBodyHandle, force: { x: number; y: number; z: number }): void {
  bloom_physics_apply_force(body, force.x, force.y, force.z);
}

export function applyImpulse(body: RigidBodyHandle, impulse: { x: number; y: number; z: number }): void {
  bloom_physics_apply_impulse(body, impulse.x, impulse.y, impulse.z);
}

export function applyTorque(body: RigidBodyHandle, torque: { x: number; y: number; z: number }): void {
  bloom_physics_apply_torque(body, torque.x, torque.y, torque.z);
}

export function applyTorqueImpulse(body: RigidBodyHandle, torque: { x: number; y: number; z: number }): void {
  bloom_physics_apply_torque_impulse(body, torque.x, torque.y, torque.z);
}

export function setLinearVelocity(body: RigidBodyHandle, vel: { x: number; y: number; z: number }): void {
  bloom_physics_set_linear_velocity(body, vel.x, vel.y, vel.z);
}

export function setAngularVelocity(body: RigidBodyHandle, vel: { x: number; y: number; z: number }): void {
  bloom_physics_set_angular_velocity(body, vel.x, vel.y, vel.z);
}

// ============================================================
// Stepping
// ============================================================

export function stepPhysics(deltaTime: number): void {
  bloom_physics_step(deltaTime);
}

export function syncPhysicsTransforms(): void {
  bloom_physics_sync_transforms();
}

// ============================================================
// Queries
// ============================================================

export function getBodyPosition(body: RigidBodyHandle): { x: number; y: number; z: number } {
  return {
    x: bloom_physics_get_body_position_x(body),
    y: bloom_physics_get_body_position_y(body),
    z: bloom_physics_get_body_position_z(body),
  };
}

export function getBodyRotation(body: RigidBodyHandle): { x: number; y: number; z: number; w: number } {
  return {
    x: bloom_physics_get_body_rotation_x(body),
    y: bloom_physics_get_body_rotation_y(body),
    z: bloom_physics_get_body_rotation_z(body),
    w: bloom_physics_get_body_rotation_w(body),
  };
}

export function getLinearVelocity(body: RigidBodyHandle): { x: number; y: number; z: number } {
  return {
    x: bloom_physics_get_linear_velocity_x(body),
    y: bloom_physics_get_linear_velocity_y(body),
    z: bloom_physics_get_linear_velocity_z(body),
  };
}

export function getAngularVelocity(body: RigidBodyHandle): { x: number; y: number; z: number } {
  return {
    x: bloom_physics_get_angular_velocity_x(body),
    y: bloom_physics_get_angular_velocity_y(body),
    z: bloom_physics_get_angular_velocity_z(body),
  };
}

// ============================================================
// Raycasting
// ============================================================

export function physicsRaycast(
  origin: { x: number; y: number; z: number },
  direction: { x: number; y: number; z: number },
  maxDistance: number,
): PhysicsRayHit {
  const didHit = bloom_physics_raycast(
    origin.x, origin.y, origin.z,
    direction.x, direction.y, direction.z,
    maxDistance,
  );
  if (didHit !== 0.0) {
    return {
      hit: true,
      body: bloom_physics_ray_hit_body(),
      distance: bloom_physics_ray_hit_distance(),
      point: {
        x: bloom_physics_ray_hit_x(),
        y: bloom_physics_ray_hit_y(),
        z: bloom_physics_ray_hit_z(),
      },
    };
  }
  return {
    hit: false,
    body: 0,
    distance: 0,
    point: { x: 0, y: 0, z: 0 },
  };
}

// ============================================================
// Collision Events
// ============================================================

export function getCollisions(): CollisionInfo[] {
  const count = bloom_physics_get_collision_count();
  const events: CollisionInfo[] = [];
  for (let i = 0; i < count; i = i + 1) {
    const bodyA = bloom_physics_get_collision_event(i);
    events.push({
      bodyA,
      bodyB: bloom_physics_get_collision_body_b(),
      started: bloom_physics_get_collision_started() !== 0.0,
    });
  }
  return events;
}

// ============================================================
// Scene Node Attachment
// ============================================================

export function attachPhysicsBody(body: RigidBodyHandle, sceneNode: number): void {
  bloom_physics_attach_scene_node(body, sceneNode);
}

// ============================================================
// Joints
// ============================================================

export function createFixedJoint(
  bodyA: RigidBodyHandle, bodyB: RigidBodyHandle,
  anchorA: { x: number; y: number; z: number },
  anchorB: { x: number; y: number; z: number },
): JointHandle {
  return bloom_physics_create_fixed_joint(
    bodyA, bodyB,
    anchorA.x, anchorA.y, anchorA.z,
    anchorB.x, anchorB.y, anchorB.z,
  );
}

export function createRevoluteJoint(
  bodyA: RigidBodyHandle, bodyB: RigidBodyHandle,
  anchor: { x: number; y: number; z: number },
  axis: { x: number; y: number; z: number },
): JointHandle {
  return bloom_physics_create_revolute_joint(
    bodyA, bodyB,
    anchor.x, anchor.y, anchor.z,
    axis.x, axis.y, axis.z,
  );
}

export function createPrismaticJoint(
  bodyA: RigidBodyHandle, bodyB: RigidBodyHandle,
  anchor: { x: number; y: number; z: number },
  axis: { x: number; y: number; z: number },
): JointHandle {
  return bloom_physics_create_prismatic_joint(
    bodyA, bodyB,
    anchor.x, anchor.y, anchor.z,
    axis.x, axis.y, axis.z,
  );
}

export function destroyJoint(joint: JointHandle): void {
  bloom_physics_destroy_joint(joint);
}
