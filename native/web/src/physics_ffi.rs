//! Physics (Jolt 5.x via JoltPhysics.js) — the web bloom_physics_* FFI
//! surface. Split out of lib.rs; every function is a thin wrapper around
//! a JS export of jolt_bridge.js.

use wasm_bindgen::prelude::*;

// ============================================================
// Physics (Jolt 5.x via JoltPhysics.js) — web FFI surface
// ============================================================
//
// Every bloom_physics_* call is a thin wrapper around a JS function defined
// in jolt_bridge.js. Block is generated from package.json manifest.

// 117 physics FFI entries — generated from package.json
#[cfg(feature = "jolt")]
#[wasm_bindgen(module = "/jolt_bridge.js")]
extern "C" {
    #[wasm_bindgen(js_name = initJolt, catch)]
    async fn jb_init_jolt(factory: JsValue) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(js_name = createWorld)]
    fn jb_create_world(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64;
    #[wasm_bindgen(js_name = destroyWorld)]
    fn jb_destroy_world(a0: f64);
    #[wasm_bindgen(js_name = setGravity)]
    fn jb_set_gravity(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = getGravity)]
    fn jb_get_gravity(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = optimizeBroadphase)]
    fn jb_optimize_broadphase(a0: f64);
    #[wasm_bindgen(js_name = step)]
    fn jb_step(a0: f64, a1: f64, a2: f64);
    #[wasm_bindgen(js_name = stepFixed)]
    fn jb_step_fixed(a0: f64, a1: f64, a2: f64) -> f64;
    #[wasm_bindgen(js_name = setFixedTimestep)]
    fn jb_set_fixed_timestep(a0: f64, a1: f64, a2: f64);
    #[wasm_bindgen(js_name = setInterpolation)]
    fn jb_set_interpolation(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = getStepAlpha)]
    fn jb_get_step_alpha(a0: f64) -> f64;
    #[wasm_bindgen(js_name = setLayerCollides)]
    fn jb_set_layer_collides(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = getLayerCollides)]
    fn jb_get_layer_collides(a0: f64, a1: f64, a2: f64) -> f64;
    #[wasm_bindgen(js_name = bodyCount)]
    fn jb_body_count(a0: f64) -> f64;
    #[wasm_bindgen(js_name = activeBodyCount)]
    fn jb_active_body_count(a0: f64) -> f64;
    #[wasm_bindgen(js_name = shapeBox)]
    fn jb_shape_box(a0: f64, a1: f64, a2: f64, a3: f64) -> f64;
    #[wasm_bindgen(js_name = shapeSphere)]
    fn jb_shape_sphere(a0: f64) -> f64;
    #[wasm_bindgen(js_name = shapeCapsule)]
    pub(crate) fn jb_shape_capsule(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = shapeCylinder)]
    fn jb_shape_cylinder(a0: f64, a1: f64, a2: f64) -> f64;
    #[wasm_bindgen(js_name = shapeScaled)]
    fn jb_shape_scaled(a0: f64, a1: f64, a2: f64, a3: f64) -> f64;
    #[wasm_bindgen(js_name = shapeOffsetCom)]
    fn jb_shape_offset_com(a0: f64, a1: f64, a2: f64, a3: f64) -> f64;
    #[wasm_bindgen(js_name = shapeRelease)]
    fn jb_shape_release(a0: f64);
    #[wasm_bindgen(js_name = scratchReset)]
    fn jb_scratch_reset();
    #[wasm_bindgen(js_name = scratchPushF32)]
    fn jb_scratch_push_f32(a0: f64);
    #[wasm_bindgen(js_name = scratchPushU32)]
    fn jb_scratch_push_u32(a0: f64);
    #[wasm_bindgen(js_name = shapeConvexHull)]
    fn jb_shape_convex_hull(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = shapeMesh)]
    fn jb_shape_mesh(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = shapeHeightfield)]
    fn jb_shape_heightfield(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64) -> f64;
    #[wasm_bindgen(js_name = compoundBegin)]
    fn jb_compound_begin();
    #[wasm_bindgen(js_name = compoundAddChild)]
    fn jb_compound_add_child(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64);
    #[wasm_bindgen(js_name = compoundEnd)]
    fn jb_compound_end() -> f64;
    #[wasm_bindgen(js_name = shapeBounds)]
    fn jb_shape_bounds(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = shapeVolume)]
    fn jb_shape_volume(a0: f64) -> f64;
    #[wasm_bindgen(js_name = bodyCreate)]
    pub(crate) fn jb_body_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64) -> f64;
    #[wasm_bindgen(js_name = bodyDestroy)]
    pub(crate) fn jb_body_destroy(a0: f64);
    #[wasm_bindgen(js_name = bodyActivate)]
    fn jb_body_activate(a0: f64);
    #[wasm_bindgen(js_name = bodyDeactivate)]
    fn jb_body_deactivate(a0: f64);
    #[wasm_bindgen(js_name = bodyIsActive)]
    fn jb_body_is_active(a0: f64) -> f64;
    #[wasm_bindgen(js_name = bodyIsValid)]
    fn jb_body_is_valid(a0: f64) -> f64;
    #[wasm_bindgen(js_name = bodyGetPosition)]
    pub(crate) fn jb_body_get_position(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = bodyGetRotation)]
    pub(crate) fn jb_body_get_rotation(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = bodySetPosition)]
    fn jb_body_set_position(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64);
    #[wasm_bindgen(js_name = bodySetRotation)]
    fn jb_body_set_rotation(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64);
    #[wasm_bindgen(js_name = bodySetTransform)]
    fn jb_body_set_transform(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64);
    #[wasm_bindgen(js_name = bodyMoveKinematic)]
    fn jb_body_move_kinematic(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64);
    #[wasm_bindgen(js_name = bodyGetLinearVelocity)]
    fn jb_body_get_linear_velocity(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = bodyGetAngularVelocity)]
    fn jb_body_get_angular_velocity(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = bodyGetPointVelocity)]
    fn jb_body_get_point_velocity(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64;
    #[wasm_bindgen(js_name = bodySetLinearVelocity)]
    fn jb_body_set_linear_velocity(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodySetAngularVelocity)]
    fn jb_body_set_angular_velocity(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyAddForce)]
    fn jb_body_add_force(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyAddImpulse)]
    pub(crate) fn jb_body_add_impulse(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyAddTorque)]
    fn jb_body_add_torque(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyAddAngularImpulse)]
    fn jb_body_add_angular_impulse(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyAddForceAt)]
    fn jb_body_add_force_at(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64);
    #[wasm_bindgen(js_name = bodyAddImpulseAt)]
    fn jb_body_add_impulse_at(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64);
    #[wasm_bindgen(js_name = bodySetFriction)]
    pub(crate) fn jb_body_set_friction(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetRestitution)]
    pub(crate) fn jb_body_set_restitution(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetLinearDamping)]
    pub(crate) fn jb_body_set_linear_damping(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetAngularDamping)]
    pub(crate) fn jb_body_set_angular_damping(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetGravityFactor)]
    fn jb_body_set_gravity_factor(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetCcd)]
    fn jb_body_set_ccd(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetMotionType)]
    fn jb_body_set_motion_type(a0: f64, a1: f64, a2: f64);
    #[wasm_bindgen(js_name = bodySetObjectLayer)]
    fn jb_body_set_object_layer(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetIsSensor)]
    fn jb_body_set_is_sensor(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetAllowSleeping)]
    fn jb_body_set_allow_sleeping(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = bodySetShape)]
    fn jb_body_set_shape(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyLockRotationAxes)]
    fn jb_body_lock_rotation_axes(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyLockTranslationAxes)]
    fn jb_body_lock_translation_axes(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = bodyGetMass)]
    fn jb_body_get_mass(a0: f64) -> f64;
    #[wasm_bindgen(js_name = bodyGetFriction)]
    fn jb_body_get_friction(a0: f64) -> f64;
    #[wasm_bindgen(js_name = bodyGetRestitution)]
    fn jb_body_get_restitution(a0: f64) -> f64;
    #[wasm_bindgen(js_name = bodyGetObjectLayer)]
    fn jb_body_get_object_layer(a0: f64) -> f64;
    #[wasm_bindgen(js_name = bodySetUserData)]
    fn jb_body_set_user_data(a0: f64, a1: f64, a2: f64);
    #[wasm_bindgen(js_name = bodyGetUserData)]
    fn jb_body_get_user_data(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = raycast)]
    fn jb_raycast(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) -> f64;
    #[wasm_bindgen(js_name = raycastAll)]
    fn jb_raycast_all(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64) -> f64;
    #[wasm_bindgen(js_name = rayHitCount)]
    fn jb_ray_hit_count() -> f64;
    #[wasm_bindgen(js_name = rayHitBody)]
    fn jb_ray_hit_body(a0: f64) -> f64;
    #[wasm_bindgen(js_name = rayHitAxis)]
    fn jb_ray_hit_axis(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = rayHitFraction)]
    fn jb_ray_hit_fraction(a0: f64) -> f64;
    #[wasm_bindgen(js_name = rayHitSubShape)]
    fn jb_ray_hit_sub_shape(a0: f64) -> f64;
    #[wasm_bindgen(js_name = overlapSphere)]
    fn jb_overlap_sphere(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64) -> f64;
    #[wasm_bindgen(js_name = overlapPoint)]
    fn jb_overlap_point(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64) -> f64;
    #[wasm_bindgen(js_name = overlapBox)]
    fn jb_overlap_box(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64) -> f64;
    #[wasm_bindgen(js_name = overlapBody)]
    fn jb_overlap_body(a0: f64) -> f64;
    #[wasm_bindgen(js_name = constraintFixed)]
    fn jb_constraint_fixed(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) -> f64;
    #[wasm_bindgen(js_name = constraintPoint)]
    fn jb_constraint_point(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) -> f64;
    #[wasm_bindgen(js_name = constraintHinge)]
    fn jb_constraint_hinge(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64) -> f64;
    #[wasm_bindgen(js_name = constraintSlider)]
    fn jb_constraint_slider(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64) -> f64;
    #[wasm_bindgen(js_name = constraintDistance)]
    fn jb_constraint_distance(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64) -> f64;
    // EN-063: the ragdoll joint — six-DOF with translation locked and
    // per-axis rotation limits. Args: bodyA, bodyB, anchorA(x,y,z),
    // anchorB(x,y,z), rotLimits(xMin,xMax,yMin,yMax,zMin,zMax), worldSpace.
    #[wasm_bindgen(js_name = constraintSixDofLockedTranslation)]
    pub(crate) fn jb_constraint_six_dof(
        a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64,
        a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64, a14: f64,
    ) -> f64;
    #[wasm_bindgen(js_name = constraintDestroy)]
    pub(crate) fn jb_constraint_destroy(a0: f64);
    #[wasm_bindgen(js_name = constraintSetEnabled)]
    fn jb_constraint_set_enabled(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = contactCount)]
    fn jb_contact_count() -> f64;
    #[wasm_bindgen(js_name = contactField)]
    fn jb_contact_field(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = clearContacts)]
    fn jb_clear_contacts(a0: f64);
    #[wasm_bindgen(js_name = characterCreate)]
    fn jb_character_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64, a14: f64, a15: f64, a16: f64, a17: f64, a18: f64) -> f64;
    #[wasm_bindgen(js_name = characterDestroy)]
    fn jb_character_destroy(a0: f64);
    #[wasm_bindgen(js_name = characterUpdate)]
    fn jb_character_update(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64);
    #[wasm_bindgen(js_name = characterGetPosition)]
    fn jb_character_get_position(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = characterGetRotation)]
    fn jb_character_get_rotation(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = characterSetPosition)]
    fn jb_character_set_position(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = characterSetRotation)]
    fn jb_character_set_rotation(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64);
    #[wasm_bindgen(js_name = characterGetLinearVelocity)]
    fn jb_character_get_linear_velocity(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = characterSetLinearVelocity)]
    fn jb_character_set_linear_velocity(a0: f64, a1: f64, a2: f64, a3: f64);
    #[wasm_bindgen(js_name = characterGetGroundState)]
    fn jb_character_get_ground_state(a0: f64) -> f64;
    #[wasm_bindgen(js_name = characterGetGroundNormal)]
    fn jb_character_get_ground_normal(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = characterGetGroundPosition)]
    fn jb_character_get_ground_position(a0: f64, a1: f64) -> f64;
    #[wasm_bindgen(js_name = characterGetGroundBody)]
    fn jb_character_get_ground_body(a0: f64) -> f64;
    #[wasm_bindgen(js_name = characterSetShape)]
    fn jb_character_set_shape(a0: f64, a1: f64);
    #[wasm_bindgen(js_name = softBodyCreate)]
    fn jb_soft_body_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64, a14: f64) -> f64;
    #[wasm_bindgen(js_name = softBodyVertexCount)]
    fn jb_soft_body_vertex_count(a0: f64) -> f64;
    #[wasm_bindgen(js_name = softBodyGetVertex)]
    fn jb_soft_body_get_vertex(a0: f64, a1: f64, a2: f64) -> f64;
    #[wasm_bindgen(js_name = softBodySetVertex)]
    fn jb_soft_body_set_vertex(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64);
    #[wasm_bindgen(js_name = softBodySetVertexInvMass)]
    fn jb_soft_body_set_vertex_inv_mass(a0: f64, a1: f64, a2: f64);
    #[wasm_bindgen(js_name = vehicleCreate)]
    fn jb_vehicle_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64, a14: f64, a15: f64, a16: f64, a17: f64, a18: f64, a19: f64, a20: f64, a21: f64, a22: f64, a23: f64, a24: f64, a25: f64, a26: f64, a27: f64, a28: f64, a29: f64, a30: f64, a31: f64, a32: f64, a33: f64, a34: f64, a35: f64, a36: f64, a37: f64) -> f64;
    #[wasm_bindgen(js_name = vehicleDestroy)]
    fn jb_vehicle_destroy(a0: f64);
    #[wasm_bindgen(js_name = vehicleGetChassis)]
    fn jb_vehicle_get_chassis(a0: f64) -> f64;
    #[wasm_bindgen(js_name = vehicleSetInput)]
    fn jb_vehicle_set_input(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64);
    #[wasm_bindgen(js_name = vehicleGetWheelTransform)]
    fn jb_vehicle_get_wheel_transform(a0: f64, a1: f64, a2: f64) -> f64;
    #[wasm_bindgen(js_name = vehicleGetEngineRpm)]
    fn jb_vehicle_get_engine_rpm(a0: f64) -> f64;
    #[wasm_bindgen(js_name = vehicleGetWheelAngularVelocity)]
    fn jb_vehicle_get_wheel_angular_velocity(a0: f64, a1: f64) -> f64;
}

#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_create_world(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64 { jb_create_world(a0, a1, a2, a3, a4) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_destroy_world(a0: f64) { jb_destroy_world(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_set_gravity(a0: f64, a1: f64, a2: f64, a3: f64) { jb_set_gravity(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_get_gravity(a0: f64, a1: f64) -> f64 { jb_get_gravity(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_optimize_broadphase(a0: f64) { jb_optimize_broadphase(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_step(a0: f64, a1: f64, a2: f64) { jb_step(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_step_fixed(a0: f64, a1: f64, a2: f64) -> f64 { jb_step_fixed(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_set_fixed_timestep(a0: f64, a1: f64, a2: f64) { jb_set_fixed_timestep(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_set_interpolation(a0: f64, a1: f64) { jb_set_interpolation(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_get_step_alpha(a0: f64) -> f64 { jb_get_step_alpha(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_set_layer_collides(a0: f64, a1: f64, a2: f64, a3: f64) { jb_set_layer_collides(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_get_layer_collides(a0: f64, a1: f64, a2: f64) -> f64 { jb_get_layer_collides(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_count(a0: f64) -> f64 { jb_body_count(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_active_body_count(a0: f64) -> f64 { jb_active_body_count(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_box(a0: f64, a1: f64, a2: f64, a3: f64) -> f64 { jb_shape_box(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_sphere(a0: f64) -> f64 { jb_shape_sphere(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_capsule(a0: f64, a1: f64) -> f64 { jb_shape_capsule(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_cylinder(a0: f64, a1: f64, a2: f64) -> f64 { jb_shape_cylinder(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_scaled(a0: f64, a1: f64, a2: f64, a3: f64) -> f64 { jb_shape_scaled(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_offset_com(a0: f64, a1: f64, a2: f64, a3: f64) -> f64 { jb_shape_offset_com(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_release(a0: f64) { jb_shape_release(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_scratch_reset() { jb_scratch_reset() }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_scratch_push_f32(a0: f64) { jb_scratch_push_f32(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_scratch_push_u32(a0: f64) { jb_scratch_push_u32(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_convex_hull(a0: f64, a1: f64) -> f64 { jb_shape_convex_hull(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_mesh(a0: f64, a1: f64) -> f64 { jb_shape_mesh(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_heightfield(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64) -> f64 { jb_shape_heightfield(a0, a1, a2, a3, a4, a5, a6, a7) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_compound_begin() { jb_compound_begin() }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_compound_add_child(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64) { jb_compound_add_child(a0, a1, a2, a3, a4, a5, a6, a7) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_compound_end() -> f64 { jb_compound_end() }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_bounds(a0: f64, a1: f64) -> f64 { jb_shape_bounds(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_shape_volume(a0: f64) -> f64 { jb_shape_volume(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64) -> f64 { jb_body_create(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_destroy(a0: f64) { jb_body_destroy(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_activate(a0: f64) { jb_body_activate(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_deactivate(a0: f64) { jb_body_deactivate(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_is_active(a0: f64) -> f64 { jb_body_is_active(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_is_valid(a0: f64) -> f64 { jb_body_is_valid(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_position(a0: f64, a1: f64) -> f64 { jb_body_get_position(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_rotation(a0: f64, a1: f64) -> f64 { jb_body_get_rotation(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_position(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) { jb_body_set_position(a0, a1, a2, a3, a4) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_rotation(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64) { jb_body_set_rotation(a0, a1, a2, a3, a4, a5) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_transform(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) { jb_body_set_transform(a0, a1, a2, a3, a4, a5, a6, a7, a8) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_move_kinematic(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) { jb_body_move_kinematic(a0, a1, a2, a3, a4, a5, a6, a7, a8) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_linear_velocity(a0: f64, a1: f64) -> f64 { jb_body_get_linear_velocity(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_angular_velocity(a0: f64, a1: f64) -> f64 { jb_body_get_angular_velocity(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_point_velocity(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64 { jb_body_get_point_velocity(a0, a1, a2, a3, a4) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_linear_velocity(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_set_linear_velocity(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_angular_velocity(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_set_angular_velocity(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_add_force(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_add_force(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_add_impulse(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_add_impulse(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_add_torque(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_add_torque(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_add_angular_impulse(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_add_angular_impulse(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_add_force_at(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64) { jb_body_add_force_at(a0, a1, a2, a3, a4, a5, a6) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_add_impulse_at(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64) { jb_body_add_impulse_at(a0, a1, a2, a3, a4, a5, a6) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_friction(a0: f64, a1: f64) { jb_body_set_friction(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_restitution(a0: f64, a1: f64) { jb_body_set_restitution(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_linear_damping(a0: f64, a1: f64) { jb_body_set_linear_damping(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_angular_damping(a0: f64, a1: f64) { jb_body_set_angular_damping(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_gravity_factor(a0: f64, a1: f64) { jb_body_set_gravity_factor(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_ccd(a0: f64, a1: f64) { jb_body_set_ccd(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_motion_type(a0: f64, a1: f64, a2: f64) { jb_body_set_motion_type(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_object_layer(a0: f64, a1: f64) { jb_body_set_object_layer(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_is_sensor(a0: f64, a1: f64) { jb_body_set_is_sensor(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_allow_sleeping(a0: f64, a1: f64) { jb_body_set_allow_sleeping(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_shape(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_set_shape(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_lock_rotation_axes(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_lock_rotation_axes(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_lock_translation_axes(a0: f64, a1: f64, a2: f64, a3: f64) { jb_body_lock_translation_axes(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_mass(a0: f64) -> f64 { jb_body_get_mass(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_friction(a0: f64) -> f64 { jb_body_get_friction(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_restitution(a0: f64) -> f64 { jb_body_get_restitution(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_object_layer(a0: f64) -> f64 { jb_body_get_object_layer(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_set_user_data(a0: f64, a1: f64, a2: f64) { jb_body_set_user_data(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_body_get_user_data(a0: f64, a1: f64) -> f64 { jb_body_get_user_data(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_raycast(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) -> f64 { jb_raycast(a0, a1, a2, a3, a4, a5, a6, a7, a8) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_raycast_all(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64) -> f64 { jb_raycast_all(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_ray_hit_count() -> f64 { jb_ray_hit_count() }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_ray_hit_body(a0: f64) -> f64 { jb_ray_hit_body(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_ray_hit_axis(a0: f64, a1: f64) -> f64 { jb_ray_hit_axis(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_ray_hit_fraction(a0: f64) -> f64 { jb_ray_hit_fraction(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_ray_hit_sub_shape(a0: f64) -> f64 { jb_ray_hit_sub_shape(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_overlap_sphere(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64) -> f64 { jb_overlap_sphere(a0, a1, a2, a3, a4, a5, a6) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_overlap_point(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64) -> f64 { jb_overlap_point(a0, a1, a2, a3, a4, a5) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_overlap_box(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64) -> f64 { jb_overlap_box(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_overlap_body(a0: f64) -> f64 { jb_overlap_body(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_constraint_fixed(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) -> f64 { jb_constraint_fixed(a0, a1, a2, a3, a4, a5, a6, a7, a8) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_constraint_point(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64) -> f64 { jb_constraint_point(a0, a1, a2, a3, a4, a5, a6, a7, a8) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_constraint_hinge(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64) -> f64 { jb_constraint_hinge(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_constraint_slider(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64) -> f64 { jb_constraint_slider(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_constraint_distance(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64) -> f64 { jb_constraint_distance(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_constraint_destroy(a0: f64) { jb_constraint_destroy(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_constraint_set_enabled(a0: f64, a1: f64) { jb_constraint_set_enabled(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_contact_count() -> f64 { jb_contact_count() }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_contact_field(a0: f64, a1: f64) -> f64 { jb_contact_field(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_clear_contacts(a0: f64) { jb_clear_contacts(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64, a14: f64, a15: f64, a16: f64, a17: f64, a18: f64) -> f64 { jb_character_create(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_destroy(a0: f64) { jb_character_destroy(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_update(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) { jb_character_update(a0, a1, a2, a3, a4) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_get_position(a0: f64, a1: f64) -> f64 { jb_character_get_position(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_get_rotation(a0: f64, a1: f64) -> f64 { jb_character_get_rotation(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_set_position(a0: f64, a1: f64, a2: f64, a3: f64) { jb_character_set_position(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_set_rotation(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) { jb_character_set_rotation(a0, a1, a2, a3, a4) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_get_linear_velocity(a0: f64, a1: f64) -> f64 { jb_character_get_linear_velocity(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_set_linear_velocity(a0: f64, a1: f64, a2: f64, a3: f64) { jb_character_set_linear_velocity(a0, a1, a2, a3) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_get_ground_state(a0: f64) -> f64 { jb_character_get_ground_state(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_get_ground_normal(a0: f64, a1: f64) -> f64 { jb_character_get_ground_normal(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_get_ground_position(a0: f64, a1: f64) -> f64 { jb_character_get_ground_position(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_get_ground_body(a0: f64) -> f64 { jb_character_get_ground_body(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_character_set_shape(a0: f64, a1: f64) { jb_character_set_shape(a0, a1) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_soft_body_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64, a14: f64) -> f64 { jb_soft_body_create(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_soft_body_vertex_count(a0: f64) -> f64 { jb_soft_body_vertex_count(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_soft_body_get_vertex(a0: f64, a1: f64, a2: f64) -> f64 { jb_soft_body_get_vertex(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_soft_body_set_vertex(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) { jb_soft_body_set_vertex(a0, a1, a2, a3, a4) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_soft_body_set_vertex_inv_mass(a0: f64, a1: f64, a2: f64) { jb_soft_body_set_vertex_inv_mass(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_vehicle_create(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64, a7: f64, a8: f64, a9: f64, a10: f64, a11: f64, a12: f64, a13: f64, a14: f64, a15: f64, a16: f64, a17: f64, a18: f64, a19: f64, a20: f64, a21: f64, a22: f64, a23: f64, a24: f64, a25: f64, a26: f64, a27: f64, a28: f64, a29: f64, a30: f64, a31: f64, a32: f64, a33: f64, a34: f64, a35: f64, a36: f64, a37: f64) -> f64 { jb_vehicle_create(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_vehicle_destroy(a0: f64) { jb_vehicle_destroy(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_vehicle_get_chassis(a0: f64) -> f64 { jb_vehicle_get_chassis(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_vehicle_set_input(a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) { jb_vehicle_set_input(a0, a1, a2, a3, a4) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_vehicle_get_wheel_transform(a0: f64, a1: f64, a2: f64) -> f64 { jb_vehicle_get_wheel_transform(a0, a1, a2) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_vehicle_get_engine_rpm(a0: f64) -> f64 { jb_vehicle_get_engine_rpm(a0) }
#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub fn bloom_physics_vehicle_get_wheel_angular_velocity(a0: f64, a1: f64) -> f64 { jb_vehicle_get_wheel_angular_velocity(a0, a1) }

#[cfg(feature = "jolt")]
#[wasm_bindgen]
pub async fn bloom_physics_init_jolt(factory: JsValue) -> Result<(), JsValue> {
    jb_init_jolt(factory).await.map(|_| ())
}
