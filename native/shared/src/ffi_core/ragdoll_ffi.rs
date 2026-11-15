//! EN-025 — ragdoll FFI.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi).
//!
//! The shape of this API is deliberate. A ragdoll is not something a game
//! *configures*; it is something a game *triggers*, once, at the moment of
//! death, and then forgets about. So: one call to build it from a skeleton the
//! engine already has, one call to fire it with the killing impulse, one call
//! per frame to pull the pose back out, one to put it away.
//!
//! Everything else — which bones get bodies, how the capsules are sized, how the
//! joints are limited, how the simulated bodies map back onto skinning matrices
//! — is the engine's problem, because getting any of it wrong produces a corpse
//! that looks *nearly* right, which is worse than none.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_ragdoll {
    () => {
        // bloom_ragdoll_create — allocate a slot. Returns a 1-based handle.
        #[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_create() -> f64 {
            $crate::ffi::guard("bloom_ragdoll_create", move || {
                engine().ragdolls.create() as f64
            })
        }
        #[cfg(not(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32"))))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_create() -> f64 {
            0.0
        }

        // bloom_ragdoll_activate  [EN-025]
        //
        // Build the bodies from the model's CURRENT pose and let go. Call it on
        // the frame the thing dies, with the same (scale, position, yaw) you
        // last passed to bloom_anim_update — that transform is what bridges
        // model space and world space, and it is frozen here: the corpse's
        // motion belongs to the bodies now, not to the dead enemy's position.
        //
        // 8 args (the ARM64 ceiling): the impulse direction and magnitude are
        // set separately by bloom_ragdoll_push, so the killing blow's shove is
        // a distinct decision from the corpse's construction.
        #[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_activate(
            rag: f64,
            anim: f64,
            world: f64,
            scale: f64,
            px: f64,
            py: f64,
            pz: f64,
            rot_y: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_ragdoll_activate", move || {
                let eng = engine();

                let cfg = match eng.ragdolls.get(rag as u32) {
                    Some(r) => r.config.clone(),
                    None => return 0.0,
                };
                let builds = {
                    let Some(a) = eng.models.get_animation(anim) else {
                        return 0.0;
                    };
                    $crate::ragdoll::plan(
                        a,
                        scale as f32,
                        [px as f32, py as f32, pz as f32],
                        rot_y as f32,
                        &cfg,
                    )
                };
                let layer = 1u32; // MOVING
                if builds.is_empty() {
                    return 0.0;
                }

                // --- bodies
                let mut bodies: Vec<f64> = Vec::with_capacity(builds.len());
                for b in builds.iter() {
                    let shape = eng.jolt.create_capsule_shape(b.half_height, b.radius);
                    if shape == 0.0 {
                        bodies.push(0.0);
                        continue;
                    }
                    // Quaternion from the capsule's world basis.
                    let q = $crate::ragdoll::quat_from_mat(&b.world);
                    let body = eng.jolt.create_body(
                        world,
                        shape,
                        2, // DYNAMIC
                        b.world[3][0],
                        b.world[3][1],
                        b.world[3][2],
                        q[0],
                        q[1],
                        q[2],
                        q[3],
                        0.0,
                        0.0,
                        0.0, // linear velocity
                        0.0,
                        0.0,
                        0.0, // angular velocity
                        layer,
                        false, // sensor
                        true,  // allow sleeping — corpses settle
                        false, // ccd: not worth it here
                        true,  // start awake
                        0.8,   // friction: corpses do not skate
                        0.02,  // restitution: they do not bounce
                        // Angular damping does most of the work of making this
                        // read as a body rather than a rag: without it the limbs
                        // keep windmilling long after the thing has landed.
                        0.20,
                        0.75, // lin / ang damping
                        1.0,  // gravity factor
                        0.0,
                        0.0,
                        0.0,
                        0.0, // mass override + inertia
                        0,
                    );
                    bodies.push(body);
                }

                // --- collisions inside the ragdoll
                //
                // A bone's capsule overlaps its parent's and its siblings' at
                // the joint they share. Left colliding, the solver spends every
                // step pushing them apart against the joints and the corpse
                // locks rigid. Everything else in the ragdoll still collides.
                let parents: Vec<i32> = builds
                    .iter()
                    .map(|b| {
                        if b.parent_bone == usize::MAX {
                            -1
                        } else {
                            b.parent_bone as i32
                        }
                    })
                    .collect();
                eng.jolt.bodies_set_linked_group(&bodies, &parents);

                // --- joints
                //
                // Swing-twist about the bone itself: a cone the bone may lean
                // in and a small twist about its own length. An over-twisted
                // limb is the single most obviously WRONG thing a ragdoll can
                // do, so twist is the tighter of the two.
                let mut constraints: Vec<f64> = Vec::new();
                for (i, b) in builds.iter().enumerate() {
                    if b.parent_bone == usize::MAX {
                        continue;
                    }
                    let pa = bodies.get(b.parent_bone).copied().unwrap_or(0.0);
                    let pb = bodies.get(i).copied().unwrap_or(0.0);
                    if pa == 0.0 || pb == 0.0 {
                        continue;
                    }
                    // Any axis perpendicular to the bone: the cone is round.
                    let plane = [b.world[0][0], b.world[0][1], b.world[0][2]];
                    let c = eng.jolt.constraint_swing_twist_world(
                        pa, pb, b.anchor, b.dir, plane, b.swing, b.swing, -b.twist, b.twist,
                    );
                    if c != 0.0 {
                        constraints.push(c);
                    }
                }

                let Some(a) = eng.models.get_animation(anim) else {
                    return 0.0;
                };
                let Some(r) = eng.ragdolls.get_mut(rag as u32) else {
                    return 0.0;
                };
                r.attach(
                    &builds,
                    &bodies,
                    constraints,
                    a,
                    scale as f32,
                    [px as f32, py as f32, pz as f32],
                    rot_y as f32,
                );
                1.0
            })
        }
        #[cfg(not(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32"))))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_activate(
            _r: f64,
            _a: f64,
            _w: f64,
            _s: f64,
            _x: f64,
            _y: f64,
            _z: f64,
            _ry: f64,
        ) -> f64 {
            0.0
        }

        // bloom_ragdoll_push — the killing blow. Applied to every body, so the
        // whole corpse is thrown rather than one limb being yanked off.
        #[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_push(rag: f64, dx: f64, dy: f64, dz: f64, impulse: f64) {
            $crate::ffi::guard("bloom_ragdoll_push", move || {
                let eng = engine();
                let Some(r) = eng.ragdolls.get(rag as u32) else {
                    return;
                };
                if !r.active {
                    return;
                }
                let bodies = r.bodies();
                if bodies.is_empty() {
                    return;
                }
                // Spread the impulse over the bodies so a 12-bone corpse and a
                // 4-bone one take off at the same speed.
                let per = (impulse as f32) / (bodies.len() as f32);
                for b in bodies {
                    eng.jolt
                        .body_add_impulse(b, dx as f32 * per, dy as f32 * per, dz as f32 * per);
                }
            })
        }
        #[cfg(not(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32"))))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_push(_r: f64, _x: f64, _y: f64, _z: f64, _i: f64) {}

        // bloom_ragdoll_update — read the simulated bodies back into the model's
        // joint matrices and upload them. Call once per frame per active
        // ragdoll, then drawModel() as usual. `dt` only ages the ragdoll (the
        // physics world is stepped by the game).
        #[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_update(rag: f64, anim: f64, dt: f64) -> f64 {
            $crate::ffi::guard("bloom_ragdoll_update", move || {
                let eng = engine();

                let (bodies, scale, pos, rot) = {
                    let Some(r) = eng.ragdolls.get(rag as u32) else {
                        return 0.0;
                    };
                    if !r.active {
                        return 0.0;
                    }
                    let (s, p, ry) = r.upload_params();
                    (r.bodies(), s, p, ry)
                };

                let mut world: Vec<[[f32; 4]; 4]> = Vec::with_capacity(bodies.len());
                for b in bodies.iter() {
                    match eng.jolt.body_transform(*b) {
                        Some((p, q)) => world.push($crate::ragdoll::from_pos_quat(p, q)),
                        None => world.push([
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]),
                    }
                }

                {
                    let Some(r) = eng.ragdolls.get_mut(rag as u32) else {
                        return 0.0;
                    };
                    r.age += dt as f32;
                }
                let age = eng.ragdolls.get(rag as u32).map(|r| r.age).unwrap_or(0.0);

                // Split borrow: apply() needs &mut ModelAnimation and &Ragdoll.
                let ragdoll_ptr: *const $crate::ragdoll::Ragdoll =
                    match eng.ragdolls.get(rag as u32) {
                        Some(r) => r,
                        None => return 0.0,
                    };
                if let Some(a) = eng.models.get_animation_mut(anim) {
                    unsafe { (*ragdoll_ptr).apply(a, &world) };
                }

                if let Some(a) = eng.models.get_animation(anim) {
                    if !a.joint_matrices.is_empty() {
                        let (s, c) = (rot.sin(), rot.cos());
                        // PT-7: anim handle = prev-palette pairing key.
                        eng.renderer.set_joint_matrices_scaled(
                            anim.to_bits(),
                            &a.joint_matrices,
                            scale,
                            pos,
                            s,
                            c,
                        );
                    }
                }
                age as f64
            })
        }
        #[cfg(not(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32"))))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_update(_r: f64, _a: f64, _d: f64) -> f64 {
            0.0
        }

        // bloom_ragdoll_release — destroy the bodies and constraints, free the
        // slot for reuse. A pooled ragdoll that is never released leaks bodies
        // into the physics world, which is a slow, invisible death.
        #[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_release(rag: f64) {
            $crate::ffi::guard("bloom_ragdoll_release", move || {
                let eng = engine();
                let (bodies, cons) = {
                    let Some(r) = eng.ragdolls.get(rag as u32) else {
                        return;
                    };
                    (r.bodies(), r.constraint_handles().to_vec())
                };
                // Constraints first: destroying a body out from under a live
                // constraint is how you get a use-after-free in the solver.
                for c in cons {
                    eng.jolt.constraint_destroy(c);
                }
                for b in bodies {
                    eng.jolt.destroy_body(b);
                }
                if let Some(r) = eng.ragdolls.get_mut(rag as u32) {
                    r.reset();
                }
            })
        }
        #[cfg(not(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32"))))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_release(_r: f64) {}

        // bloom_ragdoll_configure — how this slot builds its ragdoll. Kept
        // across release, so a pooled slot is set up once for its rig. A
        // value <= 0 leaves that setting alone.
        #[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_configure(
            rag: f64,
            max_bodies: f64,
            radius_scale: f64,
            swing: f64,
            twist: f64,
        ) {
            $crate::ffi::guard("bloom_ragdoll_configure", move || {
                let Some(r) = engine().ragdolls.get_mut(rag as u32) else {
                    return;
                };
                if max_bodies >= 1.0 {
                    r.config.max_bodies = (max_bodies as usize).min(64);
                }
                if radius_scale > 0.0 {
                    r.config.radius_scale = (radius_scale as f32).clamp(0.02, 0.5);
                }
                if swing > 0.0 {
                    r.config.swing = (swing as f32).min(3.0);
                }
                if twist > 0.0 {
                    r.config.twist = (twist as f32).min(3.0);
                }
            })
        }
        #[cfg(not(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32"))))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_configure(_r: f64, _m: f64, _s: f64, _w: f64, _t: f64) {}

        // bloom_ragdoll_set_joint — one joint's part in the ragdoll.
        // simulate = 0 keeps it out (it rides its parent as it died);
        // swing / twist < 0 mean "the slot's defaults". joint < 0 clears every
        // per-joint setting.
        #[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_set_joint(
            rag: f64,
            joint: f64,
            simulate: f64,
            swing: f64,
            twist: f64,
        ) {
            $crate::ffi::guard("bloom_ragdoll_set_joint", move || {
                let Some(r) = engine().ragdolls.get_mut(rag as u32) else {
                    return;
                };
                if joint < 0.0 {
                    r.config.joints.clear();
                    return;
                }
                let j = joint as usize;
                r.config.joints.retain(|o| o.0 != j);
                r.config
                    .joints
                    .push((j, simulate != 0.0, swing as f32, twist as f32));
            })
        }
        #[cfg(not(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32"))))]
        #[no_mangle]
        pub extern "C" fn bloom_ragdoll_set_joint(_r: f64, _j: f64, _s: f64, _w: f64, _t: f64) {}
    };
}
