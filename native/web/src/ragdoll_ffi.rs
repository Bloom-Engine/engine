//! EN-063 — ragdolls on web.
//!
//! The native ragdoll FFI (`bloom-shared/src/ffi_core/ragdoll_ffi.rs`) is a
//! thin choreography layer: `ragdoll::plan()` decides the capsules, Jolt
//! creates them, `Ragdoll::attach/apply` map simulated bodies back onto the
//! skinning palette. All of that planning/pose math is pure Rust and compiles
//! on wasm32 — the only thing missing on web was the Jolt half, which lives
//! in JS (`jolt_bridge.js`). So this module is the same choreography with
//! `eng.jolt.*` swapped for the bridge externs, plus one new bridge function
//! (`constraintSixDofLockedTranslation`) for the joint type ragdolls use.
//!
//! The plan and its limits come from the slot's `RagdollConfig`, the
//! friction/damping set is copied from the native FFI verbatim — a corpse
//! must read the same on every platform.

use crate::engine;
use crate::physics_ffi::{
    jb_body_add_impulse, jb_body_create, jb_body_destroy, jb_body_get_position,
    jb_body_get_rotation, jb_body_set_angular_damping, jb_body_set_friction,
    jb_body_set_linear_damping, jb_body_set_restitution, jb_constraint_destroy,
    jb_bodies_set_linked_group, jb_constraint_six_dof, jb_constraint_swing_twist, jb_shape_capsule,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn bloom_ragdoll_create() -> f64 {
    engine().ragdolls.create() as f64
}

#[wasm_bindgen]
pub fn bloom_ragdoll_activate(
    rag: f64,
    anim: f64,
    world: f64,
    scale: f64,
    px: f64,
    py: f64,
    pz: f64,
    rot_y: f64,
) -> f64 {
    let eng = engine();

    let builds = {
        let Some(a) = eng.models.get_animation(anim) else {
            return 0.0;
        };
        let cfg = match eng.ragdolls.get(rag as u32) {
            Some(r) => r.config.clone(),
            None => return 0.0,
        };
        bloom_shared::ragdoll::plan(
            a,
            scale as f32,
            [px as f32, py as f32, pz as f32],
            rot_y as f32,
            &cfg,
        )
    };
    if builds.is_empty() {
        return 0.0;
    }

    // --- bodies (layer 1 = MOVING, motion type 2 = DYNAMIC)
    let mut bodies: Vec<f64> = Vec::with_capacity(builds.len());
    for b in builds.iter() {
        let shape = jb_shape_capsule(b.half_height as f64, b.radius as f64);
        if shape == 0.0 {
            bodies.push(0.0);
            continue;
        }
        let q = bloom_shared::ragdoll::quat_from_mat(&b.world);
        let body = jb_body_create(
            world,
            shape,
            2.0,
            b.world[3][0] as f64,
            b.world[3][1] as f64,
            b.world[3][2] as f64,
            q[0] as f64,
            q[1] as f64,
            q[2] as f64,
            q[3] as f64,
            1.0,
        );
        if body != 0.0 {
            jb_body_set_friction(body, 0.8); // corpses do not skate
            jb_body_set_restitution(body, 0.02); // they do not bounce
            jb_body_set_linear_damping(body, 0.20);
            // Angular damping does most of the work of making this read as a
            // body rather than a rag.
            jb_body_set_angular_damping(body, 0.75);
        }
        bodies.push(body);
    }

    // --- collisions inside the ragdoll: a bone ignores its parent and its
    // siblings, whose capsules overlap its own at the shared joint.
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
    jb_bodies_set_linked_group(&bodies, &parents);

    // --- joints: swing-twist about the bone, as native. A JoltPhysics.js
    // build without that type falls back to the six-DOF joint on world axes.
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
        let mut c = jb_constraint_swing_twist(
            pa,
            pb,
            b.anchor[0] as f64,
            b.anchor[1] as f64,
            b.anchor[2] as f64,
            b.dir[0] as f64,
            b.dir[1] as f64,
            b.dir[2] as f64,
            b.world[0][0] as f64,
            b.world[0][1] as f64,
            b.world[0][2] as f64,
            b.swing as f64,
            b.twist as f64,
        );
        if c == 0.0 {
            let (s, t) = (b.swing as f64, b.twist as f64);
            c = jb_constraint_six_dof(
                pa,
                pb,
                b.anchor[0] as f64,
                b.anchor[1] as f64,
                b.anchor[2] as f64,
                b.anchor[0] as f64,
                b.anchor[1] as f64,
                b.anchor[2] as f64,
                -s,
                s,
                -t,
                t,
                -s,
                s,
                1.0, // anchors given in world space
            );
        }
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
}

#[wasm_bindgen]
pub fn bloom_ragdoll_push(rag: f64, dx: f64, dy: f64, dz: f64, impulse: f64) {
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
    // Spread the impulse over the bodies so a 12-bone corpse and a 4-bone one
    // take off at the same speed.
    let per = impulse / (bodies.len() as f64);
    for b in bodies {
        jb_body_add_impulse(b, dx * per, dy * per, dz * per);
    }
}

#[wasm_bindgen]
pub fn bloom_ragdoll_update(rag: f64, anim: f64, dt: f64) -> f64 {
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
        let p = [
            jb_body_get_position(*b, 0.0) as f32,
            jb_body_get_position(*b, 1.0) as f32,
            jb_body_get_position(*b, 2.0) as f32,
        ];
        let q = [
            jb_body_get_rotation(*b, 0.0) as f32,
            jb_body_get_rotation(*b, 1.0) as f32,
            jb_body_get_rotation(*b, 2.0) as f32,
            jb_body_get_rotation(*b, 3.0) as f32,
        ];
        world.push(bloom_shared::ragdoll::from_pos_quat(p, q));
    }

    {
        let Some(r) = eng.ragdolls.get_mut(rag as u32) else {
            return 0.0;
        };
        r.age += dt as f32;
    }
    let age = eng.ragdolls.get(rag as u32).map(|r| r.age).unwrap_or(0.0);

    // Split borrow: apply() needs &mut ModelAnimation and &Ragdoll.
    let ragdoll_ptr: *const bloom_shared::ragdoll::Ragdoll = match eng.ragdolls.get(rag as u32) {
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
}

#[wasm_bindgen]
pub fn bloom_ragdoll_release(rag: f64) {
    let eng = engine();
    let (bodies, cons) = {
        let Some(r) = eng.ragdolls.get(rag as u32) else {
            return;
        };
        (r.bodies(), r.constraint_handles().to_vec())
    };
    // Constraints first: destroying a body out from under a live constraint
    // is how you get a use-after-free in the solver.
    for c in cons {
        jb_constraint_destroy(c);
    }
    for b in bodies {
        jb_body_destroy(b);
    }
    if let Some(r) = eng.ragdolls.get_mut(rag as u32) {
        r.reset();
    }
}

#[wasm_bindgen]
pub fn bloom_ragdoll_configure(rag: f64, max_bodies: f64, radius_scale: f64, swing: f64, twist: f64) {
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
}

#[wasm_bindgen]
pub fn bloom_ragdoll_set_joint(rag: f64, joint: f64, simulate: f64, swing: f64, twist: f64) {
    let Some(r) = engine().ragdolls.get_mut(rag as u32) else {
        return;
    };
    if joint < 0.0 {
        r.config.joints.clear();
        return;
    }
    let j = joint as usize;
    r.config.joints.retain(|o| o.0 != j);
    r.config.joints.push((j, simulate != 0.0, swing as f32, twist as f32));
}
