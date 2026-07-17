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
//! Tuning constants (12 bodies, 0.38 radius fraction, the rot limits, the
//! friction/damping set) are copied from the native FFI verbatim — a corpse
//! must read the same on every platform.

use crate::engine;
use crate::physics_ffi::{
    jb_body_add_impulse, jb_body_create, jb_body_destroy, jb_body_get_position,
    jb_body_get_rotation, jb_body_set_angular_damping, jb_body_set_friction,
    jb_body_set_linear_damping, jb_body_set_restitution, jb_constraint_destroy,
    jb_constraint_six_dof, jb_shape_capsule,
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
        // 12 bodies / 0.38 radius fraction — same sweet spot as native (spine
        // + limbs; past that you buy fingers, which cost solver time and buy
        // jitter).
        bloom_shared::ragdoll::plan(
            a,
            scale as f32,
            [px as f32, py as f32, pz as f32],
            rot_y as f32,
            12,
            0.38,
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

    // --- joints. Tight-ish; twist (y) held hardest — an over-twisted limb is
    // the single most obviously WRONG thing a ragdoll can do.
    let rot_limits: [f64; 6] = [-0.8, 0.8, -0.25, 0.25, -0.8, 0.8];
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
        let c = jb_constraint_six_dof(
            pa,
            pb,
            b.anchor[0] as f64,
            b.anchor[1] as f64,
            b.anchor[2] as f64,
            b.anchor[0] as f64,
            b.anchor[1] as f64,
            b.anchor[2] as f64,
            rot_limits[0],
            rot_limits[1],
            rot_limits[2],
            rot_limits[3],
            rot_limits[4],
            rot_limits[5],
            1.0, // anchors given in world space
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
    let ragdoll_ptr: *const bloom_shared::ragdoll::Ragdoll =
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
            eng.renderer
                .set_joint_matrices_scaled(anim.to_bits(), &a.joint_matrices, scale, pos, s, c);
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
        *r = bloom_shared::ragdoll::Ragdoll::new();
    }
}
