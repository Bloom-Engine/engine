//! Jolt pieces a ragdoll needs beyond the general constraint set: the
//! swing-twist joint and a collision group whose linked bodies ignore each
//! other. Separate from `physics_jolt.rs` only because that file is at its
//! line ceiling.

use crate::jolt_sys::*;
use crate::physics_jolt::JoltPhysics;

impl JoltPhysics {
    /// Swing-twist joint between two bodies at a world-space anchor: the
    /// ragdoll articulation. `twist` runs down the child bone, `plane` is
    /// perpendicular to it. Angles in radians.
    #[allow(clippy::too_many_arguments)]
    pub fn constraint_swing_twist_world(
        &mut self,
        body_a: f64,
        body_b: f64,
        anchor: [f32; 3],
        twist: [f32; 3],
        plane: [f32; 3],
        normal_half_cone: f32,
        plane_half_cone: f32,
        twist_min: f32,
        twist_max: f32,
    ) -> f64 {
        let (w, anchors) = match self.make_anchors(
            body_a, body_b, anchor[0], anchor[1], anchor[2], anchor[0], anchor[1], anchor[2], true,
        ) {
            Some(v) => v,
            None => return 0.0,
        };
        let c = unsafe {
            bj_constraint_swing_twist(
                w,
                &anchors,
                BjVec3 {
                    x: twist[0],
                    y: twist[1],
                    z: twist[2],
                },
                BjVec3 {
                    x: plane[0],
                    y: plane[1],
                    z: plane[2],
                },
                normal_half_cone,
                plane_half_cone,
                twist_min,
                twist_max,
            )
        };
        if c == BJ_INVALID {
            0.0
        } else {
            self.constraints.alloc((w, c))
        }
    }

    /// Put `bodies` into one collision group in which body i ignores body
    /// `parents[i]` (-1 = none) and any body sharing that parent. Bodies whose
    /// handle does not resolve are skipped; all must be in one world.
    pub fn bodies_set_linked_group(&self, bodies: &[f64], parents: &[i32]) {
        static NEXT_GROUP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
        if bodies.is_empty() || bodies.len() != parents.len() {
            return;
        }
        let mut world: Option<bj_world> = None;
        let mut raw: Vec<bj_body> = Vec::with_capacity(bodies.len());
        for &h in bodies {
            match self.resolve_body(h) {
                Some((w, b)) if world.is_none() || world == Some(w) => {
                    world = Some(w);
                    raw.push(b);
                }
                _ => raw.push(BJ_INVALID),
            }
        }
        let Some(w) = world else { return };
        let group = NEXT_GROUP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            bj_bodies_set_linked_group(w, raw.as_ptr(), parents.as_ptr(), raw.len() as u32, group);
        }
    }
}
