//! EN-064 — the procedural half of a character rig, applied on top of the
//! EN-028 mixer's sampled pose:
//!
//! 1. **Blend partners**: up to `BLEND_SLOTS` extra clips mixed into the base
//!    clip by weight, phase-locked to it (a strafe blend space: forward as
//!    the base, left/right/back as partners).
//! 2. **Joint rotations**: up to `ROT_SLOTS` joints turned about a
//!    model-space axis after sampling (an aim offset spread over the spine).
//! 3. **Two-bone IK**: up to `IK_SLOTS` chains (hand, forearm, upper arm)
//!    pulled to a model-space target, with an optional pole for the elbow.
//!
//! All of it is per-instance state the game restates when it changes; the
//! solve runs inside `advance_and_update`, after the masked layer and before
//! the skinning matrices are built. Everything here is model space: the
//! space `jointWorld` reports, before the scale / position / yaw handed to
//! `animUpdate`.
//!
//! The maths is plain `f32` arrays so this module compiles with or without
//! `models3d` and can be unit-tested without a GLB.

pub const BLEND_SLOTS: usize = 3;
pub const ROT_SLOTS: usize = 8;
pub const IK_SLOTS: usize = 2;

#[derive(Clone, Copy)]
pub struct BlendSlot {
    /// `< 0` = unused.
    pub clip: i32,
    pub weight: f32,
}

#[derive(Clone, Copy)]
pub struct RotSlot {
    /// `< 0` = unused.
    pub joint: i32,
    pub axis: [f32; 3],
    pub angle: f32,
}

#[derive(Clone, Copy)]
pub struct IkSlot {
    /// The END joint (hand / foot); its parent and grandparent are the chain.
    /// `< 0` = unused.
    pub end: i32,
    pub target: [f32; 3],
    pub weight: f32,
    /// Restore the end joint's pre-solve model-space rotation afterwards, so
    /// a hand keeps the grip the clip gave it instead of turning with the arm.
    pub keep_end_rotation: bool,
    pub pole: [f32; 3],
    pub pole_on: bool,
}

#[derive(Clone)]
pub struct AnimRig {
    pub blends: [BlendSlot; BLEND_SLOTS],
    pub rots: [RotSlot; ROT_SLOTS],
    pub iks: [IkSlot; IK_SLOTS],
}

impl Default for AnimRig {
    fn default() -> Self {
        Self {
            blends: [BlendSlot {
                clip: -1,
                weight: 0.0,
            }; BLEND_SLOTS],
            rots: [RotSlot {
                joint: -1,
                axis: [0.0, 1.0, 0.0],
                angle: 0.0,
            }; ROT_SLOTS],
            iks: [IkSlot {
                end: -1,
                target: [0.0; 3],
                weight: 0.0,
                keep_end_rotation: false,
                pole: [0.0; 3],
                pole_on: false,
            }; IK_SLOTS],
        }
    }
}

impl AnimRig {
    /// True when nothing after the blend stage is active (the common case for
    /// every enemy): lets the caller skip the FK passes entirely.
    pub fn has_post(&self) -> bool {
        self.rots.iter().any(|r| r.joint >= 0 && r.angle != 0.0)
            || self.iks.iter().any(|k| k.end >= 0 && k.weight > 0.0)
    }
}

// ---- small vector / quaternion kit (xyzw quaternions) ------------------------

type V3 = [f32; 3];
type Q = [f32; 4];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn len(a: V3) -> f32 {
    dot(a, a).sqrt()
}
fn norm(a: V3) -> V3 {
    let l = len(a);
    if l > 1e-8 {
        mul(a, 1.0 / l)
    } else {
        [0.0; 3]
    }
}
fn qmul(a: Q, b: Q) -> Q {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}
fn qconj(a: Q) -> Q {
    [-a[0], -a[1], -a[2], a[3]]
}
fn qnorm(a: Q) -> Q {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
    if l > 1e-8 {
        [a[0] / l, a[1] / l, a[2] / l, a[3] / l]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}
fn qrot(q: Q, v: V3) -> V3 {
    let u = [q[0], q[1], q[2]];
    let t = mul(cross(u, v), 2.0);
    add(add(v, mul(t, q[3])), cross(u, t))
}
fn axis_angle(axis: V3, angle: f32) -> Q {
    let a = norm(axis);
    if a == [0.0; 3] {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let (s, c) = (angle * 0.5).sin_cos();
    [a[0] * s, a[1] * s, a[2] * s, c]
}

// ---- forward kinematics ----------------------------------------------------------

/// Parent of every joint (`usize::MAX` for a root) and a parents-first visit
/// order, both derived from the children lists.
pub struct Topology {
    pub parent: Vec<usize>,
    pub order: Vec<usize>,
}

pub fn topology(children: &[&[usize]], roots: &[usize]) -> Topology {
    let n = children.len();
    let mut parent = vec![usize::MAX; n];
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<usize> = roots.iter().rev().copied().filter(|&r| r < n).collect();
    let mut seen = vec![false; n];
    while let Some(j) = stack.pop() {
        if seen[j] {
            continue;
        }
        seen[j] = true;
        order.push(j);
        for &c in children[j].iter().rev() {
            if c < n && !seen[c] {
                parent[c] = j;
                stack.push(c);
            }
        }
    }
    Topology { parent, order }
}

/// Model-space position and rotation of every joint. Scale is carried into
/// the positions (uniform scale assumed for the rotations, which is what a
/// character skeleton has).
fn fk(topo: &Topology, t: &[V3], r: &[Q], s: &[V3], pos: &mut [V3], rot: &mut [Q], scl: &mut [V3]) {
    for &j in &topo.order {
        let p = topo.parent[j];
        if p == usize::MAX {
            pos[j] = t[j];
            rot[j] = qnorm(r[j]);
            scl[j] = s[j];
        } else {
            let lt = [
                t[j][0] * scl[p][0],
                t[j][1] * scl[p][1],
                t[j][2] * scl[p][2],
            ];
            pos[j] = add(pos[p], qrot(rot[p], lt));
            rot[j] = qnorm(qmul(rot[p], r[j]));
            scl[j] = [
                scl[p][0] * s[j][0],
                scl[p][1] * s[j][1],
                scl[p][2] * s[j][2],
            ];
        }
    }
}

/// Turn joint `j` (and so its whole subtree) by the model-space rotation
/// `delta`, by rewriting its LOCAL rotation.
fn rotate_model_space(topo: &Topology, r: &mut [Q], rot: &[Q], j: usize, delta: Q) {
    let p = topo.parent[j];
    r[j] = if p == usize::MAX {
        qnorm(qmul(delta, r[j]))
    } else {
        // local' = P^-1 * delta * P * local
        qnorm(qmul(qmul(qconj(rot[p]), qmul(delta, rot[p])), r[j]))
    };
}

/// Apply the joint rotations and the IK chains to a local pose, in that order
/// (an arm is solved against the torso the aim offset has already turned).
pub fn apply_post(rig: &AnimRig, topo: &Topology, t: &[V3], r: &mut [Q], s: &[V3]) {
    let n = r.len().min(t.len()).min(s.len()).min(topo.parent.len());
    if n == 0 {
        return;
    }
    let mut pos = vec![[0.0f32; 3]; n];
    let mut rot = vec![[0.0f32, 0.0, 0.0, 1.0]; n];
    let mut scl = vec![[1.0f32; 3]; n];
    fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);

    for slot in &rig.rots {
        if slot.joint < 0 || (slot.joint as usize) >= n || slot.angle == 0.0 {
            continue;
        }
        let j = slot.joint as usize;
        rotate_model_space(topo, r, &rot, j, axis_angle(slot.axis, slot.angle));
        fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);
    }

    for ik in &rig.iks {
        if ik.end < 0 || (ik.end as usize) >= n || ik.weight <= 0.0 {
            continue;
        }
        let c_j = ik.end as usize;
        let b_j = topo.parent[c_j];
        if b_j == usize::MAX {
            continue;
        }
        let a_j = topo.parent[b_j];
        if a_j == usize::MAX {
            continue;
        }
        let end_rot_before = rot[c_j];
        let (a, b, c) = (pos[a_j], pos[b_j], pos[c_j]);
        let w = ik.weight.min(1.0);
        let target = add(c, mul(sub(ik.target, c), w));

        let lab = len(sub(b, a));
        let lcb = len(sub(c, b));
        if lab < 1e-5 || lcb < 1e-5 {
            continue;
        }
        let lat = len(sub(target, a)).clamp(0.01, lab + lcb - 0.001);
        let acos = |x: f32| x.clamp(-1.0, 1.0).acos();
        let ac_ab_0 = acos(dot(norm(sub(c, a)), norm(sub(b, a))));
        let ba_bc_0 = acos(dot(norm(sub(a, b)), norm(sub(c, b))));
        let ac_at_0 = acos(dot(norm(sub(c, a)), norm(sub(target, a))));
        let ac_ab_1 = acos((lcb * lcb - lab * lab - lat * lat) / (-2.0 * lab * lat));
        let ba_bc_1 = acos((lat * lat - lab * lab - lcb * lcb) / (-2.0 * lab * lcb));

        // Bend axis: the chain's own plane. A dead-straight arm has none, so
        // fall back to the pole, then to anything perpendicular to the arm.
        let mut axis0 = norm(cross(sub(c, a), sub(b, a)));
        if axis0 == [0.0; 3] && ik.pole_on {
            axis0 = norm(cross(sub(c, a), sub(ik.pole, a)));
        }
        if axis0 == [0.0; 3] {
            let d = norm(sub(c, a));
            let helper = if d[1].abs() < 0.9 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            axis0 = norm(cross(d, helper));
        }
        let axis1 = norm(cross(sub(c, a), sub(target, a)));

        // Lengthen / shorten the chain in its plane (root and middle), then
        // swing the root so the end lands on the target.
        let bend_root = axis_angle(axis0, ac_ab_1 - ac_ab_0);
        let bend_mid = axis_angle(axis0, ba_bc_1 - ba_bc_0);
        rotate_model_space(topo, r, &rot, a_j, bend_root);
        fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);
        rotate_model_space(
            topo,
            r,
            &rot,
            b_j,
            qmul(bend_root, qmul(bend_mid, qconj(bend_root))),
        );
        fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);
        if axis1 != [0.0; 3] {
            rotate_model_space(topo, r, &rot, a_j, axis_angle(axis1, ac_at_0));
            fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);
        }

        // Pole: spin the chain about the root->target line until the middle
        // joint is on the pole's side.
        if ik.pole_on {
            let a2 = pos[a_j];
            let line = norm(sub(pos[c_j], a2));
            if line != [0.0; 3] {
                let u0 = sub(pos[b_j], a2);
                let v0 = sub(ik.pole, a2);
                let u = sub(u0, mul(line, dot(u0, line)));
                let v = sub(v0, mul(line, dot(v0, line)));
                if len(u) > 1e-4 && len(v) > 1e-4 {
                    let (un, vn) = (norm(u), norm(v));
                    let ang = dot(cross(un, vn), line).atan2(dot(un, vn)) * w;
                    rotate_model_space(topo, r, &rot, a_j, axis_angle(line, ang));
                    fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);
                }
            }
        }

        if ik.keep_end_rotation {
            r[c_j] = qnorm(qmul(qconj(rot[b_j]), end_rot_before));
            fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);
        }
    }
}

// ---- setters (the state lives in each instance's mixer) -------------------------

impl crate::models::ModelManager {
    pub fn anim_set_blend(&mut self, handle: f64, slot: usize, clip: i32, weight: f32) {
        if let Some(ma) = self.animations.get_mut(handle) {
            if slot < BLEND_SLOTS {
                let ok = clip >= 0 && (clip as usize) < ma.animations.len() && weight > 0.0;
                ma.mixer.rig.blends[slot] = BlendSlot {
                    clip: if ok { clip } else { -1 },
                    weight: if ok { weight.min(1.0) } else { 0.0 },
                };
            }
        }
    }

    pub fn anim_set_joint_rotation(
        &mut self,
        handle: f64,
        slot: usize,
        joint: i32,
        axis: [f32; 3],
        angle: f32,
    ) {
        if let Some(ma) = self.animations.get_mut(handle) {
            if slot < ROT_SLOTS {
                ma.mixer.rig.rots[slot] = RotSlot { joint, axis, angle };
            }
        }
    }

    pub fn anim_set_ik(
        &mut self,
        handle: f64,
        slot: usize,
        end: i32,
        target: [f32; 3],
        weight: f32,
        keep_end_rotation: bool,
    ) {
        if let Some(ma) = self.animations.get_mut(handle) {
            if slot < IK_SLOTS {
                let k = &mut ma.mixer.rig.iks[slot];
                k.end = end;
                k.target = target;
                k.weight = weight.clamp(0.0, 1.0);
                k.keep_end_rotation = keep_end_rotation;
            }
        }
    }

    pub fn anim_set_ik_pole(&mut self, handle: f64, slot: usize, pole: [f32; 3], on: bool) {
        if let Some(ma) = self.animations.get_mut(handle) {
            if slot < IK_SLOTS {
                ma.mixer.rig.iks[slot].pole = pole;
                ma.mixer.rig.iks[slot].pole_on = on;
            }
        }
    }

    /// Mixer clocks, in seconds: 0 base clip time, 1 layer clip time,
    /// 2 base clip time as a 0..1 phase.
    pub fn anim_time(&self, handle: f64, which: u32) -> f32 {
        let Some(ma) = self.animations.get(handle) else {
            return 0.0;
        };
        match which {
            0 => ma.mixer.cur_time,
            1 => ma.mixer.layer_time,
            2 => {
                let d = ma
                    .animations
                    .get(ma.mixer.cur_clip)
                    .map(|a| a.duration)
                    .unwrap_or(0.0);
                if d > 0.0 {
                    ma.mixer.cur_time / d
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: Q = [0.0, 0.0, 0.0, 1.0];

    /// shoulder at the origin, elbow 0.3 along +X, hand 0.25 further, with a
    /// slight pre-bend so the chain has a plane.
    fn arm() -> (Topology, Vec<V3>, Vec<Q>, Vec<V3>) {
        let ch: Vec<Vec<usize>> = vec![vec![1], vec![2], vec![3], vec![]];
        let refs: Vec<&[usize]> = ch.iter().map(|c| c.as_slice()).collect();
        let topo = topology(&refs, &[0]);
        let t = vec![
            [0.0, 1.4, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ];
        let r = vec![ID, ID, axis_angle([0.0, 1.0, 0.0], 0.3), ID];
        let s = vec![[1.0; 3]; 4];
        (topo, t, r, s)
    }

    fn end_of(topo: &Topology, t: &[V3], r: &[Q], s: &[V3], j: usize) -> (V3, Q) {
        let n = t.len();
        let mut pos = vec![[0.0; 3]; n];
        let mut rot = vec![ID; n];
        let mut scl = vec![[1.0; 3]; n];
        fk(topo, t, r, s, &mut pos, &mut rot, &mut scl);
        (pos[j], rot[j])
    }

    #[test]
    fn topology_is_parents_first() {
        let (topo, ..) = arm();
        assert_eq!(topo.parent, vec![usize::MAX, 0, 1, 2]);
        assert_eq!(topo.order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn ik_reaches_targets_in_range() {
        for target in [
            [0.5, 1.5, 0.2],
            [0.3, 1.0, -0.25],
            [0.45, 1.75, 0.1],
            [0.2, 1.4, 0.4],
        ] {
            let (topo, t, mut r, s) = arm();
            let mut rig = AnimRig::default();
            rig.iks[0] = IkSlot {
                end: 3,
                target,
                weight: 1.0,
                keep_end_rotation: false,
                pole: [0.0; 3],
                pole_on: false,
            };
            apply_post(&rig, &topo, &t, &mut r, &s);
            let (p, _) = end_of(&topo, &t, &r, &s, 3);
            assert!(
                len(sub(p, target)) < 2e-3,
                "target {:?} got {:?}",
                target,
                p
            );
            // bone lengths are untouched: the solve only rotates
            let (e, _) = end_of(&topo, &t, &r, &s, 2);
            let (sh, _) = end_of(&topo, &t, &r, &s, 1);
            assert!((len(sub(e, sh)) - 0.3).abs() < 1e-4);
        }
    }

    #[test]
    fn ik_out_of_reach_points_at_target() {
        let (topo, t, mut r, s) = arm();
        let target = [3.0, 1.4, 0.0];
        let mut rig = AnimRig::default();
        rig.iks[0] = IkSlot {
            end: 3,
            target,
            weight: 1.0,
            keep_end_rotation: false,
            pole: [0.0; 3],
            pole_on: false,
        };
        apply_post(&rig, &topo, &t, &mut r, &s);
        let (p, _) = end_of(&topo, &t, &r, &s, 3);
        let (sh, _) = end_of(&topo, &t, &r, &s, 1);
        assert!((len(sub(p, sh)) - 0.549).abs() < 2e-3);
        assert!(dot(norm(sub(p, sh)), norm(sub(target, sh))) > 0.9999);
    }

    #[test]
    fn pole_picks_the_elbow_side_and_keeps_the_hand() {
        let target = [0.45, 1.4, 0.0];
        for pole_y in [3.0f32, -3.0] {
            let (topo, t, mut r, s) = arm();
            let mut rig = AnimRig::default();
            rig.iks[0] = IkSlot {
                end: 3,
                target,
                weight: 1.0,
                keep_end_rotation: false,
                pole: [0.3, 1.4 + pole_y, 0.0],
                pole_on: true,
            };
            apply_post(&rig, &topo, &t, &mut r, &s);
            let (p, _) = end_of(&topo, &t, &r, &s, 3);
            assert!(len(sub(p, target)) < 2e-3);
            let (e, _) = end_of(&topo, &t, &r, &s, 2);
            assert!(
                (e[1] - 1.4) * pole_y > 0.0,
                "elbow {:?} for pole_y {}",
                e,
                pole_y
            );
        }
    }

    #[test]
    fn keep_end_rotation_restores_the_hand() {
        let (topo, t, mut r, s) = arm();
        let (_, before) = end_of(&topo, &t, &r, &s, 3);
        let mut rig = AnimRig::default();
        rig.iks[0] = IkSlot {
            end: 3,
            target: [0.3, 1.1, 0.3],
            weight: 1.0,
            keep_end_rotation: true,
            pole: [0.0; 3],
            pole_on: false,
        };
        apply_post(&rig, &topo, &t, &mut r, &s);
        let (_, after) = end_of(&topo, &t, &r, &s, 3);
        let d = (before[0] * after[0]
            + before[1] * after[1]
            + before[2] * after[2]
            + before[3] * after[3])
            .abs();
        assert!(d > 0.9999);
    }

    #[test]
    fn joint_rotation_turns_the_subtree_about_a_model_axis() {
        let (topo, t, mut r, s) = arm();
        let mut rig = AnimRig::default();
        // quarter turn about model +Y at the shoulder joint (index 1)
        rig.rots[0] = RotSlot {
            joint: 1,
            axis: [0.0, 1.0, 0.0],
            angle: std::f32::consts::FRAC_PI_2,
        };
        let (before, _) = end_of(&topo, &t, &r, &s, 2);
        apply_post(&rig, &topo, &t, &mut r, &s);
        let (after, _) = end_of(&topo, &t, &r, &s, 2);
        // elbow was 0.3 along +X from the shoulder; +90 deg about Y sends +X to -Z
        assert!((before[0] - 0.5).abs() < 1e-5);
        assert!(
            (after[0] - 0.2).abs() < 1e-4 && (after[2] + 0.3).abs() < 1e-4,
            "{:?}",
            after
        );
    }

    #[test]
    fn zero_weight_and_empty_rig_change_nothing() {
        let (topo, t, mut r, s) = arm();
        let orig = r.clone();
        let mut rig = AnimRig::default();
        assert!(!rig.has_post());
        rig.iks[0].end = 3;
        rig.iks[0].target = [9.0, 9.0, 9.0];
        apply_post(&rig, &topo, &t, &mut r, &s);
        assert_eq!(r, orig);
    }
}
