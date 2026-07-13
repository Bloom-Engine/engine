//! EN-025 — ragdolls.
//!
//! Jolt has shipped `Ragdoll.cpp` in the vendored submodule since day one; what
//! was missing was any way to reach it. This is that. It does NOT use Jolt's
//! `Ragdoll` class, though — that wants a `RagdollSettings` asset authored
//! alongside the skeleton, and we don't have one. Instead it builds the ragdoll
//! from the skeleton we DO have, at runtime: one capsule per bone, one limited
//! six-DOF joint per articulation. Same result, no new authoring step, and it
//! works for any skinned model the game already loads.
//!
//! # The coordinate problem, and how it is solved
//!
//! Physics runs in world space. Skinning runs in model space. The bridge is the
//! model transform `M` (scale, position, yaw) the game was drawing with at the
//! moment of death — and the key decision here is that **`M` is frozen at
//! activation**. The corpse does not follow the enemy's old position, because
//! the enemy is gone; the bodies carry the motion now, and `M` is only a fixed
//! frame to express them in.
//!
//! So for each bodied joint we store, at activation, the joint's transform in
//! the *body's* frame:
//!
//! ```text
//!   offset_j = (M⁻¹ · B_j(0))⁻¹ · jointWorld_j(0)
//! ```
//!
//! and every frame after, recover the pose by pushing the simulated body back
//! through it:
//!
//! ```text
//!   jointWorld_j(t) = (M⁻¹ · B_j(t)) · offset_j
//! ```
//!
//! Joints with no body of their own (fingers, tails, the long chains that would
//! cost bodies and buy nothing) simply ride their nearest bodied ancestor at
//! their rest pose. That is what `maxBodies` is really buying: a coarse
//! skeleton that behaves, instead of a fine one that jitters.

use crate::models::{ModelAnimation, SkeletonData};

/// One simulated bone.
struct Bone {
    /// Joint this body drives.
    joint: usize,
    /// Physics body handle (as the f64 the physics registry uses).
    body: f64,
    /// Joint transform expressed in the body's frame — see the module note.
    offset: [[f32; 4]; 4],
}

pub struct Ragdoll {
    pub active: bool,
    bones: Vec<Bone>,
    constraints: Vec<f64>,
    /// Model transform frozen at activation, and its inverse.
    model: [[f32; 4]; 4],
    inv_model: [[f32; 4]; 4],
    /// The (scale, pos, rot) that `model` was built from — the joint upload
    /// needs them again in exactly that form.
    scale: f32,
    pos: [f32; 3],
    rot: f32,
    /// Seconds since activation, so the game can settle/despawn on a timer.
    pub age: f32,
}

impl Ragdoll {
    pub fn new() -> Self {
        Self {
            active: false,
            bones: Vec::new(),
            constraints: Vec::new(),
            model: mat4_identity(),
            inv_model: mat4_identity(),
            scale: 1.0,
            pos: [0.0; 3],
            rot: 0.0,
            age: 0.0,
        }
    }
}

/// Which joints get a body.
///
/// Breadth-first from the roots, taking any joint whose bone (parent → self) is
/// long enough to be worth simulating, until `max_bodies` is reached. BFS is the
/// point: it spends the budget on the spine and the limbs — the things whose
/// motion you actually read — and runs out before it reaches the fingers.
fn select_bones(skel: &SkeletonData, min_len: f32, max_bodies: usize,
                joint_world: &[[[f32; 4]; 4]]) -> Vec<usize> {
    let n = skel.joints.len();
    let mut parent = vec![usize::MAX; n];
    for (i, j) in skel.joints.iter().enumerate() {
        for &c in &j.children {
            if c < n { parent[c] = i; }
        }
    }

    let mut order: Vec<usize> = Vec::new();
    let mut queue: std::collections::VecDeque<usize> = skel.root_joints.iter().copied().collect();
    while let Some(j) = queue.pop_front() {
        order.push(j);
        for &c in &skel.joints[j].children {
            if c < n { queue.push_back(c); }
        }
    }

    let mut picked = Vec::new();
    for &j in &order {
        if picked.len() >= max_bodies { break; }
        let p = parent[j];
        if p == usize::MAX { continue; }              // roots carry no bone
        let a = translation(&joint_world[p]);
        let b = translation(&joint_world[j]);
        let len = dist(a, b);
        if len >= min_len {
            picked.push(j);
        }
    }
    picked
}

pub struct RagdollBuild {
    pub joint: usize,
    /// Capsule half-height and radius, in WORLD units (model scale applied).
    pub half_height: f32,
    pub radius: f32,
    /// World transform of the capsule at activation (column-major 4x4).
    pub world: [[f32; 4]; 4],
    /// Parent bone index within the build list, or usize::MAX for none.
    pub parent_bone: usize,
    /// World-space anchor for the joint to the parent bone.
    pub anchor: [f32; 3],
}

/// Compute everything the physics side needs to create the bodies. Kept separate
/// from body creation so this module never has to know about JoltPhysics — the
/// FFI layer owns that, and this stays testable.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    anim: &ModelAnimation,
    scale: f32, pos: [f32; 3], rot: f32,
    max_bodies: usize,
    radius_scale: f32,
) -> Vec<RagdollBuild> {
    let Some(skel) = &anim.skeleton else { return Vec::new() };
    if anim.joint_world.len() != skel.joints.len() { return Vec::new() }

    let model = compose(scale, pos, rot);
    let picked = select_bones(skel, 0.06, max_bodies, &anim.joint_world);

    // joint -> index in `picked`, so a bone can find its parent bone.
    let mut bone_of = vec![usize::MAX; skel.joints.len()];
    for (bi, &j) in picked.iter().enumerate() { bone_of[j] = bi; }

    let n = skel.joints.len();
    let mut parent = vec![usize::MAX; n];
    for (i, j) in skel.joints.iter().enumerate() {
        for &c in &j.children { if c < n { parent[c] = i; } }
    }

    let mut out = Vec::with_capacity(picked.len());
    for &j in &picked {
        let p = parent[j];
        let a_local = translation(&anim.joint_world[p]);
        let b_local = translation(&anim.joint_world[j]);

        // World-space endpoints of the bone.
        let a = xform_point(&model, a_local);
        let b = xform_point(&model, b_local);
        let len = dist(a, b);
        if len < 1e-4 { continue; }

        // Capsule spans the bone: centre at the midpoint, +Y down its length
        // (Jolt capsules are Y-aligned), radius a fraction of the length.
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
        let dir = normalize([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
        let radius = (len * radius_scale).clamp(0.02, len * 0.45);
        let half_height = (len * 0.5 - radius).max(0.01);

        // Rotation taking +Y onto the bone direction.
        let world = compose_from_dir(mid, dir);

        // Find the nearest ANCESTOR that has a body — not necessarily the
        // immediate parent joint, since that one may not have been picked.
        let mut anc = p;
        while anc != usize::MAX && bone_of[anc] == usize::MAX {
            anc = parent[anc];
        }
        let parent_bone = if anc == usize::MAX { usize::MAX } else { bone_of[anc] };

        out.push(RagdollBuild {
            joint: j,
            half_height,
            radius,
            world,
            parent_bone,
            // Joint sits at the bone's proximal end — that IS the articulation.
            anchor: a,
        });
    }
    out
}

impl Ragdoll {
    /// Called by the FFI once the bodies exist. `bodies[i]` pairs with
    /// `builds[i]`.
    pub fn attach(&mut self, builds: &[RagdollBuild], bodies: &[f64],
                  constraints: Vec<f64>,
                  anim: &ModelAnimation,
                  scale: f32, pos: [f32; 3], rot: f32) {
        self.model = compose(scale, pos, rot);
        self.inv_model = invert_rigid(&self.model);
        self.scale = scale;
        self.pos = pos;
        self.rot = rot;
        self.constraints = constraints;
        self.bones.clear();
        self.age = 0.0;

        for (i, b) in builds.iter().enumerate() {
            if i >= bodies.len() || bodies[i] == 0.0 { continue; }
            // offset = (M⁻¹ · B(0))⁻¹ · jointWorld(0)
            let body_model = mat4_mul(&self.inv_model, &b.world);
            let offset = mat4_mul(&invert_rigid(&body_model), &anim.joint_world[b.joint]);
            self.bones.push(Bone { joint: b.joint, body: bodies[i], offset });
        }
        self.active = true;
    }

    pub fn bodies(&self) -> Vec<f64> { self.bones.iter().map(|b| b.body).collect() }
    pub fn constraint_handles(&self) -> &[f64] { &self.constraints }

    pub fn upload_params(&self) -> (f32, [f32; 3], f32) { (self.scale, self.pos, self.rot) }

    /// Rebuild `anim.joint_matrices` from the simulated bodies.
    /// `body_world[i]` is the world transform of `self.bones[i]`'s body.
    pub fn apply(&self, anim: &mut ModelAnimation, body_world: &[[[f32; 4]; 4]]) {
        let Some(skel) = &anim.skeleton else { return };
        let n = skel.joints.len();
        if anim.joint_world.len() != n { return }

        // 1. Bodied joints: push the body back through the stored offset.
        let mut have = vec![false; n];
        let mut world = vec![mat4_identity(); n];
        for (i, bone) in self.bones.iter().enumerate() {
            if i >= body_world.len() { break; }
            let body_model = mat4_mul(&self.inv_model, &body_world[i]);
            world[bone.joint] = mat4_mul(&body_model, &bone.offset);
            have[bone.joint] = true;
        }

        // 2. Everything else rides its parent at the rest pose. Walk from the
        //    roots so a parent is always resolved before its children.
        let mut stack: Vec<(usize, [[f32; 4]; 4])> =
            skel.root_joints.iter().map(|&r| (r, mat4_identity())).collect();
        while let Some((j, parent_world)) = stack.pop() {
            if j >= n { continue; }
            if !have[j] {
                let local = mat4_from_trs(
                    &skel.joints[j].rest_translation,
                    &skel.joints[j].rest_rotation,
                    &skel.joints[j].rest_scale,
                );
                world[j] = mat4_mul(&parent_world, &local);
            }
            for &c in &skel.joints[j].children {
                stack.push((c, world[j]));
            }
        }

        for i in 0..n {
            anim.joint_matrices[i] = mat4_mul(&world[i], &skel.joints[i].inverse_bind);
        }
        anim.joint_world.copy_from_slice(&world);
    }
}

// ---- registry ----------------------------------------------------------------

pub struct RagdollManager {
    pub slots: Vec<Option<Ragdoll>>,
}

impl RagdollManager {
    pub fn new() -> Self { Self { slots: Vec::new() } }
    pub fn create(&mut self) -> u32 {
        self.slots.push(Some(Ragdoll::new()));
        self.slots.len() as u32
    }
    pub fn get_mut(&mut self, h: u32) -> Option<&mut Ragdoll> {
        if h == 0 { return None }
        self.slots.get_mut(h as usize - 1)?.as_mut()
    }
    pub fn get(&self, h: u32) -> Option<&Ragdoll> {
        if h == 0 { return None }
        self.slots.get(h as usize - 1)?.as_ref()
    }
}

impl Default for RagdollManager {
    fn default() -> Self { Self::new() }
}

// ---- math --------------------------------------------------------------------

fn mat4_identity() -> [[f32; 4]; 4] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut o = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            o[col][row] = a[0][row] * b[col][0] + a[1][row] * b[col][1]
                        + a[2][row] * b[col][2] + a[3][row] * b[col][3];
        }
    }
    o
}

fn translation(m: &[[f32; 4]; 4]) -> [f32; 3] { [m[3][0], m[3][1], m[3][2]] }

fn xform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-6 { [v[0] / l, v[1] / l, v[2] / l] } else { [0.0, 1.0, 0.0] }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Model transform: translate · rotateY · scale.
pub fn compose(scale: f32, pos: [f32; 3], rot: f32) -> [[f32; 4]; 4] {
    let (s, c) = (rot.sin(), rot.cos());
    [
        [c * scale, 0.0, -s * scale, 0.0],
        [0.0, scale, 0.0, 0.0],
        [s * scale, 0.0, c * scale, 0.0],
        [pos[0], pos[1], pos[2], 1.0],
    ]
}

/// A rigid frame whose +Y points along `dir`, centred at `p`.
fn compose_from_dir(p: [f32; 3], dir: [f32; 3]) -> [[f32; 4]; 4] {
    let up = dir;
    // Any axis not parallel to `up` works as the seed for the basis.
    let seed = if up[1].abs() > 0.99 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let right = normalize(cross(seed, up));
    let fwd = cross(up, right);
    [
        [right[0], right[1], right[2], 0.0],
        [up[0], up[1], up[2], 0.0],
        [fwd[0], fwd[1], fwd[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}

/// Inverse of a transform with uniform scale + rotation + translation.
/// (A general inverse would be wasted here and less numerically pleasant.)
fn invert_rigid(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Recover uniform scale from the first basis vector.
    let s2 = m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2];
    let s = s2.sqrt().max(1e-8);
    let inv_s = 1.0 / s;
    // R⁻¹ = Rᵀ, with the scale divided out twice (once for the transpose's own
    // scale, once for the inverse scale).
    let r = [
        [m[0][0] * inv_s * inv_s, m[1][0] * inv_s * inv_s, m[2][0] * inv_s * inv_s],
        [m[0][1] * inv_s * inv_s, m[1][1] * inv_s * inv_s, m[2][1] * inv_s * inv_s],
        [m[0][2] * inv_s * inv_s, m[1][2] * inv_s * inv_s, m[2][2] * inv_s * inv_s],
    ];
    let t = [m[3][0], m[3][1], m[3][2]];
    let it = [
        -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
        -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
        -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
    ];
    [
        [r[0][0], r[0][1], r[0][2], 0.0],
        [r[1][0], r[1][1], r[1][2], 0.0],
        [r[2][0], r[2][1], r[2][2], 0.0],
        [it[0], it[1], it[2], 1.0],
    ]
}

fn mat4_from_trs(t: &[f32; 3], q: &[f32; 4], s: &[f32; 3]) -> [[f32; 4]; 4] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    [
        [(1.0 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0.0],
        [(xy - wz) * s[1], (1.0 - (xx + zz)) * s[1], (yz + wx) * s[1], 0.0],
        [(xz + wy) * s[2], (yz - wx) * s[2], (1.0 - (xx + yy)) * s[2], 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

/// Build a world transform from a physics body's position + quaternion.
pub fn from_pos_quat(p: [f32; 3], q: [f32; 4]) -> [[f32; 4]; 4] {
    mat4_from_trs(&p, &q, &[1.0, 1.0, 1.0])
}

/// Quaternion (x, y, z, w) from an orthonormal rotation matrix. Shepperd's
/// method — pick the largest diagonal term so the division never blows up on a
/// 180° rotation, which the naive `w = sqrt(1+trace)/2` form does.
pub fn quat_from_mat(m: &[[f32; 4]; 4]) -> [f32; 4] {
    let (m00, m01, m02) = (m[0][0], m[0][1], m[0][2]);
    let (m10, m11, m12) = (m[1][0], m[1][1], m[1][2]);
    let (m20, m21, m22) = (m[2][0], m[2][1], m[2][2]);
    let trace = m00 + m11 + m22;

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(m12 - m21) / s, (m20 - m02) / s, (m01 - m10) / s, 0.25 * s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [0.25 * s, (m10 + m01) / s, (m20 + m02) / s, (m12 - m21) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m10 + m01) / s, 0.25 * s, (m21 + m12) / s, (m20 - m02) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m20 + m02) / s, (m21 + m12) / s, 0.25 * s, (m01 - m10) / s]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_rigid_round_trips() {
        let m = compose(2.5, [1.0, -3.0, 4.0], 0.7);
        let i = invert_rigid(&m);
        let id = mat4_mul(&m, &i);
        for c in 0..4 {
            for r in 0..4 {
                let want = if c == r { 1.0 } else { 0.0 };
                assert!((id[c][r] - want).abs() < 1e-3,
                    "M·M⁻¹ is not identity at [{c}][{r}]: {}", id[c][r]);
            }
        }
    }

    #[test]
    fn compose_from_dir_points_y_along_the_bone() {
        let dir = normalize([0.3, -0.9, 0.2]);
        let m = compose_from_dir([1.0, 2.0, 3.0], dir);
        // Column 1 is the +Y basis vector.
        assert!((m[1][0] - dir[0]).abs() < 1e-5);
        assert!((m[1][1] - dir[1]).abs() < 1e-5);
        assert!((m[1][2] - dir[2]).abs() < 1e-5);
        // And the frame stays orthonormal, or the capsule shears.
        let right = [m[0][0], m[0][1], m[0][2]];
        let dot = right[0] * dir[0] + right[1] * dir[1] + right[2] * dir[2];
        assert!(dot.abs() < 1e-5, "basis is not orthogonal: {dot}");
    }
}
