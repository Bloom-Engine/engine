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
//!   offset_j = (M⁻¹ · B_j(0))⁻¹ · jointWorld_j(0)      (B_j: the body of the bone LEAVING joint j)
//! ```
//!
//! and every frame after, recover the pose by pushing the simulated body back
//! through it:
//!
//! ```text
//!   jointWorld_j(t) = (M⁻¹ · B_j(t)) · offset_j
//! ```
//!
//! Joints with no body of their own (fingers, bones too short to simulate, any
//! joint the game switched off) ride their nearest bodied ancestor in the pose
//! they had at activation.
//!
//! # Which joint a body drives
//!
//! A body spans one bone, from joint j to its main child, and it drives joint
//! j: the skin weighted to a joint lies along the bone LEAVING it. Neighbouring
//! bodies overlap at the joint they share, so the FFI layer puts a ragdoll's
//! bodies in one collision group where a bone ignores its parent and its
//! siblings; everything else still collides, which is what stops a corpse
//! folding through itself.

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
    /// Each joint's transform relative to its parent at activation. Joints
    /// without a body keep it, so a hand stays in the shape it died in
    /// instead of snapping to the bind pose.
    locals: Vec<[[f32; 4]; 4]>,
    pub config: RagdollConfig,
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
            locals: Vec::new(),
            config: RagdollConfig::default(),
        }
    }

    /// Back to an empty slot, keeping how it was configured.
    pub fn reset(&mut self) {
        let config = std::mem::take(&mut self.config);
        *self = Self::new();
        self.config = config;
    }
}

/// How a ragdoll is built. Lives on the slot and survives `release`, so a
/// pooled ragdoll is configured once for its rig.
#[derive(Clone)]
pub struct RagdollConfig {
    /// Upper bound on simulated bones. Past it the shortest bones are dropped
    /// (their joints ride the nearest simulated ancestor).
    pub max_bodies: usize,
    /// Limb capsule radius as a fraction of the bone's length.
    pub radius_scale: f32,
    /// Half angle of the swing cone at every joint, radians.
    pub swing: f32,
    /// Twist allowed either way about the bone, radians.
    pub twist: f32,
    /// Bones shorter than this (world units) get no body.
    pub min_len: f32,
    /// Per-joint overrides: `(joint, simulate, swing, twist)`. `simulate ==
    /// false` keeps the joint out of the simulation; negative angles mean
    /// "use the defaults above".
    pub joints: Vec<(usize, bool, f32, f32)>,
}

impl Default for RagdollConfig {
    fn default() -> Self {
        Self {
            max_bodies: 24,
            radius_scale: 0.22,
            swing: 0.7,
            twist: 0.3,
            min_len: 0.05,
            joints: Vec::new(),
        }
    }
}

impl RagdollConfig {
    fn joint_override(&self, joint: usize) -> Option<(bool, f32, f32)> {
        self.joints
            .iter()
            .rev()
            .find(|o| o.0 == joint)
            .map(|o| (o.1, o.2, o.3))
    }
}

pub struct RagdollBuild {
    /// The joint this body drives: the one at the bone's PROXIMAL end. The
    /// skin weighted to a joint lies along the bone leaving it, so that is the
    /// body it has to follow. (Until 2026-11 the body drove the joint at the
    /// far end, which put every stretch of skin on the bone before its own and
    /// turned a corpse into a plank.)
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
    /// World-space unit direction of the bone (the joint's twist axis).
    pub dir: [f32; 3],
    /// Swing half-cone and twist range for the joint to the parent, radians.
    pub swing: f32,
    pub twist: f32,
}

fn parents_of(skel: &SkeletonData) -> Vec<usize> {
    let n = skel.joints.len();
    let mut parent = vec![usize::MAX; n];
    for (i, j) in skel.joints.iter().enumerate() {
        for &c in &j.children {
            if c < n {
                parent[c] = i;
            }
        }
    }
    parent
}

/// Half the widest gap between the children of the first branching joint at or
/// below `j` on its main chain: the hips' leg sockets, the chest's shoulders, a
/// hand's knuckles. It is the only measure of a body part's girth a skeleton
/// carries, and it keeps the trunk from being simulated as a thin rod.
fn fan_radius(skel: &SkeletonData, joint_world: &[[[f32; 4]; 4]], j: usize, next: &[usize]) -> f32 {
    let mut at = j;
    for _ in 0..3 {
        let kids = &skel.joints[at].children;
        if kids.len() >= 2 {
            let mut widest = 0.0f32;
            for (i, &a) in kids.iter().enumerate() {
                for &b in kids.iter().skip(i + 1) {
                    if a < joint_world.len() && b < joint_world.len() {
                        widest = widest.max(dist(
                            translation(&joint_world[a]),
                            translation(&joint_world[b]),
                        ));
                    }
                }
            }
            return widest * 0.5;
        }
        if next[at] == usize::MAX {
            break;
        }
        at = next[at];
    }
    0.0
}

/// Compute everything the physics side needs to create the bodies. Kept separate
/// from body creation so this module never has to know about JoltPhysics — the
/// FFI layer owns that, and this stays testable.
///
/// One body per joint that has a parent: a capsule from the joint to its main
/// child (the child that continues the incoming bone most directly), or, for a
/// leaf such as a head or a hand, a stub continuing the incoming bone. The
/// skeleton's first joint(s), which have no parent, get no body: on a Mixamo
/// rig that is the static `Root` at the feet.
pub fn plan(
    anim: &ModelAnimation,
    scale: f32,
    pos: [f32; 3],
    rot: f32,
    cfg: &RagdollConfig,
) -> Vec<RagdollBuild> {
    let Some(skel) = &anim.skeleton else {
        return Vec::new();
    };
    plan_skeleton(skel, &anim.joint_world, scale, pos, rot, cfg)
}

/// [`plan`] on a bare skeleton and its model-space joint transforms.
pub fn plan_skeleton(
    skel: &SkeletonData,
    jw: &[[[f32; 4]; 4]],
    scale: f32,
    pos: [f32; 3],
    rot: f32,
    cfg: &RagdollConfig,
) -> Vec<RagdollBuild> {
    let n = skel.joints.len();
    if jw.len() != n {
        return Vec::new();
    }
    let model = compose(scale, pos, rot);
    let parent = parents_of(skel);

    // Bone of each joint in model space: start, end, and the joint it runs to.
    let mut seg_end = vec![[0.0f32; 3]; n];
    let mut next = vec![usize::MAX; n];
    let mut has_seg = vec![false; n];
    for j in 0..n {
        let p = parent[j];
        if p == usize::MAX {
            continue;
        }
        let a = translation(&jw[j]);
        let pa = translation(&jw[p]);
        let in_len = dist(pa, a);
        let incoming = if in_len > 1e-5 {
            normalize([a[0] - pa[0], a[1] - pa[1], a[2] - pa[2]])
        } else {
            [0.0, 1.0, 0.0]
        };
        let mut best = usize::MAX;
        let mut best_dot = -2.0f32;
        for &c in &skel.joints[j].children {
            if c >= n {
                continue;
            }
            let cp = translation(&jw[c]);
            if dist(a, cp) < 1e-5 {
                continue;
            }
            let d = normalize([cp[0] - a[0], cp[1] - a[1], cp[2] - a[2]]);
            let dot = d[0] * incoming[0] + d[1] * incoming[1] + d[2] * incoming[2];
            if dot > best_dot {
                best_dot = dot;
                best = c;
            }
        }
        if best != usize::MAX {
            seg_end[j] = translation(&jw[best]);
            next[j] = best;
            has_seg[j] = true;
        } else if in_len > 1e-5 {
            let stub = in_len * 0.6;
            seg_end[j] = [
                a[0] + incoming[0] * stub,
                a[1] + incoming[1] * stub,
                a[2] + incoming[2] * stub,
            ];
            has_seg[j] = true;
        }
    }

    // Candidates, longest first when the budget is short.
    let mut picked: Vec<(usize, f32)> = Vec::new();
    for j in 0..n {
        if !has_seg[j] {
            continue;
        }
        if let Some((false, _, _)) = cfg.joint_override(j) {
            continue;
        }
        let len = dist(translation(&jw[j]), seg_end[j]) * scale;
        if len < cfg.min_len {
            continue;
        }
        picked.push((j, len));
    }
    if picked.len() > cfg.max_bodies {
        picked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        picked.truncate(cfg.max_bodies);
        picked.sort_by_key(|p| p.0);
    }

    let mut bone_of = vec![usize::MAX; n];
    for (bi, &(j, _)) in picked.iter().enumerate() {
        bone_of[j] = bi;
    }

    let mut out = Vec::with_capacity(picked.len());
    for &(j, len) in &picked {
        let a = xform_point(&model, translation(&jw[j]));
        let b = xform_point(&model, seg_end[j]);
        let mid = [
            (a[0] + b[0]) * 0.5,
            (a[1] + b[1]) * 0.5,
            (a[2] + b[2]) * 0.5,
        ];
        let dir = normalize([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
        // Limbs are a fraction of their length thick; a trunk bone is as
        // thick as the sockets that hang off it are wide.
        let girth = fan_radius(skel, jw, j, &next) * scale;
        let radius = (len * cfg.radius_scale).max(girth).max(0.02);
        let half_height = (len * 0.5 - radius).max(0.01);

        let mut anc = parent[j];
        while anc != usize::MAX && bone_of[anc] == usize::MAX {
            anc = parent[anc];
        }
        let parent_bone = if anc == usize::MAX {
            usize::MAX
        } else {
            bone_of[anc]
        };
        let (swing, twist) = match cfg.joint_override(j) {
            Some((_, s, t)) => (
                if s >= 0.0 { s } else { cfg.swing },
                if t >= 0.0 { t } else { cfg.twist },
            ),
            None => (cfg.swing, cfg.twist),
        };

        out.push(RagdollBuild {
            joint: j,
            half_height,
            radius,
            world: compose_from_dir(mid, dir),
            parent_bone,
            anchor: a,
            dir,
            swing,
            twist,
        });
    }
    out
}

impl Ragdoll {
    /// Called by the FFI once the bodies exist. `bodies[i]` pairs with
    /// `builds[i]`.
    pub fn attach(
        &mut self,
        builds: &[RagdollBuild],
        bodies: &[f64],
        constraints: Vec<f64>,
        anim: &ModelAnimation,
        scale: f32,
        pos: [f32; 3],
        rot: f32,
    ) {
        self.model = compose(scale, pos, rot);
        self.inv_model = invert_rigid(&self.model);
        self.scale = scale;
        self.pos = pos;
        self.rot = rot;
        self.constraints = constraints;
        self.bones.clear();
        self.age = 0.0;
        self.locals.clear();
        if let Some(skel) = &anim.skeleton {
            let n = skel.joints.len();
            if anim.joint_world.len() == n {
                let parent = parents_of(skel);
                self.locals = (0..n)
                    .map(|j| {
                        if parent[j] == usize::MAX {
                            anim.joint_world[j]
                        } else {
                            mat4_mul(
                                &invert_rigid(&anim.joint_world[parent[j]]),
                                &anim.joint_world[j],
                            )
                        }
                    })
                    .collect();
            }
        }

        for (i, b) in builds.iter().enumerate() {
            if i >= bodies.len() || bodies[i] == 0.0 {
                continue;
            }
            // offset = (M⁻¹ · B(0))⁻¹ · jointWorld(0)
            let body_model = mat4_mul(&self.inv_model, &b.world);
            let offset = mat4_mul(&invert_rigid(&body_model), &anim.joint_world[b.joint]);
            self.bones.push(Bone {
                joint: b.joint,
                body: bodies[i],
                offset,
            });
        }
        self.active = true;
    }

    pub fn bodies(&self) -> Vec<f64> {
        self.bones.iter().map(|b| b.body).collect()
    }
    pub fn constraint_handles(&self) -> &[f64] {
        &self.constraints
    }

    pub fn upload_params(&self) -> (f32, [f32; 3], f32) {
        (self.scale, self.pos, self.rot)
    }

    /// Rebuild `anim.joint_matrices` from the simulated bodies.
    /// `body_world[i]` is the world transform of `self.bones[i]`'s body.
    pub fn apply(&self, anim: &mut ModelAnimation, body_world: &[[[f32; 4]; 4]]) {
        let Some(skel) = &anim.skeleton else { return };
        let n = skel.joints.len();
        if anim.joint_world.len() != n {
            return;
        }

        // 1. Bodied joints: push the body back through the stored offset.
        let mut have = vec![false; n];
        let mut world = vec![mat4_identity(); n];
        for (i, bone) in self.bones.iter().enumerate() {
            if i >= body_world.len() {
                break;
            }
            let body_model = mat4_mul(&self.inv_model, &body_world[i]);
            world[bone.joint] = mat4_mul(&body_model, &bone.offset);
            have[bone.joint] = true;
        }

        // 2. Everything else rides its parent as it was at activation. Walk from the
        //    roots so a parent is always resolved before its children.
        let mut stack: Vec<(usize, [[f32; 4]; 4])> = skel
            .root_joints
            .iter()
            .map(|&r| (r, mat4_identity()))
            .collect();
        while let Some((j, parent_world)) = stack.pop() {
            if j >= n {
                continue;
            }
            if !have[j] {
                let local = match self.locals.get(j) {
                    Some(l) => *l,
                    None => mat4_from_trs(
                        &skel.joints[j].rest_translation,
                        &skel.joints[j].rest_rotation,
                        &skel.joints[j].rest_scale,
                    ),
                };
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
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }
    pub fn create(&mut self) -> u32 {
        self.slots.push(Some(Ragdoll::new()));
        self.slots.len() as u32
    }
    pub fn get_mut(&mut self, h: u32) -> Option<&mut Ragdoll> {
        if h == 0 {
            return None;
        }
        self.slots.get_mut(h as usize - 1)?.as_mut()
    }
    pub fn get(&self, h: u32) -> Option<&Ragdoll> {
        if h == 0 {
            return None;
        }
        self.slots.get(h as usize - 1)?.as_ref()
    }
}

impl Default for RagdollManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---- math --------------------------------------------------------------------

fn mat4_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut o = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            o[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    o
}

fn translation(m: &[[f32; 4]; 4]) -> [f32; 3] {
    [m[3][0], m[3][1], m[3][2]]
}

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
    if l > 1e-6 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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
    let seed = if up[1].abs() > 0.99 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
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
        [
            m[0][0] * inv_s * inv_s,
            m[1][0] * inv_s * inv_s,
            m[2][0] * inv_s * inv_s,
        ],
        [
            m[0][1] * inv_s * inv_s,
            m[1][1] * inv_s * inv_s,
            m[2][1] * inv_s * inv_s,
        ],
        [
            m[0][2] * inv_s * inv_s,
            m[1][2] * inv_s * inv_s,
            m[2][2] * inv_s * inv_s,
        ],
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
        [
            (1.0 - (yy + zz)) * s[0],
            (xy + wz) * s[0],
            (xz - wy) * s[0],
            0.0,
        ],
        [
            (xy - wz) * s[1],
            (1.0 - (xx + zz)) * s[1],
            (yz + wx) * s[1],
            0.0,
        ],
        [
            (xz + wy) * s[2],
            (yz - wx) * s[2],
            (1.0 - (xx + yy)) * s[2],
            0.0,
        ],
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

    use crate::models::JointData;

    /// 0 root (at the feet) - 1 hips - 2 spine - 3 chest - 4 neck - 5 head;
    /// chest - 6 shoulder - 7 arm - 8 hand; hips - 9 thigh - 10 shin - 11 foot;
    /// hips - 12 other thigh (so the hips branch three ways, like a real rig).
    fn rig() -> (SkeletonData, Vec<[[f32; 4]; 4]>) {
        let pos: [[f32; 3]; 13] = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.15, 0.0],
            [0.0, 1.4, 0.0],
            [0.0, 1.55, 0.0],
            [0.0, 1.65, 0.0],
            [0.08, 1.45, 0.0],
            [0.2, 1.45, 0.0],
            [0.75, 1.45, 0.0],
            [0.1, 0.95, 0.0],
            [0.1, 0.5, 0.0],
            [0.1, 0.08, 0.0],
            [-0.1, 0.95, 0.0],
        ];
        let kids: [&[usize]; 13] = [
            &[1],
            &[2, 9, 12],
            &[3],
            &[4, 6],
            &[5],
            &[],
            &[7],
            &[8],
            &[],
            &[10],
            &[11],
            &[],
            &[],
        ];
        let joints = (0..13)
            .map(|i| JointData {
                inverse_bind: mat4_identity(),
                children: kids[i].to_vec(),
                name: format!("j{i}"),
                rest_translation: [0.0; 3],
                rest_rotation: [0.0, 0.0, 0.0, 1.0],
                rest_scale: [1.0; 3],
            })
            .collect();
        let world = pos
            .iter()
            .map(|p| {
                let mut m = mat4_identity();
                m[3] = [p[0], p[1], p[2], 1.0];
                m
            })
            .collect();
        (
            SkeletonData {
                joints,
                root_joints: vec![0],
            },
            world,
        )
    }

    fn build_for(builds: &[RagdollBuild], joint: usize) -> Option<&RagdollBuild> {
        builds.iter().find(|b| b.joint == joint)
    }

    #[test]
    fn a_body_drives_the_joint_its_bone_leaves() {
        let (skel, world) = rig();
        let builds = plan_skeleton(&skel, &world, 1.0, [0.0; 3], 0.0, &RagdollConfig::default());
        // The static root at the feet has no parent and gets no body.
        assert!(build_for(&builds, 0).is_none());
        // The upper arm (joint 7) is a capsule from the shoulder end to the
        // hand, anchored at joint 7 itself and hung off the shoulder's body.
        let arm = build_for(&builds, 7).expect("arm has a body");
        assert!((arm.anchor[0] - 0.2).abs() < 1e-4 && (arm.anchor[1] - 1.45).abs() < 1e-4);
        assert!((arm.world[3][0] - 0.475).abs() < 1e-3, "centre is mid-bone");
        assert!((arm.dir[0] - 1.0).abs() < 1e-4);
        let parent = &builds[arm.parent_bone];
        assert_eq!(parent.joint, 6);
    }

    #[test]
    fn hips_follow_the_spine_and_leaves_get_a_stub() {
        let (skel, world) = rig();
        let builds = plan_skeleton(&skel, &world, 1.0, [0.0; 3], 0.0, &RagdollConfig::default());
        // Three children; the one continuing root -> hips (straight up) wins.
        let hips = build_for(&builds, 1).expect("hips have a body");
        assert!(hips.dir[1] > 0.99);
        assert_eq!(hips.parent_bone, usize::MAX);
        // Hips are as thick as the leg sockets are wide, not a thin rod.
        assert!(hips.radius >= 0.099, "hips radius {}", hips.radius);
        // The hand has no child: a stub continuing the forearm.
        let hand = build_for(&builds, 8).expect("hand has a body");
        assert!(hand.dir[0] > 0.99);
        // Every joint with a parent got one, and each exactly once.
        assert_eq!(builds.len(), 12);
    }

    #[test]
    fn the_budget_drops_the_shortest_bones_and_reparents() {
        let (skel, world) = rig();
        let cfg = RagdollConfig {
            max_bodies: 6,
            ..RagdollConfig::default()
        };
        let builds = plan_skeleton(&skel, &world, 1.0, [0.0; 3], 0.0, &cfg);
        assert_eq!(builds.len(), 6);
        for j in [7, 9, 10] {
            assert!(build_for(&builds, j).is_some(), "joint {j} kept");
        }
        for (i, b) in builds.iter().enumerate() {
            assert!(
                b.parent_bone == usize::MAX || (b.parent_bone < builds.len() && b.parent_bone != i)
            );
        }
    }

    #[test]
    fn a_joint_switched_off_rides_its_parent() {
        let (skel, world) = rig();
        let mut cfg = RagdollConfig::default();
        cfg.joints.push((4, false, -1.0, -1.0)); // the neck
        cfg.joints.push((10, true, 0.2, 0.05)); // a stiff knee
        let builds = plan_skeleton(&skel, &world, 1.0, [0.0; 3], 0.0, &cfg);
        assert!(build_for(&builds, 4).is_none());
        // The head hangs off the chest now.
        let head = build_for(&builds, 5).unwrap();
        assert_eq!(builds[head.parent_bone].joint, 3);
        let knee = build_for(&builds, 10).unwrap();
        assert!((knee.swing - 0.2).abs() < 1e-6 && (knee.twist - 0.05).abs() < 1e-6);
    }

    #[test]
    fn invert_rigid_round_trips() {
        let m = compose(2.5, [1.0, -3.0, 4.0], 0.7);
        let i = invert_rigid(&m);
        let id = mat4_mul(&m, &i);
        for c in 0..4 {
            for r in 0..4 {
                let want = if c == r { 1.0 } else { 0.0 };
                assert!(
                    (id[c][r] - want).abs() < 1e-3,
                    "M·M⁻¹ is not identity at [{c}][{r}]: {}",
                    id[c][r]
                );
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
