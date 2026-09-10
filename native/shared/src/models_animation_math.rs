//! Local-pose sampling, transforms, and animation interpolation.

use super::*;

/// Sample one clip into a local TRS pose, rest pose as the fallback for
/// joints the clip does not animate.
pub(super) fn sample_local_pose(
    skeleton: &SkeletonData,
    anim: &AnimationData,
    time: f32,
    strip_root: bool,
) -> LocalPose {
    let joint_count = skeleton.joints.len();
    let mut t: Vec<[f32; 3]> = skeleton.joints.iter().map(|j| j.rest_translation).collect();
    let mut r: Vec<[f32; 4]> = skeleton.joints.iter().map(|j| j.rest_rotation).collect();
    let mut s: Vec<[f32; 3]> = skeleton.joints.iter().map(|j| j.rest_scale).collect();

    let time = if anim.duration > 0.0 {
        time.rem_euclid(anim.duration)
    } else {
        0.0
    };

    for channel in &anim.channels {
        let ji = channel.joint_index;
        if ji >= joint_count {
            continue;
        }
        if !channel.translations.is_empty() && !channel.timestamps.is_empty() {
            t[ji] = sample_vec3(&channel.timestamps, &channel.translations, time);
        }
        if !channel.rotations.is_empty() {
            let ts = if !channel.rotation_timestamps.is_empty() {
                &channel.rotation_timestamps
            } else {
                &channel.timestamps
            };
            if !ts.is_empty() {
                r[ji] = sample_quat(ts, &channel.rotations, time);
            }
        }
        if !channel.scales.is_empty() {
            let ts = if !channel.scale_timestamps.is_empty() {
                &channel.scale_timestamps
            } else {
                &channel.timestamps
            };
            if !ts.is_empty() {
                s[ji] = sample_vec3(ts, &channel.scales, time);
            }
        }
    }

    if strip_root && joint_count > 0 {
        t[0] = skeleton.joints[0].rest_translation;
    }
    (t, r, s)
}

/// The root joint's authored translation at `time` — the raw channel value,
/// *not* the rest-locked one, which is the whole point of root motion.
pub(super) fn root_translation_at(
    skeleton: &SkeletonData,
    anim: &AnimationData,
    time: f32,
) -> [f32; 3] {
    if skeleton.joints.is_empty() {
        return [0.0; 3];
    }
    let time = if anim.duration > 0.0 {
        time.clamp(0.0, anim.duration)
    } else {
        0.0
    };
    for channel in &anim.channels {
        if channel.joint_index == 0
            && !channel.translations.is_empty()
            && !channel.timestamps.is_empty()
        {
            return sample_vec3(&channel.timestamps, &channel.translations, time);
        }
    }
    skeleton.joints[0].rest_translation
}

/// `dst = lerp(dst, src, w * mask[j])`. Rotations use nlerp with a
/// hemisphere fix — without the dot-sign flip, two clips whose quaternions
/// land on opposite hemispheres blend the *long* way round and the limb
/// visibly swings through the body.
pub(super) fn blend_pose(dst: &mut LocalPose, src: &LocalPose, w: f32, mask: Option<&[f32]>) {
    let n = dst.0.len().min(src.0.len());
    for j in 0..n {
        let jw = match mask {
            Some(m) => w * m.get(j).copied().unwrap_or(0.0),
            None => w,
        };
        if jw <= 0.0 {
            continue;
        }
        let jw = jw.min(1.0);
        for k in 0..3 {
            dst.0[j][k] = dst.0[j][k] + (src.0[j][k] - dst.0[j][k]) * jw;
            dst.2[j][k] = dst.2[j][k] + (src.2[j][k] - dst.2[j][k]) * jw;
        }
        let a = dst.1[j];
        let mut b = src.1[j];
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        if dot < 0.0 {
            b = [-b[0], -b[1], -b[2], -b[3]];
        }
        let mut q = [
            a[0] + (b[0] - a[0]) * jw,
            a[1] + (b[1] - a[1]) * jw,
            a[2] + (b[2] - a[2]) * jw,
            a[3] + (b[3] - a[3]) * jw,
        ];
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if len > 1e-6 {
            q = [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
        } else {
            q = a;
        }
        dst.1[j] = q;
    }
}

/// 1.0 for every joint at or below `root`, 0.0 elsewhere. `root < 0` means
/// "whole skeleton" so a layer with no mask is a plain full-body override.
pub(super) fn build_mask_weights(skeleton: &SkeletonData, root: i32) -> Vec<f32> {
    let n = skeleton.joints.len();
    if root < 0 || (root as usize) >= n {
        return vec![1.0; n];
    }
    let mut w = vec![0.0f32; n];
    let mut stack = vec![root as usize];
    while let Some(j) = stack.pop() {
        if j >= n || w[j] > 0.0 {
            continue;
        }
        w[j] = 1.0;
        for &c in &skeleton.joints[j].children {
            stack.push(c);
        }
    }
    w
}

// ============================================================
// Matrix / quaternion helpers for skeletal animation
// ============================================================

pub(super) fn mat4_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub(super) fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

pub(super) fn mat4_from_trs(t: &[f32; 3], r: &[f32; 4], s: &[f32; 3]) -> [[f32; 4]; 4] {
    let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    // Column-major: m[col][row]
    [
        [
            (1.0 - (yy + zz)) * s[0],
            (xy + wz) * s[0],
            (xz - wy) * s[0],
            0.0,
        ], // column 0
        [
            (xy - wz) * s[1],
            (1.0 - (xx + zz)) * s[1],
            (yz + wx) * s[1],
            0.0,
        ], // column 1
        [
            (xz + wy) * s[2],
            (yz - wx) * s[2],
            (1.0 - (xx + yy)) * s[2],
            0.0,
        ], // column 2
        [t[0], t[1], t[2], 1.0], // column 3 (translation)
    ]
}

pub(super) fn quat_slerp(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut b2 = *b;
    if dot < 0.0 {
        dot = -dot;
        b2 = [-b[0], -b[1], -b[2], -b[3]];
    }
    if dot > 0.9995 {
        let mut out = [
            a[0] + t * (b2[0] - a[0]),
            a[1] + t * (b2[1] - a[1]),
            a[2] + t * (b2[2] - a[2]),
            a[3] + t * (b2[3] - a[3]),
        ];
        let len = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
        if len > 0.0 {
            for v in &mut out {
                *v /= len;
            }
        }
        return out;
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;
    [
        wa * a[0] + wb * b2[0],
        wa * a[1] + wb * b2[1],
        wa * a[2] + wb * b2[2],
        wa * a[3] + wb * b2[3],
    ]
}

pub(super) fn lerp_vec3(a: &[f32; 3], b: &[f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

pub(super) fn find_keyframe_pair(timestamps: &[f32], time: f32) -> (usize, usize, f32) {
    if timestamps.len() <= 1 {
        return (0, 0, 0.0);
    }
    if time <= timestamps[0] {
        return (0, 0, 0.0);
    }
    if time >= timestamps[timestamps.len() - 1] {
        let last = timestamps.len() - 1;
        return (last, last, 0.0);
    }
    for i in 0..timestamps.len() - 1 {
        if time >= timestamps[i] && time < timestamps[i + 1] {
            let dt = timestamps[i + 1] - timestamps[i];
            let t = if dt > 0.0 {
                (time - timestamps[i]) / dt
            } else {
                0.0
            };
            return (i, i + 1, t);
        }
    }
    let last = timestamps.len() - 1;
    (last, last, 0.0)
}

pub(super) fn sample_vec3(timestamps: &[f32], values: &[[f32; 3]], time: f32) -> [f32; 3] {
    if values.is_empty() {
        return [0.0; 3];
    }
    if values.len() == 1 {
        return values[0];
    }
    let (i0, i1, t) = find_keyframe_pair(timestamps, time);
    if i0 >= values.len() {
        return values[values.len() - 1];
    }
    if i1 >= values.len() {
        return values[values.len() - 1];
    }
    lerp_vec3(&values[i0], &values[i1], t)
}

pub(super) fn sample_quat(timestamps: &[f32], values: &[[f32; 4]], time: f32) -> [f32; 4] {
    if values.is_empty() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if values.len() == 1 {
        return values[0];
    }
    let (i0, i1, t) = find_keyframe_pair(timestamps, time);
    if i0 >= values.len() {
        return values[values.len() - 1];
    }
    if i1 >= values.len() {
        return values[values.len() - 1];
    }
    quat_slerp(&values[i0], &values[i1], t)
}

pub(super) fn compute_joint_transforms(
    skeleton: &SkeletonData,
    joint_idx: usize,
    parent_transform: &[[f32; 4]; 4],
    translations: &[[f32; 3]],
    rotations: &[[f32; 4]],
    scales: &[[f32; 3]],
    world_transforms: &mut [[[f32; 4]; 4]],
) {
    if joint_idx >= skeleton.joints.len() {
        return;
    }
    let local = mat4_from_trs(
        &translations[joint_idx],
        &rotations[joint_idx],
        &scales[joint_idx],
    );
    let world = mat4_mul(parent_transform, &local);
    world_transforms[joint_idx] = world;
    let children = skeleton.joints[joint_idx].children.clone();
    for &child in &children {
        compute_joint_transforms(
            skeleton,
            child,
            &world,
            translations,
            rotations,
            scales,
            world_transforms,
        );
    }
}

// ---- Catmull-Rom spline helpers (Q9) ----

pub(super) fn catmull_rom_point(points: &[f32], n: usize, segment: usize, t: f32) -> [f32; 3] {
    // Indices: p0 = segment - 1, p1 = segment, p2 = segment + 1, p3 = segment + 2.
    // Clamp at boundaries.
    let i0 = if segment > 0 { segment - 1 } else { 0 };
    let i1 = segment;
    let i2 = if segment + 1 < n { segment + 1 } else { n - 1 };
    let i3 = if segment + 2 < n { segment + 2 } else { n - 1 };

    let p0 = [points[i0 * 3], points[i0 * 3 + 1], points[i0 * 3 + 2]];
    let p1 = [points[i1 * 3], points[i1 * 3 + 1], points[i1 * 3 + 2]];
    let p2 = [points[i2 * 3], points[i2 * 3 + 1], points[i2 * 3 + 2]];
    let p3 = [points[i3 * 3], points[i3 * 3 + 1], points[i3 * 3 + 2]];

    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        out[k] = 0.5
            * ((2.0 * p1[k])
                + (-p0[k] + p2[k]) * t
                + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2
                + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3);
    }
    out
}

pub(super) fn update_bounds(bmin: &mut [f32; 3], bmax: &mut [f32; 3], x: f32, y: f32, z: f32) {
    if x < bmin[0] {
        bmin[0] = x;
    }
    if y < bmin[1] {
        bmin[1] = y;
    }
    if z < bmin[2] {
        bmin[2] = z;
    }
    if x > bmax[0] {
        bmax[0] = x;
    }
    if y > bmax[1] {
        bmax[1] = y;
    }
    if z > bmax[2] {
        bmax[2] = z;
    }
}
