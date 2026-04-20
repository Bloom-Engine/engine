//! Pure utility helpers extracted from the renderer monolith.
//!
//! - `IDENTITY_MAT4` + the `mat4_*` family: column-major 4×4 matrix
//!   math used by the renderer internally *and* by sibling modules
//!   (`shadows`, `picking`) via `crate::renderer::mat4_*`. Reexported
//!   from `renderer/mod.rs` with `pub use util::*;` so external paths
//!   stay stable.
//! - `encode_png_simple`: RGB → PNG byte blob, used by the screenshot
//!   path. `pub(super)` — only `renderer::` needs it.

pub const IDENTITY_MAT4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// ============================================================
// Matrix math helpers (column-major for WGSL)
// ============================================================

pub fn mat4_perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy / 2.0).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}

pub fn mat4_ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    // wgpu NDC z range is [0, 1] (not OpenGL's [-1, 1]). Matching
    // this so shadow-map fragments at near half-depth don't get
    // clipped and sample_shadow's in-frustum test (z in [0, 1])
    // actually works.
    let lr = 1.0 / (left - right);
    let bt = 1.0 / (bottom - top);
    let nf = 1.0 / (near - far);
    [
        [-2.0 * lr, 0.0, 0.0, 0.0],
        [0.0, -2.0 * bt, 0.0, 0.0],
        [0.0, 0.0, nf,   0.0],
        [(left + right) * lr, (top + bottom) * bt, near * nf, 1.0],
    ]
}

pub fn mat4_look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let fx = center[0] - eye[0];
    let fy = center[1] - eye[1];
    let fz = center[2] - eye[2];
    let flen = (fx*fx + fy*fy + fz*fz).sqrt();
    let (fx, fy, fz) = (fx/flen, fy/flen, fz/flen);

    let sx = fy * up[2] - fz * up[1];
    let sy = fz * up[0] - fx * up[2];
    let sz = fx * up[1] - fy * up[0];
    let slen = (sx*sx + sy*sy + sz*sz).sqrt();
    let (sx, sy, sz) = (sx/slen, sy/slen, sz/slen);

    let ux = sy * fz - sz * fy;
    let uy = sz * fx - sx * fz;
    let uz = sx * fy - sy * fx;

    [
        [sx, ux, -fx, 0.0],
        [sy, uy, -fy, 0.0],
        [sz, uz, -fz, 0.0],
        [-(sx*eye[0]+sy*eye[1]+sz*eye[2]), -(ux*eye[0]+uy*eye[1]+uz*eye[2]), fx*eye[0]+fy*eye[1]+fz*eye[2], 1.0],
    ]
}

pub fn mat4_multiply(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row]*b[col][0] + a[1][row]*b[col][1] + a[2][row]*b[col][2] + a[3][row]*b[col][3];
        }
    }
    out
}

/// Multiply a column-major 4x4 matrix by a column vector.
pub fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
    [
        m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2] + m[3][0]*v[3],
        m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2] + m[3][1]*v[3],
        m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2] + m[3][2]*v[3],
        m[0][3]*v[0] + m[1][3]*v[1] + m[2][3]*v[2] + m[3][3]*v[3],
    ]
}

pub fn mat4_translate(m: [[f32; 4]; 4], v: [f32; 3]) -> [[f32; 4]; 4] {
    let mut out = m;
    for i in 0..4 {
        out[3][i] += m[0][i]*v[0] + m[1][i]*v[1] + m[2][i]*v[2];
    }
    out
}

pub fn mat4_scale(m: [[f32; 4]; 4], v: [f32; 3]) -> [[f32; 4]; 4] {
    let mut out = m;
    for i in 0..4 { out[0][i] *= v[0]; }
    for i in 0..4 { out[1][i] *= v[1]; }
    for i in 0..4 { out[2][i] *= v[2]; }
    out
}

pub fn mat4_invert(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let m = |r: usize, c: usize| m[c][r]; // accessor for row-major style
    let mut inv = [0.0f32; 16];
    inv[0]  =  m(1,1)*m(2,2)*m(3,3) - m(1,1)*m(2,3)*m(3,2) - m(2,1)*m(1,2)*m(3,3) + m(2,1)*m(1,3)*m(3,2) + m(3,1)*m(1,2)*m(2,3) - m(3,1)*m(1,3)*m(2,2);
    inv[4]  = -m(1,0)*m(2,2)*m(3,3) + m(1,0)*m(2,3)*m(3,2) + m(2,0)*m(1,2)*m(3,3) - m(2,0)*m(1,3)*m(3,2) - m(3,0)*m(1,2)*m(2,3) + m(3,0)*m(1,3)*m(2,2);
    inv[8]  =  m(1,0)*m(2,1)*m(3,3) - m(1,0)*m(2,3)*m(3,1) - m(2,0)*m(1,1)*m(3,3) + m(2,0)*m(1,3)*m(3,1) + m(3,0)*m(1,1)*m(2,3) - m(3,0)*m(1,3)*m(2,1);
    inv[12] = -m(1,0)*m(2,1)*m(3,2) + m(1,0)*m(2,2)*m(3,1) + m(2,0)*m(1,1)*m(3,2) - m(2,0)*m(1,2)*m(3,1) - m(3,0)*m(1,1)*m(2,2) + m(3,0)*m(1,2)*m(2,1);
    inv[1]  = -m(0,1)*m(2,2)*m(3,3) + m(0,1)*m(2,3)*m(3,2) + m(2,1)*m(0,2)*m(3,3) - m(2,1)*m(0,3)*m(3,2) - m(3,1)*m(0,2)*m(2,3) + m(3,1)*m(0,3)*m(2,2);
    inv[5]  =  m(0,0)*m(2,2)*m(3,3) - m(0,0)*m(2,3)*m(3,2) - m(2,0)*m(0,2)*m(3,3) + m(2,0)*m(0,3)*m(3,2) + m(3,0)*m(0,2)*m(2,3) - m(3,0)*m(0,3)*m(2,2);
    inv[9]  = -m(0,0)*m(2,1)*m(3,3) + m(0,0)*m(2,3)*m(3,1) + m(2,0)*m(0,1)*m(3,3) - m(2,0)*m(0,3)*m(3,1) - m(3,0)*m(0,1)*m(2,3) + m(3,0)*m(0,3)*m(2,1);
    inv[13] =  m(0,0)*m(2,1)*m(3,2) - m(0,0)*m(2,2)*m(3,1) - m(2,0)*m(0,1)*m(3,2) + m(2,0)*m(0,2)*m(3,1) + m(3,0)*m(0,1)*m(2,2) - m(3,0)*m(0,2)*m(2,1);
    inv[2]  =  m(0,1)*m(1,2)*m(3,3) - m(0,1)*m(1,3)*m(3,2) - m(1,1)*m(0,2)*m(3,3) + m(1,1)*m(0,3)*m(3,2) + m(3,1)*m(0,2)*m(1,3) - m(3,1)*m(0,3)*m(1,2);
    inv[6]  = -m(0,0)*m(1,2)*m(3,3) + m(0,0)*m(1,3)*m(3,2) + m(1,0)*m(0,2)*m(3,3) - m(1,0)*m(0,3)*m(3,2) - m(3,0)*m(0,2)*m(1,3) + m(3,0)*m(0,3)*m(1,2);
    inv[10] =  m(0,0)*m(1,1)*m(3,3) - m(0,0)*m(1,3)*m(3,1) - m(1,0)*m(0,1)*m(3,3) + m(1,0)*m(0,3)*m(3,1) + m(3,0)*m(0,1)*m(1,3) - m(3,0)*m(0,3)*m(1,1);
    inv[14] = -m(0,0)*m(1,1)*m(3,2) + m(0,0)*m(1,2)*m(3,1) + m(1,0)*m(0,1)*m(3,2) - m(1,0)*m(0,2)*m(3,1) - m(3,0)*m(0,1)*m(1,2) + m(3,0)*m(0,2)*m(1,1);
    inv[3]  = -m(0,1)*m(1,2)*m(2,3) + m(0,1)*m(1,3)*m(2,2) + m(1,1)*m(0,2)*m(2,3) - m(1,1)*m(0,3)*m(2,2) - m(2,1)*m(0,2)*m(1,3) + m(2,1)*m(0,3)*m(1,2);
    inv[7]  =  m(0,0)*m(1,2)*m(2,3) - m(0,0)*m(1,3)*m(2,2) - m(1,0)*m(0,2)*m(2,3) + m(1,0)*m(0,3)*m(2,2) + m(2,0)*m(0,2)*m(1,3) - m(2,0)*m(0,3)*m(1,2);
    inv[11] = -m(0,0)*m(1,1)*m(2,3) + m(0,0)*m(1,3)*m(2,1) + m(1,0)*m(0,1)*m(2,3) - m(1,0)*m(0,3)*m(2,1) - m(2,0)*m(0,1)*m(1,3) + m(2,0)*m(0,3)*m(1,1);
    inv[15] =  m(0,0)*m(1,1)*m(2,2) - m(0,0)*m(1,2)*m(2,1) - m(1,0)*m(0,1)*m(2,2) + m(1,0)*m(0,2)*m(2,1) + m(2,0)*m(0,1)*m(1,2) - m(2,0)*m(0,2)*m(1,1);

    let det = m(0,0)*inv[0] + m(0,1)*inv[4] + m(0,2)*inv[8] + m(0,3)*inv[12];
    if det.abs() < 1e-10 { return IDENTITY_MAT4; }
    let inv_det = 1.0 / det;
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = inv[col * 4 + row] * inv_det;
        }
    }
    out
}

/// Encode an RGB byte buffer (no alpha) as a PNG. Used by the
/// pending-screenshot path so callers can hand us a path and get a
/// PNG written to disk without worrying about cross-FFI buffer handoff.
pub(super) fn encode_png_simple(width: u32, height: u32, rgb: &[u8]) -> Option<Vec<u8>> {
    use image::{ImageBuffer, Rgb};
    let buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb.to_vec())?;
    let mut out = std::io::Cursor::new(Vec::new());
    buf.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}
