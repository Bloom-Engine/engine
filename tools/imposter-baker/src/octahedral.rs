//! Octahedral encode/decode — Rust port of
//! `native/shared/shaders/common/imposter.wgsl`.
//!
//! Keep these in sync with the WGSL or the runtime atlas sampler will
//! pick a different cell than the baker wrote.

/// Decode an octahedral UV in `[-1, 1]^2` to a unit direction.
/// Mirror of `octahedral_decode` in imposter.wgsl.
pub fn octahedral_decode(uv: [f32; 2]) -> [f32; 3] {
    let abs_uv = [uv[0].abs(), uv[1].abs()];
    let y = 1.0 - abs_uv[0] - abs_uv[1];
    let v = if y >= 0.0 {
        [uv[0], y, uv[1]]
    } else {
        let sx = if uv[0] >= 0.0 { 1.0 } else { -1.0 };
        let sy = if uv[1] >= 0.0 { 1.0 } else { -1.0 };
        [sx * (1.0 - abs_uv[1]), y, sy * (1.0 - abs_uv[0])]
    };
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Cell-center UV in `[-1, 1]^2` for grid coords `(i, j)` of an `N×N`
/// atlas. Matches the runtime `floor(oct * GRID)` cell selection: the
/// runtime samples the cell whose lower-left corner is `cell / N`, so
/// the bake direction must be the *center* of that cell, i.e.
/// `(cell + 0.5) / N` mapped from `[0, 1]` to `[-1, 1]`.
pub fn cell_center_uv(i: u32, j: u32, n: u32) -> [f32; 2] {
    let nf = n as f32;
    let u = ((i as f32 + 0.5) / nf) * 2.0 - 1.0;
    let v = ((j as f32 + 0.5) / nf) * 2.0 - 1.0;
    [u, v]
}

/// Direction (model → camera) for cell `(i, j)`.
pub fn cell_direction(i: u32, j: u32, n: u32) -> [f32; 3] {
    octahedral_decode(cell_center_uv(i, j, n))
}
