//! Catmull-Rom spline helpers of `gen_mesh_spline_ribbon` (split from models.rs).

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
