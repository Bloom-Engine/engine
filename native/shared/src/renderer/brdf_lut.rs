//! Split-sum BRDF LUT generator.
//!
//! Pre-integrates the GGX BRDF into a 2D `Rg16Float` table indexed by
//! `(NdotV, roughness)` that the PBR shader samples at runtime via
//! the Karis-2013 split-sum approximation. Entirely CPU-side —
//! builds once at startup via `build_brdf_lut(size)` and hands the
//! result to `queue.write_texture` in `Renderer::new`.
//!
//! Private helpers (Hammersley, GGX importance sampling, Smith
//! geometry) stay module-scoped; only `build_brdf_lut` is `pub` so
//! `Renderer::new` can reach it.

// ============================================================
// Split-sum BRDF LUT
// ============================================================
//
// Pre-integrate the GGX BRDF over hemisphere directions. The output
// 2D table is sampled at runtime as `brdf_lut(NdotV, roughness)` and
// gives a (scale, bias) pair such that:
//   IBL_specular = prefiltered_env_sample * (F0 * scale + bias)
// This is the second sum of the Karis 2013 split-sum approximation.
//
// Importance-samples GGX in the H direction, integrates the visibility
// × Fresnel(VdotH) part. The Fresnel split into scale (F0) and bias
// (1) lets us factor F0 out of the integral.

const BRDF_LUT_SAMPLES: u32 = 1024;

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    (bits as f32) * 2.328_306_4e-10
}

fn hammersley(i: u32, n: u32) -> (f32, f32) {
    (i as f32 / n as f32, radical_inverse_vdc(i))
}

fn importance_sample_ggx(xi: (f32, f32), n: [f32; 3], roughness: f32) -> [f32; 3] {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * xi.0;
    let cos_theta = ((1.0 - xi.1) / (1.0 + (a * a - 1.0) * xi.1)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let h_local = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];

    // Build TBN around N.
    let up = if n[2].abs() < 0.999 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let t = normalize3(cross3(up, n));
    let b = cross3(n, t);
    [
        t[0] * h_local[0] + b[0] * h_local[1] + n[0] * h_local[2],
        t[1] * h_local[0] + b[1] * h_local[1] + n[1] * h_local[2],
        t[2] * h_local[0] + b[2] * h_local[1] + n[2] * h_local[2],
    ]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn geometry_smith_ggx_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    // IBL geometry uses k = (alpha²)/2 (Disney) — different from
    // direct-lighting k. Returns G1(V) * G1(L).
    let a = roughness;
    let k = (a * a) / 2.0;
    let g1v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    g1v * g1l
}

fn build_brdf_lut_row(y: usize, size: usize) -> Vec<u16> {
    let n = [0.0_f32, 0.0, 1.0];
    let roughness = ((y as f32) + 0.5) / size as f32;
    let mut row = Vec::with_capacity(size * 2);
    for x in 0..size {
        let n_dot_v = ((x as f32) + 0.5) / size as f32;
        let v = [
            (1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(),
            0.0,
            n_dot_v,
        ];
        let mut a_sum = 0.0_f32;
        let mut b_sum = 0.0_f32;
        for i in 0..BRDF_LUT_SAMPLES {
            let xi = hammersley(i, BRDF_LUT_SAMPLES);
            let h = importance_sample_ggx(xi, n, roughness);
            let v_dot_h = dot3(v, h).max(0.0);
            let l = [
                2.0 * v_dot_h * h[0] - v[0],
                2.0 * v_dot_h * h[1] - v[1],
                2.0 * v_dot_h * h[2] - v[2],
            ];
            let n_dot_l = l[2].max(0.0);
            let n_dot_h = h[2].max(0.0);
            if n_dot_l > 0.0 {
                let g = geometry_smith_ggx_ibl(n_dot_v, n_dot_l, roughness);
                let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v + 1e-6);
                let fc = (1.0 - v_dot_h).powi(5);
                a_sum += (1.0 - fc) * g_vis;
                b_sum += fc * g_vis;
            }
        }
        let scale = a_sum / BRDF_LUT_SAMPLES as f32;
        let bias = b_sum / BRDF_LUT_SAMPLES as f32;
        row.push(half::f16::from_f32(scale).to_bits());
        row.push(half::f16::from_f32(bias).to_bits());
    }
    row
}

/// Build a `size × size` BRDF LUT as packed Rg16Float texels. Each
/// row is constant `roughness` (v axis), each column constant `NdotV`
/// (u axis). Output is row-major suitable for write_texture. Splits
/// across `available_parallelism()` threads since cells are
/// independent — keeps startup latency manageable even at 1024 spp.
pub fn build_brdf_lut(size: usize) -> Vec<u16> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let rows_per_thread = (size + nthreads - 1) / nthreads;
        let mut all_rows: Vec<Option<Vec<Vec<u16>>>> = (0..nthreads).map(|_| None).collect();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(nthreads);
            for t in 0..nthreads {
                let y_start = t * rows_per_thread;
                let y_end = ((t + 1) * rows_per_thread).min(size);
                let h = s.spawn(move || {
                    (y_start..y_end).map(|y| build_brdf_lut_row(y, size)).collect::<Vec<_>>()
                });
                handles.push(h);
            }
            for (t, h) in handles.into_iter().enumerate() {
                all_rows[t] = Some(h.join().unwrap());
            }
        });
        all_rows.into_iter().flatten().flatten().flatten().collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..size).flat_map(|y| build_brdf_lut_row(y, size)).collect()
    }
}
