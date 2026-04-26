//! Hillaire 2020 atmosphere LUTs — transmittance + multi-scattering.
//!
//! Two CPU-baked tables that capture the time-invariant part of an
//! Earth-like atmosphere:
//!
//! - **Transmittance** `T(r, μ)` — fraction of light surviving the
//!   journey from a point at radius `r` along a ray with zenith
//!   cosine `μ` to the top of the atmosphere. Indexed by
//!   `(view-zenith-cosine, altitude)`.
//! - **Multi-scattering** `ψ_ms(r, μ_s)` — Hillaire's energy-conserving
//!   second-and-higher-order scattering term, sampled per pixel of the
//!   sky-view LUT to add the bounce light that single-scattering alone
//!   misses (the bright zenith on overcast days, dome glow at sunset).
//!
//! Both bake once at renderer init and never change, so they live
//! CPU-side. The pattern mirrors `brdf_lut.rs`: pure Rust producing
//! `Rgba16Float` texel data ready for `queue.write_texture`, parallel
//! across cores on native, single-thread on wasm.
//!
//! Sizes are platform-tiered (RFC 0002): desktop gets Hillaire's
//! defaults, web/mobile get a smaller tier to halve memory and bake
//! cost on the targets that hurt most.

use half::f16;

// ----------------------------------------------------------------------------
// Platform-tiered LUT dimensions (RFC 0002)
// ----------------------------------------------------------------------------

#[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
pub const TRANSMITTANCE_W: u32 = 128;
#[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
pub const TRANSMITTANCE_H: u32 = 32;

#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub const TRANSMITTANCE_W: u32 = 256;
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub const TRANSMITTANCE_H: u32 = 64;

// Multi-scattering LUT is small enough to be the same across tiers.
pub const MULTI_SCATTERING_SIZE: u32 = 32;

// Sky-view LUT — recomputed every sun move via GPU compute pass.
#[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
pub const SKY_VIEW_W: u32 = 128;
#[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
pub const SKY_VIEW_H: u32 = 72;

#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub const SKY_VIEW_W: u32 = 192;
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub const SKY_VIEW_H: u32 = 108;

// Aerial-perspective 3D LUT — recomputed each frame (camera moves).
// Indexed by (NDC.x, NDC.y, depth-slice). Stores per-voxel
// (in-scatter rgb, mean transmittance). Smaller tier on web/mobile
// keeps the per-frame compute under 1 ms on integrated GPUs.
#[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
pub const AERIAL_W: u32 = 16;
#[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
pub const AERIAL_H: u32 = 16;
#[cfg(any(target_arch = "wasm32", target_os = "ios", target_os = "android"))]
pub const AERIAL_D: u32 = 16;

#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub const AERIAL_W: u32 = 32;
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub const AERIAL_H: u32 = 32;
#[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android")))]
pub const AERIAL_D: u32 = 32;

/// Maximum view-space distance the aerial-perspective LUT covers, in
/// kilometres. Beyond this the shader clamps to the deepest slice (which
/// has the longest accumulated path, so it's a sensible fallback for
/// truly distant geometry like skybox-far mountains).
pub const AERIAL_MAX_DIST_KM: f32 = 32.0;

// ----------------------------------------------------------------------------
// Earth-like atmosphere constants (Hillaire 2020, Bruneton-derived)
// All distances in kilometres; coefficients in 1/km.
// ----------------------------------------------------------------------------

const GROUND_RADIUS: f32 = 6360.0;
const ATMOSPHERE_TOP: f32 = 6460.0;
const ATMOSPHERE_THICKNESS: f32 = ATMOSPHERE_TOP - GROUND_RADIUS;

const RAYLEIGH_SCALE_HEIGHT: f32 = 8.0;
const MIE_SCALE_HEIGHT: f32 = 1.2;

// Rayleigh scattering = extinction (no absorption).
const RAYLEIGH_SCATTERING: [f32; 3] = [5.802e-3, 13.558e-3, 33.100e-3];

// Mie is dust/aerosol — small absorption beyond scattering.
const MIE_SCATTERING: f32 = 3.996e-3;
const MIE_EXTINCTION: f32 = 4.440e-3;

// Ozone — triangular density profile peaked at 25 km, half-width 15 km.
// Pure absorption, no scattering. Tints sunsets blue at high zenith.
const OZONE_ABSORPTION: [f32; 3] = [0.650e-3, 1.881e-3, 0.085e-3];
const OZONE_PEAK_ALTITUDE: f32 = 25.0;
const OZONE_HALF_WIDTH: f32 = 15.0;

const TRANSMITTANCE_STEPS: u32 = 40;
const MULTI_SCATTERING_SQRT_SAMPLES: u32 = 8; // 64 sphere directions per texel

// ----------------------------------------------------------------------------
// Density profiles
// ----------------------------------------------------------------------------

#[inline]
fn rayleigh_density(altitude_km: f32) -> f32 {
    (-altitude_km / RAYLEIGH_SCALE_HEIGHT).exp()
}

#[inline]
fn mie_density(altitude_km: f32) -> f32 {
    (-altitude_km / MIE_SCALE_HEIGHT).exp()
}

#[inline]
fn ozone_density(altitude_km: f32) -> f32 {
    (1.0 - (altitude_km - OZONE_PEAK_ALTITUDE).abs() / OZONE_HALF_WIDTH).max(0.0)
}

// Combined extinction at altitude (per channel, per km).
fn extinction(altitude_km: f32) -> [f32; 3] {
    let r_d = rayleigh_density(altitude_km);
    let m_d = mie_density(altitude_km);
    let o_d = ozone_density(altitude_km);
    [
        RAYLEIGH_SCATTERING[0] * r_d + MIE_EXTINCTION * m_d + OZONE_ABSORPTION[0] * o_d,
        RAYLEIGH_SCATTERING[1] * r_d + MIE_EXTINCTION * m_d + OZONE_ABSORPTION[1] * o_d,
        RAYLEIGH_SCATTERING[2] * r_d + MIE_EXTINCTION * m_d + OZONE_ABSORPTION[2] * o_d,
    ]
}

// ----------------------------------------------------------------------------
// Ray-sphere intersection — distance from point at radius `r` with
// zenith cosine `mu` to either the atmosphere top or the planet ground.
// Returns None if the ray doesn't hit the target sphere.
// ----------------------------------------------------------------------------

fn ray_sphere_intersect(r: f32, mu: f32, sphere_radius: f32) -> Option<f32> {
    let discriminant = r * r * (mu * mu - 1.0) + sphere_radius * sphere_radius;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_d = discriminant.sqrt();
    let t1 = -r * mu - sqrt_d;
    let t2 = -r * mu + sqrt_d;
    // We want the first positive intersection ahead of us.
    if t1 > 0.0 {
        Some(t1)
    } else if t2 > 0.0 {
        Some(t2)
    } else {
        None
    }
}

/// Distance from `(r, mu)` to whichever boundary the ray hits first.
/// If the ray hits the ground, returns the ground distance — caller
/// will see zero transmittance through opaque earth.
fn distance_to_boundary(r: f32, mu: f32) -> f32 {
    let to_top = ray_sphere_intersect(r, mu, ATMOSPHERE_TOP);
    let to_ground = ray_sphere_intersect(r, mu, GROUND_RADIUS);
    match (to_top, to_ground) {
        (Some(t), Some(g)) => t.min(g),
        (Some(t), None) => t,
        (None, Some(g)) => g,
        (None, None) => 0.0,
    }
}

// ----------------------------------------------------------------------------
// Transmittance LUT
// ----------------------------------------------------------------------------

/// LUT parameterization (V1, simple linear):
///   u = (μ + 1) / 2   — view-zenith cosine in [-1, 1] linearly mapped to texel
///   v = (r - rg) / (rt - rg)  — altitude fraction
fn transmittance_uv_to_state(u: f32, v: f32) -> (f32, f32) {
    let r = GROUND_RADIUS + v * ATMOSPHERE_THICKNESS;
    let mu = (u * 2.0 - 1.0).clamp(-1.0, 1.0);
    (r, mu)
}

/// Optical depth from `(r, mu)` along its ray to the atmosphere top
/// (or ground, whichever it hits first), per channel.
fn compute_optical_depth(r: f32, mu: f32) -> [f32; 3] {
    let total_dist = distance_to_boundary(r, mu);
    if total_dist <= 0.0 {
        return [0.0; 3];
    }
    let dx = total_dist / TRANSMITTANCE_STEPS as f32;
    let mut sum = [0.0_f32; 3];
    // Trapezoidal rule — sample at midpoints of each segment.
    for i in 0..TRANSMITTANCE_STEPS {
        let d = (i as f32 + 0.5) * dx;
        // Radius at distance d along the ray.
        let r_d = (r * r + d * d + 2.0 * r * mu * d).sqrt();
        let altitude = (r_d - GROUND_RADIUS).max(0.0);
        let e = extinction(altitude);
        sum[0] += e[0] * dx;
        sum[1] += e[1] * dx;
        sum[2] += e[2] * dx;
    }
    sum
}

fn build_transmittance_row(y: u32, w: u32, h: u32) -> Vec<u16> {
    let v = (y as f32 + 0.5) / h as f32;
    let mut row = Vec::with_capacity(w as usize * 4);
    for x in 0..w {
        let u = (x as f32 + 0.5) / w as f32;
        let (r, mu) = transmittance_uv_to_state(u, v);
        let tau = compute_optical_depth(r, mu);
        let t = [(-tau[0]).exp(), (-tau[1]).exp(), (-tau[2]).exp()];
        row.push(f16::from_f32(t[0]).to_bits());
        row.push(f16::from_f32(t[1]).to_bits());
        row.push(f16::from_f32(t[2]).to_bits());
        row.push(f16::from_f32(1.0).to_bits()); // alpha unused
    }
    row
}

/// Build the transmittance LUT as `Rgba16Float` texels, row-major,
/// suitable for `queue.write_texture`. Each texel stores per-channel
/// transmittance; alpha is unused (set to 1).
pub fn build_transmittance_lut(w: u32, h: u32) -> Vec<u16> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let rows_per_thread = (h as usize + nthreads - 1) / nthreads;
        let mut all_rows: Vec<Option<Vec<Vec<u16>>>> = (0..nthreads).map(|_| None).collect();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(nthreads);
            for t in 0..nthreads {
                let y_start = (t * rows_per_thread) as u32;
                let y_end = (((t + 1) * rows_per_thread).min(h as usize)) as u32;
                let h_handle = s.spawn(move || {
                    (y_start..y_end)
                        .map(|y| build_transmittance_row(y, w, h))
                        .collect::<Vec<_>>()
                });
                handles.push(h_handle);
            }
            for (t, handle) in handles.into_iter().enumerate() {
                all_rows[t] = Some(handle.join().unwrap());
            }
        });
        all_rows.into_iter().flatten().flatten().flatten().collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..h).flat_map(|y| build_transmittance_row(y, w, h)).collect()
    }
}

/// Sample the transmittance LUT bilinearly. Returns linear RGB
/// transmittance ∈ [0, 1]³. Used by tests and Phase 3 (sun-shaft +
/// directional-light coloring).
pub fn sample_transmittance_lut(lut: &[u16], w: u32, h: u32, r: f32, mu: f32) -> [f32; 3] {
    let v = ((r - GROUND_RADIUS) / ATMOSPHERE_THICKNESS).clamp(0.0, 1.0);
    let u = ((mu + 1.0) * 0.5).clamp(0.0, 1.0);
    let fx = u * w as f32 - 0.5;
    let fy = v * h as f32 - 0.5;
    let x0 = (fx.floor() as i32).clamp(0, w as i32 - 1) as u32;
    let y0 = (fy.floor() as i32).clamp(0, h as i32 - 1) as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = (fx - x0 as f32).clamp(0.0, 1.0);
    let ty = (fy - y0 as f32).clamp(0.0, 1.0);
    let fetch = |x: u32, y: u32| -> [f32; 3] {
        let base = ((y * w + x) * 4) as usize;
        [
            f16::from_bits(lut[base]).to_f32(),
            f16::from_bits(lut[base + 1]).to_f32(),
            f16::from_bits(lut[base + 2]).to_f32(),
        ]
    };
    let c00 = fetch(x0, y0);
    let c10 = fetch(x1, y0);
    let c01 = fetch(x0, y1);
    let c11 = fetch(x1, y1);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| {
        [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
    };
    let bottom = lerp3(c00, c10, tx);
    let top = lerp3(c01, c11, tx);
    lerp3(bottom, top, ty)
}

// ----------------------------------------------------------------------------
// Multi-scattering LUT
// ----------------------------------------------------------------------------
//
// Hillaire's "F_ms" energy-conserving multi-scattering term. For each
// (μ_s, altitude) texel: place point P at that altitude with the sun
// at zenith cosine μ_s. Integrate over the unit sphere — for each
// sphere direction, ray-march and accumulate the single-scattering
// that would arrive along it from the sun. The integrated isotropic
// in-scattered light gives ψ_ms, which the per-frame sky-view LUT
// uses to add the multi-bounce contribution analytically.
//
// We use the Hillaire 2020 closed-form: integrate L_2 (single-
// scattered light hitting P from all sphere directions) and F_ms
// (the fraction of energy scattered back into each direction), then
// the total multi-scattering is L_2 / (1 - F_ms) — geometric series
// for infinite bounces under the isotropic-phase approximation.

fn ms_uv_to_state(u: f32, v: f32) -> (f32, f32) {
    // u: μ_s ∈ [-1, 1] linear (sun zenith cosine)
    // v: altitude fraction
    let mu_s = (u * 2.0 - 1.0).clamp(-1.0, 1.0);
    let r = GROUND_RADIUS + v * ATMOSPHERE_THICKNESS;
    (r, mu_s)
}

fn scattering_at(altitude_km: f32) -> ([f32; 3], f32) {
    // Returns (rayleigh_scattering, mie_scattering) coefficients at altitude.
    let r_d = rayleigh_density(altitude_km);
    let m_d = mie_density(altitude_km);
    (
        [
            RAYLEIGH_SCATTERING[0] * r_d,
            RAYLEIGH_SCATTERING[1] * r_d,
            RAYLEIGH_SCATTERING[2] * r_d,
        ],
        MIE_SCATTERING * m_d,
    )
}

/// March from `(r, mu)` toward the boundary, accumulating single-
/// scattered radiance from a sun at zenith cosine `mu_s`. Uses the
/// LUT (already-baked transmittance) for the sun-to-point path.
fn integrate_single_scatter_along_ray(
    transmittance_lut: &[u16],
    tw: u32,
    th: u32,
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32, // cos(angle between view ray and sun direction) — for phase
    steps: u32,
) -> ([f32; 3], [f32; 3]) {
    // Returns (L_2_contribution, F_ms_contribution) — both per-channel.
    let total_dist = distance_to_boundary(r, mu);
    if total_dist <= 0.0 {
        return ([0.0; 3], [0.0; 3]);
    }
    let dx = total_dist / steps as f32;
    let mut l_sum = [0.0_f32; 3];
    let mut f_sum = [0.0_f32; 3];
    let mut tau = [0.0_f32; 3]; // running optical depth from (r, mu) origin
    for i in 0..steps {
        let d = (i as f32 + 0.5) * dx;
        let r_d = (r * r + d * d + 2.0 * r * mu * d).sqrt();
        let altitude = (r_d - GROUND_RADIUS).max(0.0);
        let e = extinction(altitude);
        tau[0] += e[0] * dx;
        tau[1] += e[1] * dx;
        tau[2] += e[2] * dx;
        let t_origin = [(-tau[0]).exp(), (-tau[1]).exp(), (-tau[2]).exp()];

        // μ_s as seen from the new sample point (assuming sun direction stays fixed in
        // world frame — to first order this is a good approximation since the atmosphere
        // is thin compared to the planet radius).
        let mu_s_at_p = ((r * mu_s + d * nu) / r_d).clamp(-1.0, 1.0);
        let t_sun = sample_transmittance_lut(transmittance_lut, tw, th, r_d, mu_s_at_p);

        let (sigma_r, sigma_m) = scattering_at(altitude);
        // Isotropic phase = 1/(4π) — Hillaire's approximation for the multi-
        // scattering bake. Folded into the L_2 / (1 - F_ms) formula at lookup time.
        for ch in 0..3 {
            let scattering = sigma_r[ch] + sigma_m;
            l_sum[ch] += t_origin[ch] * t_sun[ch] * scattering * dx;
            f_sum[ch] += t_origin[ch] * scattering * dx;
        }
    }
    (l_sum, f_sum)
}

fn build_multi_scattering_row(transmittance_lut: &[u16], tw: u32, th: u32, y: u32, size: u32) -> Vec<u16> {
    let v = (y as f32 + 0.5) / size as f32;
    let mut row = Vec::with_capacity(size as usize * 4);
    let n_dirs = MULTI_SCATTERING_SQRT_SAMPLES;
    let inv_n = 1.0 / (n_dirs * n_dirs) as f32;
    for x in 0..size {
        let u = (x as f32 + 0.5) / size as f32;
        let (r, mu_s) = ms_uv_to_state(u, v);

        let mut l_total = [0.0_f32; 3];
        let mut f_total = [0.0_f32; 3];
        // Spherical Fibonacci-like distribution over the unit sphere (n_dirs²
        // samples). For each direction, integrate single-scattered radiance
        // along that direction; accumulate L_2 and F_ms per channel.
        for i in 0..n_dirs {
            for j in 0..n_dirs {
                let theta = std::f32::consts::PI * (i as f32 + 0.5) / n_dirs as f32;
                let phi = 2.0 * std::f32::consts::PI * (j as f32 + 0.5) / n_dirs as f32;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                // Direction in a frame where sun = (sin_su, 0, cos_su).
                let sin_su = (1.0 - mu_s * mu_s).max(0.0).sqrt();
                let dir = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
                let mu = dir[2];
                let nu = dir[0] * sin_su + dir[2] * mu_s;
                let (l, f) = integrate_single_scatter_along_ray(
                    transmittance_lut,
                    tw,
                    th,
                    r,
                    mu,
                    mu_s,
                    nu,
                    20,
                );
                // Solid angle element sin(θ) dθ dφ; uniform sampling already.
                let weight = sin_theta * inv_n * std::f32::consts::PI; // ∫sinθdθ over [0,π] = 2; π/n_dirs² normalises
                for ch in 0..3 {
                    l_total[ch] += l[ch] * weight;
                    f_total[ch] += f[ch] * weight;
                }
            }
        }
        // Geometric series: total multi-scattering = L_2 / (1 - F_ms),
        // clamped to avoid runaway when F_ms approaches 1 (deep
        // atmosphere paths).
        let mut psi_ms = [0.0_f32; 3];
        for ch in 0..3 {
            let denom = (1.0 - f_total[ch]).max(0.05);
            psi_ms[ch] = (l_total[ch] / denom).max(0.0);
        }
        row.push(f16::from_f32(psi_ms[0]).to_bits());
        row.push(f16::from_f32(psi_ms[1]).to_bits());
        row.push(f16::from_f32(psi_ms[2]).to_bits());
        row.push(f16::from_f32(1.0).to_bits());
    }
    row
}

/// Build the multi-scattering LUT as `Rgba16Float` texels, row-major.
/// Requires the transmittance LUT (must be built first); samples it
/// internally to get the sun-to-point transmittance for each integration step.
pub fn build_multi_scattering_lut(transmittance_lut: &[u16], tw: u32, th: u32, size: u32) -> Vec<u16> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let rows_per_thread = (size as usize + nthreads - 1) / nthreads;
        let mut all_rows: Vec<Option<Vec<Vec<u16>>>> = (0..nthreads).map(|_| None).collect();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(nthreads);
            for t in 0..nthreads {
                let y_start = (t * rows_per_thread) as u32;
                let y_end = (((t + 1) * rows_per_thread).min(size as usize)) as u32;
                let lut_ref = transmittance_lut;
                let h_handle = s.spawn(move || {
                    (y_start..y_end)
                        .map(|y| build_multi_scattering_row(lut_ref, tw, th, y, size))
                        .collect::<Vec<_>>()
                });
                handles.push(h_handle);
            }
            for (t, handle) in handles.into_iter().enumerate() {
                all_rows[t] = Some(handle.join().unwrap());
            }
        });
        all_rows.into_iter().flatten().flatten().flatten().collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..size)
            .flat_map(|y| build_multi_scattering_row(transmittance_lut, tw, th, y, size))
            .collect()
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lerp_lut_at(lut: &[u16], w: u32, h: u32, r: f32, mu: f32) -> [f32; 3] {
        sample_transmittance_lut(lut, w, h, r, mu)
    }

    #[test]
    fn transmittance_zenith_sea_level_matches_hillaire_reference() {
        // Reference values derived analytically from Earth atmosphere
        // constants (Rayleigh + Mie + ozone) at sea level looking up:
        //   τ_R ≈ scattering × scale_height (zenith path)
        //   τ_M ≈ extinction × scale_height
        //   τ_O ≈ absorption × half_width
        // Per channel transmittance T = exp(-τ_total). Typical values
        // are ~(0.94, 0.87, 0.76) — atmosphere is more transparent
        // to red than blue, which is why the sky is blue.
        let w = TRANSMITTANCE_W;
        let h = TRANSMITTANCE_H;
        let lut = build_transmittance_lut(w, h);
        let t = lerp_lut_at(&lut, w, h, GROUND_RADIUS, 1.0);
        // Loose tolerance — exact values depend on integration step
        // count, ozone profile shape, and bilinear bias near the edge.
        assert!((t[0] - 0.94).abs() < 0.05, "R channel: got {}, expected ~0.94", t[0]);
        assert!((t[1] - 0.87).abs() < 0.05, "G channel: got {}, expected ~0.87", t[1]);
        assert!((t[2] - 0.76).abs() < 0.05, "B channel: got {}, expected ~0.76", t[2]);
        // R > G > B — Rayleigh scattering scales with 1/λ⁴.
        assert!(t[0] > t[1] && t[1] > t[2], "expected R > G > B, got {:?}", t);
    }

    #[test]
    fn transmittance_decreases_toward_horizon() {
        // At sea level, transmittance must monotonically drop as we
        // tilt from zenith (μ=1) down toward grazing (μ→0): longer
        // path through atmosphere = more extinction.
        let w = TRANSMITTANCE_W;
        let h = TRANSMITTANCE_H;
        let lut = build_transmittance_lut(w, h);
        let mu_samples = [1.0_f32, 0.8, 0.5, 0.3, 0.1];
        let mut prev = f32::INFINITY;
        for &mu in &mu_samples {
            let t = lerp_lut_at(&lut, w, h, GROUND_RADIUS, mu);
            // Use green channel as representative.
            assert!(
                t[1] < prev,
                "transmittance not decreasing: μ={} t.g={} (prev {})",
                mu,
                t[1],
                prev
            );
            prev = t[1];
        }
    }

    #[test]
    fn transmittance_grazing_horizon_substantially_attenuated() {
        // A grazing ray near the horizon traverses the longest path
        // through the densest atmosphere — blue should drop hard
        // (Rayleigh ~λ⁻⁴ over a long path), red survives more. This
        // is what makes sunrises and sunsets red: the sun's blue is
        // scattered out before it reaches the eye.
        let w = TRANSMITTANCE_W;
        let h = TRANSMITTANCE_H;
        let lut = build_transmittance_lut(w, h);
        let t = lerp_lut_at(&lut, w, h, GROUND_RADIUS, 0.05);
        assert!(t[2] < 0.2, "blue should be heavily attenuated at horizon, got {}", t[2]);
        assert!(t[0] > t[1] && t[1] > t[2], "horizon transmittance should still preserve R>G>B ordering, got {:?}", t);
        assert!(t[0] - t[2] > 0.2, "expected strong red/blue separation at horizon, got R-B={}", t[0] - t[2]);
    }

    #[test]
    fn multi_scattering_non_negative_and_finite() {
        // Multi-scattering is energy added on top of single-scattering;
        // it must be ≥ 0 everywhere and finite (the F_ms denominator
        // clamp guards against blow-up).
        let tw = TRANSMITTANCE_W;
        let th = TRANSMITTANCE_H;
        let t_lut = build_transmittance_lut(tw, th);
        let ms = build_multi_scattering_lut(&t_lut, tw, th, MULTI_SCATTERING_SIZE);
        assert_eq!(ms.len() as u32, MULTI_SCATTERING_SIZE * MULTI_SCATTERING_SIZE * 4);
        for chunk in ms.chunks(4) {
            for &bits in &chunk[..3] {
                let v = f16::from_bits(bits).to_f32();
                assert!(v.is_finite(), "non-finite multi-scattering value");
                assert!(v >= 0.0, "negative multi-scattering value: {}", v);
            }
        }
    }
}
