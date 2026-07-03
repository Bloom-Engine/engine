//! Hi-Z linear-depth pyramid + GTAO and its bilateral blur.
//! Split from renderer/shaders.rs.

/// GTAO (Ground Truth Ambient Occlusion) shader.
///
/// Horizon-based AO: for each pixel, reconstruct view-space position
/// from depth + inverse projection, then march 8 directions around
/// the pixel (4 steps each). At each step, compute the horizon angle
/// (elevation of the sample above the surface tangent plane). The
/// final AO is the average fraction of the hemisphere that is
/// unoccluded across all directions.
///
/// Uses interleaved gradient noise (IGN) for per-pixel direction
/// jitter to break banding. Output is Rg8Unorm at half resolution
/// — R = GTAO occlusion, G = contact-shadow factor (both 1 =
/// unoccluded, 0 = fully occluded).
/// Linearize the hardware depth buffer into mip 0 of the Hi-Z
/// pyramid. Stores *positive* view-space distance (|view_z|); sky
/// pixels (depth ≥ 0.9999) receive the sentinel `HIZ_SKY_Z` so a
/// subsequent `min` downsample never picks sky over real geometry.
pub(in crate::renderer) const HIZ_LINEARIZE_SHADER_WGSL: &str = "
struct Params {
    /// xy = inv_size (1/mip0_w, 1/mip0_h)
    /// z  = proj[2][2]
    /// w  = proj[3][2]
    params: vec4<f32>,
    /// xy = mip-0 size (u32). zw unused.
    size: vec4<u32>,
};

const HIZ_SKY_Z: f32 = 10000.0;

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var hiz_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.xy;
    if (px.x >= u.size.x || px.y >= u.size.y) { return; }
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) * u.params.xy;
    let d = textureSampleLevel(depth_tex, depth_samp, uv, 0);
    var linear_z: f32;
    // Sky = the cleared depth (exactly 1.0; one-ulp slack for driver
    // rounding). The previous 0.9999 threshold was calibrated for a
    // reversed-Z mindset: with this forward-Z projection, depth crams
    // toward 1.0 so fast that 0.9999 already fires at ~35 m — every
    // mid/far surface was written as 'sky' (10000).
    if (d >= 0.99999994) {
        linear_z = HIZ_SKY_Z;
    } else {
        // view_z = p32 / (d + p22); store |view_z|. For this projection
        // (p22 = far/(near-far), p32 = near*far/(near-far), both negative)
        // the numerator and denominator are BOTH negative for geometry, so
        // the quotient is the positive view depth. The previous leading
        // minus flipped every geometry pixel negative, and the max() then
        // clamped the whole depth pyramid to 0.0001 — GTAO's horizon scan
        // and the occlusion grid have been reading a flat 0.0001/10000
        // two-value field instead of real depth.
        linear_z = u.params.w / (d + u.params.z);
        linear_z = max(linear_z, 0.0001);
    }
    textureStore(hiz_out, vec2<i32>(px), vec4<f32>(linear_z, 0.0, 0.0, 0.0));
}
";

/// Downsample one Hi-Z mip into the next. Uses `min` so the coarser
/// mip reports the nearest occluder in its footprint — exactly what
/// the GTAO horizon scan wants when picking a coarser mip for a
/// far step.
pub(in crate::renderer) const HIZ_DOWNSAMPLE_SHADER_WGSL: &str = "
struct Params {
    /// xy = dst-mip size (u32). zw unused.
    size: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = gid.xy;
    if (dst.x >= u.size.x || dst.y >= u.size.y) { return; }
    let src = vec2<i32>(dst * 2u);
    let a = textureLoad(src_tex, src + vec2<i32>(0, 0), 0).r;
    let b = textureLoad(src_tex, src + vec2<i32>(1, 0), 0).r;
    let c = textureLoad(src_tex, src + vec2<i32>(0, 1), 0).r;
    let d = textureLoad(src_tex, src + vec2<i32>(1, 1), 0).r;
    let m = min(min(a, b), min(c, d));
    textureStore(dst_tex, vec2<i32>(dst), vec4<f32>(m, 0.0, 0.0, 0.0));
}
";

/// GTAO compute shader — same 8-dir × 8-step horizon scan +
/// 12-step contact-shadow ray march as ticket 002, but:
///   1. Samples a hierarchical linear-depth pyramid instead of the
///      raw depth buffer. Each scan step `s` picks its mip from
///      `log2(step_pixels)`, so far steps hit progressively coarser
///      (cache-resident) data. This is the win Apple TBDR wants to
///      compensate for the free tile-memory depth cache we lost
///      going compute.
///   2. Linear depth lives in the pyramid directly, so `view_pos`
///      skips the per-sample `-p32 / (depth + p22)` reconstruction.
///   3. Contact-shadow march is gated on `dot(N, light_vs) > 0.1` —
///      back-facing pixels skip the whole 12-step march.
/// Output layout is unchanged: R = noisy AO, G = contact shadow.
pub(in crate::renderer) const SSAO_SHADER_WGSL: &str = "
struct SsaoParams {
    /// xy = inv_size (1/half_w, 1/half_h), z = radius_ws, w = strength
    params: vec4<f32>,
    /// x = proj[0][0], y = proj[1][1], z = proj[2][0], w = proj[2][1]
    proj_row01: vec4<f32>,
    /// x = proj[2][2], y = proj[3][2] (unused here — Hi-Z already
    /// linearized; kept so host-side SsaoParams stays symmetric),
    /// z = 1/proj[0][0], w = 1/proj[1][1] (contact-shadow NDC->UV).
    proj_z: vec4<f32>,
    light_dir_vs: vec4<f32>,
    /// x = half_w, y = half_h, z = frame_phase (0..3), w = first_frames
    /// flag (non-zero → force alpha=1 so history seeds from the current
    /// frame instead of a stale clear).
    size: vec4<u32>,
    /// x = temporal blend alpha (steady state, 4-frame EMA ≈ 0.25);
    /// y = per-frame Halton-5 rotation offset for direction basis
    /// (uncorrelated with TAA's Halton-2/3 pixel jitter so the two
    /// patterns don't resonate); zw unused.
    temporal: vec4<f32>,
};

const HIZ_SKY_Z: f32 = 10000.0;

@group(0) @binding(0) var<uniform> u: SsaoParams;
@group(0) @binding(1) var ao_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var hiz_samp: sampler;
@group(0) @binding(3) var hiz0: texture_2d<f32>;
@group(0) @binding(4) var hiz1: texture_2d<f32>;
@group(0) @binding(5) var hiz2: texture_2d<f32>;
@group(0) @binding(6) var hiz3: texture_2d<f32>;
@group(0) @binding(7) var hiz4: texture_2d<f32>;
@group(0) @binding(8) var velocity_tex: texture_2d<f32>;
@group(0) @binding(9) var history_in: texture_2d<f32>;
@group(0) @binding(10) var filt_samp: sampler;
@group(0) @binding(11) var history_out: texture_storage_2d<rgba8unorm, write>;

// Temporal accumulation fan-out: every frame scans 2 of 8 directions;
// 4 frames rotate through all 8. The 4-frame EMA folded in history
// reconstructs the full 8-dir × 8-step signal at steady state for a
// nominal 4× perf win on the GTAO pass itself.
const N_DIRS_TOTAL: u32 = 8u;
const N_DIRS_PER_FRAME: u32 = 2u;
const N_PHASES: u32 = 4u;
const N_STEPS: u32 = 8u;
const PI: f32 = 3.14159265;
const HIZ_MAX_MIP: i32 = 4;


fn hiz_sample(uv: vec2<f32>, mip: i32) -> f32 {
    switch (clamp(mip, 0, HIZ_MAX_MIP)) {
        case 0: { return textureSampleLevel(hiz0, hiz_samp, uv, 0.0).r; }
        case 1: { return textureSampleLevel(hiz1, hiz_samp, uv, 0.0).r; }
        case 2: { return textureSampleLevel(hiz2, hiz_samp, uv, 0.0).r; }
        case 3: { return textureSampleLevel(hiz3, hiz_samp, uv, 0.0).r; }
        default: { return textureSampleLevel(hiz4, hiz_samp, uv, 0.0).r; }
    }
}

fn view_pos_from_linear(uv: vec2<f32>, linear_z: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;
    let view_z = -linear_z;
    let view_x = -(ndc_x + p20) * view_z / p00;
    let view_y = -(ndc_y + p21) * view_z / p11;
    return vec3<f32>(view_x, view_y, view_z);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.xy;
    let in_bounds = px.x < u.size.x && px.y < u.size.y;

    let inv_sz = u.params.xy;
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) * inv_sz;

    if (!in_bounds) { return; }

    let center_z = hiz_sample(uv, 0);
    // Sky early-out: write full visibility (no AO, no shadow) both
    // to the bilateral-blur input and to history. Seeding history
    // with 1.0 keeps next-frame's 4-tap cross clamp well-behaved
    // where sky neighbours geometry.
    if (center_z >= HIZ_SKY_Z * 0.5) {
        textureStore(ao_out, vec2<i32>(px), vec4<f32>(1.0, 1.0, 0.0, 1.0));
        textureStore(history_out, vec2<i32>(px), vec4<f32>(1.0, 1.0, 0.0, 1.0));
        return;
    }

    let P = view_pos_from_linear(uv, center_z);
    let uv_r = uv + vec2<f32>(inv_sz.x, 0.0);
    let uv_u = uv + vec2<f32>(0.0, -inv_sz.y);
    let P_r = view_pos_from_linear(uv_r, hiz_sample(uv_r, 0));
    let P_u = view_pos_from_linear(uv_u, hiz_sample(uv_u, 0));
    let N = normalize(cross(P_r - P, P_u - P));

    let coord = vec2<f32>(px);
    let ign = fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y));
    // Per-pixel IGN rotation plus a per-frame Halton-5 rotation of
    // the direction basis. Halton base 5 is uncorrelated with TAA's
    // Halton 2/3 pixel jitter so the two noise patterns don't
    // resonate into a visible crawling artefact.
    let jitter_angle = ign * PI + u.temporal.y * PI;

    let radius_ws = u.params.z;
    let strength = u.params.w;

    let proj_scale_x = abs(u.proj_row01.x);
    let proj_scale_y = abs(u.proj_row01.y);
    let screen_radius = radius_ws * 0.5 * (proj_scale_x + proj_scale_y) / abs(P.z);
    let clamped_radius = clamp(screen_radius, 2.0 * max(inv_sz.x, inv_sz.y), 0.25);

    var ao_sum = 0.0;
    let step_size = clamped_radius / f32(N_STEPS);
    let inv_radius = 1.0 / radius_ws;

    // Mip-per-step: step s covers ~s * step_size * mip0_w pixels;
    // take floor(log2(that)). Precomputed once per pixel.
    let step_pixels = step_size * f32(u.size.x);
    var mip_per_step: array<i32, 8>;
    for (var s = 0u; s < N_STEPS; s = s + 1u) {
        let d_px = step_pixels * f32(s + 1u);
        mip_per_step[s] = clamp(i32(floor(log2(max(d_px, 1.0)))), 0, HIZ_MAX_MIP);
    }

    // Split the 8-direction scan across 4 frames. Frame `phase = k`
    // samples directions { phase, phase + 4 } — complementary angles
    // in the 0..π range so every frame covers roughly the full
    // horizon and history doesn't bias toward one hemisphere between
    // refreshes. Over 4 frames the full 8-direction set is sampled.
    let phase = u.size.z;
    for (var k = 0u; k < N_DIRS_PER_FRAME; k = k + 1u) {
        let d = phase + k * N_PHASES;
        let angle = (f32(d) / f32(N_DIRS_TOTAL)) * PI + jitter_angle;
        let dir = vec2<f32>(cos(angle), sin(angle));

        var max_horizon_pos = -1.0;
        var max_horizon_neg = -1.0;
        var pos_on_sky = false;
        var neg_on_sky = false;

        for (var s = 1u; s <= N_STEPS; s = s + 1u) {
            let offset = dir * step_size * f32(s);
            let mip = mip_per_step[s - 1u];

            if (!pos_on_sky) {
                let uv_pos = uv + offset;
                let z_pos = hiz_sample(uv_pos, mip);
                if (z_pos >= HIZ_SKY_Z * 0.5) {
                    pos_on_sky = true;
                } else {
                    let S_pos = view_pos_from_linear(uv_pos, z_pos);
                    let diff_pos = S_pos - P;
                    let dist_pos = length(diff_pos);
                    if (dist_pos > 0.001) {
                        let h_pos = dot(diff_pos, N) / dist_pos;
                        let atten = saturate(1.0 - dist_pos * inv_radius);
                        max_horizon_pos = max(max_horizon_pos, mix(-1.0, h_pos, atten));
                    }
                }
            }

            if (!neg_on_sky) {
                let uv_neg = uv - offset;
                let z_neg = hiz_sample(uv_neg, mip);
                if (z_neg >= HIZ_SKY_Z * 0.5) {
                    neg_on_sky = true;
                } else {
                    let S_neg = view_pos_from_linear(uv_neg, z_neg);
                    let diff_neg = S_neg - P;
                    let dist_neg = length(diff_neg);
                    if (dist_neg > 0.001) {
                        let h_neg = dot(diff_neg, N) / dist_neg;
                        let atten = saturate(1.0 - dist_neg * inv_radius);
                        max_horizon_neg = max(max_horizon_neg, mix(-1.0, h_neg, atten));
                    }
                }
            }

            if (pos_on_sky && neg_on_sky) { break; }
        }

        let vis_pos = 1.0 - saturate(max_horizon_pos);
        let vis_neg = 1.0 - saturate(max_horizon_neg);
        ao_sum = ao_sum + (vis_pos + vis_neg) * 0.5;
    }

    // Per-frame partial AO: normalise by the N_DIRS_PER_FRAME we
    // actually scanned. Temporal blend below reconstructs the full
    // 8-direction signal via the 4-frame EMA.
    let ao_raw = ao_sum / f32(N_DIRS_PER_FRAME);

    // --- Screen-space contact shadows ---
    // Gate on surface/light orientation — a back-facing pixel has no
    // contact shadow to find, skip the 12-step march entirely.
    // Contact shadow runs at its full 12-step count every frame (no
    // temporal accumulation) because the directional light can change
    // between frames and stale history here would trail behind.
    let light_vs = normalize(u.light_dir_vs.xyz);
    var contact = 1.0;
    let n_dot_l = dot(N, light_vs);
    if (n_dot_l > 0.1) {
        let cs_steps = 12u;
        let cs_max_dist = 0.2;
        let cs_step = cs_max_dist / f32(cs_steps);
        let inv_p00 = u.proj_z.z;
        let inv_p11 = u.proj_z.w;
        for (var i = 1u; i <= cs_steps; i = i + 1u) {
            let march_pos = P + light_vs * cs_step * f32(i);
            let ndc_x = march_pos.x / ((-march_pos.z) * inv_p00);
            let ndc_y = march_pos.y / ((-march_pos.z) * inv_p11);
            let march_uv = vec2<f32>(ndc_x * 0.5 + 0.5, 1.0 - (ndc_y * 0.5 + 0.5));
            if (march_uv.x < 0.0 || march_uv.x > 1.0 || march_uv.y < 0.0 || march_uv.y > 1.0) {
                continue;
            }
            let march_z = hiz_sample(march_uv, 0);
            if (march_z >= HIZ_SKY_Z * 0.5) { continue; }
            let scene_pos = view_pos_from_linear(march_uv, march_z);
            if (scene_pos.z > march_pos.z + 0.01) {
                let t = f32(i) / f32(cs_steps);
                contact = min(contact, t);
            }
        }
    }

    // --- Temporal accumulation ---
    // Reproject previous-frame history via the velocity buffer. Use
    // `textureLoad` (nearest, no sampler — velocity's bind-group
    // entry is filterable:false). With TSR on (the default since
    // ticket 001), the velocity RT is created at `render_extent()`
    // = half-res, matching the SSAO pass dimensions 1:1 — so the
    // integer coord is just `px`, NOT `px*2`. The earlier `px*2`
    // read velocity at 2× offset, which sent 75% of SSAO pixels
    // out of bounds (returns zero → history reprojected from the
    // same screen pixel even during camera motion → stale geometry
    // blended in over 4 frames → scene-wide darkening while
    // turning + per-pixel floor sparkle from stale-history noise).
    let vel = textureLoad(velocity_tex, vec2<i32>(px), 0).rg;
    let prev_uv = vec2<f32>(uv.x - vel.x, uv.y + vel.y);
    let reproj_oob = prev_uv.x < 0.0 || prev_uv.x > 1.0
                  || prev_uv.y < 0.0 || prev_uv.y > 1.0;

    var history_ao = ao_raw;
    if (!reproj_oob) {
        history_ao = textureSampleLevel(history_in, filt_samp, prev_uv, 0.0).r;
    }

    // Disocclusion protection via AO delta: if the reprojected
    // history deviates from the current raw sample by more than
    // 0.35 we treat it as a hard break and refresh to `ao_raw`.
    // Cheaper than a spatial neighborhood clamp and sufficient
    // given the 4-frame EMA absorbs short-term drift.
    let ao_delta = abs(ao_raw - history_ao);
    let force_refresh = reproj_oob || u.size.w != 0u || ao_delta > 0.35;
    let alpha = select(u.temporal.x, 1.0, force_refresh);
    let ao = mix(history_ao, ao_raw, alpha);

    // Contrast + floor (exact curve preserved).
    let ao_contrasted = pow(ao, 2.0);
    let ao_floored = max(ao_contrasted, 0.15);
    let final_ao = mix(1.0, ao_floored, strength);
    let contact_scaled = mix(0.1, 1.0, contact);

    // Write the blended *pre-contrast* AO to history so next frame's
    // blend stays in linear AO space (otherwise `pow(pow(x,2),2)` on
    // every frame collapses toward black). The bilateral-blur input
    // (ao_out) gets the contrasted + strength-modulated value so the
    // composite pass stays visually identical to pre-temporal SSAO.
    textureStore(
        ao_out,
        vec2<i32>(px),
        vec4<f32>(saturate(final_ao), saturate(contact_scaled), 0.0, 1.0),
    );
    textureStore(
        history_out,
        vec2<i32>(px),
        vec4<f32>(saturate(ao), saturate(contact_scaled), 0.0, 1.0),
    );
}
";

/// Bilateral blur applied to the raw GTAO output.
///
/// A 5×5 cross-bilateral filter: for each tap we weight by a spatial
/// Gaussian AND by depth similarity so the blur stops at depth edges,
/// preserving contact-shadow / crease detail while suppressing the
/// per-pixel noise introduced by the horizon-sampling in GTAO.
///
/// Bindings:
///   0 – uniform  (SsaoBlurParams)
///   1 – ssao_rt  (Rg8Unorm: R = noisy GTAO, G = contact shadow)
///   2 – ao sampler (filtering)
///   3 – depth_tex (Depth32Float for edge-stopping)
///   4 – depth sampler (non-filtering)
pub(in crate::renderer) const SSAO_BLUR_SHADER_WGSL: &str = "
struct SsaoBlurParams {
    // xy = texel_size (1/w, 1/h of the SSAO RT, i.e. half-res)
    // z  = depth_sigma (edge-stop threshold, ~0.01–0.1 in NDC depth)
    // w  = unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsaoBlurParams;
@group(0) @binding(1) var ao_tex:    texture_2d<f32>;
@group(0) @binding(2) var ao_samp:   sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_samp: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Pre-computed Gaussian weights for a 5-tap 1-D kernel (sigma ≈ 1.4).
// Offsets: -2, -1, 0, +1, +2
const GAUSS5: array<f32, 5> = array<f32, 5>(
    0.0625, 0.25, 0.375, 0.25, 0.0625
);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel  = u.params.xy;
    let d_sigma = u.params.z;

    let center_depth = textureSample(depth_tex, depth_samp, in.uv);

    // Sky pixels: no occlusion, no contact shadow.
    if (center_depth >= 0.9999) {
        return vec4<f32>(1.0, 1.0, 0.0, 1.0);
    }

    var ao_sum     = 0.0;
    var weight_sum = 0.0;

    // 5×5 separable-style bilateral gather on the AO (R) channel
    // only. Contact shadows (G) pass through from the center tap
    // untouched — they're a sharp binary ray-march result and any
    // smoothing here erases the fine-detail shadows the pass is
    // there to capture.
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            let s_uv   = in.uv + offset;

            let s_ao    = textureSample(ao_tex, ao_samp, s_uv).r;
            let s_depth = textureSample(depth_tex, depth_samp, s_uv);

            let gx = GAUSS5[dx + 2];
            let gy = GAUSS5[dy + 2];
            let spatial = gx * gy;

            let depth_diff = abs(center_depth - s_depth);
            let range_w = exp(-depth_diff / d_sigma);

            let w = spatial * range_w;
            ao_sum     += s_ao * w;
            weight_sum += w;
        }
    }

    let center = textureSample(ao_tex, ao_samp, in.uv);
    let ao_blurred = select(center.r, ao_sum / weight_sum, weight_sum > 0.0001);
    return vec4<f32>(ao_blurred, center.g, 0.0, 1.0);
}
";

