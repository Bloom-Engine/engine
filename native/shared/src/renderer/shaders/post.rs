//! Post-processing: bloom, DoF, motion blur, SSS, compose, TAA, exposure, composite, upscale, RCAS.
//! Split from renderer/shaders.rs.

/// Bloom mip-chain shader. Three fragment entry points share a
/// single vertex stage and uniform layout:
///
/// - `fs_downsample`: 4-tap box-filter downsample for mips ≥ 1.
/// - `fs_threshold_downsample`: same downsample but applies a
///   Karis-style soft threshold first to extract HDR brights and
///   suppress fireflies. Used only when sampling the source HDR
///   into bloom mip 0.
/// - `fs_upsample`: 9-tap tent-filter upsample, additive blend
///   (set via wgpu's blend state on the upsample pipeline).
///
/// Uniform: `params.xy` = source texel size (1/src_w, 1/src_h);
/// `params.z` = bloom intensity (only used by upsample); `params.w`
/// reserved.
pub(in crate::renderer) const BLOOM_SHADER_WGSL: &str = "
struct BloomParams {
    /// xy = source texel size (1/src_w, 1/src_h),
    /// z = filter radius (upsample tent),
    /// w = HDR threshold (downsample-threshold variant only).
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: BloomParams;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_samp: sampler;

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

fn karis_average(c: vec4<f32>) -> vec4<f32> {
    // 1 / (luma + 1) weighting to suppress fireflies — heavy bright
    // pixels get downweighted so a single hot texel can't dominate
    // a bloom kernel and create visible specks.
    let luma = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let weight = 1.0 / (1.0 + luma);
    return c * weight;
}

// Soft HDR threshold (UE-style). Pixels with luminance below
// `threshold - knee` get zero contribution; above `threshold + knee`
// pass through fully; in-between blends smoothly. Without this,
// bloom would brighten EVERY pixel in the scene rather than just
// the visibly-overbright ones (sun glare, emissive accents, etc.).
//
// The min() clamps the input to a large finite value — prevents
// Inf/NaN from PBR edge cases (grazing-angle divides in the scene
// shader, specular hotspots hitting Rgba16Float's max) from
// propagating through bloom and poisoning every downstream pass.
fn extract_brights(c_in: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    // NaN-safe clamp: max(.,0) first (flushes NaN→0 on most
    // platforms), then cap at a large finite to avoid Inf.
    let c = min(max(c_in, vec3<f32>(0.0)), vec3<f32>(64000.0));
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let lower = max(threshold - knee, 0.0);
    let upper = threshold + knee;
    let factor = smoothstep(lower, upper, luma);
    return c * factor;
}

// 13-tap downsample (Sledgehammer / Karis 2013). Takes 5 dual-2x2
// box samples + 4 cross samples around each fragment for a smoother
// reduction than a naive 4-tap.
// Sanitize a sample: force NaN→0 via max-with-0, cap Inf via min.
// Needed because the HDR source can contain Inf from PBR edge cases
// (grazing-angle specular, division-by-zero-ish terms). Without
// this, one bad texel poisons the whole downsample kernel.
fn sanitize(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(min(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(64000.0)), c.a);
}

fn downsample_13(uv: vec2<f32>, src_size: vec2<f32>, do_threshold: bool) -> vec3<f32> {
    let dx = src_size.x;
    let dy = src_size.y;

    let a = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>(-2.0 * dx, -2.0 * dy)));
    let b = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>( 0.0,       -2.0 * dy)));
    let c = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>( 2.0 * dx, -2.0 * dy)));
    let d = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>(-2.0 * dx,  0.0)));
    let e = sanitize(textureSample(src_tex, src_samp, uv));
    let f = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>( 2.0 * dx,  0.0)));
    let g = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>(-2.0 * dx,  2.0 * dy)));
    let h = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>( 0.0,        2.0 * dy)));
    let i = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>( 2.0 * dx,  2.0 * dy)));
    let j = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>(-1.0 * dx, -1.0 * dy)));
    let k = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>( 1.0 * dx, -1.0 * dy)));
    let l = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>(-1.0 * dx,  1.0 * dy)));
    let m = sanitize(textureSample(src_tex, src_samp, uv + vec2<f32>( 1.0 * dx,  1.0 * dy)));

    // Five 2x2 boxes weighted to eliminate aliasing.
    var groups: array<vec4<f32>, 5>;
    groups[0] = (a + b + d + e) * 0.25;
    groups[1] = (b + c + e + f) * 0.25;
    groups[2] = (d + e + g + h) * 0.25;
    groups[3] = (e + f + h + i) * 0.25;
    groups[4] = (j + k + l + m) * 0.25;

    if (do_threshold) {
        // First extract HDR brights via soft threshold, then Karis
        // weight to keep fireflies from poking through.
        //
        // threshold = 2.5, knee = 0.5 (fade-in band 2..3). Raised
        // slightly from the original 1.5/0.3 because Sponza uses
        // setManualExposure(1.0) — the comment below was written
        // for auto-exposure normalised HDR (mid-gray at 0.18).
        // The 'glitter on the floor' bug that briefly pushed this
        // to 8.0 turned out to be the irradiance-convolution
        // firefly leak (see fix_ibl commit), not bloom at all, so
        // the aggressive threshold is no longer needed. 2.5 leaves
        // diffuse sunlit stone (luma 2-3) right at the knee —
        // barely blooming — while sky / emissive / specular
        // peaks still get a proper halo.
        let thr = 2.5;
        let knee = 0.5;
        for (var n = 0u; n < 5u; n = n + 1u) {
            let bright = extract_brights(groups[n].rgb, thr, knee);
            let weighted = karis_average(vec4<f32>(bright, 1.0));
            groups[n] = weighted;
        }
    }

    let weights = array<f32, 5>(0.125, 0.125, 0.125, 0.125, 0.5);
    var sum = vec4<f32>(0.0);
    for (var n = 0u; n < 5u; n = n + 1u) {
        sum = sum + groups[n] * weights[n];
    }
    return sum.rgb;
}

@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample_13(in.uv, u.params.xy, false), 1.0);
}

@fragment
fn fs_threshold_downsample(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample_13(in.uv, u.params.xy, true), 1.0);
}

// 9-tap tent filter upsample (Sledgehammer). Texel-radius scaled by
// the small radius factor in u.params.z (defaults to ~1.0 — wider
// = more blurry overlap). Output is BLENDED additively into the
// destination via the upsample pipeline's blend state.
@fragment
fn fs_upsample(in: VsOut) -> @location(0) vec4<f32> {
    let dx = u.params.x * u.params.z;
    let dy = u.params.y * u.params.z;
    let uv = in.uv;

    var sum = vec3<f32>(0.0);
    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>(-dx,  dy)).rgb * 1.0;
    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>( 0.0,  dy)).rgb * 2.0;
    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>( dx,  dy)).rgb * 1.0;

    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>(-dx,  0.0)).rgb * 2.0;
    sum = sum + textureSample(src_tex, src_samp, uv).rgb                          * 4.0;
    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>( dx,  0.0)).rgb * 2.0;

    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>(-dx, -dy)).rgb * 1.0;
    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>( 0.0, -dy)).rgb * 2.0;
    sum = sum + textureSample(src_tex, src_samp, uv + vec2<f32>( dx, -dy)).rgb * 1.0;

    return vec4<f32>(sum * (1.0 / 16.0), 1.0);
}
";

pub(in crate::renderer) const DOF_SHADER_WGSL: &str = "
struct DofParams {
    params: vec4<f32>,  // x = focus_distance (view-space Z, positive), y = aperture (CoC scale), z = max_blur_radius (UV), w = unused
    inv_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: DofParams;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_samp: sampler;
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

// Reconstruct view-space Z from depth buffer value via inverse projection.
fn linearize_depth(depth: f32) -> f32 {
    let ndc = vec4<f32>(0.0, 0.0, depth, 1.0);
    let vp = u.inv_proj * ndc;
    return -vp.z / vp.w; // positive distance from camera
}

// 16-sample Poisson disc (same offsets used by shadow PCF).
const POISSON_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790),
);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let focus_dist = u.params.x;
    let aperture   = u.params.y;
    let max_blur   = u.params.z;

    let center_depth_raw = textureSample(depth_tex, depth_samp, in.uv);

    // Sky pixels (depth ~1.0) get maximum blur — they are at infinity.
    var view_z: f32;
    if (center_depth_raw >= 0.9999) {
        view_z = 1000.0; // treat as very far
    } else {
        view_z = linearize_depth(center_depth_raw);
    }

    // Circle of confusion: thin-lens approximation.
    // Dividing by view_z ensures distant objects don't get disproportionately
    // blurred — CoC grows with defocus distance but falls off with depth.
    // max(view_z, 0.1) prevents division by zero for geometry very close to
    // the camera.
    let coc = clamp(aperture * abs(view_z - focus_dist) / max(view_z, 0.1), 0.0, max_blur);

    // If CoC is negligibly small, return the source pixel unchanged.
    let threshold = 0.0005;
    if (coc < threshold) {
        return textureSample(color_tex, color_samp, in.uv);
    }

    // Gather 16 Poisson disc samples scaled by CoC.
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    let center_color = textureSample(color_tex, color_samp, in.uv).rgb;

    for (var i = 0u; i < 16u; i = i + 1u) {
        let offset = POISSON_16[i] * coc;
        let sample_uv = in.uv + offset;

        let sample_color = textureSample(color_tex, color_samp, sample_uv).rgb;

        // Read the depth at the sample location to compute its own CoC.
        // This prevents sharp foreground objects from bleeding into
        // blurred background — only samples that are themselves blurry
        // (or at least as blurry as this pixel) contribute fully.
        let sample_depth_raw = textureSample(depth_tex, depth_samp, sample_uv);
        var sample_z: f32;
        if (sample_depth_raw >= 0.9999) {
            sample_z = 1000.0;
        } else {
            sample_z = linearize_depth(sample_depth_raw);
        }
        let sample_coc = clamp(abs(sample_z - focus_dist) * aperture, 0.0, max_blur);

        // Weight: accept the sample if its CoC is at least as large as
        // the center pixel's CoC, or if the sample is behind the center
        // (background blurring into foreground is expected). Otherwise
        // attenuate by the ratio of sample_coc / coc.
        var w = 1.0;
        if (sample_z < view_z) {
            // Sample is in front of center — only contribute if it is
            // itself blurry enough.
            w = saturate(sample_coc / coc);
        }

        color_sum += sample_color * w;
        weight_sum += w;
    }

    // Also blend in the center pixel with weight 1.
    color_sum += center_color;
    weight_sum += 1.0;

    let result = color_sum / weight_sum;
    return vec4<f32>(result, 1.0);
}
";

pub(in crate::renderer) const MOTION_BLUR_SHADER_WGSL: &str = "
struct MotionBlurParams {
    /// x = strength (velocity multiplier), y = max_blur (UV clamp), zw = unused.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: MotionBlurParams;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_samp: sampler;
@group(0) @binding(3) var velocity_tex: texture_2d<f32>;
@group(0) @binding(4) var velocity_samp: sampler;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let strength = u.params.x;
    let max_blur = u.params.y;

    let vel_raw = textureSample(velocity_tex, velocity_samp, in.uv).rg;
    // Scale velocity by strength and clamp to max_blur radius.
    var vel = vel_raw * strength;
    let vel_len = length(vel);
    if (vel_len > max_blur) {
        vel = vel * (max_blur / vel_len);
    }

    // If velocity is negligible, return source pixel unchanged.
    if (vel_len < 0.0001) {
        return textureSample(color_tex, color_samp, in.uv);
    }

    // 8-tap directional blur with tent (linear) weighting.
    // Samples are placed symmetrically around the center pixel
    // along the velocity vector. The tent filter peaks at the
    // center and falls off linearly toward the endpoints.
    let n_samples: i32 = 8;
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var i: i32 = 0; i < n_samples; i = i + 1) {
        let t = (f32(i) + 0.5) / f32(n_samples) - 0.5; // range [-0.5, 0.5)
        let sample_uv = in.uv + vel * t;
        let w = 1.0 - abs(t * 2.0); // tent weight: 1.0 at center, 0 at edges
        color_sum += textureSample(color_tex, color_samp, sample_uv).rgb * w;
        weight_sum += w;
    }

    return vec4<f32>(color_sum / weight_sum, 1.0);
}
";

pub(in crate::renderer) const SSS_SHADER_WGSL: &str = "
struct SssParams {
    /// x = strength (0 = off, 1 = full blend), y = width (screen-space
    /// blur radius in UV units, e.g. 0.01), z = falloff (bilateral
    /// depth edge-stop steepness), w = unused.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SssParams;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_samp: sampler;
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

// 9-tap disc pattern (unit disc, slightly stratified).
// Kept intentionally modest — SSS scatter radius is small.
const DISC_9: array<vec2<f32>, 9> = array<vec2<f32>, 9>(
    vec2<f32>( 0.0,      0.0),
    vec2<f32>( 1.0,      0.0),
    vec2<f32>(-1.0,      0.0),
    vec2<f32>( 0.0,      1.0),
    vec2<f32>( 0.0,     -1.0),
    vec2<f32>( 0.7071,  0.7071),
    vec2<f32>(-0.7071,  0.7071),
    vec2<f32>( 0.7071, -0.7071),
    vec2<f32>(-0.7071, -0.7071),
);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let strength = u.params.x;
    let width    = u.params.y;
    let falloff  = u.params.z;

    let center_color = textureSample(color_tex, color_samp, in.uv);

    // Sky pixels (raw depth ~1.0) skip SSS entirely — they have no
    // geometry to scatter through. This also avoids depth-edge halos
    // at the horizon.
    let center_depth = textureSample(depth_tex, depth_samp, in.uv);
    if (center_depth >= 0.9999) {
        return center_color;
    }

    // Chromatic diffusion profile: red scatters furthest (skin
    // absorbs blue/green more than red). Width multipliers:
    //   red   = 1.0 × width
    //   green = 0.5 × width
    //   blue  = 0.25 × width
    var sum_r = 0.0;
    var sum_g = 0.0;
    var sum_b = 0.0;
    var weight_r = 0.0;
    var weight_g = 0.0;
    var weight_b = 0.0;

    for (var i = 0u; i < 9u; i = i + 1u) {
        let tap_r = in.uv + DISC_9[i] * width;
        let tap_g = in.uv + DISC_9[i] * (width * 0.5);
        let tap_b = in.uv + DISC_9[i] * (width * 0.25);

        // Bilateral depth weight — each channel uses its own tap UV,
        // so we sample depth at each channel's location independently.
        let d_r = textureSample(depth_tex, depth_samp, tap_r);
        let d_g = textureSample(depth_tex, depth_samp, tap_g);
        let d_b = textureSample(depth_tex, depth_samp, tap_b);

        let w_r = exp(-abs(d_r - center_depth) * falloff);
        let w_g = exp(-abs(d_g - center_depth) * falloff);
        let w_b = exp(-abs(d_b - center_depth) * falloff);

        // Spatial Gaussian (unit disc → standard Gaussian weight from
        // the squared distance within the disc).
        let dist2 = dot(DISC_9[i], DISC_9[i]);
        let gauss = exp(-dist2 * 2.0); // sigma ≈ 0.7 in disc-space

        let c_r = textureSample(color_tex, color_samp, tap_r).r;
        let c_g = textureSample(color_tex, color_samp, tap_g).g;
        let c_b = textureSample(color_tex, color_samp, tap_b).b;

        sum_r += c_r * w_r * gauss;
        sum_g += c_g * w_g * gauss;
        sum_b += c_b * w_b * gauss;
        weight_r += w_r * gauss;
        weight_g += w_g * gauss;
        weight_b += w_b * gauss;
    }

    let blurred = vec3<f32>(
        sum_r / max(weight_r, 1e-5),
        sum_g / max(weight_g, 1e-5),
        sum_b / max(weight_b, 1e-5),
    );

    // Blend blurred result with original by strength.
    let result = mix(center_color.rgb, blurred, strength);
    return vec4<f32>(result, center_color.a);
}
";

/// Scene-compose shader. Merges direct scene HDR with every
/// screen-space post effect (SSR, albedo-modulated SSGI, bloom) and
/// then applies volumetric fog + sun shafts. The output is a single
/// "composed HDR" texture that downstream passes consume:
///
///   - TAA-on: the TAA pass reads composed_rt as its current frame
///     and only performs temporal reprojection + neighborhood clamp.
///   - TAA-off: the composite pass reads composed_rt directly.
///
/// This keeps fog / shafts consistent across both TAA states and
/// removes the need for TAA / composite to re-compose the same
/// ingredients separately.
pub(in crate::renderer) const SCENE_COMPOSE_SHADER_WGSL: &str = "
struct SceneComposeParams {
    /// x = bloom intensity; y/z/w padding.
    misc: vec4<f32>,
    /// Inverse of the current-frame view-projection (world-pos reconstruction).
    inv_vp: mat4x4<f32>,
    /// Fog tint (rgb) + density (w).
    fog_color_density: vec4<f32>,
    /// Fog: x = height_ref, y = falloff rate, zw padding.
    fog_params: vec4<f32>,
    /// Sun shafts: xy = projected sun UV, z = strength, w = decay.
    sun_shaft_uv_strength: vec4<f32>,
    /// Sun shaft tint (rgb, w padding).
    sun_shaft_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SceneComposeParams;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var hdr_samp: sampler;
@group(0) @binding(3) var ssr_tex: texture_2d<f32>;
@group(0) @binding(4) var ssr_samp: sampler;
@group(0) @binding(5) var ssgi_tex: texture_2d<f32>;
@group(0) @binding(6) var ssgi_samp: sampler;
@group(0) @binding(7) var bloom_tex: texture_2d<f32>;
@group(0) @binding(8) var bloom_samp: sampler;
@group(0) @binding(9) var albedo_tex: texture_2d<f32>;
@group(0) @binding(10) var albedo_samp: sampler;
@group(0) @binding(11) var depth_tex: texture_depth_2d;
@group(0) @binding(12) var depth_samp: sampler;
@group(0) @binding(13) var aerial_tex: texture_3d<f32>;
@group(0) @binding(14) var aerial_samp: sampler;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Pre-tonemap HDR composition. SSR is already Fresnel/edge faded
    // at its pass; SSGI is raw indirect radiance, multiplied here by
    // the receiver albedo so dark materials absorb correctly. Bloom
    // is scaled by the user-tuned intensity.
    let hdr = textureSample(hdr_tex, hdr_samp, in.uv).rgb;
    let ssr = textureSample(ssr_tex, ssr_samp, in.uv).rgb;
    let ssgi = textureSample(ssgi_tex, ssgi_samp, in.uv).rgb;
    let albedo_sample = textureSample(albedo_tex, albedo_samp, in.uv);
    let albedo = albedo_sample.rgb;
    // albedo.a carries `1 - shadow_factor` from the scene pass — how
    // much of this pixel's illumination is indirect (IBL + bounce) vs.
    // direct (sun). Forwarded through the composed RT's alpha channel
    // so the composite/tonemap pass can apply SSAO only to the
    // indirect-dominated portion. Sky pixels carry 0 here so fog / AO
    // don't touch them.
    let indirect_weight = albedo_sample.a;
    let bloom = textureSample(bloom_tex, bloom_samp, in.uv).rgb;
    var color = hdr + ssr + ssgi * albedo + bloom * u.misc.x;

    // World-space position from depth for fog ray march.
    let depth = textureSample(depth_tex, depth_samp, in.uv);
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_h = u.inv_vp * ndc;
    let world = world_h.xyz / world_h.w;

    // EN-005 V2 — when procedural sky is on (misc.y > 0), sample the
    // pre-baked aerial-perspective 3D LUT instead of running the
    // Beer-Lambert march. The LUT is indexed by (NDC.xy, depth-slice)
    // where depth-slice = world_distance / max_dist_km.
    if (u.misc.y > 0.5) {
        let cam_pos = vec3<f32>(
            u.inv_vp[3][0] / u.inv_vp[3][3],
            u.inv_vp[3][1] / u.inv_vp[3][3],
            u.inv_vp[3][2] / u.inv_vp[3][3],
        );
        // Engine units are metres; LUT covers misc.z km. Skip far-
        // plane / sky pixels: depth == 1.0 means no scene geometry,
        // and the procedural sky pass already drew the right colour
        // for that pixel — fogging it again would double-tint.
        if (depth < 1.0) {
            let dist_m = length(world - cam_pos);
            let dist_km = dist_m * 0.001;
            let max_km = u.misc.z;
            let depth_slice = clamp(dist_km / max_km, 0.0, 1.0);
            let aerial = textureSampleLevel(
                aerial_tex,
                aerial_samp,
                vec3<f32>(in.uv.x, in.uv.y, depth_slice),
                0.0,
            );
            let in_scatter = aerial.rgb;
            let mean_t = aerial.a;
            color = color * mean_t + in_scatter;
        }
    } else {
    let fog_density = u.fog_color_density.w;
    if (fog_density > 0.0) {
        let height_ref = u.fog_params.x;
        let height_falloff = u.fog_params.y;
        let cam_pos = vec3<f32>(
            u.inv_vp[3][0] / u.inv_vp[3][3],
            u.inv_vp[3][1] / u.inv_vp[3][3],
            u.inv_vp[3][2] / u.inv_vp[3][3],
        );
        let ray = world - cam_pos;
        let dist = length(ray);
        let ray_dir = ray / max(dist, 0.001);

        let n_steps = 16u;
        let step_size = dist / f32(n_steps);
        var transmittance = 1.0;
        var in_scatter = vec3<f32>(0.0);
        for (var i = 0u; i < n_steps; i = i + 1u) {
            let t = (f32(i) + 0.5) * step_size;
            let p = cam_pos + ray_dir * t;
            let height_fade = exp(-height_falloff * max(p.y - height_ref, 0.0));
            let local_density = fog_density * height_fade;
            let step_extinction = exp(-local_density * step_size);
            in_scatter += u.fog_color_density.rgb * local_density * step_size * transmittance;
            transmittance *= step_extinction;
        }
        color = color * transmittance + in_scatter;
    }
    }

    // Sun shafts: 32-tap march from the pixel toward the projected
    // sun UV, accumulating sky-visibility with per-sample decay.
    // Strength 0 disables.
    let shaft_strength = u.sun_shaft_uv_strength.z;
    if (shaft_strength > 0.0) {
        let sun_uv = u.sun_shaft_uv_strength.xy;
        let decay = u.sun_shaft_uv_strength.w;
        let n_samples: i32 = 32;
        let delta = (sun_uv - in.uv) / f32(n_samples);
        var pos = in.uv;
        var weight = 1.0;
        var accum = 0.0;
        for (var i: i32 = 0; i < n_samples; i = i + 1) {
            pos = pos + delta;
            if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0) {
                continue;
            }
            let d = textureSample(depth_tex, depth_samp, pos);
            let sky = smoothstep(0.998, 1.0, d);
            accum = accum + sky * weight;
            weight = weight * decay;
        }
        let norm = accum / f32(n_samples);
        color = color + u.sun_shaft_color.rgb * norm * shaft_strength;
    }

    // Compose-wide NaN scrub + defensive luma cap. The actual
    // stone-floor speckle was fixed upstream at the irradiance
    // convolution shader (sun disc was leaking into the 'diffuse'
    // map); this cap stays as a safety net against future
    // over-bright contributors (rare ssr/ssgi anomaly + bloom).
    // 50 is high enough to never clip normal scene brightness.
    let color_clean = select(vec3<f32>(0.0), color, color == color);
    let color_luma = dot(color_clean, vec3<f32>(0.2126, 0.7152, 0.0722));
    let compose_cap = 50.0;
    let color_scale = select(1.0, compose_cap / color_luma, color_luma > compose_cap);
    return vec4<f32>(color_clean * color_scale, indirect_weight);
}
";

/// TAA shader. Reads `composed_rt` (scene HDR + post-effects + fog +
/// shafts already merged upstream) and performs only temporal
/// reprojection with neighborhood clamp, blending against the
/// history RT. For static scenes the blend converges in ~10 frames
/// to a fully sub-pixel-resolved image.
pub(in crate::renderer) const TAA_SHADER_WGSL: &str = "
struct TaaParams {
    /// x = blend factor (current-frame weight), yzw padding.
    params: vec4<f32>,
    /// Inverse of the current-frame view-projection matrix —
    /// reconstructs world-space position for history reprojection.
    inv_vp: mat4x4<f32>,
    /// Previous-frame view-projection — projects world pos into
    /// history UV.
    prev_vp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: TaaParams;
@group(0) @binding(1) var composed_tex: texture_2d<f32>;
@group(0) @binding(2) var composed_samp: sampler;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var history_samp: sampler;
@group(0) @binding(5) var depth_tex: texture_depth_2d;
@group(0) @binding(6) var depth_samp: sampler;
@group(0) @binding(7) var velocity_tex: texture_2d<f32>;
@group(0) @binding(8) var velocity_samp: sampler;

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

// RGB <-> YCoCg conversions. Reversible, linear, cheap (no matrix
// multiply). Used by the TAA neighborhood clamp so we can bound
// history's luma (Y) against the source neighborhood's statistical
// range while leaving chroma (Co, Cg) alone — the per-channel RGB
// clamp was causing chromatic sparkle on grazing-angle stone.
fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    let Co = c.r - c.b;
    let tmp = c.b + Co * 0.5;
    let Cg = c.g - tmp;
    let Y  = tmp + Cg * 0.5;
    return vec3<f32>(Y, Co, Cg);
}
fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let tmp = c.x - c.z * 0.5;
    let g   = c.z + tmp;
    let b   = tmp - c.y * 0.5;
    let r   = c.y + b;
    return vec3<f32>(r, g, b);
}

// 5-tap Catmull-Rom upsample (Karis formulation). When the source
// (composed_tex) is half-res relative to the destination, naive
// bilinear loses sharpness; Catmull-Rom reconstructs a cubic-Hermite
// curve through 4 source taps which preserves edges. Costs 5 bilinear
// fetches vs 1 — worth it for the TSR upscale because the alternative
// is a perceptibly blurrier image.
fn sample_catmull_rom(uv: vec2<f32>) -> vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(composed_tex));
    let inv_size = 1.0 / tex_size;
    let sample_pos = uv * tex_size;
    let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos1;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let offset12 = w2 / w12;

    let tp0 = (tex_pos1 - 1.0) * inv_size;
    let tp3 = (tex_pos1 + 2.0) * inv_size;
    let tp12 = (tex_pos1 + offset12) * inv_size;

    var result = vec4<f32>(0.0);
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp0.y), 0.0) * w12.x * w0.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x, tp12.y), 0.0) * w0.x * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp12.y), 0.0) * w12.x * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x, tp12.y), 0.0) * w3.x * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp3.y), 0.0) * w12.x * w3.y;
    return max(result, vec4<f32>(0.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // composed_tex already carries HDR + SSR + SSGI*albedo + bloom +
    // fog + shafts — TAA only needs to reproject history and blend.
    // Alpha carries `indirect_weight` (see scene_compose) which the
    // composite pass reads to apply AO only to indirect-dominated
    // pixels; pass it through blended with the colour so history
    // stays consistent.
    let current_sample = sample_catmull_rom(in.uv);
    let current = current_sample.rgb;
    let current_w = current_sample.a;

    let depth = textureSample(depth_tex, depth_samp, in.uv);
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_h = u.inv_vp * ndc;
    let world = world_h.xyz / world_h.w;

    let vel = textureSample(velocity_tex, velocity_samp, in.uv).rg;
    let vel_len = length(vel);
    var prev_uv: vec2<f32>;
    if (vel_len > 0.00001) {
        prev_uv = vec2<f32>(in.uv.x - vel.x, in.uv.y + vel.y);
    } else if (depth >= 0.9999) {
        // Sky / far plane: the positional reconstruction divides by a
        // near-zero w and reprojects sky pixels onto arbitrary scene
        // points — the luma-only history clamp then locks that wrong
        // chroma in forever (uniform green/red sky tint). The sky is at
        // infinity, so reproject the view DIRECTION instead: exact under
        // camera rotation, translation-invariant by definition.
        let dir = world_h.xyz; // w ~ 0 at the far plane: xyz IS the direction
        let prev_clip = u.prev_vp * vec4<f32>(dir, 0.0);
        if (prev_clip.w > 0.00001) {
            let prev_ndc = prev_clip.xyz / prev_clip.w;
            prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));
        } else {
            prev_uv = in.uv;
        }
    } else {
        let prev_clip = u.prev_vp * vec4<f32>(world, 1.0);
        let prev_ndc = prev_clip.xyz / prev_clip.w;
        prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));
    }

    var history = current;
    var history_w = current_w;
    if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
        let h_sample = textureSample(history_tex, history_samp, prev_uv);
        history = h_sample.rgb;
        history_w = h_sample.a;
    }

    // Variance clamp in YCoCg (Karis 2014). Per-channel RGB min/max
    // clamping was producing chromatic sparkle on the stone floor
    // at grazing angles: high-frequency normal-map specular makes
    // each jittered frame's Cg/Co vary significantly, and clamping
    // each channel independently lets history's chroma get pinned
    // to whatever the current frame's specific Cg/Co range was.
    // Clamping only the *luma* axis (Y) preserves chroma stability
    // across frames; the 1σ variance range is a statistical clamp
    // that absorbs single-pixel outliers without collapsing to a
    // hard min/max bound.
    let texel = vec2<f32>(1.0 / f32(textureDimensions(composed_tex).x),
                          1.0 / f32(textureDimensions(composed_tex).y));
    let center_rgb = textureSample(composed_tex, composed_samp, in.uv).rgb;
    var m1 = rgb_to_ycocg(center_rgb);
    var m2 = m1 * m1;
    let n_samples = 9.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            if (x == 0 && y == 0) { continue; }
            let s_uv = in.uv + vec2<f32>(f32(x), f32(y)) * texel;
            let s_rgb = textureSample(composed_tex, composed_samp, s_uv).rgb;
            let s = rgb_to_ycocg(s_rgb);
            m1 = m1 + s;
            m2 = m2 + s * s;
        }
    }
    let mean = m1 / n_samples;
    let variance = max(m2 / n_samples - mean * mean, vec3<f32>(0.0));
    let stddev = sqrt(variance);

    // Motion-aware γ + alpha. At rest γ=1.25 lets sub-pixel jitter
    // history through for smooth accumulation. Under any camera
    // motion γ collapses fast to 0.25 — forces reprojected
    // history within a quarter-sigma of the neighborhood mean,
    // which is tight enough to reject the 'dark column in
    // history, bright wall in current' case that the wider band
    // let slip. alpha ramps to 0.85 at the same time so remaining
    // history contributes only 15 %.
    let motion_alpha = smoothstep(0.0005, 0.008, vel_len);
    let gamma = mix(1.25, 0.25, motion_alpha);
    let y_min = mean.x - gamma * stddev.x;
    let y_max = mean.x + gamma * stddev.x;

    let history_ycocg = rgb_to_ycocg(history);
    let history_y_clamped = clamp(history_ycocg.x, y_min, y_max);
    let clamped_history = ycocg_to_rgb(vec3<f32>(history_y_clamped, history_ycocg.yz));

    // Per-pixel disocclusion reject. If the history (already
    // variance-clamped) still sits far from the current
    // neighborhood's center, the reprojection sampled a very
    // different world point and should be dropped. Absolute
    // threshold keyed to stddev so tight-gradient regions
    // reject aggressively; flat regions stay accumulating.
    let history_dist = abs(history_y_clamped - mean.x);
    let disocclusion = smoothstep(stddev.x * 0.25, stddev.x * 1.0, history_dist);

    let motion_ramped = mix(u.params.x, 0.85, motion_alpha);
    let alpha = max(motion_ramped, disocclusion);
    let blended = mix(clamped_history, current, alpha);
    let blended_w = mix(history_w, current_w, alpha);
    return vec4<f32>(blended, blended_w);
}
";

/// Auto-exposure update shader. Runs at 1×1 viewport → single
/// fragment. Samples hdr_rt at a 4×4 grid (16 taps), averages
/// luminance, derives a target exposure via `key / avg_luma`,
/// smooths toward it from last frame's exposure. One fragment's
/// worth of work — way cheaper than having every composite
/// fragment redundantly do the same average.
pub(in crate::renderer) const EXPOSURE_SHADER_WGSL: &str = "
struct ExposureParams {
    /// x = target key value (0.18 = photography 18%-gray).
    /// y = smoothing rate (0 = no adapt, 1 = instant).
    /// z = min exposure clamp (prevents pitch-black scenes from
    ///     exploding to max brightness).
    /// w = max exposure clamp (prevents sun scenes from crushing
    ///     to zero).
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ExposureParams;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var hdr_samp: sampler;
@group(0) @binding(3) var prev_exposure_tex: texture_2d<f32>;
@group(0) @binding(4) var prev_exposure_samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Histogram-based auto-exposure. 1024-tap (32×32) log-luma
    // sampling into 64 bins, then target the 50th-percentile
    // (median) luma. Much more robust than log-average on scenes
    // with small bright outliers (windows, sun, skylights) — the
    // median ignores outliers while the average gets dragged
    // toward them.
    var bins: array<u32, 64>;
    for (var i = 0u; i < 64u; i = i + 1u) { bins[i] = 0u; }

    // Histogram log-luma range: 2^-8 (≈0.004) to 2^6 (64). Covers
    // the common HDR exposure range for natural scenes; values
    // outside get clamped into the edge bins so they still count.
    let log_min = -8.0;
    let log_max =  6.0;
    let log_range = log_max - log_min;

    var total: u32 = 0u;
    let n = 32u;
    for (var y = 0u; y < n; y = y + 1u) {
        for (var x = 0u; x < n; x = x + 1u) {
            let sx = (f32(x) + 0.5) / f32(n);
            let sy = (f32(y) + 0.5) / f32(n);
            let s = textureSample(hdr_tex, hdr_samp, vec2<f32>(sx, sy)).rgb;
            let luma = max(dot(s, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-4);
            let lg = log2(luma);
            let t = clamp((lg - log_min) / log_range, 0.0, 0.9999);
            let bin = u32(t * 64.0);
            bins[bin] = bins[bin] + 1u;
            total = total + 1u;
        }
    }

    // Find the bin whose cumulative-below-count passes 50% of total.
    let target_count = total / 2u;
    var accum: u32 = 0u;
    var median_bin: u32 = 32u;
    for (var i = 0u; i < 64u; i = i + 1u) {
        accum = accum + bins[i];
        if (accum >= target_count) {
            median_bin = i;
            break;
        }
    }
    let median_log = log_min + (f32(median_bin) + 0.5) / 64.0 * log_range;
    let median_luma = exp2(median_log);

    let key = u.params.x;
    let rate = u.params.y;
    let min_e = u.params.z;
    let max_e = u.params.w;

    let target_exp = clamp(key / max(median_luma, 0.01), min_e, max_e);
    let prev = textureSample(prev_exposure_tex, prev_exposure_samp, vec2<f32>(0.5, 0.5)).r;
    // First frame: prev is 0; snap to target instead of crawling up.
    var smoothed = mix(prev, target_exp, rate);
    if (prev < min_e * 0.5) {
        smoothed = target_exp;
    }
    return vec4<f32>(smoothed, 0.0, 0.0, 1.0);
}
";

/// Composite + tonemap fragment shader. Single fullscreen triangle
/// reads hdr_rt and writes ACES-tonemapped linear-RGB. Hardware
/// performs the linear→sRGB encode on write because the surface
/// format is sRGB.
pub(in crate::renderer) const COMPOSITE_SHADER_WGSL: &str = "
struct CompositeParams {
    /// x = tonemap mode (0 = ACES, 1 = AgX)
    /// y = auto-exposure enabled (0 = off, uses manual x)
    /// z = manual exposure multiplier (used when auto is off)
    /// w = auto-exposure target key value (0.18 = 18% gray photo standard)
    params: vec4<f32>,
    /// Filmic-look knobs — all default to 0 (effect off).
    /// x = chromatic aberration strength (0..~0.01 radial UV offset)
    /// y = vignette strength (0..1, darkens corners)
    /// z = vignette softness (0..1, smaller = harder edge)
    /// w = film grain strength (0..~0.1 amplitude added to luma)
    filmic: vec4<f32>,
    /// x = grain seed (frame index, randomizes the noise per frame);
    /// y = sharpen strength (0 = off, ~0.25 subtle, ~0.5 punchy);
    /// zw padding.
    misc: vec4<f32>,
};

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;
@group(0) @binding(2) var<uniform> u: CompositeParams;
@group(0) @binding(3) var exposure_tex: texture_2d<f32>;
@group(0) @binding(4) var exposure_samp: sampler;
@group(0) @binding(5) var ssao_tex: texture_2d<f32>;
@group(0) @binding(6) var ssao_samp: sampler;

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

fn aces_tone(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let cc = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((c * (c * a + b)) / (c * (c * cc + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- AgX tonemap (Blender/Filament reference) ---
// Better hue preservation than ACES in saturated regions — reds
// stay red instead of shifting toward orange, blues stay blue
// instead of shifting toward cyan. Same sigmoid shape overall,
// so the overall contrast is similar.

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return vec3<f32>(15.5) * x4 * x2
         - vec3<f32>(40.14) * x4 * x
         + vec3<f32>(31.96) * x4
         - vec3<f32>(6.868) * x2 * x
         + vec3<f32>(0.4298) * x2
         + vec3<f32>(0.1191) * x
         - vec3<f32>(0.00232);
}

fn agx_tone(val_in: vec3<f32>) -> vec3<f32> {
    // AgX input transform — compresses the input color gamut.
    let agx_mat = mat3x3<f32>(
        vec3<f32>(0.842479062253094,  0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772,  0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    // Log2-space normalization range. Anything outside gets clamped
    // — the sigmoid maps this window to [0, 1].
    let min_ev = -12.47393;
    let max_ev = 4.026069;

    var val = agx_mat * val_in;
    // Log2 encode, clamp to range, normalize to [0, 1].
    val = max(val, vec3<f32>(1e-10));
    val = clamp(log2(val), vec3<f32>(min_ev), vec3<f32>(max_ev));
    val = (val - vec3<f32>(min_ev)) / (max_ev - min_ev);

    // Sigmoid contrast curve.
    val = agx_default_contrast_approx(val);
    return val;
}

fn agx_eotf(val_in: vec3<f32>) -> vec3<f32> {
    // AgX inverse input transform — re-expands back to target
    // display gamut. The surface is sRGB-format so hardware
    // applies the sRGB EOTF on write; we output linear here.
    let agx_mat_inv = mat3x3<f32>(
        vec3<f32>( 1.19687900512017,   -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368,  1.15190312990417,   -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433,  1.15107367264116),
    );
    return agx_mat_inv * val_in;
}

// Tonemap — branches on u.params.x (0 = ACES, 1 = AgX). Extracted
// so the sharpen pass can tonemap neighbour HDR samples through the
// same path as the center pixel.
fn tonemap_select(hdr: vec3<f32>) -> vec3<f32> {
    if (u.params.x < 0.5) {
        return aces_tone(hdr);
    }
    // Full Filament AgX pipeline:
    //   agx_tone  (inset + log2 + sigmoid)
    //   agx_eotf  (outset = inverse inset)
    //   pow(2.2)  (display-encoded → linear, required for sRGB surface)
    //
    // Filament's polynomial-approximation AgX (what we use) systematically
    // under-saturates vs Blender/Cycles' LUT-based AgX. Apply the 'Punchy'
    // look — saturation 1.4, slight contrast — as a post step to bring
    // colours closer to the Cycles ground-truth reference. Matches
    // Filament::ToneMapper::AGX_PUNCHY.
    let hdr_safe = max(hdr, vec3<f32>(0.0));
    let agx_display = clamp(agx_eotf(agx_tone(hdr_safe)), vec3<f32>(0.0), vec3<f32>(1.0));
    let linear = pow(agx_display, vec3<f32>(2.2));
    // Punchy look: post-tonemap saturation + contrast in display space.
    // slope/offset/power are per-channel lift/gamma/gain; saturation
    // multiplies chroma around the pixel's luma.
    let agx_punchy = agx_look_punchy(linear);
    return agx_punchy;
}

fn agx_look_punchy(val: vec3<f32>) -> vec3<f32> {
    // Per-channel slope (gain): subtle warm white-balance toward the
    // 'golden hour' feel of Cycles-AgX renders — +3% red, -2% blue.
    // The Bistro outdoor.hdr leans cool; without this, the shadows
    // and IBL fill pull the overall image toward blue-grey vs.
    // Cycles's warmer midtone neutral.
    let slope = vec3<f32>(1.03, 1.00, 0.98);
    let offset = vec3<f32>(0.0);
    // A whisker above neutral — Cycles-AgX with its LUT-based DRT
    // preserves enough chroma on its own that our polynomial fit only
    // needs a gentle post-boost to catch up.
    let power = vec3<f32>(1.1);
    let saturation = 1.1;
    // ASC-CDL-ish: (val * slope + offset) ^ power
    let toned = pow(max(val * slope + offset, vec3<f32>(0.0)), power);
    let luma = dot(toned, vec3<f32>(0.2126, 0.7152, 0.0722));
    return vec3<f32>(luma) + saturation * (toned - vec3<f32>(luma));
}

// Hash-based pseudo-random in [0, 1). Cheap noise function for grain;
// not great for cryptography or stratified sampling, but visually
// indistinguishable from white noise at film-grain strengths.
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Composite samples the TAA-blended HDR. The TAA pass has
    // already combined HDR + SSAO + bloom + SSR + fog into one
    // linear-HDR value, so all that's left is exposure, tonemap,
    // and the optional filmic-look layer (CA / vignette / grain).
    var sample_uv = in.uv;

    // Sample the composed HDR at the centre pixel. Chromatic
    // aberration used to sample here at 3 offsets and take R/G/B
    // from different UVs, but doing that on pre-tonemap HDR
    // (values 1-20) turns any sub-pixel specular highlight into
    // a magenta/green speckle: the R tap lands on a bright pixel,
    // the B tap on a dark one, and the colour ratio survives
    // tonemap as visible noise all over the frame. CA was moved
    // after tonemap below, where the values are bounded to [0,1]
    // and the offset reads a colour difference of a few percent
    // instead of a 10× ratio.
    let centre_sample = textureSample(hdr_tex, hdr_samp, sample_uv);
    let indirect_weight = centre_sample.a;

    // Last-chance NaN/Inf scrub before tonemap. The main HDR pass
    // already self-compares its output, but compose mixes SSR + SSGI
    // + bloom + fog, any of which can introduce a non-finite value
    // (degenerate view rays, env UV pole, multi-scatter fresnel
    // divide). With TAA off, no neighborhood clamp exists upstream
    // to catch it — and tonemap(NaN) on Metal produces pink.
    let hdr_raw = select(vec3<f32>(0.0), centre_sample.rgb, centre_sample.rgb == centre_sample.rgb);

    // SSAO RT packs AO in R and contact shadow in G — multiplying
    // both gives combined darkening. The AO channel is bilaterally
    // blurred; G is the raw pixel-accurate contact result.
    //
    // Apply AO only in proportion to how INDIRECT this pixel's light
    // is. indirect_weight = 1 means fully shadowed (indirect lighting
    // only — full AO applies), weight = 0 means fully sunlit (direct
    // lighting dominates — AO shouldn't darken it). That's physically
    // correct: AO models the fact that nearby geometry occludes
    // ambient/bounce light, but it has nothing to say about a direct
    // ray from the sun, which the shadow map already handles. Sky
    // pixels get indirect_weight = 0 via the scene_compose pass (sky
    // albedo = 0) so AO also leaves them alone.
    let ao_pair = textureSample(ssao_tex, ssao_samp, sample_uv).rg;
    let ao_combined = ao_pair.r * ao_pair.g;
    let ao_weighted = mix(1.0, ao_combined, indirect_weight);
    let hdr_ao = hdr_raw * ao_weighted;

    // Exposure. Two modes:
    //   auto off → manual exposure multiplier (u.params.z).
    //   auto on  → read the smoothed exposure value from a 1×1
    //              texture populated by the exposure update pass.
    var exposure: f32;
    if (u.params.y < 0.5) {
        exposure = u.params.z;
    } else {
        exposure = textureSample(exposure_tex, exposure_samp, vec2<f32>(0.5, 0.5)).r;
    }
    let hdr = hdr_ao * exposure;

    // Branch between ACES and AgX via the uniform. Costs one
    // compare per fragment; the dead branch gets DCE'd per-draw
    // since the uniform is constant across the frame.
    var ldr = tonemap_select(hdr);

    // --- Chromatic aberration (UE5 formula, post-tonemap) ---
    // Port of the CA block in UE5's PostProcessTonemap.usf lines
    // 320-342. Three pieces we take directly from Epic's shader:
    //
    //   1. LensUV-space (-1..+1) per-axis formula with a StartOffset
    //      dead-zone around the centre: shift = sign(lens) *
    //      saturate(|lens| - start) * scale. Inside the dead-zone
    //      the centre of the frame stays perfectly sharp; only the
    //      edges pick up the fringe, matching the way a real
    //      photographic lens disperses light at the periphery.
    //
    //   2. Separate R and B scales for wavelength dispersion.
    //      Red refracts less than blue in a real lens, so the B
    //      shift is ~1.5× the R shift (matches the typical default
    //      that ships in UE5's post-process volume).
    //
    //   3. Green is sampled at the centre UV (no shift). Only R and
    //      B fringe outward.
    //
    // We diverge from UE5 in one spot: UE5 samples HDR pre-tonemap
    // and relies on TAA's neighborhood clamp to bound the HDR
    // ratios. With TAA optional here, we run the offset samples
    // through the same AO + exposure + tonemap path and compose R
    // and B from the LDR results — so a bright specular firefly
    // can't ride the HDR ratio and survive tonemap as a magenta /
    // green speck across the whole frame.
    let ca_strength = u.filmic.x;
    if (ca_strength > 0.0) {
        let ca_r_scale = ca_strength;
        let ca_b_scale = ca_strength * 1.5;
        let start_offset = 0.25;

        let lens_uv = sample_uv * 2.0 - 1.0;
        let beyond = max(abs(lens_uv) - vec2<f32>(start_offset), vec2<f32>(0.0));
        let sign_lens = sign(lens_uv);
        let uv_r_lens = lens_uv - sign_lens * beyond * ca_r_scale;
        let uv_b_lens = lens_uv - sign_lens * beyond * ca_b_scale;
        let uv_r = uv_r_lens * 0.5 + 0.5;
        let uv_b = uv_b_lens * 0.5 + 0.5;

        let s_r = textureSample(hdr_tex, hdr_samp, uv_r);
        let s_b = textureSample(hdr_tex, hdr_samp, uv_b);
        let clean_r = select(vec3<f32>(0.0), s_r.rgb, s_r.rgb == s_r.rgb);
        let clean_b = select(vec3<f32>(0.0), s_b.rgb, s_b.rgb == s_b.rgb);
        let ao_r_pair = textureSample(ssao_tex, ssao_samp, uv_r).rg;
        let ao_b_pair = textureSample(ssao_tex, ssao_samp, uv_b).rg;
        let ao_r = mix(1.0, ao_r_pair.r * ao_r_pair.g, s_r.a);
        let ao_b = mix(1.0, ao_b_pair.r * ao_b_pair.g, s_b.a);
        let ldr_r_full = tonemap_select(clean_r * ao_r * exposure);
        let ldr_b_full = tonemap_select(clean_b * ao_b * exposure);
        ldr = vec3<f32>(ldr_r_full.r, ldr.g, ldr_b_full.b);
    }

    // --- Sharpen (post-tonemap unsharp mask) ---
    // Subtle lens-like crispening. Samples 4 neighbour HDR values,
    // applies the same AO + exposure + tonemap path as the centre,
    // averages them in LDR, and adds the (centre - avg) difference
    // back scaled by `sharpen_strength`. Operating in LDR post-tonemap
    // avoids the classic problem of HDR sharpen blowing out highlights
    // (the unsharp of a bright pixel against a dark one gets amplified
    // into an ugly rim). The cost is 4 extra tonemap calls, which on
    // Metal is trivially cheap.
    let sharpen_strength = u.misc.y;
    if (sharpen_strength > 0.0) {
        let dims = vec2<f32>(textureDimensions(hdr_tex));
        let t = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
        let ox = vec2<f32>(t.x, 0.0);
        let oy = vec2<f32>(0.0, t.y);
        let h_r = textureSample(hdr_tex, hdr_samp, sample_uv + ox).rgb * ao_weighted * exposure;
        let h_l = textureSample(hdr_tex, hdr_samp, sample_uv - ox).rgb * ao_weighted * exposure;
        let h_d = textureSample(hdr_tex, hdr_samp, sample_uv + oy).rgb * ao_weighted * exposure;
        let h_u = textureSample(hdr_tex, hdr_samp, sample_uv - oy).rgb * ao_weighted * exposure;
        let avg = (tonemap_select(h_r) + tonemap_select(h_l)
                 + tonemap_select(h_d) + tonemap_select(h_u)) * 0.25;
        let detail = ldr - avg;
        ldr = clamp(ldr + detail * sharpen_strength, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // --- Vignette (post-tonemap) ---
    // Smooth radial darkening. Applied after tonemap so it stays
    // perceptually uniform across exposures (otherwise bright
    // scenes wash out the vignette).
    let vig_strength = u.filmic.y;
    if (vig_strength > 0.0) {
        let vig_softness = max(u.filmic.z, 0.001);
        let dist = length(in.uv - vec2<f32>(0.5, 0.5));
        // smoothstep gives a natural falloff; remap so strength=1
        // fully blackens the corner and softness controls width.
        let edge = smoothstep(0.5 - vig_softness, 0.75, dist);
        ldr = ldr * (1.0 - edge * vig_strength);
    }

    // --- Film grain (post-tonemap) ---
    // Per-pixel noise added to luma. Animated by frame seed in
    // misc.x so grain crawls naturally; if seed stays fixed (e.g.
    // headless screenshots) the grain freezes.
    let grain_strength = u.filmic.w;
    if (grain_strength > 0.0) {
        let seed = u.misc.x;
        let n = hash21(in.uv * 1024.0 + vec2<f32>(seed, seed * 1.7)) - 0.5;
        ldr = ldr + vec3<f32>(n * grain_strength);
    }

    return vec4<f32>(ldr, 1.0);
}
";

// -----------------------------------------------------------------
// Upscale pass — render-res → full-surface. Engages only when
// `render_scale < 1.0 && !taa_enabled` (when TAA is on the TAA pass
// does its own Catmull-Rom reconstruction). Mode 0 = bilinear (cheap,
// soft), mode 1 = Catmull-Rom 5-tap (sharper edge reconstruction,
// same kernel as the TAA pass).
// -----------------------------------------------------------------
pub(in crate::renderer) const UPSCALE_SHADER_WGSL: &str = "
struct UpscaleParams {
    // x = mode (0 = bilinear, 1 = catmull-rom), yzw padding.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: UpscaleParams;
@group(0) @binding(1) var composed_tex: texture_2d<f32>;
@group(0) @binding(2) var composed_samp: sampler;

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

// 5-tap Catmull-Rom (Karis formulation) — same kernel as the TAA
// shader's upsample. Costs 5 bilinear fetches vs 1; reconstructs a
// cubic-Hermite curve through 4 source taps which preserves edges
// where naive bilinear goes mushy.
fn sample_catmull_rom(uv: vec2<f32>) -> vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(composed_tex));
    let inv_size = 1.0 / tex_size;
    let sample_pos = uv * tex_size;
    let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let offset12 = w2 / w12;
    let tp0 = (tex_pos1 - 1.0) * inv_size;
    let tp3 = (tex_pos1 + 2.0) * inv_size;
    let tp12 = (tex_pos1 + offset12) * inv_size;
    var result = vec4<f32>(0.0);
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp0.y),  0.0) * w12.x * w0.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x,  tp12.y), 0.0) * w0.x  * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp12.y), 0.0) * w12.x * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x,  tp12.y), 0.0) * w3.x  * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp3.y),  0.0) * w12.x * w3.y;
    return max(result, vec4<f32>(0.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let mode = u32(u.params.x);
    if (mode == 1u) {
        return sample_catmull_rom(in.uv);
    }
    return textureSample(composed_tex, composed_samp, in.uv);
}
";

// -----------------------------------------------------------------
// Contrast-adaptive sharpen — a simplified RCAS (FidelityFX). 5-tap
// cross kernel, negative-lobe FIR with lobe amplitude adapted to
// per-pixel luma headroom so flat areas don't amplify noise. Runs
// before composite on whichever texture feeds composite today
// (sss/mb/dof/taa/upscale/composed). Gated on `strength > 0`;
// default 0 so the pass is a no-op unless the user opts in.
// -----------------------------------------------------------------
pub(in crate::renderer) const RCAS_SHADER_WGSL: &str = "
struct RcasParams {
    // x = strength (0 = off, 0.3 = subtle, 0.6 = punchy, 1.0 = max).
    // yzw padding.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: RcasParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var input_samp: sampler;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let strength = u.params.x;
    let center = textureSample(input_tex, input_samp, in.uv);
    if (strength <= 0.0) {
        return center;
    }
    let tex_size = vec2<f32>(textureDimensions(input_tex));
    let px = 1.0 / tex_size;

    let c = center.rgb;
    let n = textureSample(input_tex, input_samp, in.uv + vec2<f32>( 0.0, -px.y)).rgb;
    let s = textureSample(input_tex, input_samp, in.uv + vec2<f32>( 0.0,  px.y)).rgb;
    let w = textureSample(input_tex, input_samp, in.uv + vec2<f32>(-px.x, 0.0)).rgb;
    let e = textureSample(input_tex, input_samp, in.uv + vec2<f32>( px.x, 0.0)).rgb;

    // Luma-based local min/max for contrast adaptation. Rec. 709.
    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let lc = dot(c, lw);
    let ln = dot(n, lw);
    let ls = dot(s, lw);
    let lwl = dot(w, lw);
    let le = dot(e, lw);
    let lmin = min(min(min(ln, ls), min(lwl, le)), lc);
    let lmax = max(max(max(ln, ls), max(lwl, le)), lc);

    // Headroom — how much room is there before we clip at 0 or the
    // local max? Small in flat areas, large at edges. This is the
    // 'Robust' part of RCAS: sharpen only where it helps.
    let headroom = clamp(lmin / max(lmax, 1e-4), 0.0, 1.0);

    // Lobe amplitude — bigger at edges (low headroom), smaller in
    // flat areas (high headroom). 0.125 cap keeps the kernel stable.
    let lobe = 0.125 * strength * (1.0 - headroom);

    // Negative-lobe FIR: center*(1+4*lobe) - lobe*(n+s+w+e).
    // Coefficients sum to 1 → DC preserved.
    let sharpened = c * (1.0 + 4.0 * lobe) - lobe * (n + s + w + e);

    return vec4<f32>(max(sharpened, vec3<f32>(0.0)), center.a);
}
";

