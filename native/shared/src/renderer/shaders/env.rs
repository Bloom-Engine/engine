//! Environment: GGX prefilter, sky pass, aerial perspective.
//! Split from renderer/shaders.rs.

pub(in crate::renderer) const PREFILTER_SHADER_WGSL: &str = "
struct PrefilterUniforms {
    /// x = roughness (∈ [0, 1]), y = sample count, zw = mip resolution
    /// in pixels (used for fragCoord → UV conversion).
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: PrefilterUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_samp: sampler;

const PI: f32 = 3.14159265;

fn radical_inverse_vdc(in_bits: u32) -> f32 {
    var bits = in_bits;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a*a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h_local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * h_local.x + b * h_local.y + n * h_local.z);
}

fn dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let phi = atan2(d.z, d.x);
    let raw_u = phi / (2.0 * PI);
    return vec2<f32>(raw_u - floor(raw_u), theta / PI);
}

fn uv_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let phi = uv.x * 2.0 * PI;
    let theta = uv.y * PI;
    let sin_theta = sin(theta);
    return vec3<f32>(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let mip_w = u.params.z;
    let mip_h = u.params.w;
    let uv = vec2<f32>(frag_pos.x / mip_w, frag_pos.y / mip_h);
    let n = uv_to_dir(uv);
    let v = n; // Karis simplification

    let n_samples = u32(u.params.y);
    let roughness = u.params.x;
    var color = vec3<f32>(0.0);
    var weight = 0.0;

    for (var i = 0u; i < n_samples; i++) {
        let xi = hammersley(i, n_samples);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l > 0.0) {
            color += textureSampleLevel(src_tex, src_samp, dir_to_uv(l), 0.0).rgb * n_dot_l;
            weight += n_dot_l;
        }
    }
    return vec4<f32>(color / max(weight, 1e-4), 1.0);
}

// Diffuse irradiance convolution (cosine-weighted). Used to populate
// the env mip chain's smallest mip — the scene shader samples that
// mip for IBL diffuse. Cosine-weighted importance sampling means the
// per-sample weight (cos θ / π) cancels the PDF, so we can just
// average the env samples directly. Much closer to a proper diffuse
// irradiance map than 'GGX with roughness = 1' would be.
@fragment
fn fs_diffuse(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let mip_w = u.params.z;
    let mip_h = u.params.w;
    let uv = vec2<f32>(frag_pos.x / mip_w, frag_pos.y / mip_h);
    let n = uv_to_dir(uv);

    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);

    let n_samples = u32(u.params.y);
    var color = vec3<f32>(0.0);

    for (var i = 0u; i < n_samples; i++) {
        let xi = hammersley(i, n_samples);
        // Cosine-weighted hemisphere sample (Malley's method).
        let phi = 2.0 * PI * xi.x;
        let cos_theta = sqrt(1.0 - xi.y);
        let sin_theta = sqrt(xi.y);
        let l_local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
        let l = t * l_local.x + b * l_local.y + n * l_local.z;
        var sample = textureSampleLevel(src_tex, src_samp, dir_to_uv(l), 0.0).rgb;
        // Firefly clamp per env sample. Raw outdoor.hdr has the sun
        // disc at luma 1000+; at N samples per output texel the
        // 1-in-N chance of a sample hitting the sun leaves a bright
        // patch in the 'diffuse' irradiance map at every normal
        // direction that aims near it. Capping each sample at luma
        // 50 bounds the estimator's variance so the convolved output
        // is actually smooth — which is what fs_main_scene expects
        // when it multiplies by base_color for the IBL diffuse term.
        // The Sponza floor's per-pixel normal-map variation was
        // picking different 'bright spots' in the unsmoothed
        // irradiance map, producing exactly the white speckle on
        // the sunlit stone.
        let sample_luma = dot(sample, vec3<f32>(0.2126, 0.7152, 0.0722));
        let cap = 50.0;
        if (sample_luma > cap) {
            sample = sample * (cap / sample_luma);
        }
        color += sample;
    }
    return vec4<f32>(color / f32(n_samples), 1.0);
}
";

pub(in crate::renderer) const SKY_SHADER_WGSL: &str = "
struct SkyUniforms {
    // Camera world basis (right, up, forward) and screen scale.
    // Reconstructing view direction from these is more numerically
    // robust than inverting the VP matrix and divides — and avoids
    // edge cases like degenerate w divisions at the far plane.
    right:    vec4<f32>,  // xyz = right*tan(fovy/2)*aspect; w unused
    up:       vec4<f32>,  // xyz = up*tan(fovy/2);            w unused
    forward:  vec4<f32>,  // xyz = forward (unit);             w unused
    intensity: vec4<f32>, // x = multiplier; yzw padding
};

@group(0) @binding(0) var<uniform> u: SkyUniforms;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

const PI: f32 = 3.14159265;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn sky_vs(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single oversized triangle covering [-1,1]^2:
    //   vid 0 → (-1, -1)
    //   vid 1 → ( 3, -1)
    //   vid 2 → (-1,  3)
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    out.ndc = vec2<f32>(x, y);
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

fn linear_to_srgb_v(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= cutoff);
}

struct SkyOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec2<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) albedo: vec4<f32>,
};

@fragment
fn sky_fs(in: VsOut) -> SkyOut {
    // View direction = forward + ndc.x * (right*tan*aspect)
    //                + ndc.y * (up*tan)
    // The right/up vectors already have the scale baked in.
    let dir = normalize(u.forward.xyz + in.ndc.x * u.right.xyz + in.ndc.y * u.up.xyz);

    // Equirectangular UV — must match bloom-reference exactly:
    //   u = (phi / 2π) wrapped to [0, 1)   → +X direction at u=0
    //   v = theta / π                       → +Y at v=0
    // Earlier `phi/2π + 0.5` gave a 180° rotation that put cloud
    // patterns on the wrong side of the helmet vs the reference.
    //
    // Below-horizon clamp: most outdoor HDRIs (including
    // assets/outdoor.hdr) only contain sky in the upper
    // hemisphere — the lower half mirrors the upper because the
    // photographer captured from a flat/reflective floor. Without
    // a clamp, the sky-dome render shows the mirror when the
    // camera looks below the horizon (e.g. standing on the
    // Sponza roof, any pitch-down shot). Clamp v at 0.5 so
    // below-horizon directions re-sample the horizon ring,
    // smoothly fading to a 'hazy ground' appearance.
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let phi = atan2(dir.z, dir.x);
    let raw_u = phi / (2.0 * PI);
    let u_coord = raw_u - floor(raw_u); // fract(); WGSL has no rem_euclid
    let v_unclamped = theta / PI;
    // Slightly below v=0.5 to sample the horizon ring and not
    // clip to a single texel row; texture filter smooths over
    // the last 2 % of vertical space.
    let v_coord = min(v_unclamped, 0.5);

    // Clamp u to half-texel inside [0,1] so the bilinear filter
    // never reaches across the ±180° seam (u wraps 0↔1).
    let tex_w = f32(textureDimensions(env_tex, 0).x);
    let half_texel = 0.5 / tex_w;
    let safe_u = clamp(u_coord, half_texel, 1.0 - half_texel);

    let radiance = textureSample(env_tex, env_samp, vec2<f32>(safe_u, v_coord)).rgb * u.intensity.x;
    // Output linear HDR radiance — the composite pass downstream does
    // the ACES tonemap + sRGB encode in one place. Sky writes to the
    // material G-buffer too: 0 metallic, 1 roughness — sky never
    // reflects, never gets reflected from (well, it gets sampled by
    // SSR via the HDR RT, but that's expected behavior).
    // Sky is at infinity — zero velocity (stationary background).
    // Sky albedo is zero — sky is the indirect-light source itself,
    // so SSGI rays landing on sky pixels must not multiply by anything
    // (otherwise the bounce would be tinted by background radiance,
    // which is wrong for a directional distant light).
    return SkyOut(vec4<f32>(radiance, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec4<f32>(0.0));
}
";

// EN-005 V2 — aerial-perspective 3D LUT bake.
//
// Each voxel (NDC.x, NDC.y, depth-slice) holds the integrated in-
// scattered radiance and accumulated transmittance from the camera
// to a sample point along the view ray. scene_compose samples this
// in place of its 16-step volumetric fog march when procedural sky
// is on, giving per-pixel angular variation (sunset side warmer than
// the opposite horizon) at a fraction of the per-frame cost.
//
// Engine units assumed metres; atmosphere math runs in kilometres.
// Camera height above sea level is treated as 0.5 km regardless of
// actual y — same approximation the sky-view shader uses, since the
// atmosphere is far thicker than typical scene height variation.

pub(in crate::renderer) const AERIAL_PERSPECTIVE_SHADER_WGSL: &str = "
struct AerialParams {
    // xyz = camera world position (metres), w = max distance (km)
    cam_pos: vec4<f32>,
    inv_vp: mat4x4<f32>,
    // xyz = sun direction (unit), w = sun intensity scale
    sun: vec4<f32>,
    // x = rayleigh density mult, y = mie density mult, zw unused
    knobs: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: AerialParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var multi_scattering_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_samp: sampler;
@group(0) @binding(4) var aerial_out: texture_storage_3d<rgba16float, write>;

const PI: f32 = 3.14159265;
const GROUND_R: f32 = 6360.0;
const ATMOS_TOP: f32 = 6460.0;
const RAY_H: f32 = 8.0;
const MIE_H: f32 = 1.2;
const RAY_SCAT: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.100e-3);
const MIE_SCAT: f32 = 3.996e-3;
const MIE_EXT: f32 = 4.440e-3;
const O3_ABS: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const O3_PEAK: f32 = 25.0;
const O3_HALF: f32 = 15.0;

const MARCH_STEPS: u32 = 16u;

fn ray_density(alt: f32) -> f32 { return exp(-alt / RAY_H); }
fn mie_density(alt: f32) -> f32 { return exp(-alt / MIE_H); }
fn o3_density(alt: f32) -> f32 { return max(0.0, 1.0 - abs(alt - O3_PEAK) / O3_HALF); }

fn extinction(alt: f32) -> vec3<f32> {
    let rd = ray_density(alt) * u.knobs.x;
    let md = mie_density(alt) * u.knobs.y;
    let od = o3_density(alt);
    return RAY_SCAT * rd + vec3<f32>(MIE_EXT) * md + O3_ABS * od;
}

fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let v = clamp((r - GROUND_R) / (ATMOS_TOP - GROUND_R), 0.0, 1.0);
    let uu = clamp((mu + 1.0) * 0.5, 0.0, 1.0);
    return textureSampleLevel(transmittance_lut, lut_samp, vec2<f32>(uu, v), 0.0).rgb;
}

fn sample_multi_scattering(r: f32, mu_s: f32) -> vec3<f32> {
    let v = clamp((r - GROUND_R) / (ATMOS_TOP - GROUND_R), 0.0, 1.0);
    let uu = clamp((mu_s + 1.0) * 0.5, 0.0, 1.0);
    return textureSampleLevel(multi_scattering_lut, lut_samp, vec2<f32>(uu, v), 0.0).rgb;
}

fn rayleigh_phase(cos_t: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_t * cos_t);
}

fn mie_phase(cos_t: f32) -> f32 {
    let g = 0.8;
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_t * cos_t);
    let den = 8.0 * PI * (2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-6), 1.5);
    return num / den;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(aerial_out);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }

    // Voxel → NDC. Note: WebGPU NDC has Y up; our render target had
    // Y down in pixel space but inv_vp produces world space directly,
    // so we use NDC y = 1 - 2*v to match (top of screen = +Y in NDC).
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
           / vec2<f32>(f32(dims.x), f32(dims.y));
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Reconstruct a far-plane world point through (ndc_xy, z=1). The
    // ray from cam_pos through that point is the per-voxel view ray.
    let far_h = u.inv_vp * vec4<f32>(ndc_xy, 1.0, 1.0);
    let far_world = far_h.xyz / far_h.w;
    let view_dir = normalize(far_world - u.cam_pos.xyz);

    // Depth slicing — linear from camera out to max_dist_km. The
    // (gid.z + 0.5) / dims.z maps 0..max evenly; near voxels resolve
    // close haze, far voxels capture distant horizon scatter.
    let max_dist_km = u.cam_pos.w;
    let target_dist_km = (f32(gid.z) + 0.5) / f32(dims.z) * max_dist_km;

    // March from camera to (camera + view_dir * target_dist_km).
    // Atmosphere math is in km; the engine world-space view ray is
    // unit-length so we scale step-distance directly.
    let n_steps = MARCH_STEPS;
    let step_km = target_dist_km / f32(n_steps);
    let sun = normalize(u.sun.xyz);

    // Sea-level approximation for camera radius — sky-view shader
    // uses the same trick. Atmosphere is so much thicker than scene
    // height variation that the per-altitude correction is < 1 LSB.
    let r0 = GROUND_R + 0.5;
    let mu = view_dir.y;
    let mu_s_cam = sun.y;
    let nu = dot(view_dir, sun);
    let phase_r = rayleigh_phase(nu);
    let phase_m = mie_phase(nu);

    var transmittance = vec3<f32>(1.0);
    var in_scatter = vec3<f32>(0.0);
    for (var i = 0u; i < n_steps; i = i + 1u) {
        let d = (f32(i) + 0.5) * step_km;
        let r_d = sqrt(r0 * r0 + d * d + 2.0 * r0 * mu * d);
        let alt = max(0.0, r_d - GROUND_R);

        let mu_s_p = clamp((r0 * mu_s_cam + d * dot(view_dir, sun)) / r_d, -1.0, 1.0);

        let rd = ray_density(alt) * u.knobs.x;
        let md = mie_density(alt) * u.knobs.y;
        let scat_r = RAY_SCAT * rd;
        let scat_m = vec3<f32>(MIE_SCAT) * md;

        let ext = extinction(alt);
        let step_t = exp(-ext * step_km);

        let t_sun = sample_transmittance(r_d, mu_s_p);
        let single = (scat_r * phase_r + scat_m * phase_m) * t_sun;
        let psi_ms = sample_multi_scattering(r_d, mu_s_p);
        let multi = (scat_r + scat_m) * psi_ms * (1.0 / (4.0 * PI));

        in_scatter = in_scatter + transmittance * (single + multi) * step_km;
        transmittance = transmittance * step_t;
    }

    in_scatter = in_scatter * u.sun.w;

    // Mean transmittance — scene_compose applies it as a single
    // scalar attenuation. Per-channel transmittance would be tighter
    // physics but pushes us off the existing fog composite shape.
    let mean_t = (transmittance.x + transmittance.y + transmittance.z) / 3.0;

    textureStore(
        aerial_out,
        vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)),
        vec4<f32>(in_scatter, mean_t),
    );
}
";

// ============================================================================
// EN-005 Phase 2 — Procedural sky (Hillaire 2020)
// ============================================================================
//
// Two shaders working off the LUTs baked in atmosphere_lut.rs:
//
//  1. SKY_VIEW_LUT_SHADER_WGSL — compute pass, dispatched on sun-move.
//     Ray-marches every (azimuth, elevation) texel of a 2D radiance
//     cache, accumulating Rayleigh + Mie + ozone single-scattering
//     and reading the multi-scattering LUT for the bounce term.
//
//  2. PROCEDURAL_SKY_SHADER_WGSL — fullscreen render pass per frame.
//     Reconstructs view direction per pixel, samples the sky-view
//     LUT for sky radiance, draws the sun disk attenuated by
//     transmittance. Slots into the existing HDR pass in place of
//     `sky_pipeline` when procedural sky is enabled.
//
// Shared atmosphere parameters are inlined as `const`s in each shader
// rather than uniformed — they're physical constants, not user knobs.
// The runtime knobs (sun direction, density multipliers) live in the
// uniform blocks.

pub(in crate::renderer) const SKY_VIEW_LUT_SHADER_WGSL: &str = "
struct SkyViewParams {
    // xyz = sun direction (world space, unit), w = sun intensity scale
    sun: vec4<f32>,
    // x = rayleigh density multiplier
    // y = mie density multiplier
    // z = ground albedo (grey)
    // w = unused
    knobs: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SkyViewParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var multi_scattering_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_samp: sampler;
@group(0) @binding(4) var sky_view_out: texture_storage_2d<rgba16float, write>;

// --- Atmosphere constants (must match atmosphere_lut.rs) ---
const PI: f32 = 3.14159265;
const GROUND_R: f32 = 6360.0;
const ATMOS_TOP: f32 = 6460.0;
const RAY_H: f32 = 8.0;
const MIE_H: f32 = 1.2;
const RAY_SCAT: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.100e-3);
const MIE_SCAT: f32 = 3.996e-3;
const MIE_EXT: f32 = 4.440e-3;
const O3_ABS: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const O3_PEAK: f32 = 25.0;
const O3_HALF: f32 = 15.0;

const VIEW_STEPS: u32 = 32u;

fn ray_density(alt: f32) -> f32 { return exp(-alt / RAY_H); }
fn mie_density(alt: f32) -> f32 { return exp(-alt / MIE_H); }
fn o3_density(alt: f32) -> f32 { return max(0.0, 1.0 - abs(alt - O3_PEAK) / O3_HALF); }

fn extinction(alt: f32) -> vec3<f32> {
    let rd = ray_density(alt) * u.knobs.x;
    let md = mie_density(alt) * u.knobs.y;
    let od = o3_density(alt);
    return RAY_SCAT * rd + vec3<f32>(MIE_EXT) * md + O3_ABS * od;
}

fn ray_atmosphere_intersect(r: f32, mu: f32, sphere_r: f32) -> f32 {
    // Returns the first positive intersection distance, or -1 if none.
    let disc = r * r * (mu * mu - 1.0) + sphere_r * sphere_r;
    if (disc < 0.0) { return -1.0; }
    let sd = sqrt(disc);
    let t1 = -r * mu - sd;
    let t2 = -r * mu + sd;
    if (t1 > 0.0) { return t1; }
    if (t2 > 0.0) { return t2; }
    return -1.0;
}

fn dist_to_boundary(r: f32, mu: f32) -> f32 {
    let to_top = ray_atmosphere_intersect(r, mu, ATMOS_TOP);
    let to_ground = ray_atmosphere_intersect(r, mu, GROUND_R);
    if (to_ground > 0.0 && (to_top < 0.0 || to_ground < to_top)) {
        return to_ground;
    }
    return max(to_top, 0.0);
}

// LUT mappings (must match atmosphere_lut.rs).
fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let v = clamp((r - GROUND_R) / (ATMOS_TOP - GROUND_R), 0.0, 1.0);
    let uu = clamp((mu + 1.0) * 0.5, 0.0, 1.0);
    return textureSampleLevel(transmittance_lut, lut_samp, vec2<f32>(uu, v), 0.0).rgb;
}

fn sample_multi_scattering(r: f32, mu_s: f32) -> vec3<f32> {
    let v = clamp((r - GROUND_R) / (ATMOS_TOP - GROUND_R), 0.0, 1.0);
    let uu = clamp((mu_s + 1.0) * 0.5, 0.0, 1.0);
    return textureSampleLevel(multi_scattering_lut, lut_samp, vec2<f32>(uu, v), 0.0).rgb;
}

fn rayleigh_phase(cos_t: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_t * cos_t);
}

fn mie_phase(cos_t: f32) -> f32 {
    // Cornette-Shanks, g=0.8 — standard for atmospheric Mie.
    let g = 0.8;
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_t * cos_t);
    let den = 8.0 * PI * (2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-6), 1.5);
    return num / den;
}

// Decode (workgroup-local) coords into a world-space view direction.
// V1 uses linear (azimuth, elevation) — simple, slightly horizon-poor;
// future revision can switch to Hillaire's sin² mapping for sharper
// horizon detail.
fn sky_view_uv_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let azimuth = uv.x * 2.0 * PI;
    let elevation = (uv.y - 0.5) * PI; // [-π/2, π/2]
    let cos_e = cos(elevation);
    return vec3<f32>(cos_e * cos(azimuth), sin(elevation), cos_e * sin(azimuth));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(sky_view_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));
    let view_dir = sky_view_uv_to_dir(uv);

    // V1: camera at sea level, atmosphere centred at origin. The y axis
    // is up; r = GROUND_R + altitude. Phase 3 lifts this once the host
    // can pass per-frame camera height.
    let r0 = GROUND_R + 0.5; // 0.5 km above ground = typical eye height + atmospheric haze
    let mu = view_dir.y;     // cos(view-zenith)
    let sun = normalize(u.sun.xyz);
    let mu_s = sun.y;
    let nu = dot(view_dir, sun); // cos(view-sun angle)

    let total_dist = dist_to_boundary(r0, mu);
    if (total_dist <= 0.0) {
        textureStore(sky_view_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let dx = total_dist / f32(VIEW_STEPS);
    var radiance = vec3<f32>(0.0);
    var transmittance = vec3<f32>(1.0);

    let phase_r = rayleigh_phase(nu);
    let phase_m = mie_phase(nu);

    for (var i: u32 = 0u; i < VIEW_STEPS; i = i + 1u) {
        let d = (f32(i) + 0.5) * dx;
        // Radius at sample point.
        let r_d = sqrt(r0 * r0 + d * d + 2.0 * r0 * mu * d);
        let alt = max(0.0, r_d - GROUND_R);

        // Sun zenith cosine at the sample point (parallel-sun
        // approximation — sun direction stays fixed in world space).
        let mu_s_p = clamp((r0 * mu_s + d * dot(view_dir, sun)) / r_d, -1.0, 1.0);

        let rd = ray_density(alt) * u.knobs.x;
        let md = mie_density(alt) * u.knobs.y;
        let scat_r = RAY_SCAT * rd;
        let scat_m = vec3<f32>(MIE_SCAT) * md;

        let ext = extinction(alt);
        let step_t = exp(-ext * dx);

        // Sun radiance at sample after travelling to top of atmosphere.
        let t_sun = sample_transmittance(r_d, mu_s_p);

        // Single-scattering — Rayleigh + Mie with their phase functions.
        let single = (scat_r * phase_r + scat_m * phase_m) * t_sun;

        // Multi-scattering — isotropic (1/4π) scaled by the LUT-baked
        // bounce coefficient. Hillaire's energy-conserving form.
        let psi_ms = sample_multi_scattering(r_d, mu_s_p);
        let multi = (scat_r + scat_m) * psi_ms * (1.0 / (4.0 * PI));

        let in_scatter = (single + multi) * dx;
        // Analytic in-scattering integral: contribution × accumulated transmittance.
        radiance = radiance + transmittance * in_scatter;
        transmittance = transmittance * step_t;
    }

    radiance = radiance * u.sun.w;

    textureStore(
        sky_view_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(radiance, 1.0),
    );
}
";

// EN-005 Phase 3 — sky-view LUT → equirect HDR re-projection. Drives
// the existing GGX prefilter chain (which is the IBL specular source
// for PBR materials), so material reflections track the procedural
// sky as the sun moves.
//
// Sky-view LUT v-axis goes nadir→zenith (0 → 1); prefilter equirect
// v-axis goes zenith→nadir (0 → 1). The only correction needed is
// `sky_view_v = 1 - equirect_v`. Destination dimensions arrive in a
// small uniform so the shader can map gl_FragCoord to UV (the
// existing prefilter shader uses the same pattern).

pub(in crate::renderer) const EQUIRECT_FROM_SKY_VIEW_SHADER_WGSL: &str = "
struct BakeParams {
    // x = dest width, y = dest height, zw unused
    dims: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: BakeParams;
@group(0) @binding(1) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Equirect destination UV (matches `prefilter`'s `dir_to_uv`
    // convention: u = phi/(2π), v = theta/π where theta is from zenith).
    let uv = vec2<f32>(frag_pos.x / u.dims.x, frag_pos.y / u.dims.y);

    // Convert to sky-view LUT UV — only the v axis differs:
    //   sky-view: v ∈ [0, 1] ↔ elevation ∈ [-π/2, π/2]   (0 = nadir, 1 = zenith)
    //   equirect: v ∈ [0, 1] ↔ theta     ∈ [0, π]        (0 = zenith, 1 = nadir)
    // → sky_view_v = 1 - equirect_v, u stays the same.
    let lut_uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return textureSampleLevel(sky_view_lut, lut_samp, lut_uv, 0.0);
}
";

pub(in crate::renderer) const PROCEDURAL_SKY_SHADER_WGSL: &str = "
struct SkyUniforms {
    right:    vec4<f32>,
    up:       vec4<f32>,
    forward:  vec4<f32>,
    intensity: vec4<f32>, // x = scene-wide intensity multiplier
};

struct SunUniforms {
    sun: vec4<f32>,        // xyz = sun direction, w = sun intensity
    params: vec4<f32>,     // x = sun angular radius (rad), y = limb darkening, zw unused
};

@group(0) @binding(0) var<uniform> u: SkyUniforms;
@group(0) @binding(1) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_samp: sampler;
@group(0) @binding(3) var<uniform> sun_u: SunUniforms;
@group(0) @binding(4) var transmittance_lut: texture_2d<f32>;

const PI: f32 = 3.14159265;
const GROUND_R: f32 = 6360.0;
const ATMOS_TOP: f32 = 6460.0;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn sky_vs(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

fn dir_to_sky_uv(dir: vec3<f32>) -> vec2<f32> {
    // Inverse of sky_view_uv_to_dir in the compute shader. Matches
    // the linear (azimuth, elevation) parameterization.
    let elevation = asin(clamp(dir.y, -1.0, 1.0));
    let azimuth = atan2(dir.z, dir.x);
    var u_norm = azimuth / (2.0 * PI);
    if (u_norm < 0.0) { u_norm = u_norm + 1.0; }
    let v_norm = elevation / PI + 0.5;
    return vec2<f32>(u_norm, v_norm);
}

// --- Procedural cloud layer (value-noise fBm) ---------------------------
fn cloud_hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}
fn cloud_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let uu = f * f * (3.0 - 2.0 * f);
    let a = cloud_hash(i);
    let b = cloud_hash(i + vec2<f32>(1.0, 0.0));
    let c = cloud_hash(i + vec2<f32>(0.0, 1.0));
    let d = cloud_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, uu.x), mix(c, d, uu.x), uu.y);
}
fn cloud_fbm(p0: vec2<f32>) -> f32 {
    var s = 0.0;
    var amp = 0.5;
    var q = p0;
    for (var i = 0; i < 5; i = i + 1) {
        s = s + amp * cloud_noise(q);
        q = q * 2.03;
        amp = amp * 0.5;
    }
    return s;
}
// Analytic cloud cover for a view ray. Projects the ray onto a virtual cloud
// plane (perspective convergence toward the horizon), samples fBm for puffy
// coverage, fades near the horizon, and thins around the sun so the disk shows
// through. Returns (coverage, sunlit-amount).
fn cloud_cover(dir: vec3<f32>, sun_dir: vec3<f32>, time: f32) -> vec2<f32> {
    if (dir.y <= 0.02) { return vec2<f32>(0.0, 0.0); }
    let p = (dir.xz / dir.y) * 2.0;
    // Slow wind drift + a slower second octave shift so the puffs also evolve.
    let drift = vec2<f32>(time * 0.006, time * 0.0025);
    var cov = cloud_fbm(p * 0.55 + vec2<f32>(23.0, 11.0) + drift);
    cov = smoothstep(0.56, 1.04, cov);
    let horizon_fade = smoothstep(0.03, 0.24, dir.y);
    let near_sun = smoothstep(0.90, 0.999, dot(dir, sun_dir));
    cov = cov * horizon_fade * (1.0 - near_sun * 0.8) * 0.9;
    let sun_amt = clamp(dot(dir, sun_dir) * 0.5 + 0.5, 0.0, 1.0);
    return vec2<f32>(cov, sun_amt);
}

fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let v = clamp((r - GROUND_R) / (ATMOS_TOP - GROUND_R), 0.0, 1.0);
    let uu = clamp((mu + 1.0) * 0.5, 0.0, 1.0);
    return textureSampleLevel(transmittance_lut, lut_samp, vec2<f32>(uu, v), 0.0).rgb;
}

struct SkyOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec2<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) albedo: vec4<f32>,
};

@fragment
fn sky_fs(in: VsOut) -> SkyOut {
    let dir = normalize(u.forward.xyz + in.ndc.x * u.right.xyz + in.ndc.y * u.up.xyz);
    let uv = dir_to_sky_uv(dir);
    var radiance = textureSampleLevel(sky_view_lut, lut_samp, uv, 0.0).rgb;

    // Sun disk: if view direction is within sun_angular_radius of sun
    // direction, add the sun's radiance attenuated by atmospheric
    // transmittance toward the sun. Below the horizon (sun.y < 0) we
    // skip — sun has set, no disk.
    let sun_dir = normalize(sun_u.sun.xyz);
    let cos_to_sun = dot(dir, sun_dir);
    let cos_radius = cos(sun_u.params.x);
    if (cos_to_sun > cos_radius && sun_dir.y > -0.05) {
        let r0 = GROUND_R + 0.5;
        let t_sun = sample_transmittance(r0, sun_dir.y);
        // Limb darkening: fade toward the disk edge by params.y.
        let edge = (cos_to_sun - cos_radius) / (1.0 - cos_radius);
        let limb = mix(1.0 - sun_u.params.y, 1.0, sqrt(max(edge, 0.0)));
        // Sun radiance ~ 20 (linear) at unit intensity; t_sun colours it.
        let sun_radiance = vec3<f32>(20.0) * sun_u.sun.w * limb * t_sun;
        radiance = radiance + sun_radiance;
    }

    radiance = radiance * u.intensity.x;

    // Procedural cloud layer, composited over the scaled sky radiance. Cloud
    // colour is absolute HDR (puffy white in sun, cool grey in shadow) so the
    // clouds read brighter than the sky behind them regardless of env intensity.
    let cc = cloud_cover(dir, sun_dir, u.intensity.y);
    if (cc.x > 0.0) {
        let lit = cc.y * cc.y;
        let cloud_col = mix(vec3<f32>(0.62, 0.66, 0.76), vec3<f32>(2.6, 2.5, 2.35), lit);
        radiance = mix(radiance, cloud_col, cc.x);
    }

    // EN-005 Phase 4 — sub-LSB dither to break up the banding that
    // Rgba16Float storage produces in low-frequency regions like the
    // zenith (where sky color changes < 1 lsb across many pixels).
    // Pseudo-blue-noise via the integer bit-mash of fragment coords;
    // amplitude scaled to ~1/2 lsb of the smallest channel so it
    // disappears once quantised.
    let dither_seed = vec2<u32>(u32(in.clip_pos.x), u32(in.clip_pos.y));
    let h = (dither_seed.x * 1597u + dither_seed.y * 31337u) ^ (dither_seed.x * 71u + dither_seed.y * 113u);
    let dither = (f32(h & 0xFFu) / 255.0 - 0.5) * (1.0 / 1024.0);
    radiance = radiance + vec3<f32>(dither);

    var out: SkyOut;
    out.color = vec4<f32>(radiance, 1.0);
    out.material = vec2<f32>(0.0, 0.0);
    out.velocity = vec2<f32>(0.0, 0.0);
    out.albedo = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
}
";

/// EN-011 — single-target reflection shader for rendering cached models
/// (trees, house, etc.) into a planar-reflection probe with a mirrored
/// view-projection. Deliberately lightweight (base colour × sun N·L + ambient,
/// alpha-cutout discard) — a water reflection doesn't need the full deferred
/// PBR/SSAO stack, and a single HDR colour target (vs the main 4-target MRT)
/// lets it draw straight into the probe RT. Group layouts are owned by the
/// renderer: g0 = dynamic per-draw model uniform, g1 = sun/ambient, g2 = the
/// scene material bind group (we only sample base colour + alpha).
pub(in crate::renderer) const REFLECT_SCENE_WGSL: &str = "
struct ReflectModelU { mvp: mat4x4<f32>, model: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: ReflectModelU;

struct ReflectLight {
    sun_dir: vec4<f32>,    // xyz dir (travel), w = intensity
    sun_color: vec4<f32>,  // rgb, w unused
    ambient: vec4<f32>,    // rgb, w = intensity
};
@group(1) @binding(0) var<uniform> light: ReflectLight;

@group(2) @binding(0) var base_tex: texture_2d<f32>;
@group(2) @binding(1) var base_samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) n: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) col: vec4<f32>,
};

@vertex
fn vs_reflect(in: VsIn) -> VsOut {
    var o: VsOut;
    o.pos = u.mvp * vec4<f32>(in.position, 1.0);
    o.n = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    o.uv = in.uv;
    o.col = in.color;
    return o;
}

fn srgb_lin(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow(max((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(0.0)), vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_reflect(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(base_tex, base_samp, in.uv);
    if (tex.a < 0.5) { discard; }   // alpha-cutout foliage reflects its shape
    let base = srgb_lin(tex.rgb) * in.col.rgb;
    let n = normalize(in.n);
    let ndl = max(dot(n, -normalize(light.sun_dir.xyz)), 0.0);
    let lit = base * (light.ambient.rgb * light.ambient.w
                      + light.sun_color.rgb * light.sun_dir.w * ndl);
    return vec4<f32>(lit, 1.0);   // alpha 1 = 'real reflection here' for the water blend
}
";

