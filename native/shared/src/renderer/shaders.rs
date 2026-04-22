//! WGSL shader strings used by the renderer.
//!
//! Pure data — no behavior, no struct definitions. Each `const`
//! is `pub(super)` so the surrounding `renderer` module (and only
//! that module) can see it, via `use super::shaders::*;` in
//! `mod.rs`. Split out so the ~11 500-line renderer file shrinks
//! to the Rust logic it actually contains.

pub(super) const SHADER_2D: &str = "
struct Uniforms {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
    view_proj: mat4x4<f32>,
};

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var tex_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    let ndc_x = (world_pos.x / uniforms.screen_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world_pos.y / uniforms.screen_size.y) * 2.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(tex, tex_sampler, in.uv);
    return tex_color * in.color;
}
";

pub(super) const SHADER_3D: &str = "
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
};

struct DirLight {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct PointLight {
    position: vec4<f32>,
    color: vec4<f32>,
};

struct Lighting {
    ambient: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    dir_light_count: vec4<f32>,
    dir_lights: array<DirLight, 4>,
    point_light_count: vec4<f32>,
    point_lights: array<PointLight, 16>,
};

struct JointMatrices {
    matrices: array<mat4x4<f32>, 128>,
};

struct VertexInput3D {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
};

struct VertexOutput3D {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_pos: vec3<f32>,
    @location(4) curr_clip: vec4<f32>,
    @location(5) prev_clip: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms3D;
@group(1) @binding(0) var<uniform> lighting: Lighting;
@group(2) @binding(0) var tex3d: texture_2d<f32>;
@group(2) @binding(1) var tex3d_sampler: sampler;
@group(3) @binding(0) var<uniform> joints: JointMatrices;

@vertex
fn vs_main_3d(in: VertexInput3D) -> VertexOutput3D {
    var out: VertexOutput3D;
    let total_weight = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    var pos = vec4<f32>(in.position, 1.0);
    var norm = vec4<f32>(in.normal, 0.0);
    if (total_weight > 0.01) {
        let j0 = u32(in.joints.x); let j1 = u32(in.joints.y);
        let j2 = u32(in.joints.z); let j3 = u32(in.joints.w);
        let skinned_pos = joints.matrices[j0] * pos * in.weights.x
                        + joints.matrices[j1] * pos * in.weights.y
                        + joints.matrices[j2] * pos * in.weights.z
                        + joints.matrices[j3] * pos * in.weights.w;
        let skinned_norm = joints.matrices[j0] * norm * in.weights.x
                         + joints.matrices[j1] * norm * in.weights.y
                         + joints.matrices[j2] * norm * in.weights.z
                         + joints.matrices[j3] * norm * in.weights.w;
        pos = skinned_pos;
        norm = skinned_norm;
    }
    let curr = u.mvp * pos;
    out.clip_position = curr;
    out.curr_clip = curr;
    out.prev_clip = u.prev_mvp * pos;
    out.normal = normalize((u.model * norm).xyz);
    out.world_pos = (u.model * pos).xyz;
    out.color = in.color * u.model_tint;
    out.uv = in.uv;
    return out;
}

struct Fs3DOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec2<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) albedo: vec4<f32>,
};

@fragment
fn fs_main_3d(in: VertexOutput3D) -> Fs3DOut {
    let n = normalize(in.normal);

    // Ambient
    var lit = lighting.ambient.rgb * lighting.ambient.a;

    // Legacy directional light (backward compat)
    let legacy_dir = normalize(lighting.light_dir.xyz);
    let legacy_diffuse = max(dot(n, legacy_dir), 0.0);
    lit += lighting.light_color.rgb * lighting.light_dir.w * legacy_diffuse;

    // Additional directional lights
    let dir_count = u32(lighting.dir_light_count.x);
    for (var i = 0u; i < dir_count; i++) {
        let dl = lighting.dir_lights[i];
        let dir = normalize(dl.direction.xyz);
        let diff = max(dot(n, dir), 0.0);
        lit += dl.color.rgb * dl.direction.w * diff;
    }

    // Point lights
    let pt_count = u32(lighting.point_light_count.x);
    for (var i = 0u; i < pt_count; i++) {
        let pl = lighting.point_lights[i];
        let to_light = pl.position.xyz - in.world_pos;
        let dist = length(to_light);
        let range = pl.position.w;
        if (dist < range) {
            let dir = to_light / dist;
            let diff = max(dot(n, dir), 0.0);
            let atten = 1.0 - (dist / range);
            let atten2 = atten * atten;
            lit += pl.color.rgb * pl.color.w * diff * atten2;
        }
    }

    let tex_color = textureSample(tex3d, tex3d_sampler, in.uv);
    // Per-pixel velocity for motion blur / TAA reprojection.
    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let vel = (curr_ndc - prev_ndc) * 0.5;
    // Immediate-mode 3D draws (drawCube etc.) aren't PBR — output
    // 0 metallic / 1 roughness so SSR doesn't try to reflect them.
    return Fs3DOut(
        vec4<f32>(0.0, 1.0, 0.0, 1.0), // DEBUG: green if pipeline_3d renders this
        vec2<f32>(0.0, 1.0),
        vel,
        vec4<f32>(0.0),
    );
}
";

pub(super) const SCENE_SHADER: &str = "
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
};

struct DirLight {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct PointLight {
    position: vec4<f32>,
    color: vec4<f32>,
};

struct Lighting {
    ambient: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    dir_light_count: vec4<f32>,
    dir_lights: array<DirLight, 4>,
    point_light_count: vec4<f32>,
    point_lights: array<PointLight, 16>,
    camera_pos: vec4<f32>,
    shadow_cascade_vps: array<mat4x4<f32>, 3>,
    shadow_cascade_splits: vec4<f32>,
    shadow_view_matrix: mat4x4<f32>,
};

struct MaterialFactors {
    metal_rough: vec4<f32>, // x=metallic, y=roughness
    emissive:    vec4<f32>, // rgb=emissive factor
};

struct VertexInputScene {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
    @location(6) tangent: vec4<f32>,
};

struct VertexOutputScene {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_pos: vec3<f32>,
    @location(4) tangent: vec4<f32>,
    @location(5) curr_clip: vec4<f32>,
    @location(6) prev_clip: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms3D;
@group(1) @binding(0) var<uniform> lighting: Lighting;
@group(1) @binding(1) var env_tex: texture_2d<f32>;
@group(1) @binding(2) var env_samp: sampler;
@group(1) @binding(3) var brdf_lut_tex: texture_2d<f32>;
@group(1) @binding(4) var brdf_lut_samp: sampler;
@group(1) @binding(5) var shadow_tex_0: texture_depth_2d;
@group(1) @binding(6) var shadow_tex_1: texture_depth_2d;
@group(1) @binding(7) var shadow_tex_2: texture_depth_2d;
@group(1) @binding(8) var shadow_samp: sampler_comparison;
@group(1) @binding(9) var env_diffuse_tex: texture_2d<f32>;
@group(2) @binding(0) var base_color_tex: texture_2d<f32>;
@group(2) @binding(1) var base_color_samp: sampler;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var normal_samp: sampler;
@group(2) @binding(4) var mr_tex: texture_2d<f32>;
@group(2) @binding(5) var mr_samp: sampler;
@group(2) @binding(6) var em_tex: texture_2d<f32>;
@group(2) @binding(7) var em_samp: sampler;
@group(2) @binding(8) var<uniform> material: MaterialFactors;
@group(2) @binding(9) var occ_tex: texture_2d<f32>;
@group(2) @binding(10) var occ_samp: sampler;

const PI: f32 = 3.14159265;

fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let phi = atan2(d.z, d.x);
    let raw_u = phi / (2.0 * PI);
    let u_coord = raw_u - floor(raw_u);
    let v_coord = theta / PI;
    return vec2<f32>(u_coord, v_coord);
}

// Clamp equirectangular UV so the bilinear filter never reaches
// across the ±180° seam (u = 0 / 1 boundary). Half a texel on
// each side keeps every tap on the correct hemisphere.
fn seamless_equirect_uv(uv: vec2<f32>) -> vec2<f32> {
    let tex_w = f32(textureDimensions(env_tex, 0).x);
    let half_texel = 0.5 / tex_w;
    return vec2<f32>(clamp(uv.x, half_texel, 1.0 - half_texel), uv.y);
}

// Sample the env map at a specific mip level, multiplied by the
// global env_intensity (lighting.camera_pos.w). Keeps IBL diffuse,
// IBL specular and the sky pass scaling in sync so loading the same
// HDR with intensity=2 brightens everything proportionally.
fn env_sample_lod(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(env_tex, env_samp, seamless_equirect_uv(dir_to_equirect_uv(dir)), lod).rgb
         * lighting.camera_pos.w;
}

fn env_sample(dir: vec3<f32>) -> vec3<f32> {
    return textureSample(env_tex, env_samp, seamless_equirect_uv(dir_to_equirect_uv(dir))).rgb
         * lighting.camera_pos.w;
}

@vertex
fn vs_main_scene(in: VertexInputScene) -> VertexOutputScene {
    var out: VertexOutputScene;
    let pos4 = vec4<f32>(in.position, 1.0);
    let curr = u.mvp * pos4;
    out.clip_position = curr;
    out.curr_clip = curr;
    out.prev_clip = u.prev_mvp * pos4;
    let world4 = u.model * pos4;
    out.world_pos = world4.xyz;
    out.normal = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.color = in.color * u.model_tint;
    out.uv = in.uv;
    out.tangent = vec4<f32>(normalize((u.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz), in.tangent.w);
    return out;
}

// Screen-space-derivative TBN. Reconstructs a tangent frame purely
// from the fragment's world-space position and UV — no vertex tangent
// attribute required. Based on Mikkelsen 2010 ('Followup: Normal
// Mapping Without Precomputed Tangents'). Gives close-to-identical
// results to pre-baked tangents for continuous UV mappings, which is
// the common case for PBR assets. We use this as a fallback when the
// mesh has no TANGENT accessor (very common — e.g., DamagedHelmet).
fn compute_tbn(world_pos: vec3<f32>, n: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let dp1 = dpdx(world_pos);
    let dp2 = dpdy(world_pos);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let denom = max(dot(t, t), dot(b, b));
    let invmax = inverseSqrt(max(denom, 1e-20));
    return mat3x3<f32>(t * invmax, b * invmax, n);
}

// Exact piecewise sRGB → linear, matching bloom-reference's
// `srgb_u8_to_linear`. The 2.2-gamma approximation we used before
// drifts by ~0.005 in mid-tones, which adds up across base color +
// emissive samples and skews IBL diffuse colors slightly bluer than
// the reference.
fn srgb_to_linear_v(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let lo = c / 12.92;
    let hi = pow(max((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(0.0)), vec3<f32>(2.4));
    return select(hi, lo, c <= cutoff);
}

fn aces_tone(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let cc = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((c * (c * a + b)) / (c * (c * cc + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- Cook-Torrance GGX building blocks ---
fn d_ggx(n_dot_h: f32, alpha2: f32) -> f32 {
    let x = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * x * x);
}

fn v_smith_ggx_correlated(n_dot_l: f32, n_dot_v: f32, alpha2: f32) -> f32 {
    // Height-correlated Smith visibility (Heitz 2014). Combines with
    // the Cook-Torrance /4*NdotL*NdotV denominator — so specular is
    // D * V * F directly (no further divide).
    let ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    let ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    return 0.5 / max(ggxv + ggxl, 1e-5);
}

fn f_schlick(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {
    let fc = pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * fc;
}

// Sample a single cascade's shadow texture with 4-tap Poisson PCF.
fn sample_cascade(cascade: i32, shadow_uv: vec2<f32>, depth_ref: f32) -> f32 {
    var dims: vec2<u32>;
    if (cascade == 0) {
        dims = textureDimensions(shadow_tex_0);
    } else if (cascade == 1) {
        dims = textureDimensions(shadow_tex_1);
    } else {
        dims = textureDimensions(shadow_tex_2);
    }
    let texel = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));
    // Tighter PCF radius (1.0 vs. prior 2.0). Softer was safer against
    // shadow acne / swim but produced a ~4-texel penumbra on every
    // shadow — for outdoor sun at this map resolution that translates
    // to 2-3m of fuzz, which reads as 'painted' rather than 'cast'.
    // The sun's real angular size gives a ~1m penumbra at typical
    // scene distances; r=1.0 roughly matches that.
    let radius = 1.0;
    var sum = 0.0;
    let poisson = array<vec2<f32>, 16>(
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
    for (var i: i32 = 0; i < 16; i = i + 1) {
        let off = poisson[i] * texel * radius;
        let uv = shadow_uv + off;
        if (cascade == 0) {
            sum += textureSampleCompare(shadow_tex_0, shadow_samp, uv, depth_ref);
        } else if (cascade == 1) {
            sum += textureSampleCompare(shadow_tex_1, shadow_samp, uv, depth_ref);
        } else {
            sum += textureSampleCompare(shadow_tex_2, shadow_samp, uv, depth_ref);
        }
    }
    return sum / 16.0;
}

// Cascaded shadow map sampling. Determines which cascade the fragment
// belongs to based on its view-space depth, projects through that
// cascade's VP, and performs PCF. Blends between cascades at boundaries
// for smooth transitions.
fn sample_shadow(world_pos: vec3<f32>) -> f32 {
    // Select cascade by world-space DISTANCE from camera (not
    // view-space Z). Distance is rotation-independent — spinning
    // the camera doesn't change which cascade a surface falls in.
    let cam = lighting.camera_pos.xyz;
    let dist = length(world_pos - cam);

    var cascade = 2;
    if (dist <= lighting.shadow_cascade_splits.x) {
        cascade = 0;
    } else if (dist <= lighting.shadow_cascade_splits.y) {
        cascade = 1;
    }

    // Project through the selected cascade's VP
    let light_clip = lighting.shadow_cascade_vps[cascade] * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    if (light_ndc.x < -1.0 || light_ndc.x > 1.0 ||
        light_ndc.y < -1.0 || light_ndc.y > 1.0 ||
        light_ndc.z < 0.0 || light_ndc.z > 1.0) {
        return 1.0;
    }
    let shadow_uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, 1.0 - (light_ndc.y * 0.5 + 0.5));
    let bias = 0.001;
    let depth_ref = light_ndc.z - bias;
    let shadow_val = sample_cascade(cascade, shadow_uv, depth_ref);

    // Blend between cascades at boundary regions for smooth transitions.
    // The blend zone is 10% of each cascade's range.
    var split_near = 0.0;
    var split_far = lighting.shadow_cascade_splits.x;
    if (cascade == 1) {
        split_near = lighting.shadow_cascade_splits.x;
        split_far = lighting.shadow_cascade_splits.y;
    } else if (cascade == 2) {
        split_near = lighting.shadow_cascade_splits.y;
        split_far = lighting.shadow_cascade_splits.z;
    }
    let blend_zone = (split_far - split_near) * 0.1;
    let dist_to_edge = split_far - dist;

    if (dist_to_edge < blend_zone && cascade < 2) {
        // In the blend zone: sample the next cascade too and lerp
        let next_cascade = cascade + 1;
        let next_clip = lighting.shadow_cascade_vps[next_cascade] * vec4<f32>(world_pos, 1.0);
        let next_ndc = next_clip.xyz / next_clip.w;
        let next_uv = vec2<f32>(next_ndc.x * 0.5 + 0.5, 1.0 - (next_ndc.y * 0.5 + 0.5));
        let next_depth_ref = next_ndc.z - bias;
        let next_val = sample_cascade(next_cascade, next_uv, next_depth_ref);
        let t = dist_to_edge / blend_zone;
        return mix(next_val, shadow_val, t);
    }

    return shadow_val;
}

// Evaluate a single directional light's PBR contribution. Returns
// linear-space radiance. `l_dir` points *from surface to light*,
// `intensity` scales the light color.
fn shade_pbr(
    n: vec3<f32>,
    v: vec3<f32>,
    l_dir: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, l_dir), 0.0);
    if (n_dot_l <= 0.0 || intensity <= 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), 1e-4);
    // `normalize(0)` is NaN. At grazing-back angles (view roughly
    // anti-parallel to the light direction on a near-flat surface)
    // l + v can reach a vector indistinguishable from zero in f32,
    // and a single NaN here survives the rest of the BRDF +
    // tonemap chain as a pink speck. Skip the specular lobe when
    // the half-vector is degenerate — diffuse still contributes.
    let h_raw = l_dir + v;
    let h_len2 = dot(h_raw, h_raw);
    if (h_len2 <= 1e-12) {
        let kd0 = (vec3<f32>(1.0) - mix(vec3<f32>(0.04), base_color, metallic)) * (1.0 - metallic);
        return kd0 * base_color / PI * light_color * intensity * n_dot_l;
    }
    let h = h_raw * inverseSqrt(h_len2);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, h), 0.0, 1.0);

    let alpha = max(roughness * roughness, 0.001);
    let alpha2 = alpha * alpha;

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let f = f_schlick(v_dot_h, f0);
    let d = d_ggx(n_dot_h, alpha2);
    let vis = v_smith_ggx_correlated(n_dot_l, n_dot_v, alpha2);

    let specular_raw = d * vis * f;

    // Dielectric direct-specular attenuation. A polished marble column
    // lit by the sun produces a narrow GGX highlight peak that pathtracers
    // average over hemisphere-sized light sources; our point sun spikes
    // D_GGX to 1000+ at the peak and survives tonemap as a bright stripe
    // even after Fresnel (Intel Sponza column vs Cycles was the test).
    // Same smoothstep-by-roughness treatment as the IBL path, applied
    // only to the specular lobe — diffuse stays physically correct.
    let dielectric_direct_amp = smoothstep(0.0, 1.0, roughness);
    let dielectric_factor = 1.0 - metallic;
    let direct_spec_scale = mix(1.0, dielectric_direct_amp, dielectric_factor);
    // Universal roughness damping on direct specular too — same
    // reasoning as the IBL path; a smooth marble column lit by a
    // point sun spikes D_GGX past any tonemap cap. Metals stay at
    // full direct spec for roughness > ~0.75.
    // Universal soft luma cap on direct specular. A smooth marble
    // cylinder hit by the sun spikes D_GGX past any reasonable
    // tonemap; Reinhard-compress the luma toward a 0.3 ceiling
    // smoothly so adjacent pixels with slightly different GGX peaks
    // scale by neighbouring cap values instead of ping-ponging
    // across a hard min() discontinuity (the cause of the sparkle on
    // Sponza's sunlit floor tiles).
    let direct_luma = dot(specular_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let direct_cap = 1.0 / (1.0 + direct_luma / 0.3);
    let universal_damp = smoothstep(0.05, 0.75, roughness);
    let specular = specular_raw * direct_spec_scale * universal_damp * direct_cap;

    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;

    return (diffuse + specular) * light_color * intensity * n_dot_l;
}

struct SceneOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec2<f32>,
    @location(2) velocity: vec2<f32>,
    /// Diffuse albedo (gamma-encoded base color). Used by post-passes
    /// (SSGI, SSR) to modulate bounce light correctly — indirect
    /// diffuse arriving at a surface is albedo × irradiance, not raw
    /// radiance. Rgba8Unorm is enough precision here.
    @location(3) albedo: vec4<f32>,
};

@fragment
fn fs_main_scene(in: VertexOutputScene) -> SceneOut {
    var n = normalize(in.normal);

    // --- Normal mapping (tangent-space) ---
    // LEADR-lite normal map sample. The texture uploader bakes
    // per-mip normal-direction variance into the alpha channel
    // (see register_texture_kind). RGB holds the vector-averaged
    // unit normal at each mip, so sampling any LOD gives a proper
    // direction for shading; the alpha contains the accumulated
    // (1 - |avg|²) disagreement across the footprint. The shader
    // uses that alpha as an additional σ² term added to GGX α²,
    // widening the lobe by exactly enough to integrate over sub-
    // pixel normal variance before it hits the BRDF as sparkle.
    //
    // We still sample at +1 LOD bias so the hardware picks a mip
    // with more accumulated variance than strictly minimal; the
    // tradeoff is a hair of softness at near-perpendicular views
    // in exchange for path-tracer-like integration at grazing.
    // shadow_cascade_splits.w carries the global LOD bias (-1 when
    // TSR is on, 0 otherwise) — added so half-res rendering still
    // reads texture detail one mip finer than hardware would pick.
    let lod_bias = lighting.shadow_cascade_splits.w;
    let nm_sample4 = textureSampleBias(normal_tex, normal_samp, in.uv, 1.0 + lod_bias);
    let nm_raw = nm_sample4.xyz * 2.0 - 1.0;
    let baked_variance = nm_sample4.w;
    let toksvig_len2 = clamp(dot(nm_raw, nm_raw), 0.01, 1.0);
    let nm_sample = nm_raw * inverseSqrt(toksvig_len2);
    let tlen2 = dot(in.tangent.xyz, in.tangent.xyz);
    if (tlen2 > 0.0001) {
        let t = normalize(in.tangent.xyz);
        let t_ortho = normalize(t - n * dot(n, t));
        let b = cross(n, t_ortho) * in.tangent.w;
        n = normalize(t_ortho * nm_sample.x + b * nm_sample.y + n * nm_sample.z);
    } else {
        let tbn = compute_tbn(in.world_pos, n, in.uv);
        n = normalize(tbn * nm_sample);
    }

    // --- Material sampling ---
    // Base color & emissive textures in glTF are encoded as sRGB, but
    // the bloom texture registrar creates them as Rgba8Unorm (no
    // hardware decode). We decode manually via the 2.2 approximation —
    // matches bloom-reference's convention so the PBR lighting math
    // operates in linear space throughout.
    let base_tex = textureSampleBias(base_color_tex, base_color_samp, in.uv, lod_bias);
    // Vertex color carries the glTF baseColorFactor (linear per spec)
    // when no per-vertex COLOR_0 stream exists, or the linear color
    // attribute when it does. Do NOT srgb-decode it — that gave
    // correct output only in the boundary case where baseColorFactor
    // was (1,1,1,1), and silently darkened every legitimate tint
    // (Bistro's spec-gloss diffuse factors land in the 0.5–0.9 range
    // where the double-conversion is visibly off).
    let base_color = srgb_to_linear_v(base_tex.rgb) * in.color.rgb;
    let base_alpha = base_tex.a * in.color.a;

    // glTF MASK / BLEND alpha mode — discard fragments below the
    // authored cutoff so alpha-cutout foliage, fences, chains, and
    // fabric render as their actual shape instead of opaque billboards.
    // OPAQUE materials carry cutoff = 0 so the branch collapses.
    // BLEND is treated as MASK @ 0.5 (via the loader) pending a real
    // sorted transparent pipeline.
    let alpha_cutoff = material.metal_rough.w;
    if (alpha_cutoff > 0.0 && base_alpha < alpha_cutoff) {
        discard;
    }

    // glTF metallicRoughnessTexture: G=roughness, B=metallic (linear).
    // When the material has no MR texture (metal_rough.z == 0), the
    // binding falls back to an arbitrary scene texture (whatever lives
    // at index 0) — multiplying its random R/G/B into our factors
    // produces incorrect material values. Use the factors directly in
    // that case.
    let mr_tex_sample = textureSample(mr_tex, mr_samp, in.uv);
    let has_mr = material.metal_rough.z > 0.5;
    var roughness_raw = select(
        clamp(material.metal_rough.y, 0.045, 1.0),
        clamp(mr_tex_sample.g * material.metal_rough.y, 0.045, 1.0),
        has_mr,
    );
    // Dielectric roughness floor. Real-world stone, wood, plaster etc.
    // rarely get below ~0.15; when FBX2glTF or similar exporters drop
    // them to 0.05, we get a mirror-like highlight strip on marble
    // columns that Cycles doesn't produce (Sponza column was the tell).
    // Metals keep the original low floor so chrome / gold stay sharp.
    let metallic_raw = select(
        clamp(material.metal_rough.x, 0.0, 1.0),
        clamp(mr_tex_sample.b * material.metal_rough.x, 0.0, 1.0),
        has_mr,
    );
    let metallic = metallic_raw;
    let dielectric_floor = 0.15;
    var roughness = max(roughness_raw,
                        dielectric_floor * (1.0 - metallic));

    // Specular antialiasing. Two sources of variance are folded into
    // GGX α² as additive corrections:
    //
    //   1. Toksvig (Kaplanyan 2016) — texture-level normal variance.
    //      The bilinearly-filtered+mipmapped normal map sample has
    //      length < 1 wherever adjacent normals disagree. σ² =
    //      (1 − r²)/r² is the Lambert-averaged normal variance,
    //      added directly to α² to widen the GGX lobe by exactly
    //      enough to integrate over the detail we can't resolve.
    //
    //   2. Screen-space kernel (Karis 2013) — geometry-level variance
    //      from per-pixel normal derivatives. Smaller cap than the
    //      pre-Toksvig version because Toksvig already handles the
    //      texture case; this term now only covers sharp geometric
    //      edges and tessellation that Toksvig can't see.
    // Toksvig formula from the hardware-bilinear/aniso vector-length
    // shortening, PLUS the per-mip variance baked into alpha during
    // normal-map upload. The baked term is the clean directional-
    // variance estimate; Toksvig adds whatever extra shortening the
    // sampler's bilinear blend produces on top.
    let sigma2_toksvig = (1.0 - toksvig_len2) / toksvig_len2;
    let sigma2_baked = baked_variance / max(1.0 - baked_variance, 0.001);
    let sigma2 = sigma2_toksvig + sigma2_baked;
    var alpha2 = roughness * roughness + sigma2;
    let nm_dx = dpdx(n);
    let nm_dy = dpdy(n);
    let curvature_sq = dot(nm_dx, nm_dx) + dot(nm_dy, nm_dy);
    // Kaplanyan 2016 screen-space kernel. Bumped aggressively: 2.0
    // coefficient / cap 0.9 to kill sparkle on Intel Sponza's sunlit
    // floor tiles where each tile edge has a high-frequency normal-
    // map bump that D_GGX spikes on at a grazing view. Integrates
    // normal variance across a larger screen-space footprint before
    // the BRDF sees it. Tradeoff: subtly softer micro-specular on
    // all surfaces, which matches the path-tracer's multi-ray
    // average.
    let kernel_alpha = min(2.0 * curvature_sq, 0.9);
    alpha2 = min(alpha2 + kernel_alpha, 1.0);
    roughness = sqrt(alpha2);

    let em_tex_sample = textureSample(em_tex, em_samp, in.uv);
    let emissive = srgb_to_linear_v(em_tex_sample.rgb) * material.emissive.rgb;

    // glTF occlusion: R channel, attenuates indirect lighting (IBL
    // diffuse + ambient) only — direct lights and specular IBL are
    // unchanged per spec. Default texture is white (idx 0) so the
    // sample is 1.0 for materials without an occlusion map.
    let occlusion = textureSample(occ_tex, occ_samp, in.uv).r;

    // --- PBR direct lighting ---
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);
    // Seed with ambient light contribution, modulated by base color
    // so white walls pick up a white ambient and darker materials
    // don't get over-brightened. This is the base illumination for
    // surfaces that receive no direct light and are outside the IBL
    // environment's strongest region (e.g. shadowed interiors).
    var lit = lighting.ambient.rgb * lighting.ambient.a * base_color;

    // Legacy primary directional (kept for back-compat). Shadow-
    // mapped: only this primary light casts because we currently
    // render a single shadow map. Multi-cascade or multi-light
    // shadowing is a future addition.
    let shadow_factor = sample_shadow(in.world_pos);
    // Never fully zero direct light — a 10% floor simulates
    // ambient bounce from surrounding surfaces and keeps shadows
    // from going pitch-black regardless of IBL intensity.
    let direct_shadow = mix(0.03, 1.0, shadow_factor);
    let legacy_dir = normalize(lighting.light_dir.xyz);
    lit += shade_pbr(n, v, legacy_dir, lighting.light_color.rgb,
                     lighting.light_dir.w, base_color, metallic, roughness)
         * direct_shadow;

    let dir_count = u32(lighting.dir_light_count.x);
    for (var i = 0u; i < dir_count; i++) {
        let dl = lighting.dir_lights[i];
        let l = normalize(dl.direction.xyz);
        lit += shade_pbr(n, v, l, dl.color.rgb, dl.direction.w,
                         base_color, metallic, roughness);
    }

    let pt_count = u32(lighting.point_light_count.x);
    for (var i = 0u; i < pt_count; i++) {
        let pl = lighting.point_lights[i];
        let to_light = pl.position.xyz - in.world_pos;
        let dist = length(to_light);
        let range = pl.position.w;
        if (dist < range && dist > 0.0) {
            let l = to_light / dist;
            let atten = 1.0 - (dist / range);
            let atten2 = atten * atten;
            lit += shade_pbr(n, v, l, pl.color.rgb, pl.color.w * atten2,
                             base_color, metallic, roughness);
        }
    }

    // --- Split-sum IBL (Karis 2013) ---
    //   IBL_diffuse  = base_color * (1 - kS_avg) * (1 - metallic)
    //                  * env_irradiance(N)
    //   IBL_specular = prefiltered_env(R, roughness)
    //                  * (F0 * brdf.scale + brdf.bias)
    //
    // env_irradiance is approximated by sampling the env map at its
    // smallest mip (heaviest blur — close enough to a cosine-
    // convolved irradiance map for low-frequency diffuse lighting).
    // prefiltered_env samples mip = roughness * (mips-1), where the
    // mip chain was box-filter downsampled. Box filter ≠ true GGX
    // convolution — that's the next refinement — but together with
    // the BRDF LUT it captures the bulk of correct PBR appearance.

    let n_dot_v_ibl = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    // Diffuse irradiance: dedicated cosine-convolved texture populated
    // at env load. Sampling it directly (mip 0) at the fragment normal
    // gives proper Lambertian diffuse — no mip-steal hack on the
    // specular chain, so specular can use every mip for GGX prefilter.
    let mips = f32(textureNumLevels(env_tex));
    let irr_uv = seamless_equirect_uv(dir_to_equirect_uv(n));
    let irradiance = textureSampleLevel(env_diffuse_tex, env_samp, irr_uv, 0.0).rgb
                   * lighting.camera_pos.w;

    // For diffuse IBL, the Schlick-with-roughness approximation
    // (Lazarov 2013) handles the average kS factor at grazing angles.
    let fc_n = pow(1.0 - n_dot_v_ibl, 5.0);
    let f_ibl = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * fc_n;
    let kd = (vec3<f32>(1.0) - f_ibl) * (1.0 - metallic);
    let ibl_diffuse = irradiance * base_color * kd * occlusion;

    // Pre-filtered specular sample at mip = roughness * (mips - 1).
    // All env_tex mips are GGX-prefiltered now that diffuse lives in
    // its own dedicated texture — roughness = 1 samples the smallest,
    // most-blurred mip, and roughness = 0 samples mip 0 (mirror).
    let r = reflect(-v, n);
    let max_spec_mip = max(mips - 1.0, 0.0);
    let prefiltered_env = env_sample_lod(r, roughness * max_spec_mip);

    // BRDF LUT lookup — (NdotV, roughness) → (scale, bias) such that
    // single-scatter specular = env * (F0 * scale + bias).
    // Pre-integrated against GGX so the directional integral is correct.
    let brdf = textureSample(brdf_lut_tex, brdf_lut_samp, vec2<f32>(n_dot_v_ibl, roughness)).rg;
    let single_spec = prefiltered_env * (f0 * brdf.x + vec3<f32>(brdf.y));

    // Multi-scattering compensation (Fdez-Aguera 2019). Single-scatter
    // GGX loses energy at high roughness — light that should bounce
    // around the microsurface gets dropped. We add it back as a second
    // term tinted by F0 * average-scatter, using the BRDF LUT energy
    // total (brdf.x + brdf.y) as 'how much energy did single-scatter
    // capture' so 1 - that_total is what we missed. Visually: rough
    // metals (gold, copper) get noticeably brighter and more saturated.
    // Multi-scatter compensation (Fdez-Aguera 2019, proper form).
    //   E_ss     = brdf.x + brdf.y        single-scatter energy
    //   E_ms     = 1 - E_ss               missing (multi-scatter) energy
    //   F_avg    = F0 + (1-F0)/21         average fresnel (Karis)
    //   F_ms     = F_avg * E_ss / (1 - F_avg * E_ms)   multi-scatter fresnel
    //   ms       = F_ms * E_ms            extra radiance to add back
    // The previous simpler form `1 + f_avg*(1/E_ss - 1)` exploded
    // as E_ss → 0 (rough dielectrics at grazing), blowing the
    // ground out to white.
    let ess = brdf.x + brdf.y;
    let ems = 1.0 - ess;
    let f_avg = f0 + (vec3<f32>(1.0) - f0) * (1.0 / 21.0);
    let f_ms = f_avg * ess / (vec3<f32>(1.0) - f_avg * ems);
    let ms_contribution = f_ms * ems;

    // Specular occlusion (Lagarde 2014, Moving Frostbite to PBR):
    // attenuate IBL specular by a roughness-weighted blend of the glTF
    // AO term and NdotV so smooth dielectrics in enclosed/shadowed
    // cavities stop reflecting bright sky patches that no path-tracer
    // would let through the occluders. For metals and mirrors this is
    // near-identity; for rough surfaces it approaches the AO value.
    let spec_occ = clamp(
        pow(n_dot_v_ibl + occlusion, exp2(-16.0 * roughness - 1.0))
            - 1.0 + occlusion,
        0.0, 1.0,
    );
    let ibl_spec_raw = prefiltered_env
        * (f0 * brdf.x + vec3<f32>(brdf.y) + ms_contribution);

    // Dielectric specular luma cap. Without a proper visibility-aware
    // specular integral, smooth non-metals like marble / varnished wood
    // end up reflecting the HDR's bright-sky region at full intensity
    // even when occluded by intervening geometry (the Intel Sponza
    // column stripe vs Cycles was the smoking gun — proven by a
    // roughness=1 test render where the stripe disappeared).
    // Path-tracers handle this via shadow rays; we approximate by
    // (1) hard luma cap at 0.8 mid-grey — barely visible — and
    // (2) scaling the dielectric spec amplitude by roughness so the
    // polished end of the scale (roughness 0.15-0.35) loses almost all
    // of its IBL specular response. Metals are left alone so chrome
    // and gold keep their full dynamic range.
    let spec_luma = dot(ibl_spec_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let dielectric_factor = 1.0 - metallic;
    let luma_cap = 0.5;
    let cap_scale = select(1.0, luma_cap / max(spec_luma, 0.0001),
                           spec_luma > luma_cap);
    // Roughness attenuation curve for dielectrics: fully off on
    // polished surfaces, on-ramp all the way to roughness 1.0 where
    // the prefiltered blur covers a full hemisphere so a wrong sample
    // is guaranteed to average with its occluded neighbours. This
    // nearly wipes the column stripe without killing specular on
    // rough natural stone — matter with roughness 0.7 still gets
    // ~50% of the IBL spec contribution.
    let dielectric_spec_amp = smoothstep(0.0, 1.0, roughness);
    let dielectric_scale = mix(1.0, cap_scale * dielectric_spec_amp,
                               dielectric_factor);
    // Universal roughness-based spec attenuation. A smooth curved
    // surface with ANY material (even metal) needs visibility-aware
    // specular to avoid bright stripes where the reflection vector
    // happens to sweep across a hot HDR sample. We don't have
    // visibility, so we dial down specular for smooth surfaces
    // regardless of metalness. Roughness 0.15 floor (applied upstream
    // to dielectrics) plus this smoothstep leaves roughness 1.0
    // surfaces untouched and mid-rough surfaces (0.3-0.5) at
    // significant but reduced strength. Metals still get the
    // dielectric_scale (via the metallic-weighted mix), so the
    // combined effect is conservative for both.
    // Universal luma cap: whatever the metallicity or the roughness,
    // the IBL specular contribution for a single fragment can't exceed
    // a hard luma ceiling. Marble / stone columns get their mirror-
    // of-a-bright-sky-strip reflection clipped to something that
    // couldn't survive a path-tracer's visibility integral,
    // and brightly-polished metals lose a little punch (they compensate
    // via direct specular which still uses Fresnel at full strength).
    // Reinhard-style soft luma cap: same 0.3 ceiling as direct spec
    // but smooth rolloff so adjacent pixels don't ping-pong across a
    // hard discontinuity (speckle on sunlit floor tiles with
    // per-pixel roughness / normal-map variation).
    let cap2_luma = dot(ibl_spec_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let cap2 = 1.0 / (1.0 + cap2_luma / 0.3);
    let roughness_amp = smoothstep(0.05, 0.75, roughness);
    let ibl_spec = ibl_spec_raw
        * dielectric_scale * spec_occ * roughness_amp * cap2;

    // Indirect-shadow attenuation. 0.15 — deep enough that windows
    // Shadow darkening floor. Prior 0.15 matched Cycles path-
    // tracer output — physically correct, but visually heavy on
    // screens calibrated against UE5 / Unity renders, which
    // preserve more sky-bounce in shaded regions. 0.35 keeps
    // shadowed areas legible (Sponza atrium under-awning stays
    // 35 % of its indirect-light budget instead of 15 %) without
    // washing out the shadow line. Matches the general look of
    // UE5's Lumen + sky-occlusion and Unity HDRP's ambient
    // probes in Sponza/Bistro test scenes.
    let indirect_shadow = mix(0.35, 1.0, shadow_factor);

    // Multi-scatter also adds a diffuse-like term back from the
    // 'lost' energy, but it gets absorbed wherever there is no metal
    // since dielectrics already account for it via the (1 - kS)
    // diffuse term. The compensation above handles the metal case;
    // dielectric path is unchanged.
    let hdr_raw = lit + (ibl_diffuse + ibl_spec) * indirect_shadow + emissive;

    // Final HDR scrub. Two things the rest of the chain can't
    // recover from:
    //
    // 1. NaN/Inf anywhere upstream (unguarded GGX at α→0 +
    //    n_dot_h→1, multi-scatter `1 / (1 - F_avg·E_ms)` at
    //    grazing smooth metals, env-sample weirdness at UV seams)
    //    — a single poisoned pixel survives TAA's neighborhood
    //    clamp on Metal (clamp(NaN,a,b) is impl-defined) and
    //    tonemaps to pink. Self-compare kills it at source.
    //
    // 2. Specular fireflies from sub-pixel normal-map variance.
    //    The LEADR baked σ² already widens the GGX lobe by the
    //    accumulated mip footprint, but there are still isolated
    //    texels where D_GGX + IBL prefilter spike an order of
    //    magnitude above neighbours. Bloom then amplifies each
    //    spike into a coloured halo. The real root cause of the
    //    stone-floor speckle was the irradiance convolution
    //    shader sampling raw HDR (sun disc unclamped) — with
    //    that fixed this cap only has to catch legitimate
    //    specular outliers. 50 leaves all normal bright content
    //    alone and trims only the rare aliased peak.
    let hdr_clean = select(vec3<f32>(0.0), hdr_raw, hdr_raw == hdr_raw);
    let luma = dot(hdr_clean, vec3<f32>(0.2126, 0.7152, 0.0722));
    let firefly_cap = 50.0;
    let luma_scale = select(1.0, firefly_cap / luma, luma > firefly_cap);
    let hdr = hdr_clean * luma_scale;

    // Per-pixel velocity: difference between current and previous NDC,
    // scaled by 0.5 so the result is in UV-space units. Used by the
    // motion blur pass and TAA per-object reprojection.
    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let vel = (curr_ndc - prev_ndc) * 0.5;

    return SceneOut(
        vec4<f32>(hdr, base_alpha),
        vec2<f32>(metallic, roughness),
        vel,
        // albedo.rgb: base color (SSGI bounce modulation).
        // albedo.a:   1 - shadow_factor — how much of this pixel's
        //             illumination is INDIRECT (IBL + bounce) vs
        //             DIRECT (sun). The compose pass uses this to
        //             apply SSAO only to indirect-dominated pixels
        //             (shadowed corners, overhangs) and leave
        //             sun-lit surfaces alone, which is the physically
        //             correct behaviour for AO (occludes indirect
        //             only). 1.0 where fully shadowed, 0.0 where
        //             sunlit. Sky shader overrides with 0.0.
        vec4<f32>(base_color, 1.0 - shadow_factor),
    );
}
";

pub(super) const PREFILTER_SHADER_WGSL: &str = "
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

pub(super) const SKY_SHADER_WGSL: &str = "
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
pub(super) const BLOOM_SHADER_WGSL: &str = "
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
pub(super) const HIZ_LINEARIZE_SHADER_WGSL: &str = "
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
    if (d >= 0.9999) {
        linear_z = HIZ_SKY_Z;
    } else {
        // view_z = -p32 / (d + p22); store |view_z|.
        linear_z = -u.params.w / (d + u.params.z);
        linear_z = max(linear_z, 0.0001);
    }
    textureStore(hiz_out, vec2<i32>(px), vec4<f32>(linear_z, 0.0, 0.0, 0.0));
}
";

/// Downsample one Hi-Z mip into the next. Uses `min` so the coarser
/// mip reports the nearest occluder in its footprint — exactly what
/// the GTAO horizon scan wants when picking a coarser mip for a
/// far step.
pub(super) const HIZ_DOWNSAMPLE_SHADER_WGSL: &str = "
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
pub(super) const SSAO_SHADER_WGSL: &str = "
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
pub(super) const SSAO_BLUR_SHADER_WGSL: &str = "
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

pub(super) const DOF_SHADER_WGSL: &str = "
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

pub(super) const MOTION_BLUR_SHADER_WGSL: &str = "
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

pub(super) const SSS_SHADER_WGSL: &str = "
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


/// Ticket 013 V3 — Mesh-Cards capture shader with dual render targets.
///
/// Location 0 (albedo) + Location 1 (emissive). Rasterises a mesh
/// orthographically along its assigned signed axis into the card slot,
/// writing baked albedo and emissive into their respective atlases in
/// one draw. Emissive reads the material's emissive texture (if any)
/// and multiplies by the `emissive_factor`. Flat `base_color_factor` +
/// a 1×1 white fallback texture cover the case where the mesh has
/// only a scalar material.
pub(super) const CARD_CAPTURE_WGSL: &str = "
struct CaptureParams {
    ortho_vp: mat4x4<f32>,
    // Mesh's base_color_factor (xyz) + `has_base_texture` flag (w).
    base_color: vec4<f32>,
    // Mesh's emissive_factor (xyz) + `has_emissive_texture` flag (w).
    emissive: vec4<f32>,
};

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FsOut {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: CaptureParams;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_samp: sampler;
@group(1) @binding(2) var emissive_tex: texture_2d<f32>;

@vertex
fn vs_main(v: VertexIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.ortho_vp * vec4<f32>(v.position, 1.0);
    out.uv = v.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var albedo = u.base_color.rgb;
    if (u.base_color.w > 0.5) {
        albedo = albedo * textureSample(albedo_tex, albedo_samp, in.uv).rgb;
    }

    var emissive = u.emissive.rgb;
    if (u.emissive.w > 0.5) {
        emissive = emissive * textureSample(emissive_tex, albedo_samp, in.uv).rgb;
    }

    var out: FsOut;
    out.albedo = vec4<f32>(albedo, 1.0);
    // Rgba8UnormSrgb clamps emissive at 1.0 per channel; that's fine
    // for Sponza-scale lanterns. Pre-divide by EMISSIVE_SCALE at write
    // and multiply back at sample if HDR emissive becomes necessary.
    out.emissive = vec4<f32>(clamp(emissive, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    return out;
}
";

/// Ticket 014 V1 — per-mesh unsigned distance field bake.
///
/// One compute invocation per voxel (32×32×32 = 32768 per mesh). Each
/// lane reads the mesh's vertex + index buffers as storage, iterates
/// all triangles, and computes `min(point-triangle distance)` from the
/// voxel centre. Output is R16Float distance (unsigned — we don't
/// attempt inside/outside classification, which is brittle on open
/// Sponza meshes; sphere-trace works with UDF either way because each
/// march step advances by `d` regardless of sign).
///
/// Vertex stride matches `Vertex3D` (12 f32 = 48 bytes). Only the
/// first 3 floats (position) are read.
pub(super) const SDF_BAKE_WGSL: &str = "
struct SdfBakeParams {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    // x = triangle_count, y = sdf_resolution, zw unused
    counts: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: SdfBakeParams;
@group(0) @binding(1) var<storage, read> vertex_buf: array<f32>;
@group(0) @binding(2) var<storage, read> index_buf: array<u32>;
@group(0) @binding(3) var sdf_out: texture_storage_3d<r32float, write>;

const VERTEX_STRIDE_F32: u32 = 12u;  // Vertex3D: pos(3) + normal(3) + color(4) + uv(2) = 12 f32

fn vtx_pos(idx: u32) -> vec3<f32> {
    let base = idx * VERTEX_STRIDE_F32;
    return vec3<f32>(vertex_buf[base], vertex_buf[base + 1u], vertex_buf[base + 2u]);
}

// Point-triangle distance, clamped-edge form. From Ericson, Real-Time
// Collision Detection. Returns unsigned distance to the closest point
// on triangle abc from point p.
fn point_triangle_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return length(ap); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return length(bp); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return length(p - (a + v * ab));
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return length(cp); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return length(p - (a + w * ac));
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return length(p - (b + w * (c - b)));
    }
    // Closest point in face interior.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return length(p - (a + ab * v + ac * w));
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = u.counts.y;
    if (gid.x >= res || gid.y >= res || gid.z >= res) { return; }

    // Voxel centre in object space.
    let uvw = (vec3<f32>(gid) + vec3<f32>(0.5)) / f32(res);
    let voxel_os = mix(u.aabb_min.xyz, u.aabb_max.xyz, uvw);

    var min_dist = 1e9;
    let tri_count = u.counts.x;
    for (var t: u32 = 0u; t < tri_count; t = t + 1u) {
        let i0 = index_buf[t * 3u + 0u];
        let i1 = index_buf[t * 3u + 1u];
        let i2 = index_buf[t * 3u + 2u];
        let a = vtx_pos(i0);
        let b = vtx_pos(i1);
        let c = vtx_pos(i2);
        let d = point_triangle_distance(voxel_os, a, b, c);
        min_dist = min(min_dist, d);
    }

    textureStore(sdf_out, vec3<i32>(gid), vec4<f32>(min_dist, 0.0, 0.0, 0.0));
}
";

/// Ticket 013 V3 — per-frame card-lighting compute pass with
/// shadow-cascade sampling + emissive contribution.
///
/// Per texel: reconstruct the world-space position of the card point
/// (slot metadata's aabb + axis + mesh transform), sample the shadow
/// cascade at that point, compute direct + sky + emissive. Writes to
/// the radiance atlas. Shadow lookup makes indirect bounce meaningfully
/// darker on sun-occluded card faces (arches, column undersides).
pub(super) const CARD_LIGHT_WGSL: &str = "
struct SlotMeta {
    // xyz = world-space card-face normal, w = signed axis (0..6 as f32)
    normal_ws: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    transform: mat4x4<f32>,
};

struct CardLightParams {
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    atlas_info: vec4<u32>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    view_matrix: mat4x4<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: CardLightParams;
@group(0) @binding(1) var albedo_atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;
@group(0) @binding(3) var<storage, read> slot_meta: array<SlotMeta>;
@group(0) @binding(4) var radiance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var emissive_atlas: texture_2d<f32>;
@group(0) @binding(6) var shadow_atlas_0: texture_depth_2d;
@group(0) @binding(7) var shadow_atlas_1: texture_depth_2d;
@group(0) @binding(8) var shadow_atlas_2: texture_depth_2d;
@group(0) @binding(9) var shadow_samp: sampler_comparison;

fn sample_cascade(cascade: i32, pos_ws: vec3<f32>, bias: f32) -> f32 {
    var clip: vec4<f32>;
    if (cascade == 0) {
        clip = u.shadow_vps[0] * vec4<f32>(pos_ws, 1.0);
    } else if (cascade == 1) {
        clip = u.shadow_vps[1] * vec4<f32>(pos_ws, 1.0);
    } else {
        clip = u.shadow_vps[2] * vec4<f32>(pos_ws, 1.0);
    }
    let ndc = clip.xyz / clip.w;
    // Outside the cascade frustum → treat as lit (no shadow).
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let ref_depth = ndc.z - bias;
    if (cascade == 0) {
        return textureSampleCompareLevel(shadow_atlas_0, shadow_samp, shadow_uv, ref_depth);
    } else if (cascade == 1) {
        return textureSampleCompareLevel(shadow_atlas_1, shadow_samp, shadow_uv, ref_depth);
    } else {
        return textureSampleCompareLevel(shadow_atlas_2, shadow_samp, shadow_uv, ref_depth);
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.xy;
    let atlas_sz = u.atlas_info.x;
    if (px.x >= atlas_sz || px.y >= atlas_sz) { return; }

    let slot_sz = u.atlas_info.y;
    let slots_per_row = u.atlas_info.z;
    let active_count = u.atlas_info.w;

    let slot_x = px.x / slot_sz;
    let slot_y = px.y / slot_sz;
    let slot_idx = slot_y * slots_per_row + slot_x;

    if (slot_idx >= active_count) {
        textureStore(radiance_out, vec2<i32>(px), vec4<f32>(0.0));
        return;
    }

    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / f32(atlas_sz);
    let albedo = textureSampleLevel(albedo_atlas, atlas_samp, uv, 0.0).rgb;
    let emissive = textureSampleLevel(emissive_atlas, atlas_samp, uv, 0.0).rgb;

    let slot_m = slot_meta[slot_idx];
    let n_ws = slot_m.normal_ws.xyz;
    let axis = u32(slot_m.normal_ws.w);

    // Local slot UV (0..1 inside this slot) → object-space card-plane
    // position. Signed-axis-aware u-flip matches the capture pass's
    // projection so the texel we light is the one the capture baked.
    let sx = f32((px.x) % slot_sz);
    let sy = f32((px.y) % slot_sz);
    let sd = f32(slot_sz);
    var u_norm = (sx + 0.5) / sd;
    let v_norm = (sy + 0.5) / sd;
    var pos_os = vec3<f32>(0.0);
    if (axis == 0u || axis == 1u) {
        // Card plane at x = bmax.x (+X) or bmin.x (-X); u=y, v=z.
        if (axis == 1u) { u_norm = 1.0 - u_norm; }
        pos_os.y = mix(slot_m.aabb_min.y, slot_m.aabb_max.y, u_norm);
        pos_os.z = mix(slot_m.aabb_min.z, slot_m.aabb_max.z, v_norm);
        if (axis == 0u) { pos_os.x = slot_m.aabb_max.x; } else { pos_os.x = slot_m.aabb_min.x; }
    } else if (axis == 2u || axis == 3u) {
        if (axis == 3u) { u_norm = 1.0 - u_norm; }
        pos_os.x = mix(slot_m.aabb_min.x, slot_m.aabb_max.x, u_norm);
        pos_os.z = mix(slot_m.aabb_min.z, slot_m.aabb_max.z, v_norm);
        if (axis == 2u) { pos_os.y = slot_m.aabb_max.y; } else { pos_os.y = slot_m.aabb_min.y; }
    } else {
        if (axis == 5u) { u_norm = 1.0 - u_norm; }
        pos_os.x = mix(slot_m.aabb_min.x, slot_m.aabb_max.x, u_norm);
        pos_os.y = mix(slot_m.aabb_min.y, slot_m.aabb_max.y, v_norm);
        if (axis == 4u) { pos_os.z = slot_m.aabb_max.z; } else { pos_os.z = slot_m.aabb_min.z; }
    }
    let pos_ws = (slot_m.transform * vec4<f32>(pos_os, 1.0)).xyz;

    // Shadow. Cascade selection uses view-space Z against splits.
    var shadow: f32 = 1.0;
    if (u.flags.y > 0.5) {
        let view_z = -(u.view_matrix * vec4<f32>(pos_ws, 1.0)).z;
        var cascade: i32 = 2;
        if (view_z <= u.shadow_splits.x) {
            cascade = 0;
        } else if (view_z <= u.shadow_splits.y) {
            cascade = 1;
        }
        shadow = sample_cascade(cascade, pos_ws, u.flags.x);
    }

    let ndotl = max(dot(n_ws, u.sun_dir.xyz), 0.0);
    let direct = u.sun_color.xyz * ndotl * shadow;
    let ndotup = max(dot(n_ws, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    let sky = u.sky_color.xyz * ndotup;

    let lit = albedo * (direct + sky) + emissive;
    textureStore(radiance_out, vec2<i32>(px), vec4<f32>(lit, 1.0));
}
";

/// Ticket 014 V6 — World-Space Radiance Cache bake.
///
/// One workgroup per probe (16×16×16 probes on a 120 m camera-following
/// cube), 8×8 threads per workgroup = one octel texel each. Output is
/// the flat rgba16float atlas read by the SDF miss path.
///
/// Per texel: analytic sun term for the octel direction, gated by a
/// shadow-cascade lookup at the probe's world position (position-
/// varying occlusion — the crucial signal that makes the cache worth
/// having; flat analytic would mean every probe holds the same
/// content). Analytic sky hemisphere for up-biased directions. No
/// per-direction geometry trace — this is the "distant envelope",
/// not another SDF march.
pub(super) const WSRC_BAKE_WGSL: &str = "
struct WsrcBakeParams {
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    // xyz = clipmap origin (world-space cube centre), w = full extent
    grid: vec4<f32>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    // x = shadow bias, y = shadows-enabled flag, zw unused
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: WsrcBakeParams;
@group(0) @binding(1) var shadow_atlas_0: texture_depth_2d;
@group(0) @binding(2) var shadow_atlas_1: texture_depth_2d;
@group(0) @binding(3) var shadow_atlas_2: texture_depth_2d;
@group(0) @binding(4) var shadow_samp: sampler_comparison;
@group(0) @binding(5) var wsrc_out: texture_storage_3d<rgba16float, write>;

fn wsrc_sample_cascade(cascade: i32, pos_ws: vec3<f32>, bias: f32) -> f32 {
    var clip: vec4<f32>;
    if (cascade == 0) {
        clip = u.shadow_vps[0] * vec4<f32>(pos_ws, 1.0);
    } else if (cascade == 1) {
        clip = u.shadow_vps[1] * vec4<f32>(pos_ws, 1.0);
    } else {
        clip = u.shadow_vps[2] * vec4<f32>(pos_ws, 1.0);
    }
    let ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let ref_depth = ndc.z - bias;
    if (cascade == 0) {
        return textureSampleCompareLevel(shadow_atlas_0, shadow_samp, shadow_uv, ref_depth);
    } else if (cascade == 1) {
        return textureSampleCompareLevel(shadow_atlas_1, shadow_samp, shadow_uv, ref_depth);
    } else {
        return textureSampleCompareLevel(shadow_atlas_2, shadow_samp, shadow_uv, ref_depth);
    }
}

// V10 — workgroup writes the 10×10 padded slab per probe. Thread
// (lid.x, lid.y) writes texel (wg.*10 + lid) in the atlas; border
// threads (lid on 0 or 9) shade for the nearest INSIDE octel
// direction so the sampler's edge-extend behaviour is baked into
// the data. The 1-texel border is what lets the hardware bilinear
// sampler do octel smoothing without leaking into adjacent probes.
const WSRC_OCT_PADDED_SIZE: u32 = 10u;

@compute @workgroup_size(10, 10, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_res: u32 = 16u;
    if (wg.x >= grid_res || wg.y >= grid_res || wg.z >= grid_res) { return; }
    if (lid.x >= WSRC_OCT_PADDED_SIZE || lid.y >= WSRC_OCT_PADDED_SIZE) { return; }

    // Probe world-space centre — cell-centre within the grid cube.
    let extent = u.grid.w;
    let cell = extent / f32(grid_res);
    let probe_pos = u.grid.xyz
        - vec3<f32>(extent * 0.5)
        + (vec3<f32>(f32(wg.x), f32(wg.y), f32(wg.z)) + vec3<f32>(0.5)) * cell;

    // V11 — map padded octel → real octel with true octahedral
    // silhouette wrap on the 4 edges. Beyond v<0 or v>1 in octel uv
    // space the octahedron folds onto itself with u ↔ 1-u; likewise
    // u<0 or u>1 folds with v ↔ 1-v. Corners (both axes out) keep
    // the V10 edge-extend fill since the double-fold has two valid
    // representations and the exact corner only matters when the
    // sampler bilinear-weights it near zero anyway.
    let px = i32(lid.x);
    let py = i32(lid.y);
    let is_edge_x = px == 0 || px == 9;
    let is_edge_y = py == 0 || py == 9;
    var real_ox: i32;
    var real_oy: i32;
    if (is_edge_x && is_edge_y) {
        // Corner — nearest-inside (edge-extend).
        real_ox = clamp(px - 1, 0, 7);
        real_oy = clamp(py - 1, 0, 7);
    } else if (is_edge_y) {
        // Top/bottom border: mirror x across the edge, same row.
        real_ox = 8 - px;
        real_oy = clamp(py - 1, 0, 7);
    } else if (is_edge_x) {
        // Left/right border: same column, mirror y across the edge.
        real_ox = clamp(px - 1, 0, 7);
        real_oy = 8 - py;
    } else {
        // Interior — direct mapping.
        real_ox = px - 1;
        real_oy = py - 1;
    }
    let dir = octel_direction(vec2<u32>(u32(real_ox), u32(real_oy)));

    // Shadow at the probe position (cascade 2 — widest, covers the
    // full 120 m cube without per-probe cascade selection).
    var shadow: f32 = 1.0;
    if (u.flags.y > 0.5) {
        shadow = wsrc_sample_cascade(2, probe_pos, u.flags.x);
    }

    let ndotl = max(dot(dir, u.sun_dir.xyz), 0.0);
    let sun = u.sun_color.xyz * ndotl * shadow;
    let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sky = u.sky_color.xyz * up * up;

    let radiance = sun + sky;

    let tex_coord = vec3<i32>(
        i32(wg.x * WSRC_OCT_PADDED_SIZE + lid.x),
        i32(wg.y * WSRC_OCT_PADDED_SIZE + lid.y),
        i32(wg.z),
    );
    textureStore(wsrc_out, tex_coord, vec4<f32>(radiance, 1.0));
}
";

// ============================================================================
// Ticket 007a — Lumen-style screen-probe SSGI (software Hi-Z trace)
//
// One probe per 16×16 half-res-pixel tile. Each probe stores 64 radiance
// samples in an 8×8 octahedral atlas. Passes: place → trace → temporal →
// resolve. The resolve pass writes the legacy `ssgi_rt` so downstream
// compositing is untouched.
// ============================================================================

/// Shared helpers prepended to every probe compute/fragment shader.
/// Contains octahedral encode/decode, view-space reconstruction, and
/// the Hi-Z sample helper. Kept as a Rust &str so it can be prepended
/// in the shader-module setup without a WGSL include mechanism.
pub(super) const PROBE_HELPERS_WGSL: &str = "
const PROBE_TILE_SIZE: u32 = 16u;
const PROBE_OCT_SIZE: u32 = 8u;
const PROBE_OCT_TEXELS: u32 = 64u;
const HIZ_SKY_Z: f32 = 10000.0;
const PI: f32 = 3.14159265;

struct ProbeHeader {
    // xyz = world-space probe position; w = valid (1.0 = on surface, 0.0 = sky/invalid)
    world_pos: vec4<f32>,
    // xyz = world-space normal at the probe surface; w = linear |view-z|
    normal: vec4<f32>,
};

fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    let s = vec2<f32>(
        select(-1.0, 1.0, v.x >= 0.0),
        select(-1.0, 1.0, v.y >= 0.0),
    );
    return (1.0 - abs(vec2<f32>(v.y, v.x))) * s;
}

fn oct_encode(n_in: vec3<f32>) -> vec2<f32> {
    let n = n_in / (abs(n_in.x) + abs(n_in.y) + abs(n_in.z));
    let xy = select(oct_wrap(n.xy), n.xy, n.z >= 0.0);
    return xy * 0.5 + 0.5;
}

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    n.x = n.x + select(t, -t, n.x >= 0.0);
    n.y = n.y + select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn octel_direction(octel: vec2<u32>) -> vec3<f32> {
    let uv = (vec2<f32>(octel) + vec2<f32>(0.5)) / f32(PROBE_OCT_SIZE);
    return oct_decode(uv);
}

fn view_pos_from_linear(uv: vec2<f32>, linear_z: f32,
                        p00: f32, p11: f32, p20: f32, p21: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let view_z = -linear_z;
    let view_x = -(ndc_x + p20) * view_z / p00;
    let view_y = -(ndc_y + p21) * view_z / p11;
    return vec3<f32>(view_x, view_y, view_z);
}

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
";

/// Probe placement. One workgroup invocation per probe tile writes a
/// ProbeHeader (world position + world normal + linear view-z). Sky
/// probes are flagged invalid (world_pos.w = 0). Per-frame Halton-style
/// jitter moves the probe within its 16×16 tile so adjacent frames
/// cover slightly different surface points, widening effective
/// coverage when combined with temporal accumulation.
pub(super) const SSGI_PROBE_PLACE_WGSL: &str = "
struct PlaceParams {
    // Full inverse view matrix — used to lift view-space positions/normals
    // back into world space so the trace can march across the scene.
    inv_view: mat4x4<f32>,
    // x = proj[0][0], y = proj[1][1], z = proj[2][0], w = proj[2][1]
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = frame_index (temporal jitter), y = tile_size_f (16.0), zw unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: PlaceParams;
@group(0) @binding(1) var hiz0: texture_2d<f32>;
@group(0) @binding(2) var hiz_samp: sampler;
@group(0) @binding(3) var<storage, read_write> probes: array<ProbeHeader>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (gid.x >= grid_w || gid.y >= grid_h) { return; }

    let probe_idx = gid.y * grid_w + gid.x;
    let half_w = f32(u.size.x);
    let half_h = f32(u.size.y);
    let tile = u.params.y;
    let frame = u.params.x;

    // Jitter UV inside the tile — 50% of tile radius so probe stays
    // comfortably away from tile borders. Golden-ratio offsets across
    // frames decorrelate the jitter from TAA/SSAO patterns.
    let jx = ign(vec2<f32>(f32(gid.x) + frame * 1.618, f32(gid.y)));
    let jy = ign(vec2<f32>(f32(gid.x), f32(gid.y) + frame * 2.236));
    let px_x = f32(gid.x) * tile + tile * 0.5 + (jx - 0.5) * tile * 0.5;
    let px_y = f32(gid.y) * tile + tile * 0.5 + (jy - 0.5) * tile * 0.5;
    let uv = vec2<f32>(px_x / half_w, px_y / half_h);

    let linear_z = textureSampleLevel(hiz0, hiz_samp, uv, 0.0).r;

    // Sky probe — mark invalid and bail.
    if (linear_z >= HIZ_SKY_Z * 0.5) {
        probes[probe_idx].world_pos = vec4<f32>(0.0);
        probes[probe_idx].normal = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        return;
    }

    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;
    let P = view_pos_from_linear(uv, linear_z, p00, p11, p20, p21);

    // Finite-difference normal from 3-tap view-pos cross product. One
    // texel to the right and one up. Uses the same Hi-Z mip 0 the
    // center tap read from.
    let texel = vec2<f32>(1.0 / half_w, 1.0 / half_h);
    let uv_r = uv + vec2<f32>(texel.x, 0.0);
    let uv_u = uv + vec2<f32>(0.0, -texel.y);
    let zr = textureSampleLevel(hiz0, hiz_samp, uv_r, 0.0).r;
    let zu = textureSampleLevel(hiz0, hiz_samp, uv_u, 0.0).r;
    let P_r = view_pos_from_linear(uv_r, zr, p00, p11, p20, p21);
    let P_u = view_pos_from_linear(uv_u, zu, p00, p11, p20, p21);
    let N_vs = normalize(cross(P_r - P, P_u - P));

    let P_world = (u.inv_view * vec4<f32>(P, 1.0)).xyz;
    let N_world = normalize((u.inv_view * vec4<f32>(N_vs, 0.0)).xyz);

    probes[probe_idx].world_pos = vec4<f32>(P_world, 1.0);
    probes[probe_idx].normal = vec4<f32>(N_world, linear_z);
}
";

/// Probe trace, software (Hi-Z) path.
///
/// One workgroup per probe; each of the 64 lanes handles one octahedral
/// texel = one ray direction. Hemisphere-cull: rays below the probe's
/// tangent plane contribute zero (not visible from this surface
/// orientation). Surviving rays march the Hi-Z depth pyramid in view
/// space and sample the HDR buffer at hit. Misses contribute zero —
/// sky/off-screen handling is the compose pass's job downstream.
pub(super) const SSGI_PROBE_TRACE_SW_WGSL: &str = "
struct TraceParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = frame_index, y = intensity, z = max_march_t_world, w = firefly_cap
    params: vec4<f32>,
    // Ticket 014 V3/V6 — rest of the shared `ProbeTraceParams` layout.
    // Ignored by Hi-Z; present only so the shader struct size matches
    // the host uniform buffer.
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    clipmap: vec4<f32>,
    wsrc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TraceParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var hiz0: texture_2d<f32>;
@group(0) @binding(3) var hiz1: texture_2d<f32>;
@group(0) @binding(4) var hiz2: texture_2d<f32>;
@group(0) @binding(5) var hiz3: texture_2d<f32>;
@group(0) @binding(6) var hiz4: texture_2d<f32>;
@group(0) @binding(7) var hiz_samp: sampler;
@group(0) @binding(8) var hdr_tex: texture_2d<f32>;
@group(0) @binding(9) var hdr_samp: sampler;
@group(0) @binding(10) var radiance_out: texture_storage_3d<rgba16float, write>;

fn hiz_sample(uv: vec2<f32>, mip: i32) -> f32 {
    switch (clamp(mip, 0, 4)) {
        case 0: { return textureSampleLevel(hiz0, hiz_samp, uv, 0.0).r; }
        case 1: { return textureSampleLevel(hiz1, hiz_samp, uv, 0.0).r; }
        case 2: { return textureSampleLevel(hiz2, hiz_samp, uv, 0.0).r; }
        case 3: { return textureSampleLevel(hiz3, hiz_samp, uv, 0.0).r; }
        default: { return textureSampleLevel(hiz4, hiz_samp, uv, 0.0).r; }
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }
    if (lid.x >= PROBE_OCT_SIZE || lid.y >= PROBE_OCT_SIZE) { return; }

    let probe_idx = wg.y * grid_w + wg.x;
    let header = probes[probe_idx];

    let dst_coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));

    // Invalid probe (sky tile) → contribute zero.
    if (header.world_pos.w < 0.5) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    let dir_ws = octel_direction(lid.xy);
    let n_ws = header.normal.xyz;

    // Hemisphere cull — rays pointing below the surface carry no diffuse contribution.
    let ndotd = dot(dir_ws, n_ws);
    if (ndotd <= 0.0) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // Trace in view space so the Hi-Z march lines up directly with the
    // rasterized depth pyramid. Start origin at probe_pos + small normal
    // offset to avoid self-intersection at shading point.
    let origin_ws = header.world_pos.xyz + n_ws * 0.02;
    let origin_vs = (u.view * vec4<f32>(origin_ws, 1.0)).xyz;
    let dir_vs = normalize((u.view * vec4<f32>(dir_ws, 0.0)).xyz);

    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;

    let max_t = u.params.z;
    var t = 0.05;
    let n_steps: i32 = 14;
    let growth = pow(max_t / t, 1.0 / f32(n_steps));

    var hit_color = vec3<f32>(0.0);
    var prev_t = 0.0;

    for (var s = 0; s < n_steps; s = s + 1) {
        let pt_vs = origin_vs + dir_vs * t;
        let clip = u.proj * vec4<f32>(pt_vs, 1.0);
        let ndc = clip.xyz / clip.w;

        // Off-screen — no hit possible, stop.
        if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
            break;
        }

        let ray_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

        // Pick the mip such that step footprint ≈ one mip texel. Longer
        // steps sample coarser mips so the early-out fires at coarse
        // resolution; only the last few steps hit mip 0.
        let step_size = t - prev_t;
        let mip = clamp(i32(floor(log2(max(step_size / 0.05, 1.0)))), 0, 4);

        let scene_z = hiz_sample(ray_uv, mip);
        // Hi-Z stores positive |view-z|. ray view-z is negative.
        let ray_abs_z = -pt_vs.z;

        // Have we marched behind a surface? The step tolerance scales
        // with step size: the final tolerance (step_size * 2 + 0.1)
        // lets coarse steps accept wider thickness, which matches the
        // existing SSGI behaviour closely enough for V1.
        if (ray_abs_z >= scene_z && scene_z < HIZ_SKY_Z * 0.5) {
            // Refine against mip 0 to reject far-off misses.
            let refined_z = hiz_sample(ray_uv, 0);
            let thickness = abs(ray_abs_z - refined_z);
            if (thickness < step_size * 2.0 + 0.1) {
                let tn = t / max_t;
                let falloff = 1.0 - tn * tn;
                var raw = textureSampleLevel(hdr_tex, hdr_samp, ray_uv, 0.0).rgb * max(falloff, 0.0);
                // Firefly clamp (cap per-sample luma).
                let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
                let cap = u.params.w;
                if (luma > cap) { raw = raw * (cap / luma); }
                hit_color = raw;
            }
            break;
        }

        prev_t = t;
        t = t * growth;
    }

    let intensity = u.params.y;
    textureStore(radiance_out, dst_coord, vec4<f32>(hit_color * intensity * ndotd, 1.0));
}
";

/// Probe trace, hardware (ray-query) path (ticket 007b).
///
/// Same workgroup shape as the SW shader — one workgroup per probe, 64
/// lanes per probe, each handling one octahedral texel. The per-ray
/// inner loop replaces Hi-Z screen-space marching with `rayQuery`
/// against the TLAS, which pulls off-screen geometry into the bounce
/// (the whole point of HW-RT here).
///
/// Hit shading is "hit-lighting-lite":
///   - flat per-instance albedo + world-space normal from
///     `instance_data[hit.instance_custom_data]`;
///   - sun direct: NdotL × sun_color (no cascade shadow lookup in V1 —
///     the bias is one bounce away and hidden by temporal averaging;
///     re-add when shadow-aware hit shading becomes worth the cost);
///   - sky: max(dot(N, up), 0) × sky_color for the upward hemisphere;
///   - emissive: per-instance scalar × albedo;
///   - distance falloff and firefly clamp match the SW path so the
///     two trace variants are visually interchangeable where they
///     both see on-screen geometry.
pub(super) const SSGI_PROBE_TRACE_HW_WGSL: &str = "
struct TraceParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    size: vec4<u32>,
    params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    // Ticket 014 V3/V6 — clipmap + WSRC padding. Ignored here; present
    // only so the struct size matches the host-side `ProbeTraceParams`
    // uniform buffer layout (HW path uses ray-query and its own miss
    // sky, not WSRC).
    clipmap: vec4<f32>,
    wsrc: vec4<f32>,
};

struct InstanceGiData {
    albedo: vec3<f32>,
    emissive_luma: f32,
    normal_ws: vec3<f32>,
    _pad0: f32,
    // Ticket 013 V2: x = first_slot_index (first of 6 consecutive
    // signed-axis slots), yz unused, w = has_card flag.
    card_slot: vec4<f32>,
    // Object-space AABB min (xyz) / max (xyz).
    card_aabb_min: vec4<f32>,
    card_aabb_max: vec4<f32>,
};

const CARD_SLOTS_PER_ROW: f32 = 64.0;
const HW_WSRC_GRID_RES: i32 = 16;

@group(0) @binding(0) var<uniform> u: TraceParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var accel: acceleration_structure;
@group(0) @binding(3) var<storage, read> instance_data: array<InstanceGiData>;
@group(0) @binding(4) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var card_atlas: texture_2d<f32>;
@group(0) @binding(6) var card_samp: sampler;
@group(0) @binding(7) var wsrc_atlas: texture_3d<f32>;
@group(0) @binding(8) var wsrc_samp: sampler;

// Ticket 014 V7/V8 — WSRC lookup shared with the SDF path. V8
// trilinear across the 8 neighbouring probes, nearest octel for
// direction. extent=0 is the cache-not-ready sentinel that the
// host writes before the first bake completes, so the HW miss
// falls back to the pre-V7 return-black behaviour.
// Ticket 014 V10 — HW mirror of the SDF sampler-based WSRC lookup.
// See the SDF path's comment block for layout + uv derivation.
fn hw_wsrc_sample_probe(gx: i32, gy: i32, gz_f: f32, ru: vec2<f32>) -> vec3<f32> {
    let gxc = clamp(gx, 0, 15);
    let gyc = clamp(gy, 0, 15);
    let ax = (f32(gxc) + 0.1 + ru.x * 0.8) / 16.0;
    let ay = (f32(gyc) + 0.1 + ru.y * 0.8) / 16.0;
    let az = gz_f / 16.0;
    return textureSampleLevel(wsrc_atlas, wsrc_samp,
        vec3<f32>(ax, ay, az), 0.0).rgb;
}

fn hw_wsrc_sample(pos_ws: vec3<f32>, dir_ws: vec3<f32>) -> vec3<f32> {
    let origin = u.wsrc.xyz;
    let extent = u.wsrc.w;
    if (extent <= 0.0) {
        return vec3<f32>(0.0);
    }
    let cell = extent / 16.0;
    let rel = pos_ws - origin + vec3<f32>(extent * 0.5);
    let pf = rel / cell - vec3<f32>(0.5);
    let pfx = floor(pf.x);
    let pfy = floor(pf.y);
    let gix = i32(pfx);
    let giy = i32(pfy);
    let fx = pf.x - pfx;
    let fy = pf.y - pfy;
    let gz_f = clamp(pf.z + 0.5, 0.5, 15.5);

    let ru = oct_encode(dir_ws);

    let c00 = hw_wsrc_sample_probe(gix,     giy,     gz_f, ru);
    let c10 = hw_wsrc_sample_probe(gix + 1, giy,     gz_f, ru);
    let c01 = hw_wsrc_sample_probe(gix,     giy + 1, gz_f, ru);
    let c11 = hw_wsrc_sample_probe(gix + 1, giy + 1, gz_f, ru);

    let ix = 1.0 - fx;
    let iy = 1.0 - fy;
    return c00 * (ix * iy) + c10 * (fx * iy)
         + c01 * (ix * fy) + c11 * (fx * fy);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }
    if (lid.x >= PROBE_OCT_SIZE || lid.y >= PROBE_OCT_SIZE) { return; }

    let probe_idx = wg.y * grid_w + wg.x;
    let header = probes[probe_idx];

    let dst_coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));

    if (header.world_pos.w < 0.5) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    let dir_ws = octel_direction(lid.xy);
    let n_ws = header.normal.xyz;
    let ndotd = dot(dir_ws, n_ws);
    if (ndotd <= 0.0) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // 2 cm normal offset — matches the SW start_t and keeps primary
    // hits from self-intersecting the surface the probe sits on.
    let origin_ws = header.world_pos.xyz + n_ws * 0.02;
    let max_t = u.params.z;

    var rq: ray_query;
    rayQueryInitialize(&rq, accel, RayDesc(
        0u,
        0xFFu,
        0.001,
        max_t,
        origin_ws,
        dir_ws,
    ));
    loop {
        if (!rayQueryProceed(&rq)) { break; }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);

    var radiance = vec3<f32>(0.0);
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        let inst = instance_data[hit.instance_custom_data];

        // Ticket 013 V2 — pick the axis facing the incoming ray and
        // sample the pre-lit radiance atlas. Each mesh has 6
        // consecutive slots laid out by signed axis (see host-side
        // capture loop). The card's world-space normal was baked at
        // capture into `card_slot_meta`; lighting was applied once
        // per frame by `card_light_pass`, so the sample IS the
        // bounce contribution — no hit-time shading math.
        if (inst.card_slot.w > 0.5) {
            let hit_world = origin_ws + dir_ws * hit.t;
            let hit_os = (hit.world_to_object * vec4<f32>(hit_world, 1.0)).xyz;
            // Dominant component of the world-space ray direction
            // picks the major axis; its sign selects the front/back
            // face of that axis.
            let abs_d = abs(dir_ws);
            var axis_idx: u32 = 0u;
            if (abs_d.y >= abs_d.x && abs_d.y >= abs_d.z) {
                axis_idx = 2u;
            } else if (abs_d.z >= abs_d.x) {
                axis_idx = 4u;
            }
            // Sign: ray going -X (dir.x < 0) lands on +X face → axis + 0.
            // ray going +X lands on -X face → axis + 1.
            var signed_axis: u32 = axis_idx;
            if (axis_idx == 0u && dir_ws.x > 0.0) { signed_axis = 1u; }
            else if (axis_idx == 2u && dir_ws.y > 0.0) { signed_axis = 3u; }
            else if (axis_idx == 4u && dir_ws.z > 0.0) { signed_axis = 5u; }

            let first_slot = u32(inst.card_slot.x);
            let slot = first_slot + signed_axis;
            let slot_x = slot % 64u;
            let slot_y = slot / 64u;

            // Project hit_os onto the card plane — same math as V1 but
            // with signed-axis-aware u sign flips so the ±pair cards
            // pick up opposite views of the mesh.
            let bmin = inst.card_aabb_min.xyz;
            let bmax = inst.card_aabb_max.xyz;
            var u_os: f32;
            var v_os: f32;
            var u_lo: f32;
            var u_hi: f32;
            var v_lo: f32;
            var v_hi: f32;
            var u_flip: f32 = 1.0;
            if (signed_axis == 0u || signed_axis == 1u) {
                u_os = hit_os.y; v_os = hit_os.z;
                u_lo = bmin.y; u_hi = bmax.y; v_lo = bmin.z; v_hi = bmax.z;
                if (signed_axis == 1u) { u_flip = -1.0; }
            } else if (signed_axis == 2u || signed_axis == 3u) {
                u_os = hit_os.x; v_os = hit_os.z;
                u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.z; v_hi = bmax.z;
                if (signed_axis == 3u) { u_flip = -1.0; }
            } else {
                u_os = hit_os.x; v_os = hit_os.y;
                u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.y; v_hi = bmax.y;
                if (signed_axis == 5u) { u_flip = -1.0; }
            }
            var u_norm = clamp((u_os - u_lo) / max(u_hi - u_lo, 1e-4), 0.0, 1.0);
            let v_norm = clamp((v_os - v_lo) / max(v_hi - v_lo, 1e-4), 0.0, 1.0);
            if (u_flip < 0.0) { u_norm = 1.0 - u_norm; }

            let slot_size_uv = 1.0 / CARD_SLOTS_PER_ROW;
            let texel_in_slot = slot_size_uv / f32(64);  // 64×64 card
            let slot_u0 = f32(slot_x) * slot_size_uv + texel_in_slot;
            let slot_v0 = f32(slot_y) * slot_size_uv + texel_in_slot;
            let slot_span = slot_size_uv - 2.0 * texel_in_slot;
            let atlas_uv = vec2<f32>(
                slot_u0 + u_norm * slot_span,
                slot_v0 + v_norm * slot_span,
            );
            let pre_lit = textureSampleLevel(card_atlas, card_samp, atlas_uv, 0.0).rgb;

            // V3 — emissive is pre-added into the radiance atlas by the
            // card-lighting pass, so the hit simply picks up the full
            // pre-lit texel and applies distance falloff + firefly cap.
            let tn = hit.t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = pre_lit * falloff;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        } else {
            // Fallback path — no card captured, use the flat
            // instance albedo × hit-time lighting same as 007b.
            let hit_n = inst.normal_ws;
            let ndotl = max(dot(hit_n, u.sun_dir.xyz), 0.0);
            let direct = u.sun_color.xyz * ndotl;
            let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
            let sky = u.sky_color.xyz * ndotup;
            let tn = hit.t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = inst.albedo * (direct + sky) * falloff
                    + inst.albedo * inst.emissive_luma;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        }
    } else {
        // Ticket 014 V7 — miss path samples the WSRC envelope so HW
        // traces that escape scene geometry still contribute sky /
        // sun-visibility signal. Terminal position is the ray's full
        // march distance; direction picks the octel on the nearest
        // probe.
        let terminal = origin_ws + dir_ws * max_t;
        var raw = hw_wsrc_sample(terminal, dir_ws);
        let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
        let cap = u.params.w;
        if (luma > cap) { raw = raw * (cap / luma); }
        radiance = raw;
    }

    let intensity = u.params.y;
    textureStore(radiance_out, dst_coord, vec4<f32>(radiance * intensity * ndotd, 1.0));
}
";

/// Ticket 014 V3 — probe trace, software SDF sphere-march path.
///
/// Third trace variant alongside the 007a Hi-Z screen-space and 007b
/// HW ray-query paths. Same workgroup shape (8×8 = one workgroup per
/// probe, each lane handles one octahedral texel). Only the per-ray
/// inner loop differs: instead of marching screen-space depth or
/// firing a `rayQuery`, each lane sphere-marches the scene-wide SDF
/// clipmap baked by ticket 014 V2.
///
/// Hit shading is intentionally minimal for V3 — no mesh-card lookup
/// (the clipmap is a merged SDF with no per-instance identity). At
/// hit we estimate the surface normal by finite-differencing the SDF
/// clipmap around the hit point, then apply analytic sun × NdotL +
/// sky × NdotUp against a constant gray albedo. That gives SW-only
/// adapters a working one-bounce indirect — lower quality than the
/// 013 Mesh-Cards HW path but self-contained.
///
/// The clipmap is a single R32Float 3D texture covering a fixed world-
/// space AABB defined at capture (see `SCENE_SDF_CLIPMAP_EXTENT` /
/// `SCENE_SDF_CLIPMAP_ORIGIN` on the Rust side). Rays whose marched
/// position leaves the AABB treat the miss as "open sky".
pub(super) const SSGI_PROBE_TRACE_SDF_WGSL: &str = "
struct TraceParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    size: vec4<u32>,
    // x = frame_index, y = intensity, z = max_march_t, w = firefly_cap
    params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    // xyz = clipmap origin, w = extent (full width, not half)
    clipmap: vec4<f32>,
    // Ticket 014 V6 — WSRC camera-follow cube: xyz = origin, w = extent.
    // Used by the miss path to project hit_pos into the WSRC atlas.
    wsrc: vec4<f32>,
};

struct SdfInstanceGiData {
    albedo: vec3<f32>,
    emissive_luma: f32,
    normal_ws: vec3<f32>,
    _pad0: f32,
    card_slot: vec4<f32>,
    card_aabb_min: vec4<f32>,
    card_aabb_max: vec4<f32>,
};

const SDF_CARD_SLOTS_PER_ROW: f32 = 64.0;
const SDF_CARD_SLOT_PX: u32 = 64u;
const WSRC_GRID_RES: i32 = 16;

@group(0) @binding(0) var<uniform> u: TraceParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var clipmap_tex: texture_3d<f32>;
@group(0) @binding(3) var clipmap_samp: sampler;
@group(0) @binding(4) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> instance_data: array<SdfInstanceGiData>;
@group(0) @binding(6) var card_atlas: texture_2d<f32>;
@group(0) @binding(7) var card_samp: sampler;
@group(0) @binding(8) var wsrc_atlas: texture_3d<f32>;
@group(0) @binding(9) var wsrc_samp: sampler;

fn clipmap_uv(pos_ws: vec3<f32>) -> vec3<f32> {
    let half_extent = u.clipmap.w * 0.5;
    let origin = u.clipmap.xyz;
    return (pos_ws - origin + vec3<f32>(half_extent)) / u.clipmap.w;
}

fn clipmap_sample(pos_ws: vec3<f32>) -> f32 {
    let uv = clipmap_uv(pos_ws);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || uv.z < 0.0 || uv.z > 1.0) {
        // Outside the clipmap — assume wide open (no hit possible).
        return 1e4;
    }
    return textureSampleLevel(clipmap_tex, clipmap_samp, uv, 0.0).r;
}

// Ticket 014 V10 — WSRC lookup via the hardware linear-filtering
// sampler. Each probe's 10×10 padded octel slab lets the sampler do
// a native bilinear XY blend inside one probe without leaking into
// the neighbouring probe's data (that's what the 1-texel border is
// for). The Z axis is unpadded — probes are contiguous in Z, so
// sampling at `(gz + 0.5 + fz) / grid_res` gives native Z trilinear
// across adjacent probes for free. X/Y cross-probe blend is still
// done manually — 4 sample calls per miss ray, one per XY corner of
// the probe cube.
//
// Atlas packing:
//   atlas texel for probe (gx, gy, gz) at padded octel (ox_p, oy_p
//   in [0, 9]) is (gx*10 + ox_p, gy*10 + oy_p, gz).
//   Real octel (ox, oy in [0, 7]) sits at padded (ox+1, oy+1).
//   Border texels (padded 0 or 9) clone the nearest inside octel
//   — edge-extend, not true octahedral wrap (V11).
//
// Sampler uv formula (atlas x-axis):
//   texel_x for real-octel `ru_x ∈ [0, 1]` at probe `gx` is
//   `gx*10 + 1 + ru_x * 8`. Atlas uv = texel_x / 160.
//   → `atlas_uv_x = (gx + 0.1 + ru_x * 0.8) / 16`
//   `ru_x = 0` → texel 1 (centre of border, edge-extends octel 0)
//   `ru_x = 0.5` → texel 5 (centre of real area)
//   `ru_x = 1` → texel 9 (centre of opposite border)
fn wsrc_sample_probe(gx: i32, gy: i32, gz_f: f32, ru: vec2<f32>) -> vec3<f32> {
    let gxc = clamp(gx, 0, 15);
    let gyc = clamp(gy, 0, 15);
    let ax = (f32(gxc) + 0.1 + ru.x * 0.8) / 16.0;
    let ay = (f32(gyc) + 0.1 + ru.y * 0.8) / 16.0;
    let az = gz_f / 16.0;
    return textureSampleLevel(wsrc_atlas, wsrc_samp,
        vec3<f32>(ax, ay, az), 0.0).rgb;
}

fn wsrc_sample(pos_ws: vec3<f32>, dir_ws: vec3<f32>) -> vec3<f32> {
    let origin = u.wsrc.xyz;
    let extent = u.wsrc.w;
    if (extent <= 0.0) {
        return vec3<f32>(0.0);
    }
    let cell = extent / 16.0;
    let rel = pos_ws - origin + vec3<f32>(extent * 0.5);
    let pf = rel / cell - vec3<f32>(0.5);
    let pfx = floor(pf.x);
    let pfy = floor(pf.y);
    let gix = i32(pfx);
    let giy = i32(pfy);
    let fx = pf.x - pfx;
    let fy = pf.y - pfy;
    // V10 — Z blend is native via the sampler. Pass the full
    // floating-point z-slice position (`gz + 0.5 + fz`) clamped to
    // the atlas range so the sampler blends between the two adjacent
    // probe z-slices.
    let gz_f = clamp(pf.z + 0.5, 0.5, 15.5);

    let ru = oct_encode(dir_ws);

    let c00 = wsrc_sample_probe(gix,     giy,     gz_f, ru);
    let c10 = wsrc_sample_probe(gix + 1, giy,     gz_f, ru);
    let c01 = wsrc_sample_probe(gix,     giy + 1, gz_f, ru);
    let c11 = wsrc_sample_probe(gix + 1, giy + 1, gz_f, ru);

    let ix = 1.0 - fx;
    let iy = 1.0 - fy;
    return c00 * (ix * iy) + c10 * (fx * iy)
         + c01 * (ix * fy) + c11 * (fx * fy);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }
    if (lid.x >= PROBE_OCT_SIZE || lid.y >= PROBE_OCT_SIZE) { return; }

    let probe_idx = wg.y * grid_w + wg.x;
    let header = probes[probe_idx];
    let dst_coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));

    if (header.world_pos.w < 0.5) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    let dir_ws = octel_direction(lid.xy);
    let n_ws = header.normal.xyz;
    let ndotd = dot(dir_ws, n_ws);
    if (ndotd <= 0.0) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // 2 cm normal offset matches the SW Hi-Z + HW ray-query paths —
    // keeps primary hits from self-intersecting the probe surface.
    let origin_ws = header.world_pos.xyz + n_ws * 0.02;
    let max_t = u.params.z;

    // Sphere-trace. Step is the UDF value; convergence when within a
    // voxel's worth of the surface or when we exhaust the budget.
    let voxel_size = u.clipmap.w / 64.0;  // 64³ clipmap resolution
    let hit_threshold = voxel_size * 1.5;
    var t: f32 = 0.0;
    var hit: bool = false;
    for (var s: i32 = 0; s < 48; s = s + 1) {
        let pos = origin_ws + dir_ws * t;
        let d = clipmap_sample(pos);
        if (d < hit_threshold) {
            hit = true;
            break;
        }
        t = t + max(d, voxel_size * 0.5);
        if (t >= max_t) { break; }
    }

    var radiance = vec3<f32>(0.0);
    if (hit) {
        let hit_pos = origin_ws + dir_ws * t;

        // UDF gradient → outward surface normal (flip since gradient
        // points AWAY from the surface in an unsigned field).
        let h = voxel_size;
        let dx = clipmap_sample(hit_pos + vec3<f32>(h, 0.0, 0.0))
               - clipmap_sample(hit_pos - vec3<f32>(h, 0.0, 0.0));
        let dy = clipmap_sample(hit_pos + vec3<f32>(0.0, h, 0.0))
               - clipmap_sample(hit_pos - vec3<f32>(0.0, h, 0.0));
        let dz = clipmap_sample(hit_pos + vec3<f32>(0.0, 0.0, h))
               - clipmap_sample(hit_pos - vec3<f32>(0.0, 0.0, h));
        var grad = vec3<f32>(dx, dy, dz);
        let glen = length(grad);
        if (glen > 1e-4) { grad = grad / glen; }
        let hit_n = -grad;

        // Ticket 014 V4 — broad-phase lookup: walk `instance_data`,
        // find the first AABB (slightly dilated) containing hit_pos.
        // Pick the axis most aligned with the outward normal; project
        // hit onto its card; sample the pre-lit radiance atlas. Falls
        // back to analytic sun/sky × gray when no instance matches
        // (clipmap sentinel voxels, hits inside unaccounted-for
        // geometry, etc.).
        let count = arrayLength(&instance_data);
        var picked: i32 = -1;
        for (var i: u32 = 0u; i < count; i = i + 1u) {
            let ad = instance_data[i];
            if (ad.card_slot.w < 0.5) { continue; }
            let bmin = ad.card_aabb_min.xyz - vec3<f32>(0.05);
            let bmax = ad.card_aabb_max.xyz + vec3<f32>(0.05);
            if (hit_pos.x >= bmin.x && hit_pos.x <= bmax.x &&
                hit_pos.y >= bmin.y && hit_pos.y <= bmax.y &&
                hit_pos.z >= bmin.z && hit_pos.z <= bmax.z) {
                picked = i32(i);
                break;
            }
        }

        if (picked >= 0) {
            let ad = instance_data[u32(picked)];
            // Pick signed axis from outward normal. Dominant component
            // picks the axis; sign picks + or - face.
            let abs_n = abs(hit_n);
            var axis_idx: u32 = 0u;
            if (abs_n.y >= abs_n.x && abs_n.y >= abs_n.z) {
                axis_idx = 2u;
            } else if (abs_n.z >= abs_n.x) {
                axis_idx = 4u;
            }
            var signed_axis: u32 = axis_idx;
            if (axis_idx == 0u && hit_n.x < 0.0) { signed_axis = 1u; }
            else if (axis_idx == 2u && hit_n.y < 0.0) { signed_axis = 3u; }
            else if (axis_idx == 4u && hit_n.z < 0.0) { signed_axis = 5u; }

            let first_slot = u32(ad.card_slot.x);
            let slot = first_slot + signed_axis;
            let slot_x = slot % 64u;
            let slot_y = slot / 64u;

            let bmin = ad.card_aabb_min.xyz;
            let bmax = ad.card_aabb_max.xyz;
            // Hit is in world space and Sponza meshes are in world
            // space too (no per-instance transform beyond identity on
            // the Sponza asset). For now use world-space hit directly;
            // instances with a non-identity transform would need the
            // transform stored on `instance_data` to round-trip.
            var u_os: f32;
            var v_os: f32;
            var u_lo: f32; var u_hi: f32;
            var v_lo: f32; var v_hi: f32;
            var u_flip: f32 = 1.0;
            if (signed_axis == 0u || signed_axis == 1u) {
                u_os = hit_pos.y; v_os = hit_pos.z;
                u_lo = bmin.y; u_hi = bmax.y; v_lo = bmin.z; v_hi = bmax.z;
                if (signed_axis == 1u) { u_flip = -1.0; }
            } else if (signed_axis == 2u || signed_axis == 3u) {
                u_os = hit_pos.x; v_os = hit_pos.z;
                u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.z; v_hi = bmax.z;
                if (signed_axis == 3u) { u_flip = -1.0; }
            } else {
                u_os = hit_pos.x; v_os = hit_pos.y;
                u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.y; v_hi = bmax.y;
                if (signed_axis == 5u) { u_flip = -1.0; }
            }
            var u_norm = clamp((u_os - u_lo) / max(u_hi - u_lo, 1e-4), 0.0, 1.0);
            let v_norm = clamp((v_os - v_lo) / max(v_hi - v_lo, 1e-4), 0.0, 1.0);
            if (u_flip < 0.0) { u_norm = 1.0 - u_norm; }
            let slot_size_uv = 1.0 / SDF_CARD_SLOTS_PER_ROW;
            let texel_in_slot = slot_size_uv / f32(SDF_CARD_SLOT_PX);
            let slot_u0 = f32(slot_x) * slot_size_uv + texel_in_slot;
            let slot_v0 = f32(slot_y) * slot_size_uv + texel_in_slot;
            let slot_span = slot_size_uv - 2.0 * texel_in_slot;
            let atlas_uv = vec2<f32>(
                slot_u0 + u_norm * slot_span,
                slot_v0 + v_norm * slot_span,
            );
            let pre_lit = textureSampleLevel(card_atlas, card_samp, atlas_uv, 0.0).rgb;

            let tn = t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = pre_lit * falloff;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        } else {
            // Fallback — analytic sun/sky × gray albedo when no
            // instance matches. Same shading as V3.
            let ndotl = max(dot(hit_n, u.sun_dir.xyz), 0.0);
            let direct = u.sun_color.xyz * ndotl;
            let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
            let sky = u.sky_color.xyz * ndotup;
            let albedo = vec3<f32>(0.55, 0.55, 0.55);
            let tn = t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = albedo * (direct + sky) * falloff;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        }
    } else {
        // Ticket 014 V6 — miss path samples the WSRC envelope instead
        // of returning black. Ray terminal position (origin + dir * t)
        // is where we project into the cache; direction picks the
        // probe's octel. Firefly-clamp to match the hit path.
        let terminal = origin_ws + dir_ws * t;
        var raw = wsrc_sample(terminal, dir_ws);
        let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
        let cap = u.params.w;
        if (luma > cap) { raw = raw * (cap / luma); }
        radiance = raw;
    }

    let intensity = u.params.y;
    textureStore(radiance_out, dst_coord, vec4<f32>(radiance * intensity * ndotd, 1.0));
}
";

/// Probe temporal accumulator. EMA in probe-octel space. No reprojection
/// in V1 — since every frame traces all 64 octels, history only smooths
/// firefly noise; per-probe world positions jitter by tile-fraction so
/// camera motion eventually converges to a stable signal without explicit
/// velocity. Disocclusion is handled implicitly: moving the camera
/// changes which tile a surface falls into, replacing that probe's
/// header, and the new probe's history blends from zero since the old
/// probe at that grid coord pointed elsewhere.
pub(super) const SSGI_PROBE_TEMPORAL_WGSL: &str = "
struct TemporalParams {
    // x = alpha (0.25 = 4-frame EMA at steady state),
    // y = force_refresh (1 → alpha 1.0),
    // z = grid_w, w = grid_h
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TemporalParams;
@group(0) @binding(1) var radiance_in: texture_3d<f32>;
@group(0) @binding(2) var history_in: texture_3d<f32>;
@group(0) @binding(3) var history_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u32(u.params.z);
    let grid_h = u32(u.params.w);
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }

    let coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));
    let curr = textureLoad(radiance_in, coord, 0).rgb;
    let hist = textureLoad(history_in, coord, 0).rgb;

    var alpha = u.params.x;
    if (u.params.y > 0.5) { alpha = 1.0; }
    let blended = mix(hist, curr, alpha);

    textureStore(history_out, coord, vec4<f32>(blended, 1.0));
}
";

/// Per-pixel probe-cache reconstruction. Writes the half-res ssgi_rt
/// that the downstream compose / TAA passes already read.
///
/// Samples the 2×2 probes whose tiles enclose the pixel's tile. For
/// each probe, evaluates the octahedral atlas along the pixel's
/// world-space normal, then bilateral-weights the contribution by
/// depth-match + normal-match with the pixel itself. Invalid probes
/// (sky) are skipped. When all 4 probes reject (pixel depth/normal
/// wildly off), fall back to a zero contribution — better than leaking
/// a stale distant probe's radiance into a foreground surface.
pub(super) const SSGI_PROBE_RESOLVE_WGSL: &str = "
struct ResolveParams {
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = tile_size (16.0), y = intensity, zw unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ResolveParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var radiance_tex: texture_3d<f32>;
@group(0) @binding(3) var radiance_samp: sampler;
@group(0) @binding(4) var hiz0: texture_2d<f32>;
@group(0) @binding(5) var hiz_samp: sampler;

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

// Sample a probe's octahedral atlas in a given world-space direction.
// Uses trilinear sampling on the 3D texture so neighbouring octels
// softly blend — some visible smear on the 8×8 atlas, at the cost of
// a cheap reconstruction.
fn sample_probe(probe_xy: vec2<i32>, dir_ws: vec3<f32>) -> vec3<f32> {
    let oct_uv = oct_encode(dir_ws);
    let u_tex = (f32(probe_xy.x) + 0.5) / f32(u.size.z);
    let v_tex = (f32(probe_xy.y) + 0.5) / f32(u.size.w);
    // z coordinate: octahedral texel index in [0, 64) normalized to [0, 1)
    let oct_x = clamp(oct_uv.x * f32(PROBE_OCT_SIZE), 0.0, f32(PROBE_OCT_SIZE) - 1.0);
    let oct_y = clamp(oct_uv.y * f32(PROBE_OCT_SIZE), 0.0, f32(PROBE_OCT_SIZE) - 1.0);
    let z_idx = floor(oct_y) * f32(PROBE_OCT_SIZE) + floor(oct_x);
    let z = (z_idx + 0.5) / f32(PROBE_OCT_TEXELS);
    return textureSampleLevel(radiance_tex, radiance_samp, vec3<f32>(u_tex, v_tex, z), 0.0).rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let linear_z = textureSampleLevel(hiz0, hiz_samp, in.uv, 0.0).r;
    if (linear_z >= HIZ_SKY_Z * 0.5) {
        return vec4<f32>(0.0);
    }

    let half_w = f32(u.size.x);
    let half_h = f32(u.size.y);
    let tile = u.params.x;
    let grid_w = i32(u.size.z);
    let grid_h = i32(u.size.w);

    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;
    let P_vs = view_pos_from_linear(in.uv, linear_z, p00, p11, p20, p21);

    // Reconstruct pixel normal (same 3-tap trick as the placement pass).
    let texel = vec2<f32>(1.0 / half_w, 1.0 / half_h);
    let zr = textureSampleLevel(hiz0, hiz_samp, in.uv + vec2<f32>(texel.x, 0.0), 0.0).r;
    let zu = textureSampleLevel(hiz0, hiz_samp, in.uv + vec2<f32>(0.0, -texel.y), 0.0).r;
    let Pr = view_pos_from_linear(in.uv + vec2<f32>(texel.x, 0.0), zr, p00, p11, p20, p21);
    let Pu = view_pos_from_linear(in.uv + vec2<f32>(0.0, -texel.y), zu, p00, p11, p20, p21);
    let N_vs = normalize(cross(Pr - P_vs, Pu - P_vs));
    let N_ws = normalize((u.inv_view * vec4<f32>(N_vs, 0.0)).xyz);

    // Pixel's grid-space fractional position (which probes surround it?).
    let px_x = in.uv.x * half_w;
    let px_y = in.uv.y * half_h;
    let fx = px_x / tile - 0.5;  // -0.5 aligns grid cells centred on tile centres
    let fy = px_y / tile - 0.5;
    let gx0 = i32(floor(fx));
    let gy0 = i32(floor(fy));
    let tx = fract(fx);
    let ty = fract(fy);

    var accum = vec3<f32>(0.0);
    var wsum = 0.0;

    for (var dy = 0; dy <= 1; dy = dy + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let gx = clamp(gx0 + dx, 0, grid_w - 1);
            let gy = clamp(gy0 + dy, 0, grid_h - 1);
            let probe = probes[u32(gy * grid_w + gx)];
            if (probe.world_pos.w < 0.5) { continue; }

            // Bilinear corner weight
            var w_corner = 1.0;
            w_corner = w_corner * select(1.0 - tx, tx, dx == 1);
            w_corner = w_corner * select(1.0 - ty, ty, dy == 1);

            // Depth + normal bilateral weights — reject probes on very
            // different surfaces from the pixel (foreground pixel vs
            // probe on a far wall, or on an orthogonal facet).
            let dz = abs(probe.normal.w - linear_z);
            let w_depth = exp(-dz * dz * 8.0);
            let ndotn = clamp(dot(probe.normal.xyz, N_ws), 0.0, 1.0);
            let w_normal = pow(ndotn, 4.0);

            let w = w_corner * w_depth * w_normal;
            if (w <= 0.0001) { continue; }

            let radiance = sample_probe(vec2<i32>(gx, gy), N_ws);
            accum = accum + radiance * w;
            wsum = wsum + w;
        }
    }

    if (wsum > 0.0001) {
        accum = (accum / wsum) * u.params.y;
    }
    return vec4<f32>(accum, 1.0);
}
";

/// SSR temporal denoiser. Same shape as the SSGI temporal pass:
/// reprojects the previous history through the motion vectors,
/// clamps against the 3×3 neighborhood of the noisy current frame,
/// and blends with a low alpha so 4–8 frames of random GGX rays
/// converge to a smooth reflection. Also pre-filters the noisy
/// current frame by the 3×3 mean, which kills single-pixel
/// glossy-ray sparkles in one frame instead of 10.
pub(super) const SSR_TEMPORAL_SHADER_WGSL: &str = "
struct SsrTemporalParams {
    /// x = blend_alpha (0.1), yzw unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsrTemporalParams;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var current_samp: sampler;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var history_samp: sampler;
@group(0) @binding(5) var velocity_tex: texture_2d<f32>;
@group(0) @binding(6) var velocity_samp: sampler;

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
    let current_raw = textureSample(current_tex, current_samp, in.uv);

    // 3×3 box pre-filter + neighborhood min/max. One texel spread
    // across 9 samples hides single-pixel glossy-ray sparkles in a
    // single frame; the min/max bounds the history so disocclusion
    // and material transitions clamp rather than ghost.
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(current_tex));
    var nmin = current_raw;
    var nmax = current_raw;
    var prefilt = vec4<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let s = textureSample(current_tex, current_samp, in.uv + vec2<f32>(f32(x), f32(y)) * texel);
            nmin = min(nmin, s);
            nmax = max(nmax, s);
            prefilt = prefilt + s;
        }
    }
    let current = prefilt * (1.0 / 9.0);

    // Velocity is full-res; UV mapping handles the half-res delta.
    // NDC-space velocity + UV Y-flip → `uv + vel.y` for the Y axis,
    // matching TAA + SSAO + the sibling SSGI temporal pass.
    let vel = textureSample(velocity_tex, velocity_samp, in.uv).xy;
    let prev_uv = vec2<f32>(in.uv.x - vel.x, in.uv.y + vel.y);
    let off_screen = prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0;
    if (off_screen) { return current; }

    let history_raw = textureSample(history_tex, history_samp, prev_uv);
    // Scrub NaN/Inf from the history read. Until a clean SSR frame
    // finishes draining the ping-pong pair, any poisoned history
    // pixel would otherwise survive the clamp (clamp(NaN, a, b) is
    // implementation-defined on Metal — frequently NaN) and keep
    // tonemapping to pink. Replace poisoned channels with the
    // current-frame mean, which is the best available estimate.
    let history = select(current, history_raw, history_raw == history_raw);
    let clamped_history = clamp(history, nmin, nmax);
    let alpha = u.params.x;
    let blended = mix(clamped_history, current, alpha);
    return select(current, blended, blended == blended);
}
";

/// SSR (screen-space reflections) shader. View-space ray march:
///
/// 1. Reconstruct view-space position from the depth buffer.
/// 2. Reconstruct view-space normal from depth derivatives
///    (cross of dpdx/dpdy of view position).
/// 3. Reflect view direction around N → reflection direction R.
/// 4. March along R in view space, project each step to screen
///    coords, sample depth there, hit if our marched z is past the
///    sampled surface.
/// 5. On hit, sample the HDR RT at the hit UV and output it
///    (faded toward edges of screen so off-screen reflections
///    don't pop into existence).
///
/// Output is half-res HDR. The TAA pass adds it on top of the
/// prefiltered IBL specular for the final image.
pub(super) const SSR_SHADER_WGSL: &str = "
struct SsrParams {
    /// Inverse of the projection matrix — depth → view-space pos.
    inv_proj: mat4x4<f32>,
    /// Projection matrix — view-space pos → clip-space.
    proj: mat4x4<f32>,
    /// x = SSR strength (0 = off, 1 = full)
    /// y = max march distance in view-space units
    /// z = number of march steps
    /// w = frame index (Hammersley rotation + march jitter)
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsrParams;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var hdr_tex: texture_2d<f32>;
@group(0) @binding(4) var hdr_samp: sampler;
@group(0) @binding(5) var mat_tex: texture_2d<f32>;
@group(0) @binding(6) var mat_samp: sampler;
@group(0) @binding(7) var albedo_tex: texture_2d<f32>;
@group(0) @binding(8) var albedo_samp: sampler;

const PI: f32 = 3.14159265;

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

fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_h = u.inv_proj * ndc;
    return view_h.xyz / view_h.w;
}

/// Interleaved gradient noise — per-pixel pseudo-random in [0, 1).
/// Varies with frame so the temporal accumulator averages over
/// different march offsets each frame.
fn ign_jitter(frag_coord: vec2<f32>, frame: f32) -> f32 {
    let shifted = frag_coord + vec2<f32>(frame * 5.588238, frame * 3.127137);
    return fract(52.9829189 * fract(0.06711056 * shifted.x + 0.00583715 * shifted.y));
}

/// Cheap 2D hash → two independent low-discrepancy values in [0,1)².
/// Used as GGX microfacet-sample coordinates; the frame index rotates
/// the hash so each pixel draws a different sample every frame and
/// the temporal denoiser averages over the GGX lobe.
fn hash2(frag_coord: vec2<f32>, frame: f32) -> vec2<f32> {
    let p1 = frag_coord + vec2<f32>(frame * 11.13, frame * 7.77);
    let p2 = frag_coord + vec2<f32>(frame * 3.17,  frame * 5.29);
    let a = fract(sin(dot(p1, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let b = fract(sin(dot(p2, vec2<f32>(37.7191, 17.1123))) * 28471.1713);
    return vec2<f32>(a, b);
}

/// GGX importance-sampled microfacet half-vector in tangent space
/// aligned to the surface normal. Isotropic GGX — α = roughness².
fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h_local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * h_local.x + b * h_local.y + n * h_local.z);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, in.uv);
    if (depth >= 0.9999) { return vec4<f32>(0.0); } // sky

    // Metals-only SSR (see pre-stochastic history for the rationale).
    // The metallic and low-roughness gates are kept: dielectric
    // specular stays with the prefiltered IBL chain, and very rough
    // metals fade out to the IBL cubemap where SSR noise would
    // dominate even after temporal accumulation.
    let mat = textureSample(mat_tex, mat_samp, in.uv).rg;
    let metallic = mat.r;
    let roughness = mat.g;
    if (metallic < 0.2) { return vec4<f32>(0.0); }
    let albedo = textureSample(albedo_tex, albedo_samp, in.uv).rgb;
    let roughness_fade = 1.0 - smoothstep(0.5, 0.85, roughness);
    if (roughness_fade <= 0.001) { return vec4<f32>(0.0); }

    let view_pos = view_pos_from_depth(in.uv, depth);
    let dx = dpdx(view_pos);
    let dy = dpdy(view_pos);
    let n = normalize(cross(dx, dy));
    let v = normalize(-view_pos);

    // Stochastic SSR — cast one GGX-importance-sampled ray per pixel
    // per frame. Different frames draw from different points on the
    // GGX lobe (rotated by frame index) so the downstream temporal
    // denoiser averages a dense roughness cone over 4–8 frames. This
    // replaces the 5-tap Gaussian blur at the hit: we pay one ray +
    // one hdr sample per frame, not 32 march steps + 5 blur taps.
    //
    // xi is clamped away from exact 0 and 1. At roughness → 0 the GGX
    // denominator `1 + (α²-1)·xi.y` collapses to 0 when xi.y = 1, so
    // cos_theta becomes sqrt(0/0) = NaN. That NaN then propagates into
    // ssr_history and each 4× upsampled texel turns into a pink hot
    // pixel after tonemapping. Sponza's mirror-smooth lamp fittings
    // (roughness near 0) are exactly the worst case.
    let xi = clamp(hash2(in.clip_pos.xy, u.params.w), vec2<f32>(1e-4), vec2<f32>(0.9999));
    let h = importance_sample_ggx(xi, n, roughness);
    let r = reflect(-v, h);

    if (r.z > 0.0) { return vec4<f32>(0.0); }

    let n_dot_v = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);

    let max_dist = u.params.y;
    let n_steps_f = u.params.z;
    let n_steps = u32(n_steps_f);
    let step_size = max_dist / n_steps_f;

    let jitter = ign_jitter(in.clip_pos.xy, u.params.w);
    var t = step_size * (0.5 + jitter);

    var hit_uv = vec2<f32>(-1.0);
    var hit_found = false;
    var prev_t = 0.0;
    for (var i = 0u; i < n_steps; i = i + 1u) {
        let ray_view = view_pos + r * t;
        let ray_clip = u.proj * vec4<f32>(ray_view, 1.0);
        let ray_ndc = ray_clip.xyz / ray_clip.w;
        if (ray_ndc.x < -1.0 || ray_ndc.x > 1.0 ||
            ray_ndc.y < -1.0 || ray_ndc.y > 1.0 ||
            ray_ndc.z < 0.0 || ray_ndc.z > 1.0) {
            break;
        }
        let ray_uv = vec2<f32>(ray_ndc.x * 0.5 + 0.5, 1.0 - (ray_ndc.y * 0.5 + 0.5));
        let scene_depth = textureSample(depth_tex, depth_samp, ray_uv);

        if (ray_ndc.z >= scene_depth) {
            let hit_view = view_pos_from_depth(ray_uv, scene_depth);
            let thickness = abs(ray_view.z - hit_view.z);
            let step_world = t - prev_t;
            if (thickness < step_world * 2.0 + 0.1) {
                hit_uv = ray_uv;
                hit_found = true;
            }
            break;
        }
        prev_t = t;
        t = t + step_size;
    }
    if (!hit_found) { return vec4<f32>(0.0); }

    let edge_fade = min(
        min(hit_uv.x, 1.0 - hit_uv.x),
        min(hit_uv.y, 1.0 - hit_uv.y),
    ) * 10.0;
    let fade = clamp(edge_fade, 0.0, 1.0);

    // NaN scrubber: WGSL has no isnan(), but NaN == NaN is false for
    // every compliant backend, so a componentwise self-compare gives us
    // a vec3<bool> that is true iff each channel is finite. This nukes
    // the one stray NaN/Inf pixel per few thousand rays that would
    // otherwise ping-pong through ssr_history and tonemap to pink. Same
    // self-compare is applied to the HDR tap in case upstream writes a
    // bad sample (autoexposure ratios, rare shader ops on degenerate
    // triangles, etc).
    let raw = textureSample(hdr_tex, hdr_samp, hit_uv).rgb;
    let reflected = select(vec3<f32>(0.0), raw, raw == raw);
    let out = reflected * fresnel * roughness_fade * u.params.x * fade;
    let out_safe = select(vec3<f32>(0.0), out, out == out);
    return vec4<f32>(out_safe, fade);
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
pub(super) const SCENE_COMPOSE_SHADER_WGSL: &str = "
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

    // Volumetric fog: 16-step Beer-Lambert march with height-based
    // density falloff. Density 0 = disabled (the mul collapses to
    // unity; an early-out avoids the loop entirely).
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
pub(super) const TAA_SHADER_WGSL: &str = "
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
pub(super) const EXPOSURE_SHADER_WGSL: &str = "
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
pub(super) const COMPOSITE_SHADER_WGSL: &str = "
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

