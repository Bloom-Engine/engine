//! Core pipeline shaders: batched 2D, legacy 3D, and the main scene shader (forward MRT).
//! Split from renderer/shaders.rs.

//! WGSL shader strings used by the renderer.
//!
//! Pure data — no behavior, no struct definitions. Each `const`
//! is `pub(super)` so the surrounding `renderer` module (and only
//! that module) can see it, via `use super::shaders::*;` in
//! `mod.rs`. Split out so the ~11 500-line renderer file shrinks
//! to the Rust logic it actually contains.

pub(in crate::renderer) const SHADER_2D: &str = "
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

pub(in crate::renderer) const SHADER_3D: &str = "
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
    // x = joint-buffer offset, y = skinned flag (cached skinned draws).
    // Always zero on the immediate path — its verts arrive with joint
    // indices pre-offset CPU-side, so vs_main_3d ignores this field.
    misc: vec4<f32>,
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
    dir_lights: array<DirLight, 8>,
    point_light_count: vec4<f32>,
    point_lights: array<PointLight, 256>,
};

struct JointMatrices {
    matrices: array<mat4x4<f32>, 1024>,
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
    //
    // Alpha comes from the TINT only. Game textures routinely carry a
    // non-opacity alpha channel (Unvanquished armor packs a gloss mask
    // there), and this batch also renders CPU-skinned characters — the
    // player turned semi-transparent through its gloss mask when texture
    // alpha fed the blend. Deliberate fades still work via tint alpha;
    // untextured effect quads bind the white texture (alpha 1) anyway.
    return Fs3DOut(
        vec4<f32>(tex_color.rgb * in.color.rgb * lit, in.color.a),
        vec2<f32>(0.0, 1.0),
        vel,
        vec4<f32>(0.0),
    );
}
";

// The cloud deck (common/clouds.wgsl) is prepended verbatim: this shader is a
// raw source const and does not run through the material preprocessor. Same
// file the sky pass and the world materials use, so a cloud shadow crossing
// the terrain also crosses the trees standing in it — which is the whole
// reason to share it.
pub(in crate::renderer) const SCENE_SHADER: &str = concat!(
    include_str!("../../../shaders/common/clouds.wgsl"),
    include_str!("../../../shaders/common/foliage_wind.wgsl"),
    r#"
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
    // x = joint-buffer offset for this draw, y = 1.0 for skinned cached
    // draws (vs_main_scene then skins in the VS), zw unused.
    misc: vec4<f32>,
};

struct JointMatrices {
    matrices: array<mat4x4<f32>, 1024>,
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
    dir_lights: array<DirLight, 8>,
    point_light_count: vec4<f32>,
    point_lights: array<PointLight, 256>,
    camera_pos: vec4<f32>,
    shadow_cascade_vps: array<mat4x4<f32>, 3>,
    shadow_cascade_splits: vec4<f32>,
    shadow_view_matrix: mat4x4<f32>,
    wind: vec4<f32>,   // xy=dir, z=amplitude, w=time (foliage sway)
    cloud: vec4<f32>,  // x=shadow strength, y=deck height, z=scale, w=drift m/s
    frame_misc: vec4<f32>, // x=delta_time (prev-frame wind, for motion vectors)
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
    // EN-044 — @invariant is load-bearing. The depth prepass and the main pass run
    // the SAME vertex entry point, but through different pipelines: the prepass's
    // fragment stage consumes almost none of the varyings, so the compiler is free
    // to optimise the position maths differently (fma contraction, reassociation)
    // and the two depths stop being bit-identical. The main pass then tests Equal
    // against a depth that is one ulp off, every fragment fails, and the entire
    // forest and the player VANISH — which is exactly what happened, and it looked
    // like a 60 fps win. @invariant forbids that: the position must be computed
    // identically in every pipeline that uses this shader.
    @invariant @builtin(position) clip_position: vec4<f32>,
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
@group(3) @binding(0) var<uniform> joints: JointMatrices;
// PT-7 — previous frame's palette, same slot offsets: skinned verts
// reconstruct last frame's world position from it, giving skeletal
// motion a REAL velocity (it was exactly zero before).
@group(3) @binding(1) var<uniform> joints_prev: JointMatrices;

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
    if (u.misc.y > 0.5) {
        // Skinned draw: u.mvp/u.prev_mvp are the bare view-projection;
        // joint matrices bake world placement for weighted verts, and
        // u.model places the rare rigid (weightless) verts. No wind
        // sway here — characters aren't foliage.
        let total_weight = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
        var world4: vec4<f32>;
        var prev_world4: vec4<f32>;
        var nrm4: vec4<f32>;
        var tan4: vec4<f32>;
        let pos4l = vec4<f32>(in.position, 1.0);
        let nrm4l = vec4<f32>(in.normal, 0.0);
        let tan4l = vec4<f32>(in.tangent.xyz, 0.0);
        if (total_weight > 0.01) {
            // The cached VB keeps RAW joint indices; misc.x is this
            // draw's base slot in the shared 1024-entry joint buffer.
            let j0 = u32(in.joints.x + u.misc.x); let j1 = u32(in.joints.y + u.misc.x);
            let j2 = u32(in.joints.z + u.misc.x); let j3 = u32(in.joints.w + u.misc.x);
            world4 = joints.matrices[j0] * pos4l * in.weights.x
                   + joints.matrices[j1] * pos4l * in.weights.y
                   + joints.matrices[j2] * pos4l * in.weights.z
                   + joints.matrices[j3] * pos4l * in.weights.w;
            // PT-7 — where this vertex WAS: previous palette, same
            // slots. Feeds the velocity MRT so TAA/TSR and the path
            // tracer can reproject skeletal motion.
            prev_world4 = joints_prev.matrices[j0] * pos4l * in.weights.x
                        + joints_prev.matrices[j1] * pos4l * in.weights.y
                        + joints_prev.matrices[j2] * pos4l * in.weights.z
                        + joints_prev.matrices[j3] * pos4l * in.weights.w;
            nrm4 = joints.matrices[j0] * nrm4l * in.weights.x
                 + joints.matrices[j1] * nrm4l * in.weights.y
                 + joints.matrices[j2] * nrm4l * in.weights.z
                 + joints.matrices[j3] * nrm4l * in.weights.w;
            tan4 = joints.matrices[j0] * tan4l * in.weights.x
                 + joints.matrices[j1] * tan4l * in.weights.y
                 + joints.matrices[j2] * tan4l * in.weights.z
                 + joints.matrices[j3] * tan4l * in.weights.w;
        } else {
            world4 = u.model * pos4l;
            prev_world4 = world4;
            nrm4 = u.model * nrm4l;
            tan4 = u.model * tan4l;
        }
        var o: VertexOutputScene;
        let c = u.mvp * world4;
        o.clip_position = c;
        o.curr_clip = c;
        o.prev_clip = u.prev_mvp * prev_world4;
        o.world_pos = world4.xyz;
        o.normal = normalize(nrm4.xyz);
        o.color = in.color * u.model_tint;
        o.uv = in.uv;
        o.tangent = vec4<f32>(normalize(tan4.xyz), in.tangent.w);
        return o;
    }
    var out: VertexOutputScene;
    var local = in.position;
    // Hierarchical foliage wind (common/foliage_wind.wgsl). u.misc.z is the
    // per-draw foliage amount — 0 for everything that is not a plant, so the
    // world does not sway. This replaces a sway that only ever moved ALPHA-CUT
    // materials, which meant leaf cards fluttered and every trunk was rigid.
    //
    // is_leaf comes from the alpha cutoff, so cards get the fast flutter layer
    // and wood does not.
    var prev_local = local;
    if (u.misc.z > 0.0 && lighting.wind.z > 0.0) {
        // is_leaf from the alpha cutoff: cards get the fast flutter layer, wood
        // does not. Same helper the shadow pass calls, so the tree and its shadow
        // bend together.
        let is_leaf = select(0.0, 1.0, material.metal_rough.w > 0.0);
        local = foliage_wind_local(in.position, u.model, lighting.wind, u.misc.z, is_leaf);
        // Last frame's offset too, so TAA gets a real velocity for a moving leaf
        // instead of 0 and stops smearing the canopy into the sky behind it.
        var w_prev = lighting.wind;
        w_prev.w = lighting.wind.w - lighting.frame_misc.x;
        prev_local = foliage_wind_local(in.position, u.model, w_prev, u.misc.z, is_leaf);
    }
    let pos4 = vec4<f32>(local, 1.0);
    let curr = u.mvp * pos4;
    out.clip_position = curr;
    out.curr_clip = curr;
    out.prev_clip = u.prev_mvp * vec4<f32>(prev_local, 1.0);
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
fn sample_shadow(world_pos: vec3<f32>, geo_n: vec3<f32>) -> f32 {
    // shadows disabled → fully lit. dir_light_count.y carries the enabled
    // flag (splits.w is the TSR mip-LOD bias — do NOT gate on it); without
    // this gate the projection below runs through identity/stale cascade
    // VPs and the garbage NDC reads as 'occluded', so turning shadows OFF
    // used to DARKEN ambient instead of removing shadows.
    if (lighting.dir_light_count.y < 0.5) {
        return 1.0;
    }
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

    // Normal-offset receiver bias: push the receiver position off the
    // surface along its geometric normal by ~1.5 shadow texels of the
    // selected cascade before projecting. The fixed depth bias alone
    // (0.001 ≈ 8 cm across cascade 2's depth range) is SMALLER than the
    // per-texel depth slope of steep receivers — a vertical wall under a
    // 40°-elevation sun changes ~12 cm of light-space depth per shadow
    // texel — so entire sun-facing faces used to self-shadow into a
    // uniform ~50% PCF dimming (measured 68 vs 127 luma on the shooter's
    // stone house). Offsetting the receiver sidesteps the slope entirely;
    // the offset is texel-proportional (≈2 cm near, ≈23 cm at cascade 2),
    // far below visible peter-panning at each cascade's viewing distance.
    // The cascade fit radius ≈ its split distance (compute_cascade_vps
    // fits a camera-centred sphere), so texel ≈ 2·split / map_dim.
    let map_dim = f32(textureDimensions(shadow_tex_0).x);
    var fit_r = lighting.shadow_cascade_splits.z;
    if (cascade == 0) {
        fit_r = lighting.shadow_cascade_splits.x;
    } else if (cascade == 1) {
        fit_r = lighting.shadow_cascade_splits.y;
    }
    let recv_pos = world_pos + geo_n * (2.0 * fit_r / map_dim) * 1.5;

    // Project through the selected cascade's VP
    let light_clip = lighting.shadow_cascade_vps[cascade] * vec4<f32>(recv_pos, 1.0);
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
        // In the blend zone: sample the next cascade too and lerp.
        // Same normal-offset receiver bias, scaled to the NEXT cascade's
        // texel size (it is coarser, so the offset grows accordingly).
        let next_cascade = cascade + 1;
        var next_fit = lighting.shadow_cascade_splits.z;
        if (next_cascade == 1) {
            next_fit = lighting.shadow_cascade_splits.y;
        }
        let next_pos = world_pos + geo_n * (2.0 * next_fit / map_dim) * 1.5;
        let next_clip = lighting.shadow_cascade_vps[next_cascade] * vec4<f32>(next_pos, 1.0);
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

// EN-044 — depth prepass. Same vertex stage as the main pass (so the foliage wind
// displaces identically and the depths match), and a fragment stage that does
// nothing but honour the alpha cutout.
//
// WHY THIS EARNS ITS PASS. The scene fragment shader can `discard` (alpha-cutout
// foliage), and a shader that may discard cannot early-Z *write* — the GPU has to
// run the whole thing before it knows if the pixel survives. So every leaf card in
// an 88-tree forest shaded the full 5-target MRT, several layers deep, and threw
// most of it away. Priming depth first lets the main pass early-Z *reject* those
// fragments before the shader ever runs.
@fragment
fn fs_depth_prepass(in: VertexOutputScene) {
    let alpha_cutoff = material.metal_rough.w;
    if (alpha_cutoff > 0.0) {
        let a = textureSample(base_color_tex, base_color_samp, in.uv).a * in.color.a;
        if (a < alpha_cutoff) { discard; }
    }
}

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

    // Two-sided foliage normal. Alpha-cutout cards (leaves, grass blades)
    // are seen from both sides, but the geometric normal only faces one
    // way — the back side otherwise shades with N pointing away from the
    // sun AND from the sky irradiance, which is why grass tufts rendered
    // as solid black cards from one side. Flip the shading normal toward
    // the viewer for cutout materials only; opaque geometry is untouched.
    if (alpha_cutoff > 0.0 && dot(n, lighting.camera_pos.xyz - in.world_pos) < 0.0) {
        n = -n;
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
    // Geometric (pre-normal-map, pre-foliage-flip) normal for the receiver
    // offset — the mapped normal can point anywhere per-texel and would
    // dither the offset; the flipped foliage normal would push the sample
    // through the card.
    let shadow_factor = sample_shadow(in.world_pos, normalize(in.normal));
    // Never fully zero direct light — a 10% floor simulates
    // ambient bounce from surrounding surfaces and keeps shadows
    // from going pitch-black regardless of IBL intensity.
    let direct_shadow_raw = mix(0.03, 1.0, shadow_factor);
    let legacy_dir = normalize(lighting.light_dir.xyz);
    // Cloud deck (common/clouds.wgsl). Folded into the SUN shadow only: a cloud
    // blocks the sun, it does not stop the sky from being blue. Multiplying it
    // into ambient as well is what makes cloud shadows read as flat grey paint
    // instead of shade. Costs nothing when strength is 0 (the default).
    let direct_shadow = direct_shadow_raw * cloud_shadow_at(
        in.world_pos, legacy_dir, lighting.wind.xy, lighting.wind.w, lighting.cloud);
    if (alpha_cutoff > 0.0) {
        // Foliage wrap-lambert (energy-conserving wrap, w = 0.45): a leaf
        // turning from the sun rolls off softly — light transmits and
        // inter-scatters through a canopy — instead of clipping to black
        // at the terminator like an opaque wall. Specular is skipped:
        // foliage cards are rough and the viewer-flipped normal would
        // produce false sparkle.
        let wrap = 0.45;
        let ndl_wrap = clamp((dot(n, legacy_dir) + wrap) / ((1.0 + wrap) * (1.0 + wrap)),
                             0.0, 1.0);
        lit += base_color / PI * lighting.light_color.rgb * lighting.light_dir.w
             * ndl_wrap * direct_shadow;
    } else {
        lit += shade_pbr(n, v, legacy_dir, lighting.light_color.rgb,
                         lighting.light_dir.w, base_color, metallic, roughness)
             * direct_shadow;
    }

    // Foliage backlit transmission — sun bleeding THROUGH alpha-cut leaf cards
    // (the bright rim glow when the sun is behind a tree). Gated on the
    // alpha-cutoff so only cut-out foliage materials get it; opaque surfaces
    // (cutoff == 0) are unaffected. Matches shade_foliage's transmission term.
    // Round-2 audit: this block was pasted TWICE (1.7x strength) and ran
    // unshadowed — a canopy in another tree's shadow still glowed at full
    // transmission. De-duplicated and multiplied by the sun shadow factor.
    if (alpha_cutoff > 0.0) {
        let trans = pow(max(dot(v, -legacy_dir), 0.0), 3.0) * 0.85;
        lit += base_color * lighting.light_color.rgb * lighting.light_dir.w * trans
             * direct_shadow;
    }

    let dir_count = u32(lighting.dir_light_count.x);
    for (var i = 0u; i < dir_count; i++) {
        let dl = lighting.dir_lights[i];
        let l = normalize(dl.direction.xyz);
        lit += shade_pbr(n, v, l, dl.color.rgb, dl.direction.w,
                         base_color, metallic, roughness);
    }

    // BEGIN-POINT-LIGHT-LOOP (replaced by the froxel-clustered variant
    // at pipeline build on storage-buffer-capable backends — see
    // renderer/froxel.rs; this plain loop is the WebGL fallback and the
    // semantic reference the clustered path must match exactly)
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
    // END-POINT-LIGHT-LOOP

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
    // EN-021 exclusive ownership: where SSR is active it owns specular —
    // hit (traced colour) or miss (env fallback inside the SSR shader).
    // Scale IBL specular by the complement of SSR's own roughness fade
    // × its strength (dir_light_count.z, written per frame; 0 when SSR
    // is disabled so the full IBL term returns). Kills the metal
    // double-count on hits (round-2 audit F10) without darkening
    // off-screen reflections.
    let ssr_own = clamp(
        lighting.dir_light_count.z * (1.0 - smoothstep(0.5, 0.85, roughness)),
        0.0, 1.0);
    let ibl_spec = ibl_spec_raw
        * dielectric_scale * spec_occ * roughness_amp * cap2 * (1.0 - ssr_own);

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

    // glTF OPAQUE materials (alpha_cutoff == 0) ignore texture alpha by
    // spec — armor/gloss masks stored in .a must not make the mesh
    // translucent. MASK materials keep the sampled alpha (post-discard)
    // for soft edges. Tint alpha survives so games can fade models.
    let out_alpha = select(in.color.a, base_alpha, alpha_cutoff > 0.0);

    return SceneOut(
        vec4<f32>(hdr, out_alpha),
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
"#);

