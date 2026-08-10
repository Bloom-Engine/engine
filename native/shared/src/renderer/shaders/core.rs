//! Core pipeline shaders: legacy 3D and the main scene shader (forward MRT).
//! Split from renderer/shaders.rs.
//! Pure WGSL data, private to the surrounding renderer module.

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
    spec_gloss:  vec4<f32>, // rgb=specular factor, a=glossiness factor
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

fn safe_scene_tangent(tangent: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(tangent, tangent);
    return select(
        vec3<f32>(0.0),
        tangent * inverseSqrt(max(length_squared, 1e-20)),
        length_squared > 1e-8,
    );
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
        o.tangent = vec4<f32>(safe_scene_tangent(tan4.xyz), in.tangent.w);
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
    out.tangent = vec4<f32>(
        safe_scene_tangent((u.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        in.tangent.w,
    );
    return out;
}

// Screen-space-derivative TBN. Reconstructs a tangent frame purely
// from the fragment's world-space position and UV — no vertex tangent
// attribute required. Based on Mikkelsen 2010 ('Followup: Normal
// Mapping Without Precomputed Tangents'). Gives close-to-identical
// results to pre-baked tangents for continuous UV mappings, which is
// the common case for PBR assets. We use this as a fallback when the
// mesh has no TANGENT accessor (very common — e.g., DamagedHelmet).
// The four screen-space derivatives are taken by the CALLER in uniform
// control flow and passed in: this function is reached from the per-fragment
// "mesh has no tangents" branch, and WGSL's uniformity analysis (enforced by
// Tint on WebGPU) rejects dpdx/dpdy inside non-uniform flow.
fn compute_tbn(dp1: vec3<f32>, dp2: vec3<f32>, duv1: vec2<f32>, duv2: vec2<f32>, n: vec3<f32>) -> mat3x3<f32> {
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let denom = max(dot(t, t), dot(b, b));
    let invmax = inverseSqrt(max(denom, 1e-20));
    return mat3x3<f32>(t * invmax, b * invmax, n);
}

// Stable 4x4 Bayer threshold. Bloom's scene targets are single-sample, so
// hardware alpha-to-coverage is unavailable; this is its deterministic
// spatial equivalent for subpixel MASK coverage. It is used only after the
// sampler footprint reaches the coverage-encoded lower mip chain.
fn mask_coverage_threshold(pixel: vec2<f32>) -> f32 {
    let bayer = array<f32, 16>(
         0.5 / 16.0,  8.5 / 16.0,  2.5 / 16.0, 10.5 / 16.0,
        12.5 / 16.0,  4.5 / 16.0, 14.5 / 16.0,  6.5 / 16.0,
         3.5 / 16.0, 11.5 / 16.0,  1.5 / 16.0,  9.5 / 16.0,
        15.5 / 16.0,  7.5 / 16.0, 13.5 / 16.0,  5.5 / 16.0,
    );
    let x = u32(floor(pixel.x)) & 3u;
    let y = u32(floor(pixel.y)) & 3u;
    return bayer[y * 4u + x];
}

fn mask_texture_lod(
    uv: vec2<f32>,
    dimensions: vec2<u32>,
    lod_bias: f32,
) -> f32 {
    let extent = vec2<f32>(dimensions);
    let dx = dpdx(uv) * extent;
    let dy = dpdy(uv) * extent;
    let footprint2 = max(dot(dx, dx), dot(dy, dy));
    return max(0.5 * log2(max(footprint2, 1.0)) + lod_bias, 0.0);
}

fn mask_coverage_survives(
    authored_alpha: f32,
    lower_mip_coverage: f32,
    cutoff: f32,
    lod: f32,
    pixel: vec2<f32>,
) -> bool {
    let hard_coverage = select(0.0, 1.0, authored_alpha >= cutoff);
    // Transition before LOD 1 avoids a pop while trilinear sampling moves
    // from authored level-zero alpha to lower levels whose alpha is coverage.
    let blend = smoothstep(0.5, 1.0, lod);
    let probability = mix(hard_coverage, lower_mip_coverage, blend);
    return probability >= mask_coverage_threshold(pixel);
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

// Khronos reference conversion for the legacy
// KHR_materials_pbrSpecularGlossiness workflow. Keeping this per pixel is
// essential: reducing a textured specular/glossiness map to scalar factors
// erases the spatial material response authored by Bistro.
// xyz = converted linear base color, w = metallic.
fn specgloss_to_metalrough_pixel(
    diffuse: vec3<f32>,
    specular: vec3<f32>,
) -> vec4<f32> {
    let dielectric_specular = 0.04;
    let epsilon = 1e-6;
    let one_minus_dielectric = 1.0 - dielectric_specular;
    let diffuse_max = max(diffuse.r, max(diffuse.g, diffuse.b));
    let specular_max = max(specular.r, max(specular.g, specular.b));
    let a = dielectric_specular;
    let b = diffuse_max * one_minus_dielectric /
        max(dielectric_specular, epsilon) + specular_max -
        2.0 * dielectric_specular;
    let c = dielectric_specular - specular_max;
    let discriminant = max(b * b - 4.0 * a * c, 0.0);
    var metallic = 0.0;
    if (specular_max >= dielectric_specular) {
        metallic = clamp((-b + sqrt(discriminant)) / (2.0 * a), 0.0, 1.0);
    }
    let diffuse_scale = one_minus_dielectric /
        max(1.0 - metallic * dielectric_specular, epsilon);
    let base_color = mix(
        diffuse * diffuse_scale,
        specular,
        metallic * metallic,
    );
    return vec4<f32>(clamp(base_color, vec3<f32>(0.0), vec3<f32>(1.0)), metallic);
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
            sum += textureSampleCompareLevel(shadow_tex_0, shadow_samp, uv, depth_ref);
        } else if (cascade == 1) {
            sum += textureSampleCompareLevel(shadow_tex_1, shadow_samp, uv, depth_ref);
        } else {
            sum += textureSampleCompareLevel(shadow_tex_2, shadow_samp, uv, depth_ref);
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
    // Match the positive view depth used to fit the camera-frustum slices.
    // Spherical distance can select an undersized cascade for side receivers.
    let view_pos = lighting.shadow_view_matrix * vec4<f32>(world_pos, 1.0);
    let view_depth = max(-view_pos.z, 0.0);

    var cascade = 2;
    if (view_depth <= lighting.shadow_cascade_splits.x) {
        cascade = 0;
    } else if (view_depth <= lighting.shadow_cascade_splits.y) {
        cascade = 1;
    }

    // Push the receiver off its surface by ~1.5 shadow texels. The fixed
    // depth bias is smaller than the per-texel depth slope of steep
    // receivers, so sun-facing walls otherwise self-shadow uniformly
    // (measured 68 vs 127 luma on the shooter's stone house). This offset
    // sidesteps that slope;
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
    var recv_pos = world_pos + geo_n * (2.0 * fit_r / map_dim) * 1.5;

    // Project through the selected cascade's VP. A retained translation-slack
    // fit can put a receiver (especially after its normal offset) just outside
    // the selected cascade even though the next, wider cascade covers it. The
    // old path returned fully lit here, cutting view-dependent holes into an
    // otherwise continuous shadow as the camera moved or turned. Fall through
    // to the next valid cascade; the normal in-fit path still performs exactly
    // one depth sample.
    var light_clip = lighting.shadow_cascade_vps[cascade] * vec4<f32>(recv_pos, 1.0);
    var light_ndc = light_clip.xyz / light_clip.w;
    for (var handoff = 0; handoff < 2; handoff = handoff + 1) {
        let outside = light_ndc.x < -1.0 || light_ndc.x > 1.0 ||
            light_ndc.y < -1.0 || light_ndc.y > 1.0 ||
            light_ndc.z < 0.0 || light_ndc.z > 1.0;
        if (!outside) {
            break;
        }
        if (cascade >= 2) {
            return 1.0;
        }
        cascade = cascade + 1;
        fit_r = lighting.shadow_cascade_splits.z;
        if (cascade == 1) {
            fit_r = lighting.shadow_cascade_splits.y;
        }
        recv_pos = world_pos + geo_n * (2.0 * fit_r / map_dim) * 1.5;
        light_clip = lighting.shadow_cascade_vps[cascade] * vec4<f32>(recv_pos, 1.0);
        light_ndc = light_clip.xyz / light_clip.w;
    }
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
    let dist_to_edge = split_far - view_depth;

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
        // The next fitted slice may not cover the inner blend zone. Never
        // turn its clamped edge texel into a moving, falsely-lit shadow gap.
        if (any(abs(next_ndc.xy) > vec2<f32>(1.0)) || next_ndc.z < 0.0 || next_ndc.z > 1.0) {
            return shadow_val;
        }
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

    // Preserve the authored direct Fresnel response. The former dielectric
    // and universal roughness ramps both reached zero for smooth paint and
    // glass, deleting the highlight that distinguishes Bistro's scooter and
    // bottles from flat-colour props. Bound the punctual/directional-light
    // approximation continuously in radiance instead: this keeps the lobe
    // present while preventing an infinitesimal GGX peak from becoming a
    // camera-tracking firefly. Normal-variance filtering above supplies the
    // spatial integration for minified normal maps.
    let direct_luma = dot(specular_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let direct_cap = 1.0 / (1.0 + direct_luma / 0.3);
    let specular = specular_raw * direct_cap;

    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;

    return (diffuse + specular) * light_color * intensity * n_dot_l;
}

// Evaluate the legacy KHR_materials_pbrSpecularGlossiness workflow without
// squeezing its independent diffuse and specular colours through the
// metallic-roughness parameterization.  That conversion is necessarily
// lossy: a painted dielectric with an authored F0 above 0.04 is interpreted
// as partly metallic, tinting its reflection with the diffuse colour and
// suppressing the diffuse lobe.  Bistro's blue scooter is the canonical
// failure case.  Keeping the authored F0 here preserves both lobes while
// sharing the established GGX distribution, visibility, energy bound, and
// roughness policy with the ordinary material path.
fn shade_specular_glossiness_pbr(
    n: vec3<f32>,
    v: vec3<f32>,
    l_dir: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    diffuse_color: vec3<f32>,
    authored_f0: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, l_dir), 0.0);
    if (n_dot_l <= 0.0 || intensity <= 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), 1e-4);
    let h_raw = l_dir + v;
    let h_len2 = dot(h_raw, h_raw);
    if (h_len2 <= 1e-12) {
        let kd0 = vec3<f32>(1.0) - authored_f0;
        return kd0 * diffuse_color / PI * light_color * intensity * n_dot_l;
    }
    let h = h_raw * inverseSqrt(h_len2);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, h), 0.0, 1.0);
    let alpha = max(roughness * roughness, 0.001);
    let alpha2 = alpha * alpha;
    let f = f_schlick(v_dot_h, authored_f0);
    let d = d_ggx(n_dot_h, alpha2);
    let vis = v_smith_ggx_correlated(n_dot_l, n_dot_v, alpha2);
    let specular_raw = d * vis * f;
    let direct_luma = dot(specular_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let direct_cap = 1.0 / (1.0 + direct_luma / 0.3);
    let specular = specular_raw * direct_cap;
    let diffuse = (vec3<f32>(1.0) - f) * diffuse_color / PI;
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
        let lod_bias = lighting.shadow_cascade_splits.w;
        var survives = true;
        if (material.emissive.w > 0.5) {
            let mask_lod = mask_texture_lod(
                in.uv,
                textureDimensions(base_color_tex),
                lod_bias,
            );
            if (mask_lod <= 0.5) {
                let authored_alpha =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, 0.0).a *
                    in.color.a;
                survives = authored_alpha >= alpha_cutoff;
            } else if (mask_lod >= 1.0) {
                let coverage =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, mask_lod).a;
                survives = coverage >= mask_coverage_threshold(in.clip_position.xy);
            } else {
                let authored_alpha =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, 0.0).a *
                    in.color.a;
                let coverage =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, 1.0).a;
                survives = mask_coverage_survives(
                    authored_alpha,
                    coverage,
                    alpha_cutoff,
                    mask_lod,
                    in.clip_position.xy,
                );
            }
        } else {
            let raw_alpha =
                textureSampleBias(base_color_tex, base_color_samp, in.uv, lod_bias).a *
                in.color.a;
            survives = raw_alpha >= alpha_cutoff;
        }
        if (!survives) { discard; }
    }
}

fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {
    var n = normalize(in.normal);
    // Keep the interpolated geometric normal separate from the normal-mapped
    // shading normal.  The texture path already contributes both Toksvig
    // shortening and LEADR-style baked mip variance below; differentiating
    // the mapped normal as well counted that same texture variation a second
    // time.  On compact glossy props (Bistro's painted scooter is the
    // clearest case) the duplicate variance widened a 0.05 paint lobe toward
    // matte and removed the environment response that gives the surface its
    // shape.  Screen-space specular AA still integrates actual geometric
    // curvature through this retained normal.
    let geometric_n = n;

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
    // Derivatives for the no-tangent TBN fallback, taken here in uniform
    // control flow (inside the branch below they would fail WGSL uniformity
    // analysis on WebGPU).
    let tbn_dp1 = dpdx(in.world_pos);
    let tbn_dp2 = dpdy(in.world_pos);
    let tbn_duv1 = dpdx(in.uv);
    let tbn_duv2 = dpdy(in.uv);
    let tlen2 = dot(in.tangent.xyz, in.tangent.xyz);
    if (tlen2 > 0.0001) {
        let t = normalize(in.tangent.xyz);
        let t_ortho = normalize(t - n * dot(n, t));
        let b = cross(n, t_ortho) * in.tangent.w;
        n = normalize(t_ortho * nm_sample.x + b * nm_sample.y + n * nm_sample.z);
    } else {
        let tbn = compute_tbn(tbn_dp1, tbn_dp2, tbn_duv1, tbn_duv2, n);
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
    var base_color = srgb_to_linear_v(base_tex.rgb) * in.color.rgb;
    let base_alpha = base_tex.a * in.color.a;

    // glTF alpha mode tag: MASK carries its positive authored cutoff,
    // OPAQUE is zero, and BLEND is negative. Only MASK discards here;
    // BLEND is routed through the sorted forward translucent pipeline.
    let alpha_cutoff = material.metal_rough.w;
    //PREPASS_STRIP_BEGIN — SH-055: removed in the prepassed-pipeline variant.
    // The prepassed main pass Equal-tests against prepass-exact depth, so a
    // would-be-discarded pixel fails the depth test anyway; keeping `discard`
    // in the shader disables Adreno LRZ/early-Z for the whole draw and made
    // the canopy overdraw shade the full lighting shader per layer (~250 ms
    // per frame on an Adreno 618). See mod.rs scene_shader_prepassed.
    if (alpha_cutoff > 0.0) {
        var survives = base_alpha >= alpha_cutoff;
        if (material.emissive.w > 0.5) {
            let mask_lod = mask_texture_lod(
                in.uv,
                textureDimensions(base_color_tex),
                lod_bias,
            );
            if (mask_lod <= 0.5) {
                let authored_alpha =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, 0.0).a *
                    in.color.a;
                survives = authored_alpha >= alpha_cutoff;
            } else if (mask_lod >= 1.0) {
                let coverage =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, mask_lod).a;
                survives = coverage >= mask_coverage_threshold(in.clip_position.xy);
            } else {
                let authored_alpha =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, 0.0).a *
                    in.color.a;
                let coverage =
                    textureSampleLevel(base_color_tex, base_color_samp, in.uv, 1.0).a;
                survives = mask_coverage_survives(
                    authored_alpha,
                    coverage,
                    alpha_cutoff,
                    mask_lod,
                    in.clip_position.xy,
                );
            }
        }
        if (!survives) { discard; }
    }
    //PREPASS_STRIP_END

    // Two-sided foliage normal. Alpha-cutout cards (leaves, grass blades)
    // are seen from both sides, but the geometric normal only faces one
    // way — the back side otherwise shades with N pointing away from the
    // sun AND from the sky irradiance, which is why grass tufts rendered
    // as solid black cards from one side. Flip the shading normal toward
    // the viewer for cutout materials only; opaque geometry is untouched.
    if (alpha_cutoff > 0.0 && dot(n, lighting.camera_pos.xyz - in.world_pos) < 0.0) {
        n = -n;
    }
    if (alpha_cutoff < 0.0 && !front_facing) {
        n = -n;
    }

    // glTF metallicRoughnessTexture: G=roughness, B=metallic (linear).
    // KHR specularGlossinessTexture: RGB=specular (sRGB), A=glossiness
    // (linear). `metal_rough.z` selects the workflow so both imported
    // material models share this existing texture binding and sample.
    // When the material has no material-response texture (workflow 0), the
    // binding falls back to an arbitrary scene texture (whatever lives
    // at index 0) — multiplying its random R/G/B into our factors
    // produces incorrect material values. Use the factors directly in
    // that case.
    let mr_tex_sample = textureSample(mr_tex, mr_samp, in.uv);
    let has_mr = material.metal_rough.z > 0.5 && material.metal_rough.z < 1.5;
    let has_spec_gloss = material.metal_rough.z > 1.5;
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
    var metallic_raw = select(
        clamp(material.metal_rough.x, 0.0, 1.0),
        clamp(mr_tex_sample.b * material.metal_rough.x, 0.0, 1.0),
        has_mr,
    );
    var authored_specular = vec3<f32>(0.04);
    var ssr_base_color = base_color;
    if (has_spec_gloss) {
        authored_specular = srgb_to_linear_v(mr_tex_sample.rgb) *
            material.spec_gloss.rgb;
        // Retain the old conversion only for the compact metallic/roughness
        // and albedo buffers consumed by screen-space effects. Main-scene
        // direct and environment lighting below use the authored diffuse and
        // F0 independently, avoiding the false-metal appearance.
        let converted = specgloss_to_metalrough_pixel(base_color, authored_specular);
        ssr_base_color = converted.rgb;
        metallic_raw = converted.a;
        roughness_raw = clamp(
            1.0 - mr_tex_sample.a * material.spec_gloss.a,
            0.045,
            1.0,
        );
    }
    let metallic = metallic_raw;
    // Preserve genuinely smooth authored spec-gloss surfaces. The standard
    // MR path retains Bloom's conservative dielectric floor; specular AA
    // below still controls highlight shimmer for this exact workflow.
    let dielectric_floor = select(0.15, 0.045, has_spec_gloss);
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
    let nm_dx = dpdx(geometric_n);
    let nm_dy = dpdy(geometric_n);
    let curvature_sq = dot(nm_dx, nm_dx) + dot(nm_dy, nm_dy);
    // Kaplanyan 2016 screen-space kernel. The aggressive coefficient/cap
    // integrates unresolved geometric curvature. Texture-space variation is
    // already represented by the two terms above and must not enter here a
    // second time.
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
        if (has_spec_gloss) {
            lit += shade_specular_glossiness_pbr(
                n, v, legacy_dir, lighting.light_color.rgb,
                lighting.light_dir.w, base_color, authored_specular, roughness,
            ) * direct_shadow;
        } else {
            lit += shade_pbr(n, v, legacy_dir, lighting.light_color.rgb,
                             lighting.light_dir.w, base_color, metallic, roughness)
                 * direct_shadow;
        }
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
        if (has_spec_gloss) {
            lit += shade_specular_glossiness_pbr(
                n, v, l, dl.color.rgb, dl.direction.w,
                base_color, authored_specular, roughness,
            );
        } else {
            lit += shade_pbr(n, v, l, dl.color.rgb, dl.direction.w,
                             base_color, metallic, roughness);
        }
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
            if (has_spec_gloss) {
                lit += shade_specular_glossiness_pbr(
                    n, v, l, pl.color.rgb, pl.color.w * atten2,
                    base_color, authored_specular, roughness,
                );
            } else {
                lit += shade_pbr(n, v, l, pl.color.rgb, pl.color.w * atten2,
                                 base_color, metallic, roughness);
            }
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

    //IBL_STRIP_BEGIN — SH-055 probe: replaced by a flat-ambient fallback when
    // BLOOM_SKIP contains "ibl" (see mod.rs), to measure the split-sum IBL
    // chain's per-fragment cost on mobile GPUs. Only ibl_diffuse/ibl_spec
    // escape this block.
    let n_dot_v_ibl = max(dot(n, v), 0.0);
    let mr_f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let f0 = select(mr_f0, authored_specular, has_spec_gloss);

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
    let diffuse_weight = select(1.0 - metallic, 1.0, has_spec_gloss);
    let kd = (vec3<f32>(1.0) - f_ibl) * diffuse_weight;
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

    // Preserve the Fresnel response at every authored roughness. The former
    // anti-stripe workaround multiplied smooth dielectrics by roughness twice,
    // driving glossy dark materials (Bistro wine bottles are the canonical
    // case) to exactly zero environment reflection. Specular occlusion above,
    // the bounded soft radiance compression below, normal-variance filtering,
    // and exclusive SSR ownership are the appropriate stability controls;
    // material roughness selects the prefiltered lobe and must not also erase
    // that lobe's energy.
    // Reinhard-style soft luma compression keeps isolated HDR environment
    // samples finite and continuous without imposing a roughness-dependent
    // energy hole.
    let cap2_luma = dot(ibl_spec_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let cap2 = 1.0 / (1.0 + cap2_luma / 0.3);
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
    let ibl_spec = ibl_spec_raw * spec_occ * cap2 * (1.0 - ssr_own);
    //IBL_STRIP_END

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
    // translucent. A surviving MASK texel (positive cutoff) is fully opaque;
    // retaining its sampled alpha would accidentally blend it with surfaces
    // behind it and make output depend on submission order. BLEND uses a
    // negative sentinel and keeps fractional alpha for forward compositing.
    let non_opaque_alpha = select(base_alpha, 1.0, alpha_cutoff > 0.0);
    let out_alpha = select(in.color.a, non_opaque_alpha, alpha_cutoff != 0.0);

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
        vec4<f32>(ssr_base_color, 1.0 - shadow_factor),
    );
}

@fragment
fn fs_main_scene(
    in: VertexOutputScene,
    @builtin(front_facing) front_facing: bool,
) -> SceneOut {
    return shade_main_scene(in, front_facing);
}

@fragment
fn fs_transparent_scene(
    in: VertexOutputScene,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    return shade_main_scene(in, front_facing).color;
}
"#
);

/// Build the dedicated imported-transmission scene shader without changing
/// the shader compiled for ordinary scene materials.
///
/// Desktop/native backends add group 4 and sample the render-graph-owned
/// pre-translucency color/depth snapshots. Four-bind-group targets
/// (`fold_scene_inputs`: WebGPU/Android) compile the same physical lobe but
/// source transmitted radiance from the environment map instead.
pub(in crate::renderer) fn scene_refractive_shader_source(
    base_scene_shader: &str,
    folded_scene_inputs: bool,
    screen_space_reflections: bool,
    secondary_uv: bool,
) -> String {
    assert!(
        !folded_scene_inputs || !screen_space_reflections,
        "folded four-bind-group targets cannot add native reflection inputs"
    );
    const JOINT_DECLARATION: &str =
        "@group(3) @binding(1) var<uniform> joints_prev: JointMatrices;";
    let scene_inputs = if folded_scene_inputs {
        ""
    } else if screen_space_reflections {
        r#"
struct RefractiveReflectionParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    params: vec4<f32>,
    planar_plane: vec4<f32>,
};

@group(4) @binding(0) var refractive_scene_color_tex: texture_2d<f32>;
@group(4) @binding(1) var refractive_scene_color_samp: sampler;
@group(4) @binding(2) var refractive_scene_depth_tex: texture_depth_2d;
@group(4) @binding(3) var<uniform> refractive_reflection: RefractiveReflectionParams;
@group(4) @binding(4) var refractive_planar_tex: texture_2d<f32>;
"#
    } else {
        r#"
@group(4) @binding(0) var refractive_scene_color_tex: texture_2d<f32>;
@group(4) @binding(1) var refractive_scene_color_samp: sampler;
@group(4) @binding(2) var refractive_scene_depth_tex: texture_depth_2d;
"#
    };
    let physical_declarations = format!(
        r#"{JOINT_DECLARATION}

struct TransmissionFactors {{
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    transmission_uv: vec4<f32>,
    transmission_rotation: vec4<f32>,
    thickness_uv: vec4<f32>,
    thickness_rotation: vec4<f32>,
}};

@group(2) @binding(11) var transmission_tex: texture_2d<f32>;
@group(2) @binding(12) var transmission_samp: sampler;
@group(2) @binding(13) var thickness_tex: texture_2d<f32>;
@group(2) @binding(14) var thickness_samp: sampler;
@group(2) @binding(15) var<uniform> transmission_material: TransmissionFactors;
{scene_inputs}"#
    );
    let mut source = base_scene_shader.replacen(JOINT_DECLARATION, &physical_declarations, 1);
    assert_ne!(
        source, base_scene_shader,
        "scene shader joint declaration changed; refractive ABI injection must be updated"
    );
    let layered_secondary_uv =
        secondary_uv && base_scene_shader.contains("fn layered_secondary_uv(in:");
    let model_scale_location = if layered_secondary_uv { 8 } else { 7 };
    // The ordinary per-draw group is vertex-visible only. Carry model scale as
    // a refractive-variant-only interpolant instead of widening that established
    // layout to the fragment stage for every scene material.
    source = source.replacen(
        "    @location(6) prev_clip: vec4<f32>,",
        &format!(
            "    @location(6) prev_clip: vec4<f32>,\n    @location({model_scale_location}) model_scale: f32,"
        ),
        1,
    );
    source = source.replacen(
        "        o.prev_clip = u.prev_mvp * prev_world4;",
        "        o.prev_clip = u.prev_mvp * prev_world4;\n\
         o.model_scale = (length(u.model[0].xyz) + length(u.model[1].xyz) \
         + length(u.model[2].xyz)) / 3.0;",
        1,
    );
    source = source.replacen(
        "    out.prev_clip = u.prev_mvp * vec4<f32>(prev_local, 1.0);",
        "    out.prev_clip = u.prev_mvp * vec4<f32>(prev_local, 1.0);\n\
         out.model_scale = (length(u.model[0].xyz) + length(u.model[1].xyz) \
         + length(u.model[2].xyz)) / 3.0;",
        1,
    );
    if secondary_uv && !layered_secondary_uv {
        source = source.replacen(
            "    @location(6) tangent: vec4<f32>,\n};",
            "    @location(6) tangent: vec4<f32>,\n\
             @location(7) secondary_uv: vec2<f32>,\n\
             };",
            1,
        );
        source = source.replacen(
            "    @location(7) model_scale: f32,",
            "    @location(7) model_scale: f32,\n\
             @location(8) secondary_uv: vec2<f32>,",
            1,
        );
        source = source.replacen(
            "        o.uv = in.uv;",
            "        o.uv = in.uv;\n        o.secondary_uv = in.secondary_uv;",
            1,
        );
        source = source.replacen(
            "    out.uv = in.uv;",
            "    out.uv = in.uv;\n    out.secondary_uv = in.secondary_uv;",
            1,
        );
        assert!(
            source.contains("@location(8) secondary_uv"),
            "scene vertex ABI changed; refractive UV1 injection must be updated"
        );
    } else if layered_secondary_uv {
        assert!(
            source.contains("@location(7) secondary_uv")
                && source.contains("@location(8) model_scale"),
            "layered refractive vertex ABI changed; specialization must be updated"
        );
    }

    let transmitted_radiance = if folded_scene_inputs {
        r#"
    // Constrained four-bind-group fallback: preserve the refractive material
    // type and its Fresnel/absorption response, but source off-screen
    // transmitted radiance from the prefiltered environment.
    let max_transmission_mip = max(f32(textureNumLevels(env_tex)) - 1.0, 0.0);
    let transmitted_radiance = env_sample_lod(
        refracted_direction,
        roughness * max_transmission_mip,
    );
    let undistorted_radiance = env_sample_lod(-v, roughness * max_transmission_mip);
"#
    } else {
        r#"
    let scene_dimensions_u = textureDimensions(refractive_scene_color_tex, 0);
    let scene_dimensions = vec2<f32>(scene_dimensions_u);
    let current_ndc = in.curr_clip.xy / max(abs(in.curr_clip.w), 0.000001);
    let current_uv = clamp(
        vec2<f32>(current_ndc.x * 0.5 + 0.5, 0.5 - current_ndc.y * 0.5),
        vec2<f32>(0.0001),
        vec2<f32>(0.9999),
    );

    // Convert the refracted world-space direction into a stable screen-space
    // travel distance using the fragment's world-position derivatives. The
    // material factor already carries static glTF node scale baked by the
    // importer; the interpolant adds the later draw/instance scale. A 64-pixel
    // cap prevents pathological assets from sampling unrelated frame regions.
    let world_dx = dpdx(in.world_pos);
    let world_dy = dpdy(in.world_pos);
    let world_dx_len = max(length(world_dx), 0.000001);
    let world_dy_len = max(length(world_dy), 0.000001);
    let screen_tangent_x = world_dx / world_dx_len;
    let screen_tangent_y = world_dy / world_dy_len;
    let ray_distance = thickness_world
        / max(abs(dot(refracted_direction, n)), 0.15);
    var offset_pixels = vec2<f32>(
        dot(refracted_direction, screen_tangent_x) * ray_distance / world_dx_len,
        dot(refracted_direction, screen_tangent_y) * ray_distance / world_dy_len,
    );
    offset_pixels = clamp(offset_pixels, vec2<f32>(-64.0), vec2<f32>(64.0));
    var refracted_uv = clamp(
        current_uv + offset_pixels / scene_dimensions,
        vec2<f32>(0.0001),
        vec2<f32>(0.9999),
    );

    // Reject offsets that cross in front of this glass surface. This keeps a
    // nearby opaque silhouette from being pulled through the refractor.
    let candidate_pixel = clamp(
        vec2<i32>(refracted_uv * scene_dimensions),
        vec2<i32>(0),
        vec2<i32>(scene_dimensions_u) - vec2<i32>(1),
    );
    let candidate_depth = textureLoad(
        refractive_scene_depth_tex,
        candidate_pixel,
        0,
    );
    if (candidate_depth + 0.0005 < in.clip_position.z) {
        refracted_uv = current_uv;
    }

    var transmitted_radiance = textureSampleLevel(
        refractive_scene_color_tex,
        refractive_scene_color_samp,
        refracted_uv,
        0.0,
    ).rgb;
    // Deterministic five-tap rough transmission. Smooth glass stays at one
    // fetch; rough glass integrates a bounded footprint without temporal noise.
    if (roughness > 0.08) {
        let blur_uv = vec2<f32>(8.0 * roughness * roughness) / scene_dimensions;
        transmitted_radiance = (
            transmitted_radiance * 4.0
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv + vec2<f32>(blur_uv.x, 0.0),
                0.0,
            ).rgb
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv - vec2<f32>(blur_uv.x, 0.0),
                0.0,
            ).rgb
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv + vec2<f32>(0.0, blur_uv.y),
                0.0,
            ).rgb
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv - vec2<f32>(0.0, blur_uv.y),
                0.0,
            ).rgb
        ) * 0.125;
    }
    let undistorted_radiance = textureSampleLevel(
        refractive_scene_color_tex,
        refractive_scene_color_samp,
        current_uv,
        0.0,
    ).rgb;
"#
    };

    let reflection_helpers = if screen_space_reflections {
        r#"
fn refractive_screen_reflection(
    in: VertexOutputScene,
    reflected_direction: vec3<f32>,
    roughness: f32,
    environment_fallback: vec3<f32>,
) -> vec3<f32> {
    // The ordinary SSR target cannot be reused here: it was traced from the
    // opaque surface behind this fragment and therefore owns a different
    // normal/reflection ray. Launch one bounded ray from the glass fragment
    // against the immutable opaque snapshots instead.
    if (refractive_reflection.params.x < 0.5
        || roughness >= refractive_reflection.params.w) {
        return environment_fallback;
    }

    let dimensions_u = textureDimensions(refractive_scene_color_tex, 0);
    let dimensions = vec2<f32>(dimensions_u);
    let step_count = u32(refractive_reflection.params.z);
    let max_distance = refractive_reflection.params.y;
    let start_view = (
        refractive_reflection.view * vec4<f32>(in.world_pos, 1.0)
    ).xyz;
    let reflected_view = normalize((
        refractive_reflection.view * vec4<f32>(reflected_direction, 0.0)
    ).xyz);
    let start_clip = refractive_reflection.proj * vec4<f32>(start_view, 1.0);
    var previous_ray_depth = start_clip.z / max(abs(start_clip.w), 0.000001);
    var hit_uv = vec2<f32>(-1.0);
    var hit_confidence = 0.0;

    // Quadratic spacing keeps the nearest samples dense enough for window
    // frames and props while still reaching the same architectural range as
    // the established opaque SSR pass. The loop bound and every texture read
    // are fixed by the lazy uniform (currently eight).
    for (var step = 0u; step < step_count; step = step + 1u) {
        let fraction = f32(step + 1u) / f32(step_count);
        let distance = max_distance * fraction * fraction;
        let ray_view = start_view + reflected_view * distance;
        let ray_clip = refractive_reflection.proj * vec4<f32>(ray_view, 1.0);
        if (ray_clip.w <= 0.000001) {
            break;
        }
        let ray_ndc = ray_clip.xyz / ray_clip.w;
        if (ray_ndc.x <= -1.0 || ray_ndc.x >= 1.0
            || ray_ndc.y <= -1.0 || ray_ndc.y >= 1.0
            || ray_ndc.z <= 0.0 || ray_ndc.z >= 1.0) {
            break;
        }
        let ray_uv = vec2<f32>(
            ray_ndc.x * 0.5 + 0.5,
            0.5 - ray_ndc.y * 0.5,
        );
        let pixel = clamp(
            vec2<i32>(ray_uv * dimensions),
            vec2<i32>(0),
            vec2<i32>(dimensions_u) - vec2<i32>(1),
        );
        let scene_depth = textureLoad(refractive_scene_depth_tex, pixel, 0);
        let depth_delta = ray_ndc.z - scene_depth;
        let depth_stride = abs(ray_ndc.z - previous_ray_depth);
        let thickness = max(depth_stride * 2.0, 0.00075);
        if (scene_depth < 0.9999
            && depth_delta >= 0.0
            && depth_delta <= thickness) {
            hit_uv = ray_uv;
            hit_confidence = 1.0 - smoothstep(
                thickness * 0.25,
                thickness,
                depth_delta,
            );
            break;
        }
        previous_ray_depth = ray_ndc.z;
    }

    if (hit_uv.x < 0.0) {
        return environment_fallback;
    }
    // Suppress the screen boundary before it can pop. Rough glass fades to
    // the prefiltered environment rather than returning an incorrectly sharp
    // scene-color tap (the snapshot deliberately has no mip chain).
    let edge_pixels = min(
        min(hit_uv.x, 1.0 - hit_uv.x) * dimensions.x,
        min(hit_uv.y, 1.0 - hit_uv.y) * dimensions.y,
    );
    let edge_weight = smoothstep(0.0, 8.0, edge_pixels);
    let roughness_weight = 1.0 - smoothstep(
        refractive_reflection.params.w * 0.45,
        refractive_reflection.params.w,
        roughness,
    );
    let raw = textureSampleLevel(
        refractive_scene_color_tex,
        refractive_scene_color_samp,
        hit_uv,
        0.0,
    ).rgb;
    let screen_radiance = select(vec3<f32>(0.0), raw, raw == raw);
    let source_weight = clamp(
        edge_weight * roughness_weight * hit_confidence,
        0.0,
        1.0,
    );
    return mix(environment_fallback, screen_radiance, source_weight);
}

fn refractive_planar_sample(
    in: VertexOutputScene,
    roughness: f32,
) -> vec4<f32> {
    let plane = refractive_reflection.planar_plane;
    if (dot(plane.xyz, plane.xyz) < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    // A plane crossing unrelated vertical glass must not make the global
    // first-probe choice leak onto that surface. Use the unperturbed vertex
    // normal so authored water waves can still perturb the sampled reflection.
    if (abs(dot(normalize(plane.xyz), normalize(in.normal))) < 0.8) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    let plane_distance = abs(dot(plane.xyz, in.world_pos) - plane.w);
    if (plane_distance > 0.075) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    let ndc = in.curr_clip.xy / max(abs(in.curr_clip.w), 0.000001);
    let uv = clamp(
        vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5),
        vec2<f32>(0.0001),
        vec2<f32>(0.9999),
    );
    let planar = textureSampleLevel(
        refractive_planar_tex,
        refractive_scene_color_samp,
        uv,
        0.0,
    );
    let planar_safe = select(vec3<f32>(0.0), planar.rgb, planar.rgb == planar.rgb);
    // The existing planar capture has no mip chain. Fade its exact reflection
    // into the lower tiers as roughness grows instead of returning an
    // incorrectly sharp image. Alpha zero is the probe's explicit miss value.
    let roughness_weight = 1.0 - smoothstep(0.18, 0.45, roughness);
    let source_weight = clamp(planar.a * roughness_weight, 0.0, 1.0);
    return vec4<f32>(planar_safe, source_weight);
}
"#
    } else {
        ""
    };
    let reflected_radiance = if screen_space_reflections {
        r#"
    let reflected_environment = env_sample_lod(
        reflected_direction,
        roughness * max(f32(textureNumLevels(env_tex)) - 1.0, 0.0),
    );
    let planar_reflected = refractive_planar_sample(in, roughness);
    var reflected = mix(
        reflected_environment,
        planar_reflected.rgb,
        max(planar_reflected.a, 0.0),
    );
    // A matching explicit probe is authoritative for its plane: alpha zero is
    // its documented geometry miss and therefore reveals the environment sky.
    // Only glass without an applicable probe pays for the bounded screen march.
    if (planar_reflected.a < 0.0) {
        reflected = refractive_screen_reflection(
            in,
            reflected_direction,
            roughness,
            reflected_environment,
        );
    }
"#
    } else {
        r#"
    let reflected = env_sample_lod(
        reflected_direction,
        roughness * max(f32(textureNumLevels(env_tex)) - 1.0, 0.0),
    );
"#
    };

    let transmission_source_uv = if secondary_uv {
        "select(\n            in.uv,\n            in.secondary_uv,\n            transmission_material.transmission_rotation.w > 0.5,\n        )"
    } else {
        "in.uv"
    };
    let thickness_source_uv = if secondary_uv {
        "select(\n            in.uv,\n            in.secondary_uv,\n            transmission_material.thickness_rotation.z > 0.5,\n        )"
    } else {
        "in.uv"
    };
    source.push_str(&format!(
        r#"

{reflection_helpers}

fn physical_texture_uv(
    uv: vec2<f32>,
    offset_scale: vec4<f32>,
    rotation: vec2<f32>,
) -> vec2<f32> {{
    let scaled = uv * offset_scale.zw;
    let rotated = vec2<f32>(
        rotation.x * scaled.x - rotation.y * scaled.y,
        rotation.y * scaled.x + rotation.x * scaled.y,
    );
    return offset_scale.xy + rotated;
}}

fn refractive_scene_normal(in: VertexOutputScene, front_facing: bool) -> vec3<f32> {{
    var n = normalize(in.normal);
    let normal_sample = textureSampleBias(
        normal_tex,
        normal_samp,
        in.uv,
        1.0 + lighting.shadow_cascade_splits.w,
    ).xyz * 2.0 - 1.0;
    let mapped = normal_sample / max(length(normal_sample), 0.000001);
    let tangent_len2 = dot(in.tangent.xyz, in.tangent.xyz);
    if (tangent_len2 > 0.0001) {{
        let tangent = normalize(in.tangent.xyz);
        let tangent_ortho = normalize(tangent - n * dot(n, tangent));
        let bitangent = cross(n, tangent_ortho) * in.tangent.w;
        n = normalize(
            tangent_ortho * mapped.x + bitangent * mapped.y + n * mapped.z,
        );
    }} else {{
        let tbn = compute_tbn(
            dpdx(in.world_pos),
            dpdy(in.world_pos),
            dpdx(in.uv),
            dpdy(in.uv),
            n,
        );
        n = normalize(tbn * mapped);
    }}
    if (!front_facing) {{
        n = -n;
    }}
    return n;
}}

struct RefractiveSceneOut {{
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
}};

@fragment
fn fs_refractive_scene(
    in: VertexOutputScene,
    @builtin(front_facing) front_facing: bool,
) -> RefractiveSceneOut {{
    // Reuse the established direct/IBL PBR evaluation for the non-transmitted
    // energy, then split the dielectric lobe below. This also preserves MASK
    // discard semantics for the unusual but legal MASK+transmission case.
    let surface = shade_main_scene(in, front_facing);
    let n = refractive_scene_normal(in, front_facing);
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);

    let base_texel = textureSample(base_color_tex, base_color_samp, in.uv);
    var base_color = srgb_to_linear_v(base_texel.rgb) * in.color.rgb;
    let base_alpha = base_texel.a * in.color.a;
    let mr_texel = textureSample(mr_tex, mr_samp, in.uv);
    let has_mr = material.metal_rough.z > 0.5 && material.metal_rough.z < 1.5;
    let has_spec_gloss = material.metal_rough.z > 1.5;
    var metallic = select(
        clamp(material.metal_rough.x, 0.0, 1.0),
        clamp(mr_texel.b * material.metal_rough.x, 0.0, 1.0),
        has_mr,
    );
    var roughness = select(
        clamp(material.metal_rough.y, 0.045, 1.0),
        clamp(mr_texel.g * material.metal_rough.y, 0.045, 1.0),
        has_mr,
    );
    if (has_spec_gloss) {{
        let authored_specular = srgb_to_linear_v(mr_texel.rgb) *
            material.spec_gloss.rgb;
        let converted = specgloss_to_metalrough_pixel(base_color, authored_specular);
        base_color = converted.rgb;
        metallic = converted.a;
        roughness = clamp(
            1.0 - mr_texel.a * material.spec_gloss.a,
            0.045,
            1.0,
        );
    }}

    let transmission_uv = physical_texture_uv(
        {transmission_source_uv},
        transmission_material.transmission_uv,
        transmission_material.transmission_rotation.xy,
    );
    let texture_transmission = select(
        1.0,
        textureSample(
            transmission_tex,
            transmission_samp,
            transmission_uv,
        ).r,
        transmission_material.transmission.w > 0.5,
    );
    let dielectric_weight = 1.0 - metallic;
    let transmission_weight = clamp(
        transmission_material.transmission.x
            * texture_transmission
            * dielectric_weight,
        0.0,
        1.0,
    );

    let thickness_uv = physical_texture_uv(
        {thickness_source_uv},
        transmission_material.thickness_uv,
        transmission_material.thickness_rotation.xy,
    );
    let texture_thickness = select(
        1.0,
        textureSample(
            thickness_tex,
            thickness_samp,
            thickness_uv,
        ).g,
        transmission_material.transmission_rotation.z > 0.5,
    );
    let mean_model_scale = max(in.model_scale, 0.0);
    let thickness_world = max(
        transmission_material.transmission.z
            * texture_thickness
            * mean_model_scale,
        0.0,
    );

    let ior = max(transmission_material.transmission.y, 1.0);
    let eta = 1.0 / ior;
    var refracted_direction = refract(-v, n, eta);
    if (dot(refracted_direction, refracted_direction) < 0.000001) {{
        refracted_direction = reflect(-v, n);
    }}
    refracted_direction = normalize(refracted_direction);

{transmitted_radiance}

    var absorption = vec3<f32>(1.0);
    if (transmission_material.attenuation.w > 0.0 && thickness_world > 0.0) {{
        let optical_distance = thickness_world
            / transmission_material.attenuation.w;
        absorption = pow(
            max(transmission_material.attenuation.rgb, vec3<f32>(0.000001)),
            vec3<f32>(optical_distance),
        );
    }}
    let transmitted = transmitted_radiance * base_color * absorption;

    let f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = f0_scalar
        + (1.0 - f0_scalar) * pow(1.0 - n_dot_v, 5.0);
    let reflected_direction = reflect(-v, n);
{reflected_radiance}

    // Energy partition: the ordinary PBR surface owns the opaque fraction;
    // the transmission fraction is split exactly between Fresnel reflection
    // and absorbed transmitted radiance.
    let dielectric_transmission = mix(transmitted, reflected, fresnel);
    var hdr = surface.color.rgb * (1.0 - transmission_weight)
        + dielectric_transmission * transmission_weight;

    // glTF BLEND+transmission additionally applies base-color alpha. Because
    // this fragment already composites against the snapshot, write alpha=1
    // to avoid applying the background a second time in fixed-function blend.
    if (material.metal_rough.w < 0.0) {{
        hdr = mix(undistorted_radiance, hdr, clamp(base_alpha, 0.0, 1.0));
    }}
    hdr = select(vec3<f32>(0.0), hdr, hdr == hdr);
    return RefractiveSceneOut(vec4<f32>(hdr, 1.0), surface.velocity);
}}
"#
    ));
    source
}
