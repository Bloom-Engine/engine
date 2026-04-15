use wgpu::util::DeviceExt;
use std::collections::HashMap;

// ============================================================
// Constants
// ============================================================

const MAX_UNIFORM_SLOTS: usize = 8;

pub const IDENTITY_MAT4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// ============================================================
// Vertex and Uniform types
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms2D {
    screen_size: [f32; 2],
    _pad: [f32; 2],
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms3D {
    mvp: [[f32; 4]; 4],
    model_tint: [f32; 4],
}

/// Scene-pipeline per-material factors — the scalar parts of a glTF
/// PBR material that get multiplied onto the corresponding texture
/// samples. Sized to a multiple of 16 bytes for UBO alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneMaterialUniforms {
    /// x = metallic_factor, y = roughness_factor, z,w = padding
    pub metal_rough: [f32; 4],
    /// rgb = emissive_factor, w = padding
    pub emissive: [f32; 4],
}

impl SceneMaterialUniforms {
    pub fn new(metallic: f32, roughness: f32, emissive: [f32; 3]) -> Self {
        Self {
            metal_rough: [metallic, roughness, 0.0, 0.0],
            emissive: [emissive[0], emissive[1], emissive[2], 0.0],
        }
    }
}

const MAX_DIR_LIGHTS: usize = 4;
const MAX_POINT_LIGHTS: usize = 16;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DirLight {
    direction: [f32; 4],  // xyz + intensity
    color: [f32; 4],      // rgb + _pad
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PointLight {
    position: [f32; 4],   // xyz + range
    color: [f32; 4],      // rgb + intensity
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniforms {
    ambient: [f32; 4],                              // rgb + intensity
    light_dir: [f32; 4],                             // xyz + intensity (legacy, kept for compat)
    light_color: [f32; 4],                           // rgb + _pad (legacy)
    dir_light_count: [f32; 4],                       // [count, 0, 0, 0]
    dir_lights: [DirLight; MAX_DIR_LIGHTS],          // additional directional lights
    point_light_count: [f32; 4],                     // [count, 0, 0, 0]
    point_lights: [PointLight; MAX_POINT_LIGHTS],    // point lights
    /// Camera world-space position (xyz) + env intensity multiplier
    /// (w). Scene shader uses xyz to compute V = normalize(camera_pos
    /// - world_pos) for GGX specular, and multiplies w into every env
    /// sample so IBL stays in sync with the sky pass when the user
    /// scales their HDR. Written once per frame before the main pass.
    camera_pos: [f32; 4],
    /// Light view-projection matrix for the primary directional
    /// light's shadow map. Scene shader projects world_pos through
    /// this to derive the shadow-map UV. Identity = no shadow cast.
    shadow_light_vp: [[f32; 4]; 4],
}

impl LightingUniforms {
    fn defaults() -> Self {
        Self {
            ambient: [1.0, 1.0, 1.0, 0.3],
            light_dir: [0.5, 1.0, 0.3, 0.7],
            light_color: [1.0, 1.0, 1.0, 0.0],
            dir_light_count: [0.0; 4],
            dir_lights: [DirLight { direction: [0.0; 4], color: [0.0; 4] }; MAX_DIR_LIGHTS],
            point_light_count: [0.0; 4],
            point_lights: [PointLight { position: [0.0; 4], color: [0.0; 4] }; MAX_POINT_LIGHTS],
            // w = env_intensity multiplier for IBL + sky. 1.0 matches
            // the path-traced reference; apps with bright HDR envs
            // typically dial to 0.2–0.5 via set_env_intensity.
            camera_pos: [0.0, 0.0, 0.0, 1.0],
            shadow_light_vp: IDENTITY_MAT4,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex2D {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub joints: [f32; 4],   // bone indices (as floats for simplicity)
    pub weights: [f32; 4],  // bone weights (sum to 1.0, or all 0.0 for unskinned)
    pub tangent: [f32; 4],  // xyz = tangent direction, w = bitangent sign (±1). All zero = no tangent data; scene shader then skips normal mapping.
}

impl Default for Vertex3D {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl Vertex3D {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },   // position
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },  // normal
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },  // color
                wgpu::VertexAttribute { offset: 40, shader_location: 3, format: wgpu::VertexFormat::Float32x2 },  // uv
                wgpu::VertexAttribute { offset: 48, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },  // joints
                wgpu::VertexAttribute { offset: 64, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },  // weights
                wgpu::VertexAttribute { offset: 80, shader_location: 6, format: wgpu::VertexFormat::Float32x4 },  // tangent
            ],
        }
    }
}

// ============================================================
// Draw call tracking
// ============================================================

struct DrawCall2D {
    texture_idx: u32,
    uniform_idx: u32,
    index_start: u32,
}

struct DrawCall3D {
    texture_idx: u32,
    index_start: u32,
}

#[derive(PartialEq, Clone, Copy)]
pub enum RenderMode {
    ScreenSpace,
    Mode2D,
    Mode3D,
}

// ============================================================
// Shaders
// ============================================================

const SHADER_2D: &str = "
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

const SHADER_3D: &str = "
struct Uniforms3D {
    mvp: mat4x4<f32>,
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
    out.clip_position = u.mvp * pos;
    out.normal = norm.xyz;
    out.world_pos = pos.xyz;
    out.color = in.color * u.model_tint;
    out.uv = in.uv;
    return out;
}

struct Fs3DOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec2<f32>,
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
    // Immediate-mode 3D draws (drawCube etc.) aren't PBR — output
    // 0 metallic / 1 roughness so SSR doesn't try to reflect them.
    return Fs3DOut(
        vec4<f32>(tex_color.rgb * in.color.rgb * lit, tex_color.a * in.color.a),
        vec2<f32>(0.0, 1.0),
    );
}
";

// ============================================================
// Scene pipeline shader (retained mode scene graph)
// ============================================================
//
// Derived from SHADER_3D but extends the material bindings with a
// normal map (and stubs out for future metallic-roughness / emissive
// additions). The only other difference vs SHADER_3D is the tangent
// vertex input and the TBN-based normal perturbation in the fragment
// shader. Kept as a separate pipeline from pipeline_3d so immediate-
// mode 3D draws (drawCube, draw_model_cached, etc.) don't pay the
// extra binding cost and don't need tangents.

const SCENE_SHADER: &str = "
struct Uniforms3D {
    mvp: mat4x4<f32>,
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
    shadow_light_vp: mat4x4<f32>,
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
};

@group(0) @binding(0) var<uniform> u: Uniforms3D;
@group(1) @binding(0) var<uniform> lighting: Lighting;
@group(1) @binding(1) var env_tex: texture_2d<f32>;
@group(1) @binding(2) var env_samp: sampler;
@group(1) @binding(3) var brdf_lut_tex: texture_2d<f32>;
@group(1) @binding(4) var brdf_lut_samp: sampler;
@group(1) @binding(5) var shadow_tex: texture_depth_2d;
@group(1) @binding(6) var shadow_samp: sampler_comparison;
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

// Sample the env map at a specific mip level, multiplied by the
// global env_intensity (lighting.camera_pos.w). Keeps IBL diffuse,
// IBL specular and the sky pass scaling in sync so loading the same
// HDR with intensity=2 brightens everything proportionally.
fn env_sample_lod(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(env_tex, env_samp, dir_to_equirect_uv(dir), lod).rgb
         * lighting.camera_pos.w;
}

fn env_sample(dir: vec3<f32>) -> vec3<f32> {
    return textureSample(env_tex, env_samp, dir_to_equirect_uv(dir)).rgb
         * lighting.camera_pos.w;
}

@vertex
fn vs_main_scene(in: VertexInputScene) -> VertexOutputScene {
    var out: VertexOutputScene;
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.world_pos = in.position;
    out.color = in.color * u.model_tint;
    out.uv = in.uv;
    out.tangent = in.tangent;
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

// Sample the shadow map with 4-tap PCF — softens the cascade-edge
// stair-stepping you'd get from a single comparison. Returns 1.0
// for fully lit, 0.0 for fully shadowed. Caller multiplies the
// directional light's diffuse + specular by the result.
fn sample_shadow(world_pos: vec3<f32>) -> f32 {
    let light_clip = lighting.shadow_light_vp * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    // Out-of-frustum: no shadow info → treat as lit.
    if (light_ndc.x < -1.0 || light_ndc.x > 1.0 ||
        light_ndc.y < -1.0 || light_ndc.y > 1.0 ||
        light_ndc.z < 0.0 || light_ndc.z > 1.0) {
        return 1.0;
    }
    let shadow_uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, 1.0 - (light_ndc.y * 0.5 + 0.5));
    let bias = 0.0008;
    let depth_ref = light_ndc.z - bias;

    // 16-tap Poisson-disc PCF. Spreads taps over a wider kernel
    // for smoother penumbras than the old 4-tap box, while keeping
    // the sample budget bounded. Disc offsets pre-baked from a
    // standard Poisson distribution (Wyman/Hardy-style sequence)
    // so we don't pay for per-pixel rotation. Filter radius is
    // 2 texels — gives a noticeable but not blurry-soft shadow
    // with the current 2048×2048 shadow map.
    let dims = textureDimensions(shadow_tex);
    let texel = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));
    let radius = 2.0;
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
        sum = sum + textureSampleCompare(shadow_tex, shadow_samp, shadow_uv + off, depth_ref);
    }
    return sum / 16.0;
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
    let h = normalize(l_dir + v);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, h), 0.0, 1.0);

    let alpha = max(roughness * roughness, 0.001);
    let alpha2 = alpha * alpha;

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let f = f_schlick(v_dot_h, f0);
    let d = d_ggx(n_dot_h, alpha2);
    let vis = v_smith_ggx_correlated(n_dot_l, n_dot_v, alpha2);

    let specular = d * vis * f;
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;

    return (diffuse + specular) * light_color * intensity * n_dot_l;
}

struct SceneOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec2<f32>,
};

@fragment
fn fs_main_scene(in: VertexOutputScene) -> SceneOut {
    var n = normalize(in.normal);

    // --- Normal mapping (tangent-space) ---
    let nm_sample = textureSample(normal_tex, normal_samp, in.uv).xyz * 2.0 - 1.0;
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
    let base_tex = textureSample(base_color_tex, base_color_samp, in.uv);
    let base_color = srgb_to_linear_v(base_tex.rgb) * srgb_to_linear_v(in.color.rgb);
    let base_alpha = base_tex.a * in.color.a;

    // glTF metallicRoughnessTexture: G=roughness, B=metallic (linear).
    let mr_tex_sample = textureSample(mr_tex, mr_samp, in.uv);
    var roughness = clamp(mr_tex_sample.g * material.metal_rough.y, 0.045, 1.0);
    let metallic  = clamp(mr_tex_sample.b * material.metal_rough.x, 0.0,   1.0);

    // Geometric specular antialiasing (Kaplanyan/Hofmann/Karis 2016).
    // High-frequency normal variation within a pixel produces aliasing
    // shimmer on smooth metals — especially on the helmet's tightly-
    // tessellated panels. Boost roughness based on the squared
    // length of dN/dx + dN/dy so under-sampled regions widen their
    // GGX lobe just enough to integrate over the missing detail.
    // The 0.25 kernel constant is the value Karis uses in UE4.
    let nm_dx = dpdx(n);
    let nm_dy = dpdy(n);
    let curvature_sq = dot(nm_dx, nm_dx) + dot(nm_dy, nm_dy);
    let kernel_alpha = min(0.25 * curvature_sq, 0.18);
    let alpha_base = roughness * roughness;
    roughness = sqrt(min(alpha_base + kernel_alpha, 1.0));

    let em_tex_sample = textureSample(em_tex, em_samp, in.uv);
    let emissive = srgb_to_linear_v(em_tex_sample.rgb) * material.emissive.rgb;

    // glTF occlusion: R channel, attenuates indirect lighting (IBL
    // diffuse + ambient) only — direct lights and specular IBL are
    // unchanged per spec. Default texture is white (idx 0) so the
    // sample is 1.0 for materials without an occlusion map.
    let occlusion = textureSample(occ_tex, occ_samp, in.uv).r;

    // --- PBR direct lighting ---
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);
    var lit = vec3<f32>(0.0);

    // Legacy primary directional (kept for back-compat). Shadow-
    // mapped: only this primary light casts because we currently
    // render a single shadow map. Multi-cascade or multi-light
    // shadowing is a future addition.
    let shadow_factor = sample_shadow(in.world_pos);
    let legacy_dir = normalize(lighting.light_dir.xyz);
    lit += shade_pbr(n, v, legacy_dir, lighting.light_color.rgb,
                     lighting.light_dir.w, base_color, metallic, roughness)
         * shadow_factor;

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

    // Diffuse irradiance: smallest mip is reserved for a proper
    // cosine-weighted irradiance convolution (run once at env load).
    // textureNumLevels gives us the actual mip count of the bound
    // env texture (1 for the default 1×1 gray, ≤7 for HDR loads).
    let mips = f32(textureNumLevels(env_tex));
    let irradiance = env_sample_lod(n, mips - 1.0);

    // For diffuse IBL, the Schlick-with-roughness approximation
    // (Lazarov 2013) handles the average kS factor at grazing angles.
    let fc_n = pow(1.0 - n_dot_v_ibl, 5.0);
    let f_ibl = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * fc_n;
    let kd = (vec3<f32>(1.0) - f_ibl) * (1.0 - metallic);
    let ibl_diffuse = irradiance * base_color * kd * occlusion;

    // Pre-filtered specular sample at mip = roughness * (mips - 2).
    // Cap one below the diffuse mip — that mip is the irradiance
    // map, no longer a GGX-prefiltered sample. With mips=7 the
    // specular range becomes 0..5 (mirror to roughest-still-specular).
    let r = reflect(-v, n);
    let max_spec_mip = max(mips - 2.0, 0.0);
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
    let f_avg = f0 + (vec3<f32>(1.0) - f0) * (1.0 / 21.0);
    let energy_compensation = vec3<f32>(1.0) + f_avg * (1.0 / max(brdf.x + brdf.y, 1e-4) - 1.0);
    let ibl_spec = single_spec * energy_compensation;

    // Multi-scatter also adds a diffuse-like term back from the
    // 'lost' energy, but it gets absorbed wherever there is no metal
    // since dielectrics already account for it via the (1 - kS)
    // diffuse term. The compensation above handles the metal case;
    // dielectric path is unchanged.
    let hdr = lit + ibl_diffuse + ibl_spec + emissive;

    // Output linear HDR + per-pixel material info (metallic /
    // roughness) so the SSR pass can modulate reflections by
    // surface response.
    return SceneOut(
        vec4<f32>(hdr, base_alpha),
        vec2<f32>(metallic, roughness),
    );
}
";

// ============================================================
// GGX prefilter shader (split-sum specular convolution)
// ============================================================
//
// One-shot pipeline: for each output mip of the env texture,
// convolve the source env with a GGX importance-sampling lobe at
// `roughness = mip / (mips-1)`. Karis 2013 simplification: assume
// V = N = R, which decouples each output texel from the view
// direction. The resulting prefiltered radiance is what the scene
// shader's split-sum sampling consumes via `env_sample_lod(R, lod)`.
//
// Sampled at HDR full radiance — this is where we'd add a brightness
// clamp if fireflies appear (high-luminance pixels with few samples).
// At 64 samples per mip we haven't seen them in the test HDRs.

const PREFILTER_SHADER_WGSL: &str = "
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
        color += textureSampleLevel(src_tex, src_samp, dir_to_uv(l), 0.0).rgb;
    }
    return vec4<f32>(color / f32(n_samples), 1.0);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefilterUniforms {
    /// x = roughness (∈ [0, 1]), y = sample count, zw = mip resolution
    params: [f32; 4],
}

// ============================================================
// Sky / equirectangular HDR background shader
// ============================================================
//
// Renders a fullscreen triangle with z=1 (far plane) and samples the
// environment map by the world-space view direction reconstructed from
// inverse VP. Tone-maps with the same ACES curve the rest of the
// renderer uses so the background blends seamlessly with lit
// geometry. Always overwrites depth — the 3D opaque pass drawn after
// will occlude wherever it has actual geometry.

const SKY_SHADER_WGSL: &str = "
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
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let phi = atan2(dir.z, dir.x);
    let raw_u = phi / (2.0 * PI);
    let u_coord = raw_u - floor(raw_u); // fract(); WGSL has no rem_euclid
    let v_coord = theta / PI;

    let radiance = textureSample(env_tex, env_samp, vec2<f32>(u_coord, v_coord)).rgb * u.intensity.x;
    // Output linear HDR radiance — the composite pass downstream does
    // the ACES tonemap + sRGB encode in one place. Sky writes to the
    // material G-buffer too: 0 metallic, 1 roughness — sky never
    // reflects, never gets reflected from (well, it gets sampled by
    // SSR via the HDR RT, but that's expected behavior).
    return SkyOut(vec4<f32>(radiance, 1.0), vec2<f32>(0.0, 1.0));
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    /// Camera right vector × tan(fovy/2) × aspect — pre-scaled so the
    /// fragment shader just multiplies by NDC.x to get the horizontal
    /// offset from the forward direction.
    right: [f32; 4],
    /// Camera up vector × tan(fovy/2).
    up: [f32; 4],
    /// Camera forward unit vector.
    forward: [f32; 4],
    /// x = intensity multiplier; yzw padding.
    intensity: [f32; 4],
}

// ============================================================
// Depth texture helper
// ============================================================

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Linear HDR format for the offscreen render target. The scene + sky
/// + immediate-mode 3D passes write here in linear space; a final
/// composite pass tonemaps to the sRGB surface format.
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Number of bloom mip levels. 5 mips gives a long-tail glow that
/// covers ~32× the source pixel size. More mips = more haloing,
/// fewer = less coverage. Each mip is half the previous size.
const BLOOM_MIP_COUNT: u32 = 5;

/// SSAO render target format. R8Unorm gives 256 occlusion levels
/// (plenty for AO) at 1 byte/pixel — half-res keeps the cost in
/// the noise on modern GPUs.
const SSAO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Material G-buffer format. Rg8Unorm: R = metallic, G = roughness.
/// Written as a second color attachment in the HDR pass; SSR (and
/// any future deferred passes) reads it for per-pixel material info.
const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // SSAO samples this texture in a separate pass after the
        // depth-write HDR pass — needs TEXTURE_BINDING in addition
        // to RENDER_ATTACHMENT.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_hdr_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the two ping-pong 1×1 exposure textures. Single fragment
/// writes to one, composite samples the other, swap each frame.
fn create_exposure_textures(device: &wgpu::Device) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let make = |label: &str| -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (a, av) = make("exposure_a");
    let (b, bv) = make("exposure_b");
    ([a, b], [av, bv])
}

/// Create the material G-buffer (Rg8Unorm, surface size).
fn create_material_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MATERIAL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the SSR render target (half-res HDR — reflections are
/// low-frequency enough that half-res hides bilinear blur).
fn create_ssr_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssr_rt"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Halton low-discrepancy sequence (base `b`, index `i`, 1-based).
/// Returns a value in [0, 1). Used to generate sub-pixel jitter
/// offsets that are well-distributed across the pixel — the TAA
/// accumulation effectively integrates over those sample points
/// to produce a stably anti-aliased image.
fn halton(mut i: u32, b: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

/// Create the two TAA history textures (HDR format, surface size).
fn create_taa_textures(device: &wgpu::Device, width: u32, height: u32) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let make = |label: &str| -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (a, av) = make("taa_a");
    let (b, bv) = make("taa_b");
    ([a, b], [av, bv])
}

/// Create the SSAO render target (single channel, half-res).
fn create_ssao_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao_rt"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SSAO_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the bloom mip-chain texture + per-mip render views + a
/// full-chain view for sampling. Mip 0 starts at surface/2 size and
/// each subsequent mip halves down to ~surface/2^N. Caller is
/// responsible for deciding N (usually BLOOM_MIP_COUNT). At least
/// 1×1 is enforced per mip.
/// Build the bloom chain as N separate single-mip textures rather
/// than one multi-mip texture. Multi-mip textures with one mip
/// bound as render target while another mip is sampled in the
/// same encoder trips wgpu/Metal's per-subresource state tracking
/// — symptoms include large black bars in the sampled output. N
/// separate textures sidestep the problem entirely (each pass's
/// read/write hits a distinct texture). `bloom_full_view` is a
/// view onto mip 0's texture, kept for backward compatibility.
fn create_bloom_chain(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    mip_count: u32,
) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>, wgpu::TextureView) {
    let mut textures = Vec::with_capacity(mip_count as usize);
    let mut views = Vec::with_capacity(mip_count as usize);
    for i in 0..mip_count {
        let w = ((width / 2) >> i).max(1);
        let h = ((height / 2) >> i).max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom_mip_tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        textures.push(tex);
        views.push(view);
    }
    let full_view = textures[0].create_view(&wgpu::TextureViewDescriptor::default());
    (textures, views, full_view)
}

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
const BLOOM_SHADER_WGSL: &str = "
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
        // Threshold defaults: bright = luminance > 1.0 (anything
        // above tonemap's display range), knee = 0.5 for a soft
        // falloff so emissive accents fade in instead of popping.
        let thr = 1.0;
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

/// SSAO fragment shader. Spiral-samples 16 nearby UVs, depth-
/// compares each against the center, accumulates occlusion with a
/// smooth range falloff so distant geometry doesn't haloed
/// silhouettes. Output is single-channel (1 = open, 0 = fully
/// occluded). Cheaper than view-space hemispheric SSAO since we
/// skip view-space reconstruction; halo artifacts on geometry
/// edges are mitigated by the range-falloff term.
const SSAO_SHADER_WGSL: &str = "
struct SsaoParams {
    /// xy = inv_size of source depth (1/width, 1/height) — used to
    /// keep sample radius pixel-coherent regardless of resolution.
    /// z = world-space radius scale (sample distance in UV units),
    /// w = strength (occlusion multiplier).
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsaoParams;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;

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

const N_SAMPLES: u32 = 16u;
const PI: f32 = 3.14159265;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let center_depth = textureSample(depth_tex, depth_samp, in.uv);

    // Skip the sky (depth == 1.0 at the far plane after sky pass).
    if (center_depth >= 0.9999) {
        return vec4<f32>(1.0);
    }

    // Per-pixel rotation jitter via interleaved-gradient noise
    // (Jorge Jimenez 2014) — breaks up the spiral pattern so the
    // residual structure looks like noise instead of bands.
    let coord = in.clip_pos.xy;
    let ign = fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y));
    let rot_offset = ign * 2.0 * PI;

    let radius = u.params.z;
    var occlusion = 0.0;
    let depth_bias = 0.0001;
    let max_delta = 0.01; // beyond this depth gap, sample is too distant to occlude

    for (var i = 0u; i < N_SAMPLES; i = i + 1u) {
        // Spiral with golden-angle rotation.
        let r_norm = sqrt((f32(i) + 0.5) / f32(N_SAMPLES));
        let theta = f32(i) * 2.39996323 + rot_offset;
        let offset = vec2<f32>(cos(theta), sin(theta)) * r_norm * radius;
        let sample_uv = in.uv + offset;
        let sample_depth = textureSample(depth_tex, depth_samp, sample_uv);

        // delta > 0 means sample is closer (occluder).
        let delta = center_depth - sample_depth - depth_bias;
        if (delta > 0.0 && delta < max_delta) {
            // Smooth falloff so silhouette edges don't get a sharp halo.
            let weight = smoothstep(max_delta, 0.0, delta);
            occlusion = occlusion + weight;
        }
    }
    let ao = 1.0 - (occlusion / f32(N_SAMPLES)) * u.params.w;
    return vec4<f32>(clamp(ao, 0.0, 1.0));
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoParams {
    /// xy = inv_size, z = radius (UV), w = strength
    params: [f32; 4],
}

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
const SSR_SHADER_WGSL: &str = "
struct SsrParams {
    /// Inverse of the projection matrix — depth → view-space pos.
    inv_proj: mat4x4<f32>,
    /// Projection matrix — view-space pos → clip-space.
    proj: mat4x4<f32>,
    /// x = SSR strength (0 = off, 1 = full)
    /// y = max march distance in view-space units
    /// z = number of march steps
    /// w = padding
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsrParams;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var hdr_tex: texture_2d<f32>;
@group(0) @binding(4) var hdr_samp: sampler;
@group(0) @binding(5) var mat_tex: texture_2d<f32>;
@group(0) @binding(6) var mat_samp: sampler;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, in.uv);
    if (depth >= 0.9999) {
        // Sky — no reflections.
        return vec4<f32>(0.0);
    }

    // Read per-pixel material info. Reflectivity is concentrated in
    // smooth metals: weight = metallic * (1 - roughness)². Skip
    // shading when the surface is too rough or non-metal to matter.
    let mat = textureSample(mat_tex, mat_samp, in.uv).rg;
    let metallic = mat.r;
    let roughness = mat.g;
    let reflectivity = metallic * (1.0 - roughness) * (1.0 - roughness);
    if (reflectivity < 0.02) {
        return vec4<f32>(0.0);
    }

    let view_pos = view_pos_from_depth(in.uv, depth);
    // Reconstruct view-space normal from screen-space derivatives.
    let dx = dpdx(view_pos);
    let dy = dpdy(view_pos);
    let n = normalize(cross(dx, dy));

    // V points from surface to camera (camera is at origin in view
    // space, so V = -view_pos / length).
    let v = normalize(-view_pos);
    // Reflection direction.
    let r = reflect(-v, n);

    // Skip rays heading toward the camera — those would walk back
    // into the surface and never find a real hit.
    if (r.z > 0.0) {
        return vec4<f32>(0.0);
    }

    let max_dist = u.params.y;
    let n_steps = u.params.z;
    let step_size = max_dist / n_steps;

    var hit_uv = vec2<f32>(-1.0);
    var hit_found = false;
    var t = step_size; // skip the first step to avoid self-intersection
    for (var i = 0u; i < u32(n_steps); i = i + 1u) {
        let ray_view = view_pos + r * t;
        let ray_clip = u.proj * vec4<f32>(ray_view, 1.0);
        let ray_ndc = ray_clip.xyz / ray_clip.w;
        // Off-screen ray — no hit possible.
        if (ray_ndc.x < -1.0 || ray_ndc.x > 1.0 ||
            ray_ndc.y < -1.0 || ray_ndc.y > 1.0 ||
            ray_ndc.z < 0.0 || ray_ndc.z > 1.0) {
            break;
        }
        let ray_uv = vec2<f32>(ray_ndc.x * 0.5 + 0.5, 1.0 - (ray_ndc.y * 0.5 + 0.5));
        let scene_depth = textureSample(depth_tex, depth_samp, ray_uv);
        // Ray has gone past the surface — record the hit.
        if (ray_ndc.z >= scene_depth) {
            hit_uv = ray_uv;
            hit_found = true;
            break;
        }
        t = t + step_size;
    }

    if (!hit_found) {
        return vec4<f32>(0.0);
    }

    // Edge fade — pull hits near the screen border to zero so the
    // reflection doesn't pop in/out as the camera moves. 0.1 of the
    // screen on each side gets feathered.
    let edge_fade = min(
        min(hit_uv.x, 1.0 - hit_uv.x),
        min(hit_uv.y, 1.0 - hit_uv.y),
    ) * 10.0;
    let fade = clamp(edge_fade, 0.0, 1.0);

    let reflected = textureSample(hdr_tex, hdr_samp, hit_uv).rgb;
    // Modulate by reflectivity (material-aware) AND user strength
    // AND screen-edge fade.
    return vec4<f32>(reflected * reflectivity * u.params.x * fade, fade);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrParams {
    inv_proj: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    /// x=strength, y=max_dist, z=n_steps, w=padding
    params: [f32; 4],
}

/// TAA shader. Combines current-frame HDR + SSAO + bloom into a
/// single linear-HDR value per fragment, then blends against the
/// reprojected history with a fixed feedback factor. For static
/// scenes (no motion vectors) the blend converges in ~10 frames to
/// a fully sub-pixel-resolved image. Neighborhood color clamp
/// reduces ghosting if the camera moves slowly.
const TAA_SHADER_WGSL: &str = "
struct TaaParams {
    /// x = blend factor for current-frame contribution (0 = pure
    /// history, 1 = pure current). 0.1 = 10% new, 90% history;
    /// converges in ~10 frames. First frame should pass 1.0 so the
    /// history initializes from the actual scene rather than zeros.
    /// y = bloom intensity, zw padding.
    params: vec4<f32>,
    /// Inverse of the current-frame view-projection matrix —
    /// reconstructs world-space position from depth + NDC.
    inv_vp: mat4x4<f32>,
    /// Previous-frame view-projection — projects the reconstructed
    /// world-space pos into the history-frame's clip space so we
    /// can sample the history texture at the right UV.
    prev_vp: mat4x4<f32>,
    /// Fog color (rgb) + density (w). Density 0 = fog disabled
    /// (exp(-0) = 1, fog factor = 0, mix stays at scene color).
    fog_color_density: vec4<f32>,
    /// Fog falloff parameters: x = height at which density is 1×
    /// (above this, density drops exponentially), y = falloff rate
    /// per world-space Y unit, zw padding.
    fog_params: vec4<f32>,
    /// Sun-shafts (god rays) state.
    /// xy = sun's screen-space UV (precomputed CPU-side from
    /// camera VP and sun direction; clipped to ±2 so the march
    /// can step beyond the screen edge cleanly)
    /// z  = strength (0 = effect off)
    /// w  = per-sample decay (0..1, controls shaft length)
    sun_shaft_uv_strength: vec4<f32>,
    /// xyz = sun shaft tint color, w = padding.
    sun_shaft_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TaaParams;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var hdr_samp: sampler;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var bloom_samp: sampler;
@group(0) @binding(5) var ssao_tex: texture_2d<f32>;
@group(0) @binding(6) var ssao_samp: sampler;
@group(0) @binding(7) var history_tex: texture_2d<f32>;
@group(0) @binding(8) var history_samp: sampler;
@group(0) @binding(9) var depth_tex: texture_depth_2d;
@group(0) @binding(10) var depth_samp: sampler;
@group(0) @binding(11) var ssr_tex: texture_2d<f32>;
@group(0) @binding(12) var ssr_samp: sampler;

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
    // Combine current-frame post-effects exactly as the composite
    // shader did before TAA was inserted. Result is pre-tonemap HDR.
    let hdr = textureSample(hdr_tex, hdr_samp, in.uv).rgb;
    let bloom = textureSample(bloom_tex, bloom_samp, in.uv).rgb;
    let ssao = textureSample(ssao_tex, ssao_samp, in.uv).r;
    let ssr = textureSample(ssr_tex, ssr_samp, in.uv).rgb;
    // SSR is added on top of HDR (pre-tonemap) — already strength-
    // and edge-faded by the SSR pass, so a flat add here is fine.
    let current = (hdr + ssr) * ssao + bloom * u.params.y;

    // Reproject history via depth + previous VP. Reconstruct the
    // fragment's world-space position from current-frame depth, then
    // project through the previous VP to get the prev-frame clip-
    // space position → prev_uv. Removes ghosting from camera motion
    // for all static geometry. (Animated meshes need per-vertex
    // motion vectors — not handled yet.)
    let depth = textureSample(depth_tex, depth_samp, in.uv);
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, depth, 1.0);
    let world_h = u.inv_vp * ndc;
    let world = world_h.xyz / world_h.w;
    let prev_clip = u.prev_vp * vec4<f32>(world, 1.0);
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));

    // If the prev_uv falls off-screen, fall back to current — no
    // history available. Otherwise sample the reprojected position.
    var history = current;
    if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
        history = textureSample(history_tex, history_samp, prev_uv).rgb;
    }

    // Neighborhood color clamp keeps the history from drifting too
    // far from what the current frame plausibly contains. Sample a
    // 3x3 neighborhood of the current frame to derive the clamp
    // range. Reduces flickering bright-pixel ghosts to within the
    // local minmax envelope.
    let texel = vec2<f32>(1.0 / f32(textureDimensions(hdr_tex).x),
                          1.0 / f32(textureDimensions(hdr_tex).y));
    var nmin = current;
    var nmax = current;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            if (x == 0 && y == 0) { continue; }
            let s_uv = in.uv + vec2<f32>(f32(x), f32(y)) * texel;
            let s_hdr = textureSample(hdr_tex, hdr_samp, s_uv).rgb;
            let s_ssao = textureSample(ssao_tex, ssao_samp, s_uv).r;
            let s_bloom = textureSample(bloom_tex, bloom_samp, s_uv).rgb;
            let s_ssr = textureSample(ssr_tex, ssr_samp, s_uv).rgb;
            let s = (s_hdr + s_ssr) * s_ssao + s_bloom * u.params.y;
            nmin = min(nmin, s);
            nmax = max(nmax, s);
        }
    }
    let clamped_history = clamp(history, nmin, nmax);

    let alpha = u.params.x;
    var blended = mix(clamped_history, current, alpha);

    // Height-based exponential fog. Density > 0 engages fog; 0 is
    // a no-op. Fog factor = 1 - exp(-density * distance * height_fade)
    // where height_fade = exp(-falloff * (world.y - height_ref))
    // so low-altitude fragments get the full density but anything
    // above `height_ref` fades out exponentially — typical 'ground
    // haze' look.
    let fog_density = u.fog_color_density.w;
    if (fog_density > 0.0) {
        let height_ref = u.fog_params.x;
        let height_falloff = u.fog_params.y;
        let cam_pos = vec3<f32>(
            u.inv_vp[3][0] / u.inv_vp[3][3],
            u.inv_vp[3][1] / u.inv_vp[3][3],
            u.inv_vp[3][2] / u.inv_vp[3][3],
        );
        let dist = length(world - cam_pos);
        let height_fade = exp(-height_falloff * max(world.y - height_ref, 0.0));
        let fog_factor = 1.0 - exp(-fog_density * dist * height_fade);
        blended = mix(blended, u.fog_color_density.rgb, clamp(fog_factor, 0.0, 1.0));
    }

    // Sun shafts (god rays). March a fixed number of taps from the
    // pixel toward the projected sun position, sampling depth at
    // each tap. Where depth ≥ ~1 (sky), the sun is visible — add
    // its tinted contribution (decayed by distance from the pixel)
    // back into the image. Defers the visibility test to the depth
    // buffer so any geometry occluding the sun naturally cuts the
    // shaft. Strength 0 = effect off; the early-out keeps the
    // 32-tap loop from running.
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
            // Skip taps that walk off-screen.
            if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0) {
                continue;
            }
            let d = textureSample(depth_tex, depth_samp, pos);
            // Sky pixels (d ≈ 1) contribute; surfaces (d < 1) shadow.
            // Soft step around 0.999 avoids hard banding.
            let sky = smoothstep(0.998, 1.0, d);
            accum = accum + sky * weight;
            weight = weight * decay;
        }
        // Normalize so a fully unobstructed shaft maps to 1.
        let norm = accum / f32(n_samples);
        blended = blended + u.sun_shaft_color.rgb * norm * shaft_strength;
    }

    return vec4<f32>(blended, 1.0);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaParams {
    /// x = blend factor (current-frame weight),
    /// y = bloom intensity, zw padding.
    params: [f32; 4],
    /// Inverse of the current frame's VP (reconstructs world pos
    /// from depth + NDC).
    inv_vp: [[f32; 4]; 4],
    /// Previous frame's VP (reprojects world pos into history UVs).
    prev_vp: [[f32; 4]; 4],
    /// Fog color (rgb) + density (w).
    fog_color_density: [f32; 4],
    /// Fog falloff: x = height reference, y = falloff rate,
    /// zw padding.
    fog_params: [f32; 4],
    /// Sun shaft state: xy = sun screen-space UV, z = strength,
    /// w = decay.
    sun_shaft_uv_strength: [f32; 4],
    /// Sun shaft tint (rgb) + padding.
    sun_shaft_color: [f32; 4],
}

/// Auto-exposure update shader. Runs at 1×1 viewport → single
/// fragment. Samples hdr_rt at a 4×4 grid (16 taps), averages
/// luminance, derives a target exposure via `key / avg_luma`,
/// smooths toward it from last frame's exposure. One fragment's
/// worth of work — way cheaper than having every composite
/// fragment redundantly do the same average.
const EXPOSURE_SHADER_WGSL: &str = "
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
    // 16-tap LOG-average luma. Linear average gets dominated by
    // a few very bright pixels — in an outdoor HDR scene with
    // the sun, the mean can land 10x higher than what the eye
    // perceives as average scene brightness. Log-average matches
    // human perception and gives far better auto-exposure on
    // high-dynamic-range content.
    var log_sum = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let sx = (f32(i % 4u) + 0.5) * 0.25;
        let sy = (f32(i / 4u) + 0.5) * 0.25;
        let s = textureSample(hdr_tex, hdr_samp, vec2<f32>(sx, sy)).rgb;
        let luma = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
        log_sum = log_sum + log(max(luma, 1e-5));
    }
    let avg_luma = exp(log_sum * (1.0 / 16.0));

    let key = u.params.x;
    let rate = u.params.y;
    let min_e = u.params.z;
    let max_e = u.params.w;

    let target_exp = clamp(key / max(avg_luma, 0.01), min_e, max_e);
    let prev = textureSample(prev_exposure_tex, prev_exposure_samp, vec2<f32>(0.5, 0.5)).r;
    // First frame the prev texture is cleared to 0 — in that case
    // the mix would converge very slowly from 0. Detect that and
    // snap to the target instead of blending from 0.
    var smoothed = mix(prev, target_exp, rate);
    if (prev < min_e * 0.5) {
        smoothed = target_exp;
    }
    return vec4<f32>(smoothed, 0.0, 0.0, 1.0);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ExposureParams {
    params: [f32; 4],
}

/// Composite + tonemap fragment shader. Single fullscreen triangle
/// reads hdr_rt and writes ACES-tonemapped linear-RGB. Hardware
/// performs the linear→sRGB encode on write because the surface
/// format is sRGB.
const COMPOSITE_SHADER_WGSL: &str = "
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
    /// x = grain seed (frame index, randomizes the noise per frame),
    /// yzw padding.
    misc: vec4<f32>,
};

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;
@group(0) @binding(2) var<uniform> u: CompositeParams;
@group(0) @binding(3) var exposure_tex: texture_2d<f32>;
@group(0) @binding(4) var exposure_samp: sampler;

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

// --- AgX tonemap (Troy Sobotka 2022) ---
// Better hue preservation than ACES in saturated regions — reds
// stay red instead of shifting toward orange, blues stay blue
// instead of shifting toward cyan. Same sigmoid shape overall,
// so the overall contrast is similar.

fn agx_contrast_approx(x: vec3<f32>) -> vec3<f32> {
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
    val = agx_contrast_approx(val);
    return clamp(val, vec3<f32>(0.0), vec3<f32>(1.0));
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

    // --- Chromatic aberration ---
    // Radial offset of the R/B channels away from the screen center
    // — gives a subtle 'cinema lens' fringe at the edges. Strength
    // is the worst-case UV offset at the corner.
    let ca_strength = u.filmic.x;
    var hdr_raw: vec3<f32>;
    if (ca_strength > 0.0) {
        let center = vec2<f32>(0.5, 0.5);
        let dir = sample_uv - center;
        let r = textureSample(hdr_tex, hdr_samp, sample_uv + dir * ca_strength).r;
        let g = textureSample(hdr_tex, hdr_samp, sample_uv).g;
        let b = textureSample(hdr_tex, hdr_samp, sample_uv - dir * ca_strength).b;
        hdr_raw = vec3<f32>(r, g, b);
    } else {
        hdr_raw = textureSample(hdr_tex, hdr_samp, sample_uv).rgb;
    }

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
    let hdr = hdr_raw * exposure;

    // Branch between ACES and AgX via the uniform. Costs one
    // compare per fragment; the dead branch gets DCE'd per-draw
    // since the uniform is constant across the frame.
    var ldr: vec3<f32>;
    if (u.params.x < 0.5) {
        ldr = aces_tone(hdr);
    } else {
        ldr = agx_eotf(agx_tone(hdr));
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomParams {
    /// xy = source texel size, z = filter radius (upsample),
    /// w = HDR threshold (downsample-threshold variant).
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    /// x = tonemap kind (0 ACES / 1 AgX), y = auto-exposure toggle,
    /// z = manual exposure, w = auto-exposure target key.
    params: [f32; 4],
    /// Filmic-look knobs — see WGSL comment.
    /// x = chromatic-aberration strength, y = vignette strength,
    /// z = vignette softness, w = grain strength.
    filmic: [f32; 4],
    /// x = grain seed (frame index, animates the noise), yzw padding.
    misc: [f32; 4],
}

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

// ============================================================
// Cached model GPU data
// ============================================================

struct GpuMesh {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    index_count: u32,
    texture_idx: u32,
    /// Pre-built scene material bind group (base color + normal +
    /// metallic-roughness + emissive + material factors). Cached at
    /// model-upload time so draw_model_cached doesn't build one per
    /// frame.
    material_bg: wgpu::BindGroup,
    /// Per-material uniform buffer backing the `material` binding in
    /// the scene pipeline's group 2. Kept alongside `material_bg` so
    /// its lifetime matches (bind groups reference buffers internally
    /// via Arc, but we also want the strong ref for future updates).
    _material_uniform: wgpu::Buffer,
}

struct CachedModelDraw {
    uniform_slot: usize,
    cache_handle: u64,
    mesh_idx: usize,
}

// ============================================================
// Renderer
// ============================================================

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,

    // Pipelines
    pipeline_2d: wgpu::RenderPipeline,
    pipeline_3d: wgpu::RenderPipeline,
    custom_pipelines: Vec<wgpu::RenderPipeline>,

    // 2D uniforms (multiple slots for mode switching)
    uniform_buffers: Vec<wgpu::Buffer>,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    current_uniform_idx: u32,
    uniform_slot_count: usize,

    // 3D uniforms
    uniform_buffer_3d: wgpu::Buffer,
    uniform_bind_group_3d: wgpu::BindGroup,

    // Lighting uniforms
    lighting_uniforms: LightingUniforms,
    lighting_buffer: wgpu::Buffer,
    lighting_bind_group: wgpu::BindGroup,

    // Joint matrices for GPU skinning (64 joints × 4x4 matrix)
    joint_buffer: wgpu::Buffer,
    joint_bind_group: wgpu::BindGroup,

    // Texture management
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_groups: Vec<wgpu::BindGroup>,
    textures: Vec<wgpu::Texture>,
    texture_sizes: Vec<(u32, u32)>,
    pub sampler: wgpu::Sampler,
    pub nearest_sampler: wgpu::Sampler,

    // Depth buffer
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Linear HDR offscreen render target the scene + sky + 3D
    /// pipelines write into. A composite-tonemap pass reads it and
    /// writes the final image to the sRGB surface. Sized to surface
    /// dimensions; recreated in `resize`.
    pub hdr_rt_texture: wgpu::Texture,
    pub hdr_rt_view: wgpu::TextureView,
    /// Material G-buffer (Rg8Unorm: R=metallic, G=roughness).
    /// Second color attachment in the HDR pass. SSR reads this so
    /// only smooth metallic surfaces reflect — rough or non-metal
    /// surfaces fade to zero.
    pub material_rt_texture: wgpu::Texture,
    pub material_rt_view: wgpu::TextureView,
    /// Composite-tonemap pipeline + bind group layout. Single full-
    /// screen draw that samples hdr_rt and writes ACES-tonemapped
    /// linear-rgb (sRGB hardware encode handles the transfer fn).
    /// Now also samples bloom_chain[0] and additively merges before
    /// the tonemap.
    pub composite_pipeline: wgpu::RenderPipeline,
    pub composite_layout: wgpu::BindGroupLayout,
    pub composite_sampler: wgpu::Sampler,
    /// 0 = ACES (default, matches bloom-reference), 1 = AgX.
    pub tonemap_kind: u32,
    /// Auto-exposure on/off. Default off so validation against
    /// the path-traced reference (fixed exposure) stays meaningful.
    pub auto_exposure: bool,
    /// Manual exposure multiplier (used when auto_exposure is off).
    /// Default 1.0 = no change.
    pub manual_exposure: f32,
    /// Auto-exposure target key value (scene-average luma target).
    /// 0.18 = photography 18%-gray standard.
    pub auto_exposure_key: f32,
    /// Auto-exposure smoothing rate per frame (0 = no adapt, 0.05
    /// = ~20-frame half-life at 60fps, 1 = instant). Only used
    /// when auto_exposure is on.
    pub auto_exposure_rate: f32,
    /// Filmic-look composite knobs. All default to 0 (effect off)
    /// so validation parity with the path-traced reference stays
    /// bit-meaningful.
    pub chromatic_aberration: f32,
    pub vignette_strength: f32,
    pub vignette_softness: f32,
    pub grain_strength: f32,
    /// Ping-pong 1×1 R16Float textures holding the smoothed
    /// exposure value. Composite reads the "current" slot; the
    /// exposure update pass reads "prev" and writes to "current".
    pub exposure_textures: [wgpu::Texture; 2],
    pub exposure_views: [wgpu::TextureView; 2],
    pub exposure_current_idx: usize,
    pub exposure_pipeline: wgpu::RenderPipeline,
    pub exposure_layout: wgpu::BindGroupLayout,
    pub exposure_uniform_buffer: wgpu::Buffer,
    /// Bloom mip-chain texture. Single texture with BLOOM_MIP_COUNT
    /// mips starting at surface/2 size — each mip is half the
    /// previous. Downsample chain (with HDR threshold on first tap)
    /// fills it, upsample chain blends back up. Composite shader
    /// reads mip 0 and adds it to the HDR sample before tonemap.
    /// One distinct texture per bloom mip — see create_bloom_chain
    /// for why this isn't a single multi-mip texture.
    pub bloom_chain_textures: Vec<wgpu::Texture>,
    pub bloom_mip_views: Vec<wgpu::TextureView>,
    pub bloom_full_view: wgpu::TextureView,
    pub bloom_pipeline_threshold_downsample: wgpu::RenderPipeline,
    pub bloom_pipeline_downsample: wgpu::RenderPipeline,
    pub bloom_pipeline_upsample: wgpu::RenderPipeline,
    pub bloom_layout: wgpu::BindGroupLayout,
    pub bloom_uniform_buffer: wgpu::Buffer,
    /// Composite-shader uniform — bloom intensity etc. Written each
    /// frame from the renderer's `bloom_intensity` field.
    pub composite_uniform_buffer: wgpu::Buffer,
    pub bloom_intensity: f32,
    /// SSAO RT (R8Unorm, half-res) + pipeline + uniforms. Run after
    /// the HDR pass; sampled by the composite to darken crevices.
    pub ssao_rt_texture: wgpu::Texture,
    pub ssao_rt_view: wgpu::TextureView,
    pub ssao_pipeline: wgpu::RenderPipeline,
    pub ssao_layout: wgpu::BindGroupLayout,
    pub ssao_uniform_buffer: wgpu::Buffer,
    pub ssao_depth_sampler: wgpu::Sampler,
    /// Strength multiplier for SSAO (0 = off, 1 = full). Default 1.0.
    pub ssao_strength: f32,
    /// Sample radius in UV units (default ~0.005, gives a soft AO
    /// signal a few pixels wide on a 1024-tall surface).
    pub ssao_radius: f32,
    /// TAA history ping-pong. Two HDR-format textures the same size
    /// as the surface — each frame writes to one, reads the other as
    /// history. `taa_current_idx` flips after every frame.
    pub taa_textures: [wgpu::Texture; 2],
    pub taa_views: [wgpu::TextureView; 2],
    pub taa_current_idx: usize,
    pub taa_pipeline: wgpu::RenderPipeline,
    pub taa_layout: wgpu::BindGroupLayout,
    pub taa_uniform_buffer: wgpu::Buffer,
    /// Frame counter used to pick a different Halton offset every
    /// frame for sub-pixel camera jitter — accumulating over the
    /// jitter sequence is what gives TAA its anti-aliasing.
    pub taa_frame_index: u32,
    /// 0 = TAA off (composite reads hdr directly, history skipped).
    /// 1 = TAA on (default). When off the renderer behaves exactly
    /// as the pre-TAA pipeline did.
    pub taa_enabled: bool,
    /// Previous frame's view-projection matrix — TAA reads this to
    /// reproject the history texture into current-frame UV space,
    /// removing ghosting under camera motion. Updated at the end
    /// of each frame from current_vp_matrix.
    pub prev_vp_matrix: [[f32; 4]; 4],
    /// Fog color (rgb) — blended into scene where fog factor > 0.
    pub fog_color: [f32; 3],
    /// Fog density. 0 = disabled (default). Positive values engage
    /// exponential fog: fog_factor = 1 - exp(-density * distance).
    pub fog_density: f32,
    /// Height above which fog density starts to fall off.
    pub fog_height_ref: f32,
    /// Fog falloff rate in world-space Y units — how quickly fog
    /// thins out with altitude above `fog_height_ref`.
    pub fog_height_falloff: f32,
    /// Sun shaft (god rays) strength — additive contribution
    /// where the depth buffer says the sun is visible. 0 = off
    /// (default — keeps validation parity).
    pub sun_shaft_strength: f32,
    /// Per-sample decay for the sun shaft march. 0.95–0.99 = long
    /// shafts, 0.85 = short, 0.5 = barely visible.
    pub sun_shaft_decay: f32,
    /// Sun shaft tint (rgb 0..1).
    pub sun_shaft_color: [f32; 3],
    /// SSR (screen-space reflections) pass output — half-res HDR
    /// holding the reflected color for each fragment. Composited
    /// into the final image by the TAA pass.
    pub ssr_rt_texture: wgpu::Texture,
    pub ssr_rt_view: wgpu::TextureView,
    pub ssr_pipeline: wgpu::RenderPipeline,
    pub ssr_layout: wgpu::BindGroupLayout,
    pub ssr_uniform_buffer: wgpu::Buffer,
    /// SSR strength multiplier (0 = off, 1 = full). Default 0.5
    /// is conservative — too much SSR makes diffuse surfaces look
    /// like wet floors. Applies on top of the prefiltered IBL.
    pub ssr_strength: f32,
    pub ssr_enabled: bool,

    // Per-frame 2D batch
    vertices_2d: Vec<Vertex2D>,
    indices_2d: Vec<u32>,
    draw_calls_2d: Vec<DrawCall2D>,

    // Per-frame 3D batch
    pub vertices_3d: Vec<Vertex3D>,
    pub indices_3d: Vec<u32>,
    draw_calls_3d: Vec<DrawCall3D>,
    current_texture_3d: u32,

    // Persistent GPU buffers (reused across frames, grown as needed)
    persistent_vb_2d: wgpu::Buffer,
    persistent_ib_2d: wgpu::Buffer,
    persistent_vb_3d: wgpu::Buffer,
    persistent_ib_3d: wgpu::Buffer,
    persistent_vb_2d_capacity: usize, // in bytes
    persistent_ib_2d_capacity: usize,
    persistent_vb_3d_capacity: usize,
    persistent_ib_3d_capacity: usize,

    // Cached model GPU buffers (static models only)
    model_gpu_cache: HashMap<u64, Option<Vec<GpuMesh>>>,
    model_draw_commands: Vec<CachedModelDraw>,
    model_uniform_buffers: Vec<wgpu::Buffer>,
    model_uniform_bind_groups: Vec<wgpu::BindGroup>,
    next_model_uniform_slot: usize,
    current_vp_matrix: [[f32; 4]; 4],
    current_view_matrix: [[f32; 4]; 4],
    current_proj_matrix: [[f32; 4]; 4],
    current_camera_pos: [f32; 3],
    uniform_3d_layout: wgpu::BindGroupLayout,

    // State
    pub render_mode: RenderMode,
    clear_color: wgpu::Color,
    debug_frame: u64,
    // Pending joint matrices (written to GPU in end_frame)
    pub pending_joint_matrices: Option<Vec<[[f32; 4]; 4]>>,
    pub model_skin_scale: f32,

    // Shadow mapping
    pub shadow_map: crate::shadows::ShadowMap,

    // Screenshot capture (set flag, captured during end_frame)
    pub screenshot_requested: bool,
    pub screenshot_data: Option<(u32, u32, Vec<u8>)>,
    /// When set, the next end_frame_with_scene captures the framebuffer
    /// and writes it directly to this path as a PNG before clearing.
    /// Used by `bloom_take_screenshot()` so TS code (and CI / diff
    /// tooling) can grab a frame without going through geisterhand.
    pub pending_screenshot_path: Option<String>,

    // Q1: Render-to-texture override. When set, end_frame renders to this
    // texture view instead of the surface. Set by begin_texture_mode,
    // cleared by end_texture_mode.
    pub rt_color_view: Option<wgpu::TextureView>,
    pub rt_depth_view: Option<wgpu::TextureView>,
    pub rt_depth_texture: Option<wgpu::Texture>,
    pub rt_width: u32,
    pub rt_height: u32,

    // Equirectangular HDR environment background. When a sky texture
    // is loaded, a full-screen pass samples it per-pixel by view
    // direction so the background matches a path-traced reference
    // (instead of a flat clear color). Populated via `load_env_from_hdr`.
    sky_texture: Option<wgpu::Texture>,
    sky_bind_group: Option<wgpu::BindGroup>,
    sky_uniform_buffer: wgpu::Buffer,
    sky_pipeline: wgpu::RenderPipeline,
    sky_bind_group_layout: wgpu::BindGroupLayout,
    sky_sampler: wgpu::Sampler,

    // Scene pipeline (retained scene graph rendering with normal
    // mapping). Distinct from pipeline_3d so immediate-mode draws
    // don't have to carry tangent vertex data or normal-map bindings.
    pub scene_pipeline: wgpu::RenderPipeline,
    pub scene_material_layout: wgpu::BindGroupLayout,
    /// 1×1 gray env fallback and its sampler — bound in the lighting
    /// bind group before any HDR is loaded. `load_env_from_hdr`
    /// rebuilds the lighting bind group to swap in the real env
    /// texture. Kept around so we can rebuild back to the default
    /// if env is ever cleared.
    _scene_env_default_texture: wgpu::Texture,
    pub scene_env_default_view: wgpu::TextureView,
    pub env_sampler: wgpu::Sampler,
    pub lighting_layout: wgpu::BindGroupLayout,
    /// Pre-computed split-sum BRDF LUT — 256x256 Rg16Float texture
    /// where (u, v) = (NdotV, roughness) and (r, g) = (scale, bias)
    /// for the GGX BRDF integral. Generated once on CPU in
    /// `Renderer::new` and never touched after.
    _brdf_lut_texture: wgpu::Texture,
    pub brdf_lut_view: wgpu::TextureView,
    pub brdf_lut_sampler: wgpu::Sampler,
    /// GGX prefilter pipeline. Run once per env load to convolve the
    /// HDR env into roughness-weighted mips, replacing the box filter
    /// stand-in. Matches Karis 2013's split-sum specular prefilter.
    pub prefilter_pipeline: wgpu::RenderPipeline,
    /// Diffuse irradiance prefilter pipeline (cosine-weighted env
    /// convolution). Run on the smallest mip so the scene shader's
    /// diffuse IBL sample is properly Lambertian, not GGX-with-rough.
    pub prefilter_diffuse_pipeline: wgpu::RenderPipeline,
    pub prefilter_layout: wgpu::BindGroupLayout,
    pub prefilter_uniform_buffer: wgpu::Buffer,
    /// Default flat-normal (tangent-space +Z) 1x1 texture view — used
    /// when a mesh has tangents but no normal map so the TBN sampling
    /// becomes a no-op (returns the geometric normal).
    ///
    /// Kept in its own field rather than pushed into `self.textures`
    /// so it does not offset the indices returned by
    /// `register_texture`. If it lived in `self.textures`, scene
    /// material bind groups would look up the wrong view — base color
    /// textures would silently point to this flat-blue normal map.
    _default_normal_texture: wgpu::Texture,
    pub default_normal_view: wgpu::TextureView,
}

impl Renderer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        surface_config: wgpu::SurfaceConfiguration,
    ) -> Self {
        // --- Shaders ---
        let shader_2d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_2d"),
            source: wgpu::ShaderSource::Wgsl(SHADER_2D.into()),
        });
        let shader_3d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_3d"),
            source: wgpu::ShaderSource::Wgsl(SHADER_3D.into()),
        });

        // --- Uniform bind group layouts ---
        let uniform_2d_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_2d_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_3d_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_3d_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // --- Pre-allocate uniform buffers ---
        let initial_uniforms = Uniforms2D {
            screen_size: [surface_config.width as f32, surface_config.height as f32],
            _pad: [0.0; 2],
            view_proj: IDENTITY_MAT4,
        };

        let mut uniform_buffers = Vec::with_capacity(MAX_UNIFORM_SLOTS);
        let mut uniform_bind_groups = Vec::with_capacity(MAX_UNIFORM_SLOTS);
        for i in 0..MAX_UNIFORM_SLOTS {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("uniform_2d_{}", i)),
                contents: bytemuck::bytes_of(&initial_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("uniform_2d_bg_{}", i)),
                layout: &uniform_2d_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            uniform_buffers.push(buf);
            uniform_bind_groups.push(bg);
        }

        // --- 3D uniform buffer ---
        let uniform_buffer_3d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_3d"),
            contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4, model_tint: [1.0, 1.0, 1.0, 1.0] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_3d = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_3d_bg"),
            layout: &uniform_3d_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer_3d.as_entire_binding(),
            }],
        });

        // --- Lighting uniform buffer ---
        // Lighting layout carries the lighting UBO + the equirect
        // environment map (mip-chained for split-sum specular) + the
        // pre-computed BRDF LUT used by the scene shader for IBL.
        // Bundling all per-frame globals here keeps us within the
        // default max_bind_groups = 4 (so we don't have to request a
        // higher device limit). pipeline_3d doesn't reference the env
        // / BRDF bindings — WGSL lets bind group layouts expose more
        // than a shader consumes.
        let lighting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let lighting_uniforms = LightingUniforms::defaults();
        let lighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lighting_buffer"),
            contents: bytemuck::bytes_of(&lighting_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Sampler ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Nearest-neighbor sampler (for pixel art) ---
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Env IBL sampler — reused by both the sky pass and the scene
        // pipeline. Clamps V to avoid pole wrap artifacts; U repeats
        // because equirect wraps horizontally. Linear mipmap filter
        // so the scene shader's roughness-driven mip lookup
        // (textureSampleLevel with a fractional level) blends between
        // mip levels smoothly — that's what gives us the prefiltered-
        // specular split-sum approximation.
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 1×1 mid-gray default env so the lighting bind group is
        // valid before any HDR is loaded. Gray (not black) gives
        // roughly neutral IBL ambient so PBR geometry is visible.
        let env_default_data_u16: [u16; 4] = [
            half::f16::from_f32(0.5).to_bits(),
            half::f16::from_f32(0.5).to_bits(),
            half::f16::from_f32(0.5).to_bits(),
            half::f16::from_f32(1.0).to_bits(),
        ];
        let scene_env_default_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_env_default_texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &scene_env_default_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&env_default_data_u16),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let scene_env_default_view = scene_env_default_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- BRDF LUT (split-sum integration) ---
        // 256x256 Rg16Float texture. f(NdotV, roughness) → (scale, bias)
        // such that final_specular = env_sample * (F0 * scale + bias).
        // Generated with importance-sampled GGX (Hammersley sequence)
        // matching Karis 2013 ('Real Shading in UE4'). 1024 samples
        // per cell × 65536 cells ≈ 67M ops — runs in well under a
        // second on a modern CPU.
        let brdf_lut_size: u32 = 256;
        let brdf_lut_pixels = build_brdf_lut(brdf_lut_size as usize);
        let brdf_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf_lut"),
            size: wgpu::Extent3d {
                width: brdf_lut_size,
                height: brdf_lut_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &brdf_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&brdf_lut_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(brdf_lut_size * 4), // 2 channels × 2 bytes
                rows_per_image: Some(brdf_lut_size),
            },
            wgpu::Extent3d {
                width: brdf_lut_size,
                height: brdf_lut_size,
                depth_or_array_layers: 1,
            },
        );
        let brdf_lut_view = brdf_lut_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // BRDF LUT sampler: linear filter, clamp-to-edge. The LUT is
        // already pre-integrated at 256×256 — no mip filtering needed.
        let brdf_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brdf_lut_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Shadow map needs to be created before the lighting bind
        // group since the bind group binds the shadow depth view.
        let shadow_map = crate::shadows::ShadowMap::new(&device, Vertex3D::desc());

        let lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_bg"),
            layout: &lighting_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lighting_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&scene_env_default_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&env_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&brdf_lut_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&shadow_map.depth_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&shadow_map.sampler) },
            ],
        });

        // --- Default 1x1 white texture ---
        let white_data = [255u8, 255, 255, 255];
        let white_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("white_texture"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &white_data,
        );
        let white_view = white_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let white_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("white_texture_bg"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&white_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        let mut texture_bind_groups = Vec::new();
        let mut textures = Vec::new();
        let mut texture_sizes = Vec::new();
        texture_bind_groups.push(white_bind_group);
        textures.push(white_texture);
        texture_sizes.push((1, 1));

        // --- Depth texture ---
        let (depth_texture, depth_view) = create_depth_texture(&device, surface_config.width, surface_config.height);
        let (hdr_rt_texture, hdr_rt_view) = create_hdr_rt(&device, surface_config.width, surface_config.height);
        let (material_rt_texture, material_rt_view) = create_material_rt(&device, surface_config.width, surface_config.height);
        let (bloom_chain_textures, bloom_mip_views, bloom_full_view) = create_bloom_chain(
            &device,
            surface_config.width,
            surface_config.height,
            BLOOM_MIP_COUNT,
        );
        let (ssao_rt_texture, ssao_rt_view) = create_ssao_rt(
            &device, surface_config.width, surface_config.height,
        );
        let (taa_textures, taa_views) = create_taa_textures(
            &device, surface_config.width, surface_config.height,
        );
        let (ssr_rt_texture, ssr_rt_view) = create_ssr_rt(
            &device, surface_config.width, surface_config.height,
        );
        let (exposure_textures, exposure_views) = create_exposure_textures(&device);

        // --- Persistent GPU buffers (reused across frames) ---
        let vb_3d_cap = 1024 * 1024; // 1MB ~= 10,900 Vertex3D
        let ib_3d_cap = 512 * 1024;  // 512KB
        let vb_2d_cap = 256 * 1024;  // 256KB
        let ib_2d_cap = 128 * 1024;  // 128KB

        let persistent_vb_3d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("persistent_vb_3d"),
            size: vb_3d_cap as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let persistent_ib_3d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("persistent_ib_3d"),
            size: ib_3d_cap as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let persistent_vb_2d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("persistent_vb_2d"),
            size: vb_2d_cap as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let persistent_ib_2d = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("persistent_ib_2d"),
            size: ib_2d_cap as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- 2D Pipeline ---
        let pipeline_layout_2d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout_2d"),
            bind_group_layouts: &[&uniform_2d_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_2d = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline_2d"),
            layout: Some(&pipeline_layout_2d),
            vertex: wgpu::VertexState {
                module: &shader_2d,
                entry_point: Some("vs_main"),
                buffers: &[Vertex2D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_2d,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            // No depth-stencil — 2D never tests or writes depth, and
            // the pipeline runs in two different passes: one with a
            // depth attachment (composited into hdr_rt) and one
            // without (drawn on top of the tonemapped surface).
            // wgpu allows a depth-less pipeline in either pass; the
            // reverse — a depth-bound pipeline in a depth-less pass
            // — is a validation error.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Joint matrix buffer for GPU skinning ---
        let joint_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("joint_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        // 64 joints × 64 bytes per mat4 = 4096 bytes
        let joint_data = vec![0u8; 8192];
        let joint_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("joint_buffer"),
            contents: &joint_data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let joint_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("joint_bg"),
            layout: &joint_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: joint_buffer.as_entire_binding(),
            }],
        });
        // Initialize with identity matrices
        {
            let mut identity_data = vec![0u8; 8192];
            for i in 0..128 {
                let offset = i * 64;
                // Identity matrix in column-major: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
                let one = 1.0f32.to_le_bytes();
                identity_data[offset..offset+4].copy_from_slice(&one);       // [0][0]
                identity_data[offset+20..offset+24].copy_from_slice(&one);   // [1][1]
                identity_data[offset+40..offset+44].copy_from_slice(&one);   // [2][2]
                identity_data[offset+60..offset+64].copy_from_slice(&one);   // [3][3]
            }
            queue.write_buffer(&joint_buffer, 0, &identity_data);
        }

        // --- 3D Pipeline ---
        let pipeline_layout_3d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout_3d"),
            bind_group_layouts: &[&uniform_3d_layout, &lighting_layout, &texture_bind_group_layout, &joint_layout],
            push_constant_ranges: &[],
        });

        let pipeline_3d = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline_3d"),
            layout: Some(&pipeline_layout_3d),
            vertex: wgpu::VertexState {
                module: &shader_3d,
                entry_point: Some("vs_main_3d"),
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_3d,
                entry_point: Some("fs_main_3d"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: MATERIAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Pre-allocate model uniform buffer pool (64 slots for cached model draws) ---
        let model_uniform_count = 64;
        let mut model_uniform_buffers = Vec::with_capacity(model_uniform_count);
        let mut model_uniform_bind_groups = Vec::with_capacity(model_uniform_count);
        for _ in 0..model_uniform_count {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("model_uniform"),
                contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4, model_tint: [1.0; 4] }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("model_uniform_bg"),
                layout: &uniform_3d_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            model_uniform_buffers.push(buf);
            model_uniform_bind_groups.push(bg);
        }

        // (shadow_map already created above before lighting bind group.)

        // Sky / equirectangular HDR environment background.
        // Compiled at startup so the pipeline is ready when the user
        // first calls bloom_set_env_map(); the texture itself is set
        // lazily on first env load.
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: wgpu::ShaderSource::Wgsl(SKY_SHADER_WGSL.into()),
        });
        let sky_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_uniform_buffer"),
            size: std::mem::size_of::<SkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let sky_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_pl"),
            bind_group_layouts: &[&sky_bind_group_layout],
            push_constant_ranges: &[],
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("sky_vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("sky_fs"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: MATERIAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            // Depth: write z=1.0 (far plane). Use Always so the sky
            // pass never gets occluded by stale depth from a previous
            // frame; the 3D opaque pass will overwrite where it draws.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ============================================================
        // Scene pipeline (retained scene-graph draws with normal maps)
        // ============================================================
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_shader"),
            source: wgpu::ShaderSource::Wgsl(SCENE_SHADER.into()),
        });
        // Scene material layout:
        //   0: base_color texture      4: metallic_roughness texture
        //   1: base_color sampler      5: metallic_roughness sampler
        //   2: normal     texture      6: emissive             texture
        //   3: normal     sampler      7: emissive             sampler
        //   8: material factors uniform (metallic/roughness/emissive)
        //   9: occlusion  texture     10: occlusion           sampler
        let tex_entry = |b| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp_entry = |b| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let scene_material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_material_layout"),
            entries: &[
                tex_entry(0),  samp_entry(1),
                tex_entry(2),  samp_entry(3),
                tex_entry(4),  samp_entry(5),
                tex_entry(6),  samp_entry(7),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(9),  samp_entry(10),
            ],
        });
        // Env IBL binding is folded into the lighting bind group
        // above (bindings 1 and 2 of group 1). That keeps the scene
        // pipeline under the default max_bind_groups = 4 limit, so we
        // don't need a separate env group here.

        // --- GGX prefilter pipeline (run on env load) ---
        let prefilter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefilter_shader"),
            source: wgpu::ShaderSource::Wgsl(PREFILTER_SHADER_WGSL.into()),
        });
        let prefilter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefilter_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let prefilter_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefilter_pipeline_layout"),
            bind_group_layouts: &[&prefilter_layout],
            push_constant_ranges: &[],
        });
        let prefilter_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prefilter_pipeline"),
            layout: Some(&prefilter_pl_layout),
            vertex: wgpu::VertexState {
                module: &prefilter_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &prefilter_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let prefilter_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefilter_uniform_buffer"),
            size: std::mem::size_of::<PrefilterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Diffuse irradiance prefilter pipeline — same vertex stage,
        // cosine-weighted convolution in the fragment stage. Reused
        // bind group layout (so we don't need to rebuild bind groups
        // when switching pipelines mid-encoder).
        let prefilter_diffuse_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prefilter_diffuse_pipeline"),
            layout: Some(&prefilter_pl_layout),
            vertex: wgpu::VertexState {
                module: &prefilter_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &prefilter_shader,
                entry_point: Some("fs_diffuse"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let scene_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_pipeline_layout"),
            bind_group_layouts: &[&uniform_3d_layout, &lighting_layout, &scene_material_layout, &joint_layout],
            push_constant_ranges: &[],
        });
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_pipeline"),
            layout: Some(&scene_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main_scene"),
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_main_scene"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: MATERIAL_FORMAT,
                        // Replace blend so the material slot reflects
                        // the topmost-fragment material, not blended.
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Default flat-normal 1×1 texture for meshes that have tangents
        // but no normal map. Encodes (0, 0, 1) in tangent space:
        //   RGB = (0.5, 0.5, 1.0) * 255 = (128, 128, 255)
        // After the shader's `sampled * 2 - 1` decode, this gives the
        // unperturbed geometric normal.
        let default_normal_data = [128u8, 128, 255, 255];
        let default_normal_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("default_normal_texture"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &default_normal_data,
        );
        let default_normal_view = default_normal_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Keep the texture owned via a dedicated field — NOT pushed
        // into `textures`, because that would offset the indices
        // returned by `register_texture` (callers store those as
        // material.texture_idx etc.) by one. A prior version did push
        // here and caused all base-color lookups to silently hit this
        // flat-blue normal map instead.

        // --- Composite-tonemap pipeline ---
        // Single fullscreen draw that samples the HDR RT and writes
        // ACES-tonemapped linear RGB into the sRGB surface.
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER_WGSL.into()),
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let composite_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pl_layout"),
            bind_group_layouts: &[&composite_layout],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite_pipeline"),
            layout: Some(&composite_pl_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite_uniform_buffer"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bloom mip-chain pipelines ---
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_SHADER_WGSL.into()),
        });
        let bloom_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bloom_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_pl_layout"),
            bind_group_layouts: &[&bloom_layout],
            push_constant_ranges: &[],
        });
        let bloom_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_uniform_buffer"),
            size: std::mem::size_of::<BloomParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let make_bloom_pipeline = |entry: &str, blend: Option<wgpu::BlendState>| -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bloom_pipeline"),
                layout: Some(&bloom_pl_layout),
                vertex: wgpu::VertexState {
                    module: &bloom_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bloom_shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let bloom_pipeline_threshold_downsample = make_bloom_pipeline("fs_threshold_downsample", None);
        let bloom_pipeline_downsample = make_bloom_pipeline("fs_downsample", None);
        // Upsample blends additively into the destination mip so each
        // pass progressively builds up the final bloom.
        let upsample_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let bloom_pipeline_upsample = make_bloom_pipeline("fs_upsample", Some(upsample_blend));

        // --- SSAO pipeline ---
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_SHADER_WGSL.into()),
        });
        let ssao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Depth32Float texture — sampled as
                        // texture_depth_2d in the shader.
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // Non-comparison sampler — ordinary linear
                    // sample of the depth texture.
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let ssao_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_pl_layout"),
            bind_group_layouts: &[&ssao_layout],
            push_constant_ranges: &[],
        });
        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_pipeline"),
            layout: Some(&ssao_pl_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SSAO_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let ssao_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_uniform_buffer"),
            size: std::mem::size_of::<SsaoParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Non-filtering sampler for the depth texture (Depth32Float
        // with non-comparison sampler is a NonFiltering combination).
        let ssao_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // --- TAA pipeline ---
        let taa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa_shader"),
            source: wgpu::ShaderSource::Wgsl(TAA_SHADER_WGSL.into()),
        });
        let taa_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // hdr / bloom / ssao / history each: tex + sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let taa_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa_pl_layout"),
            bind_group_layouts: &[&taa_layout],
            push_constant_ranges: &[],
        });
        let taa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("taa_pipeline"),
            layout: Some(&taa_pl_layout),
            vertex: wgpu::VertexState {
                module: &taa_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &taa_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT, blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None, front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false, conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        });
        let taa_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa_uniform_buffer"),
            size: std::mem::size_of::<TaaParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- SSR pipeline ---
        let ssr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr_shader"),
            source: wgpu::ShaderSource::Wgsl(SSR_SHADER_WGSL.into()),
        });
        let ssr_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ssr_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_pl_layout"),
            bind_group_layouts: &[&ssr_layout],
            push_constant_ranges: &[],
        });
        let ssr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssr_pipeline"),
            layout: Some(&ssr_pl_layout),
            vertex: wgpu::VertexState {
                module: &ssr_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssr_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT, blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None, front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false, conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        });
        let ssr_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr_uniform_buffer"),
            size: std::mem::size_of::<SsrParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Auto-exposure pipeline ---
        let exposure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exposure_shader"),
            source: wgpu::ShaderSource::Wgsl(EXPOSURE_SHADER_WGSL.into()),
        });
        let exposure_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exposure_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let exposure_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("exposure_pl_layout"),
            bind_group_layouts: &[&exposure_layout],
            push_constant_ranges: &[],
        });
        let exposure_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("exposure_pipeline"),
            layout: Some(&exposure_pl_layout),
            vertex: wgpu::VertexState {
                module: &exposure_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &exposure_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None, front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false, conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        });
        let exposure_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure_uniform_buffer"),
            size: std::mem::size_of::<ExposureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            surface,
            surface_config,
            pipeline_2d,
            pipeline_3d,
            uniform_buffers,
            uniform_bind_groups,
            current_uniform_idx: 0,
            uniform_slot_count: 0,
            uniform_buffer_3d,
            uniform_bind_group_3d,
            lighting_uniforms,
            lighting_buffer,
            lighting_bind_group,
            joint_buffer,
            joint_bind_group,
            texture_bind_group_layout,
            texture_bind_groups,
            textures,
            texture_sizes,
            sampler,
            nearest_sampler,
            depth_texture,
            depth_view,
            hdr_rt_texture,
            hdr_rt_view,
            material_rt_texture,
            material_rt_view,
            composite_pipeline,
            composite_layout,
            composite_sampler,
            tonemap_kind: 0,
            auto_exposure: false,
            manual_exposure: 1.0,
            auto_exposure_key: 0.18,
            auto_exposure_rate: 0.05,
            chromatic_aberration: 0.0,
            vignette_strength: 0.0,
            vignette_softness: 0.25,
            grain_strength: 0.0,
            exposure_textures,
            exposure_views,
            exposure_current_idx: 0,
            exposure_pipeline,
            exposure_layout,
            exposure_uniform_buffer,
            composite_uniform_buffer,
            bloom_chain_textures,
            bloom_mip_views,
            bloom_full_view,
            bloom_pipeline_threshold_downsample,
            bloom_pipeline_downsample,
            bloom_pipeline_upsample,
            bloom_layout,
            bloom_uniform_buffer,
            bloom_intensity: 0.04,
            ssao_rt_texture,
            ssao_rt_view,
            ssao_pipeline,
            ssao_layout,
            ssao_uniform_buffer,
            ssao_depth_sampler,
            ssao_strength: 1.0,
            ssao_radius: 0.006,
            taa_textures,
            taa_views,
            taa_current_idx: 0,
            taa_pipeline,
            taa_layout,
            taa_uniform_buffer,
            taa_frame_index: 0,
            taa_enabled: true,
            prev_vp_matrix: IDENTITY_MAT4,
            fog_color: [0.7, 0.75, 0.82],
            fog_density: 0.0,
            fog_height_ref: 0.0,
            fog_height_falloff: 0.25,
            sun_shaft_strength: 0.0,
            sun_shaft_decay: 0.96,
            sun_shaft_color: [1.0, 0.92, 0.78],
            ssr_rt_texture,
            ssr_rt_view,
            ssr_pipeline,
            ssr_layout,
            ssr_uniform_buffer,
            ssr_strength: 0.5,
            ssr_enabled: true,
            vertices_2d: Vec::with_capacity(4096),
            indices_2d: Vec::with_capacity(8192),
            draw_calls_2d: Vec::new(),
            vertices_3d: Vec::with_capacity(16384),
            indices_3d: Vec::with_capacity(32768),
            draw_calls_3d: Vec::new(),
            current_texture_3d: 0,
            persistent_vb_2d,
            persistent_ib_2d,
            persistent_vb_3d,
            persistent_ib_3d,
            persistent_vb_2d_capacity: vb_2d_cap,
            persistent_ib_2d_capacity: ib_2d_cap,
            persistent_vb_3d_capacity: vb_3d_cap,
            persistent_ib_3d_capacity: ib_3d_cap,
            model_gpu_cache: HashMap::new(),
            model_draw_commands: Vec::with_capacity(64),
            model_uniform_buffers,
            model_uniform_bind_groups,
            next_model_uniform_slot: 0,
            current_vp_matrix: IDENTITY_MAT4,
            current_view_matrix: IDENTITY_MAT4,
            current_proj_matrix: IDENTITY_MAT4,
            current_camera_pos: [0.0, 0.0, 0.0],
            uniform_3d_layout,
            render_mode: RenderMode::ScreenSpace,
            debug_frame: 0,
            pending_joint_matrices: None,
            model_skin_scale: 1.0,
            clear_color: wgpu::Color::RED,
            custom_pipelines: Vec::new(),
            shadow_map,
            screenshot_requested: false,
            screenshot_data: None,
            pending_screenshot_path: None,
            rt_color_view: None,
            rt_depth_view: None,
            rt_depth_texture: None,
            rt_width: 0,
            rt_height: 0,
            sky_texture: None,
            sky_bind_group: None,
            sky_uniform_buffer,
            sky_pipeline,
            sky_bind_group_layout,
            sky_sampler,
            scene_pipeline,
            scene_material_layout,
            _scene_env_default_texture: scene_env_default_texture,
            scene_env_default_view,
            env_sampler,
            lighting_layout,
            _brdf_lut_texture: brdf_lut_texture,
            brdf_lut_view,
            brdf_lut_sampler,
            prefilter_pipeline,
            prefilter_diffuse_pipeline,
            prefilter_layout,
            prefilter_uniform_buffer,
            _default_normal_texture: default_normal_tex,
            default_normal_view,
        }
    }

    /// Q1: Set up a render target override. The next end_frame will render to
    /// this texture view instead of the surface. Call end_texture_mode to clear.
    pub fn begin_texture_mode(&mut self, texture: &wgpu::Texture, width: u32, height: u32) {
        let color_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rt_depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.rt_color_view = Some(color_view);
        self.rt_depth_view = Some(depth_view);
        self.rt_depth_texture = Some(depth_tex);
        self.rt_width = width;
        self.rt_height = height;
    }

    /// Q1: Create a render texture and register it for sampling via drawTexture.
    /// Returns (bind_group_index, texture_vec_index).
    pub fn create_render_texture(&mut self, width: u32, height: u32) -> (u32, usize) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_bg"), layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tex_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let idx = self.texture_bind_groups.len() as u32;
        let tex_idx = self.textures.len();
        self.texture_bind_groups.push(bind_group);
        self.textures.push(texture);
        self.texture_sizes.push((width, height));
        (idx, tex_idx)
    }

    /// Q1: Get a reference to an internal texture by index.
    pub fn get_texture_ref(&self, index: usize) -> Option<&wgpu::Texture> {
        self.textures.get(index)
    }

    /// Q1: Clear the render target override.
    pub fn end_texture_mode(&mut self) {
        self.rt_color_view = None;
        self.rt_depth_view = None;
        self.rt_depth_texture = None;
        self.rt_width = 0;
        self.rt_height = 0;
    }

    // ============================================================
    // Lifecycle
    // ============================================================

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);

            let (dt, dv) = create_depth_texture(&self.device, width, height);
            self.depth_texture = dt;
            self.depth_view = dv;
            let (hdr_t, hdr_v) = create_hdr_rt(&self.device, width, height);
            self.hdr_rt_texture = hdr_t;
            self.hdr_rt_view = hdr_v;
            let (mat_t, mat_v) = create_material_rt(&self.device, width, height);
            self.material_rt_texture = mat_t;
            self.material_rt_view = mat_v;
            let (bt, bm, bf) = create_bloom_chain(&self.device, width, height, BLOOM_MIP_COUNT);
            self.bloom_chain_textures = bt;
            self.bloom_mip_views = bm;
            self.bloom_full_view = bf;
            let (st, sv) = create_ssao_rt(&self.device, width, height);
            self.ssao_rt_texture = st;
            self.ssao_rt_view = sv;
            let (taa_t, taa_v) = create_taa_textures(&self.device, width, height);
            self.taa_textures = taa_t;
            self.taa_views = taa_v;
            self.taa_frame_index = 0; // reset jitter sequence on resize
            let (sr_t, sr_v) = create_ssr_rt(&self.device, width, height);
            self.ssr_rt_texture = sr_t;
            self.ssr_rt_view = sr_v;
        }
    }

    pub fn set_clear_color(&mut self, r: f64, g: f64, b: f64, a: f64) {
        self.clear_color = wgpu::Color {
            r: r / 255.0,
            g: g / 255.0,
            b: b / 255.0,
            a: a / 255.0,
        };
    }

    /// Set the multiplier applied to every env-map sample (sky pass +
    /// IBL diffuse + IBL specular). Defaults to 1.0. Storing in the
    /// lighting uniform's camera_pos.w avoids a new bind point.
    /// Bloom additive intensity (0 = off, 0.04 = default, 1.0 = very
    /// strong). Affects only pixels above the HDR threshold (1.0
    /// luminance), so dim scenes look unchanged regardless of value.
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        self.bloom_intensity = intensity.max(0.0);
    }

    /// SSAO strength (0 = off, 1 = default, ≥3 = stylized). Always
    /// works since SSAO darkens crevices regardless of HDR levels.
    pub fn set_ssao_strength(&mut self, strength: f32) {
        self.ssao_strength = strength.max(0.0);
    }

    /// SSAO sample radius in UV units. 0.006 (~0.6% of viewport
    /// height) is the default — wider radius catches larger AO
    /// features but also blurs detail and increases halo risk.
    pub fn set_ssao_radius(&mut self, radius: f32) {
        self.ssao_radius = radius.max(0.0001);
    }

    /// Toggle TAA on/off. Off = no jitter, no history blend, no
    /// extra texture writes. On = sub-pixel super-sampling for
    /// static and slow-camera scenes.
    pub fn set_taa_enabled(&mut self, enabled: bool) {
        if enabled != self.taa_enabled {
            self.taa_enabled = enabled;
            self.taa_frame_index = 0;
        }
    }

    /// Toggle SSR on/off. SSR contributes nothing in scenes with
    /// no on-screen geometry to reflect (e.g., single object
    /// against sky) — turning it off there saves a fullscreen
    /// pass.
    pub fn set_ssr_enabled(&mut self, enabled: bool) {
        self.ssr_enabled = enabled;
    }

    /// SSR strength multiplier (0 = off, 0.5 = default, 1+ = strong).
    /// Applies on top of the prefiltered IBL specular reflection,
    /// adding sharp on-screen reflections where they exist.
    pub fn set_ssr_strength(&mut self, strength: f32) {
        self.ssr_strength = strength.max(0.0);
    }

    /// Select the display tonemap curve. 0 = ACES (default, used
    /// by the bloom-reference path tracer so validation diffs stay
    /// meaningful). 1 = AgX (Troy Sobotka 2022) — better hue
    /// preservation in saturated colors, matches Blender 4.0+ /
    /// UE5 "PBR Neutral" look.
    pub fn set_tonemap_kind(&mut self, kind: u32) {
        self.tonemap_kind = kind;
    }

    /// Toggle auto-exposure. Off (default) = manual exposure
    /// multiplier. On = per-frame average scene luminance drives
    /// exposure toward `auto_exposure_key` (0.18 photography
    /// standard). Instant adapt — no inter-frame smoothing yet,
    /// so scene cuts pop. Fine for static or slow-motion cameras.
    pub fn set_auto_exposure(&mut self, on: bool) {
        self.auto_exposure = on;
    }

    /// Manual exposure multiplier. Applied when auto_exposure
    /// is off. 1.0 = no change. 2.0 = twice as bright. Clamp is
    /// [0, +∞) — negative silently becomes 0.
    pub fn set_manual_exposure(&mut self, value: f32) {
        self.manual_exposure = value.max(0.0);
    }

    /// Auto-exposure target scene key (average luminance to drive
    /// toward). Lower = darker overall, higher = brighter. 0.18
    /// is the 18%-gray photography standard; 0.14 gives a slightly
    /// moodier look, 0.25 a brighter one.
    pub fn set_auto_exposure_key(&mut self, key: f32) {
        self.auto_exposure_key = key.clamp(0.01, 1.0);
    }

    /// Auto-exposure smoothing rate per frame. 0 = no adapt (stuck
    /// at whatever the current texture holds), 0.05 ≈ 20-frame
    /// half-life at 60 fps (default — feels natural for camera
    /// moves), 1 = instant (pops on scene cuts).
    pub fn set_auto_exposure_rate(&mut self, rate: f32) {
        self.auto_exposure_rate = rate.clamp(0.0, 1.0);
    }

    /// Fog color that distant geometry fades to (rgb, 0-1).
    pub fn set_fog_color(&mut self, r: f32, g: f32, b: f32) {
        self.fog_color = [r, g, b];
    }

    /// Fog density. 0 (default) = fog disabled. 0.02 = gentle
    /// atmospheric haze, 0.1 = heavy smog, 1+ = soup. Applied
    /// exponentially over world-space distance.
    pub fn set_fog_density(&mut self, density: f32) {
        self.fog_density = density.max(0.0);
    }

    /// Fog altitude-based falloff. `height_ref` is the world Y
    /// below which density stays at the full value; `falloff_rate`
    /// controls how fast density drops as you go above it. Default
    /// 0.0 / 0.25 gives a natural ground-haze look.
    pub fn set_fog_height_falloff(&mut self, height_ref: f32, falloff_rate: f32) {
        self.fog_height_ref = height_ref;
        self.fog_height_falloff = falloff_rate.max(0.0);
    }

    /// Chromatic aberration strength — radial RGB-channel split at
    /// the screen edges. 0 (default) = off. 0.002 ≈ subtle film
    /// fringe, 0.01 ≈ obvious lens defect.
    pub fn set_chromatic_aberration(&mut self, strength: f32) {
        self.chromatic_aberration = strength.max(0.0);
    }

    /// Vignette darkening of the screen corners. `strength` 0..1
    /// (0 = off, 1 = corners fully black). `softness` 0..1
    /// controls the falloff width — smaller = harder edge.
    pub fn set_vignette(&mut self, strength: f32, softness: f32) {
        self.vignette_strength = strength.clamp(0.0, 1.0);
        self.vignette_softness = softness.clamp(0.001, 1.0);
    }

    /// Animated film-grain strength (added to luma post-tonemap).
    /// 0 (default) = off. 0.02 = subtle, 0.08 = noticeable.
    /// Grain reseeds per frame so it crawls naturally; freezes when
    /// the renderer's frame index isn't advancing.
    pub fn set_film_grain(&mut self, strength: f32) {
        self.grain_strength = strength.max(0.0);
    }

    /// Sun shaft (screen-space god ray) strength. 0 (default) = off.
    /// 0.4 = subtle haze, 1.0+ = obvious cinematic shafts. The
    /// shafts are sampled from the depth buffer along a screen-space
    /// line toward the sun's projected position, so any geometry
    /// occluding the sun naturally cuts the shafts.
    pub fn set_sun_shaft_strength(&mut self, strength: f32) {
        self.sun_shaft_strength = strength.max(0.0);
    }

    /// Per-sample decay (0..1). Larger = longer shafts. 0.96 default
    /// gives ~32-tap visible falloff.
    pub fn set_sun_shaft_decay(&mut self, decay: f32) {
        self.sun_shaft_decay = decay.clamp(0.0, 1.0);
    }

    /// Sun shaft tint (rgb).
    pub fn set_sun_shaft_color(&mut self, r: f32, g: f32, b: f32) {
        self.sun_shaft_color = [r, g, b];
    }

    pub fn set_env_intensity(&mut self, intensity: f32) {
        self.lighting_uniforms.camera_pos[3] = intensity;
        self.queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_uniforms),
        );
    }

    /// Upload an HDR equirectangular environment map. The `data` is
    /// `width * height * 3` packed f32 RGB triples in linear space —
    /// the output of `image::codecs::hdr::HdrDecoder::read_image()`
    /// laid out row-major. Replaces any previously-loaded env.
    ///
    /// Generates a mip chain by GGX-convolving the source env at
    /// roughness = mip / (mips - 1) for each mip ≥ 1. This is the
    /// Karis 2013 split-sum specular prefilter; combined with the
    /// pre-baked BRDF LUT it gives correct PBR specular reflections
    /// at any roughness without per-frame importance sampling.
    /// Mip 0 is the original radiance (used by the sky pass).
    pub fn load_env_from_hdr(&mut self, width: u32, height: u32, rgb_f32: &[f32]) {
        let max_dim = width.max(height);
        let mip_count = ((max_dim as f32).log2().floor() as u32 + 1).min(7);

        // Pack f32 RGB → packed f16 RGBA for the GPU.
        let texel_count = (width as usize) * (height as usize);
        let mut packed: Vec<u16> = Vec::with_capacity(texel_count * 4);
        for px in 0..texel_count {
            packed.push(half::f16::from_f32(rgb_f32[px * 3]).to_bits());
            packed.push(half::f16::from_f32(rgb_f32[px * 3 + 1]).to_bits());
            packed.push(half::f16::from_f32(rgb_f32[px * 3 + 2]).to_bits());
            packed.push(half::f16::from_f32(1.0).to_bits());
        }

        // Source texture — single mip, holds the original radiance.
        // We sample from this when prefiltering each output mip so a
        // single texture isn't both read and written in the same pass.
        let src_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky_env_src"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_DST
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&packed),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        // Destination texture — full mip chain, RENDER_ATTACHMENT for
        // the prefilter passes plus TEXTURE_BINDING for sampling at
        // draw time.
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky_env_texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_DST
                 | wgpu::TextureUsages::COPY_SRC
                 | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Mip 0 = exact copy of source (mirror reflection — no convolution).
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("env_prefilter_encoder"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        // Source bind group — same for every mip's prefilter pass.
        let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let prefilter_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefilter_src_bg"),
            layout: &self.prefilter_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.prefilter_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.env_sampler) },
            ],
        });

        // GGX-prefilter mips 1..N-1 with roughness scaling. The
        // smallest mip (level N-1) instead gets a cosine-weighted
        // diffuse irradiance convolution so the scene shader's IBL
        // diffuse sample is properly Lambertian — that mip is
        // dedicated to diffuse, while specular reads at fractional
        // mips ≤ (N-2) for normal roughness values.
        let last_mip = mip_count - 1;
        for level in 1..mip_count {
            let mip_w = (width >> level).max(1);
            let mip_h = (height >> level).max(1);
            let is_diffuse = level == last_mip;
            let roughness = level as f32 / (mip_count - 1) as f32;
            let sample_count = if is_diffuse {
                // 1024 cosine samples gives a smooth irradiance map
                // even for high-contrast HDRs.
                1024.0
            } else {
                (32.0 + 96.0 * roughness).round()
            };

            let uniforms = PrefilterUniforms {
                params: [roughness, sample_count, mip_w as f32, mip_h as f32],
            };
            self.queue.write_buffer(&self.prefilter_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            let mip_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("prefilter_dst_mip_view"),
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prefilter_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &mip_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let pl = if is_diffuse {
                &self.prefilter_diffuse_pipeline
            } else {
                &self.prefilter_pipeline
            };
            pass.set_pipeline(pl);
            pass.set_bind_group(0, &prefilter_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_bg"),
            layout: &self.sky_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sky_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sky_sampler),
                },
            ],
        });

        // Rebuild lighting_bind_group so the scene shader's group 1
        // binding points at this env texture (for IBL ambient and
        // specular reflections). The lighting uniform buffer + BRDF
        // LUT bindings stay put — only env tex/sampler change.
        let new_lighting_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_bg"),
            layout: &self.lighting_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.lighting_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.env_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.brdf_lut_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
            ],
        });

        self.sky_texture = Some(texture);
        self.sky_bind_group = Some(bg);
        self.lighting_bind_group = new_lighting_bg;
    }

    /// Whether a sky env map has been uploaded — controls whether
    /// end_frame_with_scene runs the sky pass.
    pub fn has_env_map(&self) -> bool {
        self.sky_bind_group.is_some()
    }

    /// Render the sky pass into `pass`. Caller must have already set
    /// up the render pass with the surface color attachment + depth
    /// attachment. Reconstructs the camera basis from the view matrix
    /// and uploads pre-scaled right/up vectors so the shader just
    /// needs to multiply by NDC components.
    fn render_sky_pass(&self, pass: &mut wgpu::RenderPass<'_>, intensity: f32) {
        let bg = match &self.sky_bind_group {
            Some(b) => b,
            None => return,
        };
        // Extract camera basis from the view matrix. View is
        // world→camera, so its rows (== columns of the inverse) are
        // the camera's world-space axes. With column-major storage,
        // view[col][row] gives the standard layout: row 0 of view is
        // the world-space right vector of the camera, etc.
        let v = self.current_view_matrix;
        // view matrix layout (column-major):
        //   row 0 = camera right (world space)
        //   row 1 = camera up
        //   row 2 = -camera forward (right-handed lookAt convention)
        // We want forward in world space, so negate row 2.
        let right_world = [v[0][0], v[1][0], v[2][0]];
        let up_world    = [v[0][1], v[1][1], v[2][1]];
        let forward_world = [-v[0][2], -v[1][2], -v[2][2]];

        // Pre-scale by tan(fovy/2) and aspect so the shader is a
        // single multiply-add per axis.
        let aspect = self.surface_config.width as f32 / self.surface_config.height as f32;
        // Recover tan(fovy/2) from the projection matrix: for a
        // standard perspective P, P[1][1] = 1 / tan(fovy/2). So
        // tan(fovy/2) = 1 / P[1][1].
        let p = self.current_proj_matrix;
        let tan_half = if p[1][1].abs() > 1e-6 { 1.0 / p[1][1] } else { 1.0 };

        let uniforms = SkyUniforms {
            right: [
                right_world[0] * tan_half * aspect,
                right_world[1] * tan_half * aspect,
                right_world[2] * tan_half * aspect,
                0.0,
            ],
            up: [
                up_world[0] * tan_half,
                up_world[1] * tan_half,
                up_world[2] * tan_half,
                0.0,
            ],
            forward: [forward_world[0], forward_world[1], forward_world[2], 0.0],
            intensity: [intensity, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.sky_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        pass.set_pipeline(&self.sky_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Get the current view-projection matrix (set by begin_mode_3d).
    pub fn vp_matrix(&self) -> [[f32; 4]; 4] {
        self.current_vp_matrix
    }

    /// Get the current camera position (set by begin_mode_3d).
    pub fn camera_pos(&self) -> [f32; 3] {
        self.current_camera_pos
    }

    /// Get the inverse VP matrix for unprojecting screen coords to world rays.
    pub fn inverse_vp_matrix(&self) -> [[f32; 4]; 4] {
        mat4_invert(self.current_vp_matrix)
    }

    /// Get the 3D uniform bind group layout (for creating per-node uniform bind groups).
    pub fn uniform_3d_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uniform_3d_layout
    }

    /// Get texture bind groups (for scene graph rendering).
    pub fn texture_bind_groups_slice(&self) -> &[wgpu::BindGroup] {
        &self.texture_bind_groups
    }

    /// Build a scene-pipeline material uniform buffer holding the
    /// per-material scalar factors. Called once per material — the
    /// bind group below references this buffer.
    pub fn create_scene_material_uniform(
        &self,
        metallic: f32,
        roughness: f32,
        emissive: [f32; 3],
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let uniforms = SceneMaterialUniforms::new(metallic, roughness, emissive);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_material_uniform"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Build a scene-pipeline material bind group.
    ///
    /// Each of the four texture indices can be zero to mean 'not
    /// present' — we substitute a sensible default so the fragment
    /// shader doesn't need per-slot presence flags:
    ///   • base color  → textures[0] (white)
    ///   • normal map  → flat (0,0,1) texture — TBN becomes a no-op
    ///   • MR texture  → white (roughness=1, metallic=1 before factor)
    ///   • emissive    → white — multiplied by emissive_factor, which
    ///     is zero for non-emissive materials, giving 0.
    pub fn create_scene_material_bg(
        &self,
        base_color_tex_idx: u32,
        normal_tex_idx: u32,
        metallic_roughness_tex_idx: u32,
        emissive_tex_idx: u32,
        occlusion_tex_idx: u32,
        material_uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let view_or_white = |idx: u32| -> wgpu::TextureView {
            let i = idx as usize;
            let tex = if idx == 0 || i >= self.textures.len() {
                &self.textures[0]
            } else {
                &self.textures[i]
            };
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let base_view = view_or_white(base_color_tex_idx);
        let mr_view = view_or_white(metallic_roughness_tex_idx);
        let em_view = view_or_white(emissive_tex_idx);
        // Occlusion default = white texture: shader does
        // `mix(1.0, occlusion, strength)`, so a white sample gives
        // 1.0 (no occlusion) regardless of strength.
        let occ_view = view_or_white(occlusion_tex_idx);

        // Normal map uses the flat-normal default when not specified
        // (white here would give incorrect perturbation since it
        // decodes to (1, 1, 1) in tangent space instead of (0, 0, 1)).
        // All four view locals live until after create_bind_group, so
        // taking references to them is safe.
        let normal_view_owned = if normal_tex_idx == 0 || (normal_tex_idx as usize) >= self.textures.len() {
            None
        } else {
            Some(self.textures[normal_tex_idx as usize].create_view(&wgpu::TextureViewDescriptor::default()))
        };
        let normal_view_ref: &wgpu::TextureView = normal_view_owned
            .as_ref()
            .unwrap_or(&self.default_normal_view);

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_material_bg"),
            layout: &self.scene_material_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&base_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal_view_ref) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&mr_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&em_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: material_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&occ_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }

    pub fn begin_frame(&mut self) {
        self.vertices_2d.clear();
        self.indices_2d.clear();
        self.draw_calls_2d.clear();
        self.vertices_3d.clear();
        self.indices_3d.clear();
        self.draw_calls_3d.clear();
        self.model_draw_commands.clear();
        self.next_model_uniform_slot = 0;
        self.current_texture_3d = 0;
        self.current_uniform_idx = 0;
        self.uniform_slot_count = 0;
        self.render_mode = RenderMode::ScreenSpace;

        // Write identity uniforms to slot 0
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;
        let uniforms = Uniforms2D {
            screen_size: [w, h],
            _pad: [0.0; 2],
            view_proj: IDENTITY_MAT4,
        };
        self.queue.write_buffer(&self.uniform_buffers[0], 0, bytemuck::bytes_of(&uniforms));

        // Reset lighting to defaults (clears additional lights too)
        self.lighting_uniforms = LightingUniforms::defaults();
        self.queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&self.lighting_uniforms));
        self.clear_additional_lights();

        // DEBUG: joint animation disabled for iOS port
        // self.debug_frame += 1;
        // let angle = (self.debug_frame as f32) * 0.03;
        // self.set_joint_test(0, angle.sin() * 0.8);
        // self.set_joint_test(5, (angle * 1.5).sin() * 0.5);
    }

    pub fn end_frame(&mut self) {
        // DEBUG: Clear all 2D content - only clear color should render
        self.vertices_2d.clear();
        self.indices_2d.clear();
        self.draw_calls_2d.clear();

        // Flush pending joint matrices to GPU right before rendering
        self.flush_joint_matrices();

        // Q1: If rendering to a texture, use the RT view. Otherwise use the surface.
        // We take ownership of the RT views (via Option::take) to avoid holding a
        // borrow on `self` while the rest of end_frame mutates it.
        let rt_color = self.rt_color_view.take();
        let rt_depth = self.rt_depth_view.take();
        let using_rt = rt_color.is_some();

        let surface_output = if using_rt {
            None
        } else {
            match self.surface.get_current_texture() {
                Ok(t) => Some(t),
                Err(e) => {
                    // Log the error to a file so we can diagnose tvOS rendering issues
                    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        let _ = std::fs::write("/tmp/bloom_surface_err.txt", format!("get_current_texture failed: {:?}\n", e));
                    }
                    self.surface.configure(&self.device, &self.surface_config);
                    // Restore RT views if they were set.
                    self.rt_color_view = rt_color;
                    self.rt_depth_view = rt_depth;
                    return;
                }
            }
        };

        let view: wgpu::TextureView;
        let owned_depth_view: wgpu::TextureView;

        if let Some(ref rt_view) = rt_color {
            view = rt_view.clone();
            owned_depth_view = rt_depth.as_ref().unwrap().clone();
        } else {
            view = surface_output.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());
            owned_depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }

        // Restore RT views so they persist across frames.
        self.rt_color_view = rt_color;
        self.rt_depth_view = rt_depth;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bloom_encoder"),
        });

        // Upload 2D data to persistent GPU buffers
        let has_2d = !self.vertices_2d.is_empty();
        if has_2d {
            let vb_size = std::mem::size_of_val(self.vertices_2d.as_slice());
            let ib_size = std::mem::size_of_val(self.indices_2d.as_slice());
            self.ensure_buffer_capacity_2d(vb_size, ib_size);
            self.queue.write_buffer(&self.persistent_vb_2d, 0, bytemuck::cast_slice(&self.vertices_2d));
            self.queue.write_buffer(&self.persistent_ib_2d, 0, bytemuck::cast_slice(&self.indices_2d));
        }

        // Upload 3D data to persistent GPU buffers
        let has_3d = !self.vertices_3d.is_empty();
        if has_3d {
            let vb_size = std::mem::size_of_val(self.vertices_3d.as_slice());
            let ib_size = std::mem::size_of_val(self.indices_3d.as_slice());
            self.ensure_buffer_capacity_3d(vb_size, ib_size);
            self.queue.write_buffer(&self.persistent_vb_3d, 0, bytemuck::cast_slice(&self.vertices_3d));
            self.queue.write_buffer(&self.persistent_ib_3d, 0, bytemuck::cast_slice(&self.indices_3d));
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &owned_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw 3D geometry first (with depth testing), batched by texture
            if has_3d {
                pass.set_pipeline(&self.pipeline_3d);
                pass.set_bind_group(0, &self.uniform_bind_group_3d, &[]);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);
                pass.set_vertex_buffer(0, self.persistent_vb_3d.slice(..));
                pass.set_index_buffer(self.persistent_ib_3d.slice(..), wgpu::IndexFormat::Uint32);

                if self.draw_calls_3d.is_empty() {
                    // No draw calls tracked — draw all with white texture (backward compat)
                    pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                    pass.draw_indexed(0..self.indices_3d.len() as u32, 0, 0..1);
                } else {
                    let num_calls = self.draw_calls_3d.len();
                    for i in 0..num_calls {
                        let call = &self.draw_calls_3d[i];
                        let next_start = if i + 1 < num_calls {
                            self.draw_calls_3d[i + 1].index_start
                        } else {
                            self.indices_3d.len() as u32
                        };
                        let count = next_start - call.index_start;
                        if count == 0 { continue; }
                        let tex_idx = call.texture_idx as usize;
                        if tex_idx < self.texture_bind_groups.len() {
                            pass.set_bind_group(2, &self.texture_bind_groups[tex_idx], &[]);
                        } else {
                            pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                        }
                        pass.draw_indexed(call.index_start..next_start, 0, 0..1);
                    }
                }
            }

            // Draw cached models (static models with GPU-resident buffers).
            // Use the scene pipeline so PBR-style material bindings (base
            // color + normal map) apply — drawModel should behave the same
            // as attachModelToNode for PBR purposes.
            if !self.model_draw_commands.is_empty() {
                pass.set_pipeline(&self.scene_pipeline);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);

                for cmd in &self.model_draw_commands {
                    if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                        if cmd.mesh_idx < meshes.len() {
                            let mesh = &meshes[cmd.mesh_idx];
                            pass.set_bind_group(0, &self.model_uniform_bind_groups[cmd.uniform_slot], &[]);
                            pass.set_bind_group(2, &mesh.material_bg, &[]);
                            pass.set_vertex_buffer(0, mesh.vb.slice(..));
                            pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                }
            }

            // Draw 2D geometry (no depth testing, always passes)
            if has_2d {
                pass.set_pipeline(&self.pipeline_2d);
                pass.set_vertex_buffer(0, self.persistent_vb_2d.slice(..));
                pass.set_index_buffer(self.persistent_ib_2d.slice(..), wgpu::IndexFormat::Uint32);

                let num_calls = self.draw_calls_2d.len();
                for i in 0..num_calls {
                    let call = &self.draw_calls_2d[i];
                    let next_start = if i + 1 < num_calls {
                        self.draw_calls_2d[i + 1].index_start
                    } else {
                        self.indices_2d.len() as u32
                    };
                    let count = next_start - call.index_start;
                    if count == 0 { continue; }

                    pass.set_bind_group(0, &self.uniform_bind_groups[call.uniform_idx as usize], &[]);
                    if (call.texture_idx as usize) < self.texture_bind_groups.len() {
                        pass.set_bind_group(1, &self.texture_bind_groups[call.texture_idx as usize], &[]);
                    }
                    pass.draw_indexed(call.index_start..next_start, 0, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Some(out) = surface_output { out.present(); }
    }

    /// Like end_frame, but also renders retained scene graph nodes.
    pub fn end_frame_with_scene(&mut self, scene: &crate::scene::SceneGraph) {
        self.flush_joint_matrices();

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bloom_encoder"),
        });

        // Shadow pass: render scene nodes from light's perspective
        if self.shadow_map.enabled {
            // Compute light VP from the primary directional light direction
            let light_dir = [
                self.lighting_uniforms.light_dir[0],
                self.lighting_uniforms.light_dir[1],
                self.lighting_uniforms.light_dir[2],
            ];
            self.shadow_map.compute_light_vp(light_dir, [0.0, 0.0, 0.0]);

            {
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_map.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                shadow_pass.set_pipeline(&self.shadow_map.pipeline);

                // Render each visible scene node into shadow map
                for (_handle, node) in scene.nodes.iter() {
                    if !node.visible || !node.cast_shadow || node.indices.is_empty() {
                        continue;
                    }
                    let Some(vb) = &node.gpu_vb else { continue };
                    let Some(ib) = &node.gpu_ib else { continue };

                    // Write shadow uniforms (light_vp * model)
                    let shadow_uniforms = crate::shadows::ShadowUniforms {
                        light_vp: self.shadow_map.light_vp,
                        model: node.transform,
                    };
                    self.queue.write_buffer(
                        &self.shadow_map.uniform_buffer,
                        0,
                        bytemuck::bytes_of(&shadow_uniforms),
                    );

                    shadow_pass.set_bind_group(0, &self.shadow_map.uniform_bind_group, &[]);
                    shadow_pass.set_vertex_buffer(0, vb.slice(..));
                    shadow_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    shadow_pass.draw_indexed(0..node.gpu_index_count, 0, 0..1);
                }
            }
        }

        // Upload immediate-mode 2D data
        let has_2d = !self.vertices_2d.is_empty();
        if has_2d {
            let vb_size = std::mem::size_of_val(self.vertices_2d.as_slice());
            let ib_size = std::mem::size_of_val(self.indices_2d.as_slice());
            self.ensure_buffer_capacity_2d(vb_size, ib_size);
            self.queue.write_buffer(&self.persistent_vb_2d, 0, bytemuck::cast_slice(&self.vertices_2d));
            self.queue.write_buffer(&self.persistent_ib_2d, 0, bytemuck::cast_slice(&self.indices_2d));
        }

        // Upload immediate-mode 3D data
        let has_3d = !self.vertices_3d.is_empty();
        if has_3d {
            let vb_size = std::mem::size_of_val(self.vertices_3d.as_slice());
            let ib_size = std::mem::size_of_val(self.indices_3d.as_slice());
            self.ensure_buffer_capacity_3d(vb_size, ib_size);
            self.queue.write_buffer(&self.persistent_vb_3d, 0, bytemuck::cast_slice(&self.vertices_3d));
            self.queue.write_buffer(&self.persistent_ib_3d, 0, bytemuck::cast_slice(&self.indices_3d));
        }

        // ============================================================
        // HDR pass: sky + 3D + scene → linear HDR offscreen RT.
        // ============================================================
        // The composite-tonemap pass downstream reads this RT and
        // writes the final image to the sRGB surface. Keeping the
        // intermediate radiance in HDR sets up a future bloom pass
        // and means tonemap + sRGB encode happen exactly once, in
        // one place.
        {
            // HDR clear: the user's clear_color is in 0-1 srgb-ish
            // range; treat it as the linear background for the HDR
            // RT. After tonemap it ends up roughly the same shade.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_hdr_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.hdr_rt_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.material_rt_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Default = (0, 1) = non-metal, fully
                            // rough — sky / 3D / blank pixels won't
                            // SSR-reflect.
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 1.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Sky uses the same env_intensity as IBL so the background
            // and lighting stay in sync — otherwise bumping IBL down
            // would leave the sky blown out.
            self.render_sky_pass(&mut pass, self.lighting_uniforms.camera_pos[3]);

            if has_3d {
                pass.set_pipeline(&self.pipeline_3d);
                pass.set_bind_group(0, &self.uniform_bind_group_3d, &[]);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);
                pass.set_vertex_buffer(0, self.persistent_vb_3d.slice(..));
                pass.set_index_buffer(self.persistent_ib_3d.slice(..), wgpu::IndexFormat::Uint32);

                if self.draw_calls_3d.is_empty() {
                    pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                    pass.draw_indexed(0..self.indices_3d.len() as u32, 0, 0..1);
                } else {
                    let num_calls = self.draw_calls_3d.len();
                    for i in 0..num_calls {
                        let call = &self.draw_calls_3d[i];
                        let next_start = if i + 1 < num_calls {
                            self.draw_calls_3d[i + 1].index_start
                        } else {
                            self.indices_3d.len() as u32
                        };
                        let count = next_start - call.index_start;
                        if count == 0 { continue; }
                        let tex_idx = call.texture_idx as usize;
                        if tex_idx < self.texture_bind_groups.len() {
                            pass.set_bind_group(2, &self.texture_bind_groups[tex_idx], &[]);
                        } else {
                            pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                        }
                        pass.draw_indexed(call.index_start..next_start, 0, 0..1);
                    }
                }
            }

            // Cached models + retained scene graph — both via scene_pipeline.
            let has_cached_models = !self.model_draw_commands.is_empty();
            if has_cached_models || scene.node_count() > 0 {
                pass.set_pipeline(&self.scene_pipeline);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);

                if has_cached_models {
                    for cmd in &self.model_draw_commands {
                        if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                            if cmd.mesh_idx < meshes.len() {
                                let mesh = &meshes[cmd.mesh_idx];
                                pass.set_bind_group(0, &self.model_uniform_bind_groups[cmd.uniform_slot], &[]);
                                pass.set_bind_group(2, &mesh.material_bg, &[]);
                                pass.set_vertex_buffer(0, mesh.vb.slice(..));
                                pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        }
                    }
                }

                scene.render(&mut pass);
            }
        }

        // ============================================================
        // SSAO: half-res spiral-sample of the depth buffer.
        // ============================================================
        let surf_w = self.surface_config.width;
        let surf_h = self.surface_config.height;
        {
            let sp = SsaoParams {
                params: [
                    1.0 / surf_w as f32,
                    1.0 / surf_h as f32,
                    self.ssao_radius,
                    self.ssao_strength,
                ],
            };
            self.queue.write_buffer(&self.ssao_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao_bg"),
                layout: &self.ssao_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ssao_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_rt_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // SSR: view-space ray march of the depth buffer + HDR sample.
        // ============================================================
        if self.ssr_enabled {
            let inv_proj = mat4_invert(self.current_proj_matrix);
            let sp = SsrParams {
                inv_proj,
                proj: self.current_proj_matrix,
                params: [self.ssr_strength, 8.0, 32.0, 0.0],
            };
            self.queue.write_buffer(&self.ssr_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssr_bg"),
                layout: &self.ssr_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ssr_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.material_rt_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssr_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssr_rt_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ssr_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        } else {
            // SSR disabled — clear the RT so TAA's read returns 0
            // (transparent black). One-time clear is cheaper than a
            // full clear+pipeline switch every frame.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssr_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssr_rt_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            drop(pass);
        }

        // ============================================================
        // Bloom: progressive downsample (Karis-thresholded first tap)
        // followed by additive upsample back up the chain.
        // ============================================================
        let mip_dims: Vec<(u32, u32)> = (0..BLOOM_MIP_COUNT)
            .map(|i| (
                ((surf_w / 2) >> i).max(1),
                ((surf_h / 2) >> i).max(1),
            ))
            .collect();

        // Build per-pass bind groups + uniform writes. Each downsample
        // reads the previous mip (or hdr_rt for the first) and writes
        // to the current mip. Each upsample reads mip i+1 and blends
        // additively into mip i.
        let bloom_filter_radius = 1.0_f32; // upsample tent radius

        // Downsample chain: mip 0 reads HDR, mips 1..N read previous mip.
        for i in 0..BLOOM_MIP_COUNT as usize {
            let (src_view, src_w, src_h, threshold_pass) = if i == 0 {
                (&self.hdr_rt_view, surf_w as f32, surf_h as f32, true)
            } else {
                let prev = &self.bloom_mip_views[i - 1];
                let (pw, ph) = mip_dims[i - 1];
                (prev, pw as f32, ph as f32, false)
            };

            let bp = BloomParams {
                params: [1.0 / src_w, 1.0 / src_h, bloom_filter_radius, 1.0],
            };
            self.queue.write_buffer(&self.bloom_uniform_buffer, 0, bytemuck::bytes_of(&bp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_downsample_bg"),
                layout: &self.bloom_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.bloom_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mip_views[i],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let pl = if threshold_pass {
                &self.bloom_pipeline_threshold_downsample
            } else {
                &self.bloom_pipeline_downsample
            };
            pass.set_pipeline(pl);
            // Force the viewport to this mip's actual size — wgpu's
            // auto-viewport derives from the surface config, not the
            // mip-view attachment, so without this the bloom pass
            // writes into a fraction of the mip and leaves the rest
            // uninitialized.
            let (mw, mh) = mip_dims[i];
            pass.set_viewport(0.0, 0.0, mw as f32, mh as f32, 0.0, 1.0);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Upsample chain: blend mip i+1 additively into mip i for
        // i = N-2..0. Final mip 0 ends up with the full bloom result.
        for i in (0..(BLOOM_MIP_COUNT as usize - 1)).rev() {
            let src_view = &self.bloom_mip_views[i + 1];
            let (sw, sh) = mip_dims[i + 1];

            let bp = BloomParams {
                params: [1.0 / sw as f32, 1.0 / sh as f32, bloom_filter_radius, 0.0],
            };
            self.queue.write_buffer(&self.bloom_uniform_buffer, 0, bytemuck::bytes_of(&bp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom_upsample_bg"),
                layout: &self.bloom_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.bloom_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_upsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mip_views[i],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load — additive blend on top of what
                        // downsample wrote.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bloom_pipeline_upsample);
            // Same viewport fix as the downsample loop above — without
            // this the upsample tents only cover a sub-region of the
            // destination mip.
            let (mw, mh) = mip_dims[i];
            pass.set_viewport(0.0, 0.0, mw as f32, mh as f32, 0.0, 1.0);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }


        // ============================================================
        // TAA pass: combine current-frame HDR + bloom + SSAO with the
        // history texture, write to the OTHER ping-pong slot. Skipped
        // when TAA is off — composite reads hdr_rt directly instead.
        // ============================================================
        let taa_dst_idx = self.taa_current_idx;
        let taa_src_idx = 1 - self.taa_current_idx;

        if self.taa_enabled {
            // First few frames blend at 1.0 (use current as-is) so
            // the history initializes from the actual scene rather
            // than the pre-zero texture. After 4 frames switch to
            // the converging weight.
            let alpha = if self.taa_frame_index < 4 { 1.0 } else { 0.1 };
            // Inverse of current VP for world-pos reconstruction.
            // Use the unjittered version when feasible — for now we
            // pass the actual jittered current_vp_matrix; the jitter
            // is sub-pixel so reprojection error is negligible.
            let inv_vp = mat4_invert(self.current_vp_matrix);
            // Sun shaft screen-space position. Project a point far
            // along the sun direction through the current VP. If it
            // ends up behind the camera (clip.w ≤ 0) the sun is
            // off-screen — disable shafts this frame.
            let sun_dir = self.lighting_uniforms.light_dir;
            let sun_world = [sun_dir[0] * 1000.0, sun_dir[1] * 1000.0, sun_dir[2] * 1000.0, 1.0];
            let clip = mat4_mul_vec4(&self.current_vp_matrix, &sun_world);
            let (sun_uv, shaft_strength_eff) = if clip[3] > 0.0 {
                let ndc_x = clip[0] / clip[3];
                let ndc_y = clip[1] / clip[3];
                let u = ndc_x * 0.5 + 0.5;
                let v = 1.0 - (ndc_y * 0.5 + 0.5);
                // Allow off-screen suns to still cast shafts that
                // streak in from the edge — clamp to a small margin
                // beyond ±[0,1] rather than disabling outright.
                let off = u < -1.0 || u > 2.0 || v < -1.0 || v > 2.0;
                if off { ([0.0, 0.0], 0.0) } else { ([u, v], self.sun_shaft_strength) }
            } else {
                ([0.0, 0.0], 0.0)
            };

            let tp = TaaParams {
                params: [alpha, self.bloom_intensity, 0.0, 0.0],
                inv_vp,
                prev_vp: self.prev_vp_matrix,
                fog_color_density: [
                    self.fog_color[0],
                    self.fog_color[1],
                    self.fog_color[2],
                    self.fog_density,
                ],
                fog_params: [self.fog_height_ref, self.fog_height_falloff, 0.0, 0.0],
                sun_shaft_uv_strength: [
                    sun_uv[0], sun_uv[1],
                    shaft_strength_eff,
                    self.sun_shaft_decay,
                ],
                sun_shaft_color: [
                    self.sun_shaft_color[0],
                    self.sun_shaft_color[1],
                    self.sun_shaft_color[2],
                    0.0,
                ],
            };
            self.queue.write_buffer(&self.taa_uniform_buffer, 0, bytemuck::bytes_of(&tp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa_bg"),
                layout: &self.taa_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.taa_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.bloom_mip_views[0]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.ssao_rt_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.taa_views[taa_src_idx]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.ssr_rt_view) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.taa_views[taa_dst_idx],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.taa_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // Composite pass: tonemap (ACES + sRGB encode)
        // ============================================================
        let composite_src_view = if self.taa_enabled {
            &self.taa_views[taa_dst_idx]
        } else {
            &self.hdr_rt_view
        };

        // ============================================================
        // Auto-exposure update pass (runs only when auto_exposure is
        // on; otherwise the composite reads the old exposure texture
        // which is fine since manual_exposure bypasses the read).
        // ============================================================
        let exposure_src_idx = self.exposure_current_idx;
        let exposure_dst_idx = 1 - self.exposure_current_idx;
        if self.auto_exposure {
            let ep = ExposureParams {
                params: [
                    self.auto_exposure_key,
                    self.auto_exposure_rate,
                    // Min exposure needs to go low enough to handle
                    // bright outdoor HDRs where scene avg_luma is
                    // > 10. At min=0.1, a 10-luma ground saturates
                    // to 1.0 (white). 0.01 lets it crush to 0.1.
                    0.01,
                    10.0,
                ],
            };
            self.queue.write_buffer(&self.exposure_uniform_buffer, 0, bytemuck::bytes_of(&ep));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("exposure_bg"),
                layout: &self.exposure_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.exposure_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(composite_src_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.exposure_views[exposure_src_idx]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("exposure_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.exposure_views[exposure_dst_idx],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.exposure_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // composite_uniform_buffer carries per-frame composite state.
        // x = tonemap kind (0 ACES / 1 AgX)
        // y = auto-exposure toggle
        // z = manual exposure multiplier
        // w = auto-exposure target key value
        let cp = CompositeParams {
            params: [
                self.tonemap_kind as f32,
                if self.auto_exposure { 1.0 } else { 0.0 },
                self.manual_exposure,
                self.auto_exposure_key,
            ],
            filmic: [
                self.chromatic_aberration,
                self.vignette_strength,
                self.vignette_softness,
                self.grain_strength,
            ],
            misc: [self.taa_frame_index as f32, 0.0, 0.0, 0.0],
        };
        self.queue.write_buffer(&self.composite_uniform_buffer, 0, bytemuck::bytes_of(&cp));

        let composite_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bg"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(composite_src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.composite_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.exposure_views[exposure_dst_idx]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Composite covers the full surface anyway,
                        // but Clear is safer than Load (cheaper too —
                        // tile-based GPUs love Clear).
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // 2D pass: immediate-mode 2D geometry on top of composited image
        // ============================================================
        if has_2d {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_2d_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline_2d);
            pass.set_vertex_buffer(0, self.persistent_vb_2d.slice(..));
            pass.set_index_buffer(self.persistent_ib_2d.slice(..), wgpu::IndexFormat::Uint32);

            let num_calls = self.draw_calls_2d.len();
            for i in 0..num_calls {
                let call = &self.draw_calls_2d[i];
                let next_start = if i + 1 < num_calls {
                    self.draw_calls_2d[i + 1].index_start
                } else {
                    self.indices_2d.len() as u32
                };
                let count = next_start - call.index_start;
                if count == 0 { continue; }

                pass.set_bind_group(0, &self.uniform_bind_groups[call.uniform_idx as usize], &[]);
                if (call.texture_idx as usize) < self.texture_bind_groups.len() {
                    pass.set_bind_group(1, &self.texture_bind_groups[call.texture_idx as usize], &[]);
                }
                pass.draw_indexed(call.index_start..next_start, 0, 0..1);
            }
        }

        // If screenshot requested, copy rendered texture to staging buffer before submitting.
        // Synchronous GPU readback is not available on WASM (device.poll(Wait) blocks).
        #[cfg(not(target_arch = "wasm32"))]
        if self.screenshot_requested {
            // Use actual texture dimensions (accounts for Retina/DPI scaling)
            let tex_size = output.texture.size();
            let width = tex_size.width;
            let height = tex_size.height;
            let bytes_per_pixel = 4u32;
            let unpadded_bpr = width * bytes_per_pixel;
            let padded_bpr = (unpadded_bpr + 255) & !255;
            let buf_size = (padded_bpr * height) as u64;

            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("screenshot_staging"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bpr),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );

            self.queue.submit(std::iter::once(encoder.finish()));

            // Read back pixels synchronously
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.device.poll(wgpu::Maintain::Wait);

            if let Ok(Ok(())) = rx.recv() {
                let data = slice.get_mapped_range();
                let mut rgba = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
                for row in 0..height {
                    let start = (row * padded_bpr) as usize;
                    let end = start + (width * bytes_per_pixel) as usize;
                    rgba.extend_from_slice(&data[start..end]);
                }
                drop(data);
                // If the user requested an inline file write (via
                // bloom_take_screenshot), do that here. RGBA is in
                // BGRA order on Metal/DX12 surfaces — swap channels
                // before encoding to PNG so colors match what was on
                // screen rather than blue-and-red flipped.
                if let Some(path) = self.pending_screenshot_path.take() {
                    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
                    for chunk in rgba.chunks_exact(4) {
                        // BGRA → RGB. (Surface format is bgra8unorm
                        // on the platforms we care about today.)
                        rgb.push(chunk[2]);
                        rgb.push(chunk[1]);
                        rgb.push(chunk[0]);
                    }
                    if let Some(png) = encode_png_simple(width, height, &rgb) {
                        let _ = std::fs::write(&path, &png);
                    }
                }
                self.screenshot_data = Some((width, height, rgba));
            }
            staging.unmap();
            self.screenshot_requested = false;
        } else {
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        output.present();

        // After present: swap TAA ping-pong + advance the jitter
        // sequence so next frame's projection picks a new sub-pixel
        // offset and the just-written texture becomes the history.
        // Snapshot current VP into prev_vp so next frame's TAA pass
        // can reproject through it.
        if self.taa_enabled {
            self.taa_current_idx = 1 - self.taa_current_idx;
            self.taa_frame_index = self.taa_frame_index.wrapping_add(1);
            self.prev_vp_matrix = self.current_vp_matrix;
        }
        // Swap exposure ping-pong so next frame's exposure pass
        // reads what we just wrote.
        if self.auto_exposure {
            self.exposure_current_idx = 1 - self.exposure_current_idx;
        }
    }

    // ============================================================
    // Texture management
    // ============================================================

    // (encode_png_simple is defined as a free function below the impl
    // block so it can be reused by other capture paths if needed.)

    pub fn register_texture(&mut self, width: u32, height: u32, data: &[u8]) -> u32 {
        let max_dim = if width > height { width } else { height };
        let mip_count = (max_dim as f32).log2().floor() as u32 + 1;

        // Generate mip chain data (box filter downsampling)
        let mut mip_data = Vec::with_capacity(data.len() * 2); // overallocate
        mip_data.extend_from_slice(data); // level 0
        let mut mip_offsets = vec![0usize]; // byte offset of each mip level
        let mut mw = width;
        let mut mh = height;
        for _ in 1..mip_count {
            let prev_offset = *mip_offsets.last().unwrap();
            let pw = mw as usize; // previous width
            let ph = mh as usize; // previous height
            mw = if mw > 1 { mw / 2 } else { 1 };
            mh = if mh > 1 { mh / 2 } else { 1 };
            mip_offsets.push(mip_data.len());
            for y in 0..mh as usize {
                for x in 0..mw as usize {
                    let sx = x * 2;
                    let sy = y * 2;
                    let sx1 = (sx + 1).min(pw - 1);
                    let sy1 = (sy + 1).min(ph - 1);
                    for c in 0..4usize {
                        let p00 = mip_data[prev_offset + (sy * pw + sx) * 4 + c] as u32;
                        let p10 = mip_data[prev_offset + (sy * pw + sx1) * 4 + c] as u32;
                        let p01 = mip_data[prev_offset + (sy1 * pw + sx) * 4 + c] as u32;
                        let p11 = mip_data[prev_offset + (sy1 * pw + sx1) * 4 + c] as u32;
                        mip_data.push(((p00 + p10 + p01 + p11 + 2) / 4) as u8);
                    }
                }
            }
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("registered_texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each mip level
        let mut lw = width;
        let mut lh = height;
        for level in 0..mip_count {
            let offset = mip_offsets[level as usize];
            let level_size = (lw * lh * 4) as usize;
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mip_data[offset..offset + level_size],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * lw),
                    rows_per_image: Some(lh),
                },
                wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
            );
            lw = if lw > 1 { lw / 2 } else { 1 };
            lh = if lh > 1 { lh / 2 } else { 1 };
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        let idx = self.texture_bind_groups.len() as u32;
        self.texture_bind_groups.push(bind_group);
        self.textures.push(texture);
        self.texture_sizes.push((width, height));
        idx
    }

    pub fn update_texture(&mut self, idx: u32, width: u32, height: u32, data: &[u8]) {
        let i = idx as usize;
        if i >= self.textures.len() { return; }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.textures[i],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }

    pub fn unload_texture(&mut self, idx: u32) {
        // Mark as unused; bind group remains but won't be referenced
        let i = idx as usize;
        if i > 0 && i < self.textures.len() {
            self.texture_sizes[i] = (0, 0);
        }
    }

    pub fn set_texture_filter(&mut self, idx: u32, nearest: bool) {
        let i = idx as usize;
        if i >= self.textures.len() { return; }
        let view = self.textures[i].create_view(&wgpu::TextureViewDescriptor::default());
        let chosen_sampler = if nearest { &self.nearest_sampler } else { &self.sampler };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bg_refiltered"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(chosen_sampler) },
            ],
        });
        self.texture_bind_groups[i] = bind_group;
    }

    pub fn get_texture_width(&self, idx: u32) -> u32 {
        self.texture_sizes.get(idx as usize).map(|s| s.0).unwrap_or(0)
    }

    pub fn get_texture_height(&self, idx: u32) -> u32 {
        self.texture_sizes.get(idx as usize).map(|s| s.1).unwrap_or(0)
    }

    // ============================================================
    // 2D drawing internals
    // ============================================================

    fn ensure_draw_state(&mut self, texture_idx: u32) {
        let needs_new = self.draw_calls_2d.is_empty()
            || {
                let last = self.draw_calls_2d.last().unwrap();
                last.texture_idx != texture_idx || last.uniform_idx != self.current_uniform_idx
            };
        if needs_new {
            self.draw_calls_2d.push(DrawCall2D {
                texture_idx,
                uniform_idx: self.current_uniform_idx,
                index_start: self.indices_2d.len() as u32,
            });
        }
    }

    fn color_to_f32(r: f64, g: f64, b: f64, a: f64) -> [f32; 4] {
        [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32]
    }

    // ============================================================
    // 2D shape drawing (uses white texture at index 0)
    // ============================================================

    pub fn draw_rect(&mut self, x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32(r, g, b, a);
        // DEBUG: skip actual geometry - only red clear should show
        let _ = (x, y, w, h, color);
        return;
        let base = self.vertices_2d.len() as u32;
        let (x, y, w, h) = (x as f32, y as f32, w as f32, h as f32);

        self.vertices_2d.push(Vertex2D { position: [x, y], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x + w, y], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x + w, y + h], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x, y + h], uv: [0.0, 0.0], color });

        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn draw_rect_lines(&mut self, x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
        let t = thickness;
        self.draw_rect(x, y, w, t, r, g, b, a);
        self.draw_rect(x, y + h - t, w, t, r, g, b, a);
        self.draw_rect(x, y + t, t, h - 2.0 * t, r, g, b, a);
        self.draw_rect(x + w - t, y + t, t, h - 2.0 * t, r, g, b, a);
    }

    pub fn draw_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32(r, g, b, a);
        let dx = (x2 - x1) as f32;
        let dy = (y2 - y1) as f32;
        let len = (dx * dx + dy * dy).sqrt();
        if len == 0.0 { return; }
        let half_t = (thickness as f32) * 0.5;
        let nx = -dy / len * half_t;
        let ny = dx / len * half_t;
        let (x1, y1, x2, y2) = (x1 as f32, y1 as f32, x2 as f32, y2 as f32);
        let base = self.vertices_2d.len() as u32;

        self.vertices_2d.push(Vertex2D { position: [x1 + nx, y1 + ny], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x1 - nx, y1 - ny], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x2 - nx, y2 - ny], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x2 + nx, y2 + ny], uv: [0.0, 0.0], color });

        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn draw_circle(&mut self, cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32(r, g, b, a);
        let segments = 36u32;
        let base = self.vertices_2d.len() as u32;
        let (cx, cy, radius) = (cx as f32, cy as f32, radius as f32);

        self.vertices_2d.push(Vertex2D { position: [cx, cy], uv: [0.0, 0.0], color });
        for i in 0..segments {
            let angle = (i as f32) / (segments as f32) * std::f32::consts::TAU;
            self.vertices_2d.push(Vertex2D {
                position: [cx + radius * angle.cos(), cy + radius * angle.sin()],
                uv: [0.0, 0.0],
                color,
            });
        }
        for i in 0..segments {
            let next = if i + 1 < segments { i + 1 } else { 0 };
            self.indices_2d.extend_from_slice(&[base, base + 1 + i, base + 1 + next]);
        }
    }

    pub fn draw_circle_lines(&mut self, cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
        let segments = 36;
        for i in 0..segments {
            let a1 = (i as f64) / (segments as f64) * std::f64::consts::TAU;
            let a2 = ((i + 1) as f64) / (segments as f64) * std::f64::consts::TAU;
            self.draw_line(
                cx + radius * a1.cos(), cy + radius * a1.sin(),
                cx + radius * a2.cos(), cy + radius * a2.sin(),
                1.0, r, g, b, a,
            );
        }
    }

    pub fn draw_triangle(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32(r, g, b, a);
        let base = self.vertices_2d.len() as u32;

        self.vertices_2d.push(Vertex2D { position: [x1 as f32, y1 as f32], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x2 as f32, y2 as f32], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x3 as f32, y3 as f32], uv: [0.0, 0.0], color });

        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2]);
    }

    pub fn draw_poly(&mut self, cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32(r, g, b, a);
        let n = sides as u32;
        if n < 3 { return; }
        let base = self.vertices_2d.len() as u32;
        let (cx, cy, radius) = (cx as f32, cy as f32, radius as f32);
        let rot_rad = (rotation as f32).to_radians();

        self.vertices_2d.push(Vertex2D { position: [cx, cy], uv: [0.0, 0.0], color });
        for i in 0..n {
            let angle = rot_rad + (i as f32) / (n as f32) * std::f32::consts::TAU;
            self.vertices_2d.push(Vertex2D {
                position: [cx + radius * angle.cos(), cy + radius * angle.sin()],
                uv: [0.0, 0.0],
                color,
            });
        }
        for i in 0..n {
            let next = if i + 1 < n { i + 1 } else { 0 };
            self.indices_2d.extend_from_slice(&[base, base + 1 + i, base + 1 + next]);
        }
    }

    // ============================================================
    // Textured 2D drawing (for text atlas, sprites, etc.)
    // ============================================================

    pub fn draw_textured_quad(
        &mut self,
        x: f32, y: f32, w: f32, h: f32,
        u0: f32, v0: f32, u1: f32, v1: f32,
        color: [f32; 4],
        texture_idx: u32,
    ) {
        self.ensure_draw_state(texture_idx);
        let base = self.vertices_2d.len() as u32;
        self.vertices_2d.push(Vertex2D { position: [x, y], uv: [u0, v0], color });
        self.vertices_2d.push(Vertex2D { position: [x + w, y], uv: [u1, v0], color });
        self.vertices_2d.push(Vertex2D { position: [x + w, y + h], uv: [u1, v1], color });
        self.vertices_2d.push(Vertex2D { position: [x, y + h], uv: [u0, v1], color });
        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn draw_texture(&mut self, bind_group_idx: u32, x: f64, y: f64, tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64) {
        let (tw, th) = self.texture_sizes.get(bind_group_idx as usize).copied().unwrap_or((0, 0));
        if tw == 0 { return; }
        let color = Self::color_to_f32(tint_r, tint_g, tint_b, tint_a);
        self.draw_textured_quad(x as f32, y as f32, tw as f32, th as f32, 0.0, 0.0, 1.0, 1.0, color, bind_group_idx);
    }

    pub fn draw_texture_rec(
        &mut self, bind_group_idx: u32,
        src_x: f64, src_y: f64, src_w: f64, src_h: f64,
        dst_x: f64, dst_y: f64,
        tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
    ) {
        let (tw, th) = self.texture_sizes.get(bind_group_idx as usize).copied().unwrap_or((0, 0));
        if tw == 0 { return; }
        let color = Self::color_to_f32(tint_r, tint_g, tint_b, tint_a);
        let u0 = src_x as f32 / tw as f32;
        let v0 = src_y as f32 / th as f32;
        let u1 = (src_x + src_w) as f32 / tw as f32;
        let v1 = (src_y + src_h) as f32 / th as f32;
        self.draw_textured_quad(dst_x as f32, dst_y as f32, src_w as f32, src_h as f32, u0, v0, u1, v1, color, bind_group_idx);
    }

    pub fn draw_texture_pro(
        &mut self, bind_group_idx: u32,
        src_x: f64, src_y: f64, src_w: f64, src_h: f64,
        dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64,
        origin_x: f64, origin_y: f64, rotation: f64,
        tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
    ) {
        let (tw, th) = self.texture_sizes.get(bind_group_idx as usize).copied().unwrap_or((0, 0));
        if tw == 0 { return; }
        let color = Self::color_to_f32(tint_r, tint_g, tint_b, tint_a);
        let u0 = src_x as f32 / tw as f32;
        let v0 = src_y as f32 / th as f32;
        let u1 = (src_x + src_w) as f32 / tw as f32;
        let v1 = (src_y + src_h) as f32 / th as f32;

        let cos_r = (rotation as f32).to_radians().cos();
        let sin_r = (rotation as f32).to_radians().sin();
        let ox = origin_x as f32;
        let oy = origin_y as f32;
        let (dx, dy, dw, dh) = (dst_x as f32, dst_y as f32, dst_w as f32, dst_h as f32);

        let corners = [
            [dx - ox, dy - oy],
            [dx + dw - ox, dy - oy],
            [dx + dw - ox, dy + dh - oy],
            [dx - ox, dy + dh - oy],
        ];

        self.ensure_draw_state(bind_group_idx);
        let base = self.vertices_2d.len() as u32;
        let uvs = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
        for (c, uv) in corners.iter().zip(uvs.iter()) {
            let rx = c[0] * cos_r - c[1] * sin_r + ox;
            let ry = c[0] * sin_r + c[1] * cos_r + oy;
            self.vertices_2d.push(Vertex2D { position: [rx, ry], uv: *uv, color });
        }
        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    // ============================================================
    // Camera 2D
    // ============================================================

    pub fn begin_mode_2d(&mut self, offset_x: f32, offset_y: f32, target_x: f32, target_y: f32, rotation: f32, zoom: f32) {
        self.uniform_slot_count += 1;
        if self.uniform_slot_count >= MAX_UNIFORM_SLOTS { return; }
        self.current_uniform_idx = self.uniform_slot_count as u32;

        let cos_r = rotation.to_radians().cos();
        let sin_r = rotation.to_radians().sin();
        let tx = target_x;
        let ty = target_y;
        let view_proj: [[f32; 4]; 4] = [
            [zoom * cos_r, -zoom * sin_r, 0.0, 0.0],
            [zoom * sin_r,  zoom * cos_r, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [offset_x - zoom * (cos_r * tx + sin_r * ty),
             offset_y + zoom * (sin_r * tx - cos_r * ty),
             0.0, 1.0],
        ];

        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;
        let uniforms = Uniforms2D { screen_size: [w, h], _pad: [0.0; 2], view_proj };
        self.queue.write_buffer(
            &self.uniform_buffers[self.current_uniform_idx as usize],
            0,
            bytemuck::bytes_of(&uniforms),
        );
        self.render_mode = RenderMode::Mode2D;
    }

    pub fn end_mode_2d(&mut self) {
        self.current_uniform_idx = 0;
        self.render_mode = RenderMode::ScreenSpace;
    }

    // ============================================================
    // Camera 3D
    // ============================================================

    pub fn begin_mode_3d(
        &mut self,
        pos_x: f32, pos_y: f32, pos_z: f32,
        target_x: f32, target_y: f32, target_z: f32,
        up_x: f32, up_y: f32, up_z: f32,
        fovy: f32, projection: f32,
    ) {
        let aspect = self.surface_config.width as f32 / self.surface_config.height as f32;
        let mut proj = if projection < 0.5 {
            mat4_perspective(fovy.to_radians(), aspect, 0.01, 1000.0)
        } else {
            let top = fovy / 2.0;
            mat4_ortho(-top * aspect, top * aspect, -top, top, 0.01, 1000.0)
        };

        // TAA jitter: nudge the projection by a sub-pixel Halton
        // offset every frame. The TAA pass blends accumulated frames,
        // so this turns the jitter into per-pixel super-sampling.
        // Skipped when TAA is disabled to keep image stable.
        if self.taa_enabled {
            let i = (self.taa_frame_index % 16) + 1;
            let jx = halton(i, 2) - 0.5;
            let jy = halton(i, 3) - 0.5;
            let surface_w = self.surface_config.width.max(1) as f32;
            let surface_h = self.surface_config.height.max(1) as f32;
            // proj is column-major; column 2 row 0/1 are the
            // perspective / Z-coupling slots. Adding a constant NDC
            // offset there shifts the whole frustum by jitter px.
            proj[2][0] += (jx * 2.0) / surface_w;
            proj[2][1] += (jy * 2.0) / surface_h;
        }

        let view = mat4_look_at(
            [pos_x, pos_y, pos_z],
            [target_x, target_y, target_z],
            [up_x, up_y, up_z],
        );
        let vp = mat4_multiply(proj, view);
        self.current_vp_matrix = vp;
        self.current_view_matrix = view;
        self.current_proj_matrix = proj;
        self.current_camera_pos = [pos_x, pos_y, pos_z];

        // Mirror camera pos into lighting uniforms so the scene shader
        // can compute V for GGX specular. Preserve the .w slot — it
        // holds the env_intensity multiplier (set via load_env_from_hdr).
        let env_intensity_w = self.lighting_uniforms.camera_pos[3];
        self.lighting_uniforms.camera_pos = [pos_x, pos_y, pos_z, env_intensity_w];
        // Pass the current shadow VP (computed in end_frame_with_scene
        // based on the primary directional light direction) so the
        // scene shader's PCF sample lands on the right map.
        self.lighting_uniforms.shadow_light_vp = self.shadow_map.light_vp;
        self.queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_uniforms),
        );

        self.queue.write_buffer(
            &self.uniform_buffer_3d,
            0,
            bytemuck::bytes_of(&Uniforms3D { mvp: vp, model_tint: [1.0, 1.0, 1.0, 1.0] }),
        );
        self.render_mode = RenderMode::Mode3D;
    }

    pub fn end_mode_3d(&mut self) {
        self.render_mode = RenderMode::ScreenSpace;
    }

    // ============================================================
    // Joint matrices (GPU skinning)
    // ============================================================

    /// Set a single joint matrix for testing (joint_index 0-63, angle in radians around X axis)
    pub fn set_joint_test(&mut self, joint_index: usize, angle: f32) {
        if joint_index >= 128 { return; }
        let c = angle.cos();
        let s = angle.sin();
        // Rotation around X axis, column-major m[col][row]
        let mat: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],   // column 0
            [0.0,   c,   s, 0.0],   // column 1
            [0.0,  -s,   c, 0.0],   // column 2
            [0.0, 0.0, 0.0, 1.0],   // column 3
        ];
        self.queue.write_buffer(&self.joint_buffer, (joint_index * 64) as u64, bytemuck::cast_slice(&mat));
    }

    pub fn set_joint_matrices(&mut self, matrices: &[[[f32; 4]; 4]]) {
        self.pending_joint_matrices = Some(matrices.to_vec());
    }

    pub fn set_model_skin_scale(&mut self, scale: f32) {
        self.model_skin_scale = scale;
    }

    pub fn set_joint_matrices_scaled(&mut self, matrices: &[[[f32; 4]; 4]], scale: f32, position: [f32; 3], rot_sin: f32, rot_cos: f32) {
        let cos_r = rot_cos;
        let sin_r = rot_sin;
        let mut scaled = Vec::with_capacity(matrices.len());
        for m in matrices {
            let mut sm = *m;
            // Scale
            for col in 0..4 {
                sm[col][0] *= scale;
                sm[col][1] *= scale;
                sm[col][2] *= scale;
            }
            // Rotate around Y axis
            for col in 0..4 {
                let x = sm[col][0];
                let z = sm[col][2];
                sm[col][0] = cos_r * x + sin_r * z;
                sm[col][2] = -sin_r * x + cos_r * z;
            }
            // Translate
            sm[3][0] += position[0];
            sm[3][1] += position[1];
            sm[3][2] += position[2];
            scaled.push(sm);
        }

        self.pending_joint_matrices = Some(scaled);
    }

    /// Ensure persistent 3D buffers are large enough. Grows with doubling strategy.
    fn ensure_buffer_capacity_3d(&mut self, vb_bytes: usize, ib_bytes: usize) {
        if vb_bytes > self.persistent_vb_3d_capacity {
            let new_cap = vb_bytes.next_power_of_two();
            self.persistent_vb_3d = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_vb_3d"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.persistent_vb_3d_capacity = new_cap;
        }
        if ib_bytes > self.persistent_ib_3d_capacity {
            let new_cap = ib_bytes.next_power_of_two();
            self.persistent_ib_3d = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_ib_3d"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.persistent_ib_3d_capacity = new_cap;
        }
    }

    /// Ensure persistent 2D buffers are large enough. Grows with doubling strategy.
    fn ensure_buffer_capacity_2d(&mut self, vb_bytes: usize, ib_bytes: usize) {
        if vb_bytes > self.persistent_vb_2d_capacity {
            let new_cap = vb_bytes.next_power_of_two();
            self.persistent_vb_2d = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_vb_2d"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.persistent_vb_2d_capacity = new_cap;
        }
        if ib_bytes > self.persistent_ib_2d_capacity {
            let new_cap = ib_bytes.next_power_of_two();
            self.persistent_ib_2d = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_ib_2d"),
                size: new_cap as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.persistent_ib_2d_capacity = new_cap;
        }
    }

    // ============================================================
    // Cached model GPU buffers
    // ============================================================

    /// Check if a model's GPU buffers are cached (or marked uncacheable).
    pub fn is_model_in_cache(&self, handle_bits: u64) -> bool {
        self.model_gpu_cache.contains_key(&handle_bits)
    }

    /// Returns true if the model was cached successfully (static model).
    /// Returns false if the model is skinned (uncacheable).
    pub fn cache_model_if_static(&mut self, handle_bits: u64, meshes: &[crate::models::MeshData]) -> bool {
        if let Some(entry) = self.model_gpu_cache.get(&handle_bits) {
            return entry.is_some();
        }

        // Check if any vertex is skinned
        let is_skinned = meshes.iter().any(|m|
            m.vertices.iter().any(|v| v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01));

        if is_skinned {
            self.model_gpu_cache.insert(handle_bits, None);
            return false;
        }

        let gpu_meshes: Vec<GpuMesh> = meshes.iter().map(|mesh| {
            let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cached_model_vb"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cached_model_ib"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let base_color_idx = mesh.texture_idx.unwrap_or(0);
            let normal_idx = mesh.normal_texture_idx.unwrap_or(0);
            let mr_idx = mesh.metallic_roughness_texture_idx.unwrap_or(0);
            let em_idx = mesh.emissive_texture_idx.unwrap_or(0);
            let occ_idx = mesh.occlusion_texture_idx.unwrap_or(0);
            let material_uniform = self.create_scene_material_uniform(
                mesh.metallic_factor,
                mesh.roughness_factor,
                mesh.emissive_factor,
            );
            let material_bg = self.create_scene_material_bg(
                base_color_idx, normal_idx, mr_idx, em_idx, occ_idx, &material_uniform,
            );
            GpuMesh {
                vb,
                ib,
                index_count: mesh.indices.len() as u32,
                texture_idx: base_color_idx,
                material_bg,
                _material_uniform: material_uniform,
            }
        }).collect();

        self.model_gpu_cache.insert(handle_bits, Some(gpu_meshes));
        true
    }

    /// Record a cached model draw command. The actual rendering happens in end_frame().
    pub fn draw_model_cached(&mut self, handle_bits: u64, position: [f32; 3], scale: f32, tint: [f32; 4]) {
        let mesh_count = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes.len(),
            _ => return,
        };

        for mesh_idx in 0..mesh_count {
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;

            // Grow uniform pool if needed
            self.ensure_model_uniform_slot(slot);

            // Compute model MVP: VP * translate(position) * scale(s)
            let model_matrix = mat4_multiply(
                mat4_translate(IDENTITY_MAT4, position),
                mat4_scale(IDENTITY_MAT4, [scale, scale, scale]),
            );
            let model_mvp = mat4_multiply(self.current_vp_matrix, model_matrix);

            // Write uniform for this draw
            self.queue.write_buffer(
                &self.model_uniform_buffers[slot],
                0,
                bytemuck::bytes_of(&Uniforms3D { mvp: model_mvp, model_tint: tint }),
            );

            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
            });
        }
    }

    fn ensure_model_uniform_slot(&mut self, slot: usize) {
        while self.model_uniform_buffers.len() <= slot {
            let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("model_uniform"),
                contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4, model_tint: [1.0; 4] }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("model_uniform_bg"),
                layout: &self.uniform_3d_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            self.model_uniform_buffers.push(buf);
            self.model_uniform_bind_groups.push(bg);
        }
    }

    fn flush_joint_matrices(&mut self) {
        if let Some(ref matrices) = self.pending_joint_matrices {
            let count = matrices.len().min(127);
            let mut all_data = vec![[[0.0f32; 4]; 4]; 128];
            for i in 0..count {
                all_data[i] = matrices[i];
            }
            self.queue.write_buffer(&self.joint_buffer, 0, bytemuck::cast_slice(&all_data));
        }
        self.pending_joint_matrices = None;
    }

    // ============================================================
    // 3D texture tracking
    // ============================================================

    fn ensure_draw_state_3d(&mut self, texture_idx: u32) {
        let needs_new = self.draw_calls_3d.is_empty()
            || self.draw_calls_3d.last().unwrap().texture_idx != texture_idx;
        if needs_new {
            self.draw_calls_3d.push(DrawCall3D {
                texture_idx,
                index_start: self.indices_3d.len() as u32,
            });
        }
    }

    pub fn set_texture_3d(&mut self, texture_idx: u32) {
        self.current_texture_3d = texture_idx;
    }

    // ============================================================
    // Lighting
    // ============================================================

    pub fn set_ambient_light(&mut self, r: f64, g: f64, b: f64, intensity: f64) {
        self.lighting_uniforms.ambient = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, intensity as f32];
        self.queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&self.lighting_uniforms));
    }

    pub fn set_directional_light(&mut self, dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
        self.lighting_uniforms.light_dir = [dx as f32, dy as f32, dz as f32, intensity as f32];
        self.lighting_uniforms.light_color = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, 0.0];
        self.queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&self.lighting_uniforms));
    }

    /// Add an additional directional light (up to MAX_DIR_LIGHTS).
    /// Color is 0-1 range (not 0-255).
    pub fn add_directional_light(&mut self, dx: f32, dy: f32, dz: f32, r: f32, g: f32, b: f32, intensity: f32) {
        let idx = self.lighting_uniforms.dir_light_count[0] as usize;
        if idx >= MAX_DIR_LIGHTS { return; }
        self.lighting_uniforms.dir_lights[idx] = DirLight {
            direction: [dx, dy, dz, intensity],
            color: [r, g, b, 0.0],
        };
        self.lighting_uniforms.dir_light_count[0] = (idx + 1) as f32;
        self.queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&self.lighting_uniforms));
    }

    /// Add a point light (up to MAX_POINT_LIGHTS).
    /// Color is 0-1 range.
    pub fn add_point_light(&mut self, x: f32, y: f32, z: f32, range: f32, r: f32, g: f32, b: f32, intensity: f32) {
        let idx = self.lighting_uniforms.point_light_count[0] as usize;
        if idx >= MAX_POINT_LIGHTS { return; }
        self.lighting_uniforms.point_lights[idx] = PointLight {
            position: [x, y, z, range],
            color: [r, g, b, intensity],
        };
        self.lighting_uniforms.point_light_count[0] = (idx + 1) as f32;
        self.queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&self.lighting_uniforms));
    }

    /// Clear all additional lights (called at begin_frame).
    pub fn clear_additional_lights(&mut self) {
        self.lighting_uniforms.dir_light_count = [0.0; 4];
        self.lighting_uniforms.point_light_count = [0.0; 4];
    }

    // ============================================================
    // 3D drawing
    // ============================================================

    fn add_line_3d(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4], thickness: f32) {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        let len = (dx*dx + dy*dy + dz*dz).sqrt();
        if len < 0.0001 { return; }
        let (dx, dy, dz) = (dx/len, dy/len, dz/len);

        // Find perpendicular using cross product with best reference axis
        let (px, py, pz) = if dy.abs() > 0.9 {
            // Cross with X axis: (0, dz, -dy)
            (0.0, dz, -dy)
        } else {
            // Cross with Y axis: (-dz, 0, dx)
            (-dz, 0.0, dx)
        };
        let plen = (px*px + py*py + pz*pz).sqrt();
        let ht = thickness * 0.5;
        let (px, py, pz) = (px/plen * ht, py/plen * ht, pz/plen * ht);
        let normal = [px/ht, py/ht, pz/ht];

        let base = self.vertices_3d.len() as u32;
        self.vertices_3d.push(Vertex3D { position: [start[0]+px, start[1]+py, start[2]+pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [start[0]-px, start[1]-py, start[2]-pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [end[0]-px, end[1]-py, end[2]-pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [end[0]+px, end[1]+py, end[2]+pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.indices_3d.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    pub fn draw_cube(&mut self, x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let (hw, hh, hd) = (w as f32 * 0.5, h as f32 * 0.5, d as f32 * 0.5);

        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            ([0.0, 0.0, -1.0], [[x-hw,y-hh,z-hd],[x+hw,y-hh,z-hd],[x+hw,y+hh,z-hd],[x-hw,y+hh,z-hd]]), // front
            ([0.0, 0.0, 1.0],  [[x+hw,y-hh,z+hd],[x-hw,y-hh,z+hd],[x-hw,y+hh,z+hd],[x+hw,y+hh,z+hd]]), // back
            ([-1.0, 0.0, 0.0], [[x-hw,y-hh,z+hd],[x-hw,y-hh,z-hd],[x-hw,y+hh,z-hd],[x-hw,y+hh,z+hd]]), // left
            ([1.0, 0.0, 0.0],  [[x+hw,y-hh,z-hd],[x+hw,y-hh,z+hd],[x+hw,y+hh,z+hd],[x+hw,y+hh,z-hd]]), // right
            ([0.0, 1.0, 0.0],  [[x-hw,y+hh,z-hd],[x+hw,y+hh,z-hd],[x+hw,y+hh,z+hd],[x-hw,y+hh,z+hd]]), // top
            ([0.0, -1.0, 0.0], [[x-hw,y-hh,z+hd],[x+hw,y-hh,z+hd],[x+hw,y-hh,z-hd],[x-hw,y-hh,z-hd]]), // bottom
        ];

        for (normal, verts) in &faces {
            let base = self.vertices_3d.len() as u32;
            for v in verts {
                self.vertices_3d.push(Vertex3D { position: *v, normal: *normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            }
            self.indices_3d.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
        }
    }

    pub fn draw_cube_wires(&mut self, x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
        let color = Self::color_to_f32(r, g, b, a);
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let (hw, hh, hd) = (w as f32 * 0.5, h as f32 * 0.5, d as f32 * 0.5);
        let t = 0.02f32;

        let corners = [
            [x-hw,y-hh,z-hd],[x+hw,y-hh,z-hd],[x+hw,y+hh,z-hd],[x-hw,y+hh,z-hd],
            [x-hw,y-hh,z+hd],[x+hw,y-hh,z+hd],[x+hw,y+hh,z+hd],[x-hw,y+hh,z+hd],
        ];
        let edges = [
            (0,1),(1,2),(2,3),(3,0), // front
            (4,5),(5,6),(6,7),(7,4), // back
            (0,4),(1,5),(2,6),(3,7), // connecting
        ];
        for (a_idx, b_idx) in &edges {
            self.add_line_3d(corners[*a_idx], corners[*b_idx], color, t);
        }
    }

    pub fn draw_sphere(&mut self, cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (cx, cy, cz, radius) = (cx as f32, cy as f32, cz as f32, radius as f32);
        let rings = 8u32;
        let slices = 8u32;

        for i in 0..rings {
            let theta1 = (i as f32) / (rings as f32) * std::f32::consts::PI;
            let theta2 = ((i + 1) as f32) / (rings as f32) * std::f32::consts::PI;
            for j in 0..slices {
                let phi1 = (j as f32) / (slices as f32) * std::f32::consts::TAU;
                let phi2 = ((j + 1) as f32) / (slices as f32) * std::f32::consts::TAU;

                let p = |theta: f32, phi: f32| -> ([f32; 3], [f32; 3]) {
                    let nx = theta.sin() * phi.cos();
                    let ny = theta.cos();
                    let nz = theta.sin() * phi.sin();
                    ([cx + radius * nx, cy + radius * ny, cz + radius * nz], [nx, ny, nz])
                };

                let (p00, n00) = p(theta1, phi1);
                let (p10, n10) = p(theta2, phi1);
                let (p11, n11) = p(theta2, phi2);
                let (p01, n01) = p(theta1, phi2);

                let base = self.vertices_3d.len() as u32;
                self.vertices_3d.push(Vertex3D { position: p00, normal: n00, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
                self.vertices_3d.push(Vertex3D { position: p10, normal: n10, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
                self.vertices_3d.push(Vertex3D { position: p11, normal: n11, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
                self.vertices_3d.push(Vertex3D { position: p01, normal: n01, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
                self.indices_3d.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
            }
        }
    }

    pub fn draw_sphere_wires(&mut self, cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
        let color = Self::color_to_f32(r, g, b, a);
        let (cx, cy, cz, radius) = (cx as f32, cy as f32, cz as f32, radius as f32);
        let segments = 16u32;

        for i in 0..segments {
            let a1 = (i as f32) / (segments as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32) / (segments as f32) * std::f32::consts::TAU;
            // XY ring
            self.add_line_3d(
                [cx + radius * a1.cos(), cy + radius * a1.sin(), cz],
                [cx + radius * a2.cos(), cy + radius * a2.sin(), cz],
                color, 0.02,
            );
            // XZ ring
            self.add_line_3d(
                [cx + radius * a1.cos(), cy, cz + radius * a1.sin()],
                [cx + radius * a2.cos(), cy, cz + radius * a2.sin()],
                color, 0.02,
            );
            // YZ ring
            self.add_line_3d(
                [cx, cy + radius * a1.cos(), cz + radius * a1.sin()],
                [cx, cy + radius * a2.cos(), cz + radius * a2.sin()],
                color, 0.02,
            );
        }
    }

    pub fn draw_cylinder(&mut self, x: f64, y: f64, z: f64, radius_top: f64, radius_bottom: f64, height: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let (rt, rb, h) = (radius_top as f32, radius_bottom as f32, height as f32);
        let slices = 16u32;

        for i in 0..slices {
            let a1 = (i as f32) / (slices as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32) / (slices as f32) * std::f32::consts::TAU;
            let (c1, s1) = (a1.cos(), a1.sin());
            let (c2, s2) = (a2.cos(), a2.sin());

            // Side face
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D { position: [x + rb*c1, y, z + rb*s1], normal: [c1, 0.0, s1], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x + rb*c2, y, z + rb*s2], normal: [c2, 0.0, s2], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x + rt*c2, y+h, z + rt*s2], normal: [c2, 0.0, s2], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x + rt*c1, y+h, z + rt*s1], normal: [c1, 0.0, s1], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.indices_3d.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

            // Top cap
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D { position: [x, y+h, z], normal: [0.0, 1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rt*c1, y+h, z+rt*s1], normal: [0.0, 1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rt*c2, y+h, z+rt*s2], normal: [0.0, 1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.indices_3d.extend_from_slice(&[base, base+1, base+2]);

            // Bottom cap
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D { position: [x, y, z], normal: [0.0, -1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rb*c2, y, z+rb*s2], normal: [0.0, -1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rb*c1, y, z+rb*s1], normal: [0.0, -1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
            self.indices_3d.extend_from_slice(&[base, base+1, base+2]);
        }
    }

    pub fn draw_plane(&mut self, cx: f64, cy: f64, cz: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (cx, cy, cz) = (cx as f32, cy as f32, cz as f32);
        let (hw, hd) = (w as f32 * 0.5, d as f32 * 0.5);
        let normal = [0.0f32, 1.0, 0.0];

        let base = self.vertices_3d.len() as u32;
        self.vertices_3d.push(Vertex3D { position: [cx-hw, cy, cz-hd], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [cx+hw, cy, cz-hd], normal, color, uv: [1.0, 0.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [cx+hw, cy, cz+hd], normal, color, uv: [1.0, 1.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [cx-hw, cy, cz+hd], normal, color, uv: [0.0, 1.0], joints: [0.0; 4], weights: [0.0; 4], tangent: [0.0; 4] });
        self.indices_3d.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    pub fn draw_grid(&mut self, slices: i32, spacing: f64) {
        let color = [0.5f32, 0.5, 0.5, 1.0];
        let spacing = spacing as f32;
        let half = slices as f32 * spacing / 2.0;

        for i in 0..=slices {
            let pos = -half + i as f32 * spacing;
            self.add_line_3d([-half, 0.0, pos], [half, 0.0, pos], color, 0.01);
            self.add_line_3d([pos, 0.0, -half], [pos, 0.0, half], color, 0.01);
        }
    }

    pub fn draw_ray(&mut self, origin_x: f64, origin_y: f64, origin_z: f64, dir_x: f64, dir_y: f64, dir_z: f64, r: f64, g: f64, b: f64, a: f64) {
        let color = Self::color_to_f32(r, g, b, a);
        let start = [origin_x as f32, origin_y as f32, origin_z as f32];
        let end = [(origin_x + dir_x) as f32, (origin_y + dir_y) as f32, (origin_z + dir_z) as f32];
        self.add_line_3d(start, end, color, 0.02);
    }

    pub fn draw_model_mesh(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32) {
        self.draw_model_mesh_tinted(vertices, indices, position, scale, [1.0, 1.0, 1.0, 1.0], 0);
    }

    pub fn draw_model_mesh_tinted(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32, tint: [f32; 4], texture_idx: u32) {
        self.ensure_draw_state_3d(texture_idx);
        let base = self.vertices_3d.len() as u32;
        for v in vertices {
            // Check if vertex is skinned (has non-zero weights)
            let is_skinned = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01;
            let pos = if is_skinned {
                // Skinned: pass raw bind-pose positions — joint matrices handle transform
                v.position
            } else {
                // Unskinned: apply CPU-side position + scale
                [v.position[0] * scale + position[0],
                 v.position[1] * scale + position[1],
                 v.position[2] * scale + position[2]]
            };
            self.vertices_3d.push(Vertex3D {
                position: pos,
                normal: v.normal,
                color: [
                    v.color[0] * tint[0],
                    v.color[1] * tint[1],
                    v.color[2] * tint[2],
                    v.color[3] * tint[3],
                ],
                uv: v.uv,
                joints: v.joints,
                weights: v.weights,
                tangent: v.tangent,
            });
        }
        for &idx in indices {
            self.indices_3d.push(base + idx);
        }
    }

    // ============================================================
    // Queries
    // ============================================================

    pub fn width(&self) -> u32 {
        self.surface_config.width
    }

    pub fn height(&self) -> u32 {
        self.surface_config.height
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    /// Capture the current framebuffer as RGBA pixels.
    /// Returns (width, height, rgba_data). Call after end_frame.
    /// Not available on WASM (requires synchronous GPU readback).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn capture_screenshot(&self) -> Option<(u32, u32, Vec<u8>)> {
        let width = self.surface_config.width;
        let height = self.surface_config.height;
        let bytes_per_pixel = 4u32;
        // wgpu requires rows aligned to 256 bytes
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        // Render one frame to a texture we can copy from
        let output = self.surface.get_current_texture().ok()?;
        let texture = &output.texture;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("screenshot_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and read pixels
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        if rx.recv().ok()?.is_err() {
            return None;
        }

        let data = buffer_slice.get_mapped_range();
        // Remove row padding
        let mut rgba = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (width * bytes_per_pixel) as usize;
            rgba.extend_from_slice(&data[start..end]);
        }
        drop(data);
        staging_buffer.unmap();
        output.present();

        Some((width, height, rgba))
    }

    /// Returns true if vsync is active (Fifo or FifoRelaxed present mode).
    pub fn vsync_active(&self) -> bool {
        matches!(self.surface_config.present_mode,
            wgpu::PresentMode::Fifo | wgpu::PresentMode::FifoRelaxed)
    }

    pub fn load_custom_shader(&mut self, wgsl_source: &str) -> usize {
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("custom_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        // Create layout matching the default 3D pipeline
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("custom_shader_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("custom_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("custom_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main_3d"),
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main_3d"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.custom_pipelines.push(pipeline);
        self.custom_pipelines.len() // 1-based index
    }
}

// ============================================================
// Matrix math helpers (column-major for WGSL)
// ============================================================

pub fn mat4_perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy / 2.0).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}

pub fn mat4_ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let lr = 1.0 / (left - right);
    let bt = 1.0 / (bottom - top);
    let nf = 1.0 / (near - far);
    [
        [-2.0 * lr, 0.0, 0.0, 0.0],
        [0.0, -2.0 * bt, 0.0, 0.0],
        [0.0, 0.0, 2.0 * nf, 0.0],
        [(left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1.0],
    ]
}

pub fn mat4_look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let fx = center[0] - eye[0];
    let fy = center[1] - eye[1];
    let fz = center[2] - eye[2];
    let flen = (fx*fx + fy*fy + fz*fz).sqrt();
    let (fx, fy, fz) = (fx/flen, fy/flen, fz/flen);

    let sx = fy * up[2] - fz * up[1];
    let sy = fz * up[0] - fx * up[2];
    let sz = fx * up[1] - fy * up[0];
    let slen = (sx*sx + sy*sy + sz*sz).sqrt();
    let (sx, sy, sz) = (sx/slen, sy/slen, sz/slen);

    let ux = sy * fz - sz * fy;
    let uy = sz * fx - sx * fz;
    let uz = sx * fy - sy * fx;

    [
        [sx, ux, -fx, 0.0],
        [sy, uy, -fy, 0.0],
        [sz, uz, -fz, 0.0],
        [-(sx*eye[0]+sy*eye[1]+sz*eye[2]), -(ux*eye[0]+uy*eye[1]+uz*eye[2]), fx*eye[0]+fy*eye[1]+fz*eye[2], 1.0],
    ]
}

pub fn mat4_multiply(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row]*b[col][0] + a[1][row]*b[col][1] + a[2][row]*b[col][2] + a[3][row]*b[col][3];
        }
    }
    out
}

/// Multiply a column-major 4x4 matrix by a column vector.
pub fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
    [
        m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2] + m[3][0]*v[3],
        m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2] + m[3][1]*v[3],
        m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2] + m[3][2]*v[3],
        m[0][3]*v[0] + m[1][3]*v[1] + m[2][3]*v[2] + m[3][3]*v[3],
    ]
}

pub fn mat4_translate(m: [[f32; 4]; 4], v: [f32; 3]) -> [[f32; 4]; 4] {
    let mut out = m;
    for i in 0..4 {
        out[3][i] += m[0][i]*v[0] + m[1][i]*v[1] + m[2][i]*v[2];
    }
    out
}

pub fn mat4_scale(m: [[f32; 4]; 4], v: [f32; 3]) -> [[f32; 4]; 4] {
    let mut out = m;
    for i in 0..4 { out[0][i] *= v[0]; }
    for i in 0..4 { out[1][i] *= v[1]; }
    for i in 0..4 { out[2][i] *= v[2]; }
    out
}

pub fn mat4_invert(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let m = |r: usize, c: usize| m[c][r]; // accessor for row-major style
    let mut inv = [0.0f32; 16];
    inv[0]  =  m(1,1)*m(2,2)*m(3,3) - m(1,1)*m(2,3)*m(3,2) - m(2,1)*m(1,2)*m(3,3) + m(2,1)*m(1,3)*m(3,2) + m(3,1)*m(1,2)*m(2,3) - m(3,1)*m(1,3)*m(2,2);
    inv[4]  = -m(1,0)*m(2,2)*m(3,3) + m(1,0)*m(2,3)*m(3,2) + m(2,0)*m(1,2)*m(3,3) - m(2,0)*m(1,3)*m(3,2) - m(3,0)*m(1,2)*m(2,3) + m(3,0)*m(1,3)*m(2,2);
    inv[8]  =  m(1,0)*m(2,1)*m(3,3) - m(1,0)*m(2,3)*m(3,1) - m(2,0)*m(1,1)*m(3,3) + m(2,0)*m(1,3)*m(3,1) + m(3,0)*m(1,1)*m(2,3) - m(3,0)*m(1,3)*m(2,1);
    inv[12] = -m(1,0)*m(2,1)*m(3,2) + m(1,0)*m(2,2)*m(3,1) + m(2,0)*m(1,1)*m(3,2) - m(2,0)*m(1,2)*m(3,1) - m(3,0)*m(1,1)*m(2,2) + m(3,0)*m(1,2)*m(2,1);
    inv[1]  = -m(0,1)*m(2,2)*m(3,3) + m(0,1)*m(2,3)*m(3,2) + m(2,1)*m(0,2)*m(3,3) - m(2,1)*m(0,3)*m(3,2) - m(3,1)*m(0,2)*m(2,3) + m(3,1)*m(0,3)*m(2,2);
    inv[5]  =  m(0,0)*m(2,2)*m(3,3) - m(0,0)*m(2,3)*m(3,2) - m(2,0)*m(0,2)*m(3,3) + m(2,0)*m(0,3)*m(3,2) + m(3,0)*m(0,2)*m(2,3) - m(3,0)*m(0,3)*m(2,2);
    inv[9]  = -m(0,0)*m(2,1)*m(3,3) + m(0,0)*m(2,3)*m(3,1) + m(2,0)*m(0,1)*m(3,3) - m(2,0)*m(0,3)*m(3,1) - m(3,0)*m(0,1)*m(2,3) + m(3,0)*m(0,3)*m(2,1);
    inv[13] =  m(0,0)*m(2,1)*m(3,2) - m(0,0)*m(2,2)*m(3,1) - m(2,0)*m(0,1)*m(3,2) + m(2,0)*m(0,2)*m(3,1) + m(3,0)*m(0,1)*m(2,2) - m(3,0)*m(0,2)*m(2,1);
    inv[2]  =  m(0,1)*m(1,2)*m(3,3) - m(0,1)*m(1,3)*m(3,2) - m(1,1)*m(0,2)*m(3,3) + m(1,1)*m(0,3)*m(3,2) + m(3,1)*m(0,2)*m(1,3) - m(3,1)*m(0,3)*m(1,2);
    inv[6]  = -m(0,0)*m(1,2)*m(3,3) + m(0,0)*m(1,3)*m(3,2) + m(1,0)*m(0,2)*m(3,3) - m(1,0)*m(0,3)*m(3,2) - m(3,0)*m(0,2)*m(1,3) + m(3,0)*m(0,3)*m(1,2);
    inv[10] =  m(0,0)*m(1,1)*m(3,3) - m(0,0)*m(1,3)*m(3,1) - m(1,0)*m(0,1)*m(3,3) + m(1,0)*m(0,3)*m(3,1) + m(3,0)*m(0,1)*m(1,3) - m(3,0)*m(0,3)*m(1,1);
    inv[14] = -m(0,0)*m(1,1)*m(3,2) + m(0,0)*m(1,2)*m(3,1) + m(1,0)*m(0,1)*m(3,2) - m(1,0)*m(0,2)*m(3,1) - m(3,0)*m(0,1)*m(1,2) + m(3,0)*m(0,2)*m(1,1);
    inv[3]  = -m(0,1)*m(1,2)*m(2,3) + m(0,1)*m(1,3)*m(2,2) + m(1,1)*m(0,2)*m(2,3) - m(1,1)*m(0,3)*m(2,2) - m(2,1)*m(0,2)*m(1,3) + m(2,1)*m(0,3)*m(1,2);
    inv[7]  =  m(0,0)*m(1,2)*m(2,3) - m(0,0)*m(1,3)*m(2,2) - m(1,0)*m(0,2)*m(2,3) + m(1,0)*m(0,3)*m(2,2) + m(2,0)*m(0,2)*m(1,3) - m(2,0)*m(0,3)*m(1,2);
    inv[11] = -m(0,0)*m(1,1)*m(2,3) + m(0,0)*m(1,3)*m(2,1) + m(1,0)*m(0,1)*m(2,3) - m(1,0)*m(0,3)*m(2,1) - m(2,0)*m(0,1)*m(1,3) + m(2,0)*m(0,3)*m(1,1);
    inv[15] =  m(0,0)*m(1,1)*m(2,2) - m(0,0)*m(1,2)*m(2,1) - m(1,0)*m(0,1)*m(2,2) + m(1,0)*m(0,2)*m(2,1) + m(2,0)*m(0,1)*m(1,2) - m(2,0)*m(0,2)*m(1,1);

    let det = m(0,0)*inv[0] + m(0,1)*inv[4] + m(0,2)*inv[8] + m(0,3)*inv[12];
    if det.abs() < 1e-10 { return IDENTITY_MAT4; }
    let inv_det = 1.0 / det;
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = inv[col * 4 + row] * inv_det;
        }
    }
    out
}

/// Encode an RGB byte buffer (no alpha) as a PNG. Used by the
/// pending-screenshot path so callers can hand us a path and get a
/// PNG written to disk without worrying about cross-FFI buffer handoff.
fn encode_png_simple(width: u32, height: u32, rgb: &[u8]) -> Option<Vec<u8>> {
    use image::{ImageBuffer, Rgb};
    let buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb.to_vec())?;
    let mut out = std::io::Cursor::new(Vec::new());
    buf.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}
