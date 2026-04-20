use wgpu::util::DeviceExt;
use std::collections::HashMap;

mod shaders;
use shaders::*;

mod util;
pub use util::{
    IDENTITY_MAT4,
    mat4_perspective, mat4_ortho, mat4_look_at,
    mat4_multiply, mat4_mul_vec4,
    mat4_translate, mat4_scale, mat4_invert,
};
use util::encode_png_simple;

mod brdf_lut;
use brdf_lut::build_brdf_lut;


// ============================================================
// Constants
// ============================================================

const MAX_UNIFORM_SLOTS: usize = 8;

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
    model: [[f32; 4]; 4],
    prev_mvp: [[f32; 4]; 4],
    model_tint: [f32; 4],
}

/// Scene-pipeline per-material factors — the scalar parts of a glTF
/// PBR material that get multiplied onto the corresponding texture
/// samples. Sized to a multiple of 16 bytes for UBO alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneMaterialUniforms {
    /// x = metallic_factor, y = roughness_factor,
    /// z = has_mr_texture (1.0 = sample mr_tex and multiply, 0.0 = ignore
    ///     mr_tex and use factors directly),
    /// w = alpha_cutoff (0.0 = OPAQUE mode, >0 = MASK/BLEND — fragments
    ///     whose base-colour alpha is below this are discarded).
    pub metal_rough: [f32; 4],
    /// rgb = emissive_factor, w = padding
    pub emissive: [f32; 4],
}

impl SceneMaterialUniforms {
    pub fn new(
        metallic: f32,
        roughness: f32,
        emissive: [f32; 3],
        has_mr_texture: bool,
        alpha_cutoff: f32,
    ) -> Self {
        Self {
            metal_rough: [
                metallic,
                roughness,
                if has_mr_texture { 1.0 } else { 0.0 },
                alpha_cutoff,
            ],
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
    /// Cascaded shadow map: 3 light view-projection matrices (one per
    /// cascade). Scene shader selects the tightest cascade based on
    /// the fragment's view-space depth and projects through the
    /// corresponding matrix for shadow-map UV.
    shadow_cascade_vps: [[[f32; 4]; 4]; 3],
    /// View-space Z split distances for cascade selection (xyz = split
    /// distances for cascades 0/1/2, w = unused). Fragment at depth z
    /// uses cascade i where z <= cascade_splits[i].
    shadow_cascade_splits: [f32; 4],
    /// Camera view matrix — passed to the shader so the fragment shader
    /// can compute view-space Z for cascade selection without an extra
    /// buffer binding.
    shadow_view_matrix: [[f32; 4]; 4],
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
            shadow_cascade_vps: [IDENTITY_MAT4; 3],
            shadow_cascade_splits: [8.0, 25.0, 80.0, 0.0],
            shadow_view_matrix: IDENTITY_MAT4,
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

/// SSAO RT layout: R = GTAO occlusion (bilaterally blurred), G =
/// contact-shadow factor (passed through blur unchanged so the fine-
/// detail ray-march result survives). Rgba8Unorm because WebGPU
/// requires `rgba8unorm` for storage-texture writes by default —
/// the compute GTAO pass (SSAO_SHADER_WGSL) uses `textureStore`.
/// Extra two channels left 0; downstream samplers only read .r/.g,
/// so the only cost is 4 B/px vs 2 B/px at half-res
/// (~180 kB extra on a 1600×900 surface).
const SSAO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Material G-buffer format. Rg8Unorm: R = metallic, G = roughness.
/// Written as a second color attachment in the HDR pass; SSR (and
/// any future deferred passes) reads it for per-pixel material info.
const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

/// Linear-depth Hi-Z pyramid format. R32Float (not R16Float) because
/// WebGPU only mandates r32-family formats for single-channel storage
/// textures. The pyramid stores *positive* view-space distance
/// (|view_z|) so compute GTAO skips per-sample linearization. Sky
/// pixels get `HIZ_SKY_Z` (10 000) and the downsample uses `min` so
/// any near-field geometry in a tile dominates surrounding sky.
const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Number of mip levels in the linear-depth Hi-Z pyramid. 5 covers
/// a 16-pixel-radius footprint at the coarsest mip — enough for
/// the 0.25 UV clamp SSAO uses (~100 px at half-res 400-wide).
/// One linearize pass plus `HIZ_MIP_COUNT - 1` downsample passes.
const HIZ_MIP_COUNT: u32 = 5;

/// Velocity buffer format. Rg16Float: two-channel 16-bit float for
/// sub-pixel precision screen-space velocity. Written as a third
/// color attachment in the HDR pass; motion blur and TAA read it.
const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

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

/// Create the albedo G-buffer (Rgba8Unorm, surface size). Written by
/// the scene pass so post-passes can modulate bounce light by the
/// receiving surface's diffuse albedo (SSGI etc.).
fn create_albedo_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("albedo_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the composed HDR render target. Scene HDR + SSR + SSGI *
/// albedo + bloom + fog + sun shafts all merged into one texture by
/// the `scene_compose` pass. Both the TAA-on path (TAA consumes this
/// as its "current frame" input) and the TAA-off path (composite
/// reads it directly) read from the same buffer, so fog / shafts /
/// post-effects stay visible regardless of TAA state.
fn create_composed_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composed_rt"),
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

/// Create the velocity render target (Rg16Float, surface size).
/// Per-pixel screen-space velocity for motion blur and TAA.
fn create_velocity_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("velocity_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: VELOCITY_FORMAT,
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

/// Create the SSGI render target (half-res HDR — indirect diffuse bounce light).
/// Same half-res HDR strategy as SSR: keeps the per-pixel ray march cheap
/// while still providing enough color resolution for colored bounce light.
fn create_ssgi_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssgi_rt"),
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

/// Create the SSGI temporal history textures (ping-pong pair, same
/// format/size as ssgi_rt — half-res HDR). Returns two textures and
/// their views.
fn create_ssgi_history_textures(
    device: &wgpu::Device, width: u32, height: u32,
) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let make = || {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssgi_history"),
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
    };
    let (t0, v0) = make();
    let (t1, v1) = make();
    ([t0, t1], [v0, v1])
}

/// Create the DoF render target (full-res HDR, same format as TAA output).
/// DoF reads the TAA output + depth, writes the blurred result here.
/// Composite then reads this instead of the TAA output.
fn create_dof_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dof_rt"),
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

/// Create the SSS render target (full-res HDR, same format as DoF/motion-blur).
fn create_sss_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sss_rt"),
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

/// Create the SSAO render target. Written by the compute GTAO pass
/// (via `STORAGE_BINDING`) and sampled by the bilateral blur +
/// downstream passes.
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
        usage: wgpu::TextureUsages::STORAGE_BINDING
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the SSAO bilateral-blur render target (same format/size as ssao_rt).
fn create_ssao_blur_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao_blur_rt"),
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

/// Build the linear-depth Hi-Z pyramid as `HIZ_MIP_COUNT` separate
/// single-mip textures. One multi-mip texture is cheaper on paper
/// but Metal's per-subresource state tracking trips when wgpu
/// writes one mip and samples another in the same encoder — the
/// bloom chain has the same issue and uses this same layout.
fn create_linear_depth_hiz_chain(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>) {
    let mut textures = Vec::with_capacity(HIZ_MIP_COUNT as usize);
    let mut views = Vec::with_capacity(HIZ_MIP_COUNT as usize);
    for i in 0..HIZ_MIP_COUNT {
        let w = ((width / 2) >> i).max(1);
        let h = ((height / 2) >> i).max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("linear_depth_hiz_mip"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        textures.push(tex);
        views.push(view);
    }
    (textures, views)
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HizLinearizeParams {
    /// xy = inv_size, z = proj[2][2], w = proj[3][2]
    params: [f32; 4],
    /// xy = mip-0 size, zw unused
    size: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HizDownsampleParams {
    /// xy = dst-mip size, zw unused
    size: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoParams {
    /// xy = inv_size (1/half_w, 1/half_h), z = radius (world units),
    /// w = strength
    params: [f32; 4],
    /// x = proj[0][0], y = proj[1][1], z = proj[2][0] (TAA jitter),
    /// w = proj[2][1] (TAA jitter). Column-major: proj[col][row].
    proj_row01: [f32; 4],
    /// x = proj[2][2], y = proj[3][2], z = 1/proj[0][0], w = 1/proj[1][1]
    proj_z: [f32; 4],
    /// Light direction in view space (xyz, w unused). For contact shadows.
    light_dir_vs: [f32; 4],
    /// xy = half-res width/height, zw = unused.
    size: [u32; 4],
}

// ============================================================
// SSAO Bilateral Blur post-process
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoBlurParams {
    /// xy = texel_size (of the half-res SSAO RT), z = depth_sigma, w = unused.
    params: [f32; 4],
}

// ============================================================
// Depth of Field (DoF) post-process
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DofParams {
    /// x = focus_distance, y = aperture, z = max_blur_radius (UV), w = unused
    params: [f32; 4],
    /// Inverse projection matrix — used to linearize depth.
    inv_proj: [[f32; 4]; 4],
}

// ============================================================
// Motion Blur post-process
// ============================================================
//
// Reads the TAA/DoF output (color) and the per-pixel velocity buffer.
// For each pixel, samples 8 taps along the velocity direction with a
// tent (linear) weight, blending them into a directionally-blurred
// result. Default OFF — no perf cost when disabled.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionBlurParams {
    /// x = strength, y = max_blur (UV), zw = unused.
    params: [f32; 4],
}

// ============================================================
// Screen-Space Subsurface Scattering (SSS) post-process
// ============================================================
//
// Single-pass 9-tap disc blur applied after the motion blur pass
// (pre-composite). Uses a chromatic diffusion profile where red
// scatters furthest (kernel width 1×), green 0.5×, blue 0.25×,
// simulating the spectral absorption of skin/wax/leaves.
// Depth-guided bilateral weighting prevents color bleeding across
// depth discontinuities (hard edges stay sharp). Default OFF.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SssParams {
    /// x = strength, y = width, z = falloff, w = unused.
    params: [f32; 4],
}

// ============================================================
// SSGI (Screen-Space Global Illumination) post-process
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsgiParams {
    inv_proj: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    /// x = intensity, y = radius, z = n_samples, w = frame_index
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsgiTemporalParams {
    /// x = blend_alpha (0.1), y = depth_reject_threshold, zw unused
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrParams {
    inv_proj: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    /// x=strength, y=max_dist, z=n_steps, w=padding
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneComposeParams {
    misc: [f32; 4],
    inv_vp: [[f32; 4]; 4],
    fog_color_density: [f32; 4],
    fog_params: [f32; 4],
    sun_shaft_uv_strength: [f32; 4],
    sun_shaft_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaParams {
    /// x = blend factor (current-frame weight), yzw padding.
    params: [f32; 4],
    inv_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ExposureParams {
    params: [f32; 4],
}

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
    /// x = grain seed (frame index, animates the noise),
    /// y = sharpen strength, zw padding.
    misc: [f32; 4],
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

    // Logical (points / CSS px) size — what user code addresses via
    // `screenWidth`/HUD coords. Physical render target size is stored
    // in `surface_config` and is `logical * scale_factor`. On non-HiDPI
    // platforms the two are identical.
    pub logical_width: u32,
    pub logical_height: u32,

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
    /// Albedo G-buffer (Rgba8Unorm). Fourth color attachment in the
    /// HDR pass. Post-passes (SSGI) multiply bounce radiance by this
    /// so dark materials absorb indirect light correctly.
    pub albedo_rt_texture: wgpu::Texture,
    pub albedo_rt_view: wgpu::TextureView,
    /// Composed HDR target — scene + SSR + SSGI*albedo + bloom + fog
    /// + shafts all merged by the `scene_compose` pass. Feeds both
    /// TAA (as the current-frame input) and composite (as the
    /// TAA-off source) so atmospherics stay consistent across paths.
    pub composed_rt_texture: wgpu::Texture,
    pub composed_rt_view: wgpu::TextureView,
    pub scene_compose_pipeline: wgpu::RenderPipeline,
    pub scene_compose_layout: wgpu::BindGroupLayout,
    pub scene_compose_uniform_buffer: wgpu::Buffer,
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
    /// Post-tonemap unsharp-mask strength (0 = off, ~0.25 subtle,
    /// ~0.5 punchy). Cheap lens-like crispening applied in LDR to
    /// avoid the highlight blowout that HDR-space sharpen causes.
    pub sharpen_strength: f32,
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
    /// SSAO RT (half-res) + compute GTAO pipeline + uniforms. Run
    /// after the HDR pass; sampled by the composite to darken
    /// crevices.
    pub ssao_rt_texture: wgpu::Texture,
    pub ssao_rt_view: wgpu::TextureView,
    pub ssao_pipeline: wgpu::ComputePipeline,
    pub ssao_layout: wgpu::BindGroupLayout,
    pub ssao_uniform_buffer: wgpu::Buffer,
    pub ssao_depth_sampler: wgpu::Sampler,
    /// Linear-depth Hi-Z pyramid (positive |view_z|). `HIZ_MIP_COUNT`
    /// separate textures so Metal multi-mip state tracking doesn't
    /// trip when writes and samples interleave in one encoder (same
    /// workaround `create_bloom_chain` uses). Built every frame
    /// before SSAO: one linearize pass + `HIZ_MIP_COUNT - 1`
    /// min-downsample passes. Sampled by SSAO with a per-step mip.
    pub hiz_textures: Vec<wgpu::Texture>,
    pub hiz_views: Vec<wgpu::TextureView>,
    pub hiz_sampler: wgpu::Sampler,
    pub hiz_linearize_pipeline: wgpu::ComputePipeline,
    pub hiz_linearize_layout: wgpu::BindGroupLayout,
    pub hiz_linearize_uniform_buffer: wgpu::Buffer,
    pub hiz_downsample_pipeline: wgpu::ComputePipeline,
    pub hiz_downsample_layout: wgpu::BindGroupLayout,
    pub hiz_downsample_uniform_buffers: Vec<wgpu::Buffer>,
    hiz_linearize_bg_cache: Option<wgpu::BindGroup>,
    hiz_downsample_bg_cache: Vec<Option<wgpu::BindGroup>>,
    /// Bilateral blur pass applied to the raw GTAO output. Reads
    /// ssao_rt, writes ssao_blur_rt (same half-res R8Unorm format).
    /// The TAA pass then samples ssao_blur_rt instead of ssao_rt.
    pub ssao_blur_rt_texture: wgpu::Texture,
    pub ssao_blur_rt_view: wgpu::TextureView,
    pub ssao_blur_pipeline: wgpu::RenderPipeline,
    pub ssao_blur_layout: wgpu::BindGroupLayout,
    pub ssao_blur_uniform_buffer: wgpu::Buffer,
    /// Strength multiplier for SSAO (0 = off, 1 = full). Default 1.0.
    pub ssao_strength: f32,
    /// Sample radius in UV units (default ~0.005, gives a soft AO
    /// signal a few pixels wide on a 1024-tall surface).
    pub ssao_radius: f32,
    /// Skip the SSAO + bilateral-blur passes entirely when false; the
    /// blur RT gets a WHITE clear (no occlusion) so composite stays
    /// correct. Cheaper than `ssao_strength = 0` which still runs
    /// the passes. Default true.
    pub ssao_enabled: bool,
    /// Skip the bloom downsample/upsample chain when false; composite
    /// receives bloom_intensity = 0 so the stale chain contributes
    /// nothing visually. Default true.
    pub bloom_enabled: bool,
    /// Cached bind groups for the post-FX passes whose inputs (RT
    /// views + uniform buffers) only change on resize. Invalidated
    /// (set to None) in `resize()` and rebuilt lazily on next use.
    /// Saves ~4 `create_bind_group` calls per frame (~15-20 µs on M1).
    ssao_bg_cache: Option<wgpu::BindGroup>,
    ssao_blur_bg_cache: Option<wgpu::BindGroup>,
    ssr_bg_cache: Option<wgpu::BindGroup>,
    ssgi_bg_cache: Option<wgpu::BindGroup>,
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
    /// TSR (temporal super-resolution): render the G-buffer + HDR
    /// chain at half-res, upscale via the TAA pass to full surface
    /// resolution. Halves fragment count on the dominant passes
    /// (main_hdr 4-MRT, scene_compose) for ~4× shading throughput.
    /// Coupled to `taa_enabled` — TAA provides the temporal jitter
    /// + history blend that reconstructs detail from sub-pixel
    /// samples. Off → render at native surface resolution.
    pub tsr_enabled: bool,
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

    /// SSGI (screen-space global illumination) pass output — half-res
    /// HDR holding the indirect diffuse bounce light for each fragment.
    /// Composited into the final image by the TAA pass.
    pub ssgi_rt_texture: wgpu::Texture,
    pub ssgi_rt_view: wgpu::TextureView,
    pub ssgi_pipeline: wgpu::RenderPipeline,
    pub ssgi_layout: wgpu::BindGroupLayout,
    pub ssgi_uniform_buffer: wgpu::Buffer,
    /// SSGI intensity multiplier (0 = off, 0.5 = default, 1+ = strong).
    pub ssgi_intensity: f32,
    /// SSGI max march distance in view-space meters.
    pub ssgi_radius: f32,
    /// SSGI master switch. Default true (temporal denoiser keeps it clean).
    pub ssgi_enabled: bool,

    /// SSGI temporal denoiser: ping-pong history textures (same format/size
    /// as ssgi_rt). Each frame blends noisy SSGI with reprojected history.
    pub ssgi_history_textures: [wgpu::Texture; 2],
    pub ssgi_history_views: [wgpu::TextureView; 2],
    pub ssgi_history_idx: usize,
    pub ssgi_temporal_pipeline: wgpu::RenderPipeline,
    pub ssgi_temporal_layout: wgpu::BindGroupLayout,
    pub ssgi_temporal_uniform_buffer: wgpu::Buffer,

    /// Depth of field render target (full-res HDR). DoF pass reads
    /// TAA output + depth, writes variable-radius Poisson disc blur
    /// here. Composite reads this instead of TAA when DoF is on.
    pub dof_rt_texture: wgpu::Texture,
    pub dof_rt_view: wgpu::TextureView,
    pub dof_pipeline: wgpu::RenderPipeline,
    pub dof_layout: wgpu::BindGroupLayout,
    pub dof_uniform_buffer: wgpu::Buffer,
    /// DoF master switch. Default false — no perf cost when off.
    pub dof_enabled: bool,
    /// Focus distance in world units from the camera. Objects at
    /// this distance are perfectly sharp. Default 10.0.
    pub dof_focus_distance: f32,
    /// Aperture (CoC scale). 0 = no blur, 0.05 = subtle, 0.2 = heavy.
    /// Default 0.0 (disabled even when dof_enabled is true).
    pub dof_aperture: f32,
    /// Maximum blur disc radius in UV units. Clamps the CoC so the
    /// blur never exceeds this radius. Default 0.02.
    pub dof_max_blur: f32,

    /// Per-pixel velocity render target (Rg16Float, surface size).
    /// Third color attachment in the HDR pass; written by the 3D and
    /// scene fragment shaders with screen-space velocity. Read by
    /// the motion blur pass and TAA for per-object reprojection.
    pub velocity_rt_texture: wgpu::Texture,
    pub velocity_rt_view: wgpu::TextureView,

    /// Motion blur render target (full-res HDR). Motion blur pass
    /// reads color + velocity, writes directionally-blurred result
    /// here. Composite reads this instead of the upstream source
    /// when motion blur is enabled.
    pub motion_blur_rt_texture: wgpu::Texture,
    pub motion_blur_rt_view: wgpu::TextureView,
    pub motion_blur_pipeline: wgpu::RenderPipeline,
    pub motion_blur_layout: wgpu::BindGroupLayout,
    pub motion_blur_uniform_buffer: wgpu::Buffer,
    /// Motion blur master switch. Default false — no perf cost when off.
    pub motion_blur_enabled: bool,
    /// Velocity multiplier. Higher = more blur for the same motion.
    /// Default 1.0.
    pub motion_blur_strength: f32,
    /// Maximum blur radius in UV units. Clamps velocity so blur never
    /// exceeds this radius. Default 0.05.
    pub motion_blur_max_blur: f32,

    /// Screen-space subsurface scattering (SSS) render target — full-res
    /// HDR. The SSS pass reads the motion-blur (or DoF/TAA/HDR) output and
    /// writes a chromatically-blurred version here. Composite reads this
    /// instead of the upstream source when SSS is on.
    pub sss_rt_texture: wgpu::Texture,
    pub sss_rt_view: wgpu::TextureView,
    pub sss_pipeline: wgpu::RenderPipeline,
    pub sss_layout: wgpu::BindGroupLayout,
    pub sss_uniform_buffer: wgpu::Buffer,
    /// SSS master switch. Default false — zero perf cost when off.
    pub sss_enabled: bool,
    /// SSS scatter strength: 0 = no blur (even when enabled), 1 = full
    /// chromatic blur blended over the source. Default 0.5.
    pub sss_strength: f32,
    /// SSS blur radius in UV units. Controls how far light scatters
    /// beneath the surface. Default 0.01 (~1% of viewport width).
    pub sss_width: f32,

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
    /// Cached inverses of the current projection and view-projection
    /// matrices, recomputed once per `begin_mode_3d` and reused by
    /// every post-FX pass (SSAO, SSR, SSGI, DoF, scene_compose).
    /// Without this cache the renderer calls `mat4_invert` 4-5 times
    /// per frame on the same matrices.
    current_inv_proj_matrix: [[f32; 4]; 4],
    current_inv_vp_matrix: [[f32; 4]; 4],
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
    /// Dedicated cosine-convolved diffuse irradiance texture. Separate
    /// from the GGX-prefiltered specular chain so both can use their
    /// full resolution range. Single mip at 128×64 equirect — ample
    /// for a low-frequency irradiance signal. `None` until an HDR is
    /// loaded; bind group falls back to `scene_env_default_view` (1×1
    /// gray) while empty.
    env_diffuse_texture: Option<wgpu::Texture>,

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
        logical_width: u32,
        logical_height: u32,
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
        // 2D uses logical (points) dimensions so user HUD coords stay
        // consistent on HiDPI displays — the rasterizer upsamples to
        // the physical render target automatically.
        let initial_uniforms = Uniforms2D {
            screen_size: [logical_width as f32, logical_height as f32],
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
            contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4, model: IDENTITY_MAT4, prev_mvp: IDENTITY_MAT4, model_tint: [1.0, 1.0, 1.0, 1.0] }),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            // 16x anisotropic filtering. Without this, surfaces viewed
            // at oblique angles (long streets of facades, floor
            // receding toward the horizon) pick an over-blurred mip to
            // avoid aliasing, producing a 'watercolor' look on distant
            // walls. Anisotropy samples along the actual footprint so
            // texture detail is preserved along the sharp axis. wgpu
            // clamps to the device max — Metal/Vulkan/DX12 all support
            // 16x; lower-end GLES hardware may clamp to 4x or 8x.
            anisotropy_clamp: 16,
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
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&shadow_map.depth_views[0]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&shadow_map.depth_views[1]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&shadow_map.depth_views[2]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&shadow_map.sampler) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&scene_env_default_view) },
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
        let (albedo_rt_texture, albedo_rt_view) = create_albedo_rt(&device, surface_config.width, surface_config.height);
        let (composed_rt_texture, composed_rt_view) = create_composed_rt(&device, surface_config.width, surface_config.height);
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
        let (ssgi_rt_texture, ssgi_rt_view) = create_ssgi_rt(
            &device, surface_config.width, surface_config.height,
        );
        let (ssgi_history_textures, ssgi_history_views) = create_ssgi_history_textures(
            &device, surface_config.width, surface_config.height,
        );
        let (dof_rt_texture, dof_rt_view) = create_dof_rt(
            &device, surface_config.width, surface_config.height,
        );
        let (velocity_rt_texture, velocity_rt_view) = create_velocity_rt(
            &device, surface_config.width, surface_config.height,
        );
        // Motion blur RT reuses the same HDR format as DoF.
        let (motion_blur_rt_texture, motion_blur_rt_view) = create_dof_rt(
            &device, surface_config.width, surface_config.height,
        );
        // SSS RT — full-res HDR, same format as DoF/motion-blur.
        let (sss_rt_texture, sss_rt_view) = create_sss_rt(
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
                    Some(wgpu::ColorTargetState {
                        format: VELOCITY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
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
                contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4, model: IDENTITY_MAT4, prev_mvp: IDENTITY_MAT4, model_tint: [1.0; 4] }),
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
                    Some(wgpu::ColorTargetState {
                        format: VELOCITY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
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
                    Some(wgpu::ColorTargetState {
                        format: VELOCITY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        // --- Hi-Z pyramid (linearize + downsample) pipelines ---
        let hiz_linearize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_linearize_shader"),
            source: wgpu::ShaderSource::Wgsl(HIZ_LINEARIZE_SHADER_WGSL.into()),
        });
        let hiz_linearize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_linearize_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HIZ_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
            ],
        });
        let hiz_linearize_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_linearize_pl_layout"),
            bind_group_layouts: &[&hiz_linearize_layout],
            push_constant_ranges: &[],
        });
        let hiz_linearize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz_linearize_pipeline"),
            layout: Some(&hiz_linearize_pl_layout),
            module: &hiz_linearize_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let hiz_linearize_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hiz_linearize_uniform_buffer"),
            size: std::mem::size_of::<HizLinearizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hiz_downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_downsample_shader"),
            source: wgpu::ShaderSource::Wgsl(HIZ_DOWNSAMPLE_SHADER_WGSL.into()),
        });
        let hiz_downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_downsample_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HIZ_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
            ],
        });
        let hiz_downsample_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_downsample_pl_layout"),
            bind_group_layouts: &[&hiz_downsample_layout],
            push_constant_ranges: &[],
        });
        let hiz_downsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz_downsample_pipeline"),
            layout: Some(&hiz_downsample_pl_layout),
            module: &hiz_downsample_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let hiz_downsample_uniform_buffers: Vec<wgpu::Buffer> = (0..HIZ_MIP_COUNT - 1)
            .map(|_| device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hiz_downsample_uniform_buffer"),
                size: std::mem::size_of::<HizDownsampleParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();
        let hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hiz_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let (hiz_textures, hiz_views) = create_linear_depth_hiz_chain(
            &device, surface_config.width, surface_config.height,
        );

        // --- SSAO (compute GTAO) pipeline ---
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_SHADER_WGSL.into()),
        });
        let ssao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: SSAO_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
            ],
        });
        let ssao_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_pl_layout"),
            bind_group_layouts: &[&ssao_layout],
            push_constant_ranges: &[],
        });
        let ssao_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssao_pipeline"),
            layout: Some(&ssao_pl_layout),
            module: &ssao_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
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

        // --- SSAO bilateral blur pipeline ---
        let ssao_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_BLUR_SHADER_WGSL.into()),
        });
        let ssao_blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_blur_layout"),
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let ssao_blur_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_blur_pl_layout"),
            bind_group_layouts: &[&ssao_blur_layout],
            push_constant_ranges: &[],
        });
        let ssao_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_blur_pipeline"),
            layout: Some(&ssao_blur_pl_layout),
            vertex: wgpu::VertexState {
                module: &ssao_blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_blur_shader,
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
        let ssao_blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_blur_uniform_buffer"),
            size: std::mem::size_of::<SsaoBlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (ssao_blur_rt_texture, ssao_blur_rt_view) = create_ssao_blur_rt(
            &device, surface_config.width, surface_config.height,
        );

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
                // composed_rt (tex + sampler)
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
                // history (tex + sampler)
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
                // depth (tex + sampler)
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // velocity (tex + sampler)
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

        // --- SSGI pipeline ---
        let ssgi_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_shader"),
            source: wgpu::ShaderSource::Wgsl(SSGI_SHADER_WGSL.into()),
        });
        let ssgi_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_layout"),
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
            ],
        });
        let ssgi_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssgi_pl_layout"),
            bind_group_layouts: &[&ssgi_layout],
            push_constant_ranges: &[],
        });
        let ssgi_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssgi_pipeline"),
            layout: Some(&ssgi_pl_layout),
            vertex: wgpu::VertexState {
                module: &ssgi_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssgi_shader, entry_point: Some("fs_main"),
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
        let ssgi_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi_uniform_buffer"),
            size: std::mem::size_of::<SsgiParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- SSGI temporal denoiser pipeline ---
        let ssgi_temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_temporal_shader"),
            source: wgpu::ShaderSource::Wgsl(SSGI_TEMPORAL_SHADER_WGSL.into()),
        });
        let ssgi_temporal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_temporal_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // binding 1: current SSGI (noisy)
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
                // binding 3: history SSGI (previous frame accumulated)
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
                // binding 5: velocity buffer (motion vectors)
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
        let ssgi_temporal_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssgi_temporal_pl_layout"),
            bind_group_layouts: &[&ssgi_temporal_layout],
            push_constant_ranges: &[],
        });
        let ssgi_temporal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssgi_temporal_pipeline"),
            layout: Some(&ssgi_temporal_pl_layout),
            vertex: wgpu::VertexState {
                module: &ssgi_temporal_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssgi_temporal_shader, entry_point: Some("fs_main"),
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
        let ssgi_temporal_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi_temporal_uniform_buffer"),
            size: std::mem::size_of::<SsgiTemporalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Scene-compose pipeline ---
        // Merges HDR + SSR + SSGI*albedo + bloom + fog + shafts into
        // composed_rt. Both TAA and composite downstream read from
        // this single output so atmospherics behave identically
        // whether TAA is on or off.
        let scene_compose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_compose_shader"),
            source: wgpu::ShaderSource::Wgsl(SCENE_COMPOSE_SHADER_WGSL.into()),
        });
        let scene_compose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_compose_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // hdr, ssr, ssgi, bloom, albedo each: tex + sampler.
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // depth (tex + sampler)
                wgpu::BindGroupLayoutEntry {
                    binding: 11, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let scene_compose_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_compose_pl_layout"),
            bind_group_layouts: &[&scene_compose_layout],
            push_constant_ranges: &[],
        });
        let scene_compose_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_compose_pipeline"),
            layout: Some(&scene_compose_pl_layout),
            vertex: wgpu::VertexState {
                module: &scene_compose_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_compose_shader, entry_point: Some("fs_main"),
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
        let scene_compose_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_compose_uniform_buffer"),
            size: std::mem::size_of::<SceneComposeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- DoF (depth of field) pipeline ---
        let dof_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_shader"),
            source: wgpu::ShaderSource::Wgsl(DOF_SHADER_WGSL.into()),
        });
        let dof_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_layout"),
            entries: &[
                // binding 0: DofParams uniform
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
                // binding 1: color input (TAA output or hdr_rt)
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
                // binding 2: color sampler (linear filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: depth texture (texture_depth_2d)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: depth sampler (non-filtering, non-comparison)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let dof_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_pl_layout"),
            bind_group_layouts: &[&dof_layout],
            push_constant_ranges: &[],
        });
        let dof_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof_pipeline"),
            layout: Some(&dof_pl_layout),
            vertex: wgpu::VertexState {
                module: &dof_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &dof_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
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
        let dof_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_uniform_buffer"),
            size: std::mem::size_of::<DofParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Motion blur pipeline ---
        let motion_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(MOTION_BLUR_SHADER_WGSL.into()),
        });
        let motion_blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion_blur_layout"),
            entries: &[
                // binding 0: MotionBlurParams uniform
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
                // binding 1: color input (upstream HDR)
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
                // binding 2: color sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: velocity texture
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
                // binding 4: velocity sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let motion_blur_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion_blur_pl_layout"),
            bind_group_layouts: &[&motion_blur_layout],
            push_constant_ranges: &[],
        });
        let motion_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion_blur_pipeline"),
            layout: Some(&motion_blur_pl_layout),
            vertex: wgpu::VertexState {
                module: &motion_blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &motion_blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
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
        let motion_blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_blur_uniform_buffer"),
            size: std::mem::size_of::<MotionBlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- SSS (screen-space subsurface scattering) pipeline ---
        let sss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sss_shader"),
            source: wgpu::ShaderSource::Wgsl(SSS_SHADER_WGSL.into()),
        });
        let sss_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sss_layout"),
            entries: &[
                // binding 0: SssParams uniform
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
                // binding 1: color input (upstream HDR)
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
                // binding 2: color sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: depth texture (texture_depth_2d — for bilateral weighting)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: depth sampler (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let sss_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sss_pl_layout"),
            bind_group_layouts: &[&sss_layout],
            push_constant_ranges: &[],
        });
        let sss_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sss_pipeline"),
            layout: Some(&sss_pl_layout),
            vertex: wgpu::VertexState {
                module: &sss_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sss_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
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
        let sss_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sss_uniform_buffer"),
            size: std::mem::size_of::<SssParams>() as u64,
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
            logical_width,
            logical_height,
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
            albedo_rt_texture,
            albedo_rt_view,
            composed_rt_texture,
            composed_rt_view,
            scene_compose_pipeline,
            scene_compose_layout,
            scene_compose_uniform_buffer,
            composite_pipeline,
            composite_layout,
            composite_sampler,
            // 1 = AgX (Troy Sobotka 2022). Matches Blender 4.0+ and
            // UE5 "PBR Neutral" look — softer highlight rolloff and
            // better hue preservation than the Narkowicz ACES fit,
            // which tends to read as "digital/plasticky" on saturated
            // primary-colour materials (red awnings, green storefronts).
            tonemap_kind: 1,
            auto_exposure: false,
            manual_exposure: 1.0,
            auto_exposure_key: 0.18,
            // 0.015 per frame at 60fps → ~45-frame (0.75s) half-life.
            // Faster than a camera pan; slow enough to not "hunt" on
            // scene detail as the camera moves between bright sky
            // and dark geometry.
            auto_exposure_rate: 0.05,  // ~0.3 second half-life at 60fps
            chromatic_aberration: 0.0,
            vignette_strength: 0.0,
            vignette_softness: 0.25,
            grain_strength: 0.0,
            sharpen_strength: 0.8,
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
            hiz_textures,
            hiz_views,
            hiz_sampler,
            hiz_linearize_pipeline,
            hiz_linearize_layout,
            hiz_linearize_uniform_buffer,
            hiz_downsample_pipeline,
            hiz_downsample_layout,
            hiz_downsample_uniform_buffers,
            hiz_linearize_bg_cache: None,
            hiz_downsample_bg_cache: vec![None; (HIZ_MIP_COUNT - 1) as usize],
            ssao_blur_rt_texture,
            ssao_blur_rt_view,
            ssao_blur_pipeline,
            ssao_blur_layout,
            ssao_blur_uniform_buffer,
            ssao_strength: 1.0,
            ssao_enabled: true,
            bloom_enabled: true,
            ssao_bg_cache: None,
            ssao_blur_bg_cache: None,
            ssr_bg_cache: None,
            ssgi_bg_cache: None,
            // World-space AO radius in meters. Sponza-scale arches
            // and columns span 3-5m, so 4m catches proper architectural
            // occlusion.
            // 2.0 view-space units — half the previous 4.0. Finer radius
            // catches sub-brick mortar-line detail and carved capital
            // grooves that 4.0 smoothed over; coarser occlusion already
            // covered by GTAO's horizon scan + indirect_shadow floor.
            ssao_radius: 2.0,
            taa_textures,
            taa_views,
            taa_current_idx: 0,
            taa_pipeline,
            taa_layout,
            taa_uniform_buffer,
            taa_frame_index: 0,
            taa_enabled: true,
            tsr_enabled: true,
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
            ssgi_rt_texture,
            ssgi_rt_view,
            ssgi_pipeline,
            ssgi_layout,
            ssgi_uniform_buffer,
            // 1.0 — stronger bounce than the earlier 0.5 default so
            // shadowed regions pick up visible color from nearby lit
            // surfaces (red awning tinting wall behind it, ground
            // bounce warming shopfronts, sky fill cooling overhangs).
            // Under-contribution was one of the reasons scenes read
            // as 'flat grey shadows' rather than 'shadow lit by
            // bounce'.
            ssgi_intensity: 1.0,
            ssgi_radius: 20.0,
            ssgi_enabled: true,
            ssgi_history_textures,
            ssgi_history_views,
            ssgi_history_idx: 0,
            ssgi_temporal_pipeline,
            ssgi_temporal_layout,
            ssgi_temporal_uniform_buffer,
            dof_rt_texture,
            dof_rt_view,
            dof_pipeline,
            dof_layout,
            dof_uniform_buffer,
            dof_enabled: false,
            dof_focus_distance: 10.0,
            dof_aperture: 0.0,
            dof_max_blur: 0.006,
            velocity_rt_texture,
            velocity_rt_view,
            motion_blur_rt_texture,
            motion_blur_rt_view,
            motion_blur_pipeline,
            motion_blur_layout,
            motion_blur_uniform_buffer,
            motion_blur_enabled: false,
            motion_blur_strength: 1.0,
            motion_blur_max_blur: 0.05,
            sss_rt_texture,
            sss_rt_view,
            sss_pipeline,
            sss_layout,
            sss_uniform_buffer,
            sss_enabled: false,
            sss_strength: 0.5,
            sss_width: 0.01,
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
            current_inv_proj_matrix: IDENTITY_MAT4,
            current_inv_vp_matrix: IDENTITY_MAT4,
            current_camera_pos: [0.0, 0.0, 0.0],
            uniform_3d_layout,
            render_mode: RenderMode::ScreenSpace,
            debug_frame: 0,
            pending_joint_matrices: None,
            model_skin_scale: 1.0,
            clear_color: wgpu::Color::BLACK,
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
            env_diffuse_texture: None,
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

    /// Resize the swapchain and all post-process render targets.
    // Debug accessors for diagnosing draw call issues
    pub fn vertices_2d_count(&self) -> usize { self.vertices_2d.len() }
    pub fn indices_2d_count(&self) -> usize { self.indices_2d.len() }
    pub fn draw_calls_2d_count(&self) -> usize { self.draw_calls_2d.len() }
    pub fn texture_count(&self) -> usize { self.texture_bind_groups.len() }
    pub fn texture_sizes_debug(&self) -> Vec<(u32, u32)> { self.texture_sizes.clone() }

    /// `width`/`height` are PHYSICAL pixels (the actual GPU surface).
    /// `logical_width`/`logical_height` are the points size reported to
    /// user code via `screenWidth`/HUD — on non-HiDPI platforms they
    /// match the physical size.
    /// Render-pass extent used by main_hdr + scene_compose. When TSR
    /// is on this is half the surface size; the TAA pass upscales to
    /// the full surface for composite. Off → native surface.
    pub fn render_extent(&self) -> (u32, u32) {
        if self.tsr_enabled {
            (
                (self.surface_config.width / 2).max(1),
                (self.surface_config.height / 2).max(1),
            )
        } else {
            (self.surface_config.width.max(1), self.surface_config.height.max(1))
        }
    }

    pub fn resize(&mut self, width: u32, height: u32, logical_width: u32, logical_height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.logical_width = logical_width.max(1);
            self.logical_height = logical_height.max(1);
            self.surface.configure(&self.device, &self.surface_config);

            // Render-resolution RTs (G-buffer + composed). Half of
            // surface when TSR is on, full surface otherwise. The
            // TAA pass upscales the half-res composed_rt to the
            // full-res history texture — the rest of the post-FX
            // chain (DoF/MB/SSS) and composite stay at full surface.
            let (rw, rh) = self.render_extent();

            let (dt, dv) = create_depth_texture(&self.device, rw, rh);
            self.depth_texture = dt;
            self.depth_view = dv;
            let (hdr_t, hdr_v) = create_hdr_rt(&self.device, rw, rh);
            self.hdr_rt_texture = hdr_t;
            self.hdr_rt_view = hdr_v;
            let (mat_t, mat_v) = create_material_rt(&self.device, rw, rh);
            self.material_rt_texture = mat_t;
            self.material_rt_view = mat_v;
            let (alb_t, alb_v) = create_albedo_rt(&self.device, rw, rh);
            let (cmp_t, cmp_v) = create_composed_rt(&self.device, rw, rh);
            self.composed_rt_texture = cmp_t;
            self.composed_rt_view = cmp_v;
            self.albedo_rt_texture = alb_t;
            self.albedo_rt_view = alb_v;
            let (bt, bm, bf) = create_bloom_chain(&self.device, width, height, BLOOM_MIP_COUNT);
            self.bloom_chain_textures = bt;
            self.bloom_mip_views = bm;
            self.bloom_full_view = bf;
            let (st, sv) = create_ssao_rt(&self.device, width, height);
            self.ssao_rt_texture = st;
            self.ssao_rt_view = sv;
            let (sbt, sbv) = create_ssao_blur_rt(&self.device, width, height);
            self.ssao_blur_rt_texture = sbt;
            self.ssao_blur_rt_view = sbv;
            let (hiz_t, hiz_v) = create_linear_depth_hiz_chain(&self.device, width, height);
            self.hiz_textures = hiz_t;
            self.hiz_views = hiz_v;
            let (taa_t, taa_v) = create_taa_textures(&self.device, width, height);
            self.taa_textures = taa_t;
            self.taa_views = taa_v;
            self.taa_frame_index = 0; // reset jitter sequence on resize
            let (sr_t, sr_v) = create_ssr_rt(&self.device, width, height);
            self.ssr_rt_texture = sr_t;
            self.ssr_rt_view = sr_v;
            let (ssgi_t, ssgi_v) = create_ssgi_rt(&self.device, width, height);
            self.ssgi_rt_texture = ssgi_t;
            self.ssgi_rt_view = ssgi_v;
            let (ssgi_ht, ssgi_hv) = create_ssgi_history_textures(&self.device, width, height);
            self.ssgi_history_textures = ssgi_ht;
            self.ssgi_history_views = ssgi_hv;
            self.ssgi_history_idx = 0;
            let (dof_t, dof_v) = create_dof_rt(&self.device, width, height);
            self.dof_rt_texture = dof_t;
            self.dof_rt_view = dof_v;
            // velocity is the 3rd MRT of main_hdr — must match the
            // render-resolution RTs above.
            let (vel_t, vel_v) = create_velocity_rt(&self.device, rw, rh);
            self.velocity_rt_texture = vel_t;
            self.velocity_rt_view = vel_v;
            let (mb_t, mb_v) = create_dof_rt(&self.device, width, height);
            self.motion_blur_rt_texture = mb_t;
            self.motion_blur_rt_view = mb_v;
            let (sss_t, sss_v) = create_sss_rt(&self.device, width, height);
            self.sss_rt_texture = sss_t;
            self.sss_rt_view = sss_v;

            // Invalidate bind-group caches that reference any of the
            // RT views we just recreated.
            self.ssao_bg_cache = None;
            self.ssao_blur_bg_cache = None;
            self.ssr_bg_cache = None;
            self.ssgi_bg_cache = None;
            self.hiz_linearize_bg_cache = None;
            for slot in self.hiz_downsample_bg_cache.iter_mut() {
                *slot = None;
            }
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
    /// extra texture writes, no TSR. On = sub-pixel super-sampling
    /// + half-res shading via TSR upscale.
    pub fn set_taa_enabled(&mut self, enabled: bool) {
        if enabled != self.taa_enabled {
            self.taa_enabled = enabled;
            self.tsr_enabled = enabled;
            self.taa_frame_index = 0;
            // TSR toggle changes render-resolution; recreate the
            // affected RTs at the new size.
            let (w, h) = (self.surface_config.width, self.surface_config.height);
            self.resize(w, h, self.logical_width, self.logical_height);
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

    /// Toggle SSGI (screen-space global illumination) on/off. Off
    /// (default) = no SSGI pass, zero perf cost. On = single-bounce
    /// indirect diffuse lighting via screen-space ray marching.
    pub fn set_ssgi_enabled(&mut self, on: bool) {
        self.ssgi_enabled = on;
    }

    /// SSGI intensity multiplier (0 = off, 0.5 = default, 1+ = strong).
    /// Controls the brightness of indirect bounce light.
    pub fn set_ssgi_intensity(&mut self, intensity: f32) {
        self.ssgi_intensity = intensity.max(0.0);
    }

    /// SSGI max march distance in view-space meters (default 20).
    /// Tune to the scene scale: small for tight rooms, large for
    /// open-world interiors.
    pub fn set_ssgi_radius(&mut self, radius: f32) {
        self.ssgi_radius = radius.max(0.1);
    }

    /// Toggle depth of field on/off. Off (default) = no DoF pass,
    /// zero perf cost. On = variable-radius Poisson disc blur driven
    /// by circle of confusion from the depth buffer.
    pub fn set_dof_enabled(&mut self, on: bool) {
        self.dof_enabled = on;
    }

    /// Set the DoF focus distance in world units from the camera.
    /// Objects at this distance are perfectly sharp; objects closer
    /// or farther blur proportionally to `dof_aperture`.
    pub fn set_dof_focus_distance(&mut self, dist: f32) {
        self.dof_focus_distance = dist.max(0.01);
    }

    /// Set the DoF aperture (CoC scale). 0 = no blur even when DoF
    /// is enabled. 0.05 = subtle. 0.2 = heavy. Higher values
    /// produce stronger blur for the same distance from focus.
    pub fn set_dof_aperture(&mut self, aperture: f32) {
        self.dof_aperture = aperture.max(0.0);
    }

    /// Toggle motion blur on/off. Off (default) = no motion blur
    /// pass, zero perf cost. On = 8-tap directional blur driven by
    /// the per-pixel velocity buffer.
    pub fn set_motion_blur_enabled(&mut self, on: bool) {
        self.motion_blur_enabled = on;
    }

    /// Set the motion blur strength (velocity multiplier). 0 = no
    /// visible blur even when enabled. 1.0 = default, subtle.
    /// Higher values amplify the blur for the same screen-space
    /// velocity.
    pub fn set_motion_blur_strength(&mut self, strength: f32) {
        self.motion_blur_strength = strength.max(0.0);
    }

    /// Toggle screen-space subsurface scattering (SSS) on/off.
    /// Off (default) — zero perf cost. On — single fullscreen pass
    /// applies a 9-tap chromatic disc blur (red scatters furthest)
    /// with depth-guided bilateral edge-stop weighting.
    pub fn set_sss_enabled(&mut self, on: bool) {
        self.sss_enabled = on;
    }

    /// SSS scatter strength (0 = transparent / no blur, 1 = full
    /// chromatic blur). 0.5 (default) blends half blurred with half
    /// original, giving a subtle translucent-skin look without
    /// completely losing surface detail.
    pub fn set_sss_strength(&mut self, strength: f32) {
        self.sss_strength = strength.clamp(0.0, 1.0);
    }

    /// SSS blur radius in UV units. Controls how far light scatters
    /// beneath the surface in screen space. 0.01 (default) ≈ 1% of
    /// viewport width — a few pixels at 1080p. Larger values look
    /// more waxy/translucent; smaller values are subtle.
    pub fn set_sss_width(&mut self, width: f32) {
        self.sss_width = width.max(0.0);
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

    // ============================================================
    // Render quality toggles — control individual post-FX / lighting
    // features at runtime. Games call these directly for fine-tuning
    // or use `apply_quality_preset()` for batch configuration.
    // ============================================================

    pub fn set_shadows_enabled(&mut self, on: bool) {
        if on { self.shadow_map.enable(); } else { self.shadow_map.disable(); }
    }
    pub fn set_bloom_enabled(&mut self, on: bool) { self.bloom_enabled = on; }
    pub fn set_ssao_enabled(&mut self, on: bool) { self.ssao_enabled = on; }

    /// Batch-configure every quality flag based on a preset level.
    /// Presets:
    ///   0 = Off     — bare minimum, for the slowest integrated GPUs.
    ///                 No shadows, no SSAO, no bloom, no TAA, no SSR/SSGI,
    ///                 no DoF/motion blur/SSS, no chromatic aberration.
    ///   1 = Low     — shadows off, SSAO off, bloom low, TAA off. Keeps
    ///                 the base HDR/tonemap pipeline only.
    ///   2 = Medium  — shadows on, SSAO on, bloom on, TAA on. No SSR/SSGI
    ///                 or cinematic effects.
    ///   3 = High    — adds SSR + SSGI + subtle chromatic aberration.
    ///   4 = Ultra   — everything on (plus DoF if aperture > 0).
    /// Individual setters override preset choices on the current frame —
    /// call `apply_quality_preset` first, then customize as needed.
    pub fn apply_quality_preset(&mut self, preset: u32) {
        let (shadows, ssao, bloom, taa, ssr, ssgi, motion_blur, sss, ca) = match preset {
            0 => (false, false, false, false, false, false, false, false, 0.0),
            1 => (false, false, true,  false, false, false, false, false, 0.0),
            2 => (true,  true,  true,  true,  false, false, false, false, 0.0),
            3 => (true,  true,  true,  true,  true,  true,  false, false, 0.002),
            _ => (true,  true,  true,  true,  true,  true,  true,  true,  0.003),
        };
        self.set_shadows_enabled(shadows);
        self.set_ssao_enabled(ssao);
        self.set_bloom_enabled(bloom);
        self.set_taa_enabled(taa);
        self.set_ssr_enabled(ssr);
        self.set_ssgi_enabled(ssgi);
        self.set_motion_blur_enabled(motion_blur);
        self.set_sss_enabled(sss);
        self.set_chromatic_aberration(ca);
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

        // GGX-prefilter every mip 1..N-1 with roughness scaling. Mip 0
        // is the unmodified mirror radiance (copied above); higher mips
        // are progressively rougher. Diffuse irradiance now lives in a
        // separate dedicated texture, so the specular chain uses every
        // available mip — roughness = 1 samples the smallest mip for
        // the widest GGX lobe, with no mip stolen for diffuse use.
        for level in 1..mip_count {
            let mip_w = (width >> level).max(1);
            let mip_h = (height >> level).max(1);
            let roughness = level as f32 / (mip_count - 1) as f32;
            let sample_count = (128.0 + 384.0 * roughness).round();

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
            pass.set_pipeline(&self.prefilter_pipeline);
            pass.set_bind_group(0, &prefilter_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Dedicated diffuse irradiance texture — 128×64 equirect with
        // cosine-convolved radiance. 1024 samples / texel, one-shot.
        let diffuse_w: u32 = 128;
        let diffuse_h: u32 = 64;
        let diffuse_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky_env_diffuse_texture"),
            size: wgpu::Extent3d { width: diffuse_w, height: diffuse_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let diffuse_uniforms = PrefilterUniforms {
            params: [1.0, 1024.0, diffuse_w as f32, diffuse_h as f32],
        };
        self.queue.write_buffer(&self.prefilter_uniform_buffer, 0, bytemuck::bytes_of(&diffuse_uniforms));
        let diffuse_view_rt = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prefilter_diffuse_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &diffuse_view_rt,
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
            pass.set_pipeline(&self.prefilter_diffuse_pipeline);
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
        // LUT bindings stay put — only env tex/sampler + diffuse view
        // change.
        let diffuse_view_bg = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let new_lighting_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_bg"),
            layout: &self.lighting_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.lighting_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.env_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.brdf_lut_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.brdf_lut_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&diffuse_view_bg) },
            ],
        });

        self.sky_texture = Some(texture);
        self.sky_bind_group = Some(bg);
        self.env_diffuse_texture = Some(diffuse_texture);
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
        self.current_inv_vp_matrix
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
        has_mr_texture: bool,
        alpha_cutoff: f32,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let uniforms = SceneMaterialUniforms::new(metallic, roughness, emissive, has_mr_texture, alpha_cutoff);
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

        // Write identity uniforms to slot 0 (2D uses logical points,
        // not physical pixels — see Renderer::new).
        let w = self.logical_width as f32;
        let h = self.logical_height as f32;
        let uniforms = Uniforms2D {
            screen_size: [w, h],
            _pad: [0.0; 2],
            view_proj: IDENTITY_MAT4,
        };
        self.queue.write_buffer(&self.uniform_buffers[0], 0, bytemuck::bytes_of(&uniforms));

        // Reset lighting to defaults (clears additional lights too).
        // Preserve env_intensity — it's set once at app init via
        // set_env_intensity, not per-frame, so the default-reset
        // would clobber it. camera_pos.xyz gets rewritten below by
        // begin_mode_3d with the actual camera position.
        let preserved_env_intensity = self.lighting_uniforms.camera_pos[3];
        self.lighting_uniforms = LightingUniforms::defaults();
        self.lighting_uniforms.camera_pos[3] = preserved_env_intensity;
        self.queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&self.lighting_uniforms));
        self.clear_additional_lights();

        // DEBUG: joint animation disabled for iOS port
        // self.debug_frame += 1;
        // let angle = (self.debug_frame as f32) * 0.03;
        // self.set_joint_test(0, angle.sin() * 0.8);
        // self.set_joint_test(5, (angle * 1.5).sin() * 0.5);
    }

    pub fn end_frame(&mut self) {
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
                Err(_) => {
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
    pub fn end_frame_with_scene(&mut self, scene: &crate::scene::SceneGraph, profiler: &mut crate::profiler::Profiler) {
        profiler.begin("joint_flush");
        self.flush_joint_matrices();
        profiler.end("joint_flush");

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

        // Shadow pass: render scene nodes from light's perspective into
        // cascaded shadow maps (3 cascades).
        profiler.begin("shadow_pass");
        if self.shadow_map.enabled {
            // Compute cascade VPs from the primary directional light and camera.
            let light_dir = [
                self.lighting_uniforms.light_dir[0],
                self.lighting_uniforms.light_dir[1],
                self.lighting_uniforms.light_dir[2],
            ];
            // Auto-fit: compute world-space AABB across every visible,
            // cast-shadow node so the ortho volume always covers the
            // scene regardless of what's loaded. No per-scene magic
            // numbers.
            let scene_bounds = scene.compute_shadow_bounds();
            self.shadow_map.compute_cascade_vps(
                light_dir,
                self.current_camera_pos,
                self.current_view_matrix,
                self.current_proj_matrix,
                0.5,   // near — start cascades slightly past the camera
                80.0,  // far — shadow coverage range
                scene_bounds,
            );

            // Re-upload lighting uniforms with cascade VPs and splits.
            self.lighting_uniforms.shadow_cascade_vps = self.shadow_map.light_vps;
            self.lighting_uniforms.shadow_cascade_splits = [
                self.shadow_map.cascade_splits[0],
                self.shadow_map.cascade_splits[1],
                self.shadow_map.cascade_splits[2],
                // .w = mip-LOD bias for material textures. -1 when
                // TSR is on (rendering at half-res selects 1 mip
                // coarser by hardware default, so bias finer to
                // recover detail).
                if self.tsr_enabled { -1.0 } else { 0.0 },
            ];
            self.lighting_uniforms.shadow_view_matrix = self.current_view_matrix;
            self.queue.write_buffer(
                &self.lighting_buffer,
                0,
                bytemuck::bytes_of(&self.lighting_uniforms),
            );

            // Build draw list once (shared across all cascades).
            // Collect per-node data before any render pass borrows the scene.
            struct ShadowDrawEntry {
                vb_idx: usize,
                ib_idx: usize,
                index_count: u32,
                transform: [[f32; 4]; 4],
            }
            let mut shadow_nodes: Vec<ShadowDrawEntry> = Vec::new();
            // Collect buffer references separately for the render pass
            let mut shadow_vbs: Vec<&wgpu::Buffer> = Vec::new();
            let mut shadow_ibs: Vec<&wgpu::Buffer> = Vec::new();
            for (_handle, node) in scene.nodes.iter() {
                if !node.visible || !node.cast_shadow || node.indices.is_empty() {
                    continue;
                }
                let Some(vb) = &node.gpu_vb else { continue };
                let Some(ib) = &node.gpu_ib else { continue };
                let vb_idx = shadow_vbs.len();
                shadow_vbs.push(vb);
                shadow_ibs.push(ib);
                shadow_nodes.push(ShadowDrawEntry {
                    vb_idx,
                    ib_idx: vb_idx,
                    index_count: node.gpu_index_count,
                    transform: node.transform,
                });
            }

            // Render each cascade
            for cascade in 0..crate::shadows::NUM_CASCADES {
                let stride = crate::shadows::SHADOW_UNIFORM_STRIDE as usize;
                let max = crate::shadows::SHADOW_MAX_NODES as usize;
                let mut uniform_data: Vec<u8> = vec![0u8; stride * max];
                let mut slot = 0usize;
                let cascade_vp = self.shadow_map.light_vps[cascade];

                for entry in &shadow_nodes {
                    if slot >= max { break; }
                    let uniforms = crate::shadows::ShadowUniforms {
                        light_vp: cascade_vp,
                        model: entry.transform,
                    };
                    let off = slot * stride;
                    uniform_data[off..off + std::mem::size_of::<crate::shadows::ShadowUniforms>()]
                        .copy_from_slice(bytemuck::bytes_of(&uniforms));
                    slot += 1;
                }

                if slot > 0 {
                    self.queue.write_buffer(
                        &self.shadow_map.uniform_buffer,
                        0,
                        &uniform_data[..slot * stride],
                    );
                }

                {
                    let shadow_ts = profiler.pass_timestamp_writes("shadow_pass");
                    let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.shadow_map.depth_views[cascade],
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: shadow_ts,
                        occlusion_query_set: None,
                    });

                    shadow_pass.set_pipeline(&self.shadow_map.pipeline);

                    for (i, entry) in shadow_nodes.iter().enumerate() {
                        if i >= max { break; }
                        let offset = (i * stride) as u32;
                        shadow_pass.set_bind_group(0, &self.shadow_map.uniform_bind_group, &[offset]);
                        shadow_pass.set_vertex_buffer(0, shadow_vbs[entry.vb_idx].slice(..));
                        shadow_pass.set_index_buffer(shadow_ibs[entry.ib_idx].slice(..), wgpu::IndexFormat::Uint32);
                        shadow_pass.draw_indexed(0..entry.index_count, 0, 0..1);
                    }
                }
            }
        }

        profiler.end("shadow_pass");

        // Upload immediate-mode 2D data
        profiler.begin("upload_geometry");
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
        profiler.end("upload_geometry");

        // ============================================================
        // HDR pass: sky + 3D + scene → linear HDR offscreen RT.
        // ============================================================
        // The composite-tonemap pass downstream reads this RT and
        // writes the final image to the sRGB surface. Keeping the
        // intermediate radiance in HDR sets up a future bloom pass
        // and means tonemap + sRGB encode happen exactly once, in
        // one place.
        profiler.begin("main_hdr_pass");
        {
            // HDR clear: the user's clear_color is in 0-1 srgb-ish
            // range; treat it as the linear background for the HDR
            // RT. After tonemap it ends up roughly the same shade.
            let hdr_ts = profiler.pass_timestamp_writes("main_hdr_pass");
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
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.velocity_rt_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Zero velocity = stationary pixel.
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.albedo_rt_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Clear to zero albedo — pixels the scene
                            // doesn't cover (before sky writes) absorb
                            // indirect light fully. Sky then writes 0
                            // too so SSGI rays landing on sky don't
                            // re-tint bounce by background radiance.
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
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
                timestamp_writes: hdr_ts,
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
        profiler.end("main_hdr_pass");

        // ============================================================
        // SSAO: half-res GTAO sampling a hierarchical linear-depth
        // pyramid. Build hiz (linearize + 4 min-downsamples), then
        // dispatch the GTAO compute pass.
        // ============================================================
        profiler.begin("post_fx");
        let surf_w = self.surface_config.width;
        let surf_h = self.surface_config.height;
        if self.ssao_enabled {
            let p = &self.current_proj_matrix;
            let p00 = p[0][0];
            let p11 = p[1][1];
            let p20 = p[2][0];
            let p21 = p[2][1];
            let p22 = p[2][2];
            let p32 = p[3][2];
            let half_w = (surf_w / 2).max(1);
            let half_h = (surf_h / 2).max(1);

            // --- Hi-Z build: linearize depth into mip 0 -----------------
            let lin_params = HizLinearizeParams {
                params: [1.0 / half_w as f32, 1.0 / half_h as f32, p22, p32],
                size: [half_w, half_h, 0, 0],
            };
            self.queue.write_buffer(&self.hiz_linearize_uniform_buffer, 0, bytemuck::bytes_of(&lin_params));
            if self.hiz_linearize_bg_cache.is_none() {
                self.hiz_linearize_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hiz_linearize_bg"),
                    layout: &self.hiz_linearize_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.hiz_linearize_uniform_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                    ],
                }));
            }
            {
                let bg = self.hiz_linearize_bg_cache.as_ref().unwrap();
                let ts = profiler.compute_pass_timestamp_writes("hiz_linearize_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hiz_linearize_pass"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.hiz_linearize_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups((half_w + 7) / 8, (half_h + 7) / 8, 1);
            }

            // --- Hi-Z build: downsample mip i -> mip i+1 ----------------
            for i in 0..(HIZ_MIP_COUNT - 1) as usize {
                let dst_w = (half_w >> (i + 1)).max(1);
                let dst_h = (half_h >> (i + 1)).max(1);
                let ds_params = HizDownsampleParams {
                    size: [dst_w, dst_h, 0, 0],
                };
                self.queue.write_buffer(&self.hiz_downsample_uniform_buffers[i], 0, bytemuck::bytes_of(&ds_params));
                if self.hiz_downsample_bg_cache[i].is_none() {
                    self.hiz_downsample_bg_cache[i] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hiz_downsample_bg"),
                        layout: &self.hiz_downsample_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: self.hiz_downsample_uniform_buffers[i].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hiz_views[i]) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.hiz_views[i + 1]) },
                        ],
                    }));
                }
                let bg = self.hiz_downsample_bg_cache[i].as_ref().unwrap();
                let ts_label: &'static str = match i {
                    0 => "hiz_downsample_pass_1",
                    1 => "hiz_downsample_pass_2",
                    2 => "hiz_downsample_pass_3",
                    _ => "hiz_downsample_pass_4",
                };
                let ts = profiler.compute_pass_timestamp_writes(ts_label);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(ts_label),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.hiz_downsample_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
            }

            // --- SSAO (compute GTAO, samples Hi-Z pyramid) --------------
            let ld = self.lighting_uniforms.light_dir;
            let v = &self.current_view_matrix;
            let light_dir_vs = [
                v[0][0]*ld[0] + v[1][0]*ld[1] + v[2][0]*ld[2],
                v[0][1]*ld[0] + v[1][1]*ld[1] + v[2][1]*ld[2],
                v[0][2]*ld[0] + v[1][2]*ld[1] + v[2][2]*ld[2],
                0.0,
            ];
            let sp = SsaoParams {
                params: [
                    1.0 / half_w as f32,
                    1.0 / half_h as f32,
                    self.ssao_radius,
                    self.ssao_strength,
                ],
                proj_row01: [p00, p11, p20, p21],
                proj_z: [p22, p32, 1.0 / p00, 1.0 / p11],
                light_dir_vs,
                size: [half_w, half_h, 0, 0],
            };
            self.queue.write_buffer(&self.ssao_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            if self.ssao_bg_cache.is_none() {
                self.ssao_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssao_bg"),
                    layout: &self.ssao_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.ssao_uniform_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ssao_rt_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.hiz_views[1]) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.hiz_views[2]) },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.hiz_views[3]) },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.hiz_views[4]) },
                    ],
                }));
            }
            let bg = self.ssao_bg_cache.as_ref().unwrap();

            let ssao_ts = profiler.compute_pass_timestamp_writes("ssao_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssao_pass"),
                timestamp_writes: ssao_ts,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((half_w + 7) / 8, (half_h + 7) / 8, 1);
        }

        // ============================================================
        // SSAO bilateral blur: smooth the noisy GTAO output while
        // preserving depth edges (depth-guided bilateral filter).
        // Reads ssao_rt → writes ssao_blur_rt.
        // ============================================================
        if self.ssao_enabled {
            // texel_size is the size of one SSAO RT texel (half-res).
            let ao_w = (surf_w / 2).max(1) as f32;
            let ao_h = (surf_h / 2).max(1) as f32;
            let bp = SsaoBlurParams {
                params: [1.0 / ao_w, 1.0 / ao_h, 0.05, 0.0],
            };
            self.queue.write_buffer(&self.ssao_blur_uniform_buffer, 0, bytemuck::bytes_of(&bp));

            if self.ssao_blur_bg_cache.is_none() {
                self.ssao_blur_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssao_blur_bg"),
                    layout: &self.ssao_blur_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.ssao_blur_uniform_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ssao_rt_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                    ],
                }));
            }
            let bg = self.ssao_blur_bg_cache.as_ref().unwrap();

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_blur_rt_view,
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
            pass.set_pipeline(&self.ssao_blur_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        } else {
            // SSAO disabled — clear the blur RT to WHITE so the
            // composite pass samples "no occlusion". Cheaper than a
            // full blur pass; the clear is the only GPU work.
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_blur_disabled_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_blur_rt_view,
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
        }

        // ============================================================
        // SSR: view-space ray march of the depth buffer + HDR sample.
        // ============================================================
        if self.ssr_enabled {
            let inv_proj = self.current_inv_proj_matrix;
            let sp = SsrParams {
                inv_proj,
                proj: self.current_proj_matrix,
                params: [self.ssr_strength, 8.0, 32.0, self.taa_frame_index as f32],
            };
            self.queue.write_buffer(&self.ssr_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            if self.ssr_bg_cache.is_none() {
                self.ssr_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view) },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    ],
                }));
            }
            let bg = self.ssr_bg_cache.as_ref().unwrap();
            let ssr_ts = profiler.pass_timestamp_writes("ssr_pass");
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
                timestamp_writes: ssr_ts,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ssr_pipeline);
            pass.set_bind_group(0, bg, &[]);
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
        // SSGI: screen-space global illumination (single-bounce indirect
        // diffuse). Half-res ray march similar to SSR but for diffuse.
        // ============================================================
        if self.ssgi_enabled {
            let inv_proj = self.current_inv_proj_matrix;
            let sp = SsgiParams {
                inv_proj,
                proj: self.current_proj_matrix,
                params: [self.ssgi_intensity, self.ssgi_radius, 8.0, self.taa_frame_index as f32],
            };
            self.queue.write_buffer(&self.ssgi_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            if self.ssgi_bg_cache.is_none() {
                self.ssgi_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssgi_bg"),
                    layout: &self.ssgi_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.ssgi_uniform_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    ],
                }));
            }
            let bg = self.ssgi_bg_cache.as_ref().unwrap();
            let ssgi_ts = profiler.pass_timestamp_writes("ssgi_pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssgi_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssgi_rt_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ssgi_ts,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ssgi_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        } else {
            // SSGI disabled — clear the RT so TAA's read returns 0
            // (transparent black).
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssgi_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssgi_rt_view,
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
        // SSGI temporal denoiser: blend noisy SSGI with reprojected
        // history. Reads ssgi_rt + ssgi_history[prev] + velocity_rt,
        // writes ssgi_history[current]. TAA reads the denoised result.
        // ============================================================
        if self.ssgi_enabled {
            let prev_idx = 1 - self.ssgi_history_idx;
            let cur_idx = self.ssgi_history_idx;

            // First frame (taa_frame_index == 0): use alpha=1.0 to
            // initialize history from the current noisy frame rather
            // than blending with uninitialized zeros.
            let alpha = if self.taa_frame_index == 0 { 1.0_f32 } else { 0.1_f32 };
            let tp = SsgiTemporalParams {
                params: [alpha, 0.1, 0.0, 0.0],
            };
            self.queue.write_buffer(&self.ssgi_temporal_uniform_buffer, 0, bytemuck::bytes_of(&tp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssgi_temporal_bg"),
                layout: &self.ssgi_temporal_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ssgi_temporal_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ssgi_rt_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.ssgi_history_views[prev_idx]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssgi_temporal_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssgi_history_views[cur_idx],
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
            pass.set_pipeline(&self.ssgi_temporal_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // The TAA pass (and composite, on the TAA-off path) reads the
        // denoised SSGI from the current history texture, or raw
        // ssgi_rt if SSGI is off.
        let ssgi_composite_view = if self.ssgi_enabled {
            &self.ssgi_history_views[self.ssgi_history_idx]
        } else {
            &self.ssgi_rt_view
        };

        // ============================================================
        // Bloom: progressive downsample (Karis-thresholded first tap)
        // followed by additive upsample back up the chain.
        // ============================================================
        if self.bloom_enabled {
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

            let bloom_ts = profiler.pass_timestamp_writes("bloom_pass");
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
                timestamp_writes: bloom_ts,
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

            let bloom_up_ts = profiler.pass_timestamp_writes("bloom_pass");
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
                timestamp_writes: bloom_up_ts,
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
        } // end if self.bloom_enabled

        // ============================================================
        // Scene-compose pass: merge HDR + SSR + SSGI*albedo + bloom
        // + fog + sun shafts into composed_rt. Runs unconditionally
        // so both the TAA-on path (TAA consumes this) and the
        // TAA-off path (composite consumes this) get the same
        // atmospherics + post-effects.
        // ============================================================
        let inv_vp_current = self.current_inv_vp_matrix;
        // Sun shaft screen-space position. Project a point far along
        // the sun direction through the current VP. If behind the
        // camera (clip.w ≤ 0), the sun is off-screen → disable.
        let sun_dir = self.lighting_uniforms.light_dir;
        let sun_world = [sun_dir[0] * 1000.0, sun_dir[1] * 1000.0, sun_dir[2] * 1000.0, 1.0];
        let clip = mat4_mul_vec4(&self.current_vp_matrix, &sun_world);
        let (sun_uv, shaft_strength_eff) = if clip[3] > 0.0 {
            let ndc_x = clip[0] / clip[3];
            let ndc_y = clip[1] / clip[3];
            let u = ndc_x * 0.5 + 0.5;
            let v = 1.0 - (ndc_y * 0.5 + 0.5);
            // Allow off-screen suns to still cast shafts that streak
            // in from the edge — clamp to a small margin beyond ±[0,1]
            // rather than disabling outright.
            let off = u < -1.0 || u > 2.0 || v < -1.0 || v > 2.0;
            if off { ([0.0, 0.0], 0.0) } else { ([u, v], self.sun_shaft_strength) }
        } else {
            ([0.0, 0.0], 0.0)
        };
        // When bloom_enabled is false we skip the downsample/upsample
        // chain entirely; forcing the composite's bloom multiplier to
        // 0 here means stale bloom_mip_views[0] contents contribute
        // nothing visually.
        let effective_bloom_intensity = if self.bloom_enabled { self.bloom_intensity } else { 0.0 };
        let cp = SceneComposeParams {
            misc: [effective_bloom_intensity, 0.0, 0.0, 0.0],
            inv_vp: inv_vp_current,
            fog_color_density: [
                self.fog_color[0], self.fog_color[1], self.fog_color[2], self.fog_density,
            ],
            fog_params: [self.fog_height_ref, self.fog_height_falloff, 0.0, 0.0],
            sun_shaft_uv_strength: [
                sun_uv[0], sun_uv[1], shaft_strength_eff, self.sun_shaft_decay,
            ],
            sun_shaft_color: [
                self.sun_shaft_color[0], self.sun_shaft_color[1], self.sun_shaft_color[2], 0.0,
            ],
        };
        self.queue.write_buffer(&self.scene_compose_uniform_buffer, 0, bytemuck::bytes_of(&cp));
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene_compose_bg"),
                layout: &self.scene_compose_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.scene_compose_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.ssr_rt_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(ssgi_composite_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.bloom_mip_views[0]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                ],
            });
            // NOTE: GPU timestamp deliberately not requested on this pass.
            // Empirically (sponza, Metal) the reported delta was ~249 ms
            // for what should be a sub-millisecond fullscreen pass. Likely
            // the end-of-pass write is synchronized to a later barrier
            // and includes idle time. CPU-side timing via the enclosing
            // `post_fx` phase captures the cost adequately.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_compose_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composed_rt_view,
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
            pass.set_pipeline(&self.scene_compose_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // TAA pass: reprojection + neighborhood clamp on composed_rt.
        // Skipped when TAA is off — composite reads composed_rt
        // directly and gets the same composed / fog / shafts output.
        // ============================================================
        let taa_dst_idx = self.taa_current_idx;
        let taa_src_idx = 1 - self.taa_current_idx;

        if self.taa_enabled {
            // TSR upscale needs a longer history window than full-res
            // TAA because each frame contributes 1/4 the per-pixel
            // sample density. 0.05 = ~20-frame effective window.
            let steady = if self.tsr_enabled { 0.05 } else { 0.1 };
            let alpha = if self.taa_frame_index < 4 { 1.0 } else { steady };
            let tp = TaaParams {
                params: [alpha, 0.0, 0.0, 0.0],
                inv_vp: inv_vp_current,
                prev_vp: self.prev_vp_matrix,
            };
            self.queue.write_buffer(&self.taa_uniform_buffer, 0, bytemuck::bytes_of(&tp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa_bg"),
                layout: &self.taa_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.taa_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.composed_rt_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.taa_views[taa_src_idx]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });
            let taa_ts = profiler.pass_timestamp_writes("taa_pass");
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
                timestamp_writes: taa_ts,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.taa_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // DoF pass: variable-radius Poisson disc blur driven by CoC
        // Reads TAA output (or hdr_rt if TAA off) + depth → dof_rt
        // ============================================================
        let pre_dof_view = if self.taa_enabled {
            &self.taa_views[taa_dst_idx]
        } else {
            &self.hdr_rt_view
        };

        if self.dof_enabled && self.dof_aperture > 0.0 {
            let inv_proj = self.current_inv_proj_matrix;
            let dp = DofParams {
                params: [self.dof_focus_distance, self.dof_aperture, self.dof_max_blur, 0.0],
                inv_proj,
            };
            self.queue.write_buffer(&self.dof_uniform_buffer, 0, bytemuck::bytes_of(&dp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dof_bg"),
                layout: &self.dof_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.dof_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(pre_dof_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.dof_rt_view,
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
            pass.set_pipeline(&self.dof_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // Motion blur pass: 8-tap directional blur along velocity
        // Reads upstream color + velocity_rt → motion_blur_rt
        // ============================================================
        let pre_mblur_view = if self.dof_enabled && self.dof_aperture > 0.0 {
            &self.dof_rt_view
        } else if self.taa_enabled {
            &self.taa_views[taa_dst_idx]
        } else {
            &self.hdr_rt_view
        };

        if self.motion_blur_enabled && self.motion_blur_strength > 0.0 {
            let mbp = MotionBlurParams {
                params: [self.motion_blur_strength, self.motion_blur_max_blur, 0.0, 0.0],
            };
            self.queue.write_buffer(&self.motion_blur_uniform_buffer, 0, bytemuck::bytes_of(&mbp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("motion_blur_bg"),
                layout: &self.motion_blur_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.motion_blur_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(pre_mblur_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.motion_blur_rt_view,
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
            pass.set_pipeline(&self.motion_blur_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // SSS pass: chromatic disc blur (skin / wax / leaves)
        // Reads upstream color + depth → sss_rt.
        // Runs after motion blur so it applies to the fully composited
        // motion state, not to individual geometry.
        // ============================================================
        let pre_sss_view = if self.motion_blur_enabled && self.motion_blur_strength > 0.0 {
            &self.motion_blur_rt_view
        } else if self.dof_enabled && self.dof_aperture > 0.0 {
            &self.dof_rt_view
        } else if self.taa_enabled {
            &self.taa_views[taa_dst_idx]
        } else {
            &self.hdr_rt_view
        };

        if self.sss_enabled && self.sss_strength > 0.0 {
            let sp = SssParams {
                params: [self.sss_strength, self.sss_width, 500.0, 0.0],
            };
            self.queue.write_buffer(&self.sss_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sss_bg"),
                layout: &self.sss_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.sss_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(pre_sss_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sss_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.sss_rt_view,
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
            pass.set_pipeline(&self.sss_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // Composite pass: tonemap (ACES + sRGB encode)
        // ============================================================
        let composite_src_view = if self.sss_enabled && self.sss_strength > 0.0 {
            &self.sss_rt_view
        } else if self.motion_blur_enabled && self.motion_blur_strength > 0.0 {
            &self.motion_blur_rt_view
        } else if self.dof_enabled && self.dof_aperture > 0.0 {
            &self.dof_rt_view
        } else if self.taa_enabled {
            &self.taa_views[taa_dst_idx]
        } else {
            // TAA off: read the composed buffer directly so SSR /
            // SSGI / bloom / fog / shafts still land in the final
            // image. Before the scene-compose split this branch
            // read raw hdr_rt and silently dropped those effects.
            &self.composed_rt_view
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
                    // Wide clamp — without SSGI, Sponza's shadowed
                    // corridors have ~7× less average luma than its
                    // sunlit courtyard, so exposure needs to span
                    // the same range to keep perceived brightness
                    // stable across rotations.
                    0.1,
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
            misc: [self.taa_frame_index as f32, self.sharpen_strength, 0.0, 0.0],
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
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.ssao_blur_rt_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
            ],
        });
        {
            let final_composite_ts = profiler.pass_timestamp_writes("final_composite_pass");
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
                timestamp_writes: final_composite_ts,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        profiler.end("post_fx");

        // ============================================================
        // 2D pass: immediate-mode 2D geometry on top of composited image
        // ============================================================
        profiler.begin("overlay_2d");
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
        profiler.end("overlay_2d");

        profiler.resolve(&mut encoder);

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
        // Swap SSGI temporal history ping-pong so next frame reads
        // what we just wrote and writes to the other buffer.
        if self.ssgi_enabled {
            self.ssgi_history_idx = 1 - self.ssgi_history_idx;
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
        self.register_texture_kind(width, height, data, false)
    }

    /// Single-mip texture for dynamically updated atlases.
    pub fn register_texture_no_mips(&mut self, width: u32, height: u32, data: &[u8]) -> u32 {
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("atlas_no_mips"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor, data,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas_bg"), layout: &self.texture_bind_group_layout,
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

    /// Replace an existing no-mips texture in-place.
    pub fn replace_texture_no_mips(&mut self, idx: u32, width: u32, height: u32, data: &[u8]) {
        let i = idx as usize;
        if i >= self.textures.len() { return; }
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("atlas_replaced"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor, data,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas_replaced_bg"), layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.textures[i] = texture;
        self.texture_bind_groups[i] = bind_group;
        self.texture_sizes[i] = (width, height);
    }

    /// Register a texture with optional normal-map preprocessing.
    ///
    /// For normal maps (is_normal_map=true), mip chain is built with
    /// vector-space averaging instead of scalar RGB averaging, and
    /// per-mip variance (1 - |vector_avg|²) is baked into the alpha
    /// channel. The shader reads alpha as a Toksvig-style σ² addition
    /// that accumulates normal-direction disagreement across the
    /// footprint the sampler ends up integrating — the simplified
    /// scalar LEADR/LEAN filter. Alpha is unused by glTF normal maps
    /// (they carry (x,y,z) in RGB) so we can safely repurpose it.
    pub fn register_texture_kind(
        &mut self,
        width: u32,
        height: u32,
        data: &[u8],
        is_normal_map: bool,
    ) -> u32 {
        let max_dim = if width > height { width } else { height };
        // On Android/Vulkan, multi-level mipmap upload can fail silently.
        // Use single mip for 2D textures; only generate mipmaps on desktop.
        #[cfg(target_os = "android")]
        let mip_count = 1u32;
        #[cfg(not(target_os = "android"))]
        let mip_count = (max_dim as f32).log2().floor() as u32 + 1;

        // Generate mip chain data
        let mut mip_data = Vec::with_capacity(data.len() * 2); // overallocate
        if is_normal_map {
            // Level 0: normalize input RGB and clear alpha to 0 (no
            // variance at the finest level — each texel is assumed unit).
            mip_data.reserve(data.len());
            for i in 0..(width as usize * height as usize) {
                let r = data[i * 4];
                let g = data[i * 4 + 1];
                let b = data[i * 4 + 2];
                mip_data.push(r);
                mip_data.push(g);
                mip_data.push(b);
                mip_data.push(0);
            }
        } else {
            mip_data.extend_from_slice(data);
        }
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
                    if is_normal_map {
                        // Decode 4 children to signed [-1, 1] vectors
                        let dec = |r: u8, g: u8, b: u8| -> [f32; 3] {
                            [
                                r as f32 * (2.0 / 255.0) - 1.0,
                                g as f32 * (2.0 / 255.0) - 1.0,
                                b as f32 * (2.0 / 255.0) - 1.0,
                            ]
                        };
                        let idx = |sx: usize, sy: usize| -> usize {
                            prev_offset + (sy * pw + sx) * 4
                        };
                        let n00 = dec(mip_data[idx(sx, sy)], mip_data[idx(sx, sy) + 1], mip_data[idx(sx, sy) + 2]);
                        let n10 = dec(mip_data[idx(sx1, sy)], mip_data[idx(sx1, sy) + 1], mip_data[idx(sx1, sy) + 2]);
                        let n01 = dec(mip_data[idx(sx, sy1)], mip_data[idx(sx, sy1) + 1], mip_data[idx(sx, sy1) + 2]);
                        let n11 = dec(mip_data[idx(sx1, sy1)], mip_data[idx(sx1, sy1) + 1], mip_data[idx(sx1, sy1) + 2]);
                        // Previous-mip baked variances
                        let v00 = mip_data[idx(sx, sy) + 3] as f32 / 255.0;
                        let v10 = mip_data[idx(sx1, sy) + 3] as f32 / 255.0;
                        let v01 = mip_data[idx(sx, sy1) + 3] as f32 / 255.0;
                        let v11 = mip_data[idx(sx1, sy1) + 3] as f32 / 255.0;
                        // Average the vectors
                        let avg_x = (n00[0] + n10[0] + n01[0] + n11[0]) * 0.25;
                        let avg_y = (n00[1] + n10[1] + n01[1] + n11[1]) * 0.25;
                        let avg_z = (n00[2] + n10[2] + n01[2] + n11[2]) * 0.25;
                        let len_sq = avg_x * avg_x + avg_y * avg_y + avg_z * avg_z;
                        let len = len_sq.sqrt().max(1e-6);
                        // Normalize direction (what the shader reads as
                        // the shading normal). Re-encode to [0, 255].
                        let encode = |v: f32| -> u8 {
                            ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
                        };
                        mip_data.push(encode(avg_x / len));
                        mip_data.push(encode(avg_y / len));
                        mip_data.push(encode(avg_z / len));
                        // Variance at this mip = disagreement among the
                        // 4 children (1 - |avg|²) PLUS the weighted mean
                        // of the children's own variances. Both live in
                        // [0, 1]; combined variance clamped.
                        let v_children_avg = (v00 + v10 + v01 + v11) * 0.25;
                        let v_local = (1.0 - len_sq).max(0.0);
                        let v_out = (v_local + v_children_avg).min(1.0);
                        mip_data.push((v_out * 255.0).round().clamp(0.0, 255.0) as u8);
                    } else {
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
        }

        let texture = if mip_count == 1 {
            // Simple path: single mip level, use create_texture_with_data
            self.device.create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
                    label: Some("registered_texture"),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &mip_data[..((width * height * 4) as usize)],
            )
        } else {
            // Multi-mip path: create texture, upload each level
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("registered_texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mut lw = width;
            let mut lh = height;
            for level in 0..mip_count {
                let offset = mip_offsets[level as usize];
                let level_size = (lw * lh * 4) as usize;
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex, mip_level: level,
                        origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                    },
                    &mip_data[offset..offset + level_size],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0, bytes_per_row: Some(4 * lw), rows_per_image: Some(lh),
                    },
                    wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
                );
                lw = if lw > 1 { lw / 2 } else { 1 };
                lh = if lh > 1 { lh / 2 } else { 1 };
            }
            tex
        };

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

        let w = self.logical_width as f32;
        let h = self.logical_height as f32;
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
            // Jitter is sub-pixel in *render* space — when TSR is
            // on the G-buffer is half-res, so each render pixel
            // covers 2× surface pixels and the offset must scale
            // accordingly. render_extent() returns surface size
            // when TSR is off.
            let (rw, rh) = self.render_extent();
            let render_w = rw.max(1) as f32;
            let render_h = rh.max(1) as f32;
            // proj is column-major; column 2 row 0/1 are the
            // perspective / Z-coupling slots. Adding a constant NDC
            // offset there shifts the whole frustum by jitter px.
            proj[2][0] += (jx * 2.0) / render_w;
            proj[2][1] += (jy * 2.0) / render_h;
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
        self.current_inv_proj_matrix = mat4_invert(proj);
        self.current_inv_vp_matrix = mat4_invert(vp);
        self.current_camera_pos = [pos_x, pos_y, pos_z];

        // Mirror camera pos into lighting uniforms so the scene shader
        // can compute V for GGX specular. Preserve the .w slot — it
        // holds the env_intensity multiplier (set via load_env_from_hdr).
        let env_intensity_w = self.lighting_uniforms.camera_pos[3];
        self.lighting_uniforms.camera_pos = [pos_x, pos_y, pos_z, env_intensity_w];
        // Pass the current cascade shadow VPs and view matrix (computed
        // in end_frame_with_scene) so the scene shader's CSM lookup
        // lands on the right cascade map.
        self.lighting_uniforms.shadow_cascade_vps = self.shadow_map.light_vps;
        self.lighting_uniforms.shadow_cascade_splits = [
            self.shadow_map.cascade_splits[0],
            self.shadow_map.cascade_splits[1],
            self.shadow_map.cascade_splits[2],
            0.0,
        ];
        self.lighting_uniforms.shadow_view_matrix = self.current_view_matrix;
        self.queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_uniforms),
        );

        self.queue.write_buffer(
            &self.uniform_buffer_3d,
            0,
            bytemuck::bytes_of(&Uniforms3D { mvp: vp, model: IDENTITY_MAT4, prev_mvp: self.prev_vp_matrix, model_tint: [1.0, 1.0, 1.0, 1.0] }),
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
                mesh.metallic_roughness_texture_idx.is_some(),
                mesh.alpha_cutoff,
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
                bytemuck::bytes_of(&Uniforms3D { mvp: model_mvp, model: model_matrix, prev_mvp: model_mvp, model_tint: tint }),
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
                contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4, model: IDENTITY_MAT4, prev_mvp: IDENTITY_MAT4, model_tint: [1.0; 4] }),
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

    /// Logical (points / CSS px) width — what user code sees via
    /// `screenWidth` and what 2D HUD coordinates are expressed in.
    /// On HiDPI displays the underlying render target is larger (see
    /// `physical_width`).
    pub fn width(&self) -> u32 {
        self.logical_width
    }

    pub fn height(&self) -> u32 {
        self.logical_height
    }

    /// Physical pixel dimensions of the swapchain and post-process
    /// render targets. Always equal to `width`/`height` on non-HiDPI
    /// platforms; `logical * scale_factor` on Retina/Web.
    pub fn physical_width(&self) -> u32 {
        self.surface_config.width
    }

    pub fn physical_height(&self) -> u32 {
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

    /// Dump a shadow cascade's depth texture to a grayscale PNG for debugging.
    /// Depth 0.0 (near) → white, depth 1.0 (far / clear) → black.
    /// `cascade` selects which cascade to dump (0, 1, or 2).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dump_shadow_map(&self, path: &str) {
        self.dump_shadow_cascade(path, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn dump_shadow_cascade(&self, path: &str, cascade: usize) {
        let cascade = cascade.min(crate::shadows::NUM_CASCADES - 1);
        let size = crate::shadows::CASCADE_MAP_SIZE;
        let bytes_per_pixel = 4u32; // Depth32Float = 4 bytes
        let unpadded_bpr = size * bytes_per_pixel;
        let padded_bpr = (unpadded_bpr + 255) & !255;
        let buf_size = (padded_bpr * size) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_dump_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shadow_dump_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.shadow_map.depth_textures[cascade],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            // Convert f32 depth values to grayscale RGB
            let mut rgb = Vec::with_capacity((size * size * 3) as usize);
            for row in 0..size {
                let row_start = (row * padded_bpr) as usize;
                for col in 0..size {
                    let offset = row_start + (col * bytes_per_pixel) as usize;
                    let depth = f32::from_le_bytes([
                        data[offset], data[offset+1], data[offset+2], data[offset+3],
                    ]);
                    // depth 0 = near (white), depth 1 = far/clear (black)
                    let gray = ((1.0 - depth.clamp(0.0, 1.0)) * 255.0) as u8;
                    rgb.push(gray);
                    rgb.push(gray);
                    rgb.push(gray);
                }
            }
            drop(data);
            if let Some(png) = encode_png_simple(size, size, &rgb) {
                let _ = std::fs::write(path, &png);
            }
        }
        staging.unmap();
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
