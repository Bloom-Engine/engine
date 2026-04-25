use wgpu::util::DeviceExt;
use std::collections::HashMap;

mod shaders;
use shaders::*;

pub mod shader_include;
pub mod shader_library;
pub mod material_pipeline;
pub mod material_system;
pub mod graph;
pub mod transient;
pub mod impulse_field;
pub mod hot_reload;

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

mod formats;
use formats::{
    DEPTH_FORMAT, HDR_FORMAT, SSAO_FORMAT, MATERIAL_FORMAT,
    HIZ_FORMAT, VELOCITY_FORMAT, BLOOM_MIP_COUNT, HIZ_MIP_COUNT,
    create_depth_texture, create_hdr_rt, create_material_rt,
    create_albedo_rt, create_velocity_rt, create_ssr_rt,
    create_ssr_history_textures,
    create_ssgi_rt, create_probe_trace_tex, create_probe_history_textures,
    probe_grid_dims, PROBE_TILE_SIZE, PROBE_OCT_SIZE, PROBE_OCT_TEXELS,
    create_mesh_card_atlas, create_mesh_card_emissive_atlas,
    create_mesh_card_radiance_atlas,
    CARD_ATLAS_SIZE, CARD_SLOT_SIZE, CARD_SLOTS_PER_ROW, CARD_MAX_SLOTS,
    CARD_AXES_PER_MESH, create_mesh_sdf_texture, MESH_SDF_RES,
    create_scene_sdf_clipmap, SCENE_SDF_CLIPMAP_RES,
    SCENE_SDF_CLIPMAP_EXTENT, SCENE_SDF_CLIPMAP_REBAKE_THRESHOLD,
    create_wsrc_atlas, WSRC_GRID_RES, WSRC_CASCADE_COUNT,
    WSRC_CASCADE_EXTENTS, WSRC_REBAKE_THRESHOLD,
    create_taa_textures,
    create_ssao_rt, create_ssao_blur_rt, create_ssao_history_textures, create_sss_rt,
    create_exposure_textures, create_composed_rt, create_dof_rt,
    create_linear_depth_hiz_chain, create_bloom_chain, halton,
};

mod types;
pub use types::{Vertex2D, Vertex3D, SceneMaterialUniforms, RenderMode};
use types::{
    MAX_UNIFORM_SLOTS, MAX_DIR_LIGHTS, MAX_POINT_LIGHTS,
    Uniforms2D, Uniforms3D, DirLight, PointLight, LightingUniforms,
    DrawCall2D, DrawCall3D,
};



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
    /// x = half-res width, y = half-res height, z = frame phase
    /// (`frame_index % 4`), w = "force-refresh" flag (non-zero on
    /// first few frames, resize, or any host-side history invalidation).
    size: [u32; 4],
    /// x = temporal blend alpha (≈0.25 steady, 4-frame EMA).
    /// y = per-frame Halton-5 rotation of the direction basis
    /// (uncorrelated with TAA's Halton-2/3 pixel jitter).
    /// zw unused.
    temporal: [f32; 4],
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

// Ticket 007a: per-pass uniform params for the screen-probe SSGI chain.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbePlaceParams {
    inv_view: [[f32; 4]; 4],
    /// x = proj[0][0], y = proj[1][1], z = proj[2][0], w = proj[2][1]
    proj_row01: [f32; 4],
    /// x = half_w, y = half_h, z = grid_w, w = grid_h
    size: [u32; 4],
    /// x = frame_index, y = tile_size (16.0), zw unused
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeTraceParams {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    proj_row01: [f32; 4],
    size: [u32; 4],
    /// x = frame_index, y = intensity, z = max_march_t (world units),
    /// w = firefly luma cap.
    params: [f32; 4],
    /// Ticket 007b HW path — sun direction in world space (xyz) +
    /// intensity (w). Ignored by the SW-HiZ shader, consumed by HW + SDF.
    sun_dir: [f32; 4],
    /// Sun colour (xyz), reserved (w). Used for analytical NdotL at hit.
    sun_color: [f32; 4],
    /// Sky dome colour (xyz), reserved (w). Flat analytic sky for
    /// up-facing hits where the ray hits a surface with an up normal.
    sky_color: [f32; 4],
    /// Ticket 014 V3 — clipmap origin (xyz) + full extent (w). Used
    /// by the SDF sphere-trace variant to map world-space march
    /// positions into clipmap sample UVs.
    clipmap: [f32; 4],
    /// Ticket 014 V6/V13 — WSRC cascade cubes. Each entry is
    /// (origin xyz, extent w). Cascades are ordered near→far; miss
    /// paths pick the smallest cascade whose cube contains the
    /// ray-terminal position. `extent = 0.0` marks an unbaked
    /// cascade (shader falls through to the next one). HW + Hi-Z
    /// carry these for uniform-buffer size parity; Hi-Z ignores
    /// them.
    wsrc_cascades: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeTemporalParams {
    /// x = alpha (EMA), y = force_refresh (1→alpha=1), z = grid_w (f32), w = grid_h (f32)
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeResolveParams {
    inv_view: [[f32; 4]; 4],
    proj_row01: [f32; 4],
    /// x = half_w, y = half_h, z = grid_w, w = grid_h
    size: [u32; 4],
    /// x = tile_size (16.0), y = intensity, zw unused
    params: [f32; 4],
}

/// On-GPU `ProbeHeader` layout (must match PROBE_HELPERS_WGSL's struct).
/// 32 bytes per probe.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeHeaderCpu {
    world_pos: [f32; 4],
    normal: [f32; 4],
}

/// Ticket 013 V3 — CardCaptureParams. ortho_vp + base_color + emissive.
/// 96 bytes; we allocate 128 for uniform-alignment headroom.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CardCaptureParams {
    ortho_vp: [[f32; 4]; 4],
    base_color: [f32; 4],  // rgb = factor, w = has_base_texture (0/1)
    emissive: [f32; 4],    // rgb = emissive_factor, w = has_emissive_texture (0/1)
}

/// Ticket 013 V2 — orthographic projection for the 6 signed axes.
/// `face_axis` encoding:
///   0 → +X, 1 → -X, 2 → +Y, 3 → -Y, 4 → +Z, 5 → -Z.
/// For each face we pick the two orthogonal AABB axes and map them
/// to clip-space [-1, +1]. The ±pair for each axis differ only in
/// the sign of the "u" clip axis so that when the HW shader picks
/// axis N at hit and projects the hit into card UV, the UV lines up
/// with the mesh geometry as seen FROM that face.
fn build_card_ortho_v2(face_axis: u32, bmin: [f32; 3], bmax: [f32; 3]) -> [[f32; 4]; 4] {
    let wx = (bmax[0] - bmin[0]).max(1e-4);
    let wy = (bmax[1] - bmin[1]).max(1e-4);
    let wz = (bmax[2] - bmin[2]).max(1e-4);
    let cx = bmin[0] + bmax[0];
    let cy = bmin[1] + bmax[1];
    let cz = bmin[2] + bmax[2];
    match face_axis {
        // +X → project onto YZ; clip.x = y, clip.y = z.
        0 => [
            [0.0, 0.0, 0.0, 0.0],
            [2.0/wy, 0.0, 0.0, 0.0],
            [0.0, 2.0/wz, 0.0, 0.0],
            [-cy/wy, -cz/wz, 0.0, 1.0],
        ],
        // -X → flip u so the -X view is the mirror of +X. clip.x = -y, clip.y = z.
        1 => [
            [0.0, 0.0, 0.0, 0.0],
            [-2.0/wy, 0.0, 0.0, 0.0],
            [0.0, 2.0/wz, 0.0, 0.0],
            [cy/wy, -cz/wz, 0.0, 1.0],
        ],
        // +Y → project onto XZ; clip.x = x, clip.y = z.
        2 => [
            [2.0/wx, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0/wz, 0.0, 0.0],
            [-cx/wx, -cz/wz, 0.0, 1.0],
        ],
        // -Y → flip u; clip.x = -x, clip.y = z.
        3 => [
            [-2.0/wx, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0/wz, 0.0, 0.0],
            [cx/wx, -cz/wz, 0.0, 1.0],
        ],
        // +Z → project onto XY; clip.x = x, clip.y = y.
        4 => [
            [2.0/wx, 0.0, 0.0, 0.0],
            [0.0, 2.0/wy, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-cx/wx, -cy/wy, 0.0, 1.0],
        ],
        // -Z → flip u; clip.x = -x, clip.y = y.
        _ => [
            [-2.0/wx, 0.0, 0.0, 0.0],
            [0.0, 2.0/wy, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [cx/wx, -cy/wy, 0.0, 1.0],
        ],
    }
}

/// Ticket 014 — per-mesh SDF bake uniform.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SdfBakeParams {
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    /// x = triangle_count, y = sdf_resolution (32), zw unused
    counts: [u32; 4],
}

/// Ticket 014 V6 — uniform for `WSRC_BAKE_WGSL`. Analytic sun × shadow
/// + analytic sky computed per probe-octel. Shadow VPs + splits +
/// flags mirror CARD_LIGHT_WGSL so the shader can re-use the same
/// cascade-sampling helper.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WsrcBakeParams {
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    sky_color: [f32; 4],
    /// xyz = WSRC cube origin, w = full extent.
    grid: [f32; 4],
    /// Cascade 0..2 VPs. Only cascade 2 is actually sampled by V6
    /// (widest cascade covers the whole 120 m cube), but we carry
    /// all three to keep the uniform layout identical to the card-
    /// lighting params.
    shadow_vps: [[[f32; 4]; 4]; 3],
    /// xyz = view-space split distances (unused by V6 because it
    /// always samples cascade 2), w = 0.
    shadow_splits: [f32; 4],
    /// x = shadow bias, y = shadows_enabled (0/1), zw unused.
    flags: [f32; 4],
}

/// Uniform struct for the card-lighting compute pass. Matches
/// CARD_LIGHT_WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CardLightParams {
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    sky_color: [f32; 4],
    /// [atlas_size, slot_size, slots_per_row, active_slot_count]
    atlas_info: [u32; 4],
    /// Shadow cascade VPs (3× mat4) — identity when shadows disabled.
    shadow_vps: [[[f32; 4]; 4]; 3],
    /// xyz = view-space split distances for cascades 0/1/2, w = 0.
    shadow_splits: [f32; 4],
    /// Camera view matrix — needed to convert card-texel world pos
    /// into view-space Z for cascade selection.
    view_matrix: [[f32; 4]; 4],
    /// x = shadow bias, y = shadows_enabled (0/1), zw unused.
    flags: [f32; 4],
}

/// Ticket 013 V3 — per-slot metadata consumed by `card_light_pass`.
/// Baked at capture time; carries enough state for the lighting
/// shader to reconstruct each texel's world-space position and query
/// the shadow cascade at that point. 128 bytes per slot × 4096 slots
/// = 512 KB — fits comfortably.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CardSlotMetaCpu {
    /// xyz = world-space card-face normal, w = signed axis (0..6 as f32).
    normal_ws: [f32; 4],
    /// Object-space AABB min (xyz) + padding (w).
    aabb_min: [f32; 4],
    /// Object-space AABB max (xyz) + padding (w).
    aabb_max: [f32; 4],
    /// Mesh's world transform. Multiplied into the object-space
    /// card-plane position to land in world space for shadow lookup.
    transform: [[f32; 4]; 4],
}

/// Ticket 007b — per-TLAS-instance GI shading input. Indexed by the
/// hit's `instance_custom_data` in the HW trace shader. Layout must
/// match the `InstanceGIData` struct in SSGI_PROBE_TRACE_HW_WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceGiDataCpu {
    /// Flat albedo (per-mesh, pre-baked from material base-color).
    /// Used as a fallback when `card_slot.w < 0` (no card captured).
    albedo: [f32; 3],
    /// Scalar emissive luminance — multiplied by `albedo` at the hit.
    emissive_luma: f32,
    /// Flat world-space mesh normal. Rough — averaged over vertex
    /// normals at BLAS build time. Used when the card atlas is
    /// unavailable; ticket 013's textured path still drives lighting
    /// from this flat normal but multiplies the sampled albedo in.
    normal_ws: [f32; 3],
    _pad0: f32,
    /// Ticket 013 — card slot for textured hit shading.
    /// `card_slot.xy` = atlas slot coord (0..CARD_SLOTS_PER_ROW).
    /// `card_slot.z` = dominant axis (0=X, 1=Y, 2=Z).
    /// `card_slot.w` = flag (1.0 = card captured, 0.0 = no card → fall
    /// back to `albedo` flat value).
    card_slot: [f32; 4],
    /// Object-space AABB min (xyz) + unused pad (w).
    card_aabb_min: [f32; 4],
    /// Object-space AABB max (xyz) + unused pad (w).
    card_aabb_max: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrTemporalParams {
    /// x = blend_alpha (0.1), yzw unused
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
    /// Cached bind groups for the SSAO compute pass. One per ping-pong
    /// index because the history input/output textures swap every
    /// frame — swapping views inside a cached BG is not allowed.
    ssao_bg_cache: [Option<wgpu::BindGroup>; 2],
    ssao_blur_bg_cache: Option<wgpu::BindGroup>,
    /// Half-res AO history ping-pong for temporal accumulation.
    /// Each frame reads `[1 - ssao_history_idx]` as history input and
    /// writes `[ssao_history_idx]` as the blended output. `ssao_rt`
    /// still receives the contrasted/strength-modulated per-frame
    /// result that the bilateral blur consumes; history stores the
    /// pre-contrast linear AO so repeated applies of pow(ao, 2) don't
    /// collapse the signal toward zero.
    pub ssao_history_textures: [wgpu::Texture; 2],
    pub ssao_history_views: [wgpu::TextureView; 2],
    pub ssao_history_idx: usize,
    /// Frames since the SSAO history was last invalidated. On 0..3
    /// we force alpha=1 so the first frames seed the history from
    /// the current frame instead of blending against an undefined
    /// clear. Resets to 0 on resize and when SSAO toggles back on.
    pub ssao_history_frame: u32,
    ssr_bg_cache: Option<wgpu::BindGroup>,
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

    /// SSR temporal denoiser: ping-pong history textures (same format/size
    /// as ssr_rt). One GGX-importance-sampled ray per pixel per frame
    /// converges over 4–8 frames of accumulation via velocity reprojection
    /// + neighborhood clamp. Compose reads ssr_history[cur] instead of
    /// ssr_rt when ssr_enabled.
    pub ssr_history_textures: [wgpu::Texture; 2],
    pub ssr_history_views: [wgpu::TextureView; 2],
    pub ssr_history_idx: usize,
    pub ssr_temporal_pipeline: wgpu::RenderPipeline,
    pub ssr_temporal_layout: wgpu::BindGroupLayout,
    pub ssr_temporal_uniform_buffer: wgpu::Buffer,

    /// SSGI (screen-space global illumination) pass output — half-res
    /// HDR holding the indirect diffuse bounce light for each fragment.
    /// Ticket 007a: written by the probe-resolve pass, composited by
    /// the TAA pass. No other code path touches `ssgi_rt_view`.
    pub ssgi_rt_texture: wgpu::Texture,
    pub ssgi_rt_view: wgpu::TextureView,
    /// SSGI intensity multiplier (0 = off, 0.5 = default, 1+ = strong).
    pub ssgi_intensity: f32,
    /// SSGI max march distance in view-space meters.
    pub ssgi_radius: f32,
    /// SSGI master switch.
    pub ssgi_enabled: bool,

    // --- Ticket 007a: Lumen-style screen-probe SSGI ---

    /// Current probe grid dimensions. Recomputed on resize.
    pub probe_grid_w: u32,
    pub probe_grid_h: u32,
    /// Per-probe header buffer (`ProbeHeader`, 32 B each). `STORAGE |
    /// COPY_DST`. Written by probe-place, read by trace + resolve.
    pub probe_header_buffer: wgpu::Buffer,
    /// Per-frame trace output — the compute trace pass writes
    /// `textureStore` into this. The temporal pass reads it as the
    /// "current" input.
    pub probe_trace_tex: wgpu::Texture,
    pub probe_trace_view: wgpu::TextureView,
    /// Ping-pong 3D history textures (Rgba16Float, gw × gh × 64).
    /// Temporal reads `[prev_idx]` + trace, writes to `[write_idx]`.
    /// Resolve samples `[write_idx]`.
    pub probe_history_textures: [wgpu::Texture; 2],
    pub probe_history_views: [wgpu::TextureView; 2],
    pub probe_history_idx: usize,

    /// Ticket 007b — true when the adapter granted
    /// `Features::EXPERIMENTAL_RAY_QUERY` at device creation. Flips the
    /// probe trace pass from the SW Hi-Z path (007a) to the HW ray-query
    /// path. `BLOOM_FORCE_SW_GI=1` at platform init leaves this false.
    pub hw_rt_enabled: bool,
    /// Top-level acceleration structure. Lazy — allocated on the first
    /// frame that finds at least one per-node BLAS ready. Capacity
    /// rounded up to 1024 instances; resized only when exceeded.
    pub tlas: Option<wgpu::Tlas>,
    pub tlas_max_instances: u32,
    /// `SceneGraph::tlas_version` the last time we rebuilt the TLAS +
    /// instance-data buffer. Mismatch → rebuild. Mirrors `shadow_version`'s
    /// cache pattern (ticket 004).
    pub tlas_built_version: u64,
    /// Per-instance GI data (flat albedo + flat normal_ws + emissive),
    /// indexed by the TLAS instance's `custom_data`. Rebuilt alongside
    /// the TLAS instance list.
    pub tlas_instance_data_buffer: Option<wgpu::Buffer>,

    pub probe_place_pipeline: wgpu::ComputePipeline,
    pub probe_place_layout: wgpu::BindGroupLayout,
    pub probe_place_uniform: wgpu::Buffer,
    probe_place_bg_cache: Option<wgpu::BindGroup>,

    pub probe_trace_pipeline: wgpu::ComputePipeline,
    pub probe_trace_layout: wgpu::BindGroupLayout,
    pub probe_trace_uniform: wgpu::Buffer,
    probe_trace_bg_cache: [Option<wgpu::BindGroup>; 2],

    /// Ticket 007b — HW-path trace pipeline and its distinct layout
    /// (includes the TLAS + instance_data buffer bindings that the SW
    /// layout doesn't carry). `Some` only when `hw_rt_enabled`; `None`
    /// otherwise so the shader module isn't even compiled.
    pub probe_trace_hw_pipeline: Option<wgpu::ComputePipeline>,
    pub probe_trace_hw_layout: Option<wgpu::BindGroupLayout>,
    /// V3 — [Option; 2] so the HW trace can bind the prev-frame
    /// probe history on a per-frame ping-pong index, matching the
    /// SW cache shape.
    probe_trace_hw_bg_cache: [Option<wgpu::BindGroup>; 2],

    /// Ticket 014 V3 — SW SDF sphere-trace pipeline + layout.
    /// Active whenever the scene clipmap has been baked; chosen at
    /// dispatch time over the Hi-Z SW fallback when available.
    pub probe_trace_sdf_pipeline: wgpu::ComputePipeline,
    pub probe_trace_sdf_layout: wgpu::BindGroupLayout,
    /// V3 — same [Option; 2] shape as the HW + SW caches.
    probe_trace_sdf_bg_cache: [Option<wgpu::BindGroup>; 2],

    // --- Ticket 013: Mesh Cards (Surface Cache, V2 — 6-axis + per-frame lighting) ---
    /// Albedo atlas, baked once per mesh at model load.
    pub mesh_card_atlas: wgpu::Texture,
    pub mesh_card_atlas_view: wgpu::TextureView,
    pub mesh_card_atlas_sampler: wgpu::Sampler,
    /// Ticket 013 V3 — emissive atlas, baked alongside albedo.
    pub mesh_card_emissive_tex: wgpu::Texture,
    pub mesh_card_emissive_view: wgpu::TextureView,
    /// Radiance atlas — lit each frame by the card-lighting compute
    /// pass, then sampled at hit by the HW probe trace. Same geometry
    /// as the albedo atlas.
    pub mesh_card_radiance_tex: wgpu::Texture,
    pub mesh_card_radiance_view: wgpu::TextureView,
    /// Per-slot metadata buffer: one vec4 per slot holding the card
    /// face's world-space normal (xyz) + unused w. Populated as
    /// captures land; consumed by the card-lighting pass for NdotL.
    pub card_slot_meta_buffer: wgpu::Buffer,

    pub card_capture_pipeline: wgpu::RenderPipeline,
    pub card_capture_uniform_layout: wgpu::BindGroupLayout,
    pub card_capture_texture_layout: wgpu::BindGroupLayout,
    pub card_capture_uniform: wgpu::Buffer,
    pub card_capture_fallback_tex: wgpu::Texture,
    pub card_capture_fallback_view: wgpu::TextureView,

    pub card_light_pipeline: wgpu::ComputePipeline,
    pub card_light_layout: wgpu::BindGroupLayout,
    pub card_light_uniform: wgpu::Buffer,
    card_light_bg_cache: Option<wgpu::BindGroup>,

    // --- Ticket 014: per-mesh UDF bake ---
    pub sdf_bake_pipeline: wgpu::ComputePipeline,
    pub sdf_bake_layout: wgpu::BindGroupLayout,
    pub sdf_bake_uniform: wgpu::Buffer,

    // --- Ticket 014 V2: scene-wide SDF clipmap ---
    pub scene_sdf_clipmap_tex: wgpu::Texture,
    pub scene_sdf_clipmap_view: wgpu::TextureView,
    /// Set once when the scene clipmap has been baked. V5 clears this
    /// flag when the camera wanders past the rebake threshold, so the
    /// bake path fires again to re-centre the clipmap.
    pub scene_sdf_clipmap_built: bool,
    /// Ticket 014 V5 — current clipmap origin in world space (voxel-
    /// snapped camera position at last bake). Read every frame by the
    /// trace uniform; updated at bake time.
    pub scene_sdf_clipmap_origin: [f32; 3],

    // --- Ticket 014 V6/V10: World-Space Radiance Cache ---
    pub wsrc_atlas_tex: wgpu::Texture,
    pub wsrc_atlas_view: wgpu::TextureView,
    /// V10 — linear-filtering sampler for the padded WSRC atlas.
    /// Clamp-to-edge in all axes: edge-extend behaves identically
    /// to the baked borders so the miss-path lookup stays well-
    /// defined at probe seams.
    pub wsrc_atlas_sampler: wgpu::Sampler,
    pub wsrc_bake_pipeline: wgpu::ComputePipeline,
    pub wsrc_bake_layout: wgpu::BindGroupLayout,
    pub wsrc_bake_uniform: wgpu::Buffer,
    wsrc_bake_bg_cache: Option<wgpu::BindGroup>,
    /// V14 — HW-ray-traced WSRC bake. Built only when
    /// `hw_rt_enabled`. Same `WsrcBakeParams` uniform as the SW
    /// bake; extra bindings carry the TLAS + per-instance GI data
    /// + card atlas so probe-octel rays can sample pre-lit Mesh
    /// Cards radiance at hit points.
    pub wsrc_bake_hw_pipeline: Option<wgpu::ComputePipeline>,
    pub wsrc_bake_hw_layout: Option<wgpu::BindGroupLayout>,
    wsrc_bake_hw_bg_cache: Option<wgpu::BindGroup>,
    /// V13 — per-cascade state. Each cascade (near/mid/far) tracks
    /// its own `built` flag, voxel-snapped origin, and lighting
    /// snapshot. Invalidation uses the cascade's own cell size
    /// as the camera-travel threshold, so the far cascade rebakes
    /// far less often than the near one despite the same relative
    /// threshold constant.
    pub wsrc_built: [bool; 3],
    pub wsrc_origin: [[f32; 3]; 3],
    /// Ticket 014 V7/V12/V13 — per-cascade last-baked lighting
    /// state. `sun_dir` is xyz + intensity in w (intensity isn't
    /// threshold-checked on its own — it's folded into `sun_color
    /// × intensity`). V12 perceptual hysteresis applies per cascade
    /// — any one cascade going out-of-threshold rebakes only that
    /// cascade.
    pub wsrc_last_sun_dir: [[f32; 4]; 3],
    pub wsrc_last_sun_color: [[f32; 3]; 3],
    pub wsrc_last_sky_color: [[f32; 3]; 3],

    pub probe_temporal_pipeline: wgpu::ComputePipeline,
    pub probe_temporal_layout: wgpu::BindGroupLayout,
    pub probe_temporal_uniform: wgpu::Buffer,
    probe_temporal_bg_cache: [Option<wgpu::BindGroup>; 2],

    pub probe_resolve_pipeline: wgpu::RenderPipeline,
    pub probe_resolve_layout: wgpu::BindGroupLayout,
    pub probe_resolve_uniform: wgpu::Buffer,
    probe_resolve_bg_cache: [Option<wgpu::BindGroup>; 2],

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
    /// Projection matrix before the TAA sub-pixel jitter is applied.
    /// Used for shadow cascade fitting — the jitter would otherwise
    /// nudge the cascade VPs every frame and defeat the shadow cache.
    current_proj_matrix_unjittered: [[f32; 4]; 4],
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
    // Multi-skin per-frame staging.
    //
    // Each `updateModelAnimation` call appends one entry to
    // `pending_skin_groups` — a pre-scaled/positioned pose for one
    // skinned model. The matching `drawModel` call consumes the
    // front entry (FIFO), copies its matrices into
    // `frame_joint_data` at the current write cursor, and offsets
    // that draw's per-vertex joint indices by the cursor so the
    // shader samples its own pose from the shared 1024-slot buffer.
    // At `end_frame` the accumulator is flushed in one write.
    pub pending_skin_groups: Vec<Vec<[[f32; 4]; 4]>>,
    pub frame_joint_data: Vec<[[f32; 4]; 4]>,
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

    /// Phase 1c — the new shader-ABI material draw path. Opt-in via
    /// `compile_material` + `submit_material_draw`; existing draws are
    /// untouched.
    pub material_system: material_system::MaterialSystem,

    /// Phase 3 — short-lived texture pool. Feeds scene-colour
    /// snapshots (Phase 4b), depth-as-sampled linearisations (Phase 4b),
    /// and future graph-managed intermediates.
    pub transient_pool: transient::TransientPool,

    /// Phase 7 — persistent world-space impulse field. Games submit
    /// splats via `bloom_splat_impulse`; the renderer dispatches a
    /// compute pass per frame to decay + accumulate, then the front
    /// view is bound at `group(4) binding(4)` in scene_inputs.
    pub impulse_field: impulse_field::ImpulseField,

    /// Phase 6 — hot-reload registry for file-backed materials. Each
    /// frame we drain pending file-change events and recompile any
    /// affected pipelines. Handles registered via
    /// `compile_material_from_file` participate; inline-string
    /// materials (compile_material) don't.
    pub material_hot_reload: hot_reload::MaterialHotReload,
}

/// Ticket 014 — re-exposed for `SceneGraph::prepare()` to allocate
/// the per-mesh SDF texture alongside BLAS creation without needing
/// `pub(super)` access to the renderer's format module.
pub fn create_mesh_sdf_texture_public(
    device: &wgpu::Device,
    label: &'static str,
) -> (wgpu::Texture, wgpu::TextureView) {
    formats::create_mesh_sdf_texture(device, label)
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
        // Ticket 007b: HW ray-tracing availability is a device-feature
        // query, set by whichever platform crate constructed this
        // device. Renderer internals branch on this flag when picking
        // the probe-trace pipeline.
        let hw_rt_enabled = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

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
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
        let (ssao_history_textures, ssao_history_views) = create_ssao_history_textures(
            &device, surface_config.width, surface_config.height,
        );
        let (taa_textures, taa_views) = create_taa_textures(
            &device, surface_config.width, surface_config.height,
        );
        let (ssr_rt_texture, ssr_rt_view) = create_ssr_rt(
            &device, surface_config.width, surface_config.height,
        );
        let (ssr_history_textures, ssr_history_views) = create_ssr_history_textures(
            &device, surface_config.width, surface_config.height,
        );
        let (ssgi_rt_texture, ssgi_rt_view) = create_ssgi_rt(
            &device, surface_config.width, surface_config.height,
        );
        let (probe_grid_w, probe_grid_h) =
            probe_grid_dims(surface_config.width, surface_config.height);
        let (probe_trace_tex, probe_trace_view) = create_probe_trace_tex(
            &device, surface_config.width, surface_config.height,
        );
        let (probe_history_textures, probe_history_views) = create_probe_history_textures(
            &device, surface_config.width, surface_config.height,
        );
        let probe_header_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_header_buffer"),
            size: (probe_grid_w * probe_grid_h) as u64
                * std::mem::size_of::<ProbeHeaderCpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
            bind_group_layouts: &[Some(&uniform_2d_layout), Some(&texture_bind_group_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
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
        // 1024 joints × 64 bytes per mat4 = 65536 bytes.
        // Sized so multiple skinned models can coexist in one frame —
        // each updateModelAnimation stages its pose into a slice of
        // this buffer, and each skinned drawModel reads its assigned
        // slice via a per-vertex joint-index offset baked at submit
        // time. 65536 is the default wgpu max_uniform_buffer_binding_size.
        let joint_data = vec![0u8; 65536];
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
            let mut identity_data = vec![0u8; 65536];
            for i in 0..1024 {
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
            bind_group_layouts: &[Some(&uniform_3d_layout), Some(&lighting_layout), Some(&texture_bind_group_layout), Some(&joint_layout)],
            immediate_size: 0,
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
            bind_group_layouts: &[Some(&sky_bind_group_layout)],
            immediate_size: 0,
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
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
            bind_group_layouts: &[Some(&prefilter_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
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
            multiview_mask: None,
            cache: None,
        });

        let scene_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_pipeline_layout"),
            bind_group_layouts: &[Some(&uniform_3d_layout), Some(&lighting_layout), Some(&scene_material_layout), Some(&joint_layout)],
            immediate_size: 0,
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
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
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
            cache: None,
        });
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
            bind_group_layouts: &[Some(&bloom_layout)],
            immediate_size: 0,
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
                multiview_mask: None,
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
            bind_group_layouts: &[Some(&hiz_linearize_layout)],
            immediate_size: 0,
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
            bind_group_layouts: &[Some(&hiz_downsample_layout)],
            immediate_size: 0,
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
                // velocity_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 8, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // history_in (ping-pong read)
                wgpu::BindGroupLayoutEntry {
                    binding: 9, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // filt_samp (history bilinear reprojection)
                wgpu::BindGroupLayoutEntry {
                    binding: 10, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // history_out (ping-pong write)
                wgpu::BindGroupLayoutEntry {
                    binding: 11, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: SSAO_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
            ],
        });
        let ssao_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_pl_layout"),
            bind_group_layouts: &[Some(&ssao_layout)],
            immediate_size: 0,
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
            bind_group_layouts: &[Some(&ssao_blur_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
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
            bind_group_layouts: &[Some(&taa_layout)],
            immediate_size: 0,
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
            multiview_mask: None, cache: None,
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
            bind_group_layouts: &[Some(&ssr_layout)],
            immediate_size: 0,
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
            multiview_mask: None, cache: None,
        });
        let ssr_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr_uniform_buffer"),
            size: std::mem::size_of::<SsrParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- SSR temporal denoiser pipeline ---
        let ssr_temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr_temporal_shader"),
            source: wgpu::ShaderSource::Wgsl(SSR_TEMPORAL_SHADER_WGSL.into()),
        });
        let ssr_temporal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_temporal_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // binding 1: current noisy SSR
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
                // binding 3: history SSR (previous frame accumulated)
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
        let ssr_temporal_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_temporal_pl_layout"),
            bind_group_layouts: &[Some(&ssr_temporal_layout)],
            immediate_size: 0,
        });
        let ssr_temporal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssr_temporal_pipeline"),
            layout: Some(&ssr_temporal_pl_layout),
            vertex: wgpu::VertexState {
                module: &ssr_temporal_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssr_temporal_shader, entry_point: Some("fs_main"),
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
            multiview_mask: None, cache: None,
        });
        let ssr_temporal_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr_temporal_uniform_buffer"),
            size: std::mem::size_of::<SsrTemporalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Ticket 007a: Lumen screen-probe SSGI pipelines ---
        // One shader module per pass; each is prepended with
        // PROBE_HELPERS_WGSL so oct_encode/decode, view-space
        // reconstruction, and `ProbeHeader` are shared.
        let probe_shader = |label: &'static str, body: &str| {
            let source = format!("{}{}", PROBE_HELPERS_WGSL, body);
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };

        let probe_place_shader = probe_shader("probe_place_shader", SSGI_PROBE_PLACE_WGSL);
        let probe_trace_shader = probe_shader("probe_trace_shader", SSGI_PROBE_TRACE_SW_WGSL);
        let probe_temporal_shader = probe_shader("probe_temporal_shader", SSGI_PROBE_TEMPORAL_WGSL);
        let probe_resolve_shader = probe_shader("probe_resolve_shader", SSGI_PROBE_RESOLVE_WGSL);

        // --- Probe placement ---
        let probe_place_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_place_layout"),
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
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });
        let probe_place_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_place_pl_layout"),
            bind_group_layouts: &[Some(&probe_place_layout)],
            immediate_size: 0,
        });
        let probe_place_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_place_pipeline"),
            layout: Some(&probe_place_pl_layout),
            module: &probe_place_shader, entry_point: Some("cs_main"),
            compilation_options: Default::default(), cache: None,
        });
        let probe_place_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_place_uniform"),
            size: std::mem::size_of::<ProbePlaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Probe trace (SW Hi-Z) ---
        let probe_trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_trace_layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // Hi-Z pyramid (5 mips, separate views)
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    }, count: None,
                },
                // V3 — prev-frame probe history (textureLoad, no
                // sampler). Non-filterable so the layout matches
                // the storage-binding constraint on adapters that
                // don't advertise FLOAT32_FILTERABLE.
                wgpu::BindGroupLayoutEntry {
                    binding: 11, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    }, count: None,
                },
            ],
        });
        let probe_trace_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_trace_pl_layout"),
            bind_group_layouts: &[Some(&probe_trace_layout)],
            immediate_size: 0,
        });
        let probe_trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_trace_pipeline"),
            layout: Some(&probe_trace_pl_layout),
            module: &probe_trace_shader, entry_point: Some("cs_main"),
            compilation_options: Default::default(), cache: None,
        });
        let probe_trace_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_trace_uniform"),
            size: std::mem::size_of::<ProbeTraceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Probe trace HW (ticket 007b, ray-query variant) ---
        // Built only when the device was created with EXPERIMENTAL_RAY_QUERY.
        // Shares the same ProbeTraceParams uniform as the SW path (wider
        // fields — sun/sky — are unused by SW but present in the layout).
        let (probe_trace_hw_pipeline, probe_trace_hw_layout) = if hw_rt_enabled {
            // `enable wgpu_ray_query;` must appear before any declaration,
            // so we emit the ray-query enable directive ahead of the
            // shared helpers rather than using the generic `probe_shader`
            // builder (which prepends helpers first).
            let hw_source = format!(
                "enable wgpu_ray_query;\n{}{}",
                PROBE_HELPERS_WGSL, SSGI_PROBE_TRACE_HW_WGSL,
            );
            let hw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("probe_trace_hw_shader"),
                source: wgpu::ShaderSource::Wgsl(hw_source.into()),
            });
            let hw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("probe_trace_hw_layout"),
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false, min_binding_size: None,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false, min_binding_size: None,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: HDR_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        }, count: None,
                    },
                    // Ticket 013 — mesh-card albedo atlas + sampler.
                    wgpu::BindGroupLayoutEntry {
                        binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Ticket 014 V7/V10 — WSRC atlas for the HW miss
                    // path. V10 filtering texture + linear sampler.
                    wgpu::BindGroupLayoutEntry {
                        binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // V3 — prev-frame probe history (textureLoad).
                    wgpu::BindGroupLayoutEntry {
                        binding: 9, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        }, count: None,
                    },
                ],
            });
            let hw_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("probe_trace_hw_pl_layout"),
                bind_group_layouts: &[Some(&hw_layout)],
                immediate_size: 0,
            });
            let hw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("probe_trace_hw_pipeline"),
                layout: Some(&hw_pl_layout),
                module: &hw_shader, entry_point: Some("cs_main"),
                compilation_options: Default::default(), cache: None,
            });
            (Some(hw_pipeline), Some(hw_layout))
        } else {
            (None, None)
        };

        // --- Ticket 014 V3: SW SDF sphere-trace pipeline ---
        // Always built. At dispatch time we pick SDF over Hi-Z when
        // `scene_sdf_clipmap_built` is true; HW (when available)
        // still wins over both.
        let sdf_trace_shader = probe_shader(
            "probe_trace_sdf_shader",
            SSGI_PROBE_TRACE_SDF_WGSL,
        );
        let probe_trace_sdf_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_trace_sdf_layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    }, count: None,
                },
                // V4 — instance_data + card radiance atlas + sampler
                // for the broad-phase textured hit lookup.
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Ticket 014 V6/V10 — WSRC atlas for the SDF miss
                // path. V10 upgrades to a filtering Rgba16Float
                // texture + linear sampler so the sampler does octel
                // bilinear + Z trilinear natively inside each
                // probe's 10×10 padded slab.
                wgpu::BindGroupLayoutEntry {
                    binding: 8, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // V3 — prev-frame probe history (textureLoad).
                wgpu::BindGroupLayoutEntry {
                    binding: 10, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    }, count: None,
                },
            ],
        });
        let probe_trace_sdf_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_trace_sdf_pl_layout"),
            bind_group_layouts: &[Some(&probe_trace_sdf_layout)],
            immediate_size: 0,
        });
        let probe_trace_sdf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_trace_sdf_pipeline"),
            layout: Some(&probe_trace_sdf_pl_layout),
            module: &sdf_trace_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Probe temporal ---
        let probe_temporal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_temporal_layout"),
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
                        view_dimension: wgpu::TextureViewDimension::D3, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    }, count: None,
                },
            ],
        });
        let probe_temporal_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_temporal_pl_layout"),
            bind_group_layouts: &[Some(&probe_temporal_layout)],
            immediate_size: 0,
        });
        let probe_temporal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_temporal_pipeline"),
            layout: Some(&probe_temporal_pl_layout),
            module: &probe_temporal_shader, entry_point: Some("cs_main"),
            compilation_options: Default::default(), cache: None,
        });
        let probe_temporal_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_temporal_uniform"),
            size: std::mem::size_of::<ProbeTemporalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Probe resolve (fragment → half-res ssgi_rt) ---
        let probe_resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_resolve_layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let probe_resolve_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_resolve_pl_layout"),
            bind_group_layouts: &[Some(&probe_resolve_layout)],
            immediate_size: 0,
        });
        let probe_resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("probe_resolve_pipeline"),
            layout: Some(&probe_resolve_pl_layout),
            vertex: wgpu::VertexState {
                module: &probe_resolve_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &probe_resolve_shader, entry_point: Some("fs_main"),
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
            multiview_mask: None, cache: None,
        });
        let probe_resolve_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_resolve_uniform"),
            size: std::mem::size_of::<ProbeResolveParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Ticket 013 V3: mesh-card atlases + capture + lighting ---
        let (mesh_card_atlas, mesh_card_atlas_view) = create_mesh_card_atlas(&device);
        let (mesh_card_emissive_tex, mesh_card_emissive_view) =
            create_mesh_card_emissive_atlas(&device);
        let (mesh_card_radiance_tex, mesh_card_radiance_view) =
            create_mesh_card_radiance_atlas(&device);
        let card_slot_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("card_slot_meta_buffer"),
            size: (CARD_MAX_SLOTS as u64) * std::mem::size_of::<CardSlotMetaCpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_card_atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh_card_atlas_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let card_capture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("card_capture_shader"),
            source: wgpu::ShaderSource::Wgsl(CARD_CAPTURE_WGSL.into()),
        });
        let card_capture_uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("card_capture_uniform_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let card_capture_texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("card_capture_texture_layout"),
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
                // Ticket 013 V3 — emissive texture binding.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let card_capture_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("card_capture_pl_layout"),
            bind_group_layouts: &[
                Some(&card_capture_uniform_layout),
                Some(&card_capture_texture_layout),
            ],
            immediate_size: 0,
        });
        let card_capture_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("card_capture_pipeline"),
            layout: Some(&card_capture_pl_layout),
            vertex: wgpu::VertexState {
                module: &card_capture_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &card_capture_shader,
                entry_point: Some("fs_main"),
                // Location 0 → albedo atlas, Location 1 → emissive atlas.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let card_capture_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("card_capture_uniform"),
            size: 128, // ortho_vp (64) + base_color (16) + pad to 128 for alignment headroom
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // 1×1 white fallback texture used when a mesh has no albedo
        // — the shader's `has_texture` flag branches to skip the sample
        // but wgpu still requires a bound texture in the pipeline layout.
        let card_capture_fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("card_capture_fallback"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &card_capture_fallback_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let card_capture_fallback_view = card_capture_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Ticket 013 V2: card-lighting compute ---
        let card_light_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("card_light_shader"),
            source: wgpu::ShaderSource::Wgsl(CARD_LIGHT_WGSL.into()),
        });
        let card_light_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("card_light_layout"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                // V3 — emissive atlas.
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                // V3 — three shadow cascade depth textures + comparison sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let card_light_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("card_light_pl_layout"),
            bind_group_layouts: &[Some(&card_light_layout)],
            immediate_size: 0,
        });
        let card_light_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("card_light_pipeline"),
            layout: Some(&card_light_pl_layout),
            module: &card_light_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let card_light_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("card_light_uniform"),
            size: std::mem::size_of::<CardLightParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Ticket 014: per-mesh UDF bake compute pipeline ---
        let sdf_bake_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_bake_shader"),
            source: wgpu::ShaderSource::Wgsl(SDF_BAKE_WGSL.into()),
        });
        let sdf_bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_bake_layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    }, count: None,
                },
            ],
        });
        let sdf_bake_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_bake_pl_layout"),
            bind_group_layouts: &[Some(&sdf_bake_layout)],
            immediate_size: 0,
        });
        let sdf_bake_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf_bake_pipeline"),
            layout: Some(&sdf_bake_pl_layout),
            module: &sdf_bake_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sdf_bake_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_bake_uniform"),
            size: std::mem::size_of::<SdfBakeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Ticket 014 V2: scene clipmap ---
        let (scene_sdf_clipmap_tex, scene_sdf_clipmap_view) =
            create_scene_sdf_clipmap(&device);

        // --- Ticket 014 V6: WSRC ---
        let (wsrc_atlas_tex, wsrc_atlas_view) = create_wsrc_atlas(&device);
        // V10 — linear sampler for the padded WSRC atlas. Rgba16Float
        // is filterable without an extra feature on every backend we
        // care about, so a plain Filtering sampler is enough.
        let wsrc_atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("wsrc_atlas_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let wsrc_bake_shader = probe_shader("wsrc_bake_shader", WSRC_BAKE_WGSL);
        let wsrc_bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wsrc_bake_layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    }, count: None,
                },
            ],
        });
        let wsrc_bake_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wsrc_bake_pl_layout"),
            bind_group_layouts: &[Some(&wsrc_bake_layout)],
            immediate_size: 0,
        });
        let wsrc_bake_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wsrc_bake_pipeline"),
            layout: Some(&wsrc_bake_pl_layout),
            module: &wsrc_bake_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let wsrc_bake_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wsrc_bake_uniform"),
            size: std::mem::size_of::<WsrcBakeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // V14 — HW-ray-traced WSRC bake pipeline. Built only on
        // RT-capable adapters. Same uniform as the SW bake (writes
        // the same `WsrcBakeParams` per cascade) plus TLAS +
        // instance_data + card_atlas bindings that let probe-octel
        // rays sample Mesh Cards at hit.
        let (wsrc_bake_hw_pipeline, wsrc_bake_hw_layout) = if hw_rt_enabled {
            let hw_source = format!(
                "enable wgpu_ray_query;\n{}{}",
                PROBE_HELPERS_WGSL, WSRC_BAKE_HW_WGSL,
            );
            let hw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wsrc_bake_hw_shader"),
                source: wgpu::ShaderSource::Wgsl(hw_source.into()),
            });
            let hw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wsrc_bake_hw_layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: HDR_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false, min_binding_size: None,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let hw_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wsrc_bake_hw_pl_layout"),
                bind_group_layouts: &[Some(&hw_layout)],
                immediate_size: 0,
            });
            let hw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("wsrc_bake_hw_pipeline"),
                layout: Some(&hw_pl_layout),
                module: &hw_shader, entry_point: Some("cs_main"),
                compilation_options: Default::default(), cache: None,
            });
            (Some(hw_pipeline), Some(hw_layout))
        } else {
            (None, None)
        };

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
            bind_group_layouts: &[Some(&scene_compose_layout)],
            immediate_size: 0,
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
            multiview_mask: None, cache: None,
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
            bind_group_layouts: &[Some(&dof_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
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
            bind_group_layouts: &[Some(&motion_blur_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
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
            bind_group_layouts: &[Some(&sss_layout)],
            immediate_size: 0,
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
            multiview_mask: None,
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
            bind_group_layouts: &[Some(&exposure_layout)],
            immediate_size: 0,
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
            multiview_mask: None, cache: None,
        });
        let exposure_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure_uniform_buffer"),
            size: std::mem::size_of::<ExposureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Phase 1c — material system: the new shader-ABI draw path.
        // Runs alongside pipeline_3d / scene_pipeline without disturbing
        // them; games opt in via compile_material + submit_material_draw.
        let material_system = material_system::MaterialSystem::new(&device, &queue, &joint_buffer);
        let transient_pool = transient::TransientPool::new();
        let impulse_field = impulse_field::ImpulseField::new(&device);
        let material_hot_reload = hot_reload::MaterialHotReload::new();

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
            ssao_bg_cache: [None, None],
            ssao_blur_bg_cache: None,
            ssao_history_textures,
            ssao_history_views,
            ssao_history_idx: 0,
            ssao_history_frame: 0,
            ssr_bg_cache: None,
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
            ssr_history_textures,
            ssr_history_views,
            ssr_history_idx: 0,
            ssr_temporal_pipeline,
            ssr_temporal_layout,
            ssr_temporal_uniform_buffer,
            ssgi_rt_texture,
            ssgi_rt_view,
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
            probe_grid_w,
            probe_grid_h,
            probe_header_buffer,
            probe_trace_tex,
            probe_trace_view,
            probe_history_textures,
            probe_history_views,
            probe_history_idx: 0,
            hw_rt_enabled,
            tlas: None,
            tlas_max_instances: 1024,
            tlas_built_version: 0,
            tlas_instance_data_buffer: None,
            probe_place_pipeline,
            probe_place_layout,
            probe_place_uniform,
            probe_place_bg_cache: None,
            probe_trace_pipeline,
            probe_trace_layout,
            probe_trace_uniform,
            probe_trace_bg_cache: [None, None],
            probe_trace_hw_pipeline,
            probe_trace_hw_layout,
            probe_trace_hw_bg_cache: [None, None],
            probe_trace_sdf_pipeline,
            probe_trace_sdf_layout,
            probe_trace_sdf_bg_cache: [None, None],
            mesh_card_atlas,
            mesh_card_atlas_view,
            mesh_card_atlas_sampler,
            mesh_card_emissive_tex,
            mesh_card_emissive_view,
            mesh_card_radiance_tex,
            mesh_card_radiance_view,
            card_slot_meta_buffer,
            card_capture_pipeline,
            card_capture_uniform_layout,
            card_capture_texture_layout,
            card_capture_uniform,
            card_capture_fallback_tex,
            card_capture_fallback_view,
            card_light_pipeline,
            card_light_layout,
            card_light_uniform,
            card_light_bg_cache: None,
            sdf_bake_pipeline,
            sdf_bake_layout,
            sdf_bake_uniform,
            scene_sdf_clipmap_tex,
            scene_sdf_clipmap_view,
            scene_sdf_clipmap_built: false,
            scene_sdf_clipmap_origin: [0.0, 0.0, 0.0],
            wsrc_atlas_tex,
            wsrc_atlas_view,
            wsrc_atlas_sampler,
            wsrc_bake_pipeline,
            wsrc_bake_layout,
            wsrc_bake_uniform,
            wsrc_bake_bg_cache: None,
            wsrc_bake_hw_pipeline,
            wsrc_bake_hw_layout,
            wsrc_bake_hw_bg_cache: None,
            wsrc_built: [false; 3],
            wsrc_origin: [[0.0; 3]; 3],
            wsrc_last_sun_dir: [[0.0; 4]; 3],
            wsrc_last_sun_color: [[0.0; 3]; 3],
            wsrc_last_sky_color: [[0.0; 3]; 3],
            probe_temporal_pipeline,
            probe_temporal_layout,
            probe_temporal_uniform,
            probe_temporal_bg_cache: [None, None],
            probe_resolve_pipeline,
            probe_resolve_layout,
            probe_resolve_uniform,
            probe_resolve_bg_cache: [None, None],
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
            current_proj_matrix_unjittered: IDENTITY_MAT4,
            current_inv_proj_matrix: IDENTITY_MAT4,
            current_inv_vp_matrix: IDENTITY_MAT4,
            current_camera_pos: [0.0, 0.0, 0.0],
            uniform_3d_layout,
            render_mode: RenderMode::ScreenSpace,
            debug_frame: 0,
            pending_skin_groups: Vec::with_capacity(8),
            frame_joint_data: Vec::with_capacity(256),
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
            material_system,
            transient_pool,
            impulse_field,
            material_hot_reload,
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
            // Cascade fit depends on the projection aspect ratio, so a
            // resize can shift the VPs even with the camera stationary.
            // Force a re-render next frame.
            self.shadow_map.invalidate();
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
            let (sht, shv) = create_ssao_history_textures(&self.device, width, height);
            self.ssao_history_textures = sht;
            self.ssao_history_views = shv;
            self.ssao_history_idx = 0;
            self.ssao_history_frame = 0;
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
            let (ssr_ht, ssr_hv) = create_ssr_history_textures(&self.device, width, height);
            self.ssr_history_textures = ssr_ht;
            self.ssr_history_views = ssr_hv;
            self.ssr_history_idx = 0;
            let (ssgi_t, ssgi_v) = create_ssgi_rt(&self.device, width, height);
            self.ssgi_rt_texture = ssgi_t;
            self.ssgi_rt_view = ssgi_v;
            // Ticket 007a: rebuild the probe grid + 3D radiance textures
            // whenever the surface size changes. Probe count scales with
            // half-res resolution, so the header buffer is resized too.
            let (pg_w, pg_h) = probe_grid_dims(width, height);
            self.probe_grid_w = pg_w;
            self.probe_grid_h = pg_h;
            let (ptr, pvr) = create_probe_trace_tex(&self.device, width, height);
            self.probe_trace_tex = ptr;
            self.probe_trace_view = pvr;
            let (pht, phv) = create_probe_history_textures(&self.device, width, height);
            self.probe_history_textures = pht;
            self.probe_history_views = phv;
            self.probe_history_idx = 0;
            self.probe_header_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("probe_header_buffer"),
                size: (pg_w * pg_h) as u64
                    * std::mem::size_of::<ProbeHeaderCpu>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
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
            self.ssao_bg_cache = [None, None];
            self.ssao_blur_bg_cache = None;
            self.ssr_bg_cache = None;
            self.probe_place_bg_cache = None;
            self.probe_trace_bg_cache = [None, None];
            // V3 — HW + SDF trace BGs reference the prev-frame
            // probe history view too, so invalidate on history
            // resize.
            self.probe_trace_hw_bg_cache = [None, None];
            self.probe_trace_sdf_bg_cache = [None, None];
            self.probe_temporal_bg_cache = [None, None];
            self.probe_resolve_bg_cache = [None, None];
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

    /// Ticket 013 — rasterise every pending mesh card into its
    /// assigned atlas slot. Called once per frame before the HW probe
    /// trace samples the atlas. Drains `scene.pending_card_captures`;
    /// subsequent frames with no new meshes are free.
    fn capture_pending_mesh_cards(
        &mut self,
        scene: &mut crate::scene::SceneGraph,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if scene.pending_card_captures.is_empty() {
            return;
        }
        // Rate-limit: at 6 axes per mesh one encoder fills up after a
        // few hundred render passes (observed hang on Sponza's 405 ×
        // 6 = 2430-pass batch). Drain in chunks and let subsequent
        // frames continue the work — Sponza finishes in a few frames.
        const CAPTURE_MAX_PER_FRAME: usize = 20;
        let take = scene.pending_card_captures.len().min(CAPTURE_MAX_PER_FRAME);
        let pending: Vec<f64> = scene.pending_card_captures.drain(..take).collect();

        // Pre-compute per-slot world-space normals for the whole batch.
        // `slot_meta` is a Vec<[f32;4]> where entry `first_slot + axis`
        // carries the card face's world-space normal (xyz) + unused w.
        // Uploaded in one `queue.write_buffer` after the capture loop
        // so the card-lighting compute pass sees the populated slots.
        let mut slot_meta_updates: Vec<(u32, CardSlotMetaCpu)> = Vec::new();

        for handle in pending {
            let (first_slot, bmin, bmax, transform, has_tex, bc, tex_idx, has_em, em_factor, em_idx, vb_ptr, ib_ptr, index_count) = {
                let Some(node) = scene.nodes.get(handle) else { continue; };
                let Some(first_slot) = node.card_first_slot else { continue; };
                if first_slot + CARD_AXES_PER_MESH > CARD_MAX_SLOTS {
                    continue;
                }
                let Some(vb) = node.gpu_vb.as_ref() else { continue; };
                let Some(ib) = node.gpu_ib.as_ref() else { continue; };
                let has_tex = node.material.texture_idx != 0;
                let em_idx = node.material.emissive_texture_idx;
                let has_em = em_idx != 0;
                (
                    first_slot,
                    node.bounds_min,
                    node.bounds_max,
                    node.transform,
                    has_tex,
                    node.material.color,
                    node.material.texture_idx,
                    has_em,
                    node.material.emissive,
                    em_idx,
                    vb.clone(),
                    ib.clone(),
                    node.gpu_index_count,
                )
            };
            if index_count == 0 {
                continue;
            }

            // Card textures + sampler binding stay the same across all
            // 6 axes, so build the bind groups once per mesh.
            let tex_view = if has_tex && (tex_idx as usize) < self.textures.len() {
                self.textures[tex_idx as usize].create_view(&wgpu::TextureViewDescriptor::default())
            } else {
                self.card_capture_fallback_tex
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            let em_view = if has_em && (em_idx as usize) < self.textures.len() {
                self.textures[em_idx as usize].create_view(&wgpu::TextureViewDescriptor::default())
            } else {
                self.card_capture_fallback_tex
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            let texture_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("card_capture_texture_bg"),
                layout: &self.card_capture_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tex_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&em_view),
                    },
                ],
            });

            for axis in 0..CARD_AXES_PER_MESH {
                let slot = first_slot + axis;

                let ortho_vp = build_card_ortho_v2(axis, bmin, bmax);
                let bc_w = if has_tex { 1.0 } else { 0.0 };
                let em_w = if has_em { 1.0 } else { 0.0 };
                let params = CardCaptureParams {
                    ortho_vp,
                    base_color: [bc[0], bc[1], bc[2], bc_w],
                    emissive: [em_factor[0], em_factor[1], em_factor[2], em_w],
                };
                self.queue.write_buffer(
                    &self.card_capture_uniform,
                    0,
                    bytemuck::bytes_of(&params),
                );
                let uniform_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("card_capture_uniform_bg"),
                    layout: &self.card_capture_uniform_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.card_capture_uniform.as_entire_binding(),
                    }],
                });

                let slot_x = slot % CARD_SLOTS_PER_ROW;
                let slot_y = slot / CARD_SLOTS_PER_ROW;
                let vp_x = (slot_x * CARD_SLOT_SIZE) as f32;
                let vp_y = (slot_y * CARD_SLOT_SIZE) as f32;
                let vp_sz = CARD_SLOT_SIZE as f32;

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("card_capture_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.mesh_card_atlas_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.mesh_card_emissive_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_viewport(vp_x, vp_y, vp_sz, vp_sz, 0.0, 1.0);
                pass.set_scissor_rect(vp_x as u32, vp_y as u32, CARD_SLOT_SIZE, CARD_SLOT_SIZE);
                pass.set_pipeline(&self.card_capture_pipeline);
                pass.set_bind_group(0, &uniform_bg, &[]);
                pass.set_bind_group(1, &texture_bg, &[]);
                pass.set_vertex_buffer(0, vb_ptr.slice(..));
                pass.set_index_buffer(ib_ptr.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..index_count, 0, 0..1);

                // Stash the world-space normal for the card-lighting
                // pass. Object-space face normals are axis unit vectors
                // signed by axis index:
                //   0 → +X, 1 → -X, 2 → +Y, 3 → -Y, 4 → +Z, 5 → -Z.
                // Transform by the upper-3×3 of the mesh transform to
                // land in world space.
                let mut n_os = [0.0_f32; 3];
                match axis {
                    0 => n_os[0] = 1.0,
                    1 => n_os[0] = -1.0,
                    2 => n_os[1] = 1.0,
                    3 => n_os[1] = -1.0,
                    4 => n_os[2] = 1.0,
                    _ => n_os[2] = -1.0,
                }
                let t = &transform;
                let nx = t[0][0]*n_os[0] + t[1][0]*n_os[1] + t[2][0]*n_os[2];
                let ny = t[0][1]*n_os[0] + t[1][1]*n_os[1] + t[2][1]*n_os[2];
                let nz = t[0][2]*n_os[0] + t[1][2]*n_os[1] + t[2][2]*n_os[2];
                let len = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-4);
                slot_meta_updates.push((
                    slot,
                    CardSlotMetaCpu {
                        normal_ws: [nx/len, ny/len, nz/len, axis as f32],
                        aabb_min: [bmin[0], bmin[1], bmin[2], 0.0],
                        aabb_max: [bmax[0], bmax[1], bmax[2], 0.0],
                        transform,
                    },
                ));
            }
        }

        // Upload the new slot-meta entries in one contiguous batch.
        // Sorting by slot index lets us coalesce adjacent writes, but
        // V1 scatters them — `queue.write_buffer` with one range per
        // entry is fine for the first-frame bulk-populate cost.
        for (slot, meta) in &slot_meta_updates {
            let offset = (*slot as u64) * std::mem::size_of::<CardSlotMetaCpu>() as u64;
            self.queue.write_buffer(
                &self.card_slot_meta_buffer,
                offset,
                bytemuck::cast_slice(&[*meta]),
            );
        }
    }

    /// Ticket 014 V2 — bake the scene-wide SDF clipmap once, on the
    /// frame when all per-mesh queues (BLAS, cards, per-mesh SDFs)
    /// have drained. Gathers every visible mesh's triangles into a
    /// world-space buffer via `scene.build_world_triangles()` and
    /// runs `SDF_BAKE_WGSL` against the unified data with the
    /// clipmap's fixed world-space AABB. 64³ voxel × scene triangle
    /// count = expensive one-shot (~100-200 ms on Sponza), but
    /// happens after a visible frame and never repeats for static
    /// scenes.
    /// Ticket 014 V5 — camera world-space position. Uses
    /// `current_camera_pos`, which `begin_mode_3d` writes every frame
    /// from the user-supplied camera position (cheaper than inverting
    /// the view matrix and always in sync with what the game sees).
    fn current_camera_world_pos(&self) -> [f32; 3] {
        self.current_camera_pos
    }

    /// Ticket 014 V5 — invalidate the SDF clipmap if the camera has
    /// moved past the rebake threshold from the current clipmap
    /// centre. Called at the top of every frame before the bake
    /// check, so a moving camera triggers exactly one re-bake per
    /// `threshold × extent` chunk of travel.
    fn maybe_invalidate_sdf_clipmap(&mut self) {
        if !self.scene_sdf_clipmap_built {
            return;
        }
        let cam = self.current_camera_world_pos();
        let dx = cam[0] - self.scene_sdf_clipmap_origin[0];
        let dy = cam[1] - self.scene_sdf_clipmap_origin[1];
        let dz = cam[2] - self.scene_sdf_clipmap_origin[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let threshold = SCENE_SDF_CLIPMAP_EXTENT * SCENE_SDF_CLIPMAP_REBAKE_THRESHOLD;
        if dist_sq > threshold * threshold {
            self.scene_sdf_clipmap_built = false;
        }
    }

    fn bake_scene_sdf_clipmap(
        &mut self,
        scene: &crate::scene::SceneGraph,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.scene_sdf_clipmap_built {
            return;
        }
        // Wait for all per-mesh queues to drain — builds the clipmap
        // from a fully-loaded scene rather than a partial one, and
        // keeps first-frame cost spread across the card/BLAS work
        // already scheduled.
        if !scene.pending_blas_builds.is_empty()
            || !scene.pending_card_captures.is_empty()
            || !scene.pending_sdf_bakes.is_empty()
        {
            return;
        }

        let (vertices, indices, tri_count) = scene.build_world_triangles();
        if tri_count == 0 {
            return;
        }

        // Upload the unified vertex + index buffers. STORAGE usage so
        // the bake shader can bind them directly. These are transient
        // — dropped at the end of this function once the dispatch is
        // encoded. (The encoder keeps them alive until the submit.)
        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_sdf_bake_vbuf"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_sdf_bake_ibuf"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // V5 — centre the clipmap on the current camera position,
        // voxel-snapped for sampling stability (sub-voxel shifts
        // would change which voxel each sphere-trace step reads).
        let half = SCENE_SDF_CLIPMAP_EXTENT * 0.5;
        let voxel = SCENE_SDF_CLIPMAP_EXTENT / SCENE_SDF_CLIPMAP_RES as f32;
        let cam = self.current_camera_world_pos();
        let origin = [
            (cam[0] / voxel).round() * voxel,
            (cam[1] / voxel).round() * voxel,
            (cam[2] / voxel).round() * voxel,
        ];
        self.scene_sdf_clipmap_origin = origin;
        let aabb_min = [origin[0] - half, origin[1] - half, origin[2] - half, 0.0];
        let aabb_max = [origin[0] + half, origin[1] + half, origin[2] + half, 0.0];
        let params = SdfBakeParams {
            aabb_min,
            aabb_max,
            counts: [tri_count, SCENE_SDF_CLIPMAP_RES, 0, 0],
        };
        self.queue.write_buffer(
            &self.sdf_bake_uniform,
            0,
            bytemuck::bytes_of(&params),
        );

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_sdf_clipmap_bake_bg"),
            layout: &self.sdf_bake_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.sdf_bake_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ibuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.scene_sdf_clipmap_view) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scene_sdf_clipmap_bake"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sdf_bake_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        // 64³ voxels → 16³ workgroups at 4×4×4 threads each.
        pass.dispatch_workgroups(
            SCENE_SDF_CLIPMAP_RES / 4,
            SCENE_SDF_CLIPMAP_RES / 4,
            SCENE_SDF_CLIPMAP_RES / 4,
        );
        drop(pass);

        self.scene_sdf_clipmap_built = true;
    }

    /// Ticket 014 V6/V7/V12/V13 — invalidate WSRC cascades on
    /// camera travel OR meaningful lighting change. V13 runs the
    /// V12 hysteresis checks per cascade, so a sun rotation or
    /// camera shift only rebakes the affected cascade(s). Typical
    /// pattern: camera moves 10 m → near cascade (1.875 m cell,
    /// ~0.47 m threshold) rebakes every few frames, mid cascade
    /// (7.5 m cell, ~1.9 m threshold) rebakes occasionally, far
    /// cascade (31 m cell, ~7.8 m threshold) stays cached for
    /// much longer.
    fn maybe_invalidate_wsrc(&mut self) {
        let cam = self.current_camera_world_pos();
        let ld = self.lighting_uniforms.light_dir;
        let lc = self.lighting_uniforms.light_color;
        let amb = self.lighting_uniforms.ambient;
        let cur_sun_color = [lc[0] * ld[3], lc[1] * ld[3], lc[2] * ld[3]];
        let cur_sky_color = [amb[0] * amb[3], amb[1] * amb[3], amb[2] * amb[3]];

        fn luma(c: [f32; 3]) -> f32 {
            c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
        }
        fn rel_diff(a: f32, b: f32) -> f32 {
            (a - b).abs() / a.max(b).max(1e-4)
        }

        for c in 0..WSRC_CASCADE_COUNT as usize {
            if !self.wsrc_built[c] {
                continue;
            }
            // Camera travel — per-cascade threshold scales with the
            // cascade's extent, so each cascade has its own
            // "moved enough" metric.
            let extent = WSRC_CASCADE_EXTENTS[c];
            let origin = self.wsrc_origin[c];
            let dx = cam[0] - origin[0];
            let dy = cam[1] - origin[1];
            let dz = cam[2] - origin[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let threshold = extent * WSRC_REBAKE_THRESHOLD;
            if dist_sq > threshold * threshold {
                self.wsrc_built[c] = false;
                continue;
            }

            // V12 hysteresis — angular sun + 5% relative luma.
            let last = self.wsrc_last_sun_dir[c];
            let sun_dot = ld[0] * last[0] + ld[1] * last[1] + ld[2] * last[2];
            if sun_dot < 0.99985 {
                self.wsrc_built[c] = false;
                continue;
            }
            if rel_diff(luma(cur_sun_color), luma(self.wsrc_last_sun_color[c])) > 0.05 {
                self.wsrc_built[c] = false;
                continue;
            }
            if rel_diff(luma(cur_sky_color), luma(self.wsrc_last_sky_color[c])) > 0.05 {
                self.wsrc_built[c] = false;
            }
        }
    }

    /// Ticket 014 V6 — bake the world-space radiance cache. One
    /// dispatch covers all `WSRC_GRID_RES³` probes × 64 octel texels.
    /// Cheap: per-texel work is one shadow-cascade lookup + analytic
    /// sun/sky math, roughly matching a single card-lighting pixel.
    /// Runs at most once per `WSRC_REBAKE_THRESHOLD × extent` of
    /// camera travel — same amortisation pattern as the clipmap.
    fn bake_wsrc(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // V13 — bake only cascades that are marked not-built. Each
        // cascade snaps to its own cell grid (cell = extent / 16)
        // and writes into its own 16-slice block of the shared
        // atlas.
        if self.wsrc_built.iter().all(|b| *b) {
            return;
        }

        // V14 — pick the HW ray-traced bake when the adapter has
        // ray-query AND the TLAS is ready. The SW path stays the
        // fallback for non-RT adapters and for the early frames
        // before BLAS / TLAS have been built.
        let use_hw = self.hw_rt_enabled
            && self.wsrc_bake_hw_pipeline.is_some()
            && self.tlas.is_some()
            && self.tlas_instance_data_buffer.is_some();

        // Resolve a single set of lighting params — they're the same
        // across all cascades in one frame. Per-cascade differences
        // come from the origin + extent passed through the uniform.
        let ld = self.lighting_uniforms.light_dir;
        let inv_len = 1.0 / (ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]).sqrt().max(1e-4);
        let sun_dir_ws = [-ld[0]*inv_len, -ld[1]*inv_len, -ld[2]*inv_len, ld[3]];
        let lc = self.lighting_uniforms.light_color;
        let sun_intensity = ld[3].max(0.0);
        let sun_color = [
            lc[0] * sun_intensity,
            lc[1] * sun_intensity,
            lc[2] * sun_intensity,
            0.0,
        ];
        let amb = self.lighting_uniforms.ambient;
        let sky_intensity = amb[3].max(0.0);
        let sky_color = [
            amb[0] * sky_intensity,
            amb[1] * sky_intensity,
            amb[2] * sky_intensity,
            0.0,
        ];

        let shadows_enabled = self.shadow_map.enabled;
        let shadow_vps: [[[f32; 4]; 4]; 3] = if shadows_enabled {
            self.shadow_map.light_vps
        } else {
            [IDENTITY_MAT4; 3]
        };
        let shadow_splits = if shadows_enabled {
            let s = self.shadow_map.cascade_splits;
            [s[0], s[1], s[2], 0.0]
        } else {
            [f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.0]
        };

        // Lazy-build whichever bind group the selected path needs.
        // The two caches are independent — switching between paths
        // (e.g. if TLAS becomes available mid-session) is fine.
        if use_hw {
            if self.wsrc_bake_hw_bg_cache.is_none() {
                let tlas = self.tlas.as_ref().unwrap();
                let instance_buf = self.tlas_instance_data_buffer.as_ref().unwrap();
                self.wsrc_bake_hw_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wsrc_bake_hw_bg"),
                    layout: self.wsrc_bake_hw_layout.as_ref().unwrap(),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.wsrc_bake_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                        wgpu::BindGroupEntry { binding: 6, resource: tlas.as_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: instance_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.mesh_card_radiance_view) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                    ],
                }));
            }
        } else if self.wsrc_bake_bg_cache.is_none() {
            self.wsrc_bake_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wsrc_bake_bg"),
                layout: &self.wsrc_bake_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.wsrc_bake_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                ],
            }));
        }

        let cam = self.current_camera_world_pos();

        for c in 0..WSRC_CASCADE_COUNT as usize {
            if self.wsrc_built[c] {
                continue;
            }
            let extent = WSRC_CASCADE_EXTENTS[c];
            let cell = extent / WSRC_GRID_RES as f32;
            let origin = [
                (cam[0] / cell).round() * cell,
                (cam[1] / cell).round() * cell,
                (cam[2] / cell).round() * cell,
            ];
            self.wsrc_origin[c] = origin;

            let params = WsrcBakeParams {
                sun_dir: sun_dir_ws,
                sun_color,
                sky_color,
                grid: [origin[0], origin[1], origin[2], extent],
                shadow_vps,
                shadow_splits,
                flags: [
                    0.002,
                    if shadows_enabled { 1.0 } else { 0.0 },
                    c as f32,
                    0.0,
                ],
            };
            self.queue.write_buffer(
                &self.wsrc_bake_uniform,
                0,
                bytemuck::bytes_of(&params),
            );

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(if use_hw { "wsrc_bake_hw_pass" } else { "wsrc_bake_pass" }),
                timestamp_writes: None,
            });
            if use_hw {
                pass.set_pipeline(self.wsrc_bake_hw_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, self.wsrc_bake_hw_bg_cache.as_ref().unwrap(), &[]);
            } else {
                pass.set_pipeline(&self.wsrc_bake_pipeline);
                pass.set_bind_group(0, self.wsrc_bake_bg_cache.as_ref().unwrap(), &[]);
            }
            // One workgroup per probe in this cascade (16³),
            // 10×10 threads per workgroup (padded octel).
            pass.dispatch_workgroups(WSRC_GRID_RES, WSRC_GRID_RES, WSRC_GRID_RES);
            drop(pass);

            self.wsrc_built[c] = true;
            self.wsrc_last_sun_dir[c] = ld;
            self.wsrc_last_sun_color[c] = [sun_color[0], sun_color[1], sun_color[2]];
            self.wsrc_last_sky_color[c] = [sky_color[0], sky_color[1], sky_color[2]];
        }
    }

    /// Ticket 014 V1 — bake per-mesh unsigned distance fields via
    /// the compute pipeline. Drains `scene.pending_sdf_bakes` with a
    /// per-frame budget; expensive workload (O(voxels × triangles)
    /// per mesh), so the rate-limit keeps first-frame stutter
    /// bounded. Static scenes amortise and never re-bake.
    fn bake_pending_sdfs(
        &mut self,
        scene: &mut crate::scene::SceneGraph,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if !self.hw_rt_enabled || scene.pending_sdf_bakes.is_empty() {
            return;
        }
        const SDF_BAKE_MAX_PER_FRAME: usize = 8;
        let take = scene.pending_sdf_bakes.len().min(SDF_BAKE_MAX_PER_FRAME);
        let pending: Vec<f64> = scene.pending_sdf_bakes.drain(..take).collect();

        for handle in pending {
            let (sdf_view, vb_ptr, ib_ptr, bmin, bmax, index_count) = {
                let Some(node) = scene.nodes.get(handle) else { continue; };
                let Some(sdf_view) = node.mesh_sdf_view.as_ref() else { continue; };
                let Some(vb) = node.gpu_vb.as_ref() else { continue; };
                let Some(ib) = node.gpu_ib.as_ref() else { continue; };
                (
                    sdf_view.clone(),
                    vb.clone(),
                    ib.clone(),
                    node.bounds_min,
                    node.bounds_max,
                    node.gpu_index_count,
                )
            };
            if index_count == 0 {
                continue;
            }
            let tri_count = index_count / 3;
            let params = SdfBakeParams {
                aabb_min: [bmin[0], bmin[1], bmin[2], 0.0],
                aabb_max: [bmax[0], bmax[1], bmax[2], 0.0],
                counts: [tri_count, MESH_SDF_RES, 0, 0],
            };
            self.queue.write_buffer(
                &self.sdf_bake_uniform,
                0,
                bytemuck::bytes_of(&params),
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sdf_bake_bg"),
                layout: &self.sdf_bake_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.sdf_bake_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: vb_ptr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ib_ptr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sdf_view) },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf_bake_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sdf_bake_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(MESH_SDF_RES / 4, MESH_SDF_RES / 4, MESH_SDF_RES / 4);
        }
    }

    /// Ticket 013 V2 — re-light every populated card texel from the
    /// scene's directional sun + analytic sky. Runs once per frame
    /// before the HW probe trace reads the radiance atlas. For each
    /// atlas pixel the compute shader looks up its slot's world-space
    /// normal (baked at capture time), reads the albedo texel, and
    /// writes `albedo × (NdotL × sun + NdotUp × sky)` into the
    /// radiance atlas. No shadow-cascade lookup in V2; that's a V3
    /// ticket.
    fn light_mesh_cards(
        &mut self,
        scene: &crate::scene::SceneGraph,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Skip entirely when no meshes have cards allocated.
        if scene.next_card_slot == 0 {
            return;
        }
        let ld = self.lighting_uniforms.light_dir;
        let inv_len = 1.0 / (ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]).sqrt().max(1e-4);
        let sun_dir_ws = [-ld[0]*inv_len, -ld[1]*inv_len, -ld[2]*inv_len, ld[3]];
        let lc = self.lighting_uniforms.light_color;
        let sun_intensity = ld[3].max(0.0);
        let sun_color = [
            lc[0] * sun_intensity,
            lc[1] * sun_intensity,
            lc[2] * sun_intensity,
            0.0,
        ];
        let amb = self.lighting_uniforms.ambient;
        let sky_intensity = amb[3].max(0.0);
        let sky_color = [
            amb[0] * sky_intensity,
            amb[1] * sky_intensity,
            amb[2] * sky_intensity,
            0.0,
        ];
        // V3 — shadow cascade VPs, splits, and view matrix for the
        // per-texel shadow lookup. Cascades are stored on the
        // ShadowMap; when shadows are disabled we pass identity VPs
        // + flags.y = 0 so the shader skips the sample.
        let shadows_enabled = self.shadow_map.enabled;
        let shadow_vps: [[[f32; 4]; 4]; 3] = if shadows_enabled {
            self.shadow_map.light_vps
        } else {
            [IDENTITY_MAT4; 3]
        };
        let shadow_splits = if shadows_enabled {
            let s = self.shadow_map.cascade_splits;
            [s[0], s[1], s[2], 0.0]
        } else {
            [f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.0]
        };

        let params = CardLightParams {
            sun_dir: sun_dir_ws,
            sun_color,
            sky_color,
            atlas_info: [CARD_ATLAS_SIZE, CARD_SLOT_SIZE, CARD_SLOTS_PER_ROW, scene.next_card_slot],
            shadow_vps,
            shadow_splits,
            view_matrix: self.current_view_matrix,
            flags: [0.002, if shadows_enabled { 1.0 } else { 0.0 }, 0.0, 0.0],
        };
        self.queue.write_buffer(
            &self.card_light_uniform,
            0,
            bytemuck::bytes_of(&params),
        );

        if self.card_light_bg_cache.is_none() {
            self.card_light_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("card_light_bg"),
                layout: &self.card_light_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.card_light_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.mesh_card_atlas_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.card_slot_meta_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.mesh_card_radiance_view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.mesh_card_emissive_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                ],
            }));
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("card_light_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.card_light_pipeline);
        pass.set_bind_group(0, self.card_light_bg_cache.as_ref().unwrap(), &[]);
        // Dispatch over the populated slot range only. Each workgroup
        // is 8×8 pixels, each slot is CARD_SLOT_SIZE² pixels.
        let wg_per_slot = CARD_SLOT_SIZE / 8;
        let total_wg_x = wg_per_slot * CARD_SLOTS_PER_ROW;
        let total_wg_y =
            wg_per_slot * ((scene.next_card_slot + CARD_SLOTS_PER_ROW - 1) / CARD_SLOTS_PER_ROW);
        pass.dispatch_workgroups(total_wg_x, total_wg_y.max(1), 1);
    }

    /// Ticket 007b — build any pending BLASes and refresh the TLAS +
    /// per-instance GI data. Called once per frame, before the SSGI
    /// probe passes that sample the TLAS. No-op when HW RT is off or
    /// when nothing has changed since the last rebuild.
    ///
    /// Encodes `build_acceleration_structures` into the caller's
    /// `encoder` so the builds are ordered ahead of the trace pass
    /// without a separate queue submit.
    /// Ticket 014 V4 — rebuild the per-instance GI data buffer. Runs
    /// regardless of `hw_rt_enabled` so the SW SDF trace can also
    /// broad-phase this buffer at hit to sample the Mesh-Cards
    /// radiance atlas. Instance ordering matches the TLAS rebuild
    /// (nodes with a card slot, in scene order) so the `instance_count`
    /// / `instance_custom_data` conventions stay consistent across
    /// HW and SW trace paths.
    fn rebuild_instance_data(
        &mut self,
        scene: &crate::scene::SceneGraph,
        instance_handles: &[f64],
    ) -> (u32, bool) {
        let instance_count = instance_handles.len() as u32;
        let mut resized = false;
        let needed_cap = instance_count.max(self.tlas_max_instances).max(64);
        if self.tlas_instance_data_buffer.is_none()
            || needed_cap > self.tlas_max_instances
        {
            let new_cap = needed_cap.next_power_of_two();
            self.tlas_instance_data_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tlas_instance_data"),
                size: (new_cap as u64) * std::mem::size_of::<InstanceGiDataCpu>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.tlas_max_instances = new_cap;
            self.probe_trace_hw_bg_cache = [None, None];
            // V4 — SDF bind group also references instance_data, so
            // invalidate when the buffer is re-allocated.
            self.probe_trace_sdf_bg_cache = [None, None];
            // V14 — WSRC HW bake bg also references the TLAS +
            // instance_data buffer; invalidate on resize.
            self.wsrc_bake_hw_bg_cache = None;
            resized = true;
        }

        let mut instance_data: Vec<InstanceGiDataCpu> =
            Vec::with_capacity(instance_count as usize);
        for &h in instance_handles {
            let n = scene.nodes.get(h).unwrap();
            let e = n.material.emissive;
            let (first_slot, has_card) = match n.card_first_slot {
                Some(s) => (s as f32, 1.0_f32),
                None => (0.0, 0.0),
            };
            instance_data.push(InstanceGiDataCpu {
                albedo: n.flat_albedo,
                emissive_luma: (e[0] + e[1] + e[2]) * (1.0 / 3.0),
                normal_ws: n.flat_normal_ws,
                _pad0: 0.0,
                card_slot: [first_slot, 0.0, 0.0, has_card],
                card_aabb_min: [n.bounds_min[0], n.bounds_min[1], n.bounds_min[2], 0.0],
                card_aabb_max: [n.bounds_max[0], n.bounds_max[1], n.bounds_max[2], 0.0],
            });
        }
        if !instance_data.is_empty() {
            self.queue.write_buffer(
                self.tlas_instance_data_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&instance_data),
            );
        }
        (instance_count, resized)
    }

    fn rebuild_acceleration_structures(
        &mut self,
        scene: &mut crate::scene::SceneGraph,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // V4 — SW path also needs a populated instance_data buffer
        // for its broad-phase hit lookup. Collect instance handles
        // (nodes with a card slot, stable scene order) first, then
        // branch on `hw_rt_enabled` for TLAS/BLAS-specific work.
        let pending_blas = !scene.pending_blas_builds.is_empty();
        let version_changed = scene.tlas_version != self.tlas_built_version;
        if !pending_blas && !version_changed {
            return;
        }

        let mut instance_handles: Vec<f64> = Vec::new();
        for (h, n) in scene.nodes.iter() {
            if n.visible && n.card_first_slot.is_some() {
                instance_handles.push(h);
            }
        }
        let (instance_count, _resized) = self.rebuild_instance_data(scene, &instance_handles);

        // Everything below this point is HW-ray-tracing-specific
        // (TLAS build + BLAS batch). On SW adapters we're done.
        if !self.hw_rt_enabled {
            self.tlas_built_version = scene.tlas_version;
            return;
        }

        let pending = pending_blas;
        if instance_count == 0 && !pending {
            return;
        }

        if self.tlas.is_none() || instance_count > self.tlas_max_instances {
            let cap = self.tlas_max_instances.max(64);
            self.tlas = Some(self.device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("bloom_tlas"),
                max_instances: cap,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            }));
            self.probe_trace_hw_bg_cache = [None, None];
            // V14 — same reason as the resize path above.
            self.wsrc_bake_hw_bg_cache = None;
        }

        // Populate TLAS instance slots. Clear stale entries from prior
        // rebuilds by writing `None` over every slot up to the current
        // instance count (extras beyond stay None from the initial vec).
        {
            let tlas = self.tlas.as_mut().unwrap();
            for slot in 0..(instance_count as usize) {
                let n = scene.nodes.get(instance_handles[slot]).unwrap();
                let blas = n.blas.as_ref().unwrap();
                let t = &n.transform;
                // TlasInstance expects a row-major 3x4 affine transform
                // (rows × columns), i.e. [m00, m01, m02, m03, m10, ...].
                // Bloom stores column-major mat4 with translation in row 3.
                let transform_3x4 = [
                    t[0][0], t[1][0], t[2][0], t[3][0],
                    t[0][1], t[1][1], t[2][1], t[3][1],
                    t[0][2], t[1][2], t[2][2], t[3][2],
                ];
                tlas[slot] = Some(wgpu::TlasInstance::new(
                    blas,
                    transform_3x4,
                    slot as u32,
                    0xff,
                ));
            }
            // Clear any slots beyond the current instance count left
            // over from a previous rebuild that had more instances.
            for slot in (instance_count as usize)..(self.tlas_max_instances as usize) {
                if tlas[slot].is_some() {
                    tlas[slot] = None;
                }
            }
        }

        // Build any freshly-created BLASes + the TLAS in one call.
        // The BLAS size descriptors and geometry entries need to outlive
        // the build call, so we stash them in a pair of Vecs indexed in
        // parallel.
        let pending_handles: Vec<f64> = scene.pending_blas_builds.drain(..).collect();
        let size_descs: Vec<wgpu::BlasTriangleGeometrySizeDescriptor> = pending_handles
            .iter()
            .filter_map(|h| {
                let n = scene.nodes.get(*h)?;
                if n.blas.is_none() {
                    return None;
                }
                Some(wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: n.gpu_vertex_count,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(n.gpu_index_count),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                })
            })
            .collect();
        let mut build_entries: Vec<wgpu::BlasBuildEntry> =
            Vec::with_capacity(size_descs.len());
        for (i, h) in pending_handles.iter().enumerate() {
            let n = match scene.nodes.get(*h) {
                Some(n) => n,
                None => continue,
            };
            let blas = match n.blas.as_ref() { Some(b) => b, None => continue };
            let vb = match n.gpu_vb.as_ref() { Some(b) => b, None => continue };
            let ib = match n.gpu_ib.as_ref() { Some(b) => b, None => continue };
            build_entries.push(wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size_descs[i],
                        vertex_buffer: vb,
                        first_vertex: 0,
                        vertex_stride: std::mem::size_of::<Vertex3D>() as u64,
                        index_buffer: Some(ib),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            });
        }

        let tlas_ref = self.tlas.as_ref().unwrap();
        encoder.build_acceleration_structures(
            build_entries.iter(),
            std::iter::once(tlas_ref),
        );

        self.tlas_built_version = scene.tlas_version;
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

    /// Disables the ticket-004 shadow-cache and re-renders cascades
    /// every frame. Useful for games that mutate lighting state from
    /// places the cache invalidation path doesn't cover (e.g. day/
    /// night cycles driving light_dir from native code, per-frame
    /// deformable casters).
    pub fn set_shadows_always_fresh(&mut self, on: bool) {
        self.shadow_map.always_fresh = on;
        if on {
            self.shadow_map.invalidate();
        }
    }
    pub fn set_bloom_enabled(&mut self, on: bool) { self.bloom_enabled = on; }
    pub fn set_ssao_enabled(&mut self, on: bool) {
        if on && !self.ssao_enabled {
            // History was frozen while SSAO was off; reset the counter
            // so the first few frames after re-enabling seed fresh
            // instead of reusing stale accumulated AO.
            self.ssao_history_frame = 0;
        }
        self.ssao_enabled = on;
    }

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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
        // Phase 1c — clear last frame's material draws so the new
        // frame's submissions start from an empty list.
        self.material_system.commands.clear();
        self.material_system.translucent_commands.clear();
        self.material_system.reset_draw_slot();

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
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                _ => {
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
            // Only attach a depth target when we're drawing 3D. pipeline_2d is
            // depth-less; on some mobile Vulkan drivers (Adreno) pairing a
            // depth-less pipeline with a pass that carries a depth attachment
            // discards all draws silently. Matches the overlay_2d pass in
            // end_frame_with_scene, which also omits depth.
            let depth_attachment = if has_3d {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &owned_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                })
            } else {
                None
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
    pub fn end_frame_with_scene(&mut self, scene: &mut crate::scene::SceneGraph, profiler: &mut crate::profiler::Profiler) {
        profiler.begin("joint_flush");
        self.flush_joint_matrices();
        profiler.end("joint_flush");

        profiler.begin("surface_acquire");
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.surface.configure(&self.device, &self.surface_config);
                profiler.end("surface_acquire");
                return;
            }
        };
        profiler.end("surface_acquire");
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bloom_encoder"),
        });

        // Ticket 017: gate the entire Lumen warmup + per-frame
        // pipeline on ssgi_enabled. Every pass below exists solely
        // to feed the SSGI probe trace — when SSGI is off, running
        // them is pure waste and breaks the --quality 0 regression
        // guard (the Mesh-Cards backlog + per-frame card relight
        // cost held the "off" preset below 60 fps). Pending queues
        // (BLAS / SDF / card-capture) stay populated so a runtime
        // setSsgiEnabled(true) flip resumes baking from where it
        // paused instead of leaving the caches empty.
        if self.ssgi_enabled {
            // Ticket 007b: build any freshly-created BLASes and refresh
            // the TLAS before the SSGI probe trace pass can sample it.
            // No-op when HW RT is off or when nothing has changed.
            profiler.begin("accel_rebuild");
            self.rebuild_acceleration_structures(scene, &mut encoder);
            profiler.end("accel_rebuild");

            // Ticket 013: rasterise any new mesh cards into the shared
            // atlas. Drains `scene.pending_card_captures` — empty and
            // free after the first frame on a static scene.
            profiler.begin("card_capture");
            self.capture_pending_mesh_cards(scene, &mut encoder);
            profiler.end("card_capture");

            // Ticket 014 V1: bake per-mesh UDFs in the same frame encoder.
            // Runs alongside card capture until the scene's pending queue
            // is drained; no-op thereafter.
            profiler.begin("sdf_bake");
            self.bake_pending_sdfs(scene, &mut encoder);
            profiler.end("sdf_bake");

            // Ticket 014 V5: check if the camera has wandered past the
            // rebake threshold from the current clipmap centre; if so,
            // clear the `built` flag so the bake below fires again with
            // a re-centred origin. SDF trace falls back to Hi-Z for the
            // single frame between invalidation and re-bake completion.
            self.maybe_invalidate_sdf_clipmap();

            // Ticket 014 V2: bake the scene-wide SDF clipmap when it's
            // not already built. First frame after startup, and any
            // frame after V5 invalidation.
            profiler.begin("scene_sdf_clipmap");
            self.bake_scene_sdf_clipmap(scene, &mut encoder);
            profiler.end("scene_sdf_clipmap");

            // Ticket 014 V6: World-Space Radiance Cache. Invalidate on
            // camera travel past 30 m; re-bake when not built. Runs even
            // when HW RT is active so a future V7 can wire the HW miss
            // path into the same cache.
            self.maybe_invalidate_wsrc();
            profiler.begin("wsrc_bake");
            self.bake_wsrc(&mut encoder);
            profiler.end("wsrc_bake");

            // Ticket 013 V2: re-light the card atlas every frame so the
            // HW probe trace can sample pre-lit radiance at hit instead
            // of running the sun/sky math per ray.
            profiler.begin("card_light");
            self.light_mesh_cards(scene, &mut encoder);
            profiler.end("card_light");
        }

        // Shadow pass: render scene nodes from light's perspective into
        // cascaded shadow maps (3 cascades).
        //
        // Cache hit path (ticket 004): if no caster moved, the light
        // didn't move, and the freshly-computed cascade VPs match the
        // ones the cached depth textures were rendered with, we skip
        // the whole pass. The depth textures retain their content and
        // the main pass samples from them as if we had redrawn.
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
                // Use the pre-jitter projection so the cascade VPs
                // stay byte-stable when the camera is actually
                // stationary (the shadow cache compares them exactly).
                self.current_proj_matrix_unjittered,
                0.5,   // near — start cascades slightly past the camera
                80.0,  // far — shadow coverage range
                scene_bounds,
            );

            // Re-upload lighting uniforms with cascade VPs and splits.
            // Always write these — even on a cache hit the cascade
            // split distances and view matrix track camera movement
            // (they drive per-pixel cascade selection in the main
            // shader), which is independent of shadow texture content.
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

            // Cache gate. Skip if nothing that affects shadow-map
            // content has changed since last render. Texel-snap +
            // radius quantization in `compute_cascade_vps` makes this
            // check exact: identical scenes + identical poses (within
            // one cascade texel) produce byte-identical light_vps.
            let scene_ver = scene.shadow_version;
            let vps_changed = self.shadow_map.rendered_light_vps
                .as_ref()
                .map(|cached| *cached != self.shadow_map.light_vps)
                .unwrap_or(true);
            let light_changed = self.shadow_map.rendered_light_dir
                .map(|cached| cached != light_dir)
                .unwrap_or(true);
            let should_render = self.shadow_map.always_fresh
                || self.shadow_map.dirty
                || vps_changed
                || light_changed
                || self.shadow_map.rendered_scene_version != scene_ver;

            if should_render {
            // Build a shared caster list + buffer-ref vectors, then
            // filter per cascade against that cascade's ortho frustum.
            // A caster outside cascade N's frustum can't write pixels
            // into cascade N; near/far pancaking already covers
            // behind-camera casters via the cascade's own far plane.
            struct ShadowDrawEntry {
                vb_idx: usize,
                ib_idx: usize,
                index_count: u32,
                transform: [[f32; 4]; 4],
                wmin: [f32; 3],
                wmax: [f32; 3],
            }
            let mut shadow_nodes: Vec<ShadowDrawEntry> = Vec::new();
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
                    wmin: node.world_bounds_min,
                    wmax: node.world_bounds_max,
                });
            }

            let cascade_planes: [[[f32; 4]; 6]; crate::shadows::NUM_CASCADES] =
                std::array::from_fn(|c| {
                    crate::scene::extract_frustum_planes(&self.shadow_map.light_vps[c])
                });
            let mut cascade_indices: [Vec<usize>; crate::shadows::NUM_CASCADES] =
                std::array::from_fn(|_| Vec::with_capacity(shadow_nodes.len()));
            for (i, entry) in shadow_nodes.iter().enumerate() {
                let has_bounds = entry.wmin[0] <= entry.wmax[0];
                for c in 0..crate::shadows::NUM_CASCADES {
                    if has_bounds
                        && crate::scene::aabb_outside_frustum(&cascade_planes[c], entry.wmin, entry.wmax)
                    {
                        continue;
                    }
                    cascade_indices[c].push(i);
                }
            }

            // Render each cascade
            for cascade in 0..crate::shadows::NUM_CASCADES {
                let stride = crate::shadows::SHADOW_UNIFORM_STRIDE as usize;
                let max = crate::shadows::SHADOW_MAX_NODES as usize;
                let entries = &cascade_indices[cascade];
                let count = entries.len().min(max);
                let mut uniform_data: Vec<u8> = vec![0u8; stride * count.max(1)];
                let cascade_vp = self.shadow_map.light_vps[cascade];

                for (slot, &ei) in entries.iter().take(count).enumerate() {
                    let entry = &shadow_nodes[ei];
                    let uniforms = crate::shadows::ShadowUniforms {
                        light_vp: cascade_vp,
                        model: entry.transform,
                    };
                    let off = slot * stride;
                    uniform_data[off..off + std::mem::size_of::<crate::shadows::ShadowUniforms>()]
                        .copy_from_slice(bytemuck::bytes_of(&uniforms));
                }

                if count > 0 {
                    self.queue.write_buffer(
                        &self.shadow_map.uniform_buffer,
                        0,
                        &uniform_data[..count * stride],
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
                        multiview_mask: None,
                    });

                    shadow_pass.set_pipeline(&self.shadow_map.pipeline);

                    for (slot, &ei) in entries.iter().take(count).enumerate() {
                        let entry = &shadow_nodes[ei];
                        let offset = (slot * stride) as u32;
                        shadow_pass.set_bind_group(0, &self.shadow_map.uniform_bind_group, &[offset]);
                        shadow_pass.set_vertex_buffer(0, shadow_vbs[entry.vb_idx].slice(..));
                        shadow_pass.set_index_buffer(shadow_ibs[entry.ib_idx].slice(..), wgpu::IndexFormat::Uint32);
                        shadow_pass.draw_indexed(0..entry.index_count, 0, 0..1);
                    }
                }
            }

            // Cache bookkeeping — next frame will short-circuit if the
            // camera, scene, and light all stay put.
            self.shadow_map.rendered_light_vps = Some(self.shadow_map.light_vps);
            self.shadow_map.rendered_light_dir = Some(light_dir);
            self.shadow_map.rendered_scene_version = scene_ver;
            self.shadow_map.dirty = false;
            } // end should_render
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
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.material_rt_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            // Blank pixels clear to metallic=0. SSR's
                            // `metallic < 0.2` gate early-outs before
                            // roughness is read, so the roughness
                            // component of the clear is dead — leaving
                            // it at 0 instead of 1 keeps the material
                            // texture black in frame captures and
                            // avoids a false "green G-buffer" readout
                            // if the RT is ever viewed as RGBA.
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.velocity_rt_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            // Zero velocity = stationary pixel.
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.albedo_rt_view,
                        resolve_target: None,
                        depth_slice: None,
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
                multiview_mask: None,
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

        // Phase 2c — schedule the material pass through the render
        // graph. First real consumer of `renderer::graph` from #35.
        // For now a one-node graph; later phases add more nodes
        // (main_hdr, ssao, bloom, translucent, composite) and the
        // graph's topological sort picks the order from read/write
        // declarations.
        //
        // All per-frame borrows that the pass body needs are captured
        // here from `&self` before we build the context that wraps
        // `&mut encoder` + `&mut profiler`. Rust's borrow checker is
        // happy because the immutable and mutable borrows are
        // disjoint fields of the same struct.
        if !self.material_system.commands.is_empty() {
            use graph::{Graph, PassNode, PassOutput};

            let hdr_rt_view       = &self.hdr_rt_view;
            let material_rt_view  = &self.material_rt_view;
            let velocity_rt_view  = &self.velocity_rt_view;
            let albedo_rt_view    = &self.albedo_rt_view;
            let depth_view        = &self.depth_view;
            let material_system   = &self.material_system;
            let model_gpu_cache   = &self.model_gpu_cache;

            struct FrameCtx<'a> {
                encoder:  &'a mut wgpu::CommandEncoder,
                profiler: &'a mut crate::profiler::Profiler,
            }

            let mut graph: Graph<FrameCtx<'_>> = Graph::new();
            graph.push(
                PassNode::new("material_pass", Box::new(move |ctx: &mut FrameCtx| {
                    ctx.profiler.begin("material_pass");
                    {
                        let mat_ts = ctx.profiler.pass_timestamp_writes("material_pass");
                        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_material_pass"),
                            color_attachments: &[
                                Some(wgpu::RenderPassColorAttachment {
                                    view: hdr_rt_view,
                                    resolve_target: None, depth_slice: None,
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                }),
                                Some(wgpu::RenderPassColorAttachment {
                                    view: material_rt_view,
                                    resolve_target: None, depth_slice: None,
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                }),
                                Some(wgpu::RenderPassColorAttachment {
                                    view: velocity_rt_view,
                                    resolve_target: None, depth_slice: None,
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                }),
                                Some(wgpu::RenderPassColorAttachment {
                                    view: albedo_rt_view,
                                    resolve_target: None, depth_slice: None,
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                }),
                            ],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: depth_view,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: mat_ts,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        material_system.dispatch(&mut pass, |handle, idx| {
                            if let Some(Some(meshes)) = model_gpu_cache.get(&handle) {
                                if idx < meshes.len() {
                                    let mesh = &meshes[idx];
                                    return Some((&mesh.vb, &mesh.ib, mesh.index_count));
                                }
                            }
                            None
                        });
                    }
                    ctx.profiler.end("material_pass");
                }))
                // Writes HdrColor + the G-buffer so Phase 2d's scheduler
                // can order downstream passes (SSAO, bloom, translucent)
                // correctly once they're nodes too.
                .with_writes(&[
                    PassOutput::HdrColor,
                    PassOutput::MaterialRt,
                    PassOutput::VelocityRt,
                    PassOutput::AlbedoRt,
                    PassOutput::Depth,
                ]),
            );

            let mut ctx = FrameCtx { encoder: &mut encoder, profiler: &mut *profiler };
            if let Err(e) = graph.execute(&mut ctx) {
                eprintln!("[graph] material_pass failed: {:?}", e);
            }
        }

        // ============================================================
        // Phase 4b — translucent / refractive / additive material pass
        // ============================================================
        //
        // Runs after opaque materials, before post-FX. Loads hdr_rt so
        // opaque output survives; alpha-blends into it. Depth is
        // bound as read-only so translucent draws participate in the
        // depth test without writing.
        //
        // If any submitted translucent material declared
        // `reads_scene = true`, we first snapshot hdr_rt into a
        // swapchain-sized transient and bind that as group 4
        // scene_color_tex for the dispatch. Free after the pass so
        // the transient pool reuses on the next frame.
        if !self.material_system.translucent_commands.is_empty() {
            profiler.begin("translucent_pass");
            let swap_w = self.surface_config.width;
            let swap_h = self.surface_config.height;
            self.transient_pool.begin_frame(swap_w, swap_h);

            // Phase 7 — run the impulse decay + splat compute BEFORE
            // we build scene_inputs so the front view reflects this
            // frame's submissions.
            self.impulse_field.update(&self.device, &self.queue, &mut encoder);

            // Does any queued translucent material need the scene
            // colour snapshot?
            let needs_scene = self.material_system.translucent_commands
                .iter()
                .any(|c| self.material_system.pipelines
                    .get(c.material as usize - 1)
                    .and_then(|p| p.as_ref())
                    .map(|p| p.reads_scene)
                    .unwrap_or(false));

            let scene_color_tid = if needs_scene {
                let desc = transient::TransientDesc::new(
                    formats::HDR_FORMAT,
                    wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    transient::SizePolicy::Swapchain,
                );
                Some(self.transient_pool.acquire(&self.device, desc))
            } else {
                None
            };

            // Phase 4c — depth snapshot. wgpu forbids sampling a
            // texture that is also a depth-stencil attachment of the
            // same pass, so we copy the opaque depth buffer into a
            // transient before beginning the translucent pass and
            // bind the transient at group 4 binding 2. Acquired
            // whenever any translucent material reads_scene (same
            // gate as colour) — cheap enough that it's not worth a
            // separate `reads_depth` flag yet.
            let scene_depth_tid = if needs_scene {
                let desc = transient::TransientDesc::new(
                    formats::DEPTH_FORMAT,
                    wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    transient::SizePolicy::Swapchain,
                );
                Some(self.transient_pool.acquire(&self.device, desc))
            } else {
                None
            };

            // Snapshot hdr_rt + live depth -> transients.
            if let (Some(ctid), Some(dtid)) = (scene_color_tid, scene_depth_tid) {
                let color_tex = self.transient_pool.texture(ctid).expect("fresh color transient");
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.hdr_rt_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: color_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d { width: swap_w, height: swap_h, depth_or_array_layers: 1 },
                );
                let depth_tex = self.transient_pool.texture(dtid).expect("fresh depth transient");
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.depth_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::DepthOnly,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: depth_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::DepthOnly,
                    },
                    wgpu::Extent3d { width: swap_w, height: swap_h, depth_or_array_layers: 1 },
                );
                let color_view = self.transient_pool.view(ctid).unwrap();
                let depth_view = self.transient_pool.view(dtid).unwrap();
                let imp_view = self.impulse_field.front_view();
                let imp_samp = self.impulse_field.sampler();
                self.material_system.update_scene_inputs(
                    &self.device, color_view, Some(depth_view),
                    Some((imp_view, imp_samp)),
                );
            } else {
                // No refractive/depth-reading materials this frame —
                // still need a valid bind group. None → internal stubs.
                self.material_system.update_scene_inputs(
                    &self.device, &self.hdr_rt_view, None, None,
                );
            }

            {
                let t_ts = profiler.pass_timestamp_writes("translucent_pass");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_translucent_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.hdr_rt_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            // Translucents don't write depth — keep
                            // the opaque pass's depth pristine so
                            // downstream post-FX (SSR/SSGI) still
                            // sees the opaque geometry.
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: t_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let cache = &self.model_gpu_cache;
                self.material_system.dispatch_translucent(&mut pass, |handle, idx| {
                    if let Some(Some(meshes)) = cache.get(&handle) {
                        if idx < meshes.len() {
                            let mesh = &meshes[idx];
                            return Some((&mesh.vb, &mesh.ib, mesh.index_count));
                        }
                    }
                    None
                });
            }

            if let Some(tid) = scene_color_tid {
                self.transient_pool.release(tid);
            }
            profiler.end("translucent_pass");
        }

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
            // Temporal accumulation: ping-pong history textures.
            // `write_idx` is the current-frame output; `read_idx` the
            // previous frame's result. First 4 frames force alpha=1
            // so the initial clear never contaminates the signal.
            let write_idx = self.ssao_history_idx;
            let read_idx = 1 - write_idx;
            let frame_phase = self.ssao_history_frame % 4;
            let force_refresh = if self.ssao_history_frame < 4 { 1u32 } else { 0u32 };
            // 4-frame EMA: alpha = 1/4 = 0.25 gives equal weight to
            // each of the 4 phases at steady state.
            let alpha = 0.25_f32;
            // Halton-5 rotation: uncorrelated with TAA's base-2/3 jitter
            // so the two noise patterns don't resonate.
            let halton5 = halton(self.ssao_history_frame + 1, 5);
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
                size: [half_w, half_h, frame_phase, force_refresh],
                temporal: [alpha, halton5, 0.0, 0.0],
            };
            self.queue.write_buffer(&self.ssao_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            if self.ssao_bg_cache[write_idx].is_none() {
                self.ssao_bg_cache[write_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.ssao_history_views[read_idx]) },
                        wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                        wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.ssao_history_views[write_idx]) },
                    ],
                }));
            }
            let bg = self.ssao_bg_cache[write_idx].as_ref().unwrap();

            let ssao_ts = profiler.compute_pass_timestamp_writes("ssao_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssao_pass"),
                timestamp_writes: ssao_ts,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((half_w + 7) / 8, (half_h + 7) / 8, 1);

            // Flip ping-pong indices for the next frame.
            self.ssao_history_idx = read_idx;
            self.ssao_history_frame = self.ssao_history_frame.wrapping_add(1);
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                // n_steps lowered from 32 → 8 for stochastic SSR: the
                // GGX-sampled ray direction + jittered start offset +
                // temporal accumulation over 4–8 frames fills in the
                // gaps that any single-frame coarse march leaves behind.
                // Thickness tolerance grows proportionally with
                // step_size so the relative-error reject heuristic
                // still works with the larger strides.
                params: [self.ssr_strength, 8.0, 8.0, self.taa_frame_index as f32],
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ssr_ts,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            drop(pass);
        }

        // ============================================================
        // SSR temporal denoiser: blend the noisy single-ray SSR with
        // the reprojected previous history so 4–8 frames of GGX-sampled
        // rays converge to a smooth reflection. 3×3 pre-filter of the
        // noisy current frame + neighborhood clamp of reprojected
        // history. Compose then reads ssr_history[cur] instead of
        // ssr_rt.
        // ============================================================
        if self.ssr_enabled {
            let prev_idx = 1 - self.ssr_history_idx;
            let cur_idx = self.ssr_history_idx;

            // First frame: alpha=1 so we initialize history from the
            // current noisy frame rather than blending with zeros.
            let alpha = if self.taa_frame_index == 0 { 1.0_f32 } else { 0.1_f32 };
            let tp = SsrTemporalParams {
                params: [alpha, 0.0, 0.0, 0.0],
            };
            self.queue.write_buffer(&self.ssr_temporal_uniform_buffer, 0, bytemuck::bytes_of(&tp));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssr_temporal_bg"),
                layout: &self.ssr_temporal_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ssr_temporal_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ssr_rt_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.ssr_history_views[prev_idx]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssr_temporal_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssr_history_views[cur_idx],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.ssr_temporal_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // The compose pass reads denoised SSR from the current history
        // texture when ssr_enabled; otherwise the raw ssr_rt (which was
        // cleared to transparent above) so it contributes nothing.
        let ssr_composite_view = if self.ssr_enabled {
            &self.ssr_history_views[self.ssr_history_idx]
        } else {
            &self.ssr_rt_view
        };

        // ============================================================
        // Ticket 007a: Lumen-style screen-probe SSGI.
        // place → trace (SW Hi-Z) → temporal (EMA ping-pong) → resolve.
        // Resolve writes `ssgi_rt_view` so downstream compositing is
        // unchanged. When disabled we just clear `ssgi_rt_view` to
        // transparent (same fallback shape as the old per-pixel path).
        // ============================================================
        let half_w = (surf_w / 2).max(1);
        let half_h = (surf_h / 2).max(1);
        let gw = self.probe_grid_w;
        let gh = self.probe_grid_h;
        let write_idx = self.probe_history_idx;
        let prev_idx = 1 - write_idx;

        if self.ssgi_enabled {
            let p00 = self.current_proj_matrix[0][0];
            let p11 = self.current_proj_matrix[1][1];
            let p20 = self.current_proj_matrix[2][0];
            let p21 = self.current_proj_matrix[2][1];
            let inv_view = mat4_invert(self.current_view_matrix);

            // ---- place ----
            let place_params = ProbePlaceParams {
                inv_view,
                proj_row01: [p00, p11, p20, p21],
                size: [half_w, half_h, gw, gh],
                params: [self.taa_frame_index as f32, PROBE_TILE_SIZE as f32, 0.0, 0.0],
            };
            self.queue.write_buffer(&self.probe_place_uniform, 0, bytemuck::bytes_of(&place_params));
            if self.probe_place_bg_cache.is_none() {
                self.probe_place_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe_place_bg"),
                    layout: &self.probe_place_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.probe_place_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: self.probe_header_buffer.as_entire_binding() },
                    ],
                }));
            }
            {
                let ts = profiler.compute_pass_timestamp_writes("probe_place_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_place_pass"), timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_place_pipeline);
                pass.set_bind_group(0, self.probe_place_bg_cache.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups((gw + 7) / 8, (gh + 7) / 8, 1);
            }

            // ---- trace ----
            // Sun direction in world space — inverted because our
            // `light_dir` points from light toward the scene, while the
            // shader's NdotL expects the vector from the shading point
            // toward the light. Normalised because the shader doesn't.
            let ld = self.lighting_uniforms.light_dir;
            let sun_inv_len = 1.0 / (ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]).sqrt().max(1e-4);
            let sun_dir_ws = [
                -ld[0] * sun_inv_len,
                -ld[1] * sun_inv_len,
                -ld[2] * sun_inv_len,
                ld[3],
            ];
            // Sun colour = light_color × light intensity (ld.w). Sky
            // colour = ambient × ambient intensity (ambient.w) — a
            // crude dome irradiance, good enough for a one-bounce
            // shading estimate. Both fields are ignored by the SW
            // shader which inherits the same uniform struct layout.
            let lc = self.lighting_uniforms.light_color;
            let sun_intensity = ld[3].max(0.0);
            let sun_color = [
                lc[0] * sun_intensity,
                lc[1] * sun_intensity,
                lc[2] * sun_intensity,
                0.0,
            ];
            let amb = self.lighting_uniforms.ambient;
            let sky_intensity = amb[3].max(0.0);
            let sky_color = [
                amb[0] * sky_intensity,
                amb[1] * sky_intensity,
                amb[2] * sky_intensity,
                0.0,
            ];
            let trace_params = ProbeTraceParams {
                view: self.current_view_matrix,
                proj: self.current_proj_matrix,
                inv_view,
                proj_row01: [p00, p11, p20, p21],
                size: [half_w, half_h, gw, gh],
                params: [
                    self.taa_frame_index as f32,
                    self.ssgi_intensity,
                    self.ssgi_radius,
                    10.0,  // firefly luma cap
                ],
                sun_dir: sun_dir_ws,
                sun_color,
                sky_color,
                // Ticket 014 V3 — clipmap origin xyz + full extent w.
                // The SDF trace variant reads these; HW + Hi-Z ignore.
                clipmap: [
                    self.scene_sdf_clipmap_origin[0],
                    self.scene_sdf_clipmap_origin[1],
                    self.scene_sdf_clipmap_origin[2],
                    SCENE_SDF_CLIPMAP_EXTENT,
                ],
                // Ticket 014 V6/V13 — WSRC cascade cubes. `extent =
                // 0` marks an unbaked cascade; the shader's
                // `pick_cascade` helper skips those and falls through
                // to the next cascade (or returns black if none are
                // ready). First frame after startup all three are
                // unbaked → miss returns black, matching pre-V6.
                wsrc_cascades: [
                    [
                        self.wsrc_origin[0][0],
                        self.wsrc_origin[0][1],
                        self.wsrc_origin[0][2],
                        if self.wsrc_built[0] { WSRC_CASCADE_EXTENTS[0] } else { 0.0 },
                    ],
                    [
                        self.wsrc_origin[1][0],
                        self.wsrc_origin[1][1],
                        self.wsrc_origin[1][2],
                        if self.wsrc_built[1] { WSRC_CASCADE_EXTENTS[1] } else { 0.0 },
                    ],
                    [
                        self.wsrc_origin[2][0],
                        self.wsrc_origin[2][1],
                        self.wsrc_origin[2][2],
                        if self.wsrc_built[2] { WSRC_CASCADE_EXTENTS[2] } else { 0.0 },
                    ],
                ],
            };
            self.queue.write_buffer(&self.probe_trace_uniform, 0, bytemuck::bytes_of(&trace_params));
            // V3 — trace BG now binds the prev-frame history view at
            // binding 11. `prev_idx` ping-pongs every frame so we
            // cache both slots independently.
            if self.probe_trace_bg_cache[prev_idx].is_none() {
                self.probe_trace_bg_cache[prev_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe_trace_bg"),
                    layout: &self.probe_trace_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.probe_trace_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hiz_views[1]) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.hiz_views[2]) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.hiz_views[3]) },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.hiz_views[4]) },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                        wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                        wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                    ],
                }));
            }
            // HW trace needs both the TLAS (at least one instance) and
            // the instance-data buffer to exist. Fall back to SDF or
            // Hi-Z when either is missing on an HW-enabled adapter
            // (e.g. first frame before the scene has loaded any
            // geometry).
            let use_hw = self.hw_rt_enabled
                && self.probe_trace_hw_pipeline.is_some()
                && self.tlas.is_some()
                && self.tlas_instance_data_buffer.is_some();
            // Ticket 014 V3/V4 — pick SDF sphere-trace over Hi-Z when
            // the scene clipmap is baked AND the instance-data buffer
            // is ready (needed for broad-phase textured hit sampling
            // added in V4). Otherwise fall through to Hi-Z. HW still
            // wins over both when the feature was granted.
            let use_sdf = !use_hw
                && self.scene_sdf_clipmap_built
                && self.tlas_instance_data_buffer.is_some();

            if use_hw {
                // Build the HW bind group lazily. V3 uses a per-
                // prev_idx slot since the prev-frame history view
                // ping-pongs each frame.
                if self.probe_trace_hw_bg_cache[prev_idx].is_none() {
                    let tlas = self.tlas.as_ref().unwrap();
                    self.probe_trace_hw_bg_cache[prev_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("probe_trace_hw_bg"),
                        layout: self.probe_trace_hw_layout.as_ref().unwrap(),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: self.probe_trace_uniform.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: tlas.as_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: self.tlas_instance_data_buffer.as_ref().unwrap().as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                            // Ticket 013 V2: the HW trace samples the
                            // *radiance* atlas (pre-lit by card_light_pass)
                            // at hit, not the raw albedo atlas.
                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.mesh_card_radiance_view) },
                            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                            // V7/V10 — WSRC atlas + linear sampler.
                            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.wsrc_atlas_sampler) },
                            // V3 — prev-frame probe history.
                            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                        ],
                    }));
                }
                let ts = profiler.compute_pass_timestamp_writes("probe_trace_hw_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_trace_hw_pass"), timestamp_writes: ts,
                });
                pass.set_pipeline(self.probe_trace_hw_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, self.probe_trace_hw_bg_cache[prev_idx].as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(gw, gh, 1);
            } else if use_sdf {
                // Ticket 014 V3 — SW SDF sphere-trace path.
                // V3 (ticket 016) uses a per-prev_idx slot for the
                // prev-frame history binding.
                if self.probe_trace_sdf_bg_cache[prev_idx].is_none() {
                    let nf_samp = self.device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("clipmap_nonfiltering_sampler"),
                        address_mode_u: wgpu::AddressMode::ClampToEdge,
                        address_mode_v: wgpu::AddressMode::ClampToEdge,
                        address_mode_w: wgpu::AddressMode::ClampToEdge,
                        mag_filter: wgpu::FilterMode::Nearest,
                        min_filter: wgpu::FilterMode::Nearest,
                        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                        ..Default::default()
                    });
                    let instance_buf = self.tlas_instance_data_buffer.as_ref()
                        .expect("V4: instance_data buffer must exist before SDF dispatch");
                    self.probe_trace_sdf_bg_cache[prev_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("probe_trace_sdf_bg"),
                        layout: &self.probe_trace_sdf_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: self.probe_trace_uniform.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.scene_sdf_clipmap_view) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&nf_samp) },
                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                            wgpu::BindGroupEntry { binding: 5, resource: instance_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.mesh_card_radiance_view) },
                            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                            // V6/V10 — WSRC atlas + linear sampler.
                            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.wsrc_atlas_sampler) },
                            // V3 — prev-frame probe history.
                            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                        ],
                    }));
                }
                let ts = profiler.compute_pass_timestamp_writes("probe_trace_sdf_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_trace_sdf_pass"), timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_trace_sdf_pipeline);
                pass.set_bind_group(0, self.probe_trace_sdf_bg_cache[prev_idx].as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(gw, gh, 1);
            } else {
                let ts = profiler.compute_pass_timestamp_writes("probe_trace_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_trace_pass"), timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_trace_pipeline);
                pass.set_bind_group(0, self.probe_trace_bg_cache[prev_idx].as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(gw, gh, 1);
            }

            // ---- temporal (EMA) ----
            // First frame forces alpha=1 so the history is seeded from
            // the current trace rather than blending against a zero clear.
            let force_refresh = if self.taa_frame_index == 0 { 1.0_f32 } else { 0.0_f32 };
            let temporal_params = ProbeTemporalParams {
                params: [0.25, force_refresh, gw as f32, gh as f32],
            };
            self.queue.write_buffer(&self.probe_temporal_uniform, 0, bytemuck::bytes_of(&temporal_params));
            // Bind group indexed by write_idx: each direction of the
            // ping-pong (read prev, write write) gets its own cached BG.
            if self.probe_temporal_bg_cache[write_idx].is_none() {
                self.probe_temporal_bg_cache[write_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe_temporal_bg"),
                    layout: &self.probe_temporal_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.probe_temporal_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[write_idx]) },
                    ],
                }));
            }
            {
                let ts = profiler.compute_pass_timestamp_writes("probe_temporal_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_temporal_pass"), timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_temporal_pipeline);
                pass.set_bind_group(0, self.probe_temporal_bg_cache[write_idx].as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(gw, gh, 1);
            }

            // ---- resolve ----
            let resolve_params = ProbeResolveParams {
                inv_view,
                proj_row01: [p00, p11, p20, p21],
                size: [half_w, half_h, gw, gh],
                params: [PROBE_TILE_SIZE as f32, 1.0, 0.0, 0.0],
            };
            self.queue.write_buffer(&self.probe_resolve_uniform, 0, bytemuck::bytes_of(&resolve_params));
            if self.probe_resolve_bg_cache[write_idx].is_none() {
                self.probe_resolve_bg_cache[write_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe_resolve_bg"),
                    layout: &self.probe_resolve_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.probe_resolve_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[write_idx]) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                    ],
                }));
            }
            let ts = profiler.pass_timestamp_writes("probe_resolve_pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("probe_resolve_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssgi_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.probe_resolve_pipeline);
            pass.set_bind_group(0, self.probe_resolve_bg_cache[write_idx].as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        } else {
            // SSGI disabled — clear the resolve target so downstream
            // composite reads contribute zero.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssgi_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssgi_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            drop(pass);
        }

        // The resolve pass writes directly into `ssgi_rt_view`, so
        // downstream composite + TAA reads are unchanged from the
        // legacy path.
        let ssgi_composite_view = &self.ssgi_rt_view;

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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: bloom_ts,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
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
                multiview_mask: None,
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
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(ssr_composite_view) },
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: taa_ts,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
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
                    depth_slice: None,
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
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        profiler.end("post_fx");

        // ============================================================
        // 2D pass: immediate-mode 2D geometry on top of composited image
        // ============================================================
        // Phase 2d — overlay_2d ported to a graph node. Second real
        // consumer of `renderer::graph` after material_pass. Lives in
        // its own one-node graph here at the end of the frame because
        // it needs to run AFTER the hand-encoded composite has
        // written the swapchain. Phase 2e+ will merge into a single
        // frame-wide graph as more passes port.
        if has_2d {
            use graph::{Graph, PassNode, PassOutput};

            let pipeline_2d        = &self.pipeline_2d;
            let persistent_vb_2d   = &self.persistent_vb_2d;
            let persistent_ib_2d   = &self.persistent_ib_2d;
            let uniform_bind_groups = &self.uniform_bind_groups;
            let texture_bind_groups = &self.texture_bind_groups;
            let draw_calls_2d      = &self.draw_calls_2d;
            let indices_2d_len     = self.indices_2d.len() as u32;
            let view_ref           = &view;

            struct OverlayCtx<'a> {
                encoder:  &'a mut wgpu::CommandEncoder,
                profiler: &'a mut crate::profiler::Profiler,
            }

            let mut graph: Graph<OverlayCtx<'_>> = Graph::new();
            graph.push(
                PassNode::new("overlay_2d", Box::new(move |ctx: &mut OverlayCtx| {
                    ctx.profiler.begin("overlay_2d");
                    {
                        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_2d_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: view_ref,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_pipeline(pipeline_2d);
                        pass.set_vertex_buffer(0, persistent_vb_2d.slice(..));
                        pass.set_index_buffer(persistent_ib_2d.slice(..), wgpu::IndexFormat::Uint32);

                        let num_calls = draw_calls_2d.len();
                        for i in 0..num_calls {
                            let call = &draw_calls_2d[i];
                            let next_start = if i + 1 < num_calls {
                                draw_calls_2d[i + 1].index_start
                            } else {
                                indices_2d_len
                            };
                            let count = next_start - call.index_start;
                            if count == 0 { continue; }

                            pass.set_bind_group(0, &uniform_bind_groups[call.uniform_idx as usize], &[]);
                            if (call.texture_idx as usize) < texture_bind_groups.len() {
                                pass.set_bind_group(1, &texture_bind_groups[call.texture_idx as usize], &[]);
                            }
                            pass.draw_indexed(call.index_start..next_start, 0, 0..1);
                        }
                    }
                    ctx.profiler.end("overlay_2d");
                }))
                .with_writes(&[PassOutput::Swapchain]),
            );

            let mut ctx = OverlayCtx { encoder: &mut encoder, profiler: &mut *profiler };
            if let Err(e) = graph.execute(&mut ctx) {
                eprintln!("[graph] overlay_2d failed: {:?}", e);
            }
        } else {
            // Empty graphs are still valid — execute a no-op so the
            // profiler bracket is symmetric with the populated path.
            profiler.begin("overlay_2d");
            profiler.end("overlay_2d");
        }

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
            let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

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
            profiler.begin("queue_submit");
            self.queue.submit(std::iter::once(encoder.finish()));
            profiler.end("queue_submit");
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        profiler.begin("swap_present");
        output.present();
        profiler.end("swap_present");

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
        // Swap probe-history ping-pong so next frame reads what we
        // just blended as the "previous" history and writes to the
        // other buffer. Ticket 007a.
        if self.ssgi_enabled {
            self.probe_history_idx = 1 - self.probe_history_idx;
        }
        // Same ping-pong for SSR temporal accumulation.
        if self.ssr_enabled {
            self.ssr_history_idx = 1 - self.ssr_history_idx;
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
        // Capture the pre-jitter projection for shadow cascade fitting.
        // TAA's sub-pixel nudge would otherwise change the cascade VPs
        // every frame and defeat the cache.
        self.current_proj_matrix_unjittered = proj;

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
        self.pending_skin_groups.push(matrices.to_vec());
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

        self.pending_skin_groups.push(scaled);
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
        // Accumulator = all skinned poses staged by skinned drawModel
        // calls this frame, packed consecutively. Each draw's vertex
        // joint indices were pre-offset at submit time, so a single
        // flat upload here is all the GPU needs.
        const MAX_JOINT_SLOTS: usize = 1024;
        let count = self.frame_joint_data.len().min(MAX_JOINT_SLOTS);
        if count > 0 {
            let mut all_data = vec![[[0.0f32; 4]; 4]; MAX_JOINT_SLOTS];
            for i in 0..count {
                all_data[i] = self.frame_joint_data[i];
            }
            self.queue.write_buffer(&self.joint_buffer, 0, bytemuck::cast_slice(&all_data));
        }
        self.frame_joint_data.clear();
        // Any leftover staged poses are stale (no draw consumed them).
        self.pending_skin_groups.clear();
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
        // Note: the shadow cache reads `lighting_uniforms.light_dir`
        // directly at gate time, so no explicit invalidate is needed
        // here. Doing it via a setter would miss light changes that
        // happen through other paths (preset reset, etc.) anyway.
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

        // If this mesh is skinned, consume the next pending pose
        // (FIFO) and pack its matrices into the frame accumulator at
        // the current cursor. Each vertex's joint indices then get
        // shifted by that cursor so the shader samples this mesh's
        // slice of the shared joint buffer. With a 1024-slot buffer,
        // multiple skinned models can coexist in one frame.
        let mesh_skinned = vertices.iter().any(|v|
            v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01);
        let joint_offset: f32 = if mesh_skinned && !self.pending_skin_groups.is_empty() {
            let group = self.pending_skin_groups.remove(0);
            let start = self.frame_joint_data.len();
            // Cap at the 1024-slot buffer. Overflowing poses land at
            // offset 0, which at least avoids an out-of-range read —
            // the model will look mis-posed but not corrupt memory.
            if start + group.len() <= 1024 {
                self.frame_joint_data.extend_from_slice(&group);
                start as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

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
            let joints_out = if is_skinned {
                [v.joints[0] + joint_offset,
                 v.joints[1] + joint_offset,
                 v.joints[2] + joint_offset,
                 v.joints[3] + joint_offset]
            } else {
                v.joints
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
                joints: joints_out,
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
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return None,
        };
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
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

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
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        self.custom_pipelines.push(pipeline);
        self.custom_pipelines.len() // 1-based index
    }
}

// ============================================================
// Phase 1c — material system (new shader-ABI draw path)
// ============================================================
//
// Separate impl block so material_system's public surface on Renderer
// stays co-located and easy to audit. All public methods either
// compile a material, submit a draw, or sync per-frame uniforms.

impl Renderer {
    /// Compile a material from user-supplied WGSL source. Returns a
    /// handle to use with `submit_material_draw`. The source may
    /// `#include "material_abi.wgsl"` and any `common/*.wgsl` header.
    pub fn compile_material(
        &mut self, wgsl_source: &str,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        self.compile_material_with_options(
            wgsl_source,
            material_pipeline::FragmentProfile::Opaque,
            material_pipeline::Bucket::Opaque,
            false,
        )
    }

    /// Phase 4a — full-control material compile. Games that want a
    /// translucent / refractive / additive material (or a non-default
    /// bucket) call this directly. Plain `compile_material` is a
    /// convenience for Opaque + no scene reads.
    pub fn compile_material_with_options(
        &mut self, wgsl_source: &str,
        profile:     material_pipeline::FragmentProfile,
        bucket:      material_pipeline::Bucket,
        reads_scene: bool,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        self.material_system.compile(
            &self.device,
            wgsl_source,
            profile,
            bucket,
            reads_scene,
            formats::HDR_FORMAT,
            formats::MATERIAL_FORMAT,
            formats::VELOCITY_FORMAT,
            wgpu::TextureFormat::Rgba8Unorm,
            formats::DEPTH_FORMAT,
        )
    }

    /// Phase 6 — compile a material from a WGSL file on disk and
    /// register the path with the hot-reload watcher. Subsequent
    /// edits to that file fire a recompile in the next
    /// `poll_material_hot_reload` call (drained from `end_frame`).
    pub fn compile_material_from_file(
        &mut self,
        path:        &std::path::Path,
        profile:     material_pipeline::FragmentProfile,
        bucket:      material_pipeline::Bucket,
        reads_scene: bool,
    ) -> Result<material_system::MaterialHandle, String> {
        let canonical = std::fs::canonicalize(path)
            .map_err(|e| format!("canonicalize {path:?}: {e}"))?;
        let source = std::fs::read_to_string(&canonical)
            .map_err(|e| format!("read {canonical:?}: {e}"))?;
        let handle = self.compile_material_with_options(&source, profile, bucket, reads_scene)
            .map_err(|e| format!("compile {canonical:?}: {e:?}"))?;
        self.material_hot_reload.register(
            handle,
            hot_reload::FileMaterialDesc { path: canonical, profile, bucket, reads_scene },
        );
        Ok(handle)
    }

    /// Drain pending hot-reload events and rebuild affected pipelines.
    /// Logs failures and keeps the previous pipeline in place — never
    /// kills the running game.
    pub fn poll_material_hot_reload(&mut self) {
        let pending = self.material_hot_reload.drain_pending();
        for (handle, desc) in pending {
            let source = match std::fs::read_to_string(&desc.path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[hot_reload] read {:?} failed: {e}", desc.path);
                    continue;
                }
            };
            // Compile fresh; on success, replace the slot. On failure,
            // log and keep the old pipeline running.
            match self.material_system.compile(
                &self.device, &source,
                desc.profile, desc.bucket, desc.reads_scene,
                formats::HDR_FORMAT,
                formats::MATERIAL_FORMAT,
                formats::VELOCITY_FORMAT,
                wgpu::TextureFormat::Rgba8Unorm,
                formats::DEPTH_FORMAT,
            ) {
                Ok(new_handle) => {
                    // material_system.compile pushes a NEW slot.
                    // Move the new pipeline into the original slot
                    // and clear the trailing one so handles don't
                    // leak.
                    let new_idx = (new_handle - 1) as usize;
                    let old_idx = (handle - 1) as usize;
                    if let Some(p) = self.material_system.pipelines.get_mut(new_idx).and_then(|s| s.take()) {
                        if let Some(slot) = self.material_system.pipelines.get_mut(old_idx) {
                            *slot = Some(p);
                        }
                    }
                    eprintln!("[hot_reload] reloaded {:?} (handle {handle})", desc.path);
                }
                Err(e) => {
                    eprintln!("[hot_reload] compile {:?} failed: {e:?} — keeping previous", desc.path);
                }
            }
        }
    }

    /// Submit a material draw against a cached mesh. `mesh_handle` is
    /// the value returned by `cache_model_if_static` (same as the
    /// scene pipeline's cached-model path).
    pub fn submit_material_draw(
        &mut self,
        material: material_system::MaterialHandle,
        mesh_handle: u64,
        mesh_idx: usize,
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
    ) {
        let model = mat4_multiply(
            mat4_translate(IDENTITY_MAT4, position),
            mat4_scale(IDENTITY_MAT4, [scale, scale, scale]),
        );
        let mvp = mat4_multiply(self.current_vp_matrix, model);
        self.material_system.submit_draw(
            &self.device, &self.queue, &self.joint_buffer,
            material, mesh_handle, mesh_idx,
            mvp, model, mvp, tint, [0, 0, 0, 0],
        );
    }

    /// Sync PerFrame + PerView uniforms from current renderer state.
    /// FFI callers drive this from their frame boundary so `PerFrame.time`
    /// reflects the real process-uptime clock.
    pub fn material_system_begin_frame(&mut self, time_seconds: f32, delta_time: f32) {
        let screen_w = self.surface_config.width as f32;
        let screen_h = self.surface_config.height as f32;
        let (rw, rh) = self.render_extent();
        let per_frame = material_system::PerFrameUniforms {
            time: time_seconds,
            delta_time,
            frame_index: self.taa_frame_index as u32,
            _pad0: 0,
            screen_resolution: [screen_w, screen_h],
            render_resolution: [rw as f32, rh as f32],
            taa_jitter: [0.0, 0.0],
            _pad1: [0.0, 0.0],
        };
        let per_view = material_system::PerViewUniforms {
            view:           self.current_view_matrix,
            proj:           self.current_proj_matrix,
            view_proj:      self.current_vp_matrix,
            prev_view_proj: self.prev_vp_matrix,
            inv_proj:       self.current_inv_proj_matrix,
            camera_pos: [
                self.current_camera_pos[0],
                self.current_camera_pos[1],
                self.current_camera_pos[2],
                self.lighting_uniforms.camera_pos[3],
            ],
            camera_dir: [0.0, 0.0, -1.0, 70.0_f32.to_radians()],
            ambient:    self.lighting_uniforms.ambient,
            fog:        [self.fog_color[0], self.fog_color[1], self.fog_color[2], self.fog_density],
            sun_dir:    self.lighting_uniforms.light_dir,
            sun_color:  self.lighting_uniforms.light_color,
            dir_light_count:   self.lighting_uniforms.dir_light_count,
            dir_lights:        std::array::from_fn(|i| material_system::PerViewDirLight {
                direction: self.lighting_uniforms.dir_lights[i].direction,
                color:     self.lighting_uniforms.dir_lights[i].color,
            }),
            point_light_count: self.lighting_uniforms.point_light_count,
            point_lights:      std::array::from_fn(|i| material_system::PerViewPointLight {
                position: self.lighting_uniforms.point_lights[i].position,
                color:    self.lighting_uniforms.point_lights[i].color,
            }),
            shadow_splits:   self.lighting_uniforms.shadow_cascade_splits,
            shadow_view:     self.lighting_uniforms.shadow_view_matrix,
            shadow_cascades: self.lighting_uniforms.shadow_cascade_vps,
        };
        self.material_system.update_frame_uniforms(&self.queue, &per_frame, &per_view);
    }
}
