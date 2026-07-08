use wgpu::util::DeviceExt;
use std::collections::HashMap;

mod shaders;
mod texture_store;
mod draw2d;
mod hiz;
mod occlusion;
mod ssr_pass;
mod ssgi_pass;
mod shadow_pass;
mod postfx_chain;
mod scene_pass;
mod gi_bake;
mod froxel;
mod lighting;
pub use occlusion::OcclusionCuller;
use shaders::*;

pub mod shader_include;
pub mod shader_library;
pub mod material_pipeline;
pub mod material_system;
pub mod planar_reflection;
pub mod graph;
pub mod transient;
pub mod impulse_field;
pub mod hot_reload;
pub mod post_pass;

mod util;
pub use util::{
    IDENTITY_MAT4,
    mat4_perspective, mat4_ortho, mat4_look_at,
    mat4_multiply, mat4_mul_vec4,
    mat4_translate, mat4_scale, mat4_invert,
};
#[cfg(not(target_arch = "wasm32"))] // file-writing screenshot path
use util::encode_png_simple;

mod brdf_lut;
use brdf_lut::build_brdf_lut;
mod atmosphere_lut;
use atmosphere_lut::{
    build_multi_scattering_lut, build_transmittance_lut, AERIAL_D, AERIAL_H,
    AERIAL_MAX_DIST_KM, AERIAL_W, MULTI_SCATTERING_SIZE, SKY_VIEW_H, SKY_VIEW_W,
    TRANSMITTANCE_H, TRANSMITTANCE_W,
};

mod formats;
use formats::{
    DEPTH_FORMAT, HDR_FORMAT, SSAO_FORMAT, MATERIAL_FORMAT,
    HIZ_FORMAT, VELOCITY_FORMAT, BLOOM_MIP_COUNT, HIZ_MIP_COUNT,
    create_depth_texture, create_hdr_rt, create_material_rt,
    create_albedo_rt, create_velocity_rt, create_ssr_rt,
    create_ssr_history_textures,
    create_ssgi_rt, create_probe_trace_tex, create_probe_history_textures,
    probe_grid_dims, PROBE_TILE_SIZE,
    create_mesh_card_atlas, create_mesh_card_emissive_atlas,
    create_mesh_card_radiance_atlas,
    CARD_ATLAS_SIZE, CARD_SLOT_SIZE, CARD_SLOTS_PER_ROW, CARD_MAX_SLOTS,
    CARD_AXES_PER_MESH, MESH_SDF_RES,
    create_scene_sdf_clipmap, create_scene_sdf_clipmap_staging,
    SCENE_SDF_CLIPMAP_RES,
    SCENE_SDF_CLIPMAP_EXTENT, SCENE_SDF_CLIPMAP_REBAKE_THRESHOLD,
    SCENE_SDF_CLIPMAP_BIN_CELLS, SCENE_SDF_CLIPMAP_LAYERS_PER_FRAME,
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

// EN-005 Phase 2 — uniforms for the procedural sky path.
//
// `SkyViewParams` drives the sky-view LUT compute shader (recomputed
// when the sun moves). `SunUniforms` drives the per-frame sun-disk
// composite in the procedural sky fragment shader.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyViewParams {
    /// xyz = sun direction (world space, unit), w = sun intensity scale.
    sun: [f32; 4],
    /// x = rayleigh density mult, y = mie density mult, z = ground albedo, w unused.
    knobs: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SunUniforms {
    /// xyz = sun direction (unit), w = sun intensity.
    sun: [f32; 4],
    /// x = sun angular radius (rad), y = limb darkening (0..1), zw unused.
    params: [f32; 4],
}

// EN-005 V2 — aerial-perspective compute uniforms.
//
// Driven each frame from the renderer (camera + sun + atmosphere
// knobs). Layout must mirror `AerialParams` in
// AERIAL_PERSPECTIVE_SHADER_WGSL exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AerialParams {
    /// xyz = camera world position (engine units, metres assumed),
    /// w = max distance the LUT covers (km).
    cam_pos: [f32; 4],
    /// World→clip inverse, used per-voxel to reconstruct view rays
    /// from NDC.
    inv_vp: [[f32; 4]; 4],
    /// xyz = sun direction (unit), w = sun intensity scalar.
    sun: [f32; 4],
    /// x = rayleigh density mult, y = mie density mult, zw unused.
    knobs: [f32; 4],
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct HizLinearizeParams {
    /// xy = inv_size, z = proj[2][2], w = proj[3][2]
    params: [f32; 4],
    /// xy = mip-0 size, zw unused
    size: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct HizDownsampleParams {
    /// xy = dst-mip size, zw unused
    size: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SsaoParams {
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
pub(super) struct SsaoBlurParams {
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

/// Fullscreen-lag fix — an in-flight amortized scene-SDF-clipmap bake.
/// A job bakes `SCENE_SDF_CLIPMAP_LAYERS_PER_FRAME` voxel Z-layers per
/// frame into the staging texture; when the last slice lands the staging
/// contents are copied over the live clipmap and the origin flips
/// atomically, so traces never observe a half-baked field. The bind
/// group keeps the job's transient buffers alive.
struct SdfClipmapBakeJob {
    origin: [f32; 3],
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    next_z: u32,
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
    /// EN-023 — xyz = scene-average albedo (mean of card-instance flat
    /// albedos) for the SW bake's ground-bounce term; w unused. The HW
    /// bake ignores it (it traces real geometry).
    ground_albedo: [f32; 4],
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
    /// Object-space AABB min (xyz) + unused pad (w). The HW paths
    /// transform hits into object space (hit.world_to_object) and
    /// compare against THESE — do not world-ify them.
    card_aabb_min: [f32; 4],
    /// Object-space AABB max (xyz) + unused pad (w).
    card_aabb_max: [f32; 4],
    /// EN-023 — WORLD-space AABB min/max. The SDF trace has no
    /// world_to_object (it marches a world-space clipmap), so its
    /// broad-phase compares the world hit against these. With the old
    /// object-space-only bounds, every transformed instance fell
    /// through to the flat-gray analytic fallback — zero colored
    /// bounce on non-RT adapters (round-2 audit F4).
    world_aabb_min: [f32; 4],
    world_aabb_max: [f32; 4],
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
    /// x=strength, y=max_dist, z=n_steps, w=frame index
    params: [f32; 4],
    /// EN-021 — view→world rotation (transpose of the view 3×3) for the
    /// env-miss fallback's direction lookup.
    inv_view_rot: [[f32; 4]; 4],
    /// EN-021 — x = env max LOD, y = env intensity, zw unused.
    params2: [f32; 4],
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
struct UpscaleParams {
    /// x = mode (0 = bilinear, 1 = Catmull-Rom), yzw padding.
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RcasParams {
    /// x = sharpen strength (0 off, 1 max), yzw padding.
    params: [f32; 4],
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
    /// Alpha-tested shadow bind group (base-colour tex + sampler + cutoff),
    /// built only for cutout materials (alpha_cutoff > 0). When present, the
    /// shadow pass renders this mesh with the cutout pipeline so foliage casts
    /// its real shape. `None` → opaque caster, uses the plain shadow pipeline.
    shadow_cutout_bg: Option<wgpu::BindGroup>,
    _shadow_cutoff_buf: Option<wgpu::Buffer>,
}

struct CachedModelDraw {
    uniform_slot: usize,
    cache_handle: u64,
    mesh_idx: usize,
    /// Object→world model matrix for this draw, kept CPU-side so the
    /// shadow pass can render the model depth-only from the light.
    model: [[f32; 4]; 4],
}

// ============================================================
// Renderer
// ============================================================

/// 0-255 sRGB channel → linear f32. The 2D pass renders into an sRGB
/// swapchain view whose hardware encode expects LINEAR shader output;
/// passing the sRGB byte value straight through double-encoded every 2D
/// color (washed-bright HUD, gamma-skewed AA edges — round-2 audit F5).
/// Alpha stays linear by definition and is NOT decoded.
pub(crate) fn srgb_u8_to_linear(c: f64) -> f32 {
    let c = (c / 255.0).clamp(0.0, 1.0);
    (if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }) as f32
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// None in headless mode (golden tests, server rendering): frames
    /// render into `headless_target` instead of a swapchain.
    pub surface: Option<wgpu::Surface<'static>>,
    headless_target: Option<wgpu::Texture>,
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
    /// Composite-shader uniform — bloom intensity etc. Written each
    /// frame from the renderer's `bloom_intensity` field.
    pub composite_uniform_buffer: wgpu::Buffer,
    pub bloom_intensity: f32,
    /// Global wind field exposed to shaders via `frame.wind`. Layout:
    /// x = direction X, y = direction Z, z = amplitude (m), w =
    /// frequency (Hz). Foliage/cloth materials sample this in their
    /// vertex stage. Default (0,0,0,0) — no wind.
    pub wind: [f32; 4],
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
    /// Hi-Z occlusion culling (coarse max-depth grid + async readback);
    /// scene.prepare tests node AABBs against last frame's grid.
    pub occlusion: OcclusionCuller,
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
    /// Render-resolution multiplier in [0.5, 1.0]. The G-buffer,
    /// HDR, and composed RTs are sized to `surface * render_scale`;
    /// TAA (or the upscale pass) brings the output back up to the
    /// full surface for composite. At 0.5 = quarter-pixel shading
    /// (former TSR default). At 1.0 = native.
    pub render_scale: f32,
    /// Set once `set_render_scale` is called explicitly. While false,
    /// `set_taa_enabled` keeps the legacy coupling (TAA on = 0.5,
    /// TAA off = 1.0). Once the user opts into explicit control, the
    /// scale they picked sticks across subsequent TAA toggles.
    pub render_scale_explicit: bool,
    /// Previous frame's view-projection matrix — TAA reads this to
    /// reproject the history texture into current-frame UV space,
    /// removing ghosting under camera motion. Updated at the end
    /// of each frame from current_vp_matrix.
    pub prev_vp_matrix: [[f32; 4]; 4],
    /// EN-022 fix — previous frame's UNJITTERED projection + view,
    /// kept separately so `begin_mode_3d` can compose the velocity
    /// reference VP (prev unjittered proj + CURRENT jitter × prev
    /// view). Building prev_mvp from the raw jittered prev VP gave
    /// every static pixel a one-texel velocity wobble (the jitter
    /// delta), which cycled TAA history between converged and
    /// rejected — periodic sharp/soft flicker on detailed surfaces.
    prev_proj_matrix_unjittered: [[f32; 4]; 4],
    prev_view_matrix: [[f32; 4]; 4],
    /// Current frame's TAA jitter as NDC offsets (0,0 when TAA off).
    current_jitter_ndc: [f32; 2],
    /// The composed velocity-reference VP for this frame — what all
    /// prev_mvp compositions must use so jitter cancels in the
    /// shader's (curr_ndc - prev_ndc).
    pub(crate) velocity_ref_vp: [[f32; 4]; 4],
    /// Fog color (rgb) — blended into scene where fog factor > 0.
    pub fog_color: [f32; 3],
    /// EN-005 Phase 4 — `true` once the user has called
    /// `set_fog_color` explicitly. Suppresses procedural-sky's
    /// auto-derived sky-tinted fog so manual overrides stick.
    fog_color_user_override: bool,
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
    /// EN-005 Phase 3 — `true` once the user has called
    /// `set_sun_shaft_color` explicitly. Suppresses the procedural
    /// sky's auto-derived transmittance tint so manual artistic
    /// overrides stick.
    sun_shaft_color_user_override: bool,
    /// EN-005 Phase 3 — CPU copy of the transmittance LUT, kept
    /// alongside the GPU texture so renderer-side code can sample
    /// it for things like sun-shaft tint without a GPU readback.
    /// Sized identically to the GPU texture (`TRANSMITTANCE_W × _H`).
    transmittance_lut_cpu: Vec<u16>,
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
    /// EN-023 — mean flat albedo over the card instances; feeds the SW
    /// WSRC bake's ground-bounce term. Neutral mid-gray until the first
    /// instance-data upload computes the real scene average.
    pub gi_scene_avg_albedo: [f32; 3],
    /// One-shot log guard: which SSGI trace backend was last reported to
    /// stderr (hw-ray-query / sdf-clipmap / hiz-screen). The silent HW→SW
    /// fallback made "why is bounce gray?" a debugger question during the
    /// round-2 audit — keep it answerable from the log.
    pub ssgi_backend_logged: Option<&'static str>,

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
    /// Ticket 022 — staging buffers awaiting readback so freshly-baked
    /// per-mesh SDFs can be written to the disk cache. Populated by
    /// `bake_pending_sdfs` (one entry per dispatch); drained by
    /// `flush_sdf_cache_writes` after the frame's main submit, which
    /// maps each buffer, copies bytes to the cache file, and drops it.
    /// Empty on cache-hit frames and after cold launch finishes.
    sdf_cache_writes: Vec<(crate::sdf_cache::MeshHash, wgpu::Buffer)>,

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
    /// Fullscreen-lag fix — dedicated binned+sliced clipmap bake
    /// pipeline (the per-mesh `sdf_bake_pipeline` stays brute-force;
    /// its 32³ per-mesh volumes are already rate-limited).
    sdf_clipmap_bake_pipeline: wgpu::ComputePipeline,
    sdf_clipmap_bake_layout: wgpu::BindGroupLayout,
    scene_sdf_clipmap_staging_tex: wgpu::Texture,
    scene_sdf_clipmap_staging_view: wgpu::TextureView,
    /// Camera drifted past the rebake threshold — start a new bake job
    /// as soon as the previous one (if any) completes.
    scene_sdf_clipmap_rebake_needed: bool,
    sdf_clipmap_job: Option<SdfClipmapBakeJob>,

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

    /// Upscale pass — runs only when `render_scale < 1.0 && !taa_enabled`,
    /// reading the render-res `composed_rt` and writing the full-surface
    /// `upscale_rt` via bilinear or Catmull-Rom. When TAA is on the TAA
    /// pass handles upscaling itself and this pass is skipped.
    pub upscale_rt_texture: wgpu::Texture,
    pub upscale_rt_view: wgpu::TextureView,
    pub upscale_pipeline: wgpu::RenderPipeline,
    pub upscale_layout: wgpu::BindGroupLayout,
    pub upscale_uniform_buffer: wgpu::Buffer,
    /// 0 = bilinear (cheap), 1 = Catmull-Rom 5-tap (sharper edges).
    /// Default 1.
    pub upscale_mode: u32,

    /// Contrast-adaptive sharpen pass — 5-tap cross kernel. Gated on
    /// `cas_strength > 0`; default 0 = pass skipped. Reads whatever
    /// texture composite would have sampled (sss/mb/dof/taa/upscale/
    /// composed) and writes full-surface `cas_rt` that composite
    /// then consumes.
    pub cas_rt_texture: wgpu::Texture,
    pub cas_rt_view: wgpu::TextureView,
    pub cas_pipeline: wgpu::RenderPipeline,
    pub cas_layout: wgpu::BindGroupLayout,
    pub cas_uniform_buffer: wgpu::Buffer,
    /// 0 = off (default), 0.3 subtle, 0.6 punchy, 1.0 max.
    pub cas_strength: f32,

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

    // ----- EN-005 Phase 2: procedural sky -----
    // Toggled by `set_procedural_sky`; when true, the HDR pass calls
    // `render_procedural_sky_pass` instead of the panorama-based
    // `render_sky_pass`. The two paths are independent — the panorama
    // bind group/texture are untouched.
    procedural_sky_enabled: bool,
    sun_direction: [f32; 3],
    sun_intensity: f32,
    rayleigh_density: f32,
    mie_density: f32,
    ground_albedo: f32,
    sky_view_dirty: bool,
    _sky_view_lut_texture: wgpu::Texture,
    _sky_view_lut_view: wgpu::TextureView,
    sky_view_compute_pipeline: wgpu::ComputePipeline,
    _sky_view_compute_bgl: wgpu::BindGroupLayout,
    sky_view_uniform_buffer: wgpu::Buffer,
    sky_view_compute_bind_group: wgpu::BindGroup,
    procedural_sky_pipeline: wgpu::RenderPipeline,
    _procedural_sky_bgl: wgpu::BindGroupLayout,
    procedural_sun_uniform_buffer: wgpu::Buffer,
    procedural_sky_bind_group: wgpu::BindGroup,

    // EN-005 Phase 3 — IBL re-bake from procedural sky. The equirect
    // texture is the source for both the GGX-prefiltered specular mip
    // chain (which `lighting_bind_group` points at) and the diffuse
    // irradiance pass. Re-rendered every sun-move via
    // `bake_procedural_ibl`. Mip 0 holds the un-prefiltered sky
    // (sampled by anything that wants raw radiance); mips 1..N are
    // GGX-roughness-stratified for split-sum specular IBL.
    procedural_ibl_pipeline: wgpu::RenderPipeline,
    _procedural_ibl_bgl: wgpu::BindGroupLayout,
    procedural_ibl_uniform_buffer: wgpu::Buffer,
    procedural_ibl_bind_group: wgpu::BindGroup,
    procedural_sky_equirect_texture: wgpu::Texture,
    procedural_sky_equirect_full_view: wgpu::TextureView,
    procedural_sky_equirect_mip0_view: wgpu::TextureView,
    procedural_env_diffuse_texture: wgpu::Texture,
    procedural_env_diffuse_view: wgpu::TextureView,
    /// Bind group used by the GGX prefilter pipeline when re-baking
    /// procedural IBL — references the mip-0 view of
    /// `procedural_sky_equirect_texture` as its source.
    procedural_prefilter_bind_group: wgpu::BindGroup,
    /// Tracks whether `lighting_bind_group` currently points at
    /// procedural IBL textures (vs the panorama path). Avoids a
    /// rebuild on every sun-move once the swap is done.
    lighting_bg_is_procedural: bool,

    // EN-005 V2 — aerial perspective. Volume LUT recomputed each
    // frame (camera moves) when procedural sky is active. Replaces
    // scene_compose's 16-step volumetric fog march with a single
    // 3D-tex sample per fragment.
    aerial_perspective_pipeline: wgpu::ComputePipeline,
    _aerial_perspective_bgl: wgpu::BindGroupLayout,
    aerial_perspective_uniform_buffer: wgpu::Buffer,
    aerial_perspective_bind_group: wgpu::BindGroup,
    _aerial_perspective_texture: wgpu::Texture,
    pub aerial_perspective_view: wgpu::TextureView,
    pub aerial_perspective_sampler: wgpu::Sampler,
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
    /// Froxel light clustering (task #23). `Some` when the device has
    /// fragment-stage storage buffers (everything but WebGL2); the
    /// scene shader is then compiled with the clustered point-light
    /// loop and `lighting_layout` gains bindings 10–12. `None` keeps
    /// the plain count-driven loop.
    pub froxel: Option<froxel::FroxelPass>,
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
    /// EN-005 Phase 1 — Hillaire 2020 atmosphere LUTs, baked once
    /// at init from CPU code in `atmosphere_lut.rs`. Not yet sampled
    /// by any pass; Phase 2 wires them into the procedural sky pass.
    _atmosphere_transmittance_texture: wgpu::Texture,
    pub atmosphere_transmittance_view: wgpu::TextureView,
    _atmosphere_multi_scattering_texture: wgpu::Texture,
    pub atmosphere_multi_scattering_view: wgpu::TextureView,
    pub atmosphere_lut_sampler: wgpu::Sampler,
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

    /// EN-011 — Planar reflection probes, indexed by 1-based handle.
    /// Slots remain `None` after destroy so previously-issued handles
    /// never alias a future allocation. The probe RT is repainted
    /// each frame in `dispatch_planar_reflections` (called from
    /// `end_frame_with_scene` before the main material pass).
    pub planar_probes: Vec<Option<planar_reflection::PlanarReflectionProbe>>,
    /// Per-probe scratch UBO + bind group for the mirrored PerView.
    /// Parallel to `planar_probes` (1-based index minus one). Allocated
    /// alongside the probe in `create_planar_reflection`.
    pub planar_probe_view_buffers: Vec<Option<wgpu::Buffer>>,
    pub planar_probe_view_bgs:     Vec<Option<wgpu::BindGroup>>,
    /// EN-011 — lazily-built resources for rendering cached models (trees,
    /// house) into the planar probe with a mirrored VP. Single-target HDR
    /// pipeline + a dynamic per-draw model uniform + a sun/ambient uniform.
    /// `None` until the first `dispatch_planar_reflections` with a probe.
    pub reflect_scene_pipeline: Option<wgpu::RenderPipeline>,
    pub reflect_model_buf:      Option<wgpu::Buffer>,
    pub reflect_model_bg:       Option<wgpu::BindGroup>,
    pub reflect_light_buf:      Option<wgpu::Buffer>,
    pub reflect_light_bg:       Option<wgpu::BindGroup>,

    /// Phase 6 — hot-reload registry for file-backed materials. Each
    /// frame we drain pending file-change events and recompile any
    /// affected pipelines. Handles registered via
    /// `compile_material_from_file` participate; inline-string
    /// materials (compile_material) don't.
    pub material_hot_reload: hot_reload::MaterialHotReload,

    /// EN-017 V2 — game-supplied fullscreen post-pass STACK. Empty ⇒
    /// original zero-cost path. With N >= 1 entries, composite writes
    /// to `composite_ldr_rt_a`; pass[0] reads A and writes the next
    /// intermediate (or the swapchain if it's the last pass), and so
    /// on, ping-ponging between A and B. The 2D overlay still renders
    /// on top of the swapchain after the last pass.
    pub post_passes: Vec<post_pass::PostPassPipeline>,
    /// LDR intermediate "slot A" the composite pass redirects into
    /// when at least one post-pass is installed. Allocated lazily on
    /// first `add_post_pass` call and resized in `resize`. Format
    /// mirrors `surface_config.format` so post-pass shaders see
    /// identical bits regardless of whether the stack is active.
    pub composite_ldr_rt_a: Option<wgpu::Texture>,
    pub composite_ldr_rt_a_view: Option<wgpu::TextureView>,
    /// LDR intermediate "slot B" — the second ping-pong target.
    /// Allocated lazily on first `add_post_pass` call where the stack
    /// length grows to >= 2 (single-pass setups never need slot B).
    pub composite_ldr_rt_b: Option<wgpu::Texture>,
    pub composite_ldr_rt_b_view: Option<wgpu::TextureView>,
    /// NonFiltering sampler used for the depth binding in post-pass
    /// shaders. Created once and reused.
    pub post_pass_depth_sampler: wgpu::Sampler,
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
        Self::new_impl(device, queue, Some(surface), surface_config, logical_width, logical_height)
    }

    /// Headless renderer: no swapchain; frames render into an offscreen
    /// texture readable via the screenshot path. Used by the golden-image
    /// tests and available for server-side rendering.
    pub fn new_headless(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Self {
        let surface_config = wgpu::SurfaceConfiguration {
            // COPY_SRC: bloom_take_screenshot reads the swapchain back;
            // without it the readback copy is a swallowed validation
            // error and screenshots silently produce nothing.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        Self::new_impl(device, queue, None, surface_config, width, height)
    }

    fn new_impl(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: Option<wgpu::Surface<'static>>,
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
        // Froxel clustering first — its presence decides whether the
        // lighting layout grows bindings 10-12 and which point-light
        // loop the scene shader is compiled with.
        let froxel = froxel::FroxelPass::supported(&device)
            .then(|| froxel::FroxelPass::new(&device));

        let lighting_layout = lighting::create_lighting_layout(&device, froxel.is_some());
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

        // --- EN-005 Phase 1 — atmosphere LUTs (Hillaire 2020) ---
        // Two CPU-baked tables for the procedural sky: transmittance
        // (per-channel survival fraction along view rays through the
        // atmosphere) and multi-scattering (energy-conserving second-
        // and-higher-order scattering). Sizes are platform-tiered —
        // see atmosphere_lut.rs for the cfg constants. Phase 2 wires
        // these into the new sky pass; for now they only exist on the
        // Renderer struct so the bake pipeline is exercised in CI.
        let transmittance_pixels = build_transmittance_lut(TRANSMITTANCE_W, TRANSMITTANCE_H);
        let atmosphere_transmittance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atmosphere_transmittance_lut"),
            size: wgpu::Extent3d {
                width: TRANSMITTANCE_W,
                height: TRANSMITTANCE_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atmosphere_transmittance_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&transmittance_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TRANSMITTANCE_W * 8), // 4 channels × 2 bytes
                rows_per_image: Some(TRANSMITTANCE_H),
            },
            wgpu::Extent3d {
                width: TRANSMITTANCE_W,
                height: TRANSMITTANCE_H,
                depth_or_array_layers: 1,
            },
        );
        let atmosphere_transmittance_view =
            atmosphere_transmittance_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let multi_scattering_pixels = build_multi_scattering_lut(
            &transmittance_pixels,
            TRANSMITTANCE_W,
            TRANSMITTANCE_H,
            MULTI_SCATTERING_SIZE,
        );
        let atmosphere_multi_scattering_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atmosphere_multi_scattering_lut"),
            size: wgpu::Extent3d {
                width: MULTI_SCATTERING_SIZE,
                height: MULTI_SCATTERING_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atmosphere_multi_scattering_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&multi_scattering_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(MULTI_SCATTERING_SIZE * 8),
                rows_per_image: Some(MULTI_SCATTERING_SIZE),
            },
            wgpu::Extent3d {
                width: MULTI_SCATTERING_SIZE,
                height: MULTI_SCATTERING_SIZE,
                depth_or_array_layers: 1,
            },
        );
        let atmosphere_multi_scattering_view = atmosphere_multi_scattering_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Linear, clamp-to-edge sampler shared between both LUTs —
        // same shape as the BRDF LUT sampler since both are pre-
        // integrated 2D tables with no mip levels.
        let atmosphere_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmosphere_lut_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Joint-matrix bind group layout for GPU skinning. Created here (ahead
        // of its buffer/bind group below) because the shadow map's skinned
        // caster pipeline needs this layout at construction time.
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

        // Shadow map needs to be created before the lighting bind
        // group since the bind group binds the shadow depth view.
        let shadow_map = crate::shadows::ShadowMap::new(&device, Vertex3D::desc(), &joint_layout);

        let lighting_bind_group = lighting::create_lighting_bind_group(
            &device,
            &lighting_layout,
            "lighting_bg",
            &lighting::LightingBindSources {
                lighting_buffer: &lighting_buffer,
                env_sampler: &env_sampler,
                brdf_lut_view: &brdf_lut_view,
                brdf_lut_sampler: &brdf_lut_sampler,
                shadow_map: &shadow_map,
                froxel: froxel.as_ref(),
            },
            &scene_env_default_view,
            &scene_env_default_view,
        );

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
        // (`joint_layout` is created earlier, above the shadow map, because the
        // skinned shadow pipeline needs it at construction time.)
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
        // EN-005 Phase 2 — procedural sky pipeline + sky-view LUT
        // compute pipeline. Both built unconditionally at init so
        // toggling `set_procedural_sky(true)` later doesn't pay any
        // shader-compile cost. Memory cost is small: ~150 KB for the
        // sky-view LUT at desktop sizes, ~3 KB compiled shader.
        // ============================================================

        // Sky-view LUT storage texture — receives the output of the
        // compute shader on every sun move. We allocate it before the
        // compute pipeline so the bind group can hold a view directly.
        let sky_view_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky_view_lut"),
            size: wgpu::Extent3d {
                width: SKY_VIEW_W,
                height: SKY_VIEW_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sky_view_lut_view =
            sky_view_lut_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sky-view LUT compute pipeline.
        let sky_view_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_view_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(SKY_VIEW_LUT_SHADER_WGSL.into()),
        });
        let sky_view_compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_view_compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let sky_view_compute_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_view_compute_pl"),
            bind_group_layouts: &[Some(&sky_view_compute_bgl)],
            immediate_size: 0,
        });
        let sky_view_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sky_view_compute_pipeline"),
            layout: Some(&sky_view_compute_pl_layout),
            module: &sky_view_compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sky_view_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_view_uniform_buffer"),
            size: std::mem::size_of::<SkyViewParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_view_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_view_compute_bg"),
            layout: &sky_view_compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sky_view_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atmosphere_transmittance_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&atmosphere_multi_scattering_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&atmosphere_lut_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&sky_view_lut_view) },
            ],
        });

        // Procedural sky render pipeline (mirror of `sky_pipeline` shape).
        let procedural_sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("procedural_sky_shader"),
            source: wgpu::ShaderSource::Wgsl(PROCEDURAL_SKY_SHADER_WGSL.into()),
        });
        let procedural_sun_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("procedural_sun_uniform_buffer"),
            size: std::mem::size_of::<SunUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let procedural_sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("procedural_sky_bgl"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
        let procedural_sky_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("procedural_sky_pl"),
            bind_group_layouts: &[Some(&procedural_sky_bgl)],
            immediate_size: 0,
        });
        let procedural_sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("procedural_sky_pipeline"),
            layout: Some(&procedural_sky_pl_layout),
            vertex: wgpu::VertexState {
                module: &procedural_sky_shader,
                entry_point: Some("sky_vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &procedural_sky_shader,
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
        let procedural_sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural_sky_bg"),
            layout: &procedural_sky_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sky_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sky_view_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&atmosphere_lut_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: procedural_sun_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&atmosphere_transmittance_view) },
            ],
        });

        // ============================================================
        // EN-005 Phase 3 — procedural IBL. Allocate the destination
        // textures + the sky-view → equirect re-projection pipeline.
        // The textures are dimensionally fixed across platforms (IBL
        // is low-frequency; 256×128 + 128×64 is plenty), but the
        // re-render cost is gated by the sky-view dirty flag, so
        // static suns pay this exactly once.
        // ============================================================

        const PROC_IBL_W: u32 = 256;
        const PROC_IBL_H: u32 = 128;
        const PROC_IBL_DIFFUSE_W: u32 = 128;
        const PROC_IBL_DIFFUSE_H: u32 = 64;
        let proc_ibl_mip_count = (PROC_IBL_W.max(PROC_IBL_H) as f32).log2().floor() as u32 + 1;

        let procedural_sky_equirect_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("procedural_sky_equirect"),
            size: wgpu::Extent3d {
                width: PROC_IBL_W,
                height: PROC_IBL_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: proc_ibl_mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let procedural_sky_equirect_full_view =
            procedural_sky_equirect_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let procedural_sky_equirect_mip0_view =
            procedural_sky_equirect_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("procedural_sky_equirect_mip0"),
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });

        let procedural_env_diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("procedural_env_diffuse"),
            size: wgpu::Extent3d {
                width: PROC_IBL_DIFFUSE_W,
                height: PROC_IBL_DIFFUSE_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let procedural_env_diffuse_view =
            procedural_env_diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Equirect re-projection pipeline (sky-view LUT → equirect).
        let procedural_ibl_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("procedural_ibl_shader"),
            source: wgpu::ShaderSource::Wgsl(EQUIRECT_FROM_SKY_VIEW_SHADER_WGSL.into()),
        });
        let procedural_ibl_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("procedural_ibl_uniform_buffer"),
            size: 16, // BakeParams = vec4<f32> dims
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let procedural_ibl_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("procedural_ibl_bgl"),
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
        let procedural_ibl_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("procedural_ibl_pl"),
            bind_group_layouts: &[Some(&procedural_ibl_bgl)],
            immediate_size: 0,
        });
        let procedural_ibl_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("procedural_ibl_pipeline"),
            layout: Some(&procedural_ibl_pl_layout),
            vertex: wgpu::VertexState {
                module: &procedural_ibl_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &procedural_ibl_shader,
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let procedural_ibl_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural_ibl_bg"),
            layout: &procedural_ibl_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: procedural_ibl_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sky_view_lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&atmosphere_lut_sampler) },
            ],
        });

        // The GGX prefilter source bind group — see the
        // `procedural_prefilter_bind_group = ...` definition below
        // (after the prefilter pipeline + layout are created). We
        // can't build it here because `prefilter_layout` and
        // `prefilter_uniform_buffer` haven't been declared yet.

        // ============================================================
        // Scene pipeline (retained scene-graph draws with normal maps)
        // ============================================================
        // Clustered devices get the froxel point-light loop spliced in
        // place of the plain reference loop (same shading math — the
        // many_point_lights golden enforces equivalence).
        let scene_shader_source: std::borrow::Cow<'static, str> = if froxel.is_some() {
            froxel::clustered_scene_shader(SCENE_SHADER).into()
        } else {
            SCENE_SHADER.into()
        };
        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_shader"),
            source: wgpu::ShaderSource::Wgsl(scene_shader_source),
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
                    // VERTEX_FRAGMENT: the vertex stage reads metal_rough.w
                    // (alpha-cutoff) to gate foliage wind sway; fragment reads
                    // the full MaterialFactors.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

        // EN-005 Phase 3 — bind group for the GGX prefilter when
        // re-baking procedural IBL. Sources from the mip-0 view of
        // the procedural sky equirect (re-rendered each sun-move).
        // Same shape as the temporary panorama-path bind group built
        // inside `load_env_from_hdr`, but persistent so we don't
        // re-allocate on every sun move.
        let procedural_prefilter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural_prefilter_src_bg"),
            layout: &prefilter_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: prefilter_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&procedural_sky_equirect_mip0_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&env_sampler) },
            ],
        });

        // ============================================================
        // EN-005 V2 — aerial-perspective 3D LUT. Compute pipeline +
        // 3D storage texture. Re-baked each frame from
        // `dispatch_aerial_perspective_lut` when procedural sky is on.
        // ============================================================
        let aerial_perspective_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aerial_perspective_lut"),
            size: wgpu::Extent3d {
                width: AERIAL_W,
                height: AERIAL_H,
                depth_or_array_layers: AERIAL_D,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let aerial_perspective_view =
            aerial_perspective_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("aerial_perspective_view"),
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            });
        let aerial_perspective_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aerial_perspective_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let aerial_perspective_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aerial_perspective_shader"),
            source: wgpu::ShaderSource::Wgsl(AERIAL_PERSPECTIVE_SHADER_WGSL.into()),
        });
        let aerial_perspective_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aerial_perspective_uniform_buffer"),
            size: std::mem::size_of::<AerialParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let aerial_perspective_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aerial_perspective_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let aerial_perspective_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aerial_perspective_pl"),
            bind_group_layouts: &[Some(&aerial_perspective_bgl)],
            immediate_size: 0,
        });
        let aerial_perspective_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aerial_perspective_pipeline"),
            layout: Some(&aerial_perspective_pl_layout),
            module: &aerial_perspective_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let aerial_perspective_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aerial_perspective_bg"),
            layout: &aerial_perspective_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: aerial_perspective_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atmosphere_transmittance_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&atmosphere_multi_scattering_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&atmosphere_lut_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&aerial_perspective_view) },
            ],
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
        // NOTE: bloom params are per-pass create_buffer_init uniforms now
        // (postfx_chain.rs) — a single shared buffer written once per pass
        // aliased to the last write at submit.

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
        let occlusion = OcclusionCuller::new(&device);
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

        // --- Upscale pipeline (render-res composed_rt → full-surface
        // upscale_rt). Engages when render_scale < 1.0 && !taa_enabled.
        // 3-binding layout: uniform(mode) + input tex + filtering sampler.
        let upscale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upscale_shader"),
            source: wgpu::ShaderSource::Wgsl(UPSCALE_SHADER_WGSL.into()),
        });
        let upscale_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("upscale_layout"),
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
            ],
        });
        let upscale_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upscale_pl_layout"),
            bind_group_layouts: &[Some(&upscale_layout)],
            immediate_size: 0,
        });
        let upscale_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("upscale_pipeline"),
            layout: Some(&upscale_pl_layout),
            vertex: wgpu::VertexState {
                module: &upscale_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &upscale_shader, entry_point: Some("fs_main"),
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
        let upscale_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upscale_uniform_buffer"),
            size: std::mem::size_of::<UpscaleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (upscale_rt_texture, upscale_rt_view) = create_dof_rt(
            &device, surface_config.width, surface_config.height,
        );

        // --- RCAS sharpen pipeline. Same 3-binding layout as upscale
        // (uniform + input tex + sampler), same HDR output format —
        // runs at full surface res sampling whatever texture composite
        // would otherwise read. Gated on cas_strength > 0 at frame time.
        let cas_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cas_shader"),
            source: wgpu::ShaderSource::Wgsl(RCAS_SHADER_WGSL.into()),
        });
        let cas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cas_layout"),
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
            ],
        });
        let cas_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cas_pl_layout"),
            bind_group_layouts: &[Some(&cas_layout)],
            immediate_size: 0,
        });
        let cas_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cas_pipeline"),
            layout: Some(&cas_pl_layout),
            vertex: wgpu::VertexState {
                module: &cas_shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cas_shader, entry_point: Some("fs_main"),
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
        let cas_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cas_uniform_buffer"),
            size: std::mem::size_of::<RcasParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (cas_rt_texture, cas_rt_view) = create_dof_rt(
            &device, surface_config.width, surface_config.height,
        );

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
                // EN-021 — env panorama for the miss fallback.
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

        // Fullscreen-lag fix — binned + sliced clipmap bake pipeline.
        // Same first four bindings as sdf_bake_layout, plus the two
        // triangle-bin buffers (cell offsets + per-cell tri indices).
        let sdf_clipmap_bake_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_clipmap_bake_shader"),
            source: wgpu::ShaderSource::Wgsl(SDF_CLIPMAP_BAKE_WGSL.into()),
        });
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sdf_clipmap_bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_clipmap_bake_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                storage_ro(1),
                storage_ro(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    }, count: None,
                },
                storage_ro(4),
                storage_ro(5),
            ],
        });
        let sdf_clipmap_bake_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_clipmap_bake_pl_layout"),
            bind_group_layouts: &[Some(&sdf_clipmap_bake_layout)],
            immediate_size: 0,
        });
        let sdf_clipmap_bake_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf_clipmap_bake_pipeline"),
            layout: Some(&sdf_clipmap_bake_pl_layout),
            module: &sdf_clipmap_bake_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Ticket 014 V2: scene clipmap ---
        let (scene_sdf_clipmap_tex, scene_sdf_clipmap_view) =
            create_scene_sdf_clipmap(&device);
        let (scene_sdf_clipmap_staging_tex, scene_sdf_clipmap_staging_view) =
            create_scene_sdf_clipmap_staging(&device);

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
                // EN-005 V2 — aerial-perspective 3D LUT (tex + sampler).
                // Sampled per fragment when `misc.y > 0` (procedural sky
                // active); otherwise unused but bound to keep the layout
                // shape stable across frames.
                wgpu::BindGroupLayoutEntry {
                    binding: 13, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                    // Rg16Float: .g carries the anchored AE target
                    // (flicker fix — see EXPOSURE_SHADER_WGSL).
                    format: wgpu::TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
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

        // EN-017 — non-filtering sampler shared by every post-pass
        // shader for the depth binding. Created up front because the
        // sampler outlives any individual post-pass pipeline.
        let post_pass_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post_pass_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            headless_target: if surface.is_none() {
                Some(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("headless_swapchain"),
                    size: wgpu::Extent3d {
                        width: surface_config.width,
                        height: surface_config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: surface_config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }))
            } else {
                None
            },
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
            // Was 0.8 — that haloed and, under half-res TSR, made the
            // silhouette edge shimmer read as crawling gray lines. 0.5
            // keeps the crispness that covers TSR softness; the detail
            // clamp in COMPOSITE_SHADER_WGSL bounds the edge overshoot.
            sharpen_strength: 0.5,
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
            bloom_intensity: 0.04,
            wind: [0.0, 0.0, 0.0, 0.0],
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
            occlusion,
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
            render_scale: 0.5,
            render_scale_explicit: false,
            prev_vp_matrix: IDENTITY_MAT4,
            prev_proj_matrix_unjittered: IDENTITY_MAT4,
            prev_view_matrix: IDENTITY_MAT4,
            current_jitter_ndc: [0.0, 0.0],
            velocity_ref_vp: IDENTITY_MAT4,
            fog_color: [0.7, 0.75, 0.82],
            fog_color_user_override: false,
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
            ssgi_backend_logged: None,
            gi_scene_avg_albedo: [0.35, 0.35, 0.35],
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
            sdf_cache_writes: Vec::new(),
            scene_sdf_clipmap_tex,
            scene_sdf_clipmap_view,
            scene_sdf_clipmap_built: false,
            scene_sdf_clipmap_origin: [0.0, 0.0, 0.0],
            sdf_clipmap_bake_pipeline,
            sdf_clipmap_bake_layout,
            scene_sdf_clipmap_staging_tex,
            scene_sdf_clipmap_staging_view,
            scene_sdf_clipmap_rebake_needed: true,
            sdf_clipmap_job: None,
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
            upscale_rt_texture,
            upscale_rt_view,
            upscale_pipeline,
            upscale_layout,
            upscale_uniform_buffer,
            upscale_mode: 1, // Catmull-Rom by default
            cas_rt_texture,
            cas_rt_view,
            cas_pipeline,
            cas_layout,
            cas_uniform_buffer,
            cas_strength: 0.0, // off by default
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
            procedural_sky_enabled: false,
            sun_direction: [0.0, 1.0, 0.0],
            sun_intensity: 1.0,
            rayleigh_density: 1.0,
            mie_density: 1.0,
            ground_albedo: 0.1,
            sky_view_dirty: true,
            _sky_view_lut_texture: sky_view_lut_texture,
            _sky_view_lut_view: sky_view_lut_view,
            sky_view_compute_pipeline,
            _sky_view_compute_bgl: sky_view_compute_bgl,
            sky_view_uniform_buffer,
            sky_view_compute_bind_group,
            procedural_sky_pipeline,
            _procedural_sky_bgl: procedural_sky_bgl,
            procedural_sun_uniform_buffer,
            procedural_sky_bind_group,
            procedural_ibl_pipeline,
            _procedural_ibl_bgl: procedural_ibl_bgl,
            procedural_ibl_uniform_buffer,
            procedural_ibl_bind_group,
            procedural_sky_equirect_texture,
            procedural_sky_equirect_full_view,
            procedural_sky_equirect_mip0_view,
            procedural_env_diffuse_texture,
            procedural_env_diffuse_view,
            procedural_prefilter_bind_group,
            lighting_bg_is_procedural: false,
            transmittance_lut_cpu: transmittance_pixels,
            sun_shaft_color_user_override: false,
            aerial_perspective_pipeline,
            _aerial_perspective_bgl: aerial_perspective_bgl,
            aerial_perspective_uniform_buffer,
            aerial_perspective_bind_group,
            _aerial_perspective_texture: aerial_perspective_texture,
            aerial_perspective_view,
            aerial_perspective_sampler,
            env_diffuse_texture: None,
            scene_pipeline,
            froxel,
            scene_material_layout,
            _scene_env_default_texture: scene_env_default_texture,
            scene_env_default_view,
            env_sampler,
            lighting_layout,
            _brdf_lut_texture: brdf_lut_texture,
            brdf_lut_view,
            brdf_lut_sampler,
            _atmosphere_transmittance_texture: atmosphere_transmittance_texture,
            atmosphere_transmittance_view,
            _atmosphere_multi_scattering_texture: atmosphere_multi_scattering_texture,
            atmosphere_multi_scattering_view,
            atmosphere_lut_sampler,
            prefilter_pipeline,
            prefilter_diffuse_pipeline,
            prefilter_layout,
            prefilter_uniform_buffer,
            _default_normal_texture: default_normal_tex,
            default_normal_view,
            material_system,
            transient_pool,
            impulse_field,
            planar_probes: Vec::new(),
            reflect_scene_pipeline: None,
            reflect_model_buf: None,
            reflect_model_bg: None,
            reflect_light_buf: None,
            reflect_light_bg: None,
            planar_probe_view_buffers: Vec::new(),
            planar_probe_view_bgs: Vec::new(),
            material_hot_reload,
            post_passes: Vec::new(),
            composite_ldr_rt_a: None,
            composite_ldr_rt_a_view: None,
            composite_ldr_rt_b: None,
            composite_ldr_rt_b_view: None,
            post_pass_depth_sampler,
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
    /// Render-pass extent used by main_hdr + scene_compose, computed
    /// from `render_scale`. At 0.5 this is half the surface size and
    /// the TAA pass (or upscale pass) brings it back to the full
    /// surface for composite; at 1.0 it matches the surface.
    pub fn render_extent(&self) -> (u32, u32) {
        let sw = self.surface_config.width as f32;
        let sh = self.surface_config.height as f32;
        let s = self.render_scale.clamp(0.5, 1.0);
        (
            ((sw * s).round() as u32).max(1),
            ((sh * s).round() as u32).max(1),
        )
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
            match &self.surface {
                Some(surface) => surface.configure(&self.device, &self.surface_config),
                None => {
                    // Recreate the offscreen target at the new size.
                    self.headless_target = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("headless_swapchain"),
                        size: wgpu::Extent3d {
                            width: self.surface_config.width,
                            height: self.surface_config.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: self.surface_config.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    }));
                }
            }

            // Render-resolution RTs (G-buffer + composed), sized to
            // `surface * render_scale`. TAA (or the upscale pass)
            // brings the composed_rt back to full surface; the rest
            // of the post-FX chain (DoF/MB/SSS) and composite stay
            // at full surface.
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
            // The screen-space stack (bloom, GTAO, HiZ pyramid, SSR, SSGI,
            // probe grid) reads render-resolution inputs (depth/normals/HDR)
            // and is consumed by UV sampling, so it lives at RENDER
            // resolution too. Building it surface-sized (the old behaviour)
            // made GTAO/bloom cost the same at render_scale 0.5 as at 1.0 —
            // ~4 ms/frame of pure waste at 4K output.
            let (bt, bm, bf) = create_bloom_chain(&self.device, rw, rh, BLOOM_MIP_COUNT);
            self.bloom_chain_textures = bt;
            self.bloom_mip_views = bm;
            self.bloom_full_view = bf;
            let (st, sv) = create_ssao_rt(&self.device, rw, rh);
            self.ssao_rt_texture = st;
            self.ssao_rt_view = sv;
            let (sht, shv) = create_ssao_history_textures(&self.device, rw, rh);
            self.ssao_history_textures = sht;
            self.ssao_history_views = shv;
            self.ssao_history_idx = 0;
            self.ssao_history_frame = 0;
            let (sbt, sbv) = create_ssao_blur_rt(&self.device, rw, rh);
            self.ssao_blur_rt_texture = sbt;
            self.ssao_blur_rt_view = sbv;
            let (hiz_t, hiz_v) = create_linear_depth_hiz_chain(&self.device, rw, rh);
            self.hiz_textures = hiz_t;
            self.hiz_views = hiz_v;
            // TAA history/output live at SURFACE size — the TAA pass is
            // also the TSR upscaler (render res in, output res out).
            let (taa_t, taa_v) = create_taa_textures(&self.device, width, height);
            self.taa_textures = taa_t;
            self.taa_views = taa_v;
            self.taa_frame_index = 0; // reset jitter sequence on resize
            let (sr_t, sr_v) = create_ssr_rt(&self.device, rw, rh);
            self.ssr_rt_texture = sr_t;
            self.ssr_rt_view = sr_v;
            let (ssr_ht, ssr_hv) = create_ssr_history_textures(&self.device, rw, rh);
            self.ssr_history_textures = ssr_ht;
            self.ssr_history_views = ssr_hv;
            self.ssr_history_idx = 0;
            let (ssgi_t, ssgi_v) = create_ssgi_rt(&self.device, rw, rh);
            self.ssgi_rt_texture = ssgi_t;
            self.ssgi_rt_view = ssgi_v;
            // Ticket 007a: rebuild the probe grid + 3D radiance textures
            // whenever the render size changes. Probe count scales with
            // half-res resolution, so the header buffer is resized too.
            let (pg_w, pg_h) = probe_grid_dims(rw, rh);
            self.probe_grid_w = pg_w;
            self.probe_grid_h = pg_h;
            let (ptr, pvr) = create_probe_trace_tex(&self.device, rw, rh);
            self.probe_trace_tex = ptr;
            self.probe_trace_view = pvr;
            let (pht, phv) = create_probe_history_textures(&self.device, rw, rh);
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
            let (up_t, up_v) = create_dof_rt(&self.device, width, height);
            self.upscale_rt_texture = up_t;
            self.upscale_rt_view = up_v;
            let (cas_t, cas_v) = create_dof_rt(&self.device, width, height);
            self.cas_rt_texture = cas_t;
            self.cas_rt_view = cas_v;

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
        self.occlusion.invalidate_bindings();
            for slot in self.hiz_downsample_bg_cache.iter_mut() {
                *slot = None;
            }

            // EN-017 V2 — keep both LDR ping-pong intermediates in
            // lockstep with the swapchain. Each slot is allocated
            // lazily (A on first add, B on the second), so titles
            // without a post-pass pay zero memory and titles with one
            // pass only pay for slot A.
            if self.composite_ldr_rt_a.is_some() {
                let (t, v) = post_pass::create_composite_ldr_rt(
                    &self.device, width, height, self.surface_config.format,
                );
                self.composite_ldr_rt_a = Some(t);
                self.composite_ldr_rt_a_view = Some(v);
            }
            if self.composite_ldr_rt_b.is_some() {
                let (t, v) = post_pass::create_composite_ldr_rt(
                    &self.device, width, height, self.surface_config.format,
                );
                self.composite_ldr_rt_b = Some(t);
                self.composite_ldr_rt_b_view = Some(v);
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

    // EN-012 V2 — SSAO backface half-strength on two-sided foliage:
    // status = DEFERRED, requires ABI v3 RFC.
    //
    // The ticket asks for "half-strength SSAO on backfaces of two-sided
    // foliage" so leaves don't go pure black when the camera sees the
    // unlit side. The clean fix needs per-fragment data that doesn't
    // exist in the G-buffer today:
    //   OpaqueOut writes hdr (loc 0), material vec2 (loc 1), velocity
    //   vec2 (loc 2), albedo vec4 (loc 3). None carries an
    //   isFrontFace bit or a shading_model id that the SSAO compute
    //   could read.
    //
    // Three viable design paths (V3 RFC needs to pick one):
    //
    // 1. Add @location(4) flags: u32 to OpaqueOut. ABI v3 bump. Every
    //    opaque material shader (built-in + game) re-emits to write
    //    flags = (is_foliage as u32) | (is_subsurface as u32 << 1).
    //    SSAO compute samples flags, halves on bit 0.
    //    Cost: 1 extra G-buffer attachment (~rgba8 = 4 bpp at 1080p =
    //    ~8 MB). Cleanest data model.
    //
    // 2. Steal the unused alpha channel of `albedo` (loc 3) — the
    //    "for SSGI bounce colour" channel comment confirms it's
    //    currently unused. Pack shading_model into albedo.a as
    //    f32 (0.0=lit, 0.5=foliage, 1.0=subsurface). No ABI bump.
    //    Cost: locks albedo.a forever. Pragmatic.
    //
    // 3. Stencil-based: opaque foliage materials get a stencil-write
    //    pipeline state (currently unused in the engine). SSAO
    //    compute samples the depth-stencil texture and reads the
    //    stencil bit. No G-buffer change. Cost: every material
    //    pipeline now has stencil state; SSAO needs a depth-stencil
    //    sample binding.
    //
    // Workaround until V3 lands: games with heavy foliage can call
    // `set_ssao_strength(0.5)` globally before drawing foliage-heavy
    // scenes. Coarse but works.
    //
    // Tracking: unblocked once the engine commits to a G-buffer ABI
    // v3 (which would also let us land #2's `flags` channel for
    // motion-blur masking, decals, and other "per-pixel material
    // intent" use cases that keep coming up).

    /// Set the global wind field exposed to shaders via `frame.wind`.
    /// Used by foliage / cloth materials. Direction is (dir_x, dir_z)
    /// in the XZ plane and need not be normalised — the shader-side
    /// magnitude scales effective amplitude. `amplitude` is the
    /// displacement scale (~0.1 m for grass), `frequency` is in Hz.
    pub fn set_wind(&mut self, dir_x: f32, dir_z: f32, amplitude: f32, frequency: f32) {
        self.wind = [dir_x, dir_z, amplitude, frequency];
    }

    /// Toggle TAA on/off. Off = no jitter, no history blend, no
    /// extra texture writes. Until `set_render_scale` is called
    /// explicitly this also flips `render_scale` between 0.5 (on)
    /// and 1.0 (off) for backwards compat with the former TSR
    /// coupling; once the user sets scale explicitly that choice
    /// sticks across subsequent TAA toggles.
    pub fn set_taa_enabled(&mut self, enabled: bool) {
        if enabled != self.taa_enabled {
            self.taa_enabled = enabled;
            if !self.render_scale_explicit {
                self.render_scale = if enabled { 0.5 } else { 1.0 };
            }
            self.taa_frame_index = 0;
            let (w, h) = (self.surface_config.width, self.surface_config.height);
            self.resize(w, h, self.logical_width, self.logical_height);
        }
    }

    /// Set the render-resolution multiplier explicitly. Clamped to
    /// [0.5, 1.0]. Triggers a resize so render-res intermediates
    /// pick up the new extent. Marks the scale as user-set so future
    /// `set_taa_enabled` calls leave it alone.
    pub fn set_render_scale(&mut self, scale: f32) {
        let s = scale.clamp(0.5, 1.0);
        self.render_scale_explicit = true;
        if (s - self.render_scale).abs() > 1e-4 {
            self.render_scale = s;
            self.taa_frame_index = 0;
            let (w, h) = (self.surface_config.width, self.surface_config.height);
            self.resize(w, h, self.logical_width, self.logical_height);
        }
    }

    /// Upscale filter when `render_scale < 1.0 && !taa_enabled`.
    /// 0 = bilinear (cheap, soft), 1 = Catmull-Rom (sharper, default).
    pub fn set_upscale_mode(&mut self, mode: u32) {
        self.upscale_mode = if mode > 1 { 1 } else { mode };
    }

    /// CAS sharpen strength. 0 = off (default, pass skipped),
    /// 0.3 = subtle, 0.6 = punchy, 1.0 = max. Clamped to [0, 1].
    pub fn set_cas_strength(&mut self, strength: f32) {
        self.cas_strength = strength.clamp(0.0, 1.0);
    }

    /// Current render-resolution multiplier in [0.5, 1.0].
    pub fn render_scale(&self) -> f32 { self.render_scale }

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


    /// Ticket 022 — drain pending SDF cache writes after the frame's
    /// main submit. Maps each staging buffer in one pass (single
    /// `device.poll(Wait)` covers all of them), unpads the row-aligned
    /// payload back to the tightly-packed on-disk layout, and writes
    /// to the cache. Best-effort throughout: a write failure is
    /// silently ignored — the next cold launch just rebakes.
    pub fn flush_sdf_cache_writes(&mut self) {
        if self.sdf_cache_writes.is_empty() { return; }
        let entries = std::mem::take(&mut self.sdf_cache_writes);

        // Issue map_async on every buffer up front so a single poll
        // resolves all of them — much cheaper than serially polling
        // per buffer when 8 cold-launch bakes complete in one frame.
        for (_, buf) in &entries {
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| { /* polled below */ });
        }
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let row_tight = (MESH_SDF_RES * 4) as usize;
        let row_padded = ((MESH_SDF_RES * 4 + 255) & !255) as usize;
        let res = MESH_SDF_RES as usize;

        for (hash, buf) in entries {
            let slice = buf.slice(..);
            let data = slice.get_mapped_range();
            // Strip the wgpu-required row padding back to a tight
            // 32³ × 4-byte payload before storing.
            let mut tight = vec![0u8; res * res * row_tight];
            for z in 0..res {
                for y in 0..res {
                    let src_off = (z * res + y) * row_padded;
                    let dst_off = (z * res + y) * row_tight;
                    tight[dst_off..dst_off + row_tight]
                        .copy_from_slice(&data[src_off..src_off + row_tight]);
                }
            }
            drop(data);
            buf.unmap();
            let _ = crate::sdf_cache::store(hash, &tight);
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
        // EN-023 — running mean of instance albedos feeds the SW WSRC
        // bake's ground-bounce term.
        let mut albedo_sum = [0.0f32; 3];
        for &h in instance_handles {
            let n = scene.nodes.get(h).unwrap();
            let e = n.material.emissive;
            let (first_slot, has_card) = match n.card_first_slot {
                Some(s) => (s as f32, 1.0_f32),
                None => (0.0, 0.0),
            };
            // EN-023 — world AABB for the SDF broad-phase. The scene's
            // bounds pass keeps world_bounds fresh; the sentinel
            // (min.x > max.x = not yet computed) falls back to the local
            // box, which matches the old behaviour for identity nodes.
            let (wmin, wmax) = if n.world_bounds_min[0] <= n.world_bounds_max[0] {
                (n.world_bounds_min, n.world_bounds_max)
            } else {
                (n.bounds_min, n.bounds_max)
            };
            albedo_sum[0] += n.flat_albedo[0];
            albedo_sum[1] += n.flat_albedo[1];
            albedo_sum[2] += n.flat_albedo[2];
            instance_data.push(InstanceGiDataCpu {
                albedo: n.flat_albedo,
                emissive_luma: (e[0] + e[1] + e[2]) * (1.0 / 3.0),
                normal_ws: n.flat_normal_ws,
                _pad0: 0.0,
                card_slot: [first_slot, 0.0, 0.0, has_card],
                card_aabb_min: [n.bounds_min[0], n.bounds_min[1], n.bounds_min[2], 0.0],
                card_aabb_max: [n.bounds_max[0], n.bounds_max[1], n.bounds_max[2], 0.0],
                world_aabb_min: [wmin[0], wmin[1], wmin[2], 0.0],
                world_aabb_max: [wmax[0], wmax[1], wmax[2], 0.0],
            });
        }
        if !instance_data.is_empty() {
            let inv = 1.0 / instance_data.len() as f32;
            self.gi_scene_avg_albedo = [
                albedo_sum[0] * inv,
                albedo_sum[1] * inv,
                albedo_sum[2] * inv,
            ];
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
        self.fog_color_user_override = true;
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

    /// Composite-pass unsharp mask. Default 0.8; 0 disables the 4 extra
    /// HDR taps + extra tonemap entirely. Round-2 audit: this was
    /// hardcoded with no runtime control while visibly haloing
    /// silhouettes at 4K output (F3/F8).
    pub fn set_sharpen_strength(&mut self, strength: f32) {
        self.sharpen_strength = strength.max(0.0);
    }

    /// Present mode: 0 = Fifo (vsync), 1 = Mailbox (uncapped, no tearing),
    /// 2 = Immediate (uncapped, tearing allowed). Round-2 audit F6: the
    /// mode was hardcoded Fifo, which also made `set_target_fps` inert
    /// (its sleep-based cap only engages when vsync is off). All three
    /// modes are supported by DXGI on Windows; on other backends an
    /// unsupported request falls back to Fifo at configure time.
    pub fn set_present_mode(&mut self, mode: u32) {
        let requested = match mode {
            1 => wgpu::PresentMode::Mailbox,
            2 => wgpu::PresentMode::Immediate,
            _ => wgpu::PresentMode::Fifo,
        };
        if self.surface_config.present_mode == requested {
            return;
        }
        self.surface_config.present_mode = requested;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.surface_config);
        }
        eprintln!("bloom: present mode = {:?}", requested);
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
        self.sun_shaft_color_user_override = true;
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
    /// Load an environment from a `.hdr` file on disk and install it as
    /// the clear environment. Decode errors and I/O errors are silently
    /// ignored (the previous env stays active), matching the historical
    /// per-platform FFI behavior.
    ///
    /// Hoisted out of the per-platform `bloom_set_env_clear_from_hdr`
    /// wrappers so the shared FFI macro stays free of `image` codec
    /// details; gated on `image-extras` because the HDR codec is.
    #[cfg(all(feature = "image-extras", not(target_arch = "wasm32")))]
    pub fn set_env_clear_from_hdr_file(&mut self, path: &str) {
        use image::ImageDecoder;
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return,
        };
        let decoder = match image::codecs::hdr::HdrDecoder::new(std::io::BufReader::new(file)) {
            Ok(d) => d,
            Err(_) => return,
        };
        let (w, h) = decoder.dimensions();
        let byte_len = (w as usize) * (h as usize) * 3 * 4;
        let mut buf = vec![0u8; byte_len];
        if decoder.read_image(&mut buf).is_err() {
            return;
        }
        let rgb_f32: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.load_env_from_hdr(w, h, &rgb_f32);
    }

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
        let new_lighting_bg = self.make_lighting_bind_group("lighting_bg", &view, &diffuse_view_bg);

        self.sky_texture = Some(texture);
        self.sky_bind_group = Some(bg);
        self.env_diffuse_texture = Some(diffuse_texture);
        self.lighting_bind_group = new_lighting_bg;
        // EN-021 — the SSR bind group holds an env view; rebuild it when
        // a new HDR panorama is uploaded.
        self.ssr_bg_cache = None;
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

    // ========================================================================
    // EN-005 Phase 2 — procedural sky public surface
    // ========================================================================

    /// Toggle the procedural-atmosphere sky. When enabled, the HDR
    /// pass renders a Hillaire 2020 atmosphere driven by `set_sun_direction`
    /// instead of sampling a static panorama. Defaults to off so games
    /// that already call `bloom_set_env_map` keep their existing look.
    pub fn set_procedural_sky(
        &mut self,
        enabled: bool,
        rayleigh_density: f32,
        mie_density: f32,
        ground_albedo: f32,
    ) {
        self.procedural_sky_enabled = enabled;
        self.rayleigh_density = rayleigh_density.max(0.0);
        self.mie_density = mie_density.max(0.0);
        self.ground_albedo = ground_albedo.clamp(0.0, 1.0);
        self.sky_view_dirty = true;
        if !enabled && self.lighting_bg_is_procedural {
            // Switching back to the panorama path — rebuild lighting
            // bind group pointing at whatever HDR (or the 1×1 default)
            // was last loaded.
            self.swap_lighting_bg_to_panorama();
        }
    }

    /// Whether procedural sky is currently active. Used by the HDR
    /// pass to pick between `render_procedural_sky_pass` and the
    /// panorama-based `render_sky_pass`.
    pub fn procedural_sky_enabled(&self) -> bool {
        self.procedural_sky_enabled
    }

    /// Set the sun direction (world space, points *toward* the sun)
    /// and intensity multiplier. Triggers a sky-view LUT rebake on
    /// the next frame; the rebake itself is a single compute dispatch
    /// that takes ~0.1 ms on desktop GPUs.
    pub fn set_sun_direction(&mut self, dx: f32, dy: f32, dz: f32, intensity: f32) {
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
        self.sun_direction = [dx / len, dy / len, dz / len];
        self.sun_intensity = intensity.max(0.0);
        self.sky_view_dirty = true;
    }

    /// Recompute the sky-view LUT if the sun has moved (or atmosphere
    /// params changed) since the last bake. Called from the HDR pass
    /// before sampling the LUT in the procedural sky shader. No-op
    /// when the dirty flag is clear, so a static sun pays this cost
    /// exactly once.
    fn maybe_update_sky_view_lut(&mut self) {
        if !self.sky_view_dirty {
            return;
        }
        let params = SkyViewParams {
            sun: [
                self.sun_direction[0],
                self.sun_direction[1],
                self.sun_direction[2],
                self.sun_intensity,
            ],
            knobs: [self.rayleigh_density, self.mie_density, self.ground_albedo, 0.0],
        };
        self.queue
            .write_buffer(&self.sky_view_uniform_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sky_view_lut_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky_view_lut_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sky_view_compute_pipeline);
            pass.set_bind_group(0, &self.sky_view_compute_bind_group, &[]);
            // 8×8 workgroups; round up to cover the full LUT.
            let gx = (SKY_VIEW_W + 7) / 8;
            let gy = (SKY_VIEW_H + 7) / 8;
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // EN-005 Phase 3 — re-bake IBL from the freshly-updated sky-
        // view LUT so material reflections + ambient track the sun.
        self.bake_procedural_ibl();
        if !self.lighting_bg_is_procedural {
            self.swap_lighting_bg_to_procedural();
        }

        // EN-005 Phase 3 — auto-derive the sun-shaft tint from
        // atmospheric transmittance unless the user has overridden
        // it. Gives free physical sunset warmth.
        self.update_sun_shaft_color_from_transmittance();

        // EN-005 Phase 4 — auto-derive fog tint from the same LUT,
        // so distance haze tracks the sun. Cheap stand-in for full
        // 3D aerial-perspective LUT — captures the dominant signal
        // (warm fog at sunset, cool blue at noon) without the per-
        // frame compute pass + 3D texture binding plumbing.
        self.update_fog_color_from_atmosphere();

        self.sky_view_dirty = false;
    }

    /// EN-005 Phase 3 — re-render the procedural sky into the IBL
    /// equirect texture and run the existing GGX prefilter chain
    /// over its mip levels (plus the diffuse irradiance pass).
    /// Cheap (sub-millisecond on desktop), but only runs when the
    /// sun has actually moved.
    fn bake_procedural_ibl(&mut self) {
        let proc_w = self.procedural_sky_equirect_texture.width();
        let proc_h = self.procedural_sky_equirect_texture.height();
        let mip_count = self.procedural_sky_equirect_texture.mip_level_count();

        // Update the bake-pipeline uniform (just dest dims).
        let dims = [proc_w as f32, proc_h as f32, 0.0_f32, 0.0_f32];
        self.queue.write_buffer(&self.procedural_ibl_uniform_buffer, 0, bytemuck::cast_slice(&dims));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("procedural_ibl_encoder"),
        });

        // --- Step 1: render sky-view LUT → mip 0 of equirect ---
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("procedural_ibl_bake_mip0"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.procedural_sky_equirect_mip0_view,
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
            pass.set_pipeline(&self.procedural_ibl_pipeline);
            pass.set_bind_group(0, &self.procedural_ibl_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Step 2: GGX prefilter mips 1..N ---
        for level in 1..mip_count {
            let mip_w = (proc_w >> level).max(1);
            let mip_h = (proc_h >> level).max(1);
            let roughness = level as f32 / (mip_count - 1) as f32;
            let sample_count = (128.0 + 384.0 * roughness).round();
            let uniforms = PrefilterUniforms {
                params: [roughness, sample_count, mip_w as f32, mip_h as f32],
            };
            self.queue
                .write_buffer(&self.prefilter_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            let mip_view = self
                .procedural_sky_equirect_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("procedural_ibl_dst_mip"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("procedural_ibl_prefilter_pass"),
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
            pass.set_bind_group(0, &self.procedural_prefilter_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Step 3: diffuse irradiance pass ---
        let diffuse_w = self.procedural_env_diffuse_texture.width();
        let diffuse_h = self.procedural_env_diffuse_texture.height();
        let diffuse_uniforms = PrefilterUniforms {
            params: [1.0, 1024.0, diffuse_w as f32, diffuse_h as f32],
        };
        self.queue
            .write_buffer(&self.prefilter_uniform_buffer, 0, bytemuck::bytes_of(&diffuse_uniforms));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("procedural_ibl_diffuse_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.procedural_env_diffuse_view,
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
            pass.set_bind_group(0, &self.procedural_prefilter_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// EN-005 Phase 3 — restore `lighting_bind_group` to the
    /// panorama path. Called when the user disables procedural sky.
    /// Falls back to the 1×1 default env if no HDR was ever loaded.
    fn swap_lighting_bg_to_panorama(&mut self) {
        let env_view = self
            .sky_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .unwrap_or_else(|| {
                self._scene_env_default_texture
                    .create_view(&wgpu::TextureViewDescriptor::default())
            });
        let diffuse_view = self
            .env_diffuse_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .unwrap_or_else(|| {
                self._scene_env_default_texture
                    .create_view(&wgpu::TextureViewDescriptor::default())
            });
        let new_bg = self.make_lighting_bind_group("lighting_bg_panorama", &env_view, &diffuse_view);
        self.lighting_bind_group = new_bg;
        self.lighting_bg_is_procedural = false;
        // EN-021 — the SSR bind group holds an env view; rebuild it when
        // the env source swaps.
        self.ssr_bg_cache = None;
    }

    /// EN-005 Phase 3 — rebuild `lighting_bind_group` so PBR materials
    /// sample the procedural sky for IBL specular + diffuse instead of
    /// whatever panorama was last loaded. Called once on first
    /// procedural bake; the textures themselves are re-written on
    /// every sun-move and the bind group's TextureView references
    /// remain valid.
    fn swap_lighting_bg_to_procedural(&mut self) {
        let new_bg = self.make_lighting_bind_group(
            "lighting_bg_procedural",
            &self.procedural_sky_equirect_full_view,
            &self.procedural_env_diffuse_view,
        );
        self.lighting_bind_group = new_bg;
        self.lighting_bg_is_procedural = true;
        // EN-021 — the SSR env-fallback binding should track the active
        // env source. (SSR keeps sampling sky_texture today; nulling the
        // cache here at least rebuilds against the current state on the
        // next frame.)
        self.ssr_bg_cache = None;
    }

    /// Sample the transmittance LUT for the current sun direction
    /// at sea level and write the resulting tint into
    /// `sun_shaft_color`. RFC 0002 sub-RFC #1: shafts auto-warm at
    /// sunset when the procedural sky is on. Skipped when the user
    /// has set a manual override via `set_sun_shaft_color`.
    fn update_sun_shaft_color_from_transmittance(&mut self) {
        if self.sun_shaft_color_user_override {
            return;
        }
        // Sample at sea level (matches sky-view LUT camera height);
        // mu = sun.y is cos(sun-zenith) for an observer looking up
        // toward the sun.
        let r0 = 6360.0 + 0.5; // matches sky-view shader
        let mu = self.sun_direction[1];
        let lut = &self.transmittance_lut_cpu;
        let t = atmosphere_lut::sample_transmittance_lut(lut, TRANSMITTANCE_W, TRANSMITTANCE_H, r0, mu);
        // Multiply by sun radiance scalar; matches sun-disk's `vec3(20)
        // * t_sun * intensity` so shafts and the disk warm together.
        let radiance_scale = 1.0_f32; // shafts are tinted, not lit-additive — keep unscaled
        self.sun_shaft_color = [
            t[0] * radiance_scale * self.sun_intensity,
            t[1] * radiance_scale * self.sun_intensity,
            t[2] * radiance_scale * self.sun_intensity,
        ];
    }

    /// EN-005 V2 — bake the aerial-perspective 3D LUT for the
    /// current camera + sun. Called each frame from
    /// `end_frame_with_scene` before scene_compose runs (which
    /// samples the LUT). The LUT depends on the camera transform,
    /// so unlike the sky-view LUT it can't be gated on a dirty
    /// flag — every frame's view is unique.
    fn dispatch_aerial_perspective_lut(&mut self) {
        // Compute inverse VP from current view + projection.
        let v = self.current_view_matrix;
        let p = self.current_proj_matrix;
        // Multiply column-major: vp = p * v
        let mut vp = [[0.0_f32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += p[k][r] * v[c][k];
                }
                vp[c][r] = s;
            }
        }
        let inv_vp = util::mat4_invert(vp);

        // Camera world position from the view matrix. View is
        // world→camera, so cam_pos = -R^T * t where R is rotation,
        // t is the translation column. Equivalent: invert view and
        // take its translation column.
        let inv_view = util::mat4_invert(v);
        let cam_pos = [inv_view[3][0], inv_view[3][1], inv_view[3][2]];

        let params = AerialParams {
            cam_pos: [cam_pos[0], cam_pos[1], cam_pos[2], AERIAL_MAX_DIST_KM],
            inv_vp,
            sun: [
                self.sun_direction[0],
                self.sun_direction[1],
                self.sun_direction[2],
                self.sun_intensity,
            ],
            knobs: [self.rayleigh_density, self.mie_density, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.aerial_perspective_uniform_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aerial_perspective_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aerial_perspective_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.aerial_perspective_pipeline);
            pass.set_bind_group(0, &self.aerial_perspective_bind_group, &[]);
            let gx = (AERIAL_W + 7) / 8;
            let gy = (AERIAL_H + 7) / 8;
            pass.dispatch_workgroups(gx, gy, AERIAL_D);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// EN-005 Phase 4 — derive a sky-tinted fog color from the
    /// atmosphere. Approximates "what color would you see looking at
    /// the distant horizon" by blending a midday-blue baseline with a
    /// transmittance-derived sunset warm tint, gated by the sun
    /// elevation. Skipped when the user has set a manual override
    /// via `set_fog_color`.
    ///
    /// This is a CPU analytic stand-in for a full 3D aerial-
    /// perspective LUT — enough to make distance haze read as "tied
    /// to the sky" without per-frame compute or per-fragment 3D
    /// texture sampling.
    fn update_fog_color_from_atmosphere(&mut self) {
        if self.fog_color_user_override {
            return;
        }
        // Sun-zenith cosine. Clamped so a sun slightly below the
        // horizon still tints the fog (matches the procedural sky's
        // -0.05 cutoff for the sun disk).
        let mu_s = self.sun_direction[1].max(-0.05);
        let day_factor = (mu_s.max(0.0) * 2.0).clamp(0.0, 1.0);

        // Sample transmittance for a horizon-grazing ray. Gives the
        // R/G/B ratio of light surviving the long path through dense
        // air — the dominant signal behind sunset orange.
        let r0 = 6360.0 + 0.5;
        let lut = &self.transmittance_lut_cpu;
        let horizon_t = atmosphere_lut::sample_transmittance_lut(
            lut,
            TRANSMITTANCE_W,
            TRANSMITTANCE_H,
            r0,
            0.05, // ~3° above horizon
        );

        // Two end-points the model interpolates between as the sun
        // descends. Numbers tuned to read well at default exposure.
        let midday_blue = [0.45_f32, 0.55, 0.70];
        let sunset_warm = [
            horizon_t[0] * 0.9,
            horizon_t[1] * 0.55,
            horizon_t[2] * 0.25,
        ];

        let mix_factor = (1.0 - day_factor).powf(0.5);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let color = [
            lerp(midday_blue[0], sunset_warm[0], mix_factor),
            lerp(midday_blue[1], sunset_warm[1], mix_factor),
            lerp(midday_blue[2], sunset_warm[2], mix_factor),
        ];

        // Dim toward night so the fog doesn't keep glowing after sunset.
        let intensity_scale = 0.4 + 0.6 * day_factor;
        self.fog_color = [
            color[0] * intensity_scale,
            color[1] * intensity_scale,
            color[2] * intensity_scale,
        ];
    }

    /// Render the procedural sky into `pass`. Mirrors `render_sky_pass`
    /// but samples the sky-view LUT (built by `maybe_update_sky_view_lut`)
    /// instead of an HDR panorama. Caller is responsible for invoking
    /// `maybe_update_sky_view_lut` *before* opening the render pass —
    /// compute dispatches can't be nested inside a render pass.
    fn render_procedural_sky_pass(&self, pass: &mut wgpu::RenderPass<'_>, intensity: f32) {
        // Update sky uniforms (same camera basis as panorama path).
        let v = self.current_view_matrix;
        let right_world = [v[0][0], v[1][0], v[2][0]];
        let up_world = [v[0][1], v[1][1], v[2][1]];
        let forward_world = [-v[0][2], -v[1][2], -v[2][2]];
        let aspect = self.surface_config.width as f32 / self.surface_config.height as f32;
        let p = self.current_proj_matrix;
        let tan_half = if p[1][1].abs() > 1e-6 { 1.0 / p[1][1] } else { 1.0 };

        let uniforms = SkyUniforms {
            right: [
                right_world[0] * tan_half * aspect,
                right_world[1] * tan_half * aspect,
                right_world[2] * tan_half * aspect,
                0.0,
            ],
            up: [up_world[0] * tan_half, up_world[1] * tan_half, up_world[2] * tan_half, 0.0],
            forward: [forward_world[0], forward_world[1], forward_world[2], 0.0],
            // .y carries current time (seconds) so the cloud layer can drift.
            intensity: [intensity, self.lighting_uniforms.wind[3], 0.0, 0.0],
        };
        self.queue.write_buffer(&self.sky_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Sun disk parameters. Real solar disk is ~0.27° (4.7 mrad);
        // we exaggerate slightly for visibility at default exposure.
        let sun = SunUniforms {
            sun: [
                self.sun_direction[0],
                self.sun_direction[1],
                self.sun_direction[2],
                self.sun_intensity,
            ],
            params: [0.012, 0.6, 0.0, 0.0], // ~0.7° disk, moderate limb darkening
        };
        self.queue.write_buffer(&self.procedural_sun_uniform_buffer, 0, bytemuck::bytes_of(&sun));

        pass.set_pipeline(&self.procedural_sky_pipeline);
        pass.set_bind_group(0, &self.procedural_sky_bind_group, &[]);
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

    /// Alpha-tested shadow-caster bind group (base colour + sampler +
    /// cutoff) for the shadow pass's cutout pipeline. Same construction the
    /// cached-model path uses; exposed so scene-graph nodes with MASK
    /// materials cast dappled shadows instead of solid card silhouettes.
    /// The cutoff uniform buffer stays alive via the bind group.
    pub fn create_shadow_cutout_bg(&self, base_color_idx: u32, cutoff: f32) -> wgpu::BindGroup {
        use wgpu::util::DeviceExt;
        let cutoff_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow_cutout_cutoff"),
            contents: bytemuck::cast_slice(&[cutoff, 0.0f32, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bi = base_color_idx as usize;
        let base_tex = if base_color_idx == 0 || bi >= self.textures.len() {
            &self.textures[0]
        } else {
            &self.textures[bi]
        };
        let base_view = base_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_cutout_bg"),
            layout: &self.shadow_map.cutout_tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&base_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: cutoff_buf.as_entire_binding() },
            ],
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
        // frame's submissions start from an empty list. EN-022: the
        // reset also rotates the per-slot model history and pins the
        // previous frame's VP for motion-vector reconstruction.
        self.material_system.commands.clear();
        self.material_system.translucent_commands.clear();
        self.material_system.reset_draw_slot(self.prev_vp_matrix);

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
            match self.acquire_frame() {
                Some(t) => Some(t),
                None => {
                    // Swapchain lost+reconfigured. Restore RT views if set.
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
            view = self.frame_texture(surface_output.as_ref().unwrap()).create_view(&wgpu::TextureViewDescriptor::default());
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
        if let Some(out) = surface_output { self.present_frame(out); }
    }

    /// Like end_frame, but also renders retained scene graph nodes.
    pub fn end_frame_with_scene(&mut self, scene: &mut crate::scene::SceneGraph, profiler: &mut crate::profiler::Profiler) {
        profiler.begin("joint_flush");
        self.flush_joint_matrices();
        profiler.end("joint_flush");

        profiler.begin("surface_acquire");
        let output = match self.acquire_frame() {
            Some(t) => t,
            None => {
                profiler.end("surface_acquire");
                return;
            }
        };
        profiler.end("surface_acquire");
        let view = self.frame_texture(&output).create_view(&wgpu::TextureViewDescriptor::default());

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

        // Every graph node that consumes `surf` (Hi-Z, occlusion, GTAO,
        // SSAO blur, SSGI, bloom) operates on render-resolution inputs
        // and render-resolution RTs, so hand them the render extent —
        // NOT the swapchain size. The output-res passes (TSR upscale,
        // DoF/MB/SSS, composite) size themselves off their own targets.
        let (surf_w, surf_h) = self.render_extent();
        let exposure_src_idx = self.exposure_current_idx;
        let exposure_dst_idx = 1 - self.exposure_current_idx;

        // ============================================================
        // Frame render graph (RFC 0001 Phase 2b — complete).
        //
        // Every render pass between geometry upload and the terminal
        // composite runs as a PassNode. Reads/writes document the real
        // data dependencies; in addition, each node carries a with_after
        // pin to its predecessor so the schedule reproduces the
        // hand-tuned order exactly. Loosening those pins (to let the
        // scheduler interleave independent passes) is the documented
        // next refinement — do it dependency-by-dependency with the
        // golden tests watching.
        //
        // The context owns &mut Renderer, so node closures borrow
        // nothing at build time and can call the record_* methods.
        // Feature toggles (ssao/ssr/ssgi/bloom) are checked inside the
        // closures (or inside the methods), never by omitting nodes —
        // with_after on a missing node is a schedule error.
        // ============================================================
        {
            use graph::{Graph, PassInput, PassNode, PassOutput};
            // Transient ordering tokens for resources the enum doesn't
            // name. The textures themselves are persistent renderer
            // fields; these ids only express producer→consumer edges.
            const HIZ_PYRAMID: u32 = 0;
            const SSAO_TEX: u32 = 1;
            const SSR_TEX: u32 = 2;
            const SSGI_TEX: u32 = 3;
            const BLOOM_CHAIN: u32 = 4;
            const COMPOSED: u32 = 5;
            const LDR_FINAL: u32 = 6;
            const FROXEL_CLUSTERS: u32 = 7;

            struct FrameCtx2<'a> {
                r: &'a mut Renderer,
                encoder: &'a mut wgpu::CommandEncoder,
                profiler: &'a mut crate::profiler::Profiler,
                scene: &'a mut crate::scene::SceneGraph,
                surf: (u32, u32),
                exposure_idx: (usize, usize),
            }

            let mut g: Graph<FrameCtx2> = Graph::new();
            g.push(
                PassNode::new("froxel_assign", Box::new(|c: &mut FrameCtx2| {
                    // No-op when self.froxel is None (the method gates);
                    // the node stays in the graph so with_after pins
                    // never dangle.
                    c.r.record_froxel_assign(c.encoder);
                }))
                .with_writes(&[PassOutput::Transient(FROXEL_CLUSTERS)]),
            );
            g.push(
                PassNode::new("shadow", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_shadow_pass(c.encoder, c.profiler, c.scene);
                }))
                .with_writes(&[PassOutput::Shadow(0), PassOutput::Shadow(1), PassOutput::Shadow(2)]),
            );
            g.push(
                PassNode::new("hdr_scene", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_hdr_scene_pass(c.encoder, c.profiler, c.scene);
                }))
                .with_reads(&[
                    PassInput::Shadow(0),
                    PassInput::Shadow(1),
                    PassInput::Shadow(2),
                    PassInput::Transient(FROXEL_CLUSTERS),
                ])
                .with_writes(&[
                    PassOutput::HdrColor,
                    PassOutput::MaterialRt,
                    PassOutput::VelocityRt,
                    PassOutput::AlbedoRt,
                    PassOutput::Depth,
                ])
                .with_after(&["shadow", "froxel_assign"]),
            );
            g.push(
                PassNode::new("translucent", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_translucent_pass(c.encoder, c.profiler);
                }))
                // Reads the opaque HDR + depth and alpha-blends back into
                // HdrColor; the pin (not a second HdrColor write) keeps a
                // single declared writer per resource.
                .with_after(&["hdr_scene"]),
            );
            g.push(
                PassNode::new("hiz_build", Box::new(|c: &mut FrameCtx2| {
                    if !c.r.ssao_enabled {
                        return;
                    }
                    let (hw, hh) = ((c.surf.0 / 2).max(1), (c.surf.1 / 2).max(1));
                    let p22 = c.r.current_proj_matrix[2][2];
                    let p32 = c.r.current_proj_matrix[3][2];
                    c.r.record_hiz_chain(c.encoder, c.profiler, hw, hh, p22, p32);
                }))
                .with_reads(&[PassInput::SceneDepth])
                .with_writes(&[PassOutput::Transient(HIZ_PYRAMID)])
                .with_after(&["translucent"]),
            );
            g.push(
                PassNode::new("occlusion_capture", Box::new(|c: &mut FrameCtx2| {
                    if !c.r.ssao_enabled {
                        return;
                    }
                    let (hw, hh) = ((c.surf.0 / 2).max(1), (c.surf.1 / 2).max(1));
                    let vp = c.r.vp_matrix();
                    let occlusion = &mut c.r.occlusion as *mut OcclusionCuller;
                    unsafe {
                        (*occlusion).record(&c.r.device, &c.r.queue, c.encoder, &c.r.hiz_views[0], (hw, hh), vp);
                    }
                }))
                .with_reads(&[PassInput::Transient(HIZ_PYRAMID)])
                .with_after(&["hiz_build"]),
            );
            g.push(
                PassNode::new("gtao", Box::new(|c: &mut FrameCtx2| {
                    if !c.r.ssao_enabled {
                        return;
                    }
                    let (hw, hh) = ((c.surf.0 / 2).max(1), (c.surf.1 / 2).max(1));
                    let p = &c.r.current_proj_matrix;
                    let (p00, p11, p20, p21) = (p[0][0], p[1][1], p[2][0], p[2][1]);
                    c.r.record_gtao(c.encoder, c.profiler, hw, hh, p00, p11, p20, p21);
                }))
                .with_reads(&[PassInput::Transient(HIZ_PYRAMID)])
                .with_after(&["occlusion_capture"]),
            );
            g.push(
                PassNode::new("ssao_blur", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_ssao_blur(c.encoder, c.surf.0, c.surf.1);
                }))
                .with_writes(&[PassOutput::Transient(SSAO_TEX)])
                .with_after(&["gtao"]),
            );
            g.push(
                PassNode::new("ssr_march", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_ssr_march(c.encoder, c.profiler);
                }))
                .with_reads(&[PassInput::SceneColor, PassInput::SceneDepth])
                .with_after(&["ssao_blur"]),
            );
            g.push(
                PassNode::new("ssr_temporal", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_ssr_temporal(c.encoder);
                }))
                .with_writes(&[PassOutput::Transient(SSR_TEX)])
                .with_after(&["ssr_march"]),
            );
            g.push(
                PassNode::new("ssgi", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_ssgi_passes(c.encoder, c.profiler, c.surf.0, c.surf.1);
                }))
                .with_reads(&[PassInput::SceneColor, PassInput::SceneDepth])
                .with_writes(&[PassOutput::Transient(SSGI_TEX)])
                .with_after(&["ssr_temporal"]),
            );
            g.push(
                PassNode::new("bloom", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_bloom_chain(c.encoder, c.profiler, c.surf.0, c.surf.1);
                }))
                .with_reads(&[PassInput::SceneColor])
                .with_writes(&[PassOutput::Transient(BLOOM_CHAIN)])
                .with_after(&["ssgi"]),
            );
            g.push(
                PassNode::new("compose", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_scene_compose(c.encoder);
                }))
                .with_reads(&[
                    PassInput::SceneColor,
                    PassInput::Transient(SSAO_TEX),
                    PassInput::Transient(SSR_TEX),
                    PassInput::Transient(SSGI_TEX),
                    PassInput::Transient(BLOOM_CHAIN),
                ])
                .with_writes(&[PassOutput::Transient(COMPOSED)])
                .with_after(&["bloom"]),
            );
            g.push(
                PassNode::new("postfx_tail", Box::new(|c: &mut FrameCtx2| {
                    c.r.record_postfx_tail(c.encoder, c.profiler);
                }))
                .with_reads(&[PassInput::Transient(COMPOSED), PassInput::MotionVectors])
                .with_writes(&[PassOutput::Transient(LDR_FINAL)])
                .with_after(&["compose"]),
            );
            g.push(
                PassNode::new("auto_exposure", Box::new(|c: &mut FrameCtx2| {
                    let (src, dst) = c.exposure_idx;
                    c.r.record_auto_exposure(c.encoder, src, dst);
                }))
                .with_reads(&[PassInput::Transient(LDR_FINAL)])
                .with_after(&["postfx_tail"]),
            );

            let mut ctx = FrameCtx2 {
                r: self,
                encoder: &mut encoder,
                profiler,
                scene,
                surf: (surf_w, surf_h),
                exposure_idx: (exposure_src_idx, exposure_dst_idx),
            };
            if let Err(e) = g.execute(&mut ctx) {
                // A schedule error means a malformed graph (cycle /
                // unknown pin) — a programming error, not a runtime
                // condition. Surface loudly; the frame still presents
                // whatever was encoded before the failure.
                eprintln!("[graph] frame graph failed: {:?}", e);
            }
        }

        // CPU bracket over the composite + custom post-pass encode tail.
        // The matching `end("post_fx")` below predates this begin — it was
        // orphaned (a no-op) for months while comments claimed the phase
        // covered scene_compose's cost.
        profiler.begin("post_fx");
        let composite_src_view = self.composite_source_view();

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
        // EN-017 V2 — when at least one post-pass is installed the
        // composite writes into ping-pong slot A, then each pass
        // ping-pongs A/B with the LAST pass writing the swapchain.
        // Otherwise composite still writes the swapchain directly
        // (zero-cost original path).
        let composite_target_view: &wgpu::TextureView = if !self.post_passes.is_empty() {
            self.composite_ldr_rt_a_view.as_ref().unwrap_or(&view)
        } else {
            &view
        };
        {
            let final_composite_ts = profiler.pass_timestamp_writes("final_composite_pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: composite_target_view,
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

        // EN-017 V2 — fullscreen post-pass STACK. Each pass samples
        // the previous pass's output (or composite_ldr_rt_a for
        // pass 0) + scene depth, then writes either the next ping-
        // pong intermediate or — for the last pass — the swapchain.
        // Runs after composite + tonemapping but before the 2D
        // overlay so the HUD stays crisp.
        //
        // Bind groups are built transiently per-frame: BG creation
        // is cheap (~µs), and ping-ponging requires per-direction
        // BGs anyway so caching adds bookkeeping for marginal gain.
        let n_passes = self.post_passes.len();
        if n_passes > 0 {
            for i in 0..n_passes {
                // input view = (i % 2 == 0) ? A : B
                // output view = last ? swapchain
                //               : (i % 2 == 0) ? B : A
                let input_view: &wgpu::TextureView = if i % 2 == 0 {
                    // Always defined: composite wrote A above.
                    self.composite_ldr_rt_a_view.as_ref().unwrap_or(&view)
                } else {
                    // Should be defined: add_post_pass allocated B
                    // when the stack grew to >= 2. Fallback to view
                    // is defensive.
                    self.composite_ldr_rt_b_view.as_ref().unwrap_or(&view)
                };
                let is_last = i == n_passes - 1;
                let output_view: &wgpu::TextureView = if is_last {
                    &view
                } else if i % 2 == 0 {
                    self.composite_ldr_rt_b_view.as_ref().unwrap_or(&view)
                } else {
                    self.composite_ldr_rt_a_view.as_ref().unwrap_or(&view)
                };

                let pp = &self.post_passes[i];
                let pp_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("post_pass_bg"),
                    layout: &pp.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.post_pass_depth_sampler),
                        },
                    ],
                });
                let pp_ts = profiler.pass_timestamp_writes("post_pass");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_post_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: pp_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pp.pipeline);
                pass.set_bind_group(0, &pp_bg, &[]);
                pass.draw(0..3, 0..1);
            }
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
            eprintln!("bloom: screenshot readback branch running");
            // Use actual texture dimensions (accounts for Retina/DPI scaling)
            let tex_size = self.frame_texture(&output).size();
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
                    texture: self.frame_texture(&output),
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
                    // Failures were silently swallowed here for months —
                    // takeScreenshot "worked" while writing nothing.
                    match encode_png_simple(width, height, &rgb) {
                        Some(png) => {
                            if let Err(e) = std::fs::write(&path, &png) {
                                eprintln!("bloom: screenshot write '{}' failed: {}", path, e);
                            }
                        }
                        None => eprintln!(
                            "bloom: screenshot PNG encode failed ({}x{})",
                            width, height
                        ),
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

        // Map the occlusion-grid readback recorded this frame (no-op if
        // none was recorded).
        self.occlusion.after_submit();

        // Ticket 022 — drain freshly-baked SDFs to the on-disk cache.
        // No-op on cache-hit frames (queue is empty); on cold-launch
        // bake frames it blocks briefly on a single device.poll(Wait)
        // covering all 8 readbacks. Skipped on wasm32 (no filesystem
        // path, sdf_cache::store returns Err immediately).
        #[cfg(not(target_arch = "wasm32"))]
        {
            profiler.begin("sdf_cache_write");
            self.flush_sdf_cache_writes();
            profiler.end("sdf_cache_write");
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        profiler.begin("swap_present");
        self.present_frame(output);
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
        // EN-022 fix — velocity reference inputs roll over every frame
        // regardless of TAA state (velocity also feeds motion blur).
        self.prev_proj_matrix_unjittered = self.current_proj_matrix_unjittered;
        self.prev_view_matrix = self.current_view_matrix;
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

    /// Raw 0-255 → 0-1, no gamma. Used by the 3D immediate batch and
    /// model tints, whose values feed the linear HDR pipeline directly —
    /// changing their interpretation would silently re-tint every
    /// existing drawCube/drawModel call in shipped games.
    fn color_to_f32(r: f64, g: f64, b: f64, a: f64) -> [f32; 4] {
        [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32]
    }

    /// 2D variant: sRGB-decodes rgb so the sRGB swapchain view's hardware
    /// encode does not double-encode (see `srgb_u8_to_linear`).
    fn color_to_f32_srgb(r: f64, g: f64, b: f64, a: f64) -> [f32; 4] {
        [srgb_u8_to_linear(r), srgb_u8_to_linear(g), srgb_u8_to_linear(b), (a / 255.0) as f32]
    }


    pub fn draw_triangle(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32_srgb(r, g, b, a);
        let base = self.vertices_2d.len() as u32;

        self.vertices_2d.push(Vertex2D { position: [x1 as f32, y1 as f32], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x2 as f32, y2 as f32], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x3 as f32, y3 as f32], uv: [0.0, 0.0], color });

        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2]);
    }

    pub fn draw_poly(&mut self, cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32_srgb(r, g, b, a);
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
        let color = Self::color_to_f32_srgb(tint_r, tint_g, tint_b, tint_a);
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
        let color = Self::color_to_f32_srgb(tint_r, tint_g, tint_b, tint_a);
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
        let color = Self::color_to_f32_srgb(tint_r, tint_g, tint_b, tint_a);
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
            // Jitter is sub-pixel in *render* space — at fractional
            // render_scale the G-buffer is smaller than the surface,
            // so each render pixel covers 1/scale surface pixels and
            // the offset must scale accordingly. render_extent()
            // already reflects render_scale.
            let (rw, rh) = self.render_extent();
            let render_w = rw.max(1) as f32;
            let render_h = rh.max(1) as f32;
            // proj is column-major; column 2 row 0/1 are the
            // perspective / Z-coupling slots. Adding a constant NDC
            // offset there shifts the whole frustum by jitter px.
            proj[2][0] += (jx * 2.0) / render_w;
            proj[2][1] += (jy * 2.0) / render_h;
            self.current_jitter_ndc = [(jx * 2.0) / render_w, (jy * 2.0) / render_h];
        } else {
            self.current_jitter_ndc = [0.0, 0.0];
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

        // EN-022 fix — compose the velocity reference VP: the previous
        // frame's UNJITTERED projection with the CURRENT frame's jitter
        // re-applied, times the previous view. Every prev_mvp built
        // from this cancels the jitter term exactly in the shader's
        // (curr_ndc - prev_ndc), so static geometry gets a true zero
        // velocity instead of one-texel jitter-delta noise (which
        // wobbled TAA history reprojection and cycled fine detail
        // between sharp and soft — the periodic material-surface
        // flicker).
        let mut prev_proj_j = self.prev_proj_matrix_unjittered;
        prev_proj_j[2][0] += self.current_jitter_ndc[0];
        prev_proj_j[2][1] += self.current_jitter_ndc[1];
        self.velocity_ref_vp = mat4_multiply(prev_proj_j, self.prev_view_matrix);
        self.material_system.set_velocity_reference_vp(self.velocity_ref_vp);

        // Mirror camera pos into lighting uniforms so the scene shader
        // can compute V for GGX specular. Preserve the .w slot — it
        // holds the env_intensity multiplier (set via load_env_from_hdr).
        let env_intensity_w = self.lighting_uniforms.camera_pos[3];
        self.lighting_uniforms.camera_pos = [pos_x, pos_y, pos_z, env_intensity_w];
        // Pass the current cascade shadow VPs and view matrix (computed
        // in end_frame_with_scene) so the scene shader's CSM lookup
        // lands on the right cascade map.
        self.lighting_uniforms.shadow_cascade_vps = self.shadow_map.light_vps;
        // .w is the TSR mip-LOD bias slot — owned by the shadow pass's
        // per-frame upload (shadow_pass.rs), which overwrites this struct
        // later in the frame anyway. Keep 0.0 here so a frame without a
        // shadow pass reads a neutral bias. The shadows-enabled flag lives
        // in dir_light_count.y (see clear_additional_lights), NOT here.
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
            bytemuck::bytes_of(&Uniforms3D { mvp: vp, model: IDENTITY_MAT4, prev_mvp: self.velocity_ref_vp, model_tint: [1.0, 1.0, 1.0, 1.0] }),
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
    #[cfg(feature = "models3d")]
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
            // Cutout casters get an alpha-test shadow bind group (base colour +
            // sampler + cutoff). wgpu keeps the buffer alive via the bind group,
            // but we hold the strong ref too (matches _material_uniform).
            let (shadow_cutout_bg, shadow_cutoff_buf) = if mesh.alpha_cutoff > 0.0 {
                let cutoff_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("shadow_cutout_cutoff"),
                    contents: bytemuck::cast_slice(&[mesh.alpha_cutoff, 0.0f32, 0.0, 0.0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bi = base_color_idx as usize;
                let base_tex = if base_color_idx == 0 || bi >= self.textures.len() {
                    &self.textures[0]
                } else {
                    &self.textures[bi]
                };
                let base_view = base_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow_cutout_bg"),
                    layout: &self.shadow_map.cutout_tex_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&base_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: cutoff_buf.as_entire_binding() },
                    ],
                });
                (Some(bg), Some(cutoff_buf))
            } else {
                (None, None)
            };
            GpuMesh {
                vb,
                ib,
                index_count: mesh.indices.len() as u32,
                material_bg,
                _material_uniform: material_uniform,
                shadow_cutout_bg,
                _shadow_cutoff_buf: shadow_cutoff_buf,
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
                model: model_matrix,
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
        // dir_light_count.y carries the shadows-enabled flag for the shadow
        // samplers (core sample_shadow + material-path sample_sun_shadow).
        // Written here because this runs unconditionally every frame before
        // any lighting upload; .x stays the actual additional-light count.
        // (shadow_cascade_splits.w is NOT usable for this — it carries the
        // TSR mip-LOD bias, written by the shadow pass each frame.)
        let shadows_flag = if self.shadow_map.enabled { 1.0 } else { 0.0 };
        // dir_light_count.z carries SSR's ownership share for EN-021's
        // IBL-specular complement in fs_main_scene: strength while SSR
        // runs, 0 when disabled (full IBL specular returns).
        let ssr_share = if self.ssr_enabled { self.ssr_strength } else { 0.0 };
        self.lighting_uniforms.dir_light_count = [0.0, shadows_flag, ssr_share, 0.0];
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
            // Outward winding (matches the declared normals). The old
            // order wound every face inward: with back-face culling you
            // saw each cube's interior — same bug that made draw_plane
            // invisible from above.
            self.indices_3d.extend_from_slice(&[base, base+2, base+1, base, base+3, base+2]);
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
        // Wind so the +Y-normal side is the front face when seen from
        // above — the previous order back-face-culled the plane from
        // every camera above it (only visible from underneath).
        self.indices_3d.extend_from_slice(&[base, base+2, base+1, base, base+3, base+2]);
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

    /// Same as `draw_model_mesh_tinted` but applies a Y-axis rotation
    /// (radians) to the mesh local space before scale + translate.
    /// Skinned meshes ignore the rotation here — pose joints already
    /// drive their orientation. CPU-side baking mirrors the unrotated
    /// path so callers can mix rotated and unrotated draws freely
    /// without extra GPU state.
    pub fn draw_model_mesh_tinted_rotated(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32, tint: [f32; 4], texture_idx: u32, rot_y: f32) {
        self.ensure_draw_state_3d(texture_idx);

        // Mirror the joint-pose plumbing in the non-rotated path so a
        // skinned mesh drawn here still consumes its pending pose.
        let mesh_skinned = vertices.iter().any(|v|
            v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01);
        let joint_offset: f32 = if mesh_skinned && !self.pending_skin_groups.is_empty() {
            let group = self.pending_skin_groups.remove(0);
            let start = self.frame_joint_data.len();
            if start + group.len() <= 1024 {
                self.frame_joint_data.extend_from_slice(&group);
                start as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        let cos_y = rot_y.cos();
        let sin_y = rot_y.sin();
        let base = self.vertices_3d.len() as u32;
        for v in vertices {
            let is_skinned = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01;
            let pos = if is_skinned {
                v.position
            } else {
                // Rotate local-space position around Y, then scale + translate.
                let lx = v.position[0];
                let ly = v.position[1];
                let lz = v.position[2];
                let rx =  cos_y * lx + sin_y * lz;
                let rz = -sin_y * lx + cos_y * lz;
                [rx * scale + position[0],
                 ly * scale + position[1],
                 rz * scale + position[2]]
            };
            // Rotate the surface normal too so lighting matches the new
            // orientation. Y-axis rotation leaves normal.y untouched.
            let n = v.normal;
            let normal = if is_skinned {
                n
            } else {
                [ cos_y * n[0] + sin_y * n[2],
                  n[1],
                 -sin_y * n[0] + cos_y * n[2] ]
            };
            // Rotate tangent.xyz the same way; preserve handedness in w.
            let t = v.tangent;
            let tangent = if is_skinned {
                t
            } else {
                [ cos_y * t[0] + sin_y * t[2],
                  t[1],
                 -sin_y * t[0] + cos_y * t[2],
                  t[3] ]
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
                normal,
                color: [
                    v.color[0] * tint[0],
                    v.color[1] * tint[1],
                    v.color[2] * tint[2],
                    v.color[3] * tint[3],
                ],
                uv: v.uv,
                joints: joints_out,
                weights: v.weights,
                tangent,
            });
        }
        for &idx in indices {
            self.indices_3d.push(base + idx);
        }
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
        let output = match self.acquire_frame() {
            Some(t) => t,
            None => return None,
        };
        let texture = self.frame_texture(&output);

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
        self.present_frame(output);

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
            false,
        )
    }

    /// Phase 4a — full-control material compile. Games that want a
    /// translucent / refractive / additive material (or a non-default
    /// bucket) call this directly. Plain `compile_material` is a
    /// convenience for Opaque + no scene reads.
    ///
    /// `wants_instancing` adds a per-instance vertex buffer layout at
    /// slot 1 (EN-001). Materials compiled with it must be drawn via
    /// `submit_material_draw_instanced` + a buffer from
    /// `create_instance_buffer`.
    pub fn compile_material_with_options(
        &mut self, wgsl_source: &str,
        profile:          material_pipeline::FragmentProfile,
        bucket:           material_pipeline::Bucket,
        reads_scene:      bool,
        wants_instancing: bool,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        self.material_system.compile(
            &self.device,
            wgsl_source,
            profile,
            bucket,
            reads_scene,
            wants_instancing,
            formats::HDR_FORMAT,
            formats::MATERIAL_FORMAT,
            formats::VELOCITY_FORMAT,
            wgpu::TextureFormat::Rgba8Unorm,
            formats::DEPTH_FORMAT,
        )
    }

    /// EN-001 — compile a material that opts into the standard per-instance
    /// vertex layout (Opaque profile + Opaque bucket + wants_instancing).
    /// Pair with `create_instance_buffer` + `submit_material_draw_instanced`.
    /// The game shader's VertexInput must declare the instance attribute
    /// locations (see `material_abi.wgsl` for the layout).
    pub fn compile_material_instanced(
        &mut self, wgsl_source: &str,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        self.material_system.compile(
            &self.device,
            wgsl_source,
            material_pipeline::FragmentProfile::Opaque,
            material_pipeline::Bucket::Opaque,
            false,
            true, // wants_instancing
            formats::HDR_FORMAT,
            formats::MATERIAL_FORMAT,
            formats::VELOCITY_FORMAT,
            wgpu::TextureFormat::Rgba8Unorm,
            formats::DEPTH_FORMAT,
        )
    }

    /// EN-001 — upload a CPU-side per-instance buffer to GPU memory.
    /// `raw` is laid out as 9 floats per instance (pos.xyz, rot_y,
    /// scale, tint.rgba). Returns a handle for use with
    /// `submit_material_draw_instanced`.
    pub fn create_instance_buffer(&mut self, raw: &[f32], count: u32) -> u32 {
        self.material_system.create_instance_buffer(&self.device, &self.queue, raw, count)
    }

    /// EN-001 — release the GPU memory backing an instance buffer.
    /// Safe to call with handle 0 or stale handles (no-op).
    pub fn destroy_instance_buffer(&mut self, handle: u32) {
        self.material_system.destroy_instance_buffer(handle);
    }

    /// EN-017 V2 — append a fullscreen WGSL post-pass to the stack.
    /// Compiles the shader, lazily allocates ping-pong LDR
    /// intermediates as the stack grows, and pushes onto the stack.
    /// Returns the 1-based handle of the newly added pass on success
    /// (so callers can treat 0 as "compile failed"), or Err on
    /// shader-compile failure; the existing stack is left intact.
    ///
    /// The fragment shader sees `scene_color_tex` (LDR, post-tonemap)
    /// + `scene_depth_tex` at `@group(0)` — see
    /// `post_pass::POST_PASS_PRELUDE` for the exact ABI.
    ///
    /// Stack order matters: the first added pass runs first, the
    /// next sees the first's output, and so on. The last pass writes
    /// the swapchain.
    pub fn add_post_pass(
        &mut self, wgsl_source: &str,
    ) -> Result<u32, post_pass::PostPassCompileError> {
        let pipeline = post_pass::compile_post_pass(
            &self.device, wgsl_source, self.surface_config.format,
        )?;

        // Slot A is needed the moment we have any post-pass at all
        // (composite redirects into A). Allocate lazily so titles
        // without a post-pass pay zero memory.
        if self.composite_ldr_rt_a.is_none() {
            let (t, v) = post_pass::create_composite_ldr_rt(
                &self.device,
                self.surface_config.width,
                self.surface_config.height,
                self.surface_config.format,
            );
            self.composite_ldr_rt_a = Some(t);
            self.composite_ldr_rt_a_view = Some(v);
        }
        // Slot B is only needed once the stack reaches 2 passes
        // (single-pass setups read A and write the swapchain).
        // Pushing brings len to >= 2 ⇒ we'll need B next dispatch.
        if self.post_passes.len() + 1 >= 2 && self.composite_ldr_rt_b.is_none() {
            let (t, v) = post_pass::create_composite_ldr_rt(
                &self.device,
                self.surface_config.width,
                self.surface_config.height,
                self.surface_config.format,
            );
            self.composite_ldr_rt_b = Some(t);
            self.composite_ldr_rt_b_view = Some(v);
        }

        self.post_passes.push(pipeline);
        // 1-based handle: 0 reserved for "failed" at the FFI layer.
        Ok(self.post_passes.len() as u32)
    }

    /// EN-017 V2 — wipe the post-pass stack. The composite output
    /// goes directly to the swapchain again (zero post-pass cost).
    /// LDR intermediates stay allocated (cheap to keep around — at
    /// most two full-screen RGBA8 — and avoids re-alloc if the game
    /// toggles the stack frequently).
    pub fn clear_all_post_passes(&mut self) {
        self.post_passes.clear();
    }

    /// EN-017 V1 backward-compat — replace the entire stack with a
    /// single post-pass. Equivalent to `clear_all_post_passes()` +
    /// `add_post_pass(wgsl)` but ignores the returned handle so old
    /// callers keep their `Result<(), _>` ABI.
    pub fn set_post_pass(
        &mut self, wgsl_source: &str,
    ) -> Result<(), post_pass::PostPassCompileError> {
        self.clear_all_post_passes();
        self.add_post_pass(wgsl_source)?;
        Ok(())
    }

    /// EN-017 V1 backward-compat — equivalent to
    /// `clear_all_post_passes()`. Kept so existing FFI symbols on
    /// every platform continue to compile against the renderer.
    pub fn clear_post_pass(&mut self) {
        self.clear_all_post_passes();
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
        let handle = self.compile_material_with_options(&source, profile, bucket, reads_scene, false)
            .map_err(|e| format!("compile {canonical:?}: {e:?}"))?;
        self.material_hot_reload.register(
            handle,
            hot_reload::FileMaterialDesc {
                path: canonical, profile, bucket, reads_scene,
                wants_instancing: false,
            },
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
                desc.wants_instancing,
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

    /// EN-001 — submit an instanced material draw. The mesh is drawn
    /// `instance_count` times via a single `draw_indexed` with the
    /// instance buffer bound at vertex slot 1. Per-draw model/MVP are
    /// identity / current VP — per-instance pos/rot_y/scale dominate
    /// from the buffer; the per-draw `tint` is multiplied against the
    /// per-instance tint in the shader.
    pub fn submit_material_draw_instanced(
        &mut self,
        material: material_system::MaterialHandle,
        mesh_handle: u64,
        mesh_idx: usize,
        instance_buffer: u32,
        instance_count: u32,
    ) {
        let model = IDENTITY_MAT4;
        let mvp = self.current_vp_matrix;
        self.material_system.submit_draw_instanced(
            &self.device, &self.queue, &self.joint_buffer,
            material, mesh_handle, mesh_idx,
            instance_buffer, instance_count,
            mvp, model, mvp, [1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0],
        );
    }

    // ============================================================
    // EN-011 — planar reflection probes
    // ============================================================

    /// Create a planar reflection probe and return its 1-based handle
    /// (0 on failure). The probe owns a square HDR colour RT + depth
    /// at `resolution²`; pass `0` for `resolution` to default to half
    /// the swapchain width (matches the V1 ticket spec). The engine
    /// renders the world (minus excluded materials) into the probe's
    /// RT every frame in `dispatch_planar_reflections`.
    ///
    /// Materials opt into sampling a probe via
    /// `set_material_reflection_probe(material, probe)`.
    pub fn create_planar_reflection(
        &mut self,
        plane_y:    f32,
        normal:     [f32; 3],
        resolution: u32,
    ) -> u32 {
        let res = if resolution == 0 {
            (self.surface_config.width / 2).max(16)
        } else {
            resolution
        };
        let probe = planar_reflection::PlanarReflectionProbe::new(
            &self.device, plane_y, normal, res,
        );

        // Allocate the per-probe PerView UBO. The matching bind group
        // is built EACH FRAME inside `dispatch_planar_reflections`
        // (EN-011 V2) so it picks up the live env / BRDF / shadow
        // views from the renderer — bound stubs in V1 left mirrored
        // draws without IBL or sun shadows. We still keep
        // `planar_probe_view_bgs` allocated as `None` to preserve
        // index-parity with `planar_probes`, but never store anything
        // into the slot.
        let view_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_probe_per_view"),
            size: std::mem::size_of::<material_system::PerViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.planar_probes.push(Some(probe));
        self.planar_probe_view_buffers.push(Some(view_buffer));
        self.planar_probe_view_bgs.push(None);
        self.planar_probes.len() as u32
    }

    /// Link a material handle to a planar reflection probe. Subsequent
    /// draws of `material` see the probe's colour RT at
    /// `@group(2) @binding(12)`. Pass `probe = 0` to revert to the
    /// default 1×1 black texture.
    ///
    /// No-op when either handle is invalid.
    pub fn set_material_reflection_probe(&mut self, material: u32, probe: u32) {
        if material == 0 { return; }
        if probe == 0 {
            // Unlink — rebind binding 12 to the default black view by
            // routing through the same helper with our own default.
            let view = self.material_system.default_black_view.clone();
            // Cloning a wgpu TextureView is a cheap Arc bump.
            if let Err(e) = self.material_system.set_reflection_probe(
                &self.device, material, 0, &view,
            ) {
                eprintln!("[planar_reflection] unlink failed for material {material}: {e}");
            }
            return;
        }
        let idx = probe as usize - 1;
        let probe_view = match self.planar_probes.get(idx).and_then(|p| p.as_ref()) {
            Some(p) => p.color_view.clone(),
            None => {
                eprintln!("[planar_reflection] unknown probe handle {probe}");
                return;
            }
        };
        if let Err(e) = self.material_system.set_reflection_probe(
            &self.device, material, probe, &probe_view,
        ) {
            eprintln!("[planar_reflection] set failed for material {material}: {e}");
        }
    }

    // ============================================================
    // EN-014 — texture-array slots for splat-mapped terrain
    // ============================================================

    /// Create a texture array from a slice of layer source data
    /// (`(rgba8 bytes, width, height)` per layer). All layers must
    /// share the same `width × height`. V1 caps the layer count at
    /// `MAX_TEXTURE_ARRAY_LAYERS` (16). Returns a 1-based handle, or
    /// 0 on failure (empty / mismatched extents / zero-sized layer).
    pub fn create_texture_array(&mut self, layers: &[(&[u8], u32, u32)]) -> u32 {
        self.material_system.create_texture_array(&self.device, &self.queue, layers)
    }

    /// EN-014 V2 — create a texture array with explicit format + mip
    /// control. See `MaterialSystem::create_texture_array_ex` for the
    /// format / mip_levels semantics. The V1 `create_texture_array`
    /// remains as a thin wrapper that forwards `(format=0, mip_levels=1)`.
    pub fn create_texture_array_ex(
        &mut self,
        layers: &[(&[u8], u32, u32)],
        format: u32,
        mip_levels: u32,
    ) -> u32 {
        self.material_system.create_texture_array_ex(
            &self.device, &self.queue, layers, format, mip_levels,
        )
    }

    /// Link a texture array to a material at one of three slots:
    ///   - 0 = albedo  (binding 14)
    ///   - 1 = normal  (binding 15)
    ///   - 2 = MR      (binding 16)
    /// Pass `array = 0` to revert the slot to the engine's 1×1×1
    /// stub. No-op for unknown handles or out-of-range slots.
    ///
    /// Resolves the material's currently-linked planar reflection
    /// probe (if any) so binding 12 stays pointing at the probe's
    /// RT across the BG rebuild — otherwise an EN-014 rebind would
    /// silently drop an EN-011 reflection link.
    pub fn set_material_texture_array(&mut self, material: u32, slot: u32, array: u32) {
        let probe_view = match self.material_system.material_reflection_probe_handle(material) {
            Some(probe) if probe != 0 => {
                let idx = probe as usize - 1;
                self.planar_probes.get(idx)
                    .and_then(|p| p.as_ref())
                    .map(|p| p.color_view.clone())
                    .unwrap_or_else(|| self.material_system.default_black_view.clone())
            }
            _ => self.material_system.default_black_view.clone(),
        };
        self.material_system.set_material_texture_array(
            &self.device, material, slot, array, &probe_view,
        );
    }

    // ============================================================
    // EN-012 — Foliage shading model
    // ============================================================

    /// EN-012 — set the shading model for `material`. Pass `0` for
    /// default lit (standard PBR), `1` for foliage (wrap-lambert +
    /// transmission), or `2` for subsurface (V2 stub — currently
    /// behaves as default lit; the shader wouldn't branch on it
    /// until V2 ships). Lazily allocates a per-material
    /// `MaterialFactors` UBO and rebuilds the per-material BG so
    /// subsequent draws see the new shading model.
    ///
    /// Resolves the material's currently-linked planar-reflection
    /// probe (if any) so binding 12 stays pointing at the probe's
    /// RT across the BG rebuild — same precedent as
    /// `set_material_texture_array`.
    pub fn set_material_shading_model(&mut self, material: u32, model: u32) {
        let probe_view = self.resolve_probe_view_for_material(material);
        if let Err(e) = self.material_system.set_material_shading_model(
            &self.device, &self.queue, material, model, &probe_view,
        ) {
            eprintln!("[foliage] set_material_shading_model failed: {e}");
        }
    }

    /// EN-012 — set the foliage shading parameters for `material`.
    /// Only takes effect when `shading_model == 1` (foliage).
    /// `trans_color` is the rgb tint for back-lit foliage,
    /// `trans_amount` is 0..1 (how much sun bleeds through), and
    /// `wrap_factor` is 0..1 (wrap-lambert intensity). See the
    /// `shade_foliage` helper in `common/pbr.wgsl`.
    pub fn set_material_foliage(
        &mut self,
        material:    u32,
        trans_color: [f32; 3],
        trans_amount: f32,
        wrap_factor:  f32,
    ) {
        let probe_view = self.resolve_probe_view_for_material(material);
        if let Err(e) = self.material_system.set_material_foliage(
            &self.device, &self.queue, material,
            trans_color, trans_amount, wrap_factor, &probe_view,
        ) {
            eprintln!("[foliage] set_material_foliage failed: {e}");
        }
    }

    /// EN-012 — shared helper: resolve the planar-reflection probe
    /// view for the material's currently-linked probe (or the
    /// default 1×1 black view when none is linked). Used by
    /// `set_material_shading_model` / `set_material_foliage` so an
    /// EN-012 BG rebuild doesn't drop an EN-011 link. Mirrors the
    /// pattern in `set_material_texture_array`.
    fn resolve_probe_view_for_material(&self, material: u32) -> wgpu::TextureView {
        match self.material_system.material_reflection_probe_handle(material) {
            Some(probe) if probe != 0 => {
                let idx = probe as usize - 1;
                self.planar_probes.get(idx)
                    .and_then(|p| p.as_ref())
                    .map(|p| p.color_view.clone())
                    .unwrap_or_else(|| self.material_system.default_black_view.clone())
            }
            _ => self.material_system.default_black_view.clone(),
        }
    }

    /// EN-011 — render every registered probe's RT for this frame.
    /// Called from `end_frame_with_scene` BEFORE the main material
    /// pass so the probe textures are ready when materials sample
    /// them.
    ///
    /// For each probe:
    ///   1. Build the mirrored PerView (camera reflected across the
    ///      probe's plane; same projection as the main camera).
    ///   2. Upload the mirrored PerView to the probe's UBO.
    ///   3. Begin a render pass against the probe's RT + depth (clear
    ///      colour to fog, depth to 1.0).
    ///   4. Walk material_system.commands and dispatch each non-
    ///      excluded draw with the mirrored per-view bind group.
    ///
    /// V1 cull list: excludes any material whose handle equals one
    /// of the materials linked to a probe (so the water plane itself
    /// doesn't reflect). Future revisions can expand this with a
    /// hardcoded foliage / particle bucket filter via the bucket
    /// metadata on each compiled pipeline.
    pub fn dispatch_planar_reflections(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &crate::scene::SceneGraph,
    ) {
        if self.planar_probes.iter().all(|p| p.is_none()) { return; }
        // Scene-graph nodes render into the probe too (they share the
        // Vertex3D layout and the scene material bind-group layout), so a
        // fully retained-mode game gets real water reflections as well.
        let scene_draws = scene.reflect_draw_list();
        if self.material_system.commands.is_empty()
            && self.model_draw_commands.is_empty()
            && scene_draws.is_empty() { return; }

        // EN-011 — lazily build the single-target reflection pipeline + buffers
        // used to render cached models (trees/house) into the probe with a
        // mirrored VP. Owned layouts: g0 dynamic per-draw model uniform, g1
        // sun/ambient; g2 reuses the scene material layout for base colour.
        const REFLECT_STRIDE: u64 = 256;
        const REFLECT_MAX_DRAWS: usize = 1024;
        if self.reflect_scene_pipeline.is_none() {
            let model_dyn_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reflect_model_dyn_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(128),
                    },
                    count: None,
                }],
            });
            let shadow_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
                binding, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            let light_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reflect_light_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false, min_binding_size: None,
                        },
                        count: None,
                    },
                    // Shadow cascades + comparison sampler so the mirrored
                    // scene is sun-shadowed like the real one (the probe
                    // previously rendered everything fully lit, which made
                    // water reflections disagree with the scene above them).
                    shadow_tex_entry(1),
                    shadow_tex_entry(2),
                    shadow_tex_entry(3),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reflect_scene_shader"),
                source: wgpu::ShaderSource::Wgsl(REFLECT_SCENE_WGSL.into()),
            });
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("reflect_scene_pl"),
                bind_group_layouts: &[Some(&model_dyn_layout), Some(&light_layout), Some(&self.scene_material_layout)],
                immediate_size: 0,
            });
            let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("reflect_scene_pipeline"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader, entry_point: Some("vs_reflect"),
                    buffers: &[Vertex3D::desc()], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader, entry_point: Some("fs_reflect"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None, ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT, depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(), bias: Default::default(),
                }),
                multisample: Default::default(), multiview_mask: None, cache: None,
            });
            let model_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reflect_model_buf"),
                size: REFLECT_STRIDE * REFLECT_MAX_DRAWS as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let model_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflect_model_bg"), layout: &model_dyn_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &model_buf, offset: 0, size: std::num::NonZeroU64::new(128),
                    }),
                }],
            });
            // sun_dir + sun_color + ambient + cam_pos + shadow_splits (5 vec4)
            // + 3 cascade mat4s = 80 + 192 = 272 bytes.
            let light_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reflect_light_buf"), size: 272,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let light_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflect_light_bg"), layout: &light_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                ],
            });
            self.reflect_scene_pipeline = Some(pipeline);
            self.reflect_model_buf = Some(model_buf);
            self.reflect_model_bg = Some(model_bg);
            self.reflect_light_buf = Some(light_buf);
            self.reflect_light_bg = Some(light_bg);
        }
        // Sun/ambient + shadow data for the reflection shading (same values
        // as the main pass, so the mirrored scene is lit AND shadowed like
        // the real one). shadow_splits.w carries the shadows-enabled flag —
        // in THIS struct .w is free (no LOD-bias tenant).
        {
            let ld = self.lighting_uniforms.light_dir;
            let lc = self.lighting_uniforms.light_color;
            let amb = self.lighting_uniforms.ambient;
            let cam = self.current_camera_pos;
            let sp = self.lighting_uniforms.shadow_cascade_splits;
            let shadows_flag = if self.shadow_map.enabled { 1.0 } else { 0.0 };
            let mut light_data = [0.0f32; 68];
            light_data[0..4].copy_from_slice(&[ld[0], ld[1], ld[2], ld[3]]);
            light_data[4..8].copy_from_slice(&[lc[0], lc[1], lc[2], 0.0]);
            light_data[8..12].copy_from_slice(&[amb[0], amb[1], amb[2], amb[3]]);
            light_data[12..16].copy_from_slice(&[cam[0], cam[1], cam[2], 0.0]);
            light_data[16..20].copy_from_slice(&[sp[0], sp[1], sp[2], shadows_flag]);
            let vps = &self.lighting_uniforms.shadow_cascade_vps;
            for c in 0..3 {
                for col in 0..4 {
                    for row in 0..4 {
                        light_data[20 + c * 16 + col * 4 + row] = vps[c][col][row];
                    }
                }
            }
            if let Some(buf) = &self.reflect_light_buf {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&light_data));
            }
        }

        // Build the V1 exclude set: every material linked to any
        // probe. The water material itself shouldn't appear in its
        // own reflection (it'd black-on-black self-occlude the
        // surface).
        let mut excluded: std::collections::HashSet<material_system::MaterialHandle> =
            std::collections::HashSet::new();
        for (i, probe_link) in self.material_system.material_reflection_probe.iter().enumerate() {
            if probe_link.is_some() {
                excluded.insert((i + 1) as material_system::MaterialHandle);
            }
        }

        // Cache main-pass per-view inputs once outside the loop.
        let main_view = self.current_view_matrix;
        let proj      = self.current_proj_matrix;
        let cam_pos   = self.current_camera_pos;

        // Snapshot the existing PerView uniforms by reconstructing
        // the same struct material_system_begin_frame writes — we
        // need a fresh copy per probe to swap view/view_proj.
        let base_per_view = material_system::PerViewUniforms {
            view:           main_view,
            proj,
            view_proj:      self.current_vp_matrix,
            // EN-022 fix: velocity reference (prev unjittered VP +
            // current jitter), so material shaders computing
            // `prev_view_proj * world` get true zero velocity on
            // static geometry instead of TAA jitter-delta noise.
            prev_view_proj: self.velocity_ref_vp,
            inv_proj:       self.current_inv_proj_matrix,
            camera_pos: [
                cam_pos[0], cam_pos[1], cam_pos[2],
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

        // Iterate probes by index — we mutate `material_system`'s
        // commands view while iterating, so collect the work first.
        let probe_count = self.planar_probes.len();
        for i in 0..probe_count {
            let (plane_y, normal, color_view, depth_view) = match &self.planar_probes[i] {
                Some(p) => (p.plane_y, p.normal, p.color_view.clone(), p.depth_view.clone()),
                None => continue,
            };
            let view_buf = match self.planar_probe_view_buffers[i].as_ref() {
                Some(b) => b, None => continue,
            };

            // Mirror the camera + recompute view_proj for the probe.
            let mirror_view = planar_reflection::mirrored_view(main_view, plane_y, normal);
            let mirror_cam  = planar_reflection::mirrored_camera_pos(cam_pos, plane_y, normal);

            // EN-011 V2 — oblique near-plane clip. Replace the
            // projection's near plane with the water plane (in
            // mirror-eye-space) so geometry below the plane is
            // clipped at the rasterizer instead of polluting the
            // reflection edge.
            //
            // World-space plane equation: `N · p + d_w = 0` with
            // kept-side `N · p + d_w > 0` (above water). For a
            // horizontal mirror at world y = plane_y with normal
            // +Y, d_w = -plane_y so the kept side is y > plane_y.
            // Transformed via mirror-view's inverse-transpose, the
            // eye-space plane defines the same physical half-space
            // (the side the kept geometry lives on after the
            // reflection has rolled it through the view).
            let d_w = -(normal[0] * 0.0 + normal[1] * plane_y + normal[2] * 0.0);
            let plane_world = [normal[0], normal[1], normal[2], d_w];
            let plane_eye   = planar_reflection::world_plane_to_eye_space(mirror_view, plane_world);
            let mirror_proj = planar_reflection::oblique_proj(proj, plane_eye);
            let mirror_vp   = mat4_multiply(mirror_proj, mirror_view);

            let mut per_view = base_per_view;
            per_view.view      = mirror_view;
            per_view.proj      = mirror_proj;
            per_view.view_proj = mirror_vp;
            per_view.inv_proj  = planar_reflection::inv_proj_for(mirror_proj);
            per_view.camera_pos[0] = mirror_cam[0];
            per_view.camera_pos[1] = mirror_cam[1];
            per_view.camera_pos[2] = mirror_cam[2];
            // prev_view_proj stays as the main camera's previous VP
            // — TAA reprojection isn't meaningful for the reflection
            // probe (we don't temporally accumulate it), so this is
            // benign.
            self.queue.write_buffer(view_buf, 0, bytemuck::bytes_of(&per_view));

            // EN-011 V2 — rebuild the per-probe PerView bind group with
            // the live env / BRDF / shadow views. V1 bound 1×1 stub
            // textures here, which left mirrored draws lit by a flat
            // grey IBL (no specular reflections, no sun shadow) — the
            // reflection painting the lit scene differently from the
            // main pass made the surface look "off". Rebuilding once
            // per probe per frame is cheap (a single bind-group
            // create); it also picks up any env hot-load that
            // happens between frames without needing explicit dirty
            // tracking on the probe side.
            //
            // The sky env (binding 1) and env_diffuse (binding 3)
            // default to the renderer's 1×1 grey fallback when no HDR
            // is loaded — same default the main pass uses, so the
            // reflection's IBL stays consistent pre/post
            // `load_env_from_hdr`. The sky_texture's view doesn't sit
            // on a struct field (it's owned by `sky_bind_group`), so
            // we build fresh views here each frame; that's a cheap
            // Arc bump on the underlying wgpu Texture.
            let sky_view_owned: Option<wgpu::TextureView> = self.sky_texture
                .as_ref()
                .map(|t| t.create_view(&Default::default()));
            let env_view: &wgpu::TextureView = sky_view_owned
                .as_ref()
                .unwrap_or(&self.scene_env_default_view);
            let diffuse_view_owned: Option<wgpu::TextureView> = self.env_diffuse_texture
                .as_ref()
                .map(|t| t.create_view(&Default::default()));
            let env_diffuse_view: &wgpu::TextureView = diffuse_view_owned
                .as_ref()
                .unwrap_or(&self.scene_env_default_view);

            let probe_view_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("planar_probe_per_view_bg_live"),
                layout: &self.material_system.layouts.per_view,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                    // env (specular) tex + sampler
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.env_sampler) },
                    // env diffuse tex
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(env_diffuse_view) },
                    // BRDF LUT tex + sampler
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.brdf_lut_view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.brdf_lut_sampler) },
                    // 3 shadow cascades — same depth views the main
                    // pass binds, so the reflection picks up sun
                    // shadows without re-rendering the cascades.
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                ],
            });

            // Write each cached-model draw's [mirror_mvp, model] into the
            // dynamic reflection uniform buffer up front (queue writes
            // happen-before the encoded pass), and record the draw list.
            // Scene-graph nodes append after the cached models in the same
            // slot space — creation order decides who survives the cap, so
            // games should create hero geometry before filler (grass).
            let mut reflect_draws: Vec<(u64, usize, u32)> = Vec::new();
            let mut node_slots: Vec<(usize, u32)> = Vec::new();
            if let Some(model_buf) = &self.reflect_model_buf {
                for cmd in self.model_draw_commands.iter() {
                    let slot = reflect_draws.len();
                    if slot >= REFLECT_MAX_DRAWS { break; }
                    let mirror_mvp = mat4_multiply(mirror_vp, cmd.model);
                    let mut data = [0u8; 128];
                    data[0..64].copy_from_slice(bytemuck::bytes_of(&mirror_mvp));
                    data[64..128].copy_from_slice(bytemuck::bytes_of(&cmd.model));
                    self.queue.write_buffer(model_buf, slot as u64 * REFLECT_STRIDE, &data);
                    reflect_draws.push((cmd.cache_handle, cmd.mesh_idx, slot as u32));
                }
                for (i, (_vb, _ib, _ic, _bg, model)) in scene_draws.iter().enumerate() {
                    let slot = reflect_draws.len() + node_slots.len();
                    if slot >= REFLECT_MAX_DRAWS { break; }
                    let mirror_mvp = mat4_multiply(mirror_vp, *model);
                    let mut data = [0u8; 128];
                    data[0..64].copy_from_slice(bytemuck::bytes_of(&mirror_mvp));
                    data[64..128].copy_from_slice(bytemuck::bytes_of(model));
                    self.queue.write_buffer(model_buf, slot as u64 * REFLECT_STRIDE, &data);
                    node_slots.push((i, slot as u32));
                }
            }

            // Clear the probe to transparent black. Geometry fragments write
            // alpha 1, so the water shader can blend the probe over its analytic
            // sky by alpha (a=0 → no reflected geometry → show the sky dome).
            let clear_color = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let view_bg = &probe_view_bg;
            let cache   = &self.model_gpu_cache;
            let mat_sys = &self.material_system;
            let refl_pipeline = self.reflect_scene_pipeline.as_ref();
            let refl_model_bg = self.reflect_model_bg.as_ref();
            let refl_light_bg = self.reflect_light_bg.as_ref();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_planar_reflection_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear_color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                mat_sys.dispatch_with_view(
                    &mut pass, view_bg,
                    |handle| !excluded.contains(&handle),
                    // EN-011 V2 — swap to each material's sibling
                    // pipeline with cull_mode flipped Front→Back.
                    // Reflection mirrors world-space, which inverts
                    // triangle winding; without the flip, single-
                    // sided opaque geometry renders inside-out in
                    // the probe's RT. Translucent / cutout materials
                    // have `reflection_pipeline = None` and gracefully
                    // fall back to the main pipeline (no cull change
                    // needed since they're already double-sided).
                    true,
                    |handle, idx| {
                        if let Some(Some(meshes)) = cache.get(&handle) {
                            if idx < meshes.len() {
                                let mesh = &meshes[idx];
                                return Some((&mesh.vb, &mesh.ib, mesh.index_count));
                            }
                        }
                        None
                    },
                );

                // Render cached models (trees/house/foliage) mirrored into the
                // probe so the water reflects the actual world, not just an
                // analytic sky. Single-target lit pipeline; cutout alpha is
                // discarded so foliage reflects its real shape.
                if let (Some(rp), Some(rmbg), Some(rlbg)) =
                    (refl_pipeline, refl_model_bg, refl_light_bg)
                {
                    if !reflect_draws.is_empty() || !node_slots.is_empty() {
                        pass.set_pipeline(rp);
                        pass.set_bind_group(1, rlbg, &[]);
                        for (handle, midx, slot) in &reflect_draws {
                            if let Some(Some(meshes)) = cache.get(handle) {
                                if *midx < meshes.len() {
                                    let mesh = &meshes[*midx];
                                    pass.set_bind_group(0, rmbg, &[*slot * REFLECT_STRIDE as u32]);
                                    pass.set_bind_group(2, &mesh.material_bg, &[]);
                                    pass.set_vertex_buffer(0, mesh.vb.slice(..));
                                    pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
                                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                                }
                            }
                        }
                        // Scene-graph nodes: same pipeline — node geometry is
                        // Vertex3D and node material bind groups share the
                        // scene material layout the pipeline's g2 expects.
                        for (i, slot) in &node_slots {
                            let (vb, ib, index_count, mat_bg, _model) = &scene_draws[*i];
                            pass.set_bind_group(0, rmbg, &[*slot * REFLECT_STRIDE as u32]);
                            pass.set_bind_group(2, *mat_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*index_count, 0, 0..1);
                        }
                    }
                }
            }
        }
    }

    /// Sync PerFrame + PerView uniforms from current renderer state.
    /// FFI callers drive this from their frame boundary so `PerFrame.time`
    /// reflects the real process-uptime clock.
    pub fn material_system_begin_frame(&mut self, time_seconds: f32, delta_time: f32) {
        // Feed wind + time to the built-in scene shader (foliage sway). Set
        // here each frame so it's current before the per-frame lighting upload.
        self.lighting_uniforms.wind =
            [self.wind[0], self.wind[1], self.wind[2], time_seconds];
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
            wind: self.wind,
        };
        let per_view = material_system::PerViewUniforms {
            view:           self.current_view_matrix,
            proj:           self.current_proj_matrix,
            view_proj:      self.current_vp_matrix,
            // EN-022 fix: velocity reference — see the main PerView
            // build above.
            prev_view_proj: self.velocity_ref_vp,
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

