//! Uniform + vertex types used throughout the renderer.
//!
//! Pure POD data (`bytemuck::Pod`-derived structs) — no Renderer state,
//! no wgpu resource ownership, no behavior beyond tiny constructors /
//! VertexBufferLayout descriptors. Split out of the 11 500-line
//! renderer monolith so the wiring and the data types are separable.
//!
//! `pub` items that external modules import (`Vertex3D`,
//! `SceneMaterialUniforms`) are re-exported from `renderer/mod.rs`
//! with `pub use types::*;` so their public paths
//! (`crate::renderer::Vertex3D`, etc.) stay stable.

use crate::renderer::IDENTITY_MAT4;

// ============================================================
// Constants
// ============================================================

pub(super) const MAX_UNIFORM_SLOTS: usize = 8;

// ============================================================
// Vertex and Uniform types
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Uniforms2D {
    pub(super) screen_size: [f32; 2],
    pub(super) _pad: [f32; 2],
    pub(super) view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Uniforms3D {
    pub(super) mvp: [[f32; 4]; 4],
    pub(super) model: [[f32; 4]; 4],
    pub(super) prev_mvp: [[f32; 4]; 4],
    pub(super) model_tint: [f32; 4],
    /// x = joint-buffer base offset for this draw (added to vertex joint
    /// indices by the scene VS), y = 1.0 for skinned cached draws else
    /// 0.0, zw unused. Lets GPU-resident skinned models share the static
    /// cached-model path: the VB keeps RAW joint indices and the per-draw
    /// uniform carries the frame's pose offset instead.
    pub(super) misc: [f32; 4],
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

// Raised from 4/16: scenes were hard-capped at 16 point lights, the
// audit's top graphics blocker. Arrays stay in a uniform buffer so the
// cap raise works on every backend including WebGL2 (whose 16KB minimum
// UBO size this still fits: 256*32B + 8*32B + header < 9KB). Shaders
// loop only over the live count, so small scenes pay nothing. Per-pixel
// cost for genuinely large light counts is the follow-up (froxel
// clustering); this change removes the capability ceiling.
pub(crate) const MAX_DIR_LIGHTS: usize = 8;
pub(crate) const MAX_POINT_LIGHTS: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DirLight {
    pub(super) direction: [f32; 4],  // xyz + intensity
    pub(super) color: [f32; 4],      // rgb + _pad
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PointLight {
    pub(super) position: [f32; 4],   // xyz + range
    pub(super) color: [f32; 4],      // rgb + intensity
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LightingUniforms {
    pub(super) ambient: [f32; 4],                              // rgb + intensity
    pub(super) light_dir: [f32; 4],                             // xyz + intensity (legacy, kept for compat)
    pub(super) light_color: [f32; 4],                           // rgb + _pad (legacy)
    pub(super) dir_light_count: [f32; 4],                       // [count, 0, 0, 0]
    pub(super) dir_lights: [DirLight; MAX_DIR_LIGHTS],          // additional directional lights
    pub(super) point_light_count: [f32; 4],                     // [count, 0, 0, 0]
    pub(super) point_lights: [PointLight; MAX_POINT_LIGHTS],    // point lights
    /// Camera world-space position (xyz) + env intensity multiplier
    /// (w). Scene shader uses xyz to compute V = normalize(camera_pos
    /// - world_pos) for GGX specular, and multiplies w into every env
    /// sample so IBL stays in sync with the sky pass when the user
    /// scales their HDR. Written once per frame before the main pass.
    pub(super) camera_pos: [f32; 4],
    /// Cascaded shadow map: 3 light view-projection matrices (one per
    /// cascade). Scene shader selects the tightest cascade based on
    /// the fragment's view-space depth and projects through the
    /// corresponding matrix for shadow-map UV.
    pub(super) shadow_cascade_vps: [[[f32; 4]; 4]; 3],
    /// View-space Z split distances for cascade selection (xyz = split
    /// distances for cascades 0/1/2, w = unused). Fragment at depth z
    /// uses cascade i where z <= cascade_splits[i].
    pub(super) shadow_cascade_splits: [f32; 4],
    /// Camera view matrix — passed to the shader so the fragment shader
    /// can compute view-space Z for cascade selection without an extra
    /// buffer binding.
    pub(super) shadow_view_matrix: [[f32; 4]; 4],
    /// Wind for foliage sway in the built-in scene vertex shader:
    /// xy = wind direction in the XZ plane (magnitude scales sway),
    /// z = amplitude, w = elapsed time (seconds) for the sway phase.
    /// Appended last so existing field offsets stay stable.
    pub(super) wind: [f32; 4],
    /// Cloud deck for the built-in scene shader: x = shadow strength,
    /// y = deck height (m), z = feature scale, w = drift speed (m/s).
    /// Strength 0 = the scene ignores the clouds. Appended last so existing
    /// field offsets stay stable.
    pub(super) cloud: [f32; 4],
    /// x = delta_time (seconds). The scene VS needs LAST frame's wind offset to
    /// emit a correct motion vector for swaying foliage — without it TAA sees
    /// velocity 0 on every moving leaf and ghosts them. Appended last so existing
    /// field offsets stay stable.
    pub(super) frame_misc: [f32; 4],
}

impl LightingUniforms {
    pub(super) fn defaults() -> Self {
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
            wind: [0.0, 0.0, 0.0, 0.0],
            cloud: [0.0, 420.0, 0.0035, 8.0],
            frame_misc: [0.0; 4],
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

/// Per-instance data for materials compiled with `wants_instancing = true`.
/// Bound at vertex buffer slot 1, step_mode = Instance. Layout is fixed
/// at engine V1; future extensions can parameterise from a material desc.
///
/// Per-vertex attributes use shader_location 0..6. Per-instance
/// attributes start at shader_location 7. The TS-side flat layout is 9
/// floats per instance (pos.xyz, rot_y, scale, tint.rgba); the Rust
/// side pads each instance to 12 floats so the GPU stride matches the
/// 48-byte vec4-aligned layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData3D {
    pub position: [f32; 3],   // world-space position
    pub rot_y:    f32,        // Y-axis rotation in radians
    pub scale:    f32,        // uniform scale multiplier (1.0 = no scale)
    pub tint:     [f32; 4],   // RGBA tint multiplier (1,1,1,1 = no tint)
    /// EN-026 — was pure padding to the 16-byte boundary; now carried to the
    /// shader as `@location(11) instance_extra: vec3<f32>`. The three floats
    /// were already being uploaded, so exposing them costs nothing: no stride
    /// change, no extra bandwidth. Particles use them for (atlas frame,
    /// velocity-stretch length, random seed); anything else can leave them 0
    /// and simply not declare location 11 — a vertex buffer may carry
    /// attributes the shader does not consume.
    pub extra:    [f32; 3],
}

impl InstanceData3D {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 7,  format: wgpu::VertexFormat::Float32x3 },  // position
                wgpu::VertexAttribute { offset: 12, shader_location: 8,  format: wgpu::VertexFormat::Float32 },    // rot_y
                wgpu::VertexAttribute { offset: 16, shader_location: 9,  format: wgpu::VertexFormat::Float32 },    // scale
                wgpu::VertexAttribute { offset: 20, shader_location: 10, format: wgpu::VertexFormat::Float32x4 },  // tint
                wgpu::VertexAttribute { offset: 36, shader_location: 11, format: wgpu::VertexFormat::Float32x3 },  // extra (EN-026)
            ],
        }
    }
}

// ============================================================
// Draw call tracking
// ============================================================

pub(super) struct DrawCall2D {
    pub(super) texture_idx: u32,
    pub(super) uniform_idx: u32,
    pub(super) index_start: u32,
}

pub(super) struct DrawCall3D {
    pub(super) texture_idx: u32,
    pub(super) index_start: u32,
    /// First vertex of this segment in `vertices_3d` — lets the shadow
    /// pass lazily scan primitive-only segments for bounds.
    pub(super) vertex_start: u32,
    /// World AABB of the segment's content. Non-skinned verts contribute
    /// their (already world-space) positions; skinned model draws
    /// contribute the union of their joint-transformed rest AABBs (a
    /// rigorous conservative bound: a skinned vertex is a convex blend
    /// of per-joint transforms of the same rest position).
    /// `wmin[0] > wmax[0]` = not yet computed.
    pub(super) wmin: [f32; 3],
    pub(super) wmax: [f32; 3],
    /// Segment contains skinned (animated) vertices → its rendered
    /// output changes every frame, so shadow caching must treat it as
    /// always-dirty for the cascades it touches.
    pub(super) has_skinned: bool,
    /// FNV-1a over the non-skinned vertex positions appended to this
    /// segment — a cheap content identity so static immediate geometry
    /// (e.g. pickups re-submitted identically every frame) doesn't
    /// dirty the shadow cascades it sits in.
    pub(super) content_hash: u64,
    /// True when bounds/hash were maintained inline (model draws).
    /// False → primitive-only segment, scanned on demand.
    pub(super) bounded: bool,
}

pub(super) const FNV_OFFSET: u64 = 0xcbf29ce484222325;

#[inline]
pub(super) fn fnv1a_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(PartialEq, Clone, Copy)]
pub enum RenderMode {
    ScreenSpace,
    Mode2D,
    Mode3D,
}

// ============================================================
// Pass / compute uniform params (moved out of mod.rs, EN-052)
// ============================================================
// Pure POD uniform structs for the sky, GI, SSR, post, and PT
// passes plus the card-ortho helper. Fields are pub(super) so the
// renderer and its pass submodules can build them by literal.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PrefilterUniforms {
    /// x = roughness (∈ [0, 1]), y = sample count, zw = mip resolution
    pub(super) params: [f32; 4],
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
pub(super) struct SkyUniforms {
    /// Camera right vector × tan(fovy/2) × aspect — pre-scaled so the
    /// fragment shader just multiplies by NDC.x to get the horizontal
    /// offset from the forward direction.
    /// `.w` = camera world position X (see below).
    pub(super) right: [f32; 4],
    /// Camera up vector × tan(fovy/2). `.w` = camera world position Y.
    pub(super) up: [f32; 4],
    /// Camera forward unit vector. `.w` = camera world position Z.
    ///
    /// The camera POSITION rides in the spare `.w` lanes of the three basis
    /// vectors. The sky pass never needed it — a sky at infinity only cares
    /// which way you are looking — but the cloud deck is anchored in world
    /// space so that its shadow lands under it, and that makes the camera's
    /// position load-bearing. Packing it here keeps the bind group unchanged.
    pub(super) forward: [f32; 4],
    /// x = intensity multiplier, y = elapsed seconds (the cloud deck's clock);
    /// zw padding.
    pub(super) intensity: [f32; 4],
    /// Cloud deck: x = shadow strength, y = deck height (m), z = feature scale
    /// (noise units per metre), w = drift speed (m/s). Same vec4 the world
    /// materials get, so the sky and the ground cannot disagree.
    pub(super) cloud: [f32; 4],
    /// xy = wind direction in the XZ plane — the clouds drift downwind, the
    /// same way the grass is leaning.
    pub(super) wind: [f32; 4],
}

// EN-005 Phase 2 — uniforms for the procedural sky path.
//
// `SkyViewParams` drives the sky-view LUT compute shader (recomputed
// when the sun moves). `SunUniforms` drives the per-frame sun-disk
// composite in the procedural sky fragment shader.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SkyViewParams {
    /// xyz = sun direction (world space, unit), w = sun intensity scale.
    pub(super) sun: [f32; 4],
    /// x = rayleigh density mult, y = mie density mult, z = ground albedo, w unused.
    pub(super) knobs: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SunUniforms {
    /// xyz = sun direction (unit), w = sun intensity.
    pub(super) sun: [f32; 4],
    /// x = sun angular radius (rad), y = limb darkening (0..1), zw unused.
    pub(super) params: [f32; 4],
}

// EN-005 V2 — aerial-perspective compute uniforms.
//
// Driven each frame from the renderer (camera + sun + atmosphere
// knobs). Layout must mirror `AerialParams` in
// AERIAL_PERSPECTIVE_SHADER_WGSL exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AerialParams {
    /// xyz = camera world position (engine units, metres assumed),
    /// w = max distance the LUT covers (km).
    pub(super) cam_pos: [f32; 4],
    /// World→clip inverse, used per-voxel to reconstruct view rays
    /// from NDC.
    pub(super) inv_vp: [[f32; 4]; 4],
    /// xyz = sun direction (unit), w = sun intensity scalar.
    pub(super) sun: [f32; 4],
    /// x = rayleigh density mult, y = mie density mult, zw unused.
    pub(super) knobs: [f32; 4],
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct HizLinearizeParams {
    /// xy = inv_size, z = proj[2][2], w = proj[3][2]
    pub(super) params: [f32; 4],
    /// xy = mip-0 size, zw unused
    pub(super) size: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct HizDownsampleParams {
    /// xy = dst-mip size, zw unused
    pub(super) size: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SsaoParams {
    /// xy = inv_size (1/half_w, 1/half_h), z = radius (world units),
    /// w = strength
    pub(super) params: [f32; 4],
    /// x = proj[0][0], y = proj[1][1], z = proj[2][0] (TAA jitter),
    /// w = proj[2][1] (TAA jitter). Column-major: proj[col][row].
    pub(super) proj_row01: [f32; 4],
    /// x = proj[2][2], y = proj[3][2], z = 1/proj[0][0], w = 1/proj[1][1]
    pub(super) proj_z: [f32; 4],
    /// Light direction in view space (xyz, w unused). For contact shadows.
    pub(super) light_dir_vs: [f32; 4],
    /// x = half-res width, y = half-res height, z = frame phase
    /// (`frame_index % 4`), w = "force-refresh" flag (non-zero on
    /// first few frames, resize, or any host-side history invalidation).
    pub(super) size: [u32; 4],
    /// x = temporal blend alpha (≈0.25 steady, 4-frame EMA).
    /// y = per-frame Halton-5 rotation of the direction basis
    /// (uncorrelated with TAA's Halton-2/3 pixel jitter).
    /// zw unused.
    pub(super) temporal: [f32; 4],
}

// ============================================================
// SSAO Bilateral Blur post-process
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SsaoBlurParams {
    /// xy = texel_size (of the half-res SSAO RT), z = depth_sigma, w = unused.
    pub(super) params: [f32; 4],
}

// ============================================================
// Depth of Field (DoF) post-process
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DofParams {
    /// x = focus_distance, y = aperture, z = max_blur_radius (UV), w = unused
    pub(super) params: [f32; 4],
    /// Inverse projection matrix — used to linearize depth.
    pub(super) inv_proj: [[f32; 4]; 4],
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
pub(super) struct MotionBlurParams {
    /// x = strength, y = max_blur (UV), zw = unused.
    pub(super) params: [f32; 4],
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
pub(super) struct SssParams {
    /// x = strength, y = width, z = falloff, w = unused.
    pub(super) params: [f32; 4],
}

// ============================================================
// SSGI (Screen-Space Global Illumination) post-process
// ============================================================

// Ticket 007a: per-pass uniform params for the screen-probe SSGI chain.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ProbePlaceParams {
    pub(super) inv_view: [[f32; 4]; 4],
    /// x = proj[0][0], y = proj[1][1], z = proj[2][0], w = proj[2][1]
    pub(super) proj_row01: [f32; 4],
    /// x = half_w, y = half_h, z = grid_w, w = grid_h
    pub(super) size: [u32; 4],
    /// x = frame_index, y = tile_size (16.0), zw unused
    pub(super) params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ProbeTraceParams {
    pub(super) view: [[f32; 4]; 4],
    pub(super) proj: [[f32; 4]; 4],
    pub(super) inv_view: [[f32; 4]; 4],
    pub(super) proj_row01: [f32; 4],
    pub(super) size: [u32; 4],
    /// x = frame_index, y = intensity, z = max_march_t (world units),
    /// w = firefly luma cap.
    pub(super) params: [f32; 4],
    /// Ticket 007b HW path — sun direction in world space (xyz) +
    /// intensity (w). Ignored by the SW-HiZ shader, consumed by HW + SDF.
    pub(super) sun_dir: [f32; 4],
    /// Sun colour (xyz), reserved (w). Used for analytical NdotL at hit.
    pub(super) sun_color: [f32; 4],
    /// Sky dome colour (xyz), reserved (w). Flat analytic sky for
    /// up-facing hits where the ray hits a surface with an up normal.
    pub(super) sky_color: [f32; 4],
    /// Ticket 014 V3 — clipmap origin (xyz) + full extent (w). Used
    /// by the SDF sphere-trace variant to map world-space march
    /// positions into clipmap sample UVs.
    pub(super) clipmap: [f32; 4],
    /// Ticket 014 V6/V13 — WSRC cascade cubes. Each entry is
    /// (origin xyz, extent w). Cascades are ordered near→far; miss
    /// paths pick the smallest cascade whose cube contains the
    /// ray-terminal position. `extent = 0.0` marks an unbaked
    /// cascade (shader falls through to the next one). HW + Hi-Z
    /// carry these for uniform-buffer size parity; Hi-Z ignores
    /// them.
    pub(super) wsrc_cascades: [[f32; 4]; 3],
}

/// PT-2 — fixed size of the kernel's texture binding array. Real texture
/// indices at or above this clamp to 0 (white); unused tail slots are
/// padded with the white texture so no PARTIALLY_BOUND feature is needed.
pub(super) const PT_MAX_TEXTURES: usize = 256;

/// Uniform for the path-tracing megakernel — WGSL mirror `PtParams` in
/// shaders/pt.rs. Field-for-field, 16-byte aligned, 672 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PtParamsCpu {
    pub(super) inv_vp: [[f32; 4]; 4],
    /// PT-3 — previous frame's unjittered VP (transposed like inv_vp)
    /// for temporal history reprojection in realtime mode.
    pub(super) prev_vp: [[f32; 4]; 4],
    /// xyz camera world pos, w unused.
    pub(super) cam_pos: [f32; 4],
    /// xyz unit vector toward the sun, w unused.
    pub(super) sun_dir: [f32; 4],
    /// rgb premultiplied by intensity, w unused.
    pub(super) sun_color: [f32; 4],
    /// rgb ambient-derived sky tint, w unused.
    pub(super) sky_color: [f32; 4],
    /// x/y = TRACE grid dims (half in realtime mode), z=frame_index,
    /// w=accum_count.
    pub(super) size: [u32; 4],
    /// x=mode (1 progressive / 2 realtime), y=max_bounces,
    /// z=point_light_count (≤16), w=debug view (BLOOM_PT_DEBUG).
    pub(super) cfg: [f32; 4],
    /// PT-3 half-res: x/y = full G-buffer dims. z = 1 → hybrid sun
    /// (shadow cascades instead of traced sun rays). w unused.
    pub(super) ext: [u32; 4],
    /// Raster shadow cascade VPs, transposed at upload like inv_vp.
    pub(super) shadow_vps: [[[f32; 4]; 4]; 3],
    /// 16 lights × (pos_range vec4, color_int vec4).
    pub(super) lights: [[f32; 4]; 32],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ProbeTemporalParams {
    /// x = alpha (EMA), y = force_refresh (1→alpha=1), z = grid_w (f32), w = grid_h (f32)
    pub(super) params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ProbeResolveParams {
    pub(super) inv_view: [[f32; 4]; 4],
    pub(super) proj_row01: [f32; 4],
    /// x = half_w, y = half_h, z = grid_w, w = grid_h
    pub(super) size: [u32; 4],
    /// x = tile_size (16.0), y = intensity, zw unused
    pub(super) params: [f32; 4],
}

/// On-GPU `ProbeHeader` layout (must match PROBE_HELPERS_WGSL's struct).
/// 32 bytes per probe.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ProbeHeaderCpu {
    pub(super) world_pos: [f32; 4],
    pub(super) normal: [f32; 4],
}

/// Ticket 013 V3 — CardCaptureParams. ortho_vp + base_color + emissive.
/// 96 bytes; we allocate 128 for uniform-alignment headroom.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CardCaptureParams {
    pub(super) ortho_vp: [[f32; 4]; 4],
    pub(super) base_color: [f32; 4],  // rgb = factor, w = has_base_texture (0/1)
    pub(super) emissive: [f32; 4],    // rgb = emissive_factor, w = has_emissive_texture (0/1)
}

/// Ticket 013 V2 — orthographic projection for the 6 signed axes.
/// `face_axis` encoding:
///   0 → +X, 1 → -X, 2 → +Y, 3 → -Y, 4 → +Z, 5 → -Z.
/// For each face we pick the two orthogonal AABB axes and map them
/// to clip-space [-1, +1]. The ±pair for each axis differ only in
/// the sign of the "u" clip axis so that when the HW shader picks
/// axis N at hit and projects the hit into card UV, the UV lines up
/// with the mesh geometry as seen FROM that face.
pub(super) fn build_card_ortho_v2(face_axis: u32, bmin: [f32; 3], bmax: [f32; 3]) -> [[f32; 4]; 4] {
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
pub(super) struct SdfBakeParams {
    pub(super) aabb_min: [f32; 4],
    pub(super) aabb_max: [f32; 4],
    /// x = triangle_count, y = sdf_resolution (32), zw unused
    pub(super) counts: [u32; 4],
}

/// Fullscreen-lag fix — an in-flight amortized scene-SDF-clipmap bake.
/// A job bakes `SCENE_SDF_CLIPMAP_LAYERS_PER_FRAME` voxel Z-layers per
/// frame into the staging texture; when the last slice lands the staging
/// contents are copied over the live clipmap and the origin flips
/// atomically, so traces never observe a half-baked field. The bind
/// group keeps the job's transient buffers alive.
pub(super) struct SdfClipmapBakeJob {
    pub(super) origin: [f32; 3],
    pub(super) aabb_min: [f32; 4],
    pub(super) aabb_max: [f32; 4],
    pub(super) uniform: wgpu::Buffer,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) next_z: u32,
}

/// Ticket 014 V6 — uniform for `WSRC_BAKE_WGSL`. Analytic sun × shadow
/// + analytic sky computed per probe-octel. Shadow VPs + splits +
/// flags mirror CARD_LIGHT_WGSL so the shader can re-use the same
/// cascade-sampling helper.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct WsrcBakeParams {
    pub(super) sun_dir: [f32; 4],
    pub(super) sun_color: [f32; 4],
    pub(super) sky_color: [f32; 4],
    /// xyz = WSRC cube origin, w = full extent.
    pub(super) grid: [f32; 4],
    /// Cascade 0..2 VPs. Only cascade 2 is actually sampled by V6
    /// (widest cascade covers the whole 120 m cube), but we carry
    /// all three to keep the uniform layout identical to the card-
    /// lighting params.
    pub(super) shadow_vps: [[[f32; 4]; 4]; 3],
    /// xyz = view-space split distances (unused by V6 because it
    /// always samples cascade 2), w = 0.
    pub(super) shadow_splits: [f32; 4],
    /// x = shadow bias, y = shadows_enabled (0/1), zw unused.
    pub(super) flags: [f32; 4],
    /// EN-023 — xyz = scene-average albedo (mean of card-instance flat
    /// albedos) for the SW bake's ground-bounce term; w unused. The HW
    /// bake ignores it (it traces real geometry).
    pub(super) ground_albedo: [f32; 4],
}

/// Uniform struct for the card-lighting compute pass. Matches
/// CARD_LIGHT_WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CardLightParams {
    pub(super) sun_dir: [f32; 4],
    pub(super) sun_color: [f32; 4],
    pub(super) sky_color: [f32; 4],
    /// [atlas_size, slot_size, slots_per_row, active_slot_count]
    pub(super) atlas_info: [u32; 4],
    /// Shadow cascade VPs (3× mat4) — identity when shadows disabled.
    pub(super) shadow_vps: [[[f32; 4]; 4]; 3],
    /// xyz = view-space split distances for cascades 0/1/2, w = 0.
    pub(super) shadow_splits: [f32; 4],
    /// Camera view matrix — needed to convert card-texel world pos
    /// into view-space Z for cascade selection.
    pub(super) view_matrix: [[f32; 4]; 4],
    /// x = shadow bias, y = shadows_enabled (0/1), zw unused.
    pub(super) flags: [f32; 4],
}

/// Ticket 013 V3 — per-slot metadata consumed by `card_light_pass`.
/// Baked at capture time; carries enough state for the lighting
/// shader to reconstruct each texel's world-space position and query
/// the shadow cascade at that point. 128 bytes per slot × 4096 slots
/// = 512 KB — fits comfortably.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CardSlotMetaCpu {
    /// xyz = world-space card-face normal, w = signed axis (0..6 as f32).
    pub(super) normal_ws: [f32; 4],
    /// Object-space AABB min (xyz) + padding (w).
    pub(super) aabb_min: [f32; 4],
    /// Object-space AABB max (xyz) + padding (w).
    pub(super) aabb_max: [f32; 4],
    /// Mesh's world transform. Multiplied into the object-space
    /// card-plane position to land in world space for shadow lookup.
    pub(super) transform: [[f32; 4]; 4],
}

/// Ticket 007b — per-TLAS-instance GI shading input. Indexed by the
/// hit's `instance_custom_data` in the HW trace shader. Layout must
/// match the `InstanceGIData` struct in SSGI_PROBE_TRACE_HW_WGSL.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct InstanceGiDataCpu {
    /// Flat albedo (per-mesh, pre-baked from material base-color).
    /// Used as a fallback when `card_slot.w < 0` (no card captured).
    pub(super) albedo: [f32; 3],
    /// Scalar emissive luminance — multiplied by `albedo` at the hit.
    pub(super) emissive_luma: f32,
    /// Flat world-space mesh normal. Rough — averaged over vertex
    /// normals at BLAS build time. Used when the card atlas is
    /// unavailable; ticket 013's textured path still drives lighting
    /// from this flat normal but multiplies the sampled albedo in.
    pub(super) normal_ws: [f32; 3],
    pub(super) _pad0: f32,
    /// Ticket 013 — card slot for textured hit shading.
    /// `card_slot.xy` = atlas slot coord (0..CARD_SLOTS_PER_ROW).
    /// `card_slot.z` = dominant axis (0=X, 1=Y, 2=Z).
    /// `card_slot.w` = flag (1.0 = card captured, 0.0 = no card → fall
    /// back to `albedo` flat value).
    pub(super) card_slot: [f32; 4],
    /// Object-space AABB min (xyz) + unused pad (w). The HW paths
    /// transform hits into object space (hit.world_to_object) and
    /// compare against THESE — do not world-ify them.
    pub(super) card_aabb_min: [f32; 4],
    /// Object-space AABB max (xyz) + unused pad (w).
    pub(super) card_aabb_max: [f32; 4],
    /// EN-023 — WORLD-space AABB min/max. The SDF trace has no
    /// world_to_object (it marches a world-space clipmap), so its
    /// broad-phase compares the world hit against these. With the old
    /// object-space-only bounds, every transformed instance fell
    /// through to the flat-gray analytic fallback — zero colored
    /// bounce on non-RT adapters (round-2 audit F4).
    pub(super) world_aabb_min: [f32; 4],
    pub(super) world_aabb_max: [f32; 4],
    /// PT-2 — geometry window into the PT megabuffers + texture id.
    /// x = first vertex (Vertex3D-stride slot) in pt_geo_vertices,
    /// y = first index in pt_geo_indices, z = index count,
    /// w = albedo texture index (renderer texture store; 0 = white).
    /// z == 0 marks "no geometry window" (kernel falls back to the
    /// flat normal + card albedo, i.e. PT-1 behaviour).
    pub(super) geo: [u32; 4],
    /// PT-2 — x = roughness, y = metalness, z/w unused.
    pub(super) mat_params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SsrTemporalParams {
    /// x = blend_alpha (0.1), yzw unused
    pub(super) params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SsrParams {
    pub(super) inv_proj: [[f32; 4]; 4],
    pub(super) proj: [[f32; 4]; 4],
    /// x=strength, y=max_dist, z=n_steps, w=frame index
    pub(super) params: [f32; 4],
    /// EN-021 — view→world rotation (transpose of the view 3×3) for the
    /// env-miss fallback's direction lookup.
    pub(super) inv_view_rot: [[f32; 4]; 4],
    /// EN-021 — x = env max LOD, y = env intensity, zw unused.
    pub(super) params2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SceneComposeParams {
    pub(super) misc: [f32; 4],
    pub(super) inv_vp: [[f32; 4]; 4],
    pub(super) fog_color_density: [f32; 4],
    pub(super) fog_params: [f32; 4],
    pub(super) sun_shaft_uv_strength: [f32; 4],
    pub(super) sun_shaft_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TaaParams {
    /// x = blend factor (current-frame weight), yzw padding.
    pub(super) params: [f32; 4],
    pub(super) inv_vp: [[f32; 4]; 4],
    pub(super) prev_vp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct UpscaleParams {
    /// x = mode (0 = bilinear, 1 = Catmull-Rom), yzw padding.
    pub(super) params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RcasParams {
    /// x = sharpen strength (0 off, 1 max), yzw padding.
    pub(super) params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ExposureParams {
    pub(super) params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BloomParams {
    /// xy = source texel size, z = filter radius (upsample),
    /// w = HDR threshold (downsample-threshold variant).
    pub(super) params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CompositeParams {
    /// x = tonemap kind (0 ACES / 1 AgX), y = auto-exposure toggle,
    /// z = manual exposure, w = auto-exposure target key.
    pub(super) params: [f32; 4],
    /// Filmic-look knobs — see WGSL comment.
    /// x = chromatic-aberration strength, y = vignette strength,
    /// z = vignette softness, w = grain strength.
    pub(super) filmic: [f32; 4],
    /// x = grain seed (frame index, animates the noise),
    /// y = sharpen strength, zw padding.
    pub(super) misc: [f32; 4],
}
