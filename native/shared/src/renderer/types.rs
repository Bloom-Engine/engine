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
    pub _pad:     [f32; 3],   // pad to 16-byte alignment (vec4 boundary)
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
}

#[derive(PartialEq, Clone, Copy)]
pub enum RenderMode {
    ScreenSpace,
    Mode2D,
    Mode3D,
}
