//! Shadow mapping for Bloom Engine.
//!
//! Implements directional light shadow mapping with PCF (Percentage-Closer Filtering).
//! The shadow system renders the scene from the light's perspective into a depth texture,
//! then samples it during the main pass to determine shadowed areas.

use crate::renderer::{Vertex3D, IDENTITY_MAT4};

/// Shadow map configuration.
pub const SHADOW_MAP_SIZE: u32 = 2048;
pub const SHADOW_NEAR: f32 = 0.1;
pub const SHADOW_FAR: f32 = 100.0;
pub const SHADOW_EXTENT: f32 = 30.0; // orthographic extent in world units

/// Depth-only shader for shadow pass.
pub const SHADOW_SHADER: &str = "
struct ShadowUniforms {
    light_vp: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> shadow_u: ShadowUniforms;

struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
};

@vertex
fn vs_shadow(in: ShadowVertexInput) -> @builtin(position) vec4<f32> {
    let world_pos = shadow_u.model * vec4<f32>(in.position, 1.0);
    return shadow_u.light_vp * world_pos;
}
";

/// Uniform data for the shadow pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    pub light_vp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

/// Shadow map resources.
pub struct ShadowMap {
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform_layout: wgpu::BindGroupLayout,
    pub light_vp: [[f32; 4]; 4],
    pub enabled: bool,
}

impl ShadowMap {
    pub fn new(device: &wgpu::Device, vertex_layout: wgpu::VertexBufferLayout<'static>) -> Self {
        // Shadow depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_depth"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Comparison sampler for PCF
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind group layout for sampling shadow map in the main pass
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_sample_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_sample_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Shadow pass uniform layout
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_uniform_layout"),
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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform_buf"),
            size: std::mem::size_of::<ShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_uniform_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Shadow depth-only pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: None, // depth only
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
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            depth_texture,
            depth_view,
            sampler,
            bind_group_layout,
            bind_group,
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            uniform_layout,
            light_vp: IDENTITY_MAT4,
            enabled: false,
        }
    }

    /// Compute the light view-projection matrix for a directional light.
    pub fn compute_light_vp(&mut self, light_dir: [f32; 3], center: [f32; 3]) {
        // Light "position" = center - light_dir * distance
        let dist = SHADOW_FAR * 0.5;
        let len = (light_dir[0]*light_dir[0] + light_dir[1]*light_dir[1] + light_dir[2]*light_dir[2]).sqrt();
        let d = if len > 1e-6 {
            [light_dir[0]/len, light_dir[1]/len, light_dir[2]/len]
        } else {
            [0.0, 1.0, 0.0]
        };

        let light_pos = [
            center[0] - d[0] * dist,
            center[1] - d[1] * dist,
            center[2] - d[2] * dist,
        ];

        let view = crate::renderer::mat4_look_at(light_pos, center, [0.0, 1.0, 0.0]);
        let proj = crate::renderer::mat4_ortho(
            -SHADOW_EXTENT, SHADOW_EXTENT,
            -SHADOW_EXTENT, SHADOW_EXTENT,
            SHADOW_NEAR, SHADOW_FAR,
        );
        self.light_vp = crate::renderer::mat4_multiply(proj, view);
    }

    /// Enable shadow mapping.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable shadow mapping.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}
