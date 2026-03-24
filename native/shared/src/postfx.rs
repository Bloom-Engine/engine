//! Post-processing effects for Bloom Engine.
//!
//! Implements a render-to-texture pipeline with:
//! - Object outlines (selected/hovered objects)
//! - Screen-space ambient occlusion (SSAO)
//! - Final composite pass
//!
//! The pipeline renders the scene to offscreen textures (color + depth + object ID),
//! then applies fullscreen post-processing passes before presenting to screen.

use wgpu::util::DeviceExt;

/// Fullscreen quad vertex shader (shared by all post-fx passes).
const FULLSCREEN_VERT: &str = "
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    // Triangle strip covering screen: 3 vertices, no vertex buffer needed
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
";

/// Outline detection shader — Sobel edge detection on object ID buffer.
const OUTLINE_FRAG: &str = "
struct OutlineParams {
    color_selected: vec4<f32>,    // rgba
    color_hovered: vec4<f32>,     // rgba
    thickness: vec4<f32>,         // [thickness, glow, pulse_time, 0]
    screen_size: vec4<f32>,       // [width, height, 0, 0]
};

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var object_id_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: OutlineParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@fragment
fn fs_outline(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_color, tex_sampler, in.uv);
    let center_id = textureSample(object_id_tex, tex_sampler, in.uv).r;

    let pixel = vec2<f32>(1.0 / params.screen_size.x, 1.0 / params.screen_size.y);
    let t = params.thickness.x;

    // Sample neighbors for edge detection
    var edge = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let offset = vec2<f32>(f32(dx), f32(dy)) * pixel * t;
            let neighbor_id = textureSample(object_id_tex, tex_sampler, in.uv + offset).r;
            if (abs(neighbor_id - center_id) > 0.001) {
                edge += 1.0;
            }
        }
    }

    // Normalize edge strength
    edge = min(edge / 3.0, 1.0);

    // Apply outline color if this pixel is on an edge
    if (edge > 0.01) {
        // Determine if selected or hovered based on the ID value
        let outline_color = params.color_selected;
        // Pulse effect
        let pulse = 0.7 + 0.3 * sin(params.thickness.z * 3.0);
        let glow = params.thickness.y * pulse;
        return mix(color, outline_color, edge * (0.8 + glow * 0.2));
    }

    return color;
}
";

/// SSAO fragment shader — screen-space ambient occlusion.
const SSAO_FRAG: &str = "
struct SSAOParams {
    screen_size: vec4<f32>,   // [width, height, 0, 0]
    radius: vec4<f32>,        // [radius, bias, intensity, 0]
};

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> ssao_params: SSAOParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

@fragment
fn fs_ssao(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, tex_sampler, in.uv).r;
    if (depth >= 1.0) { return vec4<f32>(1.0); } // sky

    let pixel = vec2<f32>(1.0 / ssao_params.screen_size.x, 1.0 / ssao_params.screen_size.y);
    let radius = ssao_params.radius.x;
    let bias = ssao_params.radius.y;
    let intensity = ssao_params.radius.z;

    // Simple depth-based AO: compare depth with neighbors
    var occlusion = 0.0;
    let samples = 8;
    for (var i = 0; i < samples; i++) {
        let angle = f32(i) * 0.785398; // PI/4 increments
        let r = radius * (f32(i) + 1.0) / f32(samples);
        let offset = vec2<f32>(cos(angle), sin(angle)) * pixel * r;

        // Add noise based on position
        let noise = hash(in.uv * ssao_params.screen_size.xy + vec2<f32>(f32(i)));
        let rotated_offset = vec2<f32>(
            offset.x * cos(noise * 6.28) - offset.y * sin(noise * 6.28),
            offset.x * sin(noise * 6.28) + offset.y * cos(noise * 6.28),
        );

        let sample_depth = textureSample(depth_tex, tex_sampler, in.uv + rotated_offset).r;
        let diff = depth - sample_depth;
        if (diff > bias && diff < radius * 0.01) {
            occlusion += 1.0;
        }
    }

    let ao = 1.0 - (occlusion / f32(samples)) * intensity;
    return vec4<f32>(ao, ao, ao, 1.0);
}
";

/// Composite shader — combines scene color with AO.
const COMPOSITE_FRAG: &str = "
@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var ao_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_color, tex_sampler, in.uv);
    let ao = textureSample(ao_tex, tex_sampler, in.uv).r;
    return vec4<f32>(color.rgb * ao, color.a);
}
";

// ============================================================
// Outline parameters
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OutlineParams {
    pub color_selected: [f32; 4],
    pub color_hovered: [f32; 4],
    pub thickness: [f32; 4],  // [thickness, glow, pulse_time, 0]
    pub screen_size: [f32; 4],
}

impl Default for OutlineParams {
    fn default() -> Self {
        Self {
            color_selected: [0.2, 0.5, 1.0, 1.0],  // blue
            color_hovered: [0.8, 0.8, 0.2, 1.0],    // yellow
            thickness: [2.0, 0.3, 0.0, 0.0],
            screen_size: [1280.0, 720.0, 0.0, 0.0],
        }
    }
}

// ============================================================
// Post-FX Pipeline
// ============================================================

pub struct PostFxPipeline {
    // Offscreen render targets
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub object_id_texture: wgpu::Texture,
    pub object_id_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,

    // Outline pass
    pub outline_pipeline: wgpu::RenderPipeline,
    pub outline_bind_group: wgpu::BindGroup,
    pub outline_params_buffer: wgpu::Buffer,
    pub outline_params: OutlineParams,

    // Common
    pub sampler: wgpu::Sampler,
    pub enabled: bool,
    pub width: u32,
    pub height: u32,

    // Selected node handles for outline rendering
    pub selected_handles: Vec<f64>,
    pub hovered_handle: f64,
    pub time: f32,
}

impl PostFxPipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        let (color_texture, color_view) = create_color_target(device, width, height, surface_format);
        let (object_id_texture, object_id_view) = create_color_target(device, width, height, wgpu::TextureFormat::R32Float);
        let (depth_texture, depth_view) = create_depth_target(device, width, height);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("postfx_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let outline_params = OutlineParams {
            screen_size: [width as f32, height as f32, 0.0, 0.0],
            ..Default::default()
        };
        let outline_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("outline_params"),
            contents: bytemuck::bytes_of(&outline_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Outline bind group layout
        let outline_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("outline_bg_layout"),
            entries: &[
                bgl_texture(0, wgpu::TextureSampleType::Float { filterable: true }),
                bgl_texture(1, wgpu::TextureSampleType::Float { filterable: true }),
                bgl_sampler(2),
                bgl_uniform(3),
            ],
        });

        let outline_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outline_bg"),
            layout: &outline_bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&object_id_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: outline_params_buffer.as_entire_binding() },
            ],
        });

        // Outline pipeline
        let outline_shader_src = format!("{}\n{}", FULLSCREEN_VERT, OUTLINE_FRAG);
        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline_shader"),
            source: wgpu::ShaderSource::Wgsl(outline_shader_src.into()),
        });

        let outline_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline_pipeline_layout"),
            bind_group_layouts: &[&outline_bg_layout],
            push_constant_ranges: &[],
        });

        let outline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline_pipeline"),
            layout: Some(&outline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &outline_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &outline_shader,
                entry_point: Some("fs_outline"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            color_texture,
            color_view,
            object_id_texture,
            object_id_view,
            depth_texture,
            depth_view,
            outline_pipeline,
            outline_bind_group,
            outline_params_buffer,
            outline_params,
            sampler,
            enabled: false,
            width,
            height,
            selected_handles: Vec::new(),
            hovered_handle: 0.0,
            time: 0.0,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, surface_format: wgpu::TextureFormat) {
        if width == self.width && height == self.height { return; }
        *self = Self::new(device, width, height, surface_format);
    }

    pub fn set_selected(&mut self, handles: Vec<f64>) {
        self.selected_handles = handles;
    }

    pub fn set_hovered(&mut self, handle: f64) {
        self.hovered_handle = handle;
    }

    pub fn update(&mut self, queue: &wgpu::Queue, dt: f32) {
        self.time += dt;
        self.outline_params.thickness[2] = self.time;
        queue.write_buffer(&self.outline_params_buffer, 0, bytemuck::bytes_of(&self.outline_params));
    }
}

// ============================================================
// Helpers
// ============================================================

fn create_color_target(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("postfx_color"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_depth_target(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("postfx_depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn bgl_texture(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_sampler(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
