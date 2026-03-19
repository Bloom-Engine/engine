use wgpu::util::DeviceExt;

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
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniforms {
    ambient: [f32; 4],      // rgb + intensity
    light_dir: [f32; 4],    // xyz + intensity
    light_color: [f32; 4],  // rgb + _pad
}

impl LightingUniforms {
    fn defaults() -> Self {
        Self {
            ambient: [1.0, 1.0, 1.0, 0.3],
            light_dir: [0.5, 1.0, 0.3, 0.7],
            light_color: [1.0, 1.0, 1.0, 0.0],
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
};

struct Lighting {
    ambient: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
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
        // GPU skinning: blend position/normal by bone matrices
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
    out.color = in.color;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main_3d(in: VertexOutput3D) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light_dir = normalize(lighting.light_dir.xyz);
    let diffuse = max(dot(n, light_dir), 0.0);
    let ambient = lighting.ambient.rgb * lighting.ambient.a;
    let direct = lighting.light_color.rgb * lighting.light_dir.w * diffuse;
    let lit = ambient + direct;
    let tex_color = textureSample(tex3d, tex3d_sampler, in.uv);
    return vec4<f32>(tex_color.rgb * in.color.rgb * lit, tex_color.a * in.color.a);
}
";

// ============================================================
// Depth texture helper
// ============================================================

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
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

    // Depth buffer
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // Per-frame 2D batch
    vertices_2d: Vec<Vertex2D>,
    indices_2d: Vec<u32>,
    draw_calls_2d: Vec<DrawCall2D>,

    // Per-frame 3D batch
    pub vertices_3d: Vec<Vertex3D>,
    pub indices_3d: Vec<u32>,
    draw_calls_3d: Vec<DrawCall3D>,
    current_texture_3d: u32,

    // State
    pub render_mode: RenderMode,
    clear_color: wgpu::Color,
    debug_frame: u64,
    // Pending joint matrices (written to GPU in end_frame)
    pub pending_joint_matrices: Option<Vec<[[f32; 4]; 4]>>,
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
            contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4 }),
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
        let lighting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let lighting_uniforms = LightingUniforms::defaults();
        let lighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lighting_buffer"),
            contents: bytemuck::bytes_of(&lighting_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_bg"),
            layout: &lighting_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lighting_buffer.as_entire_binding(),
            }],
        });

        // --- Sampler ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
            depth_texture,
            depth_view,
            vertices_2d: Vec::with_capacity(4096),
            indices_2d: Vec::with_capacity(8192),
            draw_calls_2d: Vec::new(),
            vertices_3d: Vec::with_capacity(4096),
            indices_3d: Vec::with_capacity(8192),
            draw_calls_3d: Vec::new(),
            current_texture_3d: 0,
            render_mode: RenderMode::ScreenSpace,
            debug_frame: 0,
            pending_joint_matrices: None,
            clear_color: wgpu::Color::BLACK,
            custom_pipelines: Vec::new(),
        }
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
        }
    }

    pub fn set_clear_color(&mut self, r: f64, g: f64, b: f64, a: f64) {
        self.clear_color = wgpu::Color { r: r / 255.0, g: g / 255.0, b: b / 255.0, a: a / 255.0 };
    }

    pub fn begin_frame(&mut self) {
        self.vertices_2d.clear();
        self.indices_2d.clear();
        self.draw_calls_2d.clear();
        self.vertices_3d.clear();
        self.indices_3d.clear();
        self.draw_calls_3d.clear();
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

        // Reset lighting to defaults
        self.lighting_uniforms = LightingUniforms::defaults();
        self.queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&self.lighting_uniforms));

        // DEBUG: joint animation disabled for iOS port
        // self.debug_frame += 1;
        // let angle = (self.debug_frame as f32) * 0.03;
        // self.set_joint_test(0, angle.sin() * 0.8);
        // self.set_joint_test(5, (angle * 1.5).sin() * 0.5);
    }

    pub fn end_frame(&mut self) {
        // Flush pending joint matrices to GPU right before rendering
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

        // Create 2D GPU buffers
        let vb_2d = if !self.vertices_2d.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vb_2d"),
                contents: bytemuck::cast_slice(&self.vertices_2d),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else { None };
        let ib_2d = if !self.indices_2d.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ib_2d"),
                contents: bytemuck::cast_slice(&self.indices_2d),
                usage: wgpu::BufferUsages::INDEX,
            }))
        } else { None };

        // Create 3D GPU buffers
        let vb_3d = if !self.vertices_3d.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vb_3d"),
                contents: bytemuck::cast_slice(&self.vertices_3d),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else { None };
        let ib_3d = if !self.indices_3d.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ib_3d"),
                contents: bytemuck::cast_slice(&self.indices_3d),
                usage: wgpu::BufferUsages::INDEX,
            }))
        } else { None };

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

            // Draw 3D geometry first (with depth testing), batched by texture
            if let (Some(vb), Some(ib)) = (&vb_3d, &ib_3d) {
                pass.set_pipeline(&self.pipeline_3d);
                pass.set_bind_group(0, &self.uniform_bind_group_3d, &[]);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);

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

            // Draw 2D geometry (no depth testing, always passes)
            if let (Some(vb), Some(ib)) = (&vb_2d, &ib_2d) {
                pass.set_pipeline(&self.pipeline_2d);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);

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
        output.present();
    }

    // ============================================================
    // Texture management
    // ============================================================

    pub fn register_texture(&mut self, width: u32, height: u32, data: &[u8]) -> u32 {
        let texture = self.device.create_texture_with_data(
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
            data,
        );
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
        let proj = if projection < 0.5 {
            mat4_perspective(fovy.to_radians(), aspect, 0.01, 1000.0)
        } else {
            let top = fovy / 2.0;
            mat4_ortho(-top * aspect, top * aspect, -top, top, 0.01, 1000.0)
        };
        let view = mat4_look_at(
            [pos_x, pos_y, pos_z],
            [target_x, target_y, target_z],
            [up_x, up_y, up_z],
        );
        let mvp = mat4_multiply(proj, view);

        self.queue.write_buffer(
            &self.uniform_buffer_3d,
            0,
            bytemuck::bytes_of(&Uniforms3D { mvp }),
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
        // Rotation around X axis, column-major in the buffer
        let mut data = [0u8; 64];
        let mat: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0,   c,   s, 0.0],
            [0.0,  -s,   c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // Write as column-major
        for col in 0..4 {
            for row in 0..4 {
                let bytes = mat[row][col].to_le_bytes();
                let idx = col * 16 + row * 4;
                data[idx..idx+4].copy_from_slice(&bytes);
            }
        }
        self.queue.write_buffer(&self.joint_buffer, (joint_index * 64) as u64, &data);
    }

    pub fn set_joint_matrices(&mut self, matrices: &[[[f32; 4]; 4]]) {
        // Store for writing in end_frame (GPU needs data right before render pass)
        self.pending_joint_matrices = Some(matrices.to_vec());
    }

    fn flush_joint_matrices(&mut self) {
        if let Some(ref matrices) = self.pending_joint_matrices {
            let count = matrices.len().min(128);
            let mut data = vec![0u8; 8192];
            for i in 0..count {
                let offset = i * 64;
                // Row-major [row][col] → column-major for WGSL
                for col in 0..4 {
                    for row in 0..4 {
                        let bytes = matrices[i][row][col].to_le_bytes();
                        let idx = offset + col * 16 + row * 4;
                        data[idx..idx+4].copy_from_slice(&bytes);
                    }
                }
            }
            self.queue.write_buffer(&self.joint_buffer, 0, &data);
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
        self.vertices_3d.push(Vertex3D { position: [start[0]+px, start[1]+py, start[2]+pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [start[0]-px, start[1]-py, start[2]-pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [end[0]-px, end[1]-py, end[2]-pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [end[0]+px, end[1]+py, end[2]+pz], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
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
                self.vertices_3d.push(Vertex3D { position: *v, normal: *normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
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
        let rings = 16u32;
        let slices = 16u32;

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
                self.vertices_3d.push(Vertex3D { position: p00, normal: n00, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
                self.vertices_3d.push(Vertex3D { position: p10, normal: n10, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
                self.vertices_3d.push(Vertex3D { position: p11, normal: n11, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
                self.vertices_3d.push(Vertex3D { position: p01, normal: n01, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
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
            self.vertices_3d.push(Vertex3D { position: [x + rb*c1, y, z + rb*s1], normal: [c1, 0.0, s1], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x + rb*c2, y, z + rb*s2], normal: [c2, 0.0, s2], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x + rt*c2, y+h, z + rt*s2], normal: [c2, 0.0, s2], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x + rt*c1, y+h, z + rt*s1], normal: [c1, 0.0, s1], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.indices_3d.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);

            // Top cap
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D { position: [x, y+h, z], normal: [0.0, 1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rt*c1, y+h, z+rt*s1], normal: [0.0, 1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rt*c2, y+h, z+rt*s2], normal: [0.0, 1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.indices_3d.extend_from_slice(&[base, base+1, base+2]);

            // Bottom cap
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D { position: [x, y, z], normal: [0.0, -1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rb*c2, y, z+rb*s2], normal: [0.0, -1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
            self.vertices_3d.push(Vertex3D { position: [x+rb*c1, y, z+rb*s1], normal: [0.0, -1.0, 0.0], color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
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
        self.vertices_3d.push(Vertex3D { position: [cx-hw, cy, cz-hd], normal, color, uv: [0.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [cx+hw, cy, cz-hd], normal, color, uv: [1.0, 0.0], joints: [0.0; 4], weights: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [cx+hw, cy, cz+hd], normal, color, uv: [1.0, 1.0], joints: [0.0; 4], weights: [0.0; 4] });
        self.vertices_3d.push(Vertex3D { position: [cx-hw, cy, cz+hd], normal, color, uv: [0.0, 1.0], joints: [0.0; 4], weights: [0.0; 4] });
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
                // Skinned: pass bind-space position (shader applies joint matrices + model transform via MVP)
                // Scale is baked into the model, position offset is handled by adjusting joint matrices
                [v.position[0] * scale, v.position[1] * scale, v.position[2] * scale]
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
