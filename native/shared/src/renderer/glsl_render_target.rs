//! Runtime GLSL fragment execution for compatibility render targets.
//!
//! This is an offscreen bridge, not a second renderer: Naga parses the source
//! to native IR and wgpu writes the result directly into Bloom-owned textures.

use super::*;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const FULLSCREEN_VERTEX: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOut;
    let position = positions[index];
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = position * 0.5 + vec2<f32>(0.5);
    return out;
}
"#;

const SOBEL_FRAGMENT: &str = r#"
struct SobelParams { texel: vec2<f32>, strength: f32, _pad: f32 };
@group(0) @binding(0) var height_tex: texture_2d<f32>;
@group(0) @binding(1) var height_sampler: sampler;
@group(0) @binding(2) var<uniform> params: SobelParams;
fn height(uv: vec2<f32>, offset: vec2<f32>) -> f32 {
    return textureSample(height_tex, height_sampler, uv + offset * params.texel).r;
}
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let tl = height(uv, vec2<f32>(-1.0, 1.0));
    let t = height(uv, vec2<f32>(0.0, 1.0));
    let tr = height(uv, vec2<f32>(1.0, 1.0));
    let l = height(uv, vec2<f32>(-1.0, 0.0));
    let r = height(uv, vec2<f32>(1.0, 0.0));
    let bl = height(uv, vec2<f32>(-1.0, -1.0));
    let b = height(uv, vec2<f32>(0.0, -1.0));
    let br = height(uv, vec2<f32>(1.0, -1.0));
    let dx = ((tr + 2.0 * r + br) - (tl + 2.0 * l + bl)) * 0.125;
    let dy = ((tl + 2.0 * t + tr) - (bl + 2.0 * b + br)) * 0.125;
    let slope_x = dx / params.texel.x;
    let slope_y = dy / params.texel.y;
    let normal = normalize(vec3<f32>(-slope_x * params.strength, -slope_y * params.strength, 1.0));
    return vec4<f32>(normal * 0.5 + vec3<f32>(0.5), 1.0);
}
"#;

impl Renderer {
    pub fn render_glsl_fragment_to_texture(
        &mut self,
        target_texture_index: usize,
        source: &str,
    ) -> Result<(), String> {
        let target = self
            .textures
            .get(target_texture_index)
            .cloned()
            .ok_or_else(|| "render target texture is not live".to_string())?;
        let mut frontend = naga::front::glsl::Frontend::default();
        let options = naga::front::glsl::Options::from(naga::ShaderStage::Fragment);
        let module = frontend
            .parse(&options, source)
            .map_err(|errors| errors.emit_to_string(source))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .map_err(|error| format!("GLSL validation failed: {error:?}"))?;
        let wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .map_err(|error| format!("GLSL translation failed: {error}"))?;

        let vertex = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("compat_fullscreen_vertex"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(FULLSCREEN_VERTEX)),
            });
        let fragment = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("compat_glsl_fragment"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compat_glsl_rt_layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("compat_glsl_rt_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some("main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target.format(),
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let view = target.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compat_glsl_rt_encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compat_glsl_rt_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn render_sobel_to_texture(
        &mut self,
        target_texture_index: usize,
        source_texture_index: usize,
        strength: f32,
    ) -> Result<(), String> {
        let target = self
            .textures
            .get(target_texture_index)
            .cloned()
            .ok_or_else(|| "Sobel target texture is not live".to_string())?;
        let source = self
            .textures
            .get(source_texture_index)
            .cloned()
            .ok_or_else(|| "Sobel source texture is not live".to_string())?;
        let (width, height) = self
            .texture_sizes
            .get(source_texture_index)
            .copied()
            .ok_or_else(|| "Sobel source dimensions are unavailable".to_string())?;
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compat_sobel_layout"),
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
                ],
            });
        let params = [
            1.0 / width.max(1) as f32,
            1.0 / height.max(1) as f32,
            strength,
            0.0,
        ];
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("compat_sobel_params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let source_view = source.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compat_sobel_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compat_sobel_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let vertex = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("compat_sobel_vertex"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(FULLSCREEN_VERTEX)),
            });
        let fragment = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("compat_sobel_fragment"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOBEL_FRAGMENT)),
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("compat_sobel_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target.format(),
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let target_view = target.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compat_sobel_encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compat_sobel_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}
