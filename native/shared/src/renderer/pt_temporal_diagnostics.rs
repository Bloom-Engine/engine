//! Capture-only realtime path-tracing temporal diagnostics.
//!
//! The pass reads the production SVGF ping-pong buffers after temporal
//! accumulation and exposes their decisions without modifying denoiser state.

use super::*;

pub(super) const PT_TEMPORAL_DIAGNOSTIC_NAMES: [&str; 4] = [
    "pt-rejection-reason",
    "pt-motion",
    "pt-reprojected-uv",
    "pt-temporal-confidence",
];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SHADER: &str = include_str!("pt_temporal_diagnostics.wgsl");

pub(super) struct PtTemporalDiagnosticResources {
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
}

impl PtTemporalDiagnosticResources {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let textures = PT_TEMPORAL_DIAGNOSTIC_NAMES
            .iter()
            .map(|name| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(name),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            })
            .collect::<Vec<_>>();
        let views = textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect::<Vec<_>>();
        let buffer = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pt_temporal_diagnostic_layout"),
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
                buffer(1, true),
                buffer(2, true),
                buffer(3, true),
                texture(4, wgpu::TextureSampleType::Depth),
                texture(5, wgpu::TextureSampleType::Float { filterable: false }),
                storage(6),
                storage(7),
                storage(8),
                storage(9),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt_temporal_diagnostic_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt_temporal_diagnostic_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pt_temporal_diagnostic_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            textures,
            views,
            pipeline,
            layout,
            width,
            height,
        }
    }
}

impl Renderer {
    pub(super) fn record_pt_temporal_diagnostics(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        current_index: usize,
        width: u32,
        height: u32,
    ) {
        let resize = self
            .pt_temporal_diagnostics
            .as_ref()
            .is_some_and(|resources| resources.width != width || resources.height != height);
        if resize {
            self.pt_temporal_diagnostics = None;
        }
        if self.pt_temporal_diagnostics.is_none() {
            self.pt_temporal_diagnostics = Some(PtTemporalDiagnosticResources::new(
                &self.device,
                width,
                height,
            ));
        }
        let resources = self.pt_temporal_diagnostics.as_ref().unwrap();
        let previous_index = 1 - current_index;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pt_temporal_diagnostic_bg"),
            layout: &resources.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.pt_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pt_accum_buffers[current_index]
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pt_moments_buffers[current_index]
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.pt_moments_buffers[previous_index]
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&resources.views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&resources.views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&resources.views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&resources.views[3]),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pt_temporal_diagnostic_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&resources.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    pub(super) fn pt_temporal_diagnostic_textures(&self) -> Option<&[wgpu::Texture]> {
        self.pt_temporal_diagnostics
            .as_ref()
            .map(|resources| resources.textures.as_slice())
    }

    pub(super) fn release_pt_temporal_diagnostics(&mut self) {
        self.pt_temporal_diagnostics = None;
    }

    pub(super) fn append_pt_temporal_diagnostic_telemetry(&self, out: &mut String) {
        let (render_width, render_height) = self.render_extent();
        let width = render_width.div_ceil(2).min(960);
        let height = render_height.div_ceil(2).min(540);
        let count = PT_TEMPORAL_DIAGNOSTIC_NAMES.len() as u64;
        let texture_bytes = u64::from(width) * u64::from(height) * count * 4;
        let row_bytes = u64::from((width * 4 + 255) & !255);
        let readback_bytes = row_bytes * u64::from(height) * count;
        out.push_str(",\"pt_diagnostic_persistent_bytes\":0");
        out.push_str(",\"pt_diagnostic_capture_texture_bytes\":");
        out.push_str(&texture_bytes.to_string());
        out.push_str(",\"pt_diagnostic_capture_readback_bytes\":");
        out.push_str(&readback_bytes.to_string());
        out.push_str(",\"pt_diagnostic_capture_passes\":1");
        out.push_str(",\"pt_diagnostic_resources_live\":");
        out.push_str(if self.pt_temporal_diagnostic_textures().is_some() {
            "true"
        } else {
            "false"
        });
    }
}

#[cfg(test)]
mod tests {
    use super::SHADER;

    #[test]
    fn diagnostic_shader_parses_without_modifying_production_svgf() {
        wgpu::naga::front::wgsl::parse_str(SHADER)
            .unwrap_or_else(|error| panic!("PT temporal diagnostics WGSL failed: {error}"));
        let production = super::super::shaders::PT_KERNEL_WGSL;
        for contract in [
            "uv_prev = vec2<f32>(uv_cur.x - vel.x, uv_cur.y + vel.y)",
            "let tol = 0.1 * rp_zl_here + 0.02",
            "let wtol = 0.1 * zl_st + 0.02",
            "let alpha_c = max(1.0 / n_new, 0.1)",
        ] {
            assert!(
                production.contains(contract),
                "PT temporal contract changed without updating diagnostics: {contract}"
            );
        }
    }
}
