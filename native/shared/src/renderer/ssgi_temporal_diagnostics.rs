//! Capture-only SSGI probe-history diagnostics.
//!
//! SSGI accumulates in a 3D probe × octel domain rather than screen space.
//! This flattens that domain into a probe atlas without changing its temporal
//! shader, textures, or normal frame graph.

use super::*;

pub(super) const SSGI_TEMPORAL_DIAGNOSTIC_NAMES: [&str; 6] = [
    "ssgi-rejection-reason",
    "ssgi-temporal-confidence",
    "ssgi-current-radiance",
    "ssgi-source-identity",
    "ssgi-current-integrated",
    "ssgi-history-integrated",
];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(super) struct SsgiTemporalDiagnosticResources {
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    integrated_pipeline: wgpu::ComputePipeline,
    integrated_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
}

const SHADER: &str = include_str!("ssgi_temporal_diagnostics.wgsl");

impl SsgiTemporalDiagnosticResources {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let textures = SSGI_TEMPORAL_DIAGNOSTIC_NAMES
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
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let texture_2d = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage_buffer = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
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
            label: Some("ssgi_temporal_diagnostic_layout"),
            entries: &[
                uniform(0),
                texture(1),
                texture(2),
                storage_buffer(3),
                storage(4),
                storage(5),
                texture_2d(6),
                storage(7),
                storage(8),
            ],
        });
        let integrated_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_integrated_diagnostic_layout"),
            entries: &[storage_buffer(3), storage(9), storage(10)],
        });
        let source = format!("{}{}", PROBE_HELPERS_WGSL, SHADER);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_temporal_diagnostic_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssgi_temporal_diagnostic_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssgi_temporal_diagnostic_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let integrated_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ssgi_integrated_diagnostic_pipeline_layout"),
                bind_group_layouts: &[Some(&integrated_layout)],
                immediate_size: 0,
            });
        let integrated_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ssgi_integrated_diagnostic_pipeline"),
                layout: Some(&integrated_pipeline_layout),
                module: &shader,
                entry_point: Some("cs_integrated"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            textures,
            views,
            pipeline,
            layout,
            integrated_pipeline,
            integrated_layout,
            width,
            height,
        }
    }
}

impl Renderer {
    pub(super) fn record_ssgi_temporal_diagnostics(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        previous_history_index: usize,
        grid_width: u32,
        grid_height: u32,
    ) {
        let (width, height) = (grid_width * PROBE_OCT_SIZE, grid_height * PROBE_OCT_SIZE);
        let resize = self
            .ssgi_temporal_diagnostics
            .as_ref()
            .is_some_and(|resources| resources.width != width || resources.height != height);
        if resize {
            self.ssgi_temporal_diagnostics = None;
        }
        if self.ssgi_temporal_diagnostics.is_none() {
            self.ssgi_temporal_diagnostics = Some(SsgiTemporalDiagnosticResources::new(
                &self.device,
                width,
                height,
            ));
        }
        let resources = self.ssgi_temporal_diagnostics.as_ref().unwrap();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssgi_temporal_diagnostic_bg"),
            layout: &resources.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.probe_temporal_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.probe_trace_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &self.probe_history_views[previous_history_index],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.probe_header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&resources.views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&resources.views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&resources.views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&resources.views[3]),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssgi_temporal_diagnostic_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&resources.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        drop(pass);

        let integrated_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssgi_integrated_diagnostic_bg"),
            layout: &resources.integrated_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.probe_header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&resources.views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&resources.views[5]),
                },
            ],
        });
        let mut integrated_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssgi_integrated_diagnostic_pass"),
            timestamp_writes: None,
        });
        integrated_pass.set_pipeline(&resources.integrated_pipeline);
        integrated_pass.set_bind_group(0, &integrated_bind_group, &[]);
        integrated_pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    pub(super) fn ssgi_temporal_diagnostic_textures(&self) -> Option<&[wgpu::Texture]> {
        self.ssgi_temporal_diagnostics
            .as_ref()
            .map(|resources| resources.textures.as_slice())
    }

    pub(super) fn release_ssgi_temporal_diagnostics(&mut self) {
        self.ssgi_temporal_diagnostics = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{PROBE_HELPERS_WGSL, SHADER};

    #[test]
    fn probe_diagnostic_shader_parses_without_touching_production_history() {
        let source = format!("{}{}", PROBE_HELPERS_WGSL, SHADER);
        wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("SSGI temporal diagnostics WGSL failed: {error}"));
        assert!(SHADER.contains("let estimator_uncertainty = max(current_probe.diffuse.w"));
        assert!(SHADER.contains("previous_diffuse.rgb"));
        assert!(SHADER.contains("current_probe.current_diffuse.rgb"));
        assert!(SHADER.contains("one_based_source"));
    }
}
