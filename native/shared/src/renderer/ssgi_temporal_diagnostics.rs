//! Capture-only SSGI probe-history diagnostics.
//!
//! SSGI accumulates in a 3D probe × octel domain rather than screen space.
//! This flattens that domain into a probe atlas without changing its temporal
//! shader, textures, or normal frame graph.

use super::*;

const PROBE_DIAGNOSTIC_COUNT: usize = 8;
const RESOLVE_DIAGNOSTIC_COUNT: usize = 4;
pub(super) const SSGI_TEMPORAL_DIAGNOSTIC_NAMES: [&str;
    PROBE_DIAGNOSTIC_COUNT + RESOLVE_DIAGNOSTIC_COUNT] = [
    "ssgi-rejection-reason",
    "ssgi-temporal-confidence",
    "ssgi-current-radiance",
    "ssgi-source-identity",
    "ssgi-current-integrated",
    "ssgi-history-integrated",
    "ssgi-spatial-integrated",
    "ssgi-ring-integrated",
    "ssgi-resolve-support",
    "ssgi-resolve-geometry",
    "ssgi-resolve-plane-ratios",
    "ssgi-resolve-plane-ratio-w",
];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(super) struct SsgiTemporalDiagnosticResources {
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    integrated_pipeline: wgpu::ComputePipeline,
    integrated_layout: wgpu::BindGroupLayout,
    resolve_support_pipeline: wgpu::ComputePipeline,
    resolve_support_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    resolve_width: u32,
    resolve_height: u32,
}

const SHADER: &str = include_str!("ssgi_temporal_diagnostics.wgsl");
const RESOLVE_SUPPORT_SHADER: &str = include_str!("ssgi_resolve_diagnostics.wgsl");

impl SsgiTemporalDiagnosticResources {
    fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        resolve_width: u32,
        resolve_height: u32,
    ) -> Self {
        let mut textures = SSGI_TEMPORAL_DIAGNOSTIC_NAMES[..PROBE_DIAGNOSTIC_COUNT]
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
        for name in &SSGI_TEMPORAL_DIAGNOSTIC_NAMES[PROBE_DIAGNOSTIC_COUNT..] {
            textures.push(device.create_texture(&wgpu::TextureDescriptor {
                label: Some(name),
                size: wgpu::Extent3d {
                    width: resolve_width,
                    height: resolve_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }));
        }
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
            entries: &[
                texture(2),
                storage_buffer(3),
                storage(9),
                storage(10),
                storage(11),
                storage(12),
            ],
        });
        let resolve_support_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssgi_resolve_support_diagnostic_layout"),
                entries: &[
                    uniform(0),
                    storage_buffer(1),
                    texture_2d(2),
                    storage(3),
                    storage(4),
                    storage(5),
                    storage(6),
                    texture_2d(7),
                    texture_2d(8),
                ],
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
        let resolve_support_source = format!("{}{}", PROBE_HELPERS_WGSL, RESOLVE_SUPPORT_SHADER);
        let resolve_support_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_resolve_support_diagnostic_shader"),
            source: wgpu::ShaderSource::Wgsl(resolve_support_source.into()),
        });
        let resolve_support_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ssgi_resolve_support_diagnostic_pipeline_layout"),
                bind_group_layouts: &[Some(&resolve_support_layout)],
                immediate_size: 0,
            });
        let resolve_support_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ssgi_resolve_support_diagnostic_pipeline"),
                layout: Some(&resolve_support_pipeline_layout),
                module: &resolve_support_module,
                entry_point: Some("cs_resolve_support"),
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
            resolve_support_pipeline,
            resolve_support_layout,
            width,
            height,
            resolve_width,
            resolve_height,
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
        resolve_width: u32,
        resolve_height: u32,
    ) {
        let (width, height) = (grid_width * PROBE_OCT_SIZE, grid_height * PROBE_OCT_SIZE);
        let resize = self
            .ssgi_temporal_diagnostics
            .as_ref()
            .is_some_and(|resources| {
                resources.width != width
                    || resources.height != height
                    || resources.resolve_width != resolve_width
                    || resources.resolve_height != resolve_height
            });
        if resize {
            self.ssgi_temporal_diagnostics = None;
        }
        if self.ssgi_temporal_diagnostics.is_none() {
            self.ssgi_temporal_diagnostics = Some(SsgiTemporalDiagnosticResources::new(
                &self.device,
                width,
                height,
                resolve_width,
                resolve_height,
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
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&resources.views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&resources.views[5]),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&resources.views[6]),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&resources.views[7]),
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

    /// Records the exact screen-space support decision used by production
    /// resolve. Red marks a visible receiver that production would leave at
    /// zero; green stores the broad compatible-probe count divided by four;
    /// blue stores strict bilateral support weight.
    pub(super) fn record_ssgi_resolve_support_diagnostic(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        previous_resolve_idx: usize,
    ) {
        let Some(resources) = self.ssgi_temporal_diagnostics.as_ref() else {
            return;
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssgi_resolve_support_diagnostic_bg"),
            layout: &resources.resolve_support_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.probe_resolve_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.probe_header_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.views[PROBE_DIAGNOSTIC_COUNT],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.views[PROBE_DIAGNOSTIC_COUNT + 1],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.views[PROBE_DIAGNOSTIC_COUNT + 2],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.views[PROBE_DIAGNOSTIC_COUNT + 3],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(
                        &self.ssgi_rt_views[previous_resolve_idx],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssgi_resolve_support_diagnostic_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&resources.resolve_support_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            resources.resolve_width.div_ceil(8),
            resources.resolve_height.div_ceil(8),
            1,
        );
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
    use super::{PROBE_HELPERS_WGSL, RESOLVE_SUPPORT_SHADER, SHADER};

    #[test]
    fn probe_diagnostic_shader_parses_without_touching_production_history() {
        let source = format!("{}{}", PROBE_HELPERS_WGSL, SHADER);
        wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("SSGI temporal diagnostics WGSL failed: {error}"));
        assert!(SHADER.contains("let estimator_uncertainty = max(current_probe.diffuse.w"));
        assert!(SHADER.contains("previous_diffuse.rgb"));
        assert!(SHADER.contains("current_probe.current_diffuse.rgb"));
        assert!(SHADER.contains("let spatial_integrated = bounded_probe_history(textureLoad("));
        assert!(SHADER.contains("let ring_integrated = bounded_probe_history"));
        assert!(SHADER.contains("var alpha = u.confidence.z"));
        assert!(SHADER.contains("one_based_source"));
    }

    #[test]
    fn resolve_support_diagnostic_shader_matches_production_decision() {
        let source = format!("{}{}", PROBE_HELPERS_WGSL, RESOLVE_SUPPORT_SHADER);
        wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("SSGI resolve diagnostics WGSL failed: {error}"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("fallback_count >= 2u"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("fallback_weight >= 0.25"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("w_corner * w_plane * w_normal"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("let fallback_corner_weight = select("));
        assert!(RESOLVE_SUPPORT_SHADER.contains("(0.08 + probe_world_spacing * 0.85) * 3.0"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("plane_error / max(fallback_plane_limit"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("normal_compatible && plane_ratio <= 1.0"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("best_normal_plane_ratio * 0.25"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("plane_ratios[corner_index]"));
        assert!(
            RESOLVE_SUPPORT_SHADER.contains("coherent_count == 4u"),
            "a fully coherent receiver footprint must not switch reconstruction kernels with strict probe-grid support",
        );
        assert!(
            RESOLVE_SUPPORT_SHADER.contains("f32(coherent_count) * 0.25"),
            "capture diagnostics must expose the complete coherent footprint, including strict samples",
        );
        assert!(RESOLVE_SUPPORT_SHADER.contains("u.prev_view * vec4<f32>(P_ws, 1.0)"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("history_accepted = history_depth > 0.0"));
        assert!(RESOLVE_SUPPORT_SHADER.contains("normalized_history_depth_error * 0.25"));
    }
}
