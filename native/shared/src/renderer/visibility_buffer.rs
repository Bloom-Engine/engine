//! Packed visibility-buffer contract for the #27 qualification path.
//!
//! This module does not enable a shipping render path. It locks the 8-byte
//! target ABI and reconstruction math that an opt-in A/B implementation will
//! use. The existing forward MRT remains authoritative until total frame cost
//! and image parity pass on every required capability tier.

pub(crate) const VISIBILITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
pub(crate) const VISIBILITY_BYTES_PER_PIXEL: u64 = 8;
pub(crate) const INVALID_DRAW_ID: u32 = u32::MAX;
pub(crate) const FRONT_FACE_BIT: u32 = 1 << 31;
pub(crate) const PRIMITIVE_ID_MASK: u32 = FRONT_FACE_BIT - 1;

/// One visibility-buffer texel. The second word reserves its high bit for the
/// rasterized face orientation and leaves 31 bits for the primitive index.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VisibilityRecord {
    pub draw_id: u32,
    pub primitive_and_face: u32,
}

impl VisibilityRecord {
    pub(crate) const BACKGROUND: Self = Self {
        draw_id: INVALID_DRAW_ID,
        primitive_and_face: u32::MAX,
    };

    pub(crate) const fn encode(
        draw_id: u32,
        primitive_id: u32,
        front_facing: bool,
    ) -> Option<Self> {
        if draw_id == INVALID_DRAW_ID || primitive_id > PRIMITIVE_ID_MASK {
            return None;
        }
        Some(Self {
            draw_id,
            primitive_and_face: primitive_id | if front_facing { FRONT_FACE_BIT } else { 0 },
        })
    }

    pub(crate) const fn decode(self) -> Option<(u32, u32, bool)> {
        if self.draw_id == INVALID_DRAW_ID {
            return None;
        }
        Some((
            self.draw_id,
            self.primitive_and_face & PRIMITIVE_ID_MASK,
            (self.primitive_and_face & FRONT_FACE_BIT) != 0,
        ))
    }
}

/// Exact allocation size of the packed visibility target, excluding backend
/// row/heap alignment that must be reported separately by the runtime A/B.
pub(crate) const fn target_bytes(width: u32, height: u32) -> Option<u64> {
    match (width as u64).checked_mul(height as u64) {
        Some(pixels) => pixels.checked_mul(VISIBILITY_BYTES_PER_PIXEL),
        None => None,
    }
}

/// Stable machine-readable contract included in renderer diagnostics even
/// while the experimental path is disabled.
pub(crate) fn contract_json() -> String {
    let format_name = match VISIBILITY_FORMAT {
        wgpu::TextureFormat::Rg32Uint => "rg32uint",
        _ => "invalid",
    };
    let background = VisibilityRecord::BACKGROUND;
    let max_record = VisibilityRecord::encode(0, PRIMITIVE_ID_MASK, true)
        .expect("the visibility ABI maximum must remain encodable");
    debug_assert_eq!(background.decode(), None);
    debug_assert_eq!(max_record.decode(), Some((0, PRIMITIVE_ID_MASK, true)));
    format!(
        concat!(
            "{{\"format\":\"{}\",\"bytes_per_pixel\":{},",
            "\"invalid_draw_id\":{},\"primitive_bits\":31,",
            "\"front_face_bits\":1,\"shipping_enabled\":false,",
            "\"required_feature\":\"primitive-index\",",
            "\"vertex_stride_bytes\":{},\"native_1080p_bytes\":{},",
            "\"reconstruction_wgsl_bytes\":{},\"geometry_wgsl_bytes\":{},",
            "\"activation\":\"opt-in A/B qualification required\"}}"
        ),
        format_name,
        VISIBILITY_BYTES_PER_PIXEL,
        INVALID_DRAW_ID,
        std::mem::size_of::<super::Vertex3D>(),
        target_bytes(1_920, 1_080).expect("1080p visibility allocation is bounded"),
        RECONSTRUCTION_WGSL.len(),
        GEOMETRY_WGSL.len(),
    )
}

pub(crate) const RECONSTRUCTION_WGSL: &str =
    include_str!("../../shaders/visibility_buffer/reconstruct.wgsl");
pub(crate) const GEOMETRY_WGSL: &str =
    include_str!("../../shaders/visibility_buffer/geometry.wgsl");

const DIAGNOSTIC_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DIAGNOSTIC_BYTES_PER_PIXEL: u64 = 8;
const WORKGROUP_SIZE: u32 = 8;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeMode {
    Off,
    Validate,
    Debug,
    Shade,
}

impl RuntimeMode {
    const fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Validate => "validate",
            Self::Debug => "debug",
            Self::Shade => "shade",
        }
    }

    pub(crate) const fn requested(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub(crate) const fn shades(self) -> bool {
        matches!(self, Self::Shade)
    }
}

fn parse_runtime_mode(value: Option<&str>) -> RuntimeMode {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "on" | "true" | "validate") => RuntimeMode::Validate,
        Some("debug" | "visualize" | "visualise") => RuntimeMode::Debug,
        Some("shade" | "pbr") => RuntimeMode::Shade,
        _ => RuntimeMode::Off,
    }
}

pub(crate) fn requested_mode() -> RuntimeMode {
    static MODE: std::sync::OnceLock<RuntimeMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        if cfg!(target_arch = "wasm32") {
            RuntimeMode::Off
        } else {
            parse_runtime_mode(std::env::var("BLOOM_VISIBILITY_BUFFER").ok().as_deref())
        }
    })
}

pub(crate) fn request_feature_if_supported(
    supported: wgpu::Features,
    required: &mut wgpu::Features,
) {
    request_feature_for_mode(requested_mode(), supported, required);
}

fn request_feature_for_mode(
    mode: RuntimeMode,
    supported: wgpu::Features,
    required: &mut wgpu::Features,
) {
    if mode.requested() && supported.contains(wgpu::Features::PRIMITIVE_INDEX) {
        *required |= wgpu::Features::PRIMITIVE_INDEX;
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResourceCreations {
    pub textures: u32,
    pub bind_groups: u32,
}

struct RuntimeResources {
    _visibility_texture: wgpu::Texture,
    visibility_view: wgpu::TextureView,
    _diagnostic_texture: Option<wgpu::Texture>,
    _diagnostic_view: Option<wgpu::TextureView>,
    reconstruct_bind_group: Option<wgpu::BindGroup>,
    overlay_bind_group: Option<wgpu::BindGroup>,
    shade_bind_group: Option<wgpu::BindGroup>,
    extent: (u32, u32),
    draw_capacity: usize,
    geometry_generation: u64,
}

pub(crate) struct VisibilityBufferRuntime {
    mode: RuntimeMode,
    enabled: bool,
    disabled_reason: &'static str,
    raster_pipeline: Option<wgpu::RenderPipeline>,
    reconstruct_pipeline: Option<wgpu::ComputePipeline>,
    reconstruct_layout: Option<wgpu::BindGroupLayout>,
    overlay_pipeline: Option<wgpu::RenderPipeline>,
    overlay_layout: Option<wgpu::BindGroupLayout>,
    shade_pipeline: Option<wgpu::RenderPipeline>,
    shade_layout: Option<wgpu::BindGroupLayout>,
    resources: Option<RuntimeResources>,
    eligible_draws: u32,
    compatibility_draws: u32,
    frame_recorded: bool,
}

impl VisibilityBufferRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &wgpu::Device,
        gpu_driven_enabled: bool,
        draw_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        global_material_layout: Option<&wgpu::BindGroupLayout>,
        joint_layout: &wgpu::BindGroupLayout,
        gpu_scene_source: Option<&str>,
    ) -> Self {
        let mode = requested_mode();
        let mut runtime = Self::disabled(mode, "not-requested");
        if !mode.requested() {
            return runtime;
        }
        if !device.features().contains(wgpu::Features::PRIMITIVE_INDEX) {
            runtime.disabled_reason = "primitive-index-unavailable";
            return runtime;
        }
        if !gpu_driven_enabled {
            runtime.disabled_reason = "gpu-driven-unavailable";
            return runtime;
        }
        let (Some(global_material_layout), Some(gpu_scene_source)) =
            (global_material_layout, gpu_scene_source)
        else {
            runtime.disabled_reason = "tier-a-materials-unavailable";
            return runtime;
        };

        let raster_source = make_visibility_raster_shader(gpu_scene_source);
        let raster_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visibility_buffer_runtime_raster_shader"),
            source: wgpu::ShaderSource::Wgsl(raster_source.into()),
        });
        let raster_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visibility_buffer_runtime_raster_pipeline_layout"),
            bind_group_layouts: &[
                Some(draw_layout),
                Some(lighting_layout),
                Some(global_material_layout),
                Some(joint_layout),
            ],
            immediate_size: 0,
        });
        let raster_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("visibility_buffer_runtime_raster_pipeline"),
            layout: Some(&raster_layout),
            vertex: wgpu::VertexState {
                module: &raster_shader,
                entry_point: Some("vs_main_scene"),
                buffers: &[super::Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &raster_shader,
                entry_point: Some("fs_visibility_buffer"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: VISIBILITY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Equal),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let (reconstruct_layout, reconstruct_pipeline) = if !mode.shades() {
            let layout = create_reconstruct_layout(device);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visibility_buffer_runtime_reconstruct_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visibility_buffer_runtime_reconstruct_shader"),
                source: wgpu::ShaderSource::Wgsl(RUNTIME_RECONSTRUCT_WGSL.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("visibility_buffer_runtime_reconstruct_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cs_visibility_reconstruct"),
                compilation_options: Default::default(),
                cache: None,
            });
            (Some(layout), Some(pipeline))
        } else {
            (None, None)
        };

        let (overlay_layout, overlay_pipeline) = if matches!(mode, RuntimeMode::Debug) {
            let layout = create_overlay_layout(device);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visibility_buffer_debug_overlay_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visibility_buffer_debug_overlay_shader"),
                source: wgpu::ShaderSource::Wgsl(DEBUG_OVERLAY_WGSL.into()),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visibility_buffer_debug_overlay_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_debug_overlay"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_debug_overlay"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: super::HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            (Some(layout), Some(pipeline))
        } else {
            (None, None)
        };

        let (shade_layout, shade_pipeline) = if mode.shades() {
            let shade_layout = super::visibility_shading::create_layout(device);
            let shade_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("visibility_buffer_pbr_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(draw_layout),
                        Some(lighting_layout),
                        Some(global_material_layout),
                        Some(joint_layout),
                        Some(&shade_layout),
                    ],
                    immediate_size: 0,
                });
            let shade_source = super::visibility_shading::make_shader(gpu_scene_source);
            let shade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visibility_buffer_pbr_shader"),
                source: wgpu::ShaderSource::Wgsl(shade_source.into()),
            });
            let shade_pipeline = super::visibility_shading::create_pipeline(
                device,
                &shade_pipeline_layout,
                &shade_shader,
            );
            (Some(shade_layout), Some(shade_pipeline))
        } else {
            (None, None)
        };

        log::info!(
            "bloom: visibility-buffer runtime enabled mode={} composition={}",
            mode.name(),
            if mode.shades() {
                "visibility-eligible+forward-compatibility"
            } else {
                "forward-authoritative"
            }
        );
        Self {
            mode,
            enabled: true,
            disabled_reason: "none",
            raster_pipeline: Some(raster_pipeline),
            reconstruct_pipeline,
            reconstruct_layout,
            overlay_pipeline,
            overlay_layout,
            shade_pipeline,
            shade_layout,
            resources: None,
            eligible_draws: 0,
            compatibility_draws: 0,
            frame_recorded: false,
        }
    }

    fn disabled(mode: RuntimeMode, disabled_reason: &'static str) -> Self {
        Self {
            mode,
            enabled: false,
            disabled_reason,
            raster_pipeline: None,
            reconstruct_pipeline: None,
            reconstruct_layout: None,
            overlay_pipeline: None,
            overlay_layout: None,
            shade_pipeline: None,
            shade_layout: None,
            resources: None,
            eligible_draws: 0,
            compatibility_draws: 0,
            frame_recorded: false,
        }
    }

    pub(crate) const fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) const fn debug_overlay_enabled(&self) -> bool {
        self.enabled && matches!(self.mode, RuntimeMode::Debug)
    }

    pub(crate) const fn reconstruction_enabled(&self) -> bool {
        self.enabled && !self.mode.shades()
    }

    pub(crate) const fn shading_active(&self) -> bool {
        self.enabled && self.mode.shades() && self.frame_recorded
    }

    pub(crate) const fn shading_requested(&self) -> bool {
        self.enabled && self.mode.shades()
    }

    pub(crate) fn begin_frame(&mut self) {
        self.frame_recorded = false;
        self.eligible_draws = 0;
        self.compatibility_draws = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ensure_resources(
        &mut self,
        device: &wgpu::Device,
        extent: (u32, u32),
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        draw_buffer: &wgpu::Buffer,
        draw_capacity: usize,
        geometry_generation: u64,
    ) -> ResourceCreations {
        if !self.enabled {
            return ResourceCreations::default();
        }
        let extent = (extent.0.max(1), extent.1.max(1));
        let current = self.resources.as_ref().is_some_and(|resources| {
            resources.extent == extent
                && resources.draw_capacity == draw_capacity
                && resources.geometry_generation == geometry_generation
        });
        if current {
            return ResourceCreations::default();
        }

        let texture_size = wgpu::Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_array_layers: 1,
        };
        let visibility_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visibility_buffer_runtime_ids"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VISIBILITY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let visibility_view = visibility_texture.create_view(&Default::default());
        let diagnostic_texture = self.reconstruction_enabled().then(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("visibility_buffer_runtime_reconstruction"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DIAGNOSTIC_OUTPUT_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let diagnostic_view = diagnostic_texture
            .as_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let reconstruct_bind_group = diagnostic_view.as_ref().map(|diagnostic_view| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("visibility_buffer_runtime_reconstruct_bind_group"),
                layout: self
                    .reconstruct_layout
                    .as_ref()
                    .expect("enabled visibility runtime owns reconstruct layout"),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&visibility_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: draw_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(diagnostic_view),
                    },
                ],
            })
        });
        let overlay_bind_group = if self.debug_overlay_enabled() {
            diagnostic_view.as_ref().map(|diagnostic_view| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("visibility_buffer_debug_overlay_bind_group"),
                    layout: self
                        .overlay_layout
                        .as_ref()
                        .expect("enabled visibility runtime owns overlay layout"),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(diagnostic_view),
                    }],
                })
            })
        } else {
            None
        };
        let shade_bind_group = if self.mode.shades() {
            Some(
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("visibility_buffer_pbr_bind_group"),
                    layout: self
                        .shade_layout
                        .as_ref()
                        .expect("visibility shade mode owns its layout"),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&visibility_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: vertex_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: index_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: draw_buffer.as_entire_binding(),
                        },
                    ],
                }),
            )
        } else {
            None
        };
        let texture_creations = 1 + u32::from(diagnostic_texture.is_some());
        let bind_group_creations = u32::from(reconstruct_bind_group.is_some())
            + u32::from(overlay_bind_group.is_some())
            + u32::from(shade_bind_group.is_some());
        self.resources = Some(RuntimeResources {
            _visibility_texture: visibility_texture,
            visibility_view,
            _diagnostic_texture: diagnostic_texture,
            _diagnostic_view: diagnostic_view,
            reconstruct_bind_group,
            overlay_bind_group,
            shade_bind_group,
            extent,
            draw_capacity,
            geometry_generation,
        });
        ResourceCreations {
            textures: texture_creations,
            bind_groups: bind_group_creations,
        }
    }

    pub(crate) fn set_draw_counts(&mut self, eligible: u32, compatibility: u32) {
        self.eligible_draws = eligible;
        self.compatibility_draws = compatibility;
    }

    pub(crate) fn raster_attachment(&self) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        let resources = self.resources.as_ref()?;
        Some(wgpu::RenderPassColorAttachment {
            view: &resources.visibility_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: u32::MAX as f64,
                    g: u32::MAX as f64,
                    b: 0.0,
                    a: 0.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_raster<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        draw_bind_group: &'a wgpu::BindGroup,
        lighting: &'a wgpu::BindGroup,
        global_materials: &'a wgpu::BindGroup,
        joints: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
        index_buffer: &'a wgpu::Buffer,
        indirect_buffer: &'a wgpu::Buffer,
        counter_buffer: &'a wgpu::Buffer,
        draw_count: u32,
        count_supported: bool,
    ) {
        pass.set_pipeline(
            self.raster_pipeline
                .as_ref()
                .expect("enabled visibility runtime owns raster pipeline"),
        );
        pass.set_bind_group(0, draw_bind_group, &[]);
        pass.set_bind_group(1, lighting, &[]);
        pass.set_bind_group(2, global_materials, &[]);
        pass.set_bind_group(3, joints, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        if count_supported {
            pass.multi_draw_indexed_indirect_count(
                indirect_buffer,
                0,
                counter_buffer,
                0,
                draw_count,
            );
        } else {
            pass.multi_draw_indexed_indirect(indirect_buffer, 0, draw_count);
        }
    }

    pub(crate) fn mark_raster_recorded(&mut self) {
        self.frame_recorded = true;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_raster(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        draw_bind_group: &wgpu::BindGroup,
        lighting: &wgpu::BindGroup,
        global_materials: &wgpu::BindGroup,
        joints: &wgpu::BindGroup,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
        counter_buffer: &wgpu::Buffer,
        draw_count: u32,
        count_supported: bool,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let Some(attachment) = self.raster_attachment() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("visibility_buffer_runtime_raster_pass"),
            color_attachments: &[Some(attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.draw_raster(
            &mut pass,
            draw_bind_group,
            lighting,
            global_materials,
            joints,
            vertex_buffer,
            index_buffer,
            indirect_buffer,
            counter_buffer,
            draw_count,
            count_supported,
        );
        drop(pass);
        self.mark_raster_recorded();
    }

    pub(crate) fn record_reconstruct(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        if !self.reconstruction_enabled() {
            return;
        }
        let Some(resources) = self.resources.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("visibility_buffer_runtime_reconstruct_pass"),
            timestamp_writes,
        });
        pass.set_pipeline(
            self.reconstruct_pipeline
                .as_ref()
                .expect("enabled visibility runtime owns reconstruct pipeline"),
        );
        pass.set_bind_group(
            0,
            resources
                .reconstruct_bind_group
                .as_ref()
                .expect("reconstruction mode owns its bind group"),
            &[],
        );
        pass.dispatch_workgroups(
            resources.extent.0.div_ceil(WORKGROUP_SIZE),
            resources.extent.1.div_ceil(WORKGROUP_SIZE),
            1,
        );
        drop(pass);
    }

    pub(crate) fn record_debug_overlay(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        if !self.debug_overlay_enabled() || !self.frame_recorded {
            return;
        }
        let Some(resources) = self.resources.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("visibility_buffer_debug_overlay_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(
            self.overlay_pipeline
                .as_ref()
                .expect("enabled visibility runtime owns overlay pipeline"),
        );
        pass.set_bind_group(
            0,
            resources
                .overlay_bind_group
                .as_ref()
                .expect("debug mode owns its overlay bind group"),
            &[],
        );
        pass.draw(0..3, 0..1);
    }

    pub(crate) fn draw_shading<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        draw_bind_group: &'a wgpu::BindGroup,
        lighting: &'a wgpu::BindGroup,
        global_materials: &'a wgpu::BindGroup,
        joints: &'a wgpu::BindGroup,
    ) {
        if !self.shading_active() {
            return;
        }
        let Some(resources) = self.resources.as_ref() else {
            return;
        };
        pass.set_pipeline(
            self.shade_pipeline
                .as_ref()
                .expect("active visibility shading owns its pipeline"),
        );
        pass.set_bind_group(0, draw_bind_group, &[]);
        pass.set_bind_group(1, lighting, &[]);
        pass.set_bind_group(2, global_materials, &[]);
        pass.set_bind_group(3, joints, &[]);
        pass.set_bind_group(
            4,
            resources
                .shade_bind_group
                .as_ref()
                .expect("active visibility shading owns its bind group"),
            &[],
        );
        pass.draw(0..3, 0..1);
    }

    pub(crate) fn report_json(&self) -> String {
        let (width, height) = self
            .resources
            .as_ref()
            .map(|resources| resources.extent)
            .unwrap_or((0, 0));
        let bytes = target_bytes(width, height)
            .and_then(|visibility| {
                if self
                    .resources
                    .as_ref()
                    .is_some_and(|resources| resources._diagnostic_texture.is_some())
                {
                    (width as u64)
                        .checked_mul(height as u64)?
                        .checked_mul(DIAGNOSTIC_BYTES_PER_PIXEL)?
                        .checked_add(visibility)
                } else {
                    Some(visibility)
                }
            })
            .unwrap_or(0);
        let shading_active = self.shading_active();
        format!(
            concat!(
                "{{\"requested_mode\":\"{}\",\"enabled\":{},",
                "\"disabled_reason\":\"{}\",\"forward_authoritative\":{},",
                "\"composition\":\"{}\",\"pbr_shading\":{},",
                "\"eligible_draws\":{},\"compatibility_draws\":{},",
                "\"width\":{},\"height\":{},\"allocated_bytes\":{},",
                "\"debug_overlay\":{},\"frame_recorded\":{}}}"
            ),
            self.mode.name(),
            self.enabled,
            self.disabled_reason,
            !shading_active,
            if shading_active {
                "visibility-eligible+forward-compatibility"
            } else {
                "forward-authoritative"
            },
            shading_active,
            self.eligible_draws,
            self.compatibility_draws,
            width,
            height,
            bytes,
            self.debug_overlay_enabled(),
            self.frame_recorded,
        )
    }
}

pub(super) fn add_visibility_draw_id(gpu_scene_source: &str) -> String {
    const OUTPUT_END: &str = "    @location(8) @interpolate(flat) draw_flags: u32,\n};";
    const OUTPUT_WITH_ID: &str = concat!(
        "    @location(8) @interpolate(flat) draw_flags: u32,\n",
        "    @location(9) @interpolate(flat) draw_id: u32,\n",
        "};"
    );
    const SKINNED_RETURN: &str = concat!(
        "        o.draw_flags = bitcast<u32>(gpu_draw.bounds_min.w);\n",
        "        return o;"
    );
    const SKINNED_WITH_ID: &str = concat!(
        "        o.draw_flags = bitcast<u32>(gpu_draw.bounds_min.w);\n",
        "        o.draw_id = draw_index;\n",
        "        return o;"
    );
    const STATIC_RETURN: &str = concat!(
        "    out.draw_flags = bitcast<u32>(gpu_draw.bounds_min.w);\n",
        "    return out;"
    );
    const STATIC_WITH_ID: &str = concat!(
        "    out.draw_flags = bitcast<u32>(gpu_draw.bounds_min.w);\n",
        "    out.draw_id = draw_index;\n",
        "    return out;"
    );
    assert_eq!(gpu_scene_source.matches(OUTPUT_END).count(), 1);
    assert_eq!(gpu_scene_source.matches(SKINNED_RETURN).count(), 1);
    assert_eq!(gpu_scene_source.matches(STATIC_RETURN).count(), 1);
    gpu_scene_source
        .replace(OUTPUT_END, OUTPUT_WITH_ID)
        .replace(SKINNED_RETURN, SKINNED_WITH_ID)
        .replace(STATIC_RETURN, STATIC_WITH_ID)
}

fn make_visibility_raster_shader(gpu_scene_source: &str) -> String {
    let source = add_visibility_draw_id(gpu_scene_source);
    format!(
        "enable primitive_index;\n{source}\n{}",
        r#"
@fragment
fn fs_visibility_buffer(
    in: VertexOutputScene,
    @builtin(primitive_index) primitive_id: u32,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec2<u32> {
    // Bit 1 admits only rigid shared geometry. Wind/deforming records stay on
    // forward until the reconstruction pass owns identical deformation state.
    if ((in.draw_flags & 2u) == 0u) { discard; }
    if ((in.draw_flags & 1u) == 0u && !front_facing) { discard; }
    let face = select(0u, 0x80000000u, front_facing);
    return vec2<u32>(in.draw_id, primitive_id | face);
}
"#
    )
}

fn create_reconstruct_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("visibility_buffer_runtime_reconstruct_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage(1, true),
            storage(2, true),
            storage(3, true),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: DIAGNOSTIC_OUTPUT_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

fn create_overlay_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("visibility_buffer_debug_overlay_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    })
}

const RUNTIME_RECONSTRUCT_WGSL: &str = concat!(
    include_str!("../../shaders/visibility_buffer/reconstruct.wgsl"),
    include_str!("../../shaders/visibility_buffer/geometry.wgsl"),
    r#"
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
    misc: vec4<f32>,
};
struct GpuDrawRecord {
    uniforms: Uniforms3D,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    draw: vec4<u32>,
};
struct VertexTable { records: array<BloomPackedVertex3D>, };
struct IndexTable { values: array<u32>, };
struct DrawTable { records: array<GpuDrawRecord>, };

@group(0) @binding(0) var visibility_texture: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> vertices: VertexTable;
@group(0) @binding(2) var<storage, read> indices: IndexTable;
@group(0) @binding(3) var<storage, read> draws: DrawTable;
@group(0) @binding(4) var diagnostic_output: texture_storage_2d<rgba16float, write>;

fn visibility_fault(pixel: vec2<i32>) {
    textureStore(diagnostic_output, pixel, vec4<f32>(8.0, 0.0, 8.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn cs_visibility_reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dimensions = textureDimensions(visibility_texture);
    if (gid.x >= dimensions.x || gid.y >= dimensions.y) { return; }
    let pixel = vec2<i32>(gid.xy);
    let raw_visibility = textureLoad(visibility_texture, pixel, 0).xy;
    if (!bloom_visibility_valid(raw_visibility)) {
        textureStore(diagnostic_output, pixel, vec4<f32>(0.0));
        return;
    }
    let visibility = bloom_decode_visibility(raw_visibility);
    if (visibility.draw_id >= arrayLength(&draws.records)) {
        visibility_fault(pixel);
        return;
    }
    let draw = draws.records[visibility.draw_id];
    let primitive_offset = visibility.primitive_id * 3u;
    if (primitive_offset + 2u >= draw.draw.x) {
        visibility_fault(pixel);
        return;
    }
    let first_index = draw.draw.y + primitive_offset;
    if (first_index + 2u >= arrayLength(&indices.values)) {
        visibility_fault(pixel);
        return;
    }
    let base_vertex = bitcast<i32>(draw.draw.z);
    let signed0 = i32(indices.values[first_index]) + base_vertex;
    let signed1 = i32(indices.values[first_index + 1u]) + base_vertex;
    let signed2 = i32(indices.values[first_index + 2u]) + base_vertex;
    if (signed0 < 0 || signed1 < 0 || signed2 < 0) {
        visibility_fault(pixel);
        return;
    }
    let index0 = u32(signed0);
    let index1 = u32(signed1);
    let index2 = u32(signed2);
    let vertex_count = arrayLength(&vertices.records);
    if (index0 >= vertex_count || index1 >= vertex_count || index2 >= vertex_count) {
        visibility_fault(pixel);
        return;
    }
    let vertex0 = bloom_decode_vertex3d(vertices.records[index0]);
    let vertex1 = bloom_decode_vertex3d(vertices.records[index1]);
    let vertex2 = bloom_decode_vertex3d(vertices.records[index2]);
    let clip0 = draw.uniforms.mvp * vec4<f32>(vertex0.position, 1.0);
    let clip1 = draw.uniforms.mvp * vec4<f32>(vertex1.position, 1.0);
    let clip2 = draw.uniforms.mvp * vec4<f32>(vertex2.position, 1.0);
    let point_ndc = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(dimensions.x) * 2.0 - 1.0,
        1.0 - (f32(gid.y) + 0.5) / f32(dimensions.y) * 2.0,
    );
    let bary = bloom_perspective_barycentrics(point_ndc, clip0, clip1, clip2);
    let object_normal = bloom_interpolate3(
        vertex0.normal,
        vertex1.normal,
        vertex2.normal,
        bary,
    );
    var world_normal = normalize((draw.uniforms.model * vec4<f32>(object_normal, 0.0)).xyz);
    if (!visibility.front_facing) { world_normal = -world_normal; }
    let normal_color = world_normal * 0.5 + vec3<f32>(0.5);
    textureStore(diagnostic_output, pixel, vec4<f32>(normal_color, 1.0));
}
"#,
);

const DEBUG_OVERLAY_WGSL: &str = r#"
@group(0) @binding(0) var diagnostic_texture: texture_2d<f32>;

struct DebugVertexOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_debug_overlay(@builtin(vertex_index) vertex_index: u32) -> DebugVertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: DebugVertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_debug_overlay(in: DebugVertexOut) -> @location(0) vec4<f32> {
    let reconstructed = textureLoad(diagnostic_texture, vec2<i32>(in.position.xy), 0);
    if (reconstructed.a == 0.0) { discard; }
    return vec4<f32>(reconstructed.rgb, 1.0);
}
"#;

#[cfg(test)]
fn screen_barycentrics(point: [f32; 2], triangle: [[f32; 2]; 3]) -> Option<[f32; 3]> {
    let edge = |a: [f32; 2], b: [f32; 2], p: [f32; 2]| {
        (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    };
    let area = edge(triangle[1], triangle[2], triangle[0]);
    if area.abs() <= 1.0e-12 {
        return None;
    }
    Some([
        edge(triangle[1], triangle[2], point) / area,
        edge(triangle[2], triangle[0], point) / area,
        edge(triangle[0], triangle[1], point) / area,
    ])
}

#[cfg(test)]
fn perspective_barycentrics(point: [f32; 2], clip: [[f32; 4]; 3]) -> Option<[f32; 3]> {
    if clip.iter().any(|vertex| vertex[3].abs() <= 1.0e-12) {
        return None;
    }
    let ndc = [
        [clip[0][0] / clip[0][3], clip[0][1] / clip[0][3]],
        [clip[1][0] / clip[1][3], clip[1][1] / clip[1][3]],
        [clip[2][0] / clip[2][3], clip[2][1] / clip[2][3]],
    ];
    let linear = screen_barycentrics(point, ndc)?;
    let weighted = [
        linear[0] / clip[0][3],
        linear[1] / clip[1][3],
        linear[2] / clip[2][3],
    ];
    let sum = weighted[0] + weighted[1] + weighted[2];
    if sum.abs() <= 1.0e-12 {
        return None;
    }
    Some([weighted[0] / sum, weighted[1] / sum, weighted[2] / sum])
}

#[cfg(test)]
mod tests {
    use super::super::{gpu_driven::GpuDrawRecord, Uniforms3D, Vertex3D};
    use super::*;
    use wgpu::util::DeviceExt;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn packed_record_is_exactly_one_rg32uint_texel() {
        assert_eq!(std::mem::size_of::<VisibilityRecord>(), 8);
        assert_eq!(std::mem::align_of::<VisibilityRecord>(), 4);
        assert_eq!(VISIBILITY_BYTES_PER_PIXEL, 8);
        assert_eq!(VISIBILITY_FORMAT, wgpu::TextureFormat::Rg32Uint);
        assert_eq!(target_bytes(1_920, 1_080), Some(16_588_800));
        assert_eq!(target_bytes(u32::MAX, u32::MAX), None);

        let report = contract_json();
        assert!(report.starts_with("{\"format\":\"rg32uint\""));
        assert!(report.contains("\"native_1080p_bytes\":16588800"));
        assert!(report.contains("\"required_feature\":\"primitive-index\""));
        assert!(report.contains("\"vertex_stride_bytes\":96"));
        assert!(report.contains("\"shipping_enabled\":false"));
    }

    #[test]
    fn ids_and_face_orientation_round_trip_without_background_collision() {
        for (draw, primitive, front) in [
            (0, 0, false),
            (17, 42, true),
            (u32::MAX - 1, PRIMITIVE_ID_MASK, false),
        ] {
            let encoded = VisibilityRecord::encode(draw, primitive, front).unwrap();
            assert_eq!(encoded.decode(), Some((draw, primitive, front)));
        }
        assert_eq!(VisibilityRecord::BACKGROUND.decode(), None);
        assert_eq!(VisibilityRecord::encode(INVALID_DRAW_ID, 0, true), None);
        assert_eq!(VisibilityRecord::encode(0, FRONT_FACE_BIT, true), None);
    }

    #[test]
    fn perspective_reconstruction_matches_vertices_and_known_depth_weighting() {
        let clip = [
            [-1.0, -1.0, 0.2, 1.0],
            [2.0, -2.0, 0.4, 2.0],
            [0.0, 4.0, 0.8, 4.0],
        ];
        for (point, expected) in [
            ([-1.0, -1.0], [1.0, 0.0, 0.0]),
            ([1.0, -1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0], [0.0, 0.0, 1.0]),
        ] {
            let actual = perspective_barycentrics(point, clip).unwrap();
            for lane in 0..3 {
                assert_close(actual[lane], expected[lane]);
            }
        }

        let center = perspective_barycentrics([0.0, -1.0 / 3.0], clip).unwrap();
        assert_close(center[0], 4.0 / 7.0);
        assert_close(center[1], 2.0 / 7.0);
        assert_close(center[2], 1.0 / 7.0);
        assert_close(center.iter().sum(), 1.0);
    }

    #[test]
    fn shared_reconstruction_header_parses_and_keeps_the_cpu_abi_constants() {
        wgpu::naga::front::wgsl::parse_str(RECONSTRUCTION_WGSL)
            .unwrap_or_else(|error| panic!("visibility reconstruction WGSL failed: {error:?}"));
        wgpu::naga::front::wgsl::parse_str(GEOMETRY_WGSL)
            .unwrap_or_else(|error| panic!("visibility geometry WGSL failed: {error:?}"));
        assert!(RECONSTRUCTION_WGSL
            .contains("const BLOOM_VISIBILITY_FRONT_FACE_BIT: u32 = 0x80000000u"));
        assert!(RECONSTRUCTION_WGSL.contains("fn bloom_perspective_barycentrics("));
        assert!(GEOMETRY_WGSL.contains("const BLOOM_VERTEX3D_WORDS: u32 = 24u"));
        assert_eq!(std::mem::size_of::<Vertex3D>(), 96);
    }

    #[test]
    fn runtime_modes_request_only_the_explicit_optional_feature() {
        assert_eq!(parse_runtime_mode(None), RuntimeMode::Off);
        assert_eq!(parse_runtime_mode(Some("off")), RuntimeMode::Off);
        assert_eq!(parse_runtime_mode(Some("validate")), RuntimeMode::Validate);
        assert_eq!(parse_runtime_mode(Some("DEBUG")), RuntimeMode::Debug);
        assert_eq!(parse_runtime_mode(Some("pbr")), RuntimeMode::Shade);
        assert!(RuntimeMode::Shade.shades());

        let supported = wgpu::Features::PRIMITIVE_INDEX | wgpu::Features::TIMESTAMP_QUERY;
        let mut required = wgpu::Features::empty();
        request_feature_for_mode(RuntimeMode::Off, supported, &mut required);
        assert!(required.is_empty());
        request_feature_for_mode(RuntimeMode::Validate, supported, &mut required);
        assert_eq!(required, wgpu::Features::PRIMITIVE_INDEX);
        request_feature_for_mode(RuntimeMode::Shade, supported, &mut required);
        assert_eq!(required, wgpu::Features::PRIMITIVE_INDEX);

        let mut unsupported = wgpu::Features::empty();
        request_feature_for_mode(
            RuntimeMode::Debug,
            wgpu::Features::TIMESTAMP_QUERY,
            &mut unsupported,
        );
        assert!(unsupported.is_empty());

        let disabled = VisibilityBufferRuntime::disabled(RuntimeMode::Off, "not-requested");
        assert!(!disabled.enabled());
        assert!(disabled.resources.is_none());
        assert!(disabled.report_json().contains("\"allocated_bytes\":0"));
    }

    #[test]
    fn runtime_raster_reconstruction_and_overlay_shaders_parse() {
        let generated =
            super::super::gpu_driven::make_gpu_scene_shader(super::super::shaders::SCENE_SHADER);
        let raster = make_visibility_raster_shader(&generated);
        let depth = super::super::visibility_shading::make_visibility_depth_shader(&generated);
        wgpu::naga::front::wgsl::parse_str(&raster)
            .unwrap_or_else(|error| panic!("visibility runtime raster WGSL failed: {error:?}"));
        wgpu::naga::front::wgsl::parse_str(&depth)
            .unwrap_or_else(|error| panic!("visibility depth WGSL failed: {error:?}"));
        wgpu::naga::front::wgsl::parse_str(RUNTIME_RECONSTRUCT_WGSL).unwrap_or_else(|error| {
            panic!("visibility runtime reconstruction WGSL failed: {error:?}")
        });
        wgpu::naga::front::wgsl::parse_str(DEBUG_OVERLAY_WGSL)
            .unwrap_or_else(|error| panic!("visibility debug overlay WGSL failed: {error:?}"));
        assert!(raster.starts_with("enable primitive_index;"));
        assert!(raster.contains("out.draw_id = draw_index"));
        assert!(raster.contains("(in.draw_flags & 2u) == 0u"));
        assert!(depth.contains("return vec2<u32>(0xffffffffu, 0xffffffffu)"));
        assert!(RUNTIME_RECONSTRUCT_WGSL.contains("arrayLength(&vertices.records)"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn try_device(required_features: wgpu::Features) -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        if !adapter.features().contains(required_features) {
            eprintln!("adapter lacks required visibility-oracle features");
            return None;
        }
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("visibility_buffer_oracle_device"),
            required_features,
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .ok()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver
            .recv()
            .expect("visibility readback callback dropped")
            .expect("visibility readback mapping failed");
        let mapped = slice.get_mapped_range();
        let bytes = mapped.to_vec();
        drop(mapped);
        buffer.unmap();
        bytes
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_raster_ids_faces_and_reconstruction_match_the_cpu_oracle() {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 16;
        const VISIBILITY_ROW_BYTES: u32 = 256;
        const BARYCENTRIC_ROW_BYTES: u32 = WIDTH * 16;
        let Some((device, queue)) = try_device(wgpu::Features::PRIMITIVE_INDEX) else {
            eprintln!("no GPU adapter — skipping visibility raster oracle");
            return;
        };

        let clip = [
            [-0.9, -0.8, 0.5, 1.0],
            [-0.2, -1.6, 1.0, 2.0],
            [-2.0, 3.2, 2.0, 4.0],
            [0.1, -0.8, 0.5, 1.0],
            [2.0, 3.2, 2.0, 4.0],
            [1.8, -1.6, 1.0, 2.0],
        ];
        let shader_source = format!(
            "enable primitive_index;\n\
             {RECONSTRUCTION_WGSL}\n\
             struct VertexOut {{ @builtin(position) position: vec4<f32>, }};\n\
             struct FragmentOut {{\n\
               @location(0) visibility: vec2<u32>,\n\
               @location(1) barycentrics: vec4<f32>,\n\
             }};\n\
             fn clip_position(index: u32) -> vec4<f32> {{\n\
               var positions = array<vec4<f32>, 6>(\n\
                 vec4<f32>(-0.9, -0.8, 0.5, 1.0),\n\
                 vec4<f32>(-0.2, -1.6, 1.0, 2.0),\n\
                 vec4<f32>(-2.0, 3.2, 2.0, 4.0),\n\
                 vec4<f32>(0.1, -0.8, 0.5, 1.0),\n\
                 vec4<f32>(2.0, 3.2, 2.0, 4.0),\n\
                 vec4<f32>(1.8, -1.6, 1.0, 2.0),\n\
               );\n\
               return positions[index];\n\
             }}\n\
             @vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {{\n\
               var out: VertexOut;\n\
               out.position = clip_position(index);\n\
               return out;\n\
             }}\n\
             @fragment fn fs_main(\n\
               in: VertexOut,\n\
               @builtin(primitive_index) primitive_id: u32,\n\
               @builtin(front_facing) front_facing: bool,\n\
             ) -> FragmentOut {{\n\
               let first = primitive_id * 3u;\n\
               let point_ndc = vec2<f32>(\n\
                 in.position.x / {WIDTH}.0 * 2.0 - 1.0,\n\
                 1.0 - in.position.y / {HEIGHT}.0 * 2.0,\n\
               );\n\
               let barycentrics = bloom_perspective_barycentrics(\n\
                 point_ndc,\n\
                 clip_position(first),\n\
                 clip_position(first + 1u),\n\
                 clip_position(first + 2u),\n\
               );\n\
               var out: FragmentOut;\n\
               out.visibility = bloom_encode_visibility(7u, primitive_id, front_facing);\n\
               out.barycentrics = vec4<f32>(barycentrics, 1.0);\n\
               return out;\n\
             }}"
        );
        wgpu::naga::front::wgsl::parse_str(&shader_source)
            .unwrap_or_else(|error| panic!("visibility raster oracle WGSL failed: {error:?}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visibility_buffer_oracle_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visibility_buffer_oracle_pipeline_layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("visibility_buffer_oracle_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: VISIBILITY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
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
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let make_target = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let visibility = make_target("visibility_buffer_oracle_ids", VISIBILITY_FORMAT);
        let barycentrics = make_target(
            "visibility_buffer_oracle_barycentrics",
            wgpu::TextureFormat::Rgba32Float,
        );
        let visibility_view = visibility.create_view(&Default::default());
        let barycentric_view = barycentrics.create_view(&Default::default());
        let visibility_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_buffer_oracle_id_readback"),
            size: (VISIBILITY_ROW_BYTES * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let barycentric_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_buffer_oracle_barycentric_readback"),
            size: (BARYCENTRIC_ROW_BYTES * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("visibility_buffer_oracle_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("visibility_buffer_oracle_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &visibility_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: u32::MAX as f64,
                                g: u32::MAX as f64,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &barycentric_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..6, 0..1);
        }
        for (texture, buffer, bytes_per_row) in [
            (&visibility, &visibility_readback, VISIBILITY_ROW_BYTES),
            (&barycentrics, &barycentric_readback, BARYCENTRIC_ROW_BYTES),
        ] {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(HEIGHT),
                    },
                },
                wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(std::iter::once(encoder.finish()));

        let id_bytes = readback(&device, &visibility_readback);
        let barycentric_bytes = readback(&device, &barycentric_readback);
        let mut primitive_pixels = [0usize; 2];
        let mut primitive_faces = [None; 2];
        let mut background_pixels = 0usize;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let id_offset = (y * VISIBILITY_ROW_BYTES + x * 8) as usize;
                let words: &[u32] = bytemuck::cast_slice(&id_bytes[id_offset..id_offset + 8]);
                let record = VisibilityRecord {
                    draw_id: words[0],
                    primitive_and_face: words[1],
                };
                let Some((draw_id, primitive_id, front_facing)) = record.decode() else {
                    background_pixels += 1;
                    continue;
                };
                assert_eq!(draw_id, 7);
                assert!(primitive_id < 2);
                let primitive = primitive_id as usize;
                primitive_pixels[primitive] += 1;
                match primitive_faces[primitive] {
                    Some(expected) => assert_eq!(front_facing, expected),
                    None => primitive_faces[primitive] = Some(front_facing),
                }

                let point_ndc = [
                    (x as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0,
                    1.0 - (y as f32 + 0.5) / HEIGHT as f32 * 2.0,
                ];
                let first = primitive * 3;
                let expected = perspective_barycentrics(
                    point_ndc,
                    [clip[first], clip[first + 1], clip[first + 2]],
                )
                .unwrap();
                let bary_offset = (y * BARYCENTRIC_ROW_BYTES + x * 16) as usize;
                let actual: &[f32] =
                    bytemuck::cast_slice(&barycentric_bytes[bary_offset..bary_offset + 16]);
                for lane in 0..3 {
                    assert!(
                        (actual[lane] - expected[lane]).abs() <= 2.0e-5,
                        "pixel ({x},{y}) primitive {primitive}: GPU {:?}, CPU {:?}",
                        &actual[..3],
                        expected,
                    );
                }
                assert_close(actual[0] + actual[1] + actual[2], 1.0);
            }
        }
        assert!(background_pixels > 0, "clear sentinel was not preserved");
        assert!(primitive_pixels.iter().all(|pixels| *pixels > 0));
        assert_ne!(
            primitive_faces[0], primitive_faces[1],
            "opposite winding must preserve distinct front-face bits"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_pulls_shared_geometry_and_reconstructs_every_vertex_lane() {
        const OUTPUT_WORDS: usize = 100;
        let Some((device, queue)) = try_device(wgpu::Features::empty()) else {
            eprintln!("no GPU adapter — skipping shared-geometry oracle");
            return;
        };

        let padding = Vertex3D {
            position: [-99.0; 3],
            normal: [-98.0; 3],
            color: [-97.0; 4],
            uv: [-96.0; 2],
            joints: [-95.0; 4],
            weights: [-94.0; 4],
            tangent: [-93.0; 4],
        };
        let vertices = [
            padding,
            Vertex3D {
                position: [-0.9, -0.8, 1.0],
                normal: [0.1, 0.2, 0.3],
                color: [0.4, 0.5, 0.6, 0.7],
                uv: [0.8, 0.9],
                joints: [1.0, 2.0, 3.0, 4.0],
                weights: [0.1, 0.2, 0.3, 0.4],
                tangent: [0.7, 0.2, 0.1, -1.0],
            },
            Vertex3D {
                position: [-0.2, -1.6, 2.0],
                normal: [1.1, 1.2, 1.3],
                color: [1.4, 1.5, 1.6, 1.7],
                uv: [1.8, 1.9],
                joints: [5.0, 6.0, 7.0, 8.0],
                weights: [0.4, 0.3, 0.2, 0.1],
                tangent: [0.1, 0.6, 0.3, 1.0],
            },
            Vertex3D {
                position: [-2.0, 3.2, 4.0],
                normal: [2.1, 2.2, 2.3],
                color: [2.4, 2.5, 2.6, 2.7],
                uv: [2.8, 2.9],
                joints: [9.0, 10.0, 11.0, 12.0],
                weights: [0.25, 0.25, 0.25, 0.25],
                tangent: [0.4, 0.2, 0.8, -1.0],
            },
        ];
        let mvp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5, 0.0],
        ];
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let draw = GpuDrawRecord {
            uniforms: Uniforms3D {
                mvp,
                model: identity,
                prev_mvp: identity,
                model_tint: [1.0; 4],
                misc: [0.0; 4],
            },
            bounds_min: [-2.0, -1.6, 1.0, 0.0],
            bounds_max: [-0.2, 3.2, 4.0, 0.0],
            draw: [3, 3, 1_i32 as u32, 1_234],
        };
        let indices = [91u32, 92, 93, 0, 1, 2];
        let point_ndc = [-0.45f32, -0.1, 0.0, 0.0];
        let visibility_record = VisibilityRecord::encode(0, 0, true).unwrap();

        let shader_source = [
            RECONSTRUCTION_WGSL,
            GEOMETRY_WGSL,
            r#"
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
    misc: vec4<f32>,
};
struct GpuDrawRecord {
    uniforms: Uniforms3D,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    draw: vec4<u32>,
};
struct VertexTable { records: array<BloomPackedVertex3D>, };
struct IndexTable { values: array<u32>, };
struct DrawTable { records: array<GpuDrawRecord>, };
struct OutputTable { words: array<u32>, };

@group(0) @binding(0) var visibility_texture: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> vertices: VertexTable;
@group(0) @binding(2) var<storage, read> indices: IndexTable;
@group(0) @binding(3) var<storage, read> draws: DrawTable;
@group(0) @binding(4) var<uniform> point_ndc: vec4<f32>;
@group(0) @binding(5) var<storage, read_write> output: OutputTable;

fn write_vertex(offset: u32, vertex: BloomVertex3D) {
    output.words[offset + 0u] = bitcast<u32>(vertex.position.x);
    output.words[offset + 1u] = bitcast<u32>(vertex.position.y);
    output.words[offset + 2u] = bitcast<u32>(vertex.position.z);
    output.words[offset + 3u] = bitcast<u32>(vertex.normal.x);
    output.words[offset + 4u] = bitcast<u32>(vertex.normal.y);
    output.words[offset + 5u] = bitcast<u32>(vertex.normal.z);
    output.words[offset + 6u] = bitcast<u32>(vertex.color.x);
    output.words[offset + 7u] = bitcast<u32>(vertex.color.y);
    output.words[offset + 8u] = bitcast<u32>(vertex.color.z);
    output.words[offset + 9u] = bitcast<u32>(vertex.color.w);
    output.words[offset + 10u] = bitcast<u32>(vertex.uv.x);
    output.words[offset + 11u] = bitcast<u32>(vertex.uv.y);
    output.words[offset + 12u] = bitcast<u32>(vertex.joints.x);
    output.words[offset + 13u] = bitcast<u32>(vertex.joints.y);
    output.words[offset + 14u] = bitcast<u32>(vertex.joints.z);
    output.words[offset + 15u] = bitcast<u32>(vertex.joints.w);
    output.words[offset + 16u] = bitcast<u32>(vertex.weights.x);
    output.words[offset + 17u] = bitcast<u32>(vertex.weights.y);
    output.words[offset + 18u] = bitcast<u32>(vertex.weights.z);
    output.words[offset + 19u] = bitcast<u32>(vertex.weights.w);
    output.words[offset + 20u] = bitcast<u32>(vertex.tangent.x);
    output.words[offset + 21u] = bitcast<u32>(vertex.tangent.y);
    output.words[offset + 22u] = bitcast<u32>(vertex.tangent.z);
    output.words[offset + 23u] = bitcast<u32>(vertex.tangent.w);
}

@compute @workgroup_size(1)
fn cs_main() {
    let raw_visibility = textureLoad(visibility_texture, vec2<i32>(0, 0), 0).xy;
    if (!bloom_visibility_valid(raw_visibility)) {
        output.words[96] = BLOOM_VISIBILITY_INVALID_DRAW_ID;
        return;
    }
    let visibility = bloom_decode_visibility(raw_visibility);
    let draw = draws.records[visibility.draw_id];
    let first_index = draw.draw.y + visibility.primitive_id * 3u;
    let base_vertex = bitcast<i32>(draw.draw.z);
    let index0 = u32(i32(indices.values[first_index]) + base_vertex);
    let index1 = u32(i32(indices.values[first_index + 1u]) + base_vertex);
    let index2 = u32(i32(indices.values[first_index + 2u]) + base_vertex);
    let vertex0 = bloom_decode_vertex3d(vertices.records[index0]);
    let vertex1 = bloom_decode_vertex3d(vertices.records[index1]);
    let vertex2 = bloom_decode_vertex3d(vertices.records[index2]);
    write_vertex(0u, vertex0);
    write_vertex(24u, vertex1);
    write_vertex(48u, vertex2);

    let clip0 = draw.uniforms.mvp * vec4<f32>(vertex0.position, 1.0);
    let clip1 = draw.uniforms.mvp * vec4<f32>(vertex1.position, 1.0);
    let clip2 = draw.uniforms.mvp * vec4<f32>(vertex2.position, 1.0);
    let bary = bloom_perspective_barycentrics(point_ndc.xy, clip0, clip1, clip2);
    let interpolated = BloomVertex3D(
        bloom_interpolate3(vertex0.position, vertex1.position, vertex2.position, bary),
        bloom_interpolate3(vertex0.normal, vertex1.normal, vertex2.normal, bary),
        bloom_interpolate4(vertex0.color, vertex1.color, vertex2.color, bary),
        bloom_interpolate2(vertex0.uv, vertex1.uv, vertex2.uv, bary),
        bloom_interpolate4(vertex0.joints, vertex1.joints, vertex2.joints, bary),
        bloom_interpolate4(vertex0.weights, vertex1.weights, vertex2.weights, bary),
        bloom_interpolate4(vertex0.tangent, vertex1.tangent, vertex2.tangent, bary),
    );
    write_vertex(72u, interpolated);
    output.words[96] = visibility.draw_id;
    output.words[97] = visibility.primitive_id;
    output.words[98] = select(0u, 1u, visibility.front_facing);
    output.words[99] = draw.draw.w;
}
"#,
        ]
        .concat();
        wgpu::naga::front::wgsl::parse_str(&shader_source)
            .unwrap_or_else(|error| panic!("shared-geometry oracle WGSL failed: {error:?}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visibility_shared_geometry_oracle_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("visibility_shared_geometry_oracle_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let visibility = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visibility_shared_geometry_oracle_ids"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VISIBILITY_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &visibility,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::bytes_of(&visibility_record),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(VISIBILITY_BYTES_PER_PIXEL as u32),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let make_storage = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let vertex_buffer = make_storage(
            "visibility_shared_geometry_oracle_vertices",
            bytemuck::cast_slice(&vertices),
        );
        let index_buffer = make_storage(
            "visibility_shared_geometry_oracle_indices",
            bytemuck::cast_slice(&indices),
        );
        let draw_buffer = make_storage(
            "visibility_shared_geometry_oracle_draws",
            bytemuck::bytes_of(&draw),
        );
        let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visibility_shared_geometry_oracle_point"),
            contents: bytemuck::cast_slice(&point_ndc),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_shared_geometry_oracle_output"),
            size: (OUTPUT_WORDS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_shared_geometry_oracle_readback"),
            size: (OUTPUT_WORDS * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let visibility_view = visibility.create_view(&Default::default());
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility_shared_geometry_oracle_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&visibility_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: draw_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("visibility_shared_geometry_oracle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visibility_shared_geometry_oracle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &readback_buffer,
            0,
            (OUTPUT_WORDS * 4) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let output_bytes = readback(&device, &readback_buffer);
        let output: &[u32] = bytemuck::cast_slice(&output_bytes);
        let expected_raw: &[u32] = bytemuck::cast_slice(&vertices[1..]);
        assert_eq!(&output[..72], expected_raw);

        let clip = [
            [-0.9, -0.8, 0.5, 1.0],
            [-0.2, -1.6, 0.5, 2.0],
            [-2.0, 3.2, 0.5, 4.0],
        ];
        let bary = perspective_barycentrics([point_ndc[0], point_ndc[1]], clip).unwrap();
        let source: &[f32] = bytemuck::cast_slice(&vertices[1..]);
        for lane in 0..24 {
            let expected =
                source[lane] * bary[0] + source[24 + lane] * bary[1] + source[48 + lane] * bary[2];
            let actual = f32::from_bits(output[72 + lane]);
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "interpolated Vertex3D lane {lane}: GPU {actual}, CPU {expected}"
            );
        }
        assert_eq!(&output[96..100], &[0, 0, 1, 1_234]);
    }
}
