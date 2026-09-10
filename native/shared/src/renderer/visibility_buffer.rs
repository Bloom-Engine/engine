//! Packed visibility-buffer contract for the #27 qualification path.
//!
//! This module does not enable a shipping render path. It locks the 8-byte
//! target ABI and reconstruction math that an opt-in A/B implementation will
//! use. The existing forward MRT remains authoritative until total frame cost
//! and image parity pass on every required capability tier.

#[cfg(test)]
pub(crate) use super::visibility_ids::FRONT_FACE_BIT;
pub(crate) use super::visibility_ids::{
    VisibilityDraw, VisibilityRecord, DRAW_INDEX_MASK, INVALID_DRAW_ID, PRIMITIVE_ID_MASK,
    VIRTUAL_DRAW_BIT, VISIBILITY_BYTES_PER_PIXEL, VISIBILITY_FORMAT,
};

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
    let virtual_record = VisibilityRecord::encode_virtual(0, 0, true)
        .expect("virtual draw zero must remain encodable");
    let virtual_namespace_valid =
        virtual_record.decode_draw() == Some((VisibilityDraw::Virtual(0), 0, true));
    debug_assert_eq!(background.decode(), None);
    debug_assert_eq!(max_record.decode(), Some((0, PRIMITIVE_ID_MASK, true)));
    debug_assert!(virtual_namespace_valid);
    format!(
        concat!(
            "{{\"format\":\"{}\",\"bytes_per_pixel\":{},",
            "\"invalid_draw_id\":{},\"primitive_bits\":31,",
            "\"front_face_bits\":1,\"draw_namespace_bits\":1,",
            "\"virtual_draw_bit\":{},\"draw_index_mask\":{},",
            "\"virtual_namespace_valid\":{},",
            "\"shipping_enabled\":false,",
            "\"required_feature\":\"primitive-index\",",
            "\"vertex_stride_bytes\":{},\"native_1080p_bytes\":{},",
            "\"reconstruction_wgsl_bytes\":{},\"geometry_wgsl_bytes\":{},",
            "\"activation\":\"opt-in A/B qualification required\"}}"
        ),
        format_name,
        VISIBILITY_BYTES_PER_PIXEL,
        INVALID_DRAW_ID,
        VIRTUAL_DRAW_BIT,
        DRAW_INDEX_MASK,
        virtual_namespace_valid,
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
    vertex_bytes: u64,
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

const VISIBILITY_VERTEX_SEGMENTS: usize = 3;

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn visibility_vertex_binding_ranges(
    limits: &wgpu::Limits,
    buffer_bytes: u64,
    used_bytes: u64,
) -> [(u64, u64); VISIBILITY_VERTEX_SEGMENTS] {
    let vertex_stride = std::mem::size_of::<super::Vertex3D>() as u64;
    let binding_alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(1);
    let segment_alignment =
        vertex_stride / gcd(vertex_stride, binding_alignment) * binding_alignment;
    let maximum_binding_bytes = u64::from(limits.max_storage_buffer_binding_size);
    let segment_bytes = maximum_binding_bytes / segment_alignment * segment_alignment;
    assert!(
        segment_bytes >= vertex_stride,
        "visibility vertex ABI requires a {vertex_stride}-byte aligned storage window, but the device limit is {maximum_binding_bytes} bytes"
    );
    assert!(
        used_bytes <= buffer_bytes,
        "visibility vertex usage {used_bytes} exceeds its {buffer_bytes}-byte arena"
    );
    assert!(
        used_bytes <= segment_bytes * VISIBILITY_VERTEX_SEGMENTS as u64,
        "visibility vertex usage {used_bytes} exceeds three device-sized storage windows of {segment_bytes} bytes"
    );
    let bound_bytes = used_bytes.max(vertex_stride);
    std::array::from_fn(|segment| {
        let offset = segment as u64 * segment_bytes;
        if offset < bound_bytes {
            let size = segment_bytes.min(bound_bytes - offset);
            debug_assert_eq!(offset % binding_alignment, 0);
            debug_assert_eq!(size % vertex_stride, 0);
            (offset, size)
        } else {
            // Every declared layout entry needs a non-empty resource. An
            // unreachable one-record alias keeps small scenes on the same ABI.
            (0, vertex_stride)
        }
    })
}

fn visibility_vertex_bindings<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
    used_bytes: u64,
) -> [wgpu::BufferBinding<'a>; VISIBILITY_VERTEX_SEGMENTS] {
    visibility_vertex_binding_ranges(&device.limits(), buffer.size(), used_bytes).map(
        |(offset, size)| wgpu::BufferBinding {
            buffer,
            offset,
            size: std::num::NonZeroU64::new(size),
        },
    )
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
        vertex_bytes: u64,
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
                && resources.vertex_bytes == vertex_bytes
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
        let vertex_bindings = visibility_vertex_bindings(device, vertex_buffer, vertex_bytes);
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
                        resource: wgpu::BindingResource::Buffer(vertex_bindings[0].clone()),
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
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(vertex_bindings[1].clone()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Buffer(vertex_bindings[2].clone()),
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
                            resource: wgpu::BindingResource::Buffer(vertex_bindings[0].clone()),
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
                            resource: wgpu::BindingResource::Buffer(vertex_bindings[1].clone()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Buffer(vertex_bindings[2].clone()),
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
            vertex_bytes,
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

    #[cfg(feature = "models3d")]
    pub(crate) fn visibility_texture(&self) -> Option<&wgpu::Texture> {
        self.resources
            .as_ref()
            .map(|resources| &resources._visibility_texture)
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
            storage(5, true),
            storage(6, true),
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
@group(0) @binding(1) var<storage, read> vertices_0: VertexTable;
@group(0) @binding(2) var<storage, read> indices: IndexTable;
@group(0) @binding(3) var<storage, read> draws: DrawTable;
@group(0) @binding(4) var diagnostic_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> vertices_1: VertexTable;
@group(0) @binding(6) var<storage, read> vertices_2: VertexTable;

fn visibility_vertex_count() -> u32 {
    return arrayLength(&vertices_0.records)
        + arrayLength(&vertices_1.records)
        + arrayLength(&vertices_2.records);
}

fn visibility_vertex(index: u32) -> BloomVertex3D {
    let count_0 = arrayLength(&vertices_0.records);
    if (index < count_0) {
        return bloom_decode_vertex3d(vertices_0.records[index]);
    }
    var local_index = index - count_0;
    let count_1 = arrayLength(&vertices_1.records);
    if (local_index < count_1) {
        return bloom_decode_vertex3d(vertices_1.records[local_index]);
    }
    local_index -= count_1;
    return bloom_decode_vertex3d(vertices_2.records[local_index]);
}

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
    let vertex_count = visibility_vertex_count();
    if (index0 >= vertex_count || index1 >= vertex_count || index2 >= vertex_count) {
        visibility_fault(pixel);
        return;
    }
    let vertex0 = visibility_vertex(index0);
    let vertex1 = visibility_vertex(index1);
    let vertex2 = visibility_vertex(index2);
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
#[path = "visibility_buffer_tests.rs"]
mod tests;
