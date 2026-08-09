//! Lazy temporal-reactive coverage for imported transparency, transmission,
//! and custom translucent materials that author an `fs_reactive` entry point.
//!
//! The established TAA, sorted-alpha, refraction, and weighted-OIT pipelines
//! remain untouched. These variants are compiled only after a TAA frame has a
//! visible imported draw whose current color depends on transparency coverage
//! or the current opaque scene behind it.

use super::*;

pub(super) const TEMPORAL_REACTIVE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

pub(super) fn temporal_reactive_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("BLOOM_TEMPORAL_REACTIVE")
                .unwrap_or_else(|_| "on".to_owned())
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "off" | "false" | "disabled"
        )
    })
}

pub(super) fn temporal_reactive_selected(
    taa_enabled: bool,
    imported_transparency_draw_count: usize,
    custom_reactive_draw: impl FnOnce() -> bool,
    imported_refraction_draw_count: impl FnOnce() -> usize,
) -> bool {
    temporal_reactive_selected_for(
        temporal_reactive_enabled(),
        taa_enabled,
        imported_transparency_draw_count,
        custom_reactive_draw,
        imported_refraction_draw_count,
    )
}

fn temporal_reactive_selected_for(
    feature_enabled: bool,
    taa_enabled: bool,
    imported_transparency_draw_count: usize,
    custom_reactive_draw: impl FnOnce() -> bool,
    imported_refraction_draw_count: impl FnOnce() -> usize,
) -> bool {
    feature_enabled
        && taa_enabled
        && (imported_transparency_draw_count > 0
            || custom_reactive_draw()
            || imported_refraction_draw_count() > 0)
}

pub(super) fn reactive_union_blend() -> wgpu::BlendState {
    let union = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrc,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: union,
        alpha: union,
    }
}

pub(super) fn create_taa_bind_group_layout(
    device: &wgpu::Device,
    label: &'static str,
    reactive: bool,
) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
    ];
    if !reactive {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
    }
    let mut reactive_entries = entries.to_vec();
    reactive_entries.push(wgpu::BindGroupLayoutEntry {
        binding: 10,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &reactive_entries,
    })
}

pub(super) fn taa_reactive_shader_source() -> String {
    let source = TAA_SHADER_WGSL.replacen(
        "@group(0) @binding(9) var history_depth_tex: texture_2d<f32>;",
        "@group(0) @binding(9) var history_depth_tex: texture_2d<f32>;\n\
         @group(0) @binding(10) var reactive_tex: texture_2d<f32>;",
        1,
    );
    assert_ne!(
        source, TAA_SHADER_WGSL,
        "TAA binding declarations changed; reactive injection must be updated"
    );
    let source = source.replacen(
        "    let reactive = 0.0;",
        "    // The mask follows the same unjittered current-frame coordinate as\n\
         // the color sample. Coverage is the minimum current-frame weight: a\n\
         // 20% glass layer rejects at least 20% stale history, while fully\n\
         // refractive pixels consume the current result immediately.\n\
         let reactive = textureSampleLevel(\n\
             reactive_tex,\n\
             composed_samp,\n\
             clamp(src_uv, vec2<f32>(0.0), vec2<f32>(1.0)),\n\
             0.0,\n\
         ).r;",
        1,
    );
    assert!(
        source.contains("var reactive_tex"),
        "reactive TAA shader must declare its coverage input"
    );
    source
}

pub(super) fn scene_transparent_reactive_shader_source(base_scene_shader: &str) -> String {
    let mut source = String::with_capacity(base_scene_shader.len() + 700);
    source.push_str(base_scene_shader);
    source.push_str(
        r#"

struct TransparentReactiveOut {
    @location(0) color: vec4<f32>,
    @location(1) reactive: f32,
};

@fragment
fn fs_transparent_scene_reactive(
    in: VertexOutputScene,
    @builtin(front_facing) front_facing: bool,
) -> TransparentReactiveOut {
    let color = shade_main_scene(in, front_facing).color;
    return TransparentReactiveOut(color, clamp(color.a, 0.0, 1.0));
}
"#,
    );
    source
}

fn scene_refractive_reactive_shader_source(
    base_scene_shader: &str,
    folded_scene_inputs: bool,
    screen_space_reflections: bool,
    secondary_uv: bool,
) -> String {
    let source = scene_refractive_shader_source(
        base_scene_shader,
        folded_scene_inputs,
        screen_space_reflections,
        secondary_uv,
    )
    .replacen(
        "    @location(1) velocity: vec2<f32>,\n};",
        "    @location(1) velocity: vec2<f32>,\n\
         @location(2) reactive: f32,\n\
         };",
        1,
    );
    let source = source.replacen(
        "    return RefractiveSceneOut(vec4<f32>(hdr, 1.0), surface.velocity);",
        "    // Opaque transmission is reactive in proportion to the portion\n\
         // sourced through the current scene/environment. BLEND+transmission\n\
         // composites the complete material response by base alpha, so its\n\
         // full visible contribution is the correct temporal coverage.\n\
         let reactive = select(\n\
             transmission_weight,\n\
             clamp(base_alpha, 0.0, 1.0),\n\
             material.metal_rough.w < 0.0,\n\
         );\n\
         return RefractiveSceneOut(\n\
             vec4<f32>(hdr, 1.0), surface.velocity, reactive,\n\
         );",
        1,
    );
    assert!(
        source.contains("@location(2) reactive"),
        "refractive output changed; reactive injection must be updated"
    );
    source
}

const WEIGHTED_TRANSPARENCY_REACTIVE_RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var accumulation_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

@vertex
fn vs_weighted_transparency_resolve(
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

struct WeightedReactiveResolveOut {
    @location(0) color: vec4<f32>,
    @location(1) reactive: f32,
};

@fragment
fn fs_weighted_transparency_resolve(
    @builtin(position) position: vec4<f32>,
) -> WeightedReactiveResolveOut {
    let pixel = vec2<i32>(position.xy);
    let accumulation = textureLoad(accumulation_tex, pixel, 0);
    let revealage = clamp(textureLoad(revealage_tex, pixel, 0).r, 0.0, 1.0);
    let opacity = 1.0 - revealage;
    let color = accumulation.rgb / max(accumulation.a, 0.00001);
    let finite_color = select(vec3<f32>(0.0), color, color == color);
    return WeightedReactiveResolveOut(vec4<f32>(finite_color, opacity), opacity);
}
"#;

impl Renderer {
    pub(super) fn ensure_taa_reactive_resources(&mut self) {
        if self.taa_reactive_pipeline.is_some() {
            return;
        }
        let layout = create_taa_bind_group_layout(&self.device, "taa_reactive_layout", true);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("taa_reactive_shader"),
                source: wgpu::ShaderSource::Wgsl(taa_reactive_shader_source().into()),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("taa_reactive_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("taa_reactive_pipeline"),
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
                            format: HDR_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: TAA_DEPTH_HISTORY_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
                        }),
                    ],
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        self.taa_reactive_layout = Some(layout);
        self.taa_reactive_pipeline = Some(pipeline);
        self.created_pipelines(1);
    }

    pub(super) fn ensure_scene_transparent_reactive_resources(&mut self) {
        if self.scene_transparent_reactive_pipeline.is_some() {
            return;
        }
        let source = scene_transparent_reactive_shader_source(&specialized_scene_shader_source(
            self.froxel.is_some(),
            self.shadow_map.virtual_map.requested(),
        ));
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_transparent_reactive_shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene_transparent_reactive_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&self.uniform_3d_layout),
                    Some(&self.lighting_layout),
                    Some(&self.scene_material_layout),
                    Some(&self.joint_layout),
                ],
                immediate_size: 0,
            });
        let create_pipeline = |label: &'static str, cull_mode: Option<wgpu::Face>| {
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main_scene"),
                        buffers: &[Vertex3D::desc()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_transparent_scene_reactive"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: HDR_FORMAT,
                                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: TEMPORAL_REACTIVE_FORMAT,
                                blend: Some(reactive_union_blend()),
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        self.scene_transparent_reactive_pipeline = Some(create_pipeline(
            "scene_transparent_reactive_pipeline",
            Some(wgpu::Face::Back),
        ));
        self.scene_transparent_reactive_double_sided_pipeline = Some(create_pipeline(
            "scene_transparent_reactive_double_sided_pipeline",
            None,
        ));
        self.created_pipelines(2);
    }

    pub(super) fn ensure_scene_refraction_reactive_resources(&mut self) {
        if self.scene_refractive_reactive_pipeline.is_some() {
            return;
        }
        self.ensure_scene_refraction_resources();
        let material_layout = self
            .scene_refractive_material_layout
            .as_ref()
            .expect("ordinary imported-refraction resources initialize first");
        #[cfg(fold_scene_inputs)]
        let screen_space_reflections = false;
        #[cfg(not(fold_scene_inputs))]
        let screen_space_reflections = self.scene_refractive_inputs_layout.is_some();
        let source = scene_refractive_reactive_shader_source(
            &specialized_scene_shader_source(
                self.froxel.is_some(),
                self.shadow_map.virtual_map.requested(),
            ),
            cfg!(fold_scene_inputs),
            screen_space_reflections,
            false,
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_refractive_reactive_shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        #[cfg(fold_scene_inputs)]
        let bind_group_layouts = vec![
            Some(&self.uniform_3d_layout),
            Some(&self.lighting_layout),
            Some(material_layout),
            Some(&self.joint_layout),
        ];
        #[cfg(not(fold_scene_inputs))]
        let bind_group_layouts = vec![
            Some(&self.uniform_3d_layout),
            Some(&self.lighting_layout),
            Some(material_layout),
            Some(&self.joint_layout),
            Some(
                self.scene_refractive_inputs_layout
                    .as_ref()
                    .unwrap_or(&self.material_system.layouts.scene_inputs),
            ),
        ];
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene_refractive_reactive_pipeline_layout"),
                bind_group_layouts: &bind_group_layouts,
                immediate_size: 0,
            });
        let create_pipeline = |label: &'static str, cull_mode: Option<wgpu::Face>| {
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main_scene"),
                        buffers: &[Vertex3D::desc()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_refractive_scene"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: HDR_FORMAT,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: VELOCITY_FORMAT,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: TEMPORAL_REACTIVE_FORMAT,
                                blend: Some(reactive_union_blend()),
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        self.scene_refractive_reactive_pipeline = Some(create_pipeline(
            "scene_refractive_reactive_pipeline",
            Some(wgpu::Face::Back),
        ));
        self.scene_refractive_reactive_double_sided_pipeline = Some(create_pipeline(
            "scene_refractive_reactive_double_sided_pipeline",
            None,
        ));
        self.created_pipelines(2);
        if self.scene_refractive_uv1_pipeline.is_some() {
            self.ensure_scene_refraction_reactive_uv1_resources();
        }
    }

    pub(super) fn ensure_scene_refraction_reactive_uv1_resources(&mut self) {
        if self.scene_refractive_reactive_uv1_pipeline.is_some() {
            return;
        }
        self.ensure_scene_refraction_uv1_resources();
        if self.scene_refractive_reactive_pipeline.is_none() {
            self.ensure_scene_refraction_reactive_resources();
            return;
        }
        let material_layout = self
            .scene_refractive_material_layout
            .as_ref()
            .expect("ordinary imported-refraction resources initialize first");
        #[cfg(fold_scene_inputs)]
        let screen_space_reflections = false;
        #[cfg(not(fold_scene_inputs))]
        let screen_space_reflections = self.scene_refractive_inputs_layout.is_some();
        let source = scene_refractive_reactive_shader_source(
            &specialized_scene_shader_source(
                self.froxel.is_some(),
                self.shadow_map.virtual_map.requested(),
            ),
            cfg!(fold_scene_inputs),
            screen_space_reflections,
            true,
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_refractive_reactive_uv1_shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        #[cfg(fold_scene_inputs)]
        let bind_group_layouts = vec![
            Some(&self.uniform_3d_layout),
            Some(&self.lighting_layout),
            Some(material_layout),
            Some(&self.joint_layout),
        ];
        #[cfg(not(fold_scene_inputs))]
        let bind_group_layouts = vec![
            Some(&self.uniform_3d_layout),
            Some(&self.lighting_layout),
            Some(material_layout),
            Some(&self.joint_layout),
            Some(
                self.scene_refractive_inputs_layout
                    .as_ref()
                    .unwrap_or(&self.material_system.layouts.scene_inputs),
            ),
        ];
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene_refractive_reactive_uv1_pipeline_layout"),
                bind_group_layouts: &bind_group_layouts,
                immediate_size: 0,
            });
        let vertex_layouts = [Vertex3D::desc(), secondary_uv_desc()];
        let create_pipeline = |label: &'static str, cull_mode: Option<wgpu::Face>| {
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main_scene"),
                        buffers: &vertex_layouts,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_refractive_scene"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: HDR_FORMAT,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: VELOCITY_FORMAT,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: TEMPORAL_REACTIVE_FORMAT,
                                blend: Some(reactive_union_blend()),
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        self.scene_refractive_reactive_uv1_pipeline = Some(create_pipeline(
            "scene_refractive_reactive_uv1_pipeline",
            Some(wgpu::Face::Back),
        ));
        self.scene_refractive_reactive_uv1_double_sided_pipeline = Some(create_pipeline(
            "scene_refractive_reactive_uv1_double_sided_pipeline",
            None,
        ));
        self.created_pipelines(2);
    }

    pub(super) fn ensure_weighted_transparency_reactive_resources(&mut self) {
        if self
            .weighted_transparency_reactive_resolve_pipeline
            .is_some()
        {
            return;
        }
        self.ensure_weighted_transparency_resources();
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("weighted_transparency_reactive_resolve_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    WEIGHTED_TRANSPARENCY_REACTIVE_RESOLVE_SHADER.into(),
                ),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("weighted_transparency_reactive_resolve_pipeline_layout"),
                bind_group_layouts: &[Some(
                    self.weighted_transparency_resolve_layout
                        .as_ref()
                        .expect("ordinary weighted resolve layout initializes first"),
                )],
                immediate_size: 0,
            });
        self.weighted_transparency_reactive_resolve_pipeline = Some(
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("weighted_transparency_reactive_resolve_pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_weighted_transparency_resolve"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_weighted_transparency_resolve"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: HDR_FORMAT,
                                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: TEMPORAL_REACTIVE_FORMAT,
                                blend: Some(reactive_union_blend()),
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
        );
        self.created_pipelines(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_reactive_shader_variants_parse() {
        for source in [
            taa_reactive_shader_source(),
            scene_transparent_reactive_shader_source(SCENE_SHADER),
            scene_refractive_reactive_shader_source(SCENE_SHADER, false, false, false),
            scene_refractive_reactive_shader_source(SCENE_SHADER, false, true, false),
            scene_refractive_reactive_shader_source(SCENE_SHADER, true, false, false),
            scene_refractive_reactive_shader_source(SCENE_SHADER, false, false, true),
            scene_refractive_reactive_shader_source(SCENE_SHADER, false, true, true),
            scene_refractive_reactive_shader_source(SCENE_SHADER, true, false, true),
            WEIGHTED_TRANSPARENCY_REACTIVE_RESOLVE_SHADER.to_owned(),
        ] {
            wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("reactive WGSL failed: {error}"));
        }
    }

    #[test]
    fn established_shader_sources_do_not_gain_reactive_bindings_or_outputs() {
        assert!(!TAA_SHADER_WGSL.contains("reactive_tex"));
        assert!(!SCENE_SHADER.contains("TransparentReactiveOut"));
        assert!(!WEIGHTED_TRANSPARENCY_RESOLVE_SHADER.contains("reactive"));
    }

    #[test]
    fn selection_requires_taa_and_a_visible_reactive_contributor() {
        assert!(!temporal_reactive_selected_for(
            false,
            true,
            1,
            || panic!("disabled feature must not scan custom materials"),
            || { panic!("disabled feature must not scan refraction") }
        ));
        assert!(!temporal_reactive_selected_for(
            true,
            false,
            1,
            || panic!("TAA-off path must not scan custom materials"),
            || { panic!("TAA-off path must not scan refraction") }
        ));
        assert!(!temporal_reactive_selected_for(
            true,
            true,
            0,
            || false,
            || 0
        ));
        assert!(temporal_reactive_selected_for(
            true,
            true,
            1,
            || { panic!("visible BLEND must short-circuit the custom-material scan") },
            || { panic!("visible BLEND must short-circuit the refraction scan") }
        ));
        assert!(temporal_reactive_selected_for(
            true,
            true,
            0,
            || true,
            || { panic!("custom reactive draw must short-circuit the refraction scan") }
        ));
        assert!(temporal_reactive_selected_for(
            true,
            true,
            0,
            || false,
            || 1
        ));
    }
}
