use super::*;

/// User-selected conventional-transparency composition policy.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum TransparencyCompositionPreference {
    Sorted,
    Auto,
    Weighted,
}

impl TransparencyCompositionPreference {
    pub(super) fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Sorted,
            2 => Self::Weighted,
            _ => Self::Auto,
        }
    }

    pub(super) fn code(self) -> u32 {
        match self {
            Self::Sorted => 0,
            Self::Auto => 1,
            Self::Weighted => 2,
        }
    }

    pub(super) fn from_environment() -> Self {
        match std::env::var("BLOOM_TRANSPARENCY")
            .unwrap_or_else(|_| "auto".to_owned())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "sorted" | "off" | "0" => Self::Sorted,
            "weighted" | "oit" | "2" => Self::Weighted,
            _ => Self::Auto,
        }
    }
}

/// Auto keeps the exact sorted-alpha path for ordinary scenes and only pays
/// for OIT when a frame submits a genuinely high-count imported BLEND set.
pub(super) const WEIGHTED_TRANSPARENCY_AUTO_DRAW_THRESHOLD: usize = 64;

fn weighted_transparency_selected(
    preference: TransparencyCompositionPreference,
    visible_draw_count: usize,
) -> bool {
    match preference {
        TransparencyCompositionPreference::Sorted => false,
        TransparencyCompositionPreference::Auto => {
            visible_draw_count >= WEIGHTED_TRANSPARENCY_AUTO_DRAW_THRESHOLD
        }
        TransparencyCompositionPreference::Weighted => visible_draw_count > 0,
    }
}

impl Renderer {
    /// Configure conventional imported-transparency composition:
    /// 0 = sorted, 1 = automatic high-count OIT, 2 = force weighted OIT.
    pub fn set_transparency_composition_mode(&mut self, mode: u32) {
        self.transparency_composition_preference =
            TransparencyCompositionPreference::from_code(mode);
    }

    pub fn transparency_composition_mode_code(&self) -> u32 {
        self.transparency_composition_preference.code()
    }

    /// Route selected for the most recently prepared/current frame:
    /// 0 = deterministic sorted alpha, 1 = weighted-blended OIT.
    pub fn active_transparency_composition_mode_code(&self) -> u32 {
        u32::from(self.weighted_transparency_active)
    }

    fn imported_transparent_draw_count(&self, scene: &crate::scene::SceneGraph) -> usize {
        let cached = if self.has_blend_model_draws {
            let camera_vp = mat4_multiply(
                self.current_proj_matrix_unjittered,
                self.current_view_matrix,
            );
            let camera_planes = crate::scene::extract_frustum_planes(&camera_vp);
            self.model_draw_commands
                .iter()
                .filter_map(|command| {
                    let mesh = self
                        .model_gpu_cache
                        .get(&command.cache_handle)
                        .and_then(|meshes| meshes.as_ref())
                        .and_then(|meshes| meshes.get(command.mesh_idx))?;
                    if mesh.alpha_mode != MaterialAlphaMode::Blend
                        || (self.imported_refraction_enabled && mesh.transmission.is_active())
                    {
                        return None;
                    }
                    let (world_min, world_max) = command.bounds_override.unwrap_or_else(|| {
                        transform_aabb(&command.model, mesh.local_min, mesh.local_max)
                    });
                    if world_min[0] <= world_max[0]
                        && crate::scene::aabb_outside_frustum(&camera_planes, world_min, world_max)
                    {
                        return None;
                    }
                    Some(())
                })
                .count()
        } else {
            0
        };
        cached + scene.visible_transparent_node_count(self.imported_refraction_enabled)
    }

    fn imported_refractive_draw_count(&self, scene: &crate::scene::SceneGraph) -> usize {
        if !self.imported_refraction_enabled {
            return 0;
        }
        let cached = if self.has_refractive_model_draws {
            let camera_vp = mat4_multiply(
                self.current_proj_matrix_unjittered,
                self.current_view_matrix,
            );
            let camera_planes = crate::scene::extract_frustum_planes(&camera_vp);
            self.model_draw_commands
                .iter()
                .filter_map(|command| {
                    let mesh = self
                        .model_gpu_cache
                        .get(&command.cache_handle)
                        .and_then(|meshes| meshes.as_ref())
                        .and_then(|meshes| meshes.get(command.mesh_idx))?;
                    if !mesh.transmission.is_active() || mesh.refractive_material_bg.is_none() {
                        return None;
                    }
                    let (world_min, world_max) = command.bounds_override.unwrap_or_else(|| {
                        transform_aabb(&command.model, mesh.local_min, mesh.local_max)
                    });
                    if world_min[0] <= world_max[0]
                        && crate::scene::aabb_outside_frustum(&camera_planes, world_min, world_max)
                    {
                        return None;
                    }
                    Some(())
                })
                .count()
        } else {
            0
        };
        cached + scene.visible_refractive_node_count()
    }

    /// Select both imported-transparency routes from one visible BLEND scan.
    /// Reactive coverage additionally considers visible physical transmission,
    /// but only while TAA can consume the result.
    pub(super) fn select_transparency_routes(
        &self,
        scene: &crate::scene::SceneGraph,
    ) -> (bool, bool) {
        let transparent_draw_count = if self.has_blend_model_draws || scene.has_transparent_nodes()
        {
            self.imported_transparent_draw_count(scene)
        } else {
            0
        };
        let weighted = weighted_transparency_selected(
            self.transparency_composition_preference,
            transparent_draw_count,
        );
        let reactive = temporal_reactive::temporal_reactive_selected(
            self.taa_enabled,
            transparent_draw_count,
            || self.imported_refractive_draw_count(scene),
        );
        (weighted, reactive)
    }

    /// Compile weighted accumulation + resolve pipelines on first activation.
    /// Sorted-only and opaque applications retain the established startup and
    /// memory footprint.
    pub(super) fn ensure_weighted_transparency_resources(&mut self) {
        if self.scene_weighted_transparent_pipeline.is_some() {
            return;
        }

        let source = scene_weighted_transparency_shader_source(&specialized_scene_shader_source(
            self.froxel.is_some(),
            self.shadow_map.virtual_map.requested(),
        ));
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_weighted_transparency_shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let scene_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene_weighted_transparency_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&self.uniform_3d_layout),
                    Some(&self.lighting_layout),
                    Some(&self.scene_material_layout),
                    Some(&self.joint_layout),
                ],
                immediate_size: 0,
            });
        let accumulation_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let revealage_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let create_accumulation_pipeline = |label: &'static str, cull_mode: Option<wgpu::Face>| {
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&scene_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main_scene"),
                        buffers: &[Vertex3D::desc()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_weighted_transparent_scene"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: Some(accumulation_blend),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::R16Float,
                                blend: Some(revealage_blend),
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
        let single_sided = create_accumulation_pipeline(
            "scene_weighted_transparency_pipeline",
            Some(wgpu::Face::Back),
        );
        let double_sided =
            create_accumulation_pipeline("scene_weighted_transparency_double_sided_pipeline", None);

        let resolve_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("weighted_transparency_resolve_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });
        let resolve_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("weighted_transparency_resolve_shader"),
                source: wgpu::ShaderSource::Wgsl(WEIGHTED_TRANSPARENCY_RESOLVE_SHADER.into()),
            });
        let resolve_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("weighted_transparency_resolve_pipeline_layout"),
                    bind_group_layouts: &[Some(&resolve_layout)],
                    immediate_size: 0,
                });
        let resolve_pipeline =
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("weighted_transparency_resolve_pipeline"),
                    layout: Some(&resolve_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &resolve_shader,
                        entry_point: Some("vs_weighted_transparency_resolve"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &resolve_shader,
                        entry_point: Some("fs_weighted_transparency_resolve"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: HDR_FORMAT,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        self.scene_weighted_transparent_pipeline = Some(single_sided);
        self.scene_weighted_transparent_double_sided_pipeline = Some(double_sided);
        self.weighted_transparency_resolve_pipeline = Some(resolve_pipeline);
        self.weighted_transparency_resolve_layout = Some(resolve_layout);
        log::info!(
            "bloom materials: weighted transparency initialized \
             (auto_threshold={}, accumulation=rgba16float, revealage=r16float)",
            WEIGHTED_TRANSPARENCY_AUTO_DRAW_THRESHOLD
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_threshold_preserves_simple_sorted_sets() {
        assert!(!weighted_transparency_selected(
            TransparencyCompositionPreference::Auto,
            WEIGHTED_TRANSPARENCY_AUTO_DRAW_THRESHOLD - 1,
        ));
        assert!(weighted_transparency_selected(
            TransparencyCompositionPreference::Auto,
            WEIGHTED_TRANSPARENCY_AUTO_DRAW_THRESHOLD,
        ));
        assert!(!weighted_transparency_selected(
            TransparencyCompositionPreference::Sorted,
            usize::MAX,
        ));
        assert!(!weighted_transparency_selected(
            TransparencyCompositionPreference::Weighted,
            0,
        ));
        assert!(weighted_transparency_selected(
            TransparencyCompositionPreference::Weighted,
            1,
        ));
    }
}
