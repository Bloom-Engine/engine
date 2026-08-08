//! Lazy combined specialization for materials that author both physical
//! transmission and layered-PBR clearcoat/specular/IOR/sheen/anisotropy.

use super::*;

const FIRST_LAYERED_BINDING: u32 = 16;

// Combined transmission + layered PBR owns eighteen sampled material
// textures. The complete fragment-stage layout also carries six lighting
// textures (eight with VSM) and three native reflection inputs. When the
// reflection hierarchy is disabled, or SceneInputs are folded into group 0,
// that last contribution is four instead. Keep the maximum in the device
// request, while the runtime gate below uses the exact active layout.
pub(super) const SCENE_LAYERED_REFRACTIVE_MAX_SAMPLED_TEXTURES: u32 = 30;

const fn scene_layered_refractive_sampled_texture_requirement(
    virtual_shadows: bool,
    folded_scene_inputs: bool,
    screen_space_reflections: bool,
) -> u32 {
    let lighting = if virtual_shadows { 8 } else { 6 };
    let scene_inputs = if !folded_scene_inputs && screen_space_reflections {
        3
    } else {
        4
    };
    18 + lighting + scene_inputs
}

pub(super) fn scene_layered_refractive_shader_source(
    base_scene_shader: &str,
    folded_scene_inputs: bool,
    screen_space_reflections: bool,
    secondary_uv: bool,
    reactive: bool,
) -> String {
    let layered = layered_pbr_scene::scene_layered_shader_source_with_bindings(
        base_scene_shader,
        secondary_uv,
        FIRST_LAYERED_BINDING,
    );
    let source = scene_refractive_shader_source(
        &layered,
        folded_scene_inputs,
        screen_space_reflections,
        secondary_uv,
    );
    let source = source.replacen(
        r#"    let f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = f0_scalar
        + (1.0 - f0_scalar) * pow(1.0 - n_dot_v, 5.0);"#,
        r#"    let layered_surface = evaluate_layered_surface(
        in,
        n,
        1.0 + lighting.shadow_cascade_splits.w,
    );
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = layered_dielectric_fresnel(layered_surface, n_dot_v);"#,
        1,
    );
    assert!(
        source.contains("let layered_surface = evaluate_layered_surface("),
        "refractive Fresnel block changed; layered specialization must be updated"
    );
    let source = source.replacen(
        "    let reflected_direction = reflect(-v, n);",
        "    let reflected_direction = layered_ibl_reflection(\n\
         layered_surface, n, v, roughness,\n\
         );",
        1,
    );
    assert!(
        source.contains("let reflected_direction = layered_ibl_reflection("),
        "refractive reflection direction changed; layered specialization must be updated"
    );
    let mut source = source.replacen(
        r#"    let dielectric_transmission = mix(transmitted, reflected, fresnel);
    var hdr = surface.color.rgb * (1.0 - transmission_weight)
        + dielectric_transmission * transmission_weight;"#,
        r#"    let dielectric_transmission = mix(transmitted, reflected, fresnel);
    let sheen_n_dot_v = max(dot(n, v), 0.0);
    let sheen_attenuation = layered_sheen_ibl_scale(
        layered_surface,
        sheen_n_dot_v,
    );
    let sheen_reflection = layered_sheen_ibl(
        layered_surface,
        n,
        v,
        max(f32(textureNumLevels(env_tex)) - 1.0, 0.0),
        1.0,
    );
    let dielectric_below_coat =
        dielectric_transmission * sheen_attenuation + sheen_reflection;
    let coat_n_dot_v = max(dot(layered_surface.clearcoat_normal, v), 0.0);
    let coat_fresnel = layered_clearcoat_fresnel(layered_surface, coat_n_dot_v);
    let coat_reflection_direction = reflect(-v, layered_surface.clearcoat_normal);
    let coat_reflection_raw = env_sample_lod(
        coat_reflection_direction,
        layered_surface.clearcoat_roughness
            * max(f32(textureNumLevels(env_tex)) - 1.0, 0.0),
    ) * coat_fresnel;
    let coat_luma = dot(
        coat_reflection_raw,
        vec3<f32>(0.2126, 0.7152, 0.0722),
    );
    let coat_cap = 1.0 / (1.0 + coat_luma / 0.3);
    let layered_dielectric_transmission =
        dielectric_below_coat
            * layered_clearcoat_ibl_attenuation(layered_surface, v)
        + coat_reflection_raw * coat_cap;
    var hdr = surface.color.rgb * (1.0 - transmission_weight)
        + layered_dielectric_transmission * transmission_weight;"#,
        1,
    );
    assert!(
        source.contains("let layered_dielectric_transmission ="),
        "refractive energy partition changed; layered specialization must be updated"
    );

    if reactive {
        source = source
            .replacen(
                "    @location(1) velocity: vec2<f32>,\n};",
                "    @location(1) velocity: vec2<f32>,\n\
                 @location(2) reactive: f32,\n\
                 };",
                1,
            )
            .replacen(
                "    return RefractiveSceneOut(vec4<f32>(hdr, 1.0), surface.velocity);",
                "    let reactive = select(\n\
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
            "layered refractive reactive output changed; specialization must be updated"
        );
    }
    source
}

pub(super) struct SceneLayeredRefractiveResources {
    material_layout: wgpu::BindGroupLayout,
    scalar: wgpu::RenderPipeline,
    scalar_double_sided: wgpu::RenderPipeline,
    uv1: Option<wgpu::RenderPipeline>,
    uv1_double_sided: Option<wgpu::RenderPipeline>,
    reactive: Option<wgpu::RenderPipeline>,
    reactive_double_sided: Option<wgpu::RenderPipeline>,
    reactive_uv1: Option<wgpu::RenderPipeline>,
    reactive_uv1_double_sided: Option<wgpu::RenderPipeline>,
}

impl SceneLayeredRefractiveResources {
    pub(super) fn pipeline(
        &self,
        secondary_uv: bool,
        double_sided: bool,
        reactive: bool,
    ) -> &wgpu::RenderPipeline {
        match (reactive, secondary_uv, double_sided) {
            (false, false, false) => &self.scalar,
            (false, false, true) => &self.scalar_double_sided,
            (false, true, false) => self
                .uv1
                .as_ref()
                .expect("layered refractive UV1 resources are initialized"),
            (false, true, true) => self
                .uv1_double_sided
                .as_ref()
                .expect("layered refractive UV1 resources are initialized"),
            (true, false, false) => self
                .reactive
                .as_ref()
                .expect("layered refractive reactive resources are initialized"),
            (true, false, true) => self
                .reactive_double_sided
                .as_ref()
                .expect("layered refractive reactive resources are initialized"),
            (true, true, false) => self
                .reactive_uv1
                .as_ref()
                .expect("layered refractive reactive UV1 resources are initialized"),
            (true, true, true) => self
                .reactive_uv1_double_sided
                .as_ref()
                .expect("layered refractive reactive UV1 resources are initialized"),
        }
    }
}

fn create_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let uniform = |binding, visibility| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_layered_refractive_material_layout"),
        entries: &[
            texture(0),
            sampler(1),
            texture(2),
            sampler(3),
            texture(4),
            sampler(5),
            texture(6),
            sampler(7),
            uniform(8, wgpu::ShaderStages::VERTEX_FRAGMENT),
            texture(9),
            sampler(10),
            texture(11),
            sampler(12),
            texture(13),
            sampler(14),
            uniform(15, wgpu::ShaderStages::FRAGMENT),
            texture(16),
            texture(17),
            texture(18),
            texture(19),
            texture(20),
            texture(21),
            texture(22),
            texture(23),
            texture(24),
            texture(25),
            texture(26),
            sampler(27),
            uniform(28, wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

fn pipeline_layout(
    renderer: &Renderer,
    material_layout: &wgpu::BindGroupLayout,
    label: &'static str,
) -> wgpu::PipelineLayout {
    #[cfg(fold_scene_inputs)]
    let bind_group_layouts = vec![
        Some(&renderer.uniform_3d_layout),
        Some(&renderer.lighting_layout),
        Some(material_layout),
        Some(&renderer.joint_layout),
    ];
    #[cfg(not(fold_scene_inputs))]
    let bind_group_layouts = vec![
        Some(&renderer.uniform_3d_layout),
        Some(&renderer.lighting_layout),
        Some(material_layout),
        Some(&renderer.joint_layout),
        Some(
            renderer
                .scene_refractive_inputs_layout
                .as_ref()
                .unwrap_or(&renderer.material_system.layouts.scene_inputs),
        ),
    ];
    renderer
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &bind_group_layouts,
            immediate_size: 0,
        })
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    secondary_uv: bool,
    double_sided: bool,
    reactive: bool,
    label: &'static str,
) -> wgpu::RenderPipeline {
    let buffers = if secondary_uv {
        vec![Vertex3D::desc(), secondary_uv_desc()]
    } else {
        vec![Vertex3D::desc()]
    };
    let mut targets = vec![
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
    ];
    if reactive {
        targets.push(Some(wgpu::ColorTargetState {
            format: temporal_reactive::TEMPORAL_REACTIVE_FORMAT,
            blend: Some(temporal_reactive::reactive_union_blend()),
            write_mask: wgpu::ColorWrites::RED,
        }));
    }
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main_scene"),
            buffers: &buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_refractive_scene"),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
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
}

impl Renderer {
    fn layered_refractive_source(&self, secondary_uv: bool, reactive: bool) -> String {
        #[cfg(fold_scene_inputs)]
        let screen_space_reflections = false;
        #[cfg(not(fold_scene_inputs))]
        let screen_space_reflections = self.scene_refractive_inputs_layout.is_some();
        scene_layered_refractive_shader_source(
            &specialized_scene_shader_source(
                self.froxel.is_some(),
                self.shadow_map.virtual_map.requested(),
            ),
            cfg!(fold_scene_inputs),
            screen_space_reflections,
            secondary_uv,
            reactive,
        )
    }

    fn scene_layered_refractive_sampled_texture_requirement(&self) -> u32 {
        #[cfg(fold_scene_inputs)]
        let screen_space_reflections = false;
        #[cfg(not(fold_scene_inputs))]
        let screen_space_reflections = self.scene_refractive_inputs_layout.is_some();
        scene_layered_refractive_sampled_texture_requirement(
            self.shadow_map.virtual_map.requested(),
            cfg!(fold_scene_inputs),
            screen_space_reflections,
        )
    }

    fn ensure_scene_layered_refraction_resources(&mut self) -> bool {
        if self.scene_layered_refractive_resources.is_some() {
            return true;
        }
        self.ensure_scene_refraction_resources();
        let required = self.scene_layered_refractive_sampled_texture_requirement();
        let granted = self.device.limits().max_sampled_textures_per_shader_stage;
        if granted < required {
            static WARN_UNAVAILABLE: std::sync::Once = std::sync::Once::new();
            WARN_UNAVAILABLE.call_once(|| {
                log::warn!(
                    "bloom materials: combined layered-PBR refraction requires {required} \
                     sampled textures per fragment stage, but the negotiated device grants \
                     {granted}; retaining physical refraction without layered lobes"
                );
            });
            return false;
        }
        let material_layout = create_material_layout(&self.device);
        let layout = pipeline_layout(
            self,
            &material_layout,
            "scene_layered_refractive_pipeline_layout",
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_refractive_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    self.layered_refractive_source(false, false).into(),
                ),
            });
        let scalar = create_pipeline(
            &self.device,
            &layout,
            &shader,
            false,
            false,
            false,
            "scene_layered_refractive_pipeline",
        );
        let scalar_double_sided = create_pipeline(
            &self.device,
            &layout,
            &shader,
            false,
            true,
            false,
            "scene_layered_refractive_double_sided_pipeline",
        );
        self.scene_layered_refractive_resources = Some(SceneLayeredRefractiveResources {
            material_layout,
            scalar,
            scalar_double_sided,
            uv1: None,
            uv1_double_sided: None,
            reactive: None,
            reactive_double_sided: None,
            reactive_uv1: None,
            reactive_uv1_double_sided: None,
        });
        self.created_pipelines(2);
        true
    }

    fn ensure_scene_layered_refraction_uv1_resources(&mut self) {
        if !self.ensure_scene_layered_refraction_resources() {
            return;
        }
        if self
            .scene_layered_refractive_resources
            .as_ref()
            .is_some_and(|resources| resources.uv1.is_some())
        {
            return;
        }
        let resources = self
            .scene_layered_refractive_resources
            .as_ref()
            .expect("layered refractive resources initialize first");
        let layout = pipeline_layout(
            self,
            &resources.material_layout,
            "scene_layered_refractive_uv1_pipeline_layout",
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_refractive_uv1_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    self.layered_refractive_source(true, false).into(),
                ),
            });
        let uv1 = create_pipeline(
            &self.device,
            &layout,
            &shader,
            true,
            false,
            false,
            "scene_layered_refractive_uv1_pipeline",
        );
        let uv1_double_sided = create_pipeline(
            &self.device,
            &layout,
            &shader,
            true,
            true,
            false,
            "scene_layered_refractive_uv1_double_sided_pipeline",
        );
        let resources = self
            .scene_layered_refractive_resources
            .as_mut()
            .expect("layered refractive resources initialize first");
        resources.uv1 = Some(uv1);
        resources.uv1_double_sided = Some(uv1_double_sided);
        self.created_pipelines(2);
    }

    pub(super) fn ensure_scene_layered_refraction_reactive_resources(&mut self) {
        let Some(resources) = self.scene_layered_refractive_resources.as_ref() else {
            return;
        };
        if resources.reactive.is_some() {
            return;
        }
        let layout = pipeline_layout(
            self,
            &resources.material_layout,
            "scene_layered_refractive_reactive_pipeline_layout",
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_refractive_reactive_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    self.layered_refractive_source(false, true).into(),
                ),
            });
        let reactive = create_pipeline(
            &self.device,
            &layout,
            &shader,
            false,
            false,
            true,
            "scene_layered_refractive_reactive_pipeline",
        );
        let reactive_double_sided = create_pipeline(
            &self.device,
            &layout,
            &shader,
            false,
            true,
            true,
            "scene_layered_refractive_reactive_double_sided_pipeline",
        );
        let resources = self
            .scene_layered_refractive_resources
            .as_mut()
            .expect("layered refractive resources initialize first");
        resources.reactive = Some(reactive);
        resources.reactive_double_sided = Some(reactive_double_sided);
        let has_uv1 = resources.uv1.is_some();
        self.created_pipelines(2);
        if has_uv1 {
            self.ensure_scene_layered_refraction_reactive_uv1_resources();
        }
    }

    fn ensure_scene_layered_refraction_reactive_uv1_resources(&mut self) {
        self.ensure_scene_layered_refraction_uv1_resources();
        if self
            .scene_layered_refractive_resources
            .as_ref()
            .is_some_and(|resources| resources.reactive_uv1.is_some())
        {
            return;
        }
        let resources = self
            .scene_layered_refractive_resources
            .as_ref()
            .expect("layered refractive resources initialize first");
        let layout = pipeline_layout(
            self,
            &resources.material_layout,
            "scene_layered_refractive_reactive_uv1_pipeline_layout",
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_refractive_reactive_uv1_shader"),
                source: wgpu::ShaderSource::Wgsl(self.layered_refractive_source(true, true).into()),
            });
        let reactive_uv1 = create_pipeline(
            &self.device,
            &layout,
            &shader,
            true,
            false,
            true,
            "scene_layered_refractive_reactive_uv1_pipeline",
        );
        let reactive_uv1_double_sided = create_pipeline(
            &self.device,
            &layout,
            &shader,
            true,
            true,
            true,
            "scene_layered_refractive_reactive_uv1_double_sided_pipeline",
        );
        let resources = self
            .scene_layered_refractive_resources
            .as_mut()
            .expect("layered refractive resources initialize first");
        resources.reactive_uv1 = Some(reactive_uv1);
        resources.reactive_uv1_double_sided = Some(reactive_uv1_double_sided);
        self.created_pipelines(2);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_scene_layered_refractive_material_bg(
        &mut self,
        base_color_tex_idx: u32,
        normal_tex_idx: u32,
        metallic_roughness_tex_idx: u32,
        emissive_tex_idx: u32,
        occlusion_tex_idx: u32,
        material_uniform: &wgpu::Buffer,
        transmission: crate::models::MaterialTransmission,
        layered: crate::models::MaterialLayeredPbr,
        has_secondary_tex_coords: bool,
    ) -> Option<(wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, bool)> {
        if !self.imported_refraction_enabled || !transmission.is_active() || !layered.is_active() {
            return None;
        }
        if !self.ensure_scene_layered_refraction_resources() {
            return None;
        }

        let usable_texture = |binding: Option<crate::models::MaterialTextureBinding>,
                              contributes: bool|
         -> Option<(u32, u32)> {
            if !contributes {
                return None;
            }
            binding.and_then(|binding| {
                match binding.transform.tex_coord {
                    0 => {}
                    1 if has_secondary_tex_coords => {}
                    1 => {
                        log::warn!(
                            "bloom materials: layered refractive texture requests TEXCOORD_1 \
                             but this primitive has no valid secondary UV stream; using its \
                             scalar factor"
                        );
                        return None;
                    }
                    tex_coord => {
                        log::warn!(
                            "bloom materials: layered refractive texture TEXCOORD_{tex_coord} \
                             is preserved but only TEXCOORD_0/1 are renderable; using its \
                             scalar factor"
                        );
                        return None;
                    }
                }
                binding
                    .runtime_texture_idx
                    .filter(|index| *index != 0 && (*index as usize) < self.textures.len())
                    .map(|index| (index, binding.transform.tex_coord))
            })
        };
        let transmission_texture = usable_texture(transmission.texture, true);
        let thickness_texture = usable_texture(transmission.thickness_texture, true);
        let layered_bindings = [
            usable_texture(layered.clearcoat_texture, layered.has_clearcoat()),
            usable_texture(layered.clearcoat_roughness_texture, layered.has_clearcoat()),
            usable_texture(layered.clearcoat_normal_texture, layered.has_clearcoat()),
            usable_texture(layered.specular_texture, layered.has_specular_ior()),
            usable_texture(layered.specular_color_texture, layered.has_specular_ior()),
            usable_texture(layered.sheen_color_texture, layered.has_sheen()),
            usable_texture(layered.sheen_roughness_texture, layered.has_sheen()),
            usable_texture(layered.anisotropy_texture, layered.has_anisotropy()),
            usable_texture(layered.iridescence_texture, layered.has_iridescence()),
            usable_texture(
                layered.iridescence_thickness_texture,
                layered.has_iridescence(),
            ),
        ];
        let uses_uv1 = transmission_texture
            .into_iter()
            .chain(thickness_texture)
            .chain(layered_bindings.into_iter().flatten())
            .any(|(_, tex_coord)| tex_coord == 1);
        if uses_uv1 {
            self.ensure_scene_layered_refraction_uv1_resources();
            if self
                .scene_layered_refractive_resources
                .as_ref()
                .is_some_and(|resources| resources.reactive.is_some())
            {
                self.ensure_scene_layered_refraction_reactive_uv1_resources();
            }
        }
        if self.shadow_map.enabled {
            self.ensure_transmitted_shadow_resources();
            if uses_uv1 {
                self.ensure_transmitted_shadow_uv1_resources();
            }
        }
        if layered.has_sheen() {
            self.ensure_scene_sheen_albedo_lut();
        }

        let layered_texture_usable = layered_bindings.map(|binding| binding.is_some());
        let transmission_factors = SceneTransmissionUniforms::new(
            transmission,
            transmission_texture.is_some(),
            thickness_texture.is_some(),
        );
        let layered_factors =
            layered_pbr_scene::SceneLayeredPbrUniforms::new(layered, layered_texture_usable);
        let transmission_uniform =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_layered_refractive_transmission_uniform"),
                    contents: bytemuck::bytes_of(&transmission_factors),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        let layered_uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene_layered_refractive_layer_uniform"),
                contents: bytemuck::bytes_of(&layered_factors),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let view_or_white = |index: u32| {
            self.textures
                .get(index as usize)
                .unwrap_or(&self.textures[0])
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let base_view = view_or_white(base_color_tex_idx);
        let mr_view = view_or_white(metallic_roughness_tex_idx);
        let emissive_view = view_or_white(emissive_tex_idx);
        let occlusion_view = view_or_white(occlusion_tex_idx);
        let transmission_view =
            view_or_white(transmission_texture.map(|binding| binding.0).unwrap_or(0));
        let thickness_view = view_or_white(thickness_texture.map(|binding| binding.0).unwrap_or(0));
        let clearcoat_factor_view =
            view_or_white(layered_bindings[0].map(|binding| binding.0).unwrap_or(0));
        let clearcoat_roughness_view =
            view_or_white(layered_bindings[1].map(|binding| binding.0).unwrap_or(0));
        let specular_factor_view =
            view_or_white(layered_bindings[3].map(|binding| binding.0).unwrap_or(0));
        let specular_color_view =
            view_or_white(layered_bindings[4].map(|binding| binding.0).unwrap_or(0));
        let sheen_color_view =
            view_or_white(layered_bindings[5].map(|binding| binding.0).unwrap_or(0));
        let sheen_roughness_view =
            view_or_white(layered_bindings[6].map(|binding| binding.0).unwrap_or(0));
        let anisotropy_view =
            view_or_white(layered_bindings[7].map(|binding| binding.0).unwrap_or(0));
        let iridescence_factor_view =
            view_or_white(layered_bindings[8].map(|binding| binding.0).unwrap_or(0));
        let iridescence_thickness_view =
            view_or_white(layered_bindings[9].map(|binding| binding.0).unwrap_or(0));
        let sheen_lut_fallback = view_or_white(0);
        let sheen_albedo_view = self
            .scene_sheen_albedo_lut
            .as_ref()
            .map(|lut| &lut.view)
            .unwrap_or(&sheen_lut_fallback);
        let base_normal_view_owned = self
            .textures
            .get(normal_tex_idx as usize)
            .filter(|_| normal_tex_idx != 0)
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let base_normal_view = base_normal_view_owned
            .as_ref()
            .unwrap_or(&self.default_normal_view);
        let coat_normal_index = layered_bindings[2].map(|binding| binding.0).unwrap_or(0);
        let coat_normal_view_owned = self
            .textures
            .get(coat_normal_index as usize)
            .filter(|_| coat_normal_index != 0)
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let coat_normal_view = coat_normal_view_owned
            .as_ref()
            .unwrap_or(&self.default_normal_view);

        let layout = &self
            .scene_layered_refractive_resources
            .as_ref()
            .expect("layered refractive resources initialize first")
            .material_layout;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_layered_refractive_material_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(base_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&emissive_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: material_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&occlusion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&transmission_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&thickness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: transmission_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&clearcoat_factor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(&clearcoat_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(coat_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::TextureView(&specular_factor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: wgpu::BindingResource::TextureView(&specular_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: wgpu::BindingResource::TextureView(&sheen_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: wgpu::BindingResource::TextureView(&sheen_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: wgpu::BindingResource::TextureView(&anisotropy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: wgpu::BindingResource::TextureView(&iridescence_factor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: wgpu::BindingResource::TextureView(&iridescence_thickness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: wgpu::BindingResource::TextureView(sheen_albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 27,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 28,
                    resource: layered_uniform.as_entire_binding(),
                },
            ],
        });
        Some((transmission_uniform, layered_uniform, bind_group, uses_uv1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_texture_contract_covers_reflection_and_vsm_layouts() {
        assert_eq!(
            scene_layered_refractive_sampled_texture_requirement(false, false, true),
            27
        );
        assert_eq!(
            scene_layered_refractive_sampled_texture_requirement(true, false, true),
            29
        );
        assert_eq!(
            scene_layered_refractive_sampled_texture_requirement(false, false, false),
            28
        );
        assert_eq!(
            scene_layered_refractive_sampled_texture_requirement(true, true, false),
            SCENE_LAYERED_REFRACTIVE_MAX_SAMPLED_TEXTURES
        );
    }

    #[test]
    fn combined_refractive_variants_parse_without_touching_base_shader() {
        for secondary_uv in [false, true] {
            for reactive in [false, true] {
                for (folded, reflections) in [(false, false), (false, true), (true, false)] {
                    let source = scene_layered_refractive_shader_source(
                        SCENE_SHADER,
                        folded,
                        reflections,
                        secondary_uv,
                        reactive,
                    );
                    wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                        panic!(
                            "combined layered refraction (secondary_uv={secondary_uv}, \
                             reactive={reactive}, folded={folded}, \
                             reflections={reflections}) failed: {error}"
                        )
                    });
                    assert!(source.contains("@group(2) @binding(15)"));
                    assert!(source.contains("@group(2) @binding(28)"));
                    assert!(source.contains("let dielectric_below_coat ="));
                    assert!(source.contains("let reflected_direction = layered_ibl_reflection("));
                }
            }
        }
        assert!(!SCENE_SHADER.contains("layered_material"));
        assert!(!SCENE_SHADER.contains("transmission_material"));
    }
}
