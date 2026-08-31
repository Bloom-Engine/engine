//! Lazy iridescence-to-SSR transport.
//!
//! The compact legacy G-buffer has room only for metallic/roughness and base
//! albedo. Re-encoding those channels would either corrupt ordinary materials
//! or quantize the thin-film response. Instead, frames that actually contain
//! visible opaque iridescence replay only those meshes into a linear Fresnel
//! target. A matching lazy SSR variant consumes it. Base-only frames retain the
//! original targets, shader, bind group, and pass count.

use super::*;

const IRIDESCENCE_SSR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(crate) struct SceneIridescenceSsrResources {
    _metadata_texture: wgpu::Texture,
    pub(crate) metadata_view: wgpu::TextureView,
    scalar_metadata_pipeline: wgpu::RenderPipeline,
    uv1_metadata_pipeline: wgpu::RenderPipeline,
    pub(crate) ssr_layout: wgpu::BindGroupLayout,
    pub(crate) ssr_pipeline: wgpu::RenderPipeline,
}

impl SceneIridescenceSsrResources {
    fn metadata_pipeline(&self, secondary_uv: bool) -> &wgpu::RenderPipeline {
        if secondary_uv {
            &self.uv1_metadata_pipeline
        } else {
            &self.scalar_metadata_pipeline
        }
    }
}

fn iridescence_metadata_shader_source(base_scene_shader: &str, secondary_uv: bool) -> String {
    let source = specialized_scene_shader_source_from(
        layered_pbr_scene::scene_layered_shader_source(base_scene_shader, secondary_uv).into(),
        false,
        false,
    )
    .into_owned();
    let tail_begin = source
        .find("    let em_tex_sample = textureSample(em_tex, em_samp, in.uv);")
        .expect("layered scene shader keeps the post-material tail anchor");
    let tail_end = source[tail_begin..]
        .find("\n}\n\n@fragment\nfn fs_main_scene(")
        .map(|offset| tail_begin + offset)
        .expect("layered scene shader keeps the scene fragment entry point");
    let replacement = r#"    // The metadata replay deliberately stops after material + normal
    // evaluation. Direct lights, IBL, shadows, velocity, and albedo outputs are
    // dead here, keeping the opt-in pass bounded to the data SSR cannot infer.
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);
    let layered_surface = evaluate_layered_surface(in, n, lod_bias);
    let raw_fresnel = layered_raw_iridescence_base_fresnel(
        layered_surface,
        max(dot(n, v), 0.0),
        base_color,
        metallic,
    );
    let finite_fresnel = select(
        vec3<f32>(0.0),
        clamp(raw_fresnel, vec3<f32>(0.0), vec3<f32>(1.0)),
        raw_fresnel == raw_fresnel,
    );
    return SceneOut(
        vec4<f32>(
            finite_fresnel,
            clamp(layered_surface.iridescence_factor, 0.0, 1.0),
        ),
        vec2<f32>(0.0),
        vec2<f32>(0.0),
        vec4<f32>(0.0),
    );"#;
    format!(
        "{}{}{}",
        &source[..tail_begin],
        replacement,
        &source[tail_end..]
    )
}

fn iridescence_ssr_shader_source(base_ssr_shader: &str) -> String {
    let declaration_anchor = "@group(0) @binding(10) var env_samp: sampler;";
    let declaration = format!(
        "{declaration_anchor}\n\
         @group(0) @binding(11) var iridescence_fresnel_tex: texture_2d<f32>;"
    );
    let source = base_ssr_shader.replacen(declaration_anchor, &declaration, 1);
    assert_ne!(
        source, base_ssr_shader,
        "SSR bindings changed; iridescence transport must be updated"
    );
    let fresnel_anchor = r#"    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);"#;
    let fresnel_replacement = r#"    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let ordinary_fresnel =
        f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);
    // RGB is the unblended thin-film response and A is the sampled glTF
    // iridescence factor. Clear pixels have A=0, so non-iridescent surfaces
    // remain bit-for-bit on the established Schlick path.
    let iridescence_sample = textureSampleLevel(
        iridescence_fresnel_tex,
        mat_samp,
        in.uv,
        0.0,
    );
    let iridescence_reflection_roughness = max(
        roughness,
        clamp(iridescence_sample.a, 0.0, 1.0) / max(u.params2.x, 1.0),
    );
    let fresnel = mix(
        ordinary_fresnel,
        iridescence_sample.rgb,
        clamp(iridescence_sample.a, 0.0, 1.0),
    );"#;
    let replaced = source.replacen(fresnel_anchor, fresnel_replacement, 1);
    assert_ne!(
        replaced, source,
        "SSR Fresnel changed; iridescence transport must be updated"
    );
    let fallback = replaced.replace(
        "env_fallback(r, roughness)",
        "env_fallback(r, iridescence_reflection_roughness)",
    );
    assert_eq!(
        fallback
            .matches("env_fallback(r, iridescence_reflection_roughness)")
            .count(),
        2,
        "SSR env fallback call count changed; iridescence mip bias must be updated"
    );
    fallback
}

fn create_metadata_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    secondary_uv: bool,
    label: &'static str,
) -> wgpu::RenderPipeline {
    let mut buffers = vec![Vertex3D::desc()];
    if secondary_uv {
        buffers.push(secondary_uv_desc());
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
            entry_point: Some("fs_transparent_scene"),
            targets: &[Some(wgpu::ColorTargetState {
                format: IRIDESCENCE_SSR_FORMAT,
                blend: None,
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
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Equal),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_layered_ssr_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler = |binding, kind| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(kind),
        count: None,
    };
    let filterable = wgpu::TextureSampleType::Float { filterable: true };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("iridescence_ssr_layout"),
        entries: &[
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
            texture(1, wgpu::TextureSampleType::Depth),
            sampler(2, wgpu::SamplerBindingType::NonFiltering),
            texture(3, filterable),
            sampler(4, wgpu::SamplerBindingType::Filtering),
            texture(5, filterable),
            sampler(6, wgpu::SamplerBindingType::Filtering),
            texture(7, filterable),
            sampler(8, wgpu::SamplerBindingType::Filtering),
            texture(9, filterable),
            sampler(10, wgpu::SamplerBindingType::Filtering),
            texture(11, filterable),
        ],
    })
}

fn create_layered_ssr_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader_source = iridescence_ssr_shader_source(SSR_SHADER_WGSL);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("iridescence_ssr_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("iridescence_ssr_pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("iridescence_ssr_pipeline"),
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
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl Renderer {
    fn ensure_scene_iridescence_ssr_resources(&mut self) {
        if self.scene_iridescence_ssr_resources.is_some() {
            return;
        }
        if !self.ensure_scene_layered_pbr_resources() {
            return;
        }
        let material_layout = &self
            .scene_layered_pbr_resources
            .as_ref()
            .expect("layered scene resources initialized")
            .material_layout;
        let metadata_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iridescence_ssr_metadata_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&self.uniform_3d_layout),
                    Some(&self.lighting_layout),
                    Some(material_layout),
                    Some(&self.joint_layout),
                ],
                immediate_size: 0,
            });
        let scalar_source = iridescence_metadata_shader_source(SCENE_SHADER, false);
        let uv1_source = iridescence_metadata_shader_source(SCENE_SHADER, true);
        let scalar_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iridescence_ssr_metadata_shader"),
                source: wgpu::ShaderSource::Wgsl(scalar_source.into()),
            });
        let uv1_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iridescence_ssr_metadata_uv1_shader"),
                source: wgpu::ShaderSource::Wgsl(uv1_source.into()),
            });
        let scalar_metadata_pipeline = create_metadata_pipeline(
            &self.device,
            &metadata_layout,
            &scalar_shader,
            false,
            "iridescence_ssr_metadata_pipeline",
        );
        let uv1_metadata_pipeline = create_metadata_pipeline(
            &self.device,
            &metadata_layout,
            &uv1_shader,
            true,
            "iridescence_ssr_metadata_uv1_pipeline",
        );
        let (width, height) = self.render_extent();
        let metadata_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iridescence_ssr_metadata"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: IRIDESCENCE_SSR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let metadata_view = metadata_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let ssr_layout = create_layered_ssr_layout(&self.device);
        let ssr_pipeline = create_layered_ssr_pipeline(&self.device, &ssr_layout);
        self.scene_iridescence_ssr_resources = Some(SceneIridescenceSsrResources {
            _metadata_texture: metadata_texture,
            metadata_view,
            scalar_metadata_pipeline,
            uv1_metadata_pipeline,
            ssr_layout,
            ssr_pipeline,
        });
        self.created_pipelines(3);
        self.ssr_layered_bg_cache = None;
        log::info!(
            "bloom materials: lazy iridescence SSR transport enabled \
             (ordinary SSR path remains unchanged)"
        );
    }

    fn cached_iridescence_visible(&self, camera_planes: &[[f32; 4]; 6]) -> bool {
        if self.dbg_skip("cached_models") {
            return false;
        }
        self.model_draw_commands.iter().any(|command| {
            let Some(Some(meshes)) = self.model_gpu_cache.get(&command.cache_handle) else {
                return false;
            };
            let Some(mesh) = meshes.get(command.mesh_idx) else {
                return false;
            };
            if !mesh.layered_pbr.has_iridescence()
                || mesh.alpha_mode == MaterialAlphaMode::Blend
                || (self.imported_refraction_enabled && mesh.transmission.is_active())
                || mesh.layered_material_bg.is_none()
                || (self.gpu_driven.submitting()
                    && !command.skinned
                    && matches!(&mesh.geometry, gpu_driven::MeshGeometry::Shared(_)))
            {
                return false;
            }
            let (world_min, world_max) = command
                .bounds_override
                .unwrap_or_else(|| transform_aabb(&command.model, mesh.local_min, mesh.local_max));
            world_min[0] > world_max[0]
                || !crate::scene::aabb_outside_frustum(camera_planes, world_min, world_max)
        })
    }

    fn append_cached_iridescence_draws<'a>(
        &'a self,
        out: &mut Vec<ImportedIridescenceDrawRef<'a>>,
        camera_planes: &[[f32; 4]; 6],
    ) {
        if self.dbg_skip("cached_models") {
            return;
        }
        for command in &self.model_draw_commands {
            let Some(Some(meshes)) = self.model_gpu_cache.get(&command.cache_handle) else {
                continue;
            };
            let Some(mesh) = meshes.get(command.mesh_idx) else {
                continue;
            };
            if !mesh.layered_pbr.has_iridescence()
                || mesh.alpha_mode == MaterialAlphaMode::Blend
                || (self.imported_refraction_enabled && mesh.transmission.is_active())
                || (self.gpu_driven.submitting()
                    && !command.skinned
                    && matches!(&mesh.geometry, gpu_driven::MeshGeometry::Shared(_)))
            {
                continue;
            }
            let Some(material) = mesh.layered_material_bg.as_ref() else {
                continue;
            };
            let (world_min, world_max) = command
                .bounds_override
                .unwrap_or_else(|| transform_aabb(&command.model, mesh.local_min, mesh.local_max));
            if world_min[0] <= world_max[0]
                && crate::scene::aabb_outside_frustum(camera_planes, world_min, world_max)
            {
                continue;
            }
            let secondary_uv = if mesh.layered_uses_uv1 {
                let Some(buffer) = mesh.layered_uv1_buffer.as_ref() else {
                    continue;
                };
                Some(buffer)
            } else {
                None
            };
            let (mesh_draw, vertex_byte_offset, index_byte_offset) = if secondary_uv.is_some() {
                self.gpu_driven
                    .mesh_draw_localized(&mesh.geometry, mesh.index_count)
            } else {
                (
                    self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count),
                    0,
                    0,
                )
            };
            out.push(ImportedIridescenceDrawRef {
                uniforms: &self.model_uniform_bind_groups[command.uniform_slot],
                material,
                mesh: mesh_draw,
                secondary_uv,
                vertex_byte_offset,
                index_byte_offset,
            });
        }
    }

    pub(super) fn record_layered_iridescence_ssr_metadata(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        scene: &crate::scene::SceneGraph,
    ) {
        self.iridescence_ssr_active = false;
        if !self.ssr_enabled || self.pt_owns_frame() || self.scene_layered_pbr_resources.is_none() {
            return;
        }
        let camera_vp = mat4_multiply(
            self.current_proj_matrix_unjittered,
            self.current_view_matrix,
        );
        let camera_planes = crate::scene::extract_frustum_planes(&camera_vp);
        let cached_visible = self.cached_iridescence_visible(&camera_planes);
        let scene_visible = !self.dbg_skip("scene_graph")
            && scene.has_visible_opaque_iridescence(self.imported_refraction_enabled);
        if !cached_visible && !scene_visible {
            return;
        }

        self.ensure_scene_iridescence_ssr_resources();
        let mut draws = Vec::new();
        if cached_visible {
            self.append_cached_iridescence_draws(&mut draws, &camera_planes);
        }
        if scene_visible {
            scene.append_opaque_iridescence_draws(&mut draws);
        }
        if draws.is_empty() {
            return;
        }

        profiler.begin("iridescence_ssr_metadata");
        let timestamp_writes = profiler.pass_timestamp_writes("iridescence_ssr_metadata");
        let resources = self
            .scene_iridescence_ssr_resources
            .as_ref()
            .expect("visible iridescence initialized SSR resources");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iridescence_ssr_metadata_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &resources.metadata_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
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
        pass.set_bind_group(1, &self.lighting_bind_group, &[]);
        pass.set_bind_group(3, &self.joint_bind_group, &[]);
        let mut current_uv1 = None;
        for draw in draws {
            let uses_uv1 = draw.secondary_uv.is_some();
            if current_uv1 != Some(uses_uv1) {
                pass.set_pipeline(resources.metadata_pipeline(uses_uv1));
                current_uv1 = Some(uses_uv1);
            }
            pass.set_bind_group(0, draw.uniforms, &[]);
            pass.set_bind_group(2, draw.material, &[]);
            pass.set_vertex_buffer(0, draw.mesh.vertex.slice(draw.vertex_byte_offset..));
            if let Some(secondary_uv) = draw.secondary_uv {
                pass.set_vertex_buffer(1, secondary_uv.slice(..));
            }
            pass.set_index_buffer(
                draw.mesh.index.slice(draw.index_byte_offset..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(draw.mesh.index_range(), draw.mesh.base_vertex, 0..1);
        }
        drop(pass);
        profiler.end("iridescence_ssr_metadata");
        self.iridescence_ssr_active = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) {
        wgpu::naga::front::wgsl::parse_str(source).unwrap_or_else(|error| {
            panic!(
                "generated WGSL failed to parse:\n{}\n{error:?}",
                error.emit_to_string(source)
            )
        });
    }

    #[test]
    fn metadata_variants_parse_and_strip_unneeded_lighting() {
        for secondary_uv in [false, true] {
            let source = iridescence_metadata_shader_source(SCENE_SHADER, secondary_uv);
            parse(&source);
            assert!(source.contains("layered_raw_iridescence_base_fresnel("));
            assert!(source.contains("iridescence_factor"));
            assert!(!source.contains("// --- Split-sum IBL"));
            assert!(!source.contains("let em_tex_sample ="));
        }
    }

    #[test]
    fn layered_ssr_variant_adds_one_texture_and_preserves_base_source() {
        let source = iridescence_ssr_shader_source(SSR_SHADER_WGSL);
        parse(&source);
        assert!(source.contains("@binding(11) var iridescence_fresnel_tex"));
        assert!(source.contains("let ordinary_fresnel"));
        assert!(!SSR_SHADER_WGSL.contains("iridescence_fresnel_tex"));
        assert!(!SSR_SHADER_WGSL.contains("@binding(11)"));
    }
}
