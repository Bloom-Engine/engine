//! EN-011 planar-reflection probe pass: mirrored PerView build,
//! per-probe bind-group cache, mirrored-frustum culling and the two
//! probe render passes (materials + cached models). Split from
//! renderer/mod.rs (2000-line file policy — same pattern as
//! shadow_pass.rs / model_draw.rs / scene_pass.rs).

use super::*;

impl Renderer {
    /// EN-011 — render every registered probe's RT for this frame.
    /// Called from `end_frame_with_scene` BEFORE the main material
    /// pass so the probe textures are ready when materials sample
    /// them.
    ///
    /// For each probe:
    ///   1. Build the mirrored PerView (camera reflected across the
    ///      probe's plane; same projection as the main camera).
    ///   2. Upload the mirrored PerView to the probe's UBO.
    ///   3. Begin a render pass against the probe's RT + depth (clear
    ///      colour to fog, depth to 1.0).
    ///   4. Walk material_system.commands and dispatch each non-
    ///      excluded draw with the mirrored per-view bind group.
    ///
    /// V1 cull list: excludes any material whose handle equals one
    /// of the materials linked to a probe (so the water plane itself
    /// doesn't reflect). Future revisions can expand this with a
    /// hardcoded foliage / particle bucket filter via the bucket
    /// metadata on each compiled pipeline.
    pub fn dispatch_planar_reflections(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &crate::scene::SceneGraph,
        profiler: &mut crate::profiler::Profiler,
    ) {
        if self.planar_probes.iter().all(|p| p.is_none()) { return; }
        // Scene-graph nodes render into the probe too (they share the
        // Vertex3D layout and the scene material bind-group layout), so a
        // fully retained-mode game gets real water reflections as well.
        let scene_draws = scene.reflect_draw_list();
        if self.material_system.commands.is_empty()
            && self.model_draw_commands.is_empty()
            && scene_draws.is_empty() { return; }

        // EN-011 — lazily build the single-target reflection pipeline + buffers
        // used to render cached models (trees/house) into the probe with a
        // mirrored VP. Owned layouts: g0 dynamic per-draw model uniform, g1
        // sun/ambient; g2 reuses the scene material layout for base colour.
        const REFLECT_STRIDE: u64 = 256;
        const REFLECT_MAX_DRAWS: usize = 1024;
        if self.reflect_scene_pipeline.is_none() {
            let model_dyn_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reflect_model_dyn_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(128),
                    },
                    count: None,
                }],
            });
            let shadow_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
                binding, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
            let light_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reflect_light_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false, min_binding_size: None,
                        },
                        count: None,
                    },
                    // Shadow cascades + comparison sampler so the mirrored
                    // scene is sun-shadowed like the real one (the probe
                    // previously rendered everything fully lit, which made
                    // water reflections disagree with the scene above them).
                    shadow_tex_entry(1),
                    shadow_tex_entry(2),
                    shadow_tex_entry(3),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reflect_scene_shader"),
                source: wgpu::ShaderSource::Wgsl(REFLECT_SCENE_WGSL.into()),
            });
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("reflect_scene_pl"),
                bind_group_layouts: &[Some(&model_dyn_layout), Some(&light_layout), Some(&self.scene_material_layout)],
                immediate_size: 0,
            });
            let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("reflect_scene_pipeline"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader, entry_point: Some("vs_reflect"),
                    buffers: &[Vertex3D::desc()], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader, entry_point: Some("fs_reflect"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None, ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT, depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(), bias: Default::default(),
                }),
                multisample: Default::default(), multiview_mask: None, cache: None,
            });
            let model_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reflect_model_buf"),
                size: REFLECT_STRIDE * REFLECT_MAX_DRAWS as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let model_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflect_model_bg"), layout: &model_dyn_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &model_buf, offset: 0, size: std::num::NonZeroU64::new(128),
                    }),
                }],
            });
            // sun_dir + sun_color + ambient + cam_pos + shadow_splits (5 vec4)
            // + 3 cascade mat4s = 80 + 192 = 272 bytes.
            let light_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reflect_light_buf"), size: 272,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let light_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflect_light_bg"), layout: &light_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                ],
            });
            self.reflect_scene_pipeline = Some(pipeline);
            self.reflect_model_buf = Some(model_buf);
            self.reflect_model_bg = Some(model_bg);
            self.reflect_light_buf = Some(light_buf);
            self.reflect_light_bg = Some(light_bg);
        }
        // Sun/ambient + shadow data for the reflection shading (same values
        // as the main pass, so the mirrored scene is lit AND shadowed like
        // the real one). shadow_splits.w carries the shadows-enabled flag —
        // in THIS struct .w is free (no LOD-bias tenant).
        {
            let ld = self.lighting_uniforms.light_dir;
            let lc = self.lighting_uniforms.light_color;
            let amb = self.lighting_uniforms.ambient;
            let cam = self.current_camera_pos;
            let sp = self.lighting_uniforms.shadow_cascade_splits;
            let shadows_flag = if self.shadow_map.enabled { 1.0 } else { 0.0 };
            let mut light_data = [0.0f32; 68];
            light_data[0..4].copy_from_slice(&[ld[0], ld[1], ld[2], ld[3]]);
            light_data[4..8].copy_from_slice(&[lc[0], lc[1], lc[2], 0.0]);
            light_data[8..12].copy_from_slice(&[amb[0], amb[1], amb[2], amb[3]]);
            light_data[12..16].copy_from_slice(&[cam[0], cam[1], cam[2], 0.0]);
            light_data[16..20].copy_from_slice(&[sp[0], sp[1], sp[2], shadows_flag]);
            let vps = &self.lighting_uniforms.shadow_cascade_vps;
            for c in 0..3 {
                for col in 0..4 {
                    for row in 0..4 {
                        light_data[20 + c * 16 + col * 4 + row] = vps[c][col][row];
                    }
                }
            }
            if let Some(buf) = &self.reflect_light_buf {
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&light_data));
            }
        }

        // Build the V1 exclude set: every material linked to any
        // probe. The water material itself shouldn't appear in its
        // own reflection (it'd black-on-black self-occlude the
        // surface).
        let mut excluded: std::collections::HashSet<material_system::MaterialHandle> =
            std::collections::HashSet::new();
        for (i, probe_link) in self.material_system.material_reflection_probe.iter().enumerate() {
            if probe_link.is_some() {
                excluded.insert((i + 1) as material_system::MaterialHandle);
            }
        }

        // Cache main-pass per-view inputs once outside the loop.
        let main_view = self.current_view_matrix;
        let proj      = self.current_proj_matrix;
        let cam_pos   = self.current_camera_pos;

        // Snapshot the existing PerView uniforms by reconstructing
        // the same struct material_system_begin_frame writes — we
        // need a fresh copy per probe to swap view/view_proj.
        let base_per_view = material_system::PerViewUniforms {
            view:           main_view,
            proj,
            view_proj:      self.current_vp_matrix,
            // EN-022 fix: velocity reference (prev unjittered VP +
            // current jitter), so material shaders computing
            // `prev_view_proj * world` get true zero velocity on
            // static geometry instead of TAA jitter-delta noise.
            prev_view_proj: self.velocity_ref_vp,
            inv_proj:       self.current_inv_proj_matrix,
            camera_pos: [
                cam_pos[0], cam_pos[1], cam_pos[2],
                self.lighting_uniforms.camera_pos[3],
            ],
            camera_dir: [0.0, 0.0, -1.0, 70.0_f32.to_radians()],
            ambient:    self.lighting_uniforms.ambient,
            fog:        [self.fog_color[0], self.fog_color[1], self.fog_color[2], self.fog_density],
            sun_dir:    self.lighting_uniforms.light_dir,
            sun_color:  self.lighting_uniforms.light_color,
            dir_light_count:   self.lighting_uniforms.dir_light_count,
            dir_lights:        std::array::from_fn(|i| material_system::PerViewDirLight {
                direction: self.lighting_uniforms.dir_lights[i].direction,
                color:     self.lighting_uniforms.dir_lights[i].color,
            }),
            point_light_count: self.lighting_uniforms.point_light_count,
            point_lights:      std::array::from_fn(|i| material_system::PerViewPointLight {
                position: self.lighting_uniforms.point_lights[i].position,
                color:    self.lighting_uniforms.point_lights[i].color,
            }),
            shadow_splits:   self.lighting_uniforms.shadow_cascade_splits,
            shadow_view:     self.lighting_uniforms.shadow_view_matrix,
            shadow_cascades: self.lighting_uniforms.shadow_cascade_vps,
        };

        // Iterate probes by index — we mutate `material_system`'s
        // commands view while iterating, so collect the work first.
        let probe_count = self.planar_probes.len();
        for i in 0..probe_count {
            let (plane_y, normal, color_view, depth_view,
                 aux_material_view, aux_velocity_view, aux_albedo_view) =
                match &self.planar_probes[i] {
                    Some(p) => (p.plane_y, p.normal, p.color_view.clone(), p.depth_view.clone(),
                                p.aux_material_view.clone(), p.aux_velocity_view.clone(),
                                p.aux_albedo_view.clone()),
                    None => continue,
                };
            let view_buf = match self.planar_probe_view_buffers[i].as_ref() {
                Some(b) => b, None => continue,
            };

            // Mirror the camera + recompute view_proj for the probe.
            let mirror_view = planar_reflection::mirrored_view(main_view, plane_y, normal);
            let mirror_cam  = planar_reflection::mirrored_camera_pos(cam_pos, plane_y, normal);

            // EN-011 V2 — oblique near-plane clip. Replace the
            // projection's near plane with the water plane (in
            // mirror-eye-space) so geometry below the plane is
            // clipped at the rasterizer instead of polluting the
            // reflection edge.
            //
            // World-space plane equation: `N · p + d_w = 0` with
            // kept-side `N · p + d_w > 0` (above water). For a
            // horizontal mirror at world y = plane_y with normal
            // +Y, d_w = -plane_y so the kept side is y > plane_y.
            // Transformed via mirror-view's inverse-transpose, the
            // eye-space plane defines the same physical half-space
            // (the side the kept geometry lives on after the
            // reflection has rolled it through the view).
            let d_w = -(normal[0] * 0.0 + normal[1] * plane_y + normal[2] * 0.0);
            let plane_world = [normal[0], normal[1], normal[2], d_w];
            let plane_eye   = planar_reflection::world_plane_to_eye_space(mirror_view, plane_world);
            let mirror_proj = planar_reflection::oblique_proj(proj, plane_eye);
            let mirror_vp   = mat4_multiply(mirror_proj, mirror_view);

            let mut per_view = base_per_view;
            per_view.view      = mirror_view;
            per_view.proj      = mirror_proj;
            per_view.view_proj = mirror_vp;
            per_view.inv_proj  = planar_reflection::inv_proj_for(mirror_proj);
            per_view.camera_pos[0] = mirror_cam[0];
            per_view.camera_pos[1] = mirror_cam[1];
            per_view.camera_pos[2] = mirror_cam[2];
            // prev_view_proj stays as the main camera's previous VP
            // — TAA reprojection isn't meaningful for the reflection
            // probe (we don't temporally accumulate it), so this is
            // benign.
            self.queue.write_buffer(view_buf, 0, bytemuck::bytes_of(&per_view));

            // EN-011 V2 — rebuild the per-probe PerView bind group with
            // the live env / BRDF / shadow views. V1 bound 1×1 stub
            // textures here, which left mirrored draws lit by a flat
            // grey IBL (no specular reflections, no sun shadow) — the
            // reflection painting the lit scene differently from the
            // main pass made the surface look "off". Rebuilding once
            // per probe per frame is cheap (a single bind-group
            // create); it also picks up any env hot-load that
            // happens between frames without needing explicit dirty
            // tracking on the probe side.
            //
            // The sky env (binding 1) and env_diffuse (binding 3)
            // default to the renderer's 1×1 grey fallback when no HDR
            // is loaded — same default the main pass uses, so the
            // reflection's IBL stays consistent pre/post
            // `load_env_from_hdr`. The sky_texture's view doesn't sit
            // on a struct field (it's owned by `sky_bind_group`), so
            // we build fresh views here each frame; that's a cheap
            // Arc bump on the underlying wgpu Texture.
            // Cached per-probe PerView bind group — inputs (env, BRDF,
            // shadow views, the per-probe UBO) are all stable objects;
            // env (re)loads and probe creation clear the cache slot.
            if self.planar_probe_view_bgs[i].is_none() {
                let sky_view_owned: Option<wgpu::TextureView> = self.sky_texture
                    .as_ref()
                    .map(|t| t.create_view(&Default::default()));
                let env_view: &wgpu::TextureView = sky_view_owned
                    .as_ref()
                    .unwrap_or(&self.scene_env_default_view);
                let diffuse_view_owned: Option<wgpu::TextureView> = self.env_diffuse_texture
                    .as_ref()
                    .map(|t| t.create_view(&Default::default()));
                let env_diffuse_view: &wgpu::TextureView = diffuse_view_owned
                    .as_ref()
                    .unwrap_or(&self.scene_env_default_view);
                self.planar_probe_view_bgs[i] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("planar_probe_per_view_bg_live"),
                    layout: &self.material_system.layouts.per_view,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                        // env (specular) tex + sampler
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.env_sampler) },
                        // env diffuse tex
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(env_diffuse_view) },
                        // BRDF LUT tex + sampler
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.brdf_lut_view) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.brdf_lut_sampler) },
                        // 3 shadow cascades — same depth views the main
                        // pass binds, so the reflection picks up sun
                        // shadows without re-rendering the cascades.
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                    ],
                }));
            }
            let probe_view_bg = self.planar_probe_view_bgs[i].as_ref().unwrap();

            // Stage each surviving draw's [mirror_mvp, model] into ONE
            // batched upload (this was ~450 individual 128 B
            // `write_buffer` calls per probe per frame), and record the
            // draw list. Draws are culled against the MIRRORED frustum —
            // the oblique near plane doubles as the water-plane clip, so
            // below-plane geometry culls here too. Scene-graph nodes
            // append after the cached models in the same slot space —
            // creation order decides who survives the cap, so games
            // should create hero geometry before filler (grass).
            let mirror_planes = crate::scene::extract_frustum_planes(&mirror_vp);
            let mut reflect_draws: Vec<(u64, usize, u32)> = Vec::new();
            let mut node_slots: Vec<(usize, u32)> = Vec::new();
            if let Some(model_buf) = &self.reflect_model_buf {
                let stride = REFLECT_STRIDE as usize;
                let mut staged: Vec<u8> = Vec::with_capacity(stride * 128);
                for cmd in self.model_draw_commands.iter() {
                    // REFLECT_SCENE_WGSL can't skin — a skinned draw would
                    // mirror its bind pose at the origin. Skinned models
                    // (enemies) were never reflected on the old immediate
                    // path either, so skipping preserves that.
                    if cmd.skinned { continue; }
                    let slot = reflect_draws.len();
                    if slot >= REFLECT_MAX_DRAWS { break; }
                    let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) else { continue };
                    if cmd.mesh_idx >= meshes.len() { continue; }
                    let mesh = &meshes[cmd.mesh_idx];
                    let (wmin, wmax) =
                        transform_aabb(&cmd.model, mesh.local_min, mesh.local_max);
                    if wmin[0] <= wmax[0]
                        && crate::scene::aabb_outside_frustum(&mirror_planes, wmin, wmax)
                    {
                        continue;
                    }
                    let mirror_mvp = mat4_multiply(mirror_vp, cmd.model);
                    let base = staged.len();
                    staged.resize(base + stride, 0);
                    staged[base..base + 64].copy_from_slice(bytemuck::bytes_of(&mirror_mvp));
                    staged[base + 64..base + 128].copy_from_slice(bytemuck::bytes_of(&cmd.model));
                    reflect_draws.push((cmd.cache_handle, cmd.mesh_idx, slot as u32));
                }
                for (i, (_vb, _ib, _ic, _bg, model, wmin, wmax)) in scene_draws.iter().enumerate() {
                    let slot = reflect_draws.len() + node_slots.len();
                    if slot >= REFLECT_MAX_DRAWS { break; }
                    if wmin[0] <= wmax[0]
                        && crate::scene::aabb_outside_frustum(&mirror_planes, *wmin, *wmax)
                    {
                        continue;
                    }
                    let mirror_mvp = mat4_multiply(mirror_vp, *model);
                    let base = staged.len();
                    staged.resize(base + stride, 0);
                    staged[base..base + 64].copy_from_slice(bytemuck::bytes_of(&mirror_mvp));
                    staged[base + 64..base + 128].copy_from_slice(bytemuck::bytes_of(model));
                    node_slots.push((i, slot as u32));
                }
                if !staged.is_empty() {
                    self.queue.write_buffer(model_buf, 0, &staged);
                }
            }

            // Clear the probe to transparent black. Geometry fragments write
            // alpha 1, so the water shader can blend the probe over its analytic
            // sky by alpha (a=0 → no reflected geometry → show the sky dome).
            let clear_color = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let view_bg = probe_view_bg;
            let cache   = &self.model_gpu_cache;
            let mat_sys = &self.material_system;
            let refl_pipeline = self.reflect_scene_pipeline.as_ref();
            let refl_model_bg = self.reflect_model_bg.as_ref();
            let refl_light_bg = self.reflect_light_bg.as_ref();
            // Pass A — user materials. Opaque-profile material pipelines
            // (and their `_reflection` siblings) target the full opaque
            // G-buffer layout, so this pass presents the same four
            // attachments; the three aux targets are probe-resolution
            // dummies cleared here and discarded at store. Only the hdr
            // attachment (and depth, which pass B tests against) is kept.
            {
                let aux_ops = wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Discard,
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_planar_reflection_materials"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &color_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(clear_color),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &aux_material_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: aux_ops,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &aux_velocity_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: aux_ops,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &aux_albedo_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: aux_ops,
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: profiler.pass_timestamp_writes("planar_probe_materials"),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                mat_sys.dispatch_with_view(
                    &mut pass, view_bg,
                    // Excluded: probe-linked materials (the water plane
                    // itself) and anything flagged not-probe-visible
                    // (authoring control — e.g. sub-pixel instanced
                    // grass in a 512² probe).
                    |handle| !excluded.contains(&handle)
                        && mat_sys.material_probe_visible(handle),
                    // EN-011 V2 — swap to each material's sibling
                    // pipeline with cull_mode flipped Front→Back.
                    // Reflection mirrors world-space, which inverts
                    // triangle winding; without the flip, single-
                    // sided opaque geometry renders inside-out in
                    // the probe's RT. Cutout materials have
                    // `reflection_pipeline = None` and gracefully fall
                    // back to the main pipeline (no cull change needed
                    // since they're already double-sided); translucent-
                    // profile materials are skipped inside (their
                    // single-target pipelines can't render into this
                    // 4-target pass).
                    true,
                    // Instance tiles cull against the MIRRORED frustum.
                    Some(&mirror_planes),
                    |handle, idx| {
                        if let Some(Some(meshes)) = cache.get(&handle) {
                            if idx < meshes.len() {
                                let mesh = &meshes[idx];
                                return Some((
                                    &mesh.vb, &mesh.ib, mesh.index_count,
                                    mesh.local_min, mesh.local_max,
                                ));
                            }
                        }
                        None
                    },
                );
            }

            // Pass B — cached models (trees/house/foliage) + scene-graph
            // nodes with the single-target REFLECT_SCENE pipeline. Loads
            // pass A's color + depth so the two batches depth-test
            // against each other.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_planar_reflection_models"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: profiler.pass_timestamp_writes("planar_probe_models"),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if let (Some(rp), Some(rmbg), Some(rlbg)) =
                    (refl_pipeline, refl_model_bg, refl_light_bg)
                {
                    if !reflect_draws.is_empty() || !node_slots.is_empty() {
                        pass.set_pipeline(rp);
                        pass.set_bind_group(1, rlbg, &[]);
                        for (handle, midx, slot) in &reflect_draws {
                            if let Some(Some(meshes)) = cache.get(handle) {
                                if *midx < meshes.len() {
                                    let mesh = &meshes[*midx];
                                    pass.set_bind_group(0, rmbg, &[*slot * REFLECT_STRIDE as u32]);
                                    pass.set_bind_group(2, &mesh.material_bg, &[]);
                                    pass.set_vertex_buffer(0, mesh.vb.slice(..));
                                    pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
                                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                                }
                            }
                        }
                        // Scene-graph nodes: same pipeline — node geometry is
                        // Vertex3D and node material bind groups share the
                        // scene material layout the pipeline's g2 expects.
                        for (i, slot) in &node_slots {
                            let (vb, ib, index_count, mat_bg, _model, _wmin, _wmax) =
                                &scene_draws[*i];
                            pass.set_bind_group(0, rmbg, &[*slot * REFLECT_STRIDE as u32]);
                            pass.set_bind_group(2, *mat_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*index_count, 0, 0..1);
                        }
                    }
                }
            }
        }
    }
}
