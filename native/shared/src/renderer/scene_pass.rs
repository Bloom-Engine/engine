//! The HDR scene pass: sky-view LUT refresh, sky + immediate-mode 3D +
//! retained scene graph rendered into the linear-HDR MRT set (HDR +
//! material + velocity + albedo + depth), followed by the opaque
//! material pass running on the inner render graph. Split from
//! end_frame_with_scene (2000-line file policy + render-graph
//! migration).

use super::*;

impl Renderer {
    pub(super) fn record_hdr_scene_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        scene: &mut crate::scene::SceneGraph,
    ) {
        // Rebind: the immediate-mode 3D upload just before this call
        // checks the same predicate; vertices_3d is untouched between.
        let has_3d = !self.vertices_3d.is_empty();
        // ============================================================
        // HDR pass: sky + 3D + scene → linear HDR offscreen RT.
        // ============================================================
        // The composite-tonemap pass downstream reads this RT and
        // writes the final image to the sRGB surface. Keeping the
        // intermediate radiance in HDR sets up a future bloom pass
        // and means tonemap + sRGB encode happen exactly once, in
        // one place.
        // EN-005 Phase 2 — refresh the sky-view LUT before the HDR
        // pass opens. The compute dispatch can't be nested inside a
        // render pass, and `maybe_update_sky_view_lut` is a no-op
        // unless the sun (or atmosphere knobs) actually changed.
        // EN-005 V2 — also re-bake the aerial-perspective volume,
        // which must happen every frame because the camera moves.
        if self.procedural_sky_enabled {
            self.maybe_update_sky_view_lut();
            self.dispatch_aerial_perspective_lut();
        }
        self.prepare_gpu_driven_camera(encoder, scene);

        // EN-044 — DEPTH PREPASS over the cached-model draws.
        //
        // The scene fragment shader can `discard` (alpha-cutout foliage), and a shader
        // that may discard cannot early-Z *write*: the GPU must run it in full before it
        // knows whether the pixel survives. So an 88-tree forest of overlapping leaf
        // cards shaded the whole 5-target MRT several layers deep and threw most of it
        // away. Measured: the forest alone was 5.6 ms of a 7.4 ms main_hdr_pass, and
        // dropping it took the title screen from 46.7 fps to the 60 fps vsync cap.
        //
        // Priming depth first turns that around. The prepass writes depth only (no MRT,
        // no lighting, alpha cutout honoured so cards keep their real silhouette), and
        // the main pass then early-Z *rejects* the occluded leaves before its shader
        // ever runs. The main pass tests LessEqual, not Less — the visible surface
        // arrives with a depth exactly equal to the one the prepass stored, and `Less`
        // would throw it away.
        //
        // Same vertex stage, so the foliage wind displaces identically in both and the
        // depths agree to the bit.
        // Runs even with no cached models, because it now owns the depth CLEAR that
        // main_hdr_pass used to do — skipping it would hand the main pass a depth
        // buffer full of last frame's garbage.
        profiler.begin("depth_prepass");
        // SH-055 diag — "prepass" skips this whole pass (clear included; the main
        // pass then loads undefined depth — visually wrong, timing-valid).
        if !self.dbg_skip("prepass") {
            let prepass_ts = profiler.pass_timestamp_writes("depth_prepass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_depth_prepass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: prepass_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // EN-063 — on wasm the prepass DRAWS are skipped (the pass still
            // owns the depth clear). The prepass/main pairing requires the two
            // pipelines to produce bit-identical depths; Tint (the browser's
            // WGSL compiler) does not preserve that even under @invariant once
            // the foliage-wind displacement is in the chain, and the main
            // pass's Equal test then discards whole leaf cards (white torn
            // canopies, streaks). The main pass runs the classic Less+write
            // pipeline on wasm instead — overdraw over early-Z, correctness
            // over speed.
            #[cfg(not(target_arch = "wasm32"))]
            if !self.dbg_skip("prepass_draws") {
                // SH-055 diag — keep pass+clear, skip draws
                if let Some(global_materials) =
                    self.material_system.indirection.global_bind_group.as_ref()
                {
                    self.gpu_driven.draw_depth(
                        &mut pass,
                        &self.lighting_bind_group,
                        global_materials,
                        &self.joint_bind_group,
                    );
                }
                pass.set_pipeline(&self.scene_depth_pipeline);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);
                let cam_vp = mat4_multiply(
                    self.current_proj_matrix_unjittered,
                    self.current_view_matrix,
                );
                let cam_planes = crate::scene::extract_frustum_planes(&cam_vp);
                for cmd in &self.model_draw_commands {
                    if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                        if cmd.mesh_idx < meshes.len() {
                            let mesh = &meshes[cmd.mesh_idx];
                            if mesh.alpha_mode == MaterialAlphaMode::Blend
                                || (self.imported_refraction_enabled
                                    && mesh.transmission.is_active())
                            {
                                continue;
                            }
                            if self.gpu_driven.submitting()
                                && !cmd.skinned
                                && matches!(&mesh.geometry, gpu_driven::MeshGeometry::Shared(_))
                            {
                                continue;
                            }
                            let (wmin, wmax) = cmd.bounds_override.unwrap_or_else(|| {
                                transform_aabb(&cmd.model, mesh.local_min, mesh.local_max)
                            });
                            if wmin[0] <= wmax[0]
                                && crate::scene::aabb_outside_frustum(&cam_planes, wmin, wmax)
                            {
                                continue;
                            }
                            let draw = self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count);
                            pass.set_bind_group(
                                0,
                                &self.model_uniform_bind_groups[cmd.uniform_slot],
                                &[],
                            );
                            pass.set_bind_group(2, &mesh.material_bg, &[]);
                            pass.set_vertex_buffer(0, draw.vertex.slice(..));
                            pass.set_index_buffer(draw.index.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(draw.index_range(), draw.base_vertex, 0..1);
                        }
                    }
                }
            }
        }
        profiler.end("depth_prepass");

        profiler.begin("main_hdr_pass");
        // SH-055 diag — "hdr_pass" skips the whole main HDR pass (prepass untouched).
        if !self.dbg_skip("hdr_pass") {
            // HDR clear: the user's clear_color is in 0-1 srgb-ish
            // range; treat it as the linear background for the HDR
            // RT. After tonemap it ends up roughly the same shade.
            let hdr_ts = profiler.pass_timestamp_writes("main_hdr_pass");
            // SH-055 — `lean_mrt` (Android): skip the material + albedo
            // attachments entirely (None at the same indices the pipelines in
            // mod.rs/material_pipeline.rs declare None for — wgpu-core requires
            // the render pass and the bound pipeline to agree index-for-index).
            // This is what actually drops main_hdr_pass's per-pixel byte
            // footprint below the Adreno GMEM-overflow threshold; see build.rs.
            #[cfg(lean_mrt)]
            let color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                None,
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.velocity_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Zero velocity = stationary pixel.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                None,
            ];
            #[cfg(not(lean_mrt))]
            let color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.material_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Blank pixels clear to metallic=0. SSR's
                        // `metallic < 0.2` gate early-outs before
                        // roughness is read, so the roughness
                        // component of the clear is dead — leaving
                        // it at 0 instead of 1 keeps the material
                        // texture black in frame captures and
                        // avoids a false "green G-buffer" readout
                        // if the RT is ever viewed as RGBA.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.velocity_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Zero velocity = stationary pixel.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.albedo_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Clear to zero albedo — pixels the scene
                        // doesn't cover (before sky writes) absorb
                        // indirect light fully. Sky then writes 0
                        // too so SSGI rays landing on sky don't
                        // re-tint bounce by background radiance.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_hdr_pass"),
                color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // EN-044 — LOAD, not Clear: the depth prepass just primed this
                        // buffer, and clearing it here would throw that away.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: hdr_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Sky uses the same env_intensity as IBL so the background
            // and lighting stay in sync — otherwise bumping IBL down
            // would leave the sky blown out.
            //
            // SH-055 diag — BLOOM_SKIP_SKY=1 (Android: `adb shell setprop
            // debug.bloom.skipsky 1`, propagated in JNI_OnLoad) skips the sky
            // draw entirely, to bisect the unexplained per-pixel frame cost on
            // Adreno. Read once; a mid-run flip needs an app restart anyway.
            static SKIP_SKY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let skip_sky = *SKIP_SKY.get_or_init(|| {
                std::env::var("BLOOM_SKIP_SKY")
                    .map(|v| v == "1")
                    .unwrap_or(false)
            });
            if !skip_sky {
                if self.procedural_sky_enabled {
                    self.render_procedural_sky_pass(
                        &mut pass,
                        self.lighting_uniforms.camera_pos[3],
                    );
                } else {
                    self.render_sky_pass(&mut pass, self.lighting_uniforms.camera_pos[3]);
                }
            }

            if has_3d && !self.dbg_skip("imm3d") {
                // SH-055 diag
                pass.set_pipeline(&self.pipeline_3d);
                pass.set_bind_group(0, &self.uniform_bind_group_3d, &[]);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);
                pass.set_vertex_buffer(0, self.persistent_vb_3d.slice(..));
                pass.set_index_buffer(self.persistent_ib_3d.slice(..), wgpu::IndexFormat::Uint32);

                if self.draw_calls_3d.is_empty() {
                    pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                    pass.draw_indexed(0..self.indices_3d.len() as u32, 0, 0..1);
                } else {
                    let num_calls = self.draw_calls_3d.len();
                    for i in 0..num_calls {
                        let call = &self.draw_calls_3d[i];
                        let next_start = if i + 1 < num_calls {
                            self.draw_calls_3d[i + 1].index_start
                        } else {
                            self.indices_3d.len() as u32
                        };
                        let count = next_start - call.index_start;
                        if count == 0 {
                            continue;
                        }
                        let tex_idx = call.texture_idx as usize;
                        if tex_idx < self.texture_bind_groups.len() {
                            pass.set_bind_group(2, &self.texture_bind_groups[tex_idx], &[]);
                        } else {
                            pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                        }
                        pass.draw_indexed(call.index_start..next_start, 0, 0..1);
                    }
                }
            }

            // Cached models + retained scene graph — both via scene_pipeline.
            let has_cached_models = !self.model_draw_commands.is_empty();
            if has_cached_models || scene.node_count() > 0 {
                if let Some(global_materials) =
                    self.material_system.indirection.global_bind_group.as_ref()
                {
                    self.gpu_driven.draw_main(
                        &mut pass,
                        &self.lighting_bind_group,
                        global_materials,
                        &self.joint_bind_group,
                        cfg!(not(target_arch = "wasm32")),
                    );
                }
                // EN-044 — cached models go through the PREPASSED pipeline (no depth
                // write, Equal test), because the depth prepass above already stored
                // their exact depth. That is what lets the hardware early-Z reject the
                // occluded leaf cards instead of shading every one of them.
                #[cfg(not(target_arch = "wasm32"))]
                pass.set_pipeline(&self.scene_pipeline_prepassed);
                // wasm: no prepass priming (see above) — classic Less + write.
                #[cfg(target_arch = "wasm32")]
                pass.set_pipeline(&self.scene_pipeline);
                #[cfg(not(target_arch = "wasm32"))]
                let base_cached_pipeline = &self.scene_pipeline_prepassed;
                #[cfg(target_arch = "wasm32")]
                let base_cached_pipeline = &self.scene_pipeline;
                let cached_prepassed = cfg!(not(target_arch = "wasm32"));
                let mut current_cached_pipeline = Some((false, false));
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);

                if has_cached_models {
                    // Frustum-cull cached-model draws against the camera. These
                    // commands (the forest: ~400 draws/frame) previously had no
                    // culling in any pass — everything behind the camera still
                    // paid bind-group switches + a full VS pass. AABBs are
                    // conservative (cache-time local bounds × model matrix), and
                    // the unjittered projection is used so TAA's sub-pixel
                    // jitter can't flicker a borderline caster (the AABB slop
                    // exceeds jitter by orders of magnitude).
                    let cam_vp = mat4_multiply(
                        self.current_proj_matrix_unjittered,
                        self.current_view_matrix,
                    );
                    let cam_planes = crate::scene::extract_frustum_planes(&cam_vp);
                    // SH-055 diag — "cached_models" skips the cached-model draws;
                    // BLOOM_MAX_MODELS (Android: setprop debug.bloom.maxmodels N)
                    // caps the count so the pathological draw can be found by
                    // binary search. One-time log lists every command.
                    {
                        static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                        LOGGED.get_or_init(|| {
                        log::warn!("[MODELS] {} cached draw commands", self.model_draw_commands.len());
                        for (i, cmd) in self.model_draw_commands.iter().enumerate() {
                            if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                                if cmd.mesh_idx < meshes.len() {
                                    let m = &meshes[cmd.mesh_idx];
                                    log::warn!(
                                        "[MODELS] #{i} handle={:x} mesh_idx={} indices={} skinned={}",
                                        cmd.cache_handle, cmd.mesh_idx, m.index_count,
                                        cmd.bounds_override.is_some()
                                    );
                                }
                            }
                        }
                    });
                    }
                    static MAX_MODELS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                    let max_models = *MAX_MODELS.get_or_init(|| {
                        std::env::var("BLOOM_MAX_MODELS")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(usize::MAX)
                    });
                    let cached_take = if self.dbg_skip("cached_models") {
                        0
                    } else {
                        max_models
                    };
                    for cmd in self.model_draw_commands.iter().take(cached_take) {
                        if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                            if cmd.mesh_idx < meshes.len() {
                                let mesh = &meshes[cmd.mesh_idx];
                                if mesh.alpha_mode == MaterialAlphaMode::Blend
                                    || (self.imported_refraction_enabled
                                        && mesh.transmission.is_active())
                                {
                                    continue;
                                }
                                if self.gpu_driven.submitting()
                                    && !cmd.skinned
                                    && matches!(&mesh.geometry, gpu_driven::MeshGeometry::Shared(_))
                                {
                                    continue;
                                }
                                // Skinned draws carry a pre-computed joint-union
                                // AABB (their rest AABB × model matrix would be
                                // wrong once posed); static draws derive theirs.
                                let (wmin, wmax) = cmd.bounds_override.unwrap_or_else(|| {
                                    transform_aabb(&cmd.model, mesh.local_min, mesh.local_max)
                                });
                                if wmin[0] <= wmax[0]
                                    && crate::scene::aabb_outside_frustum(&cam_planes, wmin, wmax)
                                {
                                    continue;
                                }
                                let layered_material = mesh.layered_material_bg.as_ref();
                                let layered_uv1 =
                                    layered_material.is_some() && mesh.layered_uses_uv1;
                                if current_cached_pipeline
                                    != Some((layered_material.is_some(), layered_uv1))
                                {
                                    if layered_material.is_some() {
                                        let resources = self
                                            .scene_layered_pbr_resources
                                            .as_ref()
                                            .expect("layered material initialized its pipelines");
                                        pass.set_pipeline(
                                            resources
                                                .opaque_pipeline(layered_uv1, cached_prepassed),
                                        );
                                    } else {
                                        pass.set_pipeline(base_cached_pipeline);
                                    }
                                    current_cached_pipeline =
                                        Some((layered_material.is_some(), layered_uv1));
                                }
                                let (draw, vertex_offset, index_offset) = if layered_uv1 {
                                    let Some(secondary_uv) = mesh.layered_uv1_buffer.as_ref()
                                    else {
                                        continue;
                                    };
                                    pass.set_vertex_buffer(1, secondary_uv.slice(..));
                                    let (draw, vertex_offset, index_offset) = self
                                        .gpu_driven
                                        .mesh_draw_localized(&mesh.geometry, mesh.index_count);
                                    (draw, vertex_offset, index_offset)
                                } else {
                                    (
                                        self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count),
                                        0,
                                        0,
                                    )
                                };
                                pass.set_bind_group(
                                    0,
                                    &self.model_uniform_bind_groups[cmd.uniform_slot],
                                    &[],
                                );
                                pass.set_bind_group(
                                    2,
                                    layered_material.unwrap_or(&mesh.material_bg),
                                    &[],
                                );
                                pass.set_vertex_buffer(0, draw.vertex.slice(vertex_offset..));
                                pass.set_index_buffer(
                                    draw.index.slice(index_offset..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                pass.draw_indexed(draw.index_range(), draw.base_vertex, 0..1);
                            }
                        }
                    }
                }

                // Retained scene-graph nodes are not in the prepass, so they still need
                // the depth-writing pipeline.
                if !self.dbg_skip("scene_graph") {
                    // SH-055 diag
                    pass.set_pipeline(&self.scene_pipeline);
                    scene.render_with_material_specializations(
                        &mut pass,
                        self.gpu_driven.submitting(),
                        self.imported_refraction_enabled,
                        &self.scene_pipeline,
                        self.scene_layered_pbr_resources.as_ref(),
                    );
                }
            }
        }
        profiler.end("main_hdr_pass");

        // EN-011 — render every registered planar reflection probe
        // BEFORE the main material pass so the probe RTs are
        // sampleable when materials run. No-op when no probes are
        // registered or no opaque material draws are queued.
        profiler.begin("planar_reflections");
        self.dispatch_planar_reflections(&mut *encoder, scene, profiler);
        profiler.end("planar_reflections");

        // User-material draws are part of the compiled `hdr_scene` node. The old
        // one-node graph rebuilt and topologically sorted this closure every frame.
        self.record_opaque_material_pass(encoder, profiler);
        // The legacy G-buffer cannot losslessly carry thin-film Fresnel. Replay
        // only visible opaque iridescent meshes into a lazy metadata target for
        // the later SSR pass; base-only frames return before allocating or
        // recording anything.
        self.record_layered_iridescence_ssr_metadata(encoder, profiler, scene);
    }
}

impl Renderer {
    fn draw_imported_refraction<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a crate::scene::SceneGraph,
        reactive: bool,
    ) {
        let camera_vp = mat4_multiply(
            self.current_proj_matrix_unjittered,
            self.current_view_matrix,
        );
        let camera_planes = crate::scene::extract_frustum_planes(&camera_vp);
        let mut draws: Vec<ImportedRefractiveDrawRef<'_>> = Vec::new();
        for (stable_id, command) in self.model_draw_commands.iter().enumerate() {
            let Some(Some(meshes)) = self.model_gpu_cache.get(&command.cache_handle) else {
                continue;
            };
            let Some(mesh) = meshes.get(command.mesh_idx) else {
                continue;
            };
            if !mesh.transmission.is_active() {
                continue;
            }
            let Some(material) = mesh.refractive_material_bg.as_ref() else {
                continue;
            };
            let (world_min, world_max) = command
                .bounds_override
                .unwrap_or_else(|| transform_aabb(&command.model, mesh.local_min, mesh.local_max));
            if world_min[0] <= world_max[0]
                && crate::scene::aabb_outside_frustum(&camera_planes, world_min, world_max)
            {
                continue;
            }
            let center = if world_min[0] <= world_max[0] {
                [
                    (world_min[0] + world_max[0]) * 0.5,
                    (world_min[1] + world_max[1]) * 0.5,
                    (world_min[2] + world_max[2]) * 0.5,
                ]
            } else {
                [
                    command.model[3][0],
                    command.model[3][1],
                    command.model[3][2],
                ]
            };
            let pivot = mat4_mul_vec4(
                &self.current_vp_matrix,
                &[center[0], center[1], center[2], 1.0],
            );
            let secondary_uv = if mesh.refractive_uses_uv1 {
                let Some(buffer) = mesh.refractive_uv1_buffer.as_ref() else {
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
            draws.push(ImportedRefractiveDrawRef {
                view_depth: pivot[3],
                stable_id,
                double_sided: mesh.double_sided,
                layered: mesh.refractive_layered,
                uniforms: &self.model_uniform_bind_groups[command.uniform_slot],
                material,
                mesh: mesh_draw,
                secondary_uv,
                vertex_byte_offset,
                index_byte_offset,
            });
        }
        scene.append_refractive_draws(
            &mut draws,
            &self.current_vp_matrix,
            self.model_draw_commands.len(),
        );
        draws.sort_by(|left, right| {
            right
                .view_depth
                .total_cmp(&left.view_depth)
                .then_with(|| left.stable_id.cmp(&right.stable_id))
        });

        pass.set_bind_group(1, &self.lighting_bind_group, &[]);
        pass.set_bind_group(3, &self.joint_bind_group, &[]);
        #[cfg(not(fold_scene_inputs))]
        pass.set_bind_group(
            4,
            self.scene_refractive_inputs_bg.as_ref().unwrap_or_else(|| {
                self.material_system
                    .scene_inputs_bg
                    .as_ref()
                    .expect("native imported refraction has scene snapshots")
            }),
            &[],
        );
        let mut current_pipeline_key = None;
        for draw in draws {
            let uses_uv1 = draw.secondary_uv.is_some();
            let pipeline_key = (draw.layered, draw.double_sided, uses_uv1);
            if current_pipeline_key != Some(pipeline_key) {
                let pipeline = if draw.layered {
                    self.scene_layered_refractive_resources
                        .as_ref()
                        .expect("layered refractive material initialized its pipelines")
                        .pipeline(uses_uv1, draw.double_sided, reactive)
                } else {
                    match (reactive, uses_uv1, draw.double_sided) {
                        (false, false, false) => self.scene_refractive_pipeline.as_ref(),
                        (false, false, true) => {
                            self.scene_refractive_double_sided_pipeline.as_ref()
                        }
                        (false, true, false) => self.scene_refractive_uv1_pipeline.as_ref(),
                        (false, true, true) => {
                            self.scene_refractive_uv1_double_sided_pipeline.as_ref()
                        }
                        (true, false, false) => self.scene_refractive_reactive_pipeline.as_ref(),
                        (true, false, true) => self
                            .scene_refractive_reactive_double_sided_pipeline
                            .as_ref(),
                        (true, true, false) => self.scene_refractive_reactive_uv1_pipeline.as_ref(),
                        (true, true, true) => self
                            .scene_refractive_reactive_uv1_double_sided_pipeline
                            .as_ref(),
                    }
                    .expect("selected imported-refraction pipeline must be initialized")
                };
                pass.set_pipeline(pipeline);
                current_pipeline_key = Some(pipeline_key);
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
    }
}

impl Renderer {
    fn prepare_gpu_driven_camera(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &crate::scene::SceneGraph,
    ) {
        self.gpu_driven.draw_scratch.clear();
        if !self.gpu_driven.enabled() {
            self.gpu_driven.stats.compatibility = self.model_draw_commands.len() as u32;
            return;
        }
        let mut compatibility = 0u32;
        for cmd in &self.model_draw_commands {
            let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) else {
                compatibility += 1;
                continue;
            };
            let Some(mesh) = meshes.get(cmd.mesh_idx) else {
                compatibility += 1;
                continue;
            };
            if mesh.alpha_mode == MaterialAlphaMode::Blend
                || (self.imported_refraction_enabled && mesh.transmission.is_active())
                || mesh.layered_pbr.is_active()
            {
                compatibility += 1;
                continue;
            }
            let gpu_driven::MeshGeometry::Shared(slice) = &mesh.geometry else {
                compatibility += 1;
                continue;
            };
            if cmd.skinned {
                compatibility += 1;
                continue;
            }
            let uniform_offset = cmd.uniform_slot * MODEL_UNIFORM_STRIDE;
            let uniform_end = uniform_offset + std::mem::size_of::<Uniforms3D>();
            let Some(bytes) = self.model_uniform_scratch.get(uniform_offset..uniform_end) else {
                compatibility += 1;
                continue;
            };
            let uniforms = bytemuck::pod_read_unaligned::<Uniforms3D>(bytes);
            let (wmin, wmax) = transform_aabb(&cmd.model, mesh.local_min, mesh.local_max);
            self.gpu_driven
                .draw_scratch
                .push(gpu_driven::GpuDrawRecord {
                    uniforms,
                    // Cached-model prepass semantics are two-sided (foliage
                    // and cutout cards rely on it). Bit 0 rides in the unused
                    // bounds lane and is consumed only by the depth shader.
                    bounds_min: [wmin[0], wmin[1], wmin[2], f32::from_bits(1)],
                    bounds_max: [wmax[0], wmax[1], wmax[2], 0.0],
                    draw: [
                        mesh.index_count,
                        slice.first_index,
                        slice.base_vertex as u32,
                        mesh.material_id.raw(),
                    ],
                });
        }
        let [scene_compatibility, frustum_visible, frustum_culled] =
            scene.append_gpu_driven_draws(&mut self.gpu_driven.draw_scratch);
        compatibility += scene_compatibility;
        let (frustum_visible, frustum_culled) =
            if self.gpu_driven.draw_scratch.len() < gpu_driven::GPU_DRIVEN_MIN_DRAWS {
                compatibility += self.gpu_driven.draw_scratch.len() as u32;
                self.gpu_driven.draw_scratch.clear();
                (0, 0)
            } else {
                (frustum_visible, frustum_culled)
            };
        let camera_vp = mat4_multiply(
            self.current_proj_matrix_unjittered,
            self.current_view_matrix,
        );
        let planes = crate::scene::extract_frustum_planes(&camera_vp);
        self.gpu_driven.prepare(
            &self.device,
            &self.queue,
            encoder,
            planes,
            compatibility,
            frustum_visible,
            frustum_culled,
        );
    }

    /// Translucent / refractive / additive material pass: after opaque,
    /// before post-FX; loads hdr_rt, depth read-only, back-to-front
    /// sorted; snapshots scene color for reads_scene materials. Split
    /// from end_frame_with_scene.
    pub(super) fn record_translucent_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        scene: &crate::scene::SceneGraph,
    ) {
        self.refractive_reflections_active = false;
        // ============================================================
        // Phase 4b — translucent / refractive / additive material pass
        // ============================================================
        //
        // Runs after opaque materials, before post-FX. Loads hdr_rt so
        // opaque output survives; alpha-blends into it. Depth is
        // bound as read-only so translucent draws participate in the
        // depth test without writing.
        //
        // If any submitted translucent material declared
        // `reads_scene = true`, we first snapshot hdr_rt into a
        // swapchain-sized transient and bind that as group 4
        // scene_color_tex for the dispatch. Free after the pass so
        // the transient pool reuses on the next frame.
        let has_imported_refraction = self.imported_refraction_enabled
            && (self.has_refractive_model_draws || scene.has_refractive_nodes());
        let has_layered_imported_transparency =
            self.has_layered_blend_model_draws || scene.has_layered_transparent_nodes();
        if !self.material_system.translucent_commands.is_empty()
            || self.has_blend_model_draws
            || scene.has_transparent_nodes()
            || has_imported_refraction
        {
            if has_imported_refraction {
                self.ensure_scene_refraction_resources();
            }
            if self.weighted_transparency_active {
                self.ensure_weighted_transparency_resources();
                if has_layered_imported_transparency {
                    self.ensure_scene_layered_pbr_weighted_resources();
                }
            }
            if self.temporal_reactive_active {
                if has_imported_refraction {
                    self.ensure_scene_refraction_reactive_resources();
                    self.ensure_scene_layered_refraction_reactive_resources();
                }
                if self.weighted_transparency_active {
                    self.ensure_weighted_transparency_reactive_resources();
                } else if self.has_blend_model_draws || scene.has_transparent_nodes() {
                    self.ensure_scene_transparent_reactive_resources();
                    if has_layered_imported_transparency {
                        self.ensure_scene_layered_pbr_reactive_resources();
                    }
                }
            }
            let has_sorted_imported_transparency = !self.weighted_transparency_active
                && (self.has_blend_model_draws || scene.has_transparent_nodes());
            let globally_interleave_sorted = has_sorted_imported_transparency
                && !self.material_system.translucent_commands.is_empty()
                && super::sorted_transparency::sorted_interleaving_enabled();
            // Custom commands retain their established stable in-place sort.
            // The mixed dispatcher merge-walks this list with the independently
            // sorted imported list, avoiding a second combined allocation.
            self.material_system.sort_translucent();
            if self.temporal_reactive_active && globally_interleave_sorted {
                self.material_system
                    .ensure_translucent_reactive_pipelines(&self.device);
            }
            profiler.begin("translucent_pass");
            let swap_w = self.surface_config.width;
            let swap_h = self.surface_config.height;
            self.transient_pool.begin_frame(swap_w, swap_h);

            // Phase 7 — run the impulse decay + splat compute BEFORE
            // we build scene_inputs so the front view reflects this
            // frame's submissions.
            self.impulse_field
                .update(&self.device, &self.queue, &mut *encoder);

            // Does any queued translucent material need the scene
            // colour snapshot?
            let custom_reads_scene = self.material_system.translucent_commands.iter().any(|c| {
                self.material_system
                    .pipelines
                    .get(c.material as usize - 1)
                    .and_then(|p| p.as_ref())
                    .map(|p| p.reads_scene)
                    .unwrap_or(false)
            });
            let needs_scene =
                custom_reads_scene || (cfg!(not(fold_scene_inputs)) && has_imported_refraction);

            // The snapshots mirror hdr_rt/depth, which live at RENDER
            // resolution (surface × render_scale) — not swapchain size.
            // Sizing them from the swapchain overruns the source copy
            // whenever render_scale < 1 (TSR upscaling).
            let (render_w, render_h) = self.render_extent();
            if needs_scene || self.weighted_transparency_active || self.temporal_reactive_active {
                let plan = self
                    .last_frame_plan
                    .as_ref()
                    .expect("active frame plan is installed before translucent execution");
                self.transient_pool
                    .prepare_compiled_plan(
                        &self.device,
                        plan,
                        (render_w, render_h),
                        (swap_w, swap_h),
                    )
                    .unwrap_or_else(|error| {
                        panic!("compiled transient allocation failed: {error}")
                    });
            }
            let compiled_snapshots = needs_scene.then(|| {
                let plan = self
                    .last_frame_plan
                    .as_ref()
                    .expect("active frame plan is installed before translucent execution");
                let color = plan
                    .resource("translucent-scene-color")
                    .expect("scene-reading topology declares its color snapshot")
                    .id;
                let depth = plan
                    .resource("translucent-scene-depth")
                    .expect("scene-reading topology declares its depth snapshot")
                    .id;
                (plan.plan_id, color, depth)
            });
            let compiled_reactive = self.temporal_reactive_active.then(|| {
                let plan = self
                    .last_frame_plan
                    .as_ref()
                    .expect("reactive transparency has an active frame plan");
                let reactive = plan
                    .resource("transparency-reactive")
                    .expect("reactive topology declares its coverage target")
                    .id;
                (plan.plan_id, reactive)
            });

            // Phase 4c — depth snapshot. wgpu forbids sampling a
            // texture that is also a depth-stencil attachment of the
            // same pass, so we copy the opaque depth buffer into a
            // transient before beginning the translucent pass and
            // bind the transient at group 4 binding 2. Acquired
            // whenever any translucent material reads_scene (same
            // gate as colour) — cheap enough that it's not worth a
            // separate `reads_depth` flag yet.
            // Snapshot hdr_rt + live depth -> transients.
            if needs_scene {
                let (plan_id, color, depth) =
                    compiled_snapshots.expect("scene-reading topology has compiled snapshots");
                let color_tex = self
                    .transient_pool
                    .compiled_texture(plan_id, color)
                    .expect("compiled color snapshot");
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.hdr_rt_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: color_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: render_w,
                        height: render_h,
                        depth_or_array_layers: 1,
                    },
                );
                let depth_tex = self
                    .transient_pool
                    .compiled_texture(plan_id, depth)
                    .expect("compiled depth snapshot");
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.depth_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::DepthOnly,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: depth_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::DepthOnly,
                    },
                    wgpu::Extent3d {
                        width: render_w,
                        height: render_h,
                        depth_or_array_layers: 1,
                    },
                );
                let color_view = self
                    .transient_pool
                    .compiled_view(plan_id, color)
                    .expect("compiled color snapshot view");
                let depth_view = self
                    .transient_pool
                    .compiled_view(plan_id, depth)
                    .expect("compiled depth snapshot view");
                let imp_view = self.impulse_field.front_view();
                let imp_samp = self.impulse_field.sampler();
                #[cfg(not(fold_scene_inputs))]
                let dedicated_refractive_inputs =
                    has_imported_refraction && self.scene_refractive_inputs_layout.is_some();
                #[cfg(fold_scene_inputs)]
                let dedicated_refractive_inputs = false;
                // The hierarchy owns a smaller dedicated group 4. Do not also
                // rebuild the legacy seven-binding group unless a custom
                // scene-reading material consumes it. The startup kill switch
                // has no dedicated layout and therefore preserves the exact
                // established bind-group route.
                if custom_reads_scene || !dedicated_refractive_inputs {
                    self.material_system.update_scene_inputs(
                        &self.device,
                        color_view,
                        Some(depth_view),
                        Some((imp_view, imp_samp)),
                    );
                }
                #[cfg(not(fold_scene_inputs))]
                if has_imported_refraction {
                    if let (Some(layout), Some(params_buffer)) = (
                        self.scene_refractive_inputs_layout.as_ref(),
                        self.scene_refractive_reflection_params_buffer.as_ref(),
                    ) {
                        let planar_probe = self.planar_probes.iter().flatten().next();
                        let planar_view = planar_probe
                            .map(|probe| &probe.color_view)
                            .unwrap_or(&self.scene_env_default_view);
                        let planar_plane = planar_probe.map(|probe| (probe.normal, probe.plane_y));
                        self.scene_refractive_inputs_bg =
                            Some(refractive_reflections::materialize_inputs(
                                &self.device,
                                &self.queue,
                                refractive_reflections::RefractiveInputs {
                                    layout,
                                    params_buffer,
                                    scene_color: color_view,
                                    scene_color_sampler: &self.composite_sampler,
                                    scene_depth: depth_view,
                                    planar_reflection: planar_view,
                                },
                                self.current_view_matrix,
                                self.current_proj_matrix,
                                self.ssr_enabled,
                                planar_plane,
                            ));
                        self.refractive_reflections_active =
                            self.ssr_enabled || planar_probe.is_some();
                    }
                }
            } else {
                // No refractive/depth-reading materials this frame —
                // still need a valid bind group. None → internal stubs.
                self.material_system.update_scene_inputs(
                    &self.device,
                    &self.hdr_rt_view,
                    None,
                    None,
                );
            }

            if self.weighted_transparency_active {
                let plan = self
                    .last_frame_plan
                    .as_ref()
                    .expect("weighted transparency has an active frame plan");
                let accumulation = plan
                    .resource("transparency-accumulation")
                    .expect("weighted topology declares accumulation")
                    .id;
                let revealage = plan
                    .resource("transparency-revealage")
                    .expect("weighted topology declares revealage")
                    .id;
                let key = (plan.plan_id, self.transient_pool.rebuild_epoch);
                if self.weighted_transparency_resolve_bind_group_key != Some(key) {
                    let accumulation_view = self
                        .transient_pool
                        .compiled_view(plan.plan_id, accumulation)
                        .expect("compiled weighted accumulation view");
                    let revealage_view = self
                        .transient_pool
                        .compiled_view(plan.plan_id, revealage)
                        .expect("compiled weighted revealage view");
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("weighted_transparency_resolve_bind_group"),
                        layout: self
                            .weighted_transparency_resolve_layout
                            .as_ref()
                            .expect("weighted resolve layout initialized"),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(accumulation_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(revealage_view),
                            },
                        ],
                    });
                    self.weighted_transparency_resolve_bind_group = Some(bind_group);
                    self.weighted_transparency_resolve_bind_group_key = Some(key);
                }
            }

            let reactive_view = compiled_reactive.map(|(plan_id, reactive)| {
                self.transient_pool
                    .compiled_view(plan_id, reactive)
                    .expect("compiled temporal reactive view")
            });
            let mut reactive_initialized = false;

            if has_imported_refraction {
                profiler.begin("refractive_pass");
                if let Some(reactive_view) = reactive_view {
                    let refractive_ts = profiler.pass_timestamp_writes("refractive_pass");
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("bloom_imported_refractive_reactive_pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.hdr_rt_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.velocity_rt_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: reactive_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: refractive_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    self.draw_imported_refraction(&mut pass, scene, true);
                    reactive_initialized = true;
                } else {
                    let refractive_ts = profiler.pass_timestamp_writes("refractive_pass");
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("bloom_imported_refractive_pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.hdr_rt_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.velocity_rt_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: refractive_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    self.draw_imported_refraction(&mut pass, scene, false);
                }
                profiler.end("refractive_pass");
            }

            if self.weighted_transparency_active {
                profiler.begin("weighted_transparency_pass");
                let plan = self
                    .last_frame_plan
                    .as_ref()
                    .expect("weighted transparency has an active frame plan");
                let accumulation = plan
                    .resource("transparency-accumulation")
                    .expect("weighted topology declares accumulation")
                    .id;
                let revealage = plan
                    .resource("transparency-revealage")
                    .expect("weighted topology declares revealage")
                    .id;
                let accumulation_view = self
                    .transient_pool
                    .compiled_view(plan.plan_id, accumulation)
                    .expect("compiled weighted accumulation view");
                let revealage_view = self
                    .transient_pool
                    .compiled_view(plan.plan_id, revealage)
                    .expect("compiled weighted revealage view");
                {
                    let weighted_ts = profiler.pass_timestamp_writes("weighted_transparency_pass");
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("bloom_weighted_transparency_accumulation_pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: accumulation_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: revealage_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: weighted_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    self.draw_imported_transparency(&mut pass, scene, true, false);
                }
                if let Some(reactive_view) = reactive_view {
                    let reactive_load = if reactive_initialized {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    };
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("bloom_weighted_transparency_reactive_resolve_pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.hdr_rt_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: reactive_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: reactive_load,
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(
                        self.weighted_transparency_reactive_resolve_pipeline
                            .as_ref()
                            .expect("weighted reactive resolve pipeline initialized"),
                    );
                    pass.set_bind_group(
                        0,
                        self.weighted_transparency_resolve_bind_group
                            .as_ref()
                            .expect("weighted resolve bind group initialized"),
                        &[],
                    );
                    pass.draw(0..3, 0..1);
                    reactive_initialized = true;
                } else {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("bloom_weighted_transparency_resolve_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.hdr_rt_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(
                        self.weighted_transparency_resolve_pipeline
                            .as_ref()
                            .expect("weighted resolve pipeline initialized"),
                    );
                    pass.set_bind_group(
                        0,
                        self.weighted_transparency_resolve_bind_group
                            .as_ref()
                            .expect("weighted resolve bind group initialized"),
                        &[],
                    );
                    pass.draw(0..3, 0..1);
                }
                profiler.end("weighted_transparency_pass");
            }

            let has_conventional_translucency =
                !self.material_system.translucent_commands.is_empty()
                    || has_sorted_imported_transparency;
            if has_conventional_translucency {
                if self.temporal_reactive_active {
                    if has_sorted_imported_transparency {
                        let reactive_view =
                            reactive_view.expect("active reactive topology has a view");
                        let reactive_load = if reactive_initialized {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        };
                        let t_ts = profiler.pass_timestamp_writes("translucent_pass");
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_imported_transparency_reactive_pass"),
                            color_attachments: &[
                                Some(wgpu::RenderPassColorAttachment {
                                    view: &self.hdr_rt_view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                }),
                                Some(wgpu::RenderPassColorAttachment {
                                    view: reactive_view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: reactive_load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                }),
                            ],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &self.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: t_ts,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        if globally_interleave_sorted {
                            self.draw_sorted_transparency(&mut pass, scene, true);
                        } else {
                            self.draw_imported_transparency(&mut pass, scene, false, true);
                        }
                        reactive_initialized = true;
                    }

                    if !globally_interleave_sorted
                        && !self.material_system.translucent_commands.is_empty()
                    {
                        let t_ts = profiler.pass_timestamp_writes("translucent_pass");
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_custom_translucent_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.hdr_rt_view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &self.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: t_ts,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        let cache = &self.model_gpu_cache;
                        let gpu_driven = &self.gpu_driven;
                        self.material_system
                            .dispatch_translucent(&mut pass, |handle, idx| {
                                if let Some(Some(meshes)) = cache.get(&handle) {
                                    if idx < meshes.len() {
                                        let mesh = &meshes[idx];
                                        return Some(
                                            gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count),
                                        );
                                    }
                                }
                                None
                            });
                    }
                } else {
                    let t_ts = profiler.pass_timestamp_writes("translucent_pass");
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("bloom_translucent_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.hdr_rt_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                // Translucents don't write depth — keep
                                // the opaque pass's depth pristine so
                                // downstream post-FX (SSR/SSGI) still
                                // sees the opaque geometry.
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: t_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    if has_sorted_imported_transparency {
                        if globally_interleave_sorted {
                            self.draw_sorted_transparency(&mut pass, scene, false);
                        } else {
                            self.draw_imported_transparency(&mut pass, scene, false, false);
                            if !self.material_system.translucent_commands.is_empty() {
                                let cache = &self.model_gpu_cache;
                                let gpu_driven = &self.gpu_driven;
                                self.material_system.dispatch_translucent(
                                    &mut pass,
                                    |handle, idx| {
                                        if let Some(Some(meshes)) = cache.get(&handle) {
                                            if idx < meshes.len() {
                                                let mesh = &meshes[idx];
                                                return Some(
                                                    gpu_driven.mesh_draw(
                                                        &mesh.geometry,
                                                        mesh.index_count,
                                                    ),
                                                );
                                            }
                                        }
                                        None
                                    },
                                );
                            }
                        }
                    } else {
                        let cache = &self.model_gpu_cache;
                        let gpu_driven = &self.gpu_driven;
                        self.material_system
                            .dispatch_translucent(&mut pass, |handle, idx| {
                                if let Some(Some(meshes)) = cache.get(&handle) {
                                    if idx < meshes.len() {
                                        let mesh = &meshes[idx];
                                        return Some(
                                            gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count),
                                        );
                                    }
                                }
                                None
                            });
                    }
                }
            }

            debug_assert!(
                !self.temporal_reactive_active || reactive_initialized,
                "active temporal coverage must be initialized by a contributing imported pass"
            );
            profiler.end("translucent_pass");
        }
    }
}
