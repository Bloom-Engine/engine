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

    profiler.begin("main_hdr_pass");
    {
        // HDR clear: the user's clear_color is in 0-1 srgb-ish
        // range; treat it as the linear background for the HDR
        // RT. After tonemap it ends up roughly the same shade.
        let hdr_ts = profiler.pass_timestamp_writes("main_hdr_pass");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom_hdr_pass"),
            color_attachments: &[
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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
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
        if self.procedural_sky_enabled {
            self.render_procedural_sky_pass(&mut pass, self.lighting_uniforms.camera_pos[3]);
        } else {
            self.render_sky_pass(&mut pass, self.lighting_uniforms.camera_pos[3]);
        }

        if has_3d {
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
                    if count == 0 { continue; }
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
            pass.set_pipeline(&self.scene_pipeline);
            pass.set_bind_group(1, &self.lighting_bind_group, &[]);
            pass.set_bind_group(3, &self.joint_bind_group, &[]);

            if has_cached_models {
                for cmd in &self.model_draw_commands {
                    if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                        if cmd.mesh_idx < meshes.len() {
                            let mesh = &meshes[cmd.mesh_idx];
                            pass.set_bind_group(0, &self.model_uniform_bind_groups[cmd.uniform_slot], &[]);
                            pass.set_bind_group(2, &mesh.material_bg, &[]);
                            pass.set_vertex_buffer(0, mesh.vb.slice(..));
                            pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                }
            }

            scene.render(&mut pass);
        }
    }
    profiler.end("main_hdr_pass");

    // EN-011 — render every registered planar reflection probe
    // BEFORE the main material pass so the probe RTs are
    // sampleable when materials run. No-op when no probes are
    // registered or no opaque material draws are queued.
    profiler.begin("planar_reflections");
    self.dispatch_planar_reflections(&mut *encoder, scene);
    profiler.end("planar_reflections");

    // Phase 2c — schedule the material pass through the render
    // graph. First real consumer of `renderer::graph` from #35.
    // For now a one-node graph; later phases add more nodes
    // (main_hdr, ssao, bloom, translucent, composite) and the
    // graph's topological sort picks the order from read/write
    // declarations.
    //
    // All per-frame borrows that the pass body needs are captured
    // here from `&self` before we build the context that wraps
    // `&mut *encoder` + `&mut profiler`. Rust's borrow checker is
    // happy because the immutable and mutable borrows are
    // disjoint fields of the same struct.
    if !self.material_system.commands.is_empty() {
        use graph::{Graph, PassNode, PassOutput};

        let hdr_rt_view       = &self.hdr_rt_view;
        let material_rt_view  = &self.material_rt_view;
        let velocity_rt_view  = &self.velocity_rt_view;
        let albedo_rt_view    = &self.albedo_rt_view;
        let depth_view        = &self.depth_view;
        let material_system   = &self.material_system;
        let model_gpu_cache   = &self.model_gpu_cache;

        struct FrameCtx<'a> {
            encoder:  &'a mut wgpu::CommandEncoder,
            profiler: &'a mut crate::profiler::Profiler,
        }

        let mut graph: Graph<FrameCtx<'_>> = Graph::new();
        graph.push(
            PassNode::new("material_pass", Box::new(move |ctx: &mut FrameCtx| {
                ctx.profiler.begin("material_pass");
                {
                    let mat_ts = ctx.profiler.pass_timestamp_writes("material_pass");
                    let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("bloom_material_pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: hdr_rt_view,
                                resolve_target: None, depth_slice: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: material_rt_view,
                                resolve_target: None, depth_slice: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: velocity_rt_view,
                                resolve_target: None, depth_slice: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: albedo_rt_view,
                                resolve_target: None, depth_slice: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: mat_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    material_system.dispatch(&mut pass, |handle, idx| {
                        if let Some(Some(meshes)) = model_gpu_cache.get(&handle) {
                            if idx < meshes.len() {
                                let mesh = &meshes[idx];
                                return Some((&mesh.vb, &mesh.ib, mesh.index_count));
                            }
                        }
                        None
                    });
                }
                ctx.profiler.end("material_pass");
            }))
            // Writes HdrColor + the G-buffer so Phase 2d's scheduler
            // can order downstream passes (SSAO, bloom, translucent)
            // correctly once they're nodes too.
            .with_writes(&[
                PassOutput::HdrColor,
                PassOutput::MaterialRt,
                PassOutput::VelocityRt,
                PassOutput::AlbedoRt,
                PassOutput::Depth,
            ]),
        );

        let mut ctx = FrameCtx { encoder: &mut *encoder, profiler: &mut *profiler };
        if let Err(e) = graph.execute(&mut ctx) {
            eprintln!("[graph] material_pass failed: {:?}", e);
        }
    }

    }
}

impl Renderer {
    /// Translucent / refractive / additive material pass: after opaque,
    /// before post-FX; loads hdr_rt, depth read-only, back-to-front
    /// sorted; snapshots scene color for reads_scene materials. Split
    /// from end_frame_with_scene.
    pub(super) fn record_translucent_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
    ) {
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
    if !self.material_system.translucent_commands.is_empty() {
        // Back-to-front by view depth — required for correct alpha
        // compositing; submission order is only kept between
        // equal-depth draws (stable sort).
        self.material_system.sort_translucent();
        profiler.begin("translucent_pass");
        let swap_w = self.surface_config.width;
        let swap_h = self.surface_config.height;
        self.transient_pool.begin_frame(swap_w, swap_h);

        // Phase 7 — run the impulse decay + splat compute BEFORE
        // we build scene_inputs so the front view reflects this
        // frame's submissions.
        self.impulse_field.update(&self.device, &self.queue, &mut *encoder);

        // Does any queued translucent material need the scene
        // colour snapshot?
        let needs_scene = self.material_system.translucent_commands
            .iter()
            .any(|c| self.material_system.pipelines
                .get(c.material as usize - 1)
                .and_then(|p| p.as_ref())
                .map(|p| p.reads_scene)
                .unwrap_or(false));

        let scene_color_tid = if needs_scene {
            let desc = transient::TransientDesc::new(
                formats::HDR_FORMAT,
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                transient::SizePolicy::Swapchain,
            );
            Some(self.transient_pool.acquire(&self.device, desc))
        } else {
            None
        };

        // Phase 4c — depth snapshot. wgpu forbids sampling a
        // texture that is also a depth-stencil attachment of the
        // same pass, so we copy the opaque depth buffer into a
        // transient before beginning the translucent pass and
        // bind the transient at group 4 binding 2. Acquired
        // whenever any translucent material reads_scene (same
        // gate as colour) — cheap enough that it's not worth a
        // separate `reads_depth` flag yet.
        let scene_depth_tid = if needs_scene {
            let desc = transient::TransientDesc::new(
                formats::DEPTH_FORMAT,
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                transient::SizePolicy::Swapchain,
            );
            Some(self.transient_pool.acquire(&self.device, desc))
        } else {
            None
        };

        // Snapshot hdr_rt + live depth -> transients.
        if let (Some(ctid), Some(dtid)) = (scene_color_tid, scene_depth_tid) {
            let color_tex = self.transient_pool.texture(ctid).expect("fresh color transient");
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
                wgpu::Extent3d { width: swap_w, height: swap_h, depth_or_array_layers: 1 },
            );
            let depth_tex = self.transient_pool.texture(dtid).expect("fresh depth transient");
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
                wgpu::Extent3d { width: swap_w, height: swap_h, depth_or_array_layers: 1 },
            );
            let color_view = self.transient_pool.view(ctid).unwrap();
            let depth_view = self.transient_pool.view(dtid).unwrap();
            let imp_view = self.impulse_field.front_view();
            let imp_samp = self.impulse_field.sampler();
            self.material_system.update_scene_inputs(
                &self.device, color_view, Some(depth_view),
                Some((imp_view, imp_samp)),
            );
        } else {
            // No refractive/depth-reading materials this frame —
            // still need a valid bind group. None → internal stubs.
            self.material_system.update_scene_inputs(
                &self.device, &self.hdr_rt_view, None, None,
            );
        }

        {
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
            let cache = &self.model_gpu_cache;
            self.material_system.dispatch_translucent(&mut pass, |handle, idx| {
                if let Some(Some(meshes)) = cache.get(&handle) {
                    if idx < meshes.len() {
                        let mesh = &meshes[idx];
                        return Some((&mesh.vb, &mesh.ib, mesh.index_count));
                    }
                }
                None
            });
        }

        if let Some(tid) = scene_color_tid {
            self.transient_pool.release(tid);
        }
        profiler.end("translucent_pass");
    }
    }
}
