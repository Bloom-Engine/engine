//! Cascaded shadow-map pass: cascade fitting from the primary
//! directional light, the ticket-004 cache-hit skip, and the per-cascade
//! depth renders. Split from end_frame_with_scene (2000-line file policy
//! + render-graph migration prep).

use super::*;

impl Renderer {
    pub(super) fn record_shadow_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        scene: &mut crate::scene::SceneGraph,
    ) {
    // Shadow pass: render scene nodes from light's perspective into
    // cascaded shadow maps (3 cascades).
    //
    // Cache hit path (ticket 004): if no caster moved, the light
    // didn't move, and the freshly-computed cascade VPs match the
    // ones the cached depth textures were rendered with, we skip
    // the whole pass. The depth textures retain their content and
    // the main pass samples from them as if we had redrawn.
    profiler.begin("shadow_pass");
    if self.shadow_map.enabled {
        // Compute cascade VPs from the primary directional light and camera.
        let light_dir = [
            self.lighting_uniforms.light_dir[0],
            self.lighting_uniforms.light_dir[1],
            self.lighting_uniforms.light_dir[2],
        ];
        // Auto-fit: compute world-space AABB across every visible,
        // cast-shadow node so the ortho volume always covers the
        // scene regardless of what's loaded. No per-scene magic
        // numbers.
        let scene_bounds = scene.compute_shadow_bounds();
        self.shadow_map.compute_cascade_vps(
            light_dir,
            self.current_camera_pos,
            self.current_view_matrix,
            // Use the pre-jitter projection so the cascade VPs
            // stay byte-stable when the camera is actually
            // stationary (the shadow cache compares them exactly).
            self.current_proj_matrix_unjittered,
            0.5,   // near — start cascades slightly past the camera
            80.0,  // far — shadow coverage range
            scene_bounds,
        );

        // Re-upload lighting uniforms with cascade VPs and splits.
        // Always write these — even on a cache hit the cascade
        // split distances and view matrix track camera movement
        // (they drive per-pixel cascade selection in the main
        // shader), which is independent of shadow texture content.
        self.lighting_uniforms.shadow_cascade_vps = self.shadow_map.light_vps;
        self.lighting_uniforms.shadow_cascade_splits = [
            self.shadow_map.cascade_splits[0],
            self.shadow_map.cascade_splits[1],
            self.shadow_map.cascade_splits[2],
            // .w = mip-LOD bias for material textures. Bias finer
            // by log2(render_scale) — recovers -1.0 at 0.5 (one
            // mip finer to offset hardware's coarser selection
            // at half-res), ~-0.42 at 0.75, 0 at native.
            if self.render_scale < 0.999 { self.render_scale.log2() } else { 0.0 },
        ];
        self.lighting_uniforms.shadow_view_matrix = self.current_view_matrix;
        self.queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_uniforms),
        );
        // Shadow-flicker fix: the material system's PerView buffer was
        // uploaded before this fit ran and still carries LAST frame's
        // cascade VPs. Patch its shadow fields so material-path
        // receivers sample the depth maps with the same matrices the
        // maps are rendered with this frame. Keep .w at 0.0 — that slot
        // is the TSR mip-LOD bias, which the material path historically
        // never received (begin_mode_3d resets it before the material
        // PerView upload); delivering -1.0 here makes hardware mip
        // selection flip per-panel under TAA jitter (visible texture
        // detail popping).
        let mut splits = self.lighting_uniforms.shadow_cascade_splits;
        splits[3] = 0.0;
        self.material_system.refresh_shadow_uniforms(
            &self.queue,
            splits,
            self.lighting_uniforms.shadow_view_matrix,
            self.shadow_map.light_vps,
        );

        // Cache gate. Skip if nothing that affects shadow-map
        // content has changed since last render. Texel-snap +
        // radius quantization in `compute_cascade_vps` makes this
        // check exact: identical scenes + identical poses (within
        // one cascade texel) produce byte-identical light_vps.
        let scene_ver = scene.shadow_version;
        let vps_changed = self.shadow_map.rendered_light_vps
            .as_ref()
            .map(|cached| *cached != self.shadow_map.light_vps)
            .unwrap_or(true);
        let light_changed = self.shadow_map.rendered_light_dir
            .map(|cached| cached != light_dir)
            .unwrap_or(true);
        let should_render = self.shadow_map.always_fresh
            || self.shadow_map.dirty
            || vps_changed
            || light_changed
            || self.shadow_map.rendered_scene_version != scene_ver
            // Immediate-mode + cached-model + material-system draws aren't
            // tracked by the scene version, and they're re-submitted (and
            // usually move) every frame, so re-render the shadow map
            // whenever any are present.
            || !self.indices_3d.is_empty()
            || !self.model_draw_commands.is_empty()
            || !self.material_system.commands.is_empty();

        if should_render {
        // Build a shared caster list + buffer-ref vectors, then
        // filter per cascade against that cascade's ortho frustum.
        // A caster outside cascade N's frustum can't write pixels
        // into cascade N; near/far pancaking already covers
        // behind-camera casters via the cascade's own far plane.
        struct ShadowDrawEntry {
            vb_idx: usize,
            ib_idx: usize,
            index_count: u32,
            transform: [[f32; 4]; 4],
            wmin: [f32; 3],
            wmax: [f32; 3],
            // Index into `cutout_bgs` for an alpha-tested caster (cutout
            // foliage), or -1 for an opaque caster (plain depth pipeline).
            cutout_idx: i32,
            // True only for the immediate-mode batch, which may contain skinned
            // characters. Rendered with the skinning-aware shadow pipeline so
            // animated player/enemies cast a posed shadow instead of a rest
            // pose at the origin. (Mixed batch: non-skinned verts in it still
            // transform by the model matrix via the shader's weight branch.)
            skinned: bool,
        }
        let mut shadow_nodes: Vec<ShadowDrawEntry> = Vec::new();
        let mut shadow_vbs: Vec<&wgpu::Buffer> = Vec::new();
        let mut shadow_ibs: Vec<&wgpu::Buffer> = Vec::new();
        let mut cutout_bgs: Vec<&wgpu::BindGroup> = Vec::new();
        for (_handle, node) in scene.nodes.iter() {
            // gi_only proxies duplicate geometry that already casts through
            // the material-command path below — including them would
            // double-render every caster.
            if !node.visible || node.gi_only || !node.cast_shadow || node.indices.is_empty() {
                continue;
            }
            let Some(vb) = &node.gpu_vb else { continue };
            let Some(ib) = &node.gpu_ib else { continue };
            let vb_idx = shadow_vbs.len();
            shadow_vbs.push(vb);
            shadow_ibs.push(ib);
            // MASK-material nodes carry an alpha-test shadow bind group so
            // foliage casts dappled shadows (same as the cached-model path).
            let cutout_idx = match &node.gpu_shadow_cutout_bg {
                Some(bg) => { let i = cutout_bgs.len(); cutout_bgs.push(bg); i as i32 }
                None => -1,
            };
            shadow_nodes.push(ShadowDrawEntry {
                vb_idx,
                ib_idx: vb_idx,
                index_count: node.gpu_index_count,
                transform: node.transform,
                wmin: node.world_bounds_min,
                wmax: node.world_bounds_max,
                cutout_idx,
                skinned: false,
            });
        }

        // Immediate-mode 3D batch (drawCube/drawSphere/non-cached models).
        // These verts are already in WORLD space, so the model matrix is
        // identity. wmin > wmax marks "no bounds" → included in every cascade.
        // Games that draw in immediate mode create no scene nodes, so without
        // this nothing they draw would cast a shadow.
        if !self.indices_3d.is_empty() {
            let vb_idx = shadow_vbs.len();
            shadow_vbs.push(&self.persistent_vb_3d);
            shadow_ibs.push(&self.persistent_ib_3d);
            shadow_nodes.push(ShadowDrawEntry {
                vb_idx,
                ib_idx: vb_idx,
                index_count: self.indices_3d.len() as u32,
                transform: IDENTITY_MAT4,
                wmin: [1.0, 1.0, 1.0],
                wmax: [-1.0, -1.0, -1.0],
                cutout_idx: -1,
                skinned: true,
            });
        }

        // Cached models (drawModel: trees, characters, etc.) — each is a
        // GpuMesh plus its object→world matrix. Skinned models cast their
        // rest-pose shadow (vs_shadow doesn't skin) — acceptable.
        for cmd in self.model_draw_commands.iter() {
            if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                if cmd.mesh_idx < meshes.len() {
                    let mesh = &meshes[cmd.mesh_idx];
                    let vb_idx = shadow_vbs.len();
                    shadow_vbs.push(&mesh.vb);
                    shadow_ibs.push(&mesh.ib);
                    // Cutout foliage → alpha-tested shadow pipeline.
                    let cutout_idx = match &mesh.shadow_cutout_bg {
                        Some(bg) => { let i = cutout_bgs.len(); cutout_bgs.push(bg); i as i32 }
                        None => -1,
                    };
                    shadow_nodes.push(ShadowDrawEntry {
                        vb_idx,
                        ib_idx: vb_idx,
                        index_count: mesh.index_count,
                        transform: cmd.model,
                        wmin: [1.0, 1.0, 1.0],
                        wmax: [-1.0, -1.0, -1.0],
                        cutout_idx,
                        skinned: false,
                    });
                }
            }
        }

        // Material-system draws (terrain / building / trees rendered through
        // compiled materials). Same GpuMesh cache as drawModel — the command
        // carries a CPU-side copy of its model matrix precisely for this
        // pass. `commands` holds only the opaque + cutout buckets, so water /
        // glass / additive effects never cast. Instanced draws (the 20k-blade
        // grass field) are skipped deliberately: vs_shadow has no instance
        // stream, and per-blade grass shadows are sub-texel noise at these
        // cascade resolutions anyway.
        for cmd in self.material_system.commands.iter() {
            if cmd.instance.is_some() { continue; }
            if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.mesh_handle) {
                if cmd.mesh_idx < meshes.len() {
                    let mesh = &meshes[cmd.mesh_idx];
                    let vb_idx = shadow_vbs.len();
                    shadow_vbs.push(&mesh.vb);
                    shadow_ibs.push(&mesh.ib);
                    // MASK-material meshes (leaf cards) keep their dappled
                    // alpha-tested shadows, same as the other two paths.
                    let cutout_idx = match &mesh.shadow_cutout_bg {
                        Some(bg) => { let i = cutout_bgs.len(); cutout_bgs.push(bg); i as i32 }
                        None => -1,
                    };
                    shadow_nodes.push(ShadowDrawEntry {
                        vb_idx,
                        ib_idx: vb_idx,
                        index_count: mesh.index_count,
                        transform: cmd.model,
                        wmin: [1.0, 1.0, 1.0],
                        wmax: [-1.0, -1.0, -1.0],
                        cutout_idx,
                        skinned: false,
                    });
                }
            }
        }

        let cascade_planes: [[[f32; 4]; 6]; crate::shadows::NUM_CASCADES] =
            std::array::from_fn(|c| {
                crate::scene::extract_frustum_planes(&self.shadow_map.light_vps[c])
            });
        let mut cascade_indices: [Vec<usize>; crate::shadows::NUM_CASCADES] =
            std::array::from_fn(|_| Vec::with_capacity(shadow_nodes.len()));
        for (i, entry) in shadow_nodes.iter().enumerate() {
            let has_bounds = entry.wmin[0] <= entry.wmax[0];
            for c in 0..crate::shadows::NUM_CASCADES {
                if has_bounds
                    && crate::scene::aabb_outside_frustum(&cascade_planes[c], entry.wmin, entry.wmax)
                {
                    continue;
                }
                cascade_indices[c].push(i);
            }
        }

        // Render each cascade
        for cascade in 0..crate::shadows::NUM_CASCADES {
            let stride = crate::shadows::SHADOW_UNIFORM_STRIDE as usize;
            let max = crate::shadows::SHADOW_MAX_NODES as usize;
            let entries = &cascade_indices[cascade];
            let count = entries.len().min(max);
            let mut uniform_data: Vec<u8> = vec![0u8; stride * count.max(1)];
            let cascade_vp = self.shadow_map.light_vps[cascade];

            for (slot, &ei) in entries.iter().take(count).enumerate() {
                let entry = &shadow_nodes[ei];
                let uniforms = crate::shadows::ShadowUniforms {
                    light_vp: cascade_vp,
                    model: entry.transform,
                };
                let off = slot * stride;
                uniform_data[off..off + std::mem::size_of::<crate::shadows::ShadowUniforms>()]
                    .copy_from_slice(bytemuck::bytes_of(&uniforms));
            }

            if count > 0 {
                self.queue.write_buffer(
                    &self.shadow_map.uniform_buffer,
                    0,
                    &uniform_data[..count * stride],
                );
            }

            {
                let shadow_ts = profiler.pass_timestamp_writes("shadow_pass");
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_map.depth_views[cascade],
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: shadow_ts,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                // Pipeline kind per caster: 0 opaque, 1 cutout, 2 skinned.
                // Track the bound kind so we only switch when it changes
                // (cutout/skinned casters are grouped at the tail, so this is
                // usually one or two switches).
                let mut cur_kind: u8 = 0;
                shadow_pass.set_pipeline(&self.shadow_map.pipeline);

                for (slot, &ei) in entries.iter().take(count).enumerate() {
                    let entry = &shadow_nodes[ei];
                    let offset = (slot * stride) as u32;
                    let kind: u8 = if entry.skinned { 2 }
                        else if entry.cutout_idx >= 0 { 1 }
                        else { 0 };
                    if kind != cur_kind {
                        shadow_pass.set_pipeline(match kind {
                            1 => &self.shadow_map.pipeline_cutout,
                            2 => &self.shadow_map.pipeline_skinned,
                            _ => &self.shadow_map.pipeline,
                        });
                        cur_kind = kind;
                    }
                    shadow_pass.set_bind_group(0, &self.shadow_map.uniform_bind_group, &[offset]);
                    if kind == 1 {
                        shadow_pass.set_bind_group(1, cutout_bgs[entry.cutout_idx as usize], &[]);
                    } else if kind == 2 {
                        // Joint matrices for skinning the animated characters in
                        // the immediate-mode batch.
                        shadow_pass.set_bind_group(1, &self.joint_bind_group, &[]);
                    }
                    shadow_pass.set_vertex_buffer(0, shadow_vbs[entry.vb_idx].slice(..));
                    shadow_pass.set_index_buffer(shadow_ibs[entry.ib_idx].slice(..), wgpu::IndexFormat::Uint32);
                    shadow_pass.draw_indexed(0..entry.index_count, 0, 0..1);
                }
            }
        }

        // Cache bookkeeping — next frame will short-circuit if the
        // camera, scene, and light all stay put.
        self.shadow_map.rendered_light_vps = Some(self.shadow_map.light_vps);
        self.shadow_map.rendered_light_dir = Some(light_dir);
        self.shadow_map.rendered_scene_version = scene_ver;
        self.shadow_map.dirty = false;
        } // end should_render
    }

    profiler.end("shadow_pass");
    }
}
