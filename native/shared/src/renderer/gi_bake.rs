//! GI bake methods — scene-wide SDF clipmap (binned + sliced amortized
//! bake), WSRC radiance cascades, and per-mesh SDF drain. Split out of
//! renderer/mod.rs to keep it under the file-line ceiling (see
//! tools/check-file-lines.js); pure code move, no behaviour change.

use super::*;

impl Renderer {
    /// Ticket 014 V2 — bake the scene-wide SDF clipmap once, on the
    /// frame when all per-mesh queues (BLAS, cards, per-mesh SDFs)
    /// have drained. Gathers every visible mesh's triangles into a
    /// world-space buffer via `scene.build_world_triangles()` and
    /// runs `SDF_BAKE_WGSL` against the unified data with the
    /// clipmap's fixed world-space AABB. 64³ voxel × scene triangle
    /// count = expensive one-shot (~100-200 ms on Sponza), but
    /// happens after a visible frame and never repeats for static
    /// scenes.
    /// Ticket 014 V5 — camera world-space position. Uses
    /// `current_camera_pos`, which `begin_mode_3d` writes every frame
    /// from the user-supplied camera position (cheaper than inverting
    /// the view matrix and always in sync with what the game sees).
    pub(super) fn current_camera_world_pos(&self) -> [f32; 3] {
        self.current_camera_pos
    }

    /// Ticket 014 V5 — flag the SDF clipmap for a re-bake if the camera
    /// has moved past the rebake threshold from the current clipmap
    /// centre. Fullscreen-lag fix: instead of clearing `built` (which
    /// used to fire a full-volume single-dispatch rebake that stalled
    /// weak GPUs for seconds), this only raises `rebake_needed`; the
    /// live clipmap keeps serving traces while the amortized job bakes
    /// the re-centred volume a few Z-slices per frame.
    pub(super) fn maybe_invalidate_sdf_clipmap(&mut self) {
        // A job in flight already re-centres on its own origin — let it
        // land before measuring drift again.
        if !self.scene_sdf_clipmap_built || self.sdf_clipmap_job.is_some() {
            return;
        }
        let cam = self.current_camera_world_pos();
        let dx = cam[0] - self.scene_sdf_clipmap_origin[0];
        let dy = cam[1] - self.scene_sdf_clipmap_origin[1];
        let dz = cam[2] - self.scene_sdf_clipmap_origin[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let threshold = SCENE_SDF_CLIPMAP_EXTENT * SCENE_SDF_CLIPMAP_REBAKE_THRESHOLD;
        if dist_sq > threshold * threshold {
            self.scene_sdf_clipmap_rebake_needed = true;
        }
    }

    pub(super) fn bake_scene_sdf_clipmap(
        &mut self,
        scene: &crate::scene::SceneGraph,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Continue an in-flight job first: one slice batch per frame.
        if let Some(job) = self.sdf_clipmap_job.take() {
            self.encode_clipmap_bake_slices(job, encoder);
            return;
        }
        if !self.scene_sdf_clipmap_rebake_needed {
            return;
        }
        // Wait for all per-mesh queues to drain — builds the clipmap
        // from a fully-loaded scene rather than a partial one, and
        // keeps first-frame cost spread across the card/BLAS work
        // already scheduled.
        if !scene.pending_blas_builds.is_empty()
            || !scene.pending_card_captures.is_empty()
            || !scene.pending_sdf_bakes.is_empty()
        {
            return;
        }

        let (vertices, indices, tri_count) = scene.build_world_triangles();
        if tri_count == 0 {
            return;
        }

        // V5 — centre the clipmap on the current camera position,
        // voxel-snapped for sampling stability (sub-voxel shifts
        // would change which voxel each sphere-trace step reads).
        // The live origin flips only when the job completes.
        let half = SCENE_SDF_CLIPMAP_EXTENT * 0.5;
        let voxel = SCENE_SDF_CLIPMAP_EXTENT / SCENE_SDF_CLIPMAP_RES as f32;
        let cam = self.current_camera_world_pos();
        let origin = [
            (cam[0] / voxel).round() * voxel,
            (cam[1] / voxel).round() * voxel,
            (cam[2] / voxel).round() * voxel,
        ];
        let aabb_min = [origin[0] - half, origin[1] - half, origin[2] - half, 0.0];
        let aabb_max = [origin[0] + half, origin[1] + half, origin[2] + half, 0.0];

        // Bin triangles into BIN_CELLS³ cells, each list expanded by one
        // cell (the shader's narrow band) so the per-cell clamp stays a
        // conservative lower bound for sphere tracing. Two-pass counting
        // sort: count, prefix-sum, fill.
        let cells = SCENE_SDF_CLIPMAP_BIN_CELLS as usize;
        let cell_size = SCENE_SDF_CLIPMAP_EXTENT / cells as f32;
        let grid_min = [aabb_min[0], aabb_min[1], aabb_min[2]];
        let cell_range = |tri: usize| -> Option<([usize; 3], [usize; 3])> {
            let i0 = indices[tri * 3] as usize * 12;
            let i1 = indices[tri * 3 + 1] as usize * 12;
            let i2 = indices[tri * 3 + 2] as usize * 12;
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for base in [i0, i1, i2] {
                for a in 0..3 {
                    let v = vertices[base + a];
                    lo[a] = lo[a].min(v);
                    hi[a] = hi[a].max(v);
                }
            }
            let mut c_lo = [0usize; 3];
            let mut c_hi = [0usize; 3];
            for a in 0..3 {
                // Expand by one cell width (= the shader's band).
                let lo_c = ((lo[a] - cell_size - grid_min[a]) / cell_size).floor();
                let hi_c = ((hi[a] + cell_size - grid_min[a]) / cell_size).floor();
                if hi_c < 0.0 || lo_c >= cells as f32 {
                    return None; // entirely outside the clipmap volume
                }
                c_lo[a] = lo_c.max(0.0) as usize;
                c_hi[a] = hi_c.min(cells as f32 - 1.0) as usize;
            }
            Some((c_lo, c_hi))
        };
        let cell_count = cells * cells * cells;
        let mut counts = vec![0u32; cell_count];
        for t in 0..tri_count as usize {
            if let Some((lo, hi)) = cell_range(t) {
                for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            counts[(z * cells + y) * cells + x] += 1;
                        }
                    }
                }
            }
        }
        let mut offsets = vec![0u32; cell_count + 1];
        for i in 0..cell_count {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        let total_refs = offsets[cell_count] as usize;
        let mut cursor: Vec<u32> = offsets[..cell_count].to_vec();
        // wgpu rejects zero-sized buffers — keep one dummy entry when no
        // triangle touches the volume (all cells then read empty ranges).
        let mut tri_refs = vec![0u32; total_refs.max(1)];
        for t in 0..tri_count as usize {
            if let Some((lo, hi)) = cell_range(t) {
                for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            let ci = (z * cells + y) * cells + x;
                            tri_refs[cursor[ci] as usize] = t as u32;
                            cursor[ci] += 1;
                        }
                    }
                }
            }
        }

        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_sdf_bake_vbuf"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_sdf_bake_ibuf"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cell_offsets_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_sdf_bake_cell_offsets"),
            contents: bytemuck::cast_slice(&offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cell_tris_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_sdf_bake_cell_tris"),
            contents: bytemuck::cast_slice(&tri_refs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Per-job uniform: sharing sdf_bake_uniform would alias with the
        // per-mesh bakes — queue.write_buffer applies before any of this
        // frame's commands, so the last write would win for every pass.
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_sdf_clipmap_bake_uniform"),
            size: std::mem::size_of::<SdfBakeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_sdf_clipmap_bake_bg"),
            layout: &self.sdf_clipmap_bake_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ibuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.scene_sdf_clipmap_staging_view) },
                wgpu::BindGroupEntry { binding: 4, resource: cell_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_tris_buf.as_entire_binding() },
            ],
        });

        self.scene_sdf_clipmap_rebake_needed = false;
        let job = SdfClipmapBakeJob {
            origin,
            aabb_min,
            aabb_max,
            uniform,
            bind_group,
            next_z: 0,
        };
        // Encode the first slice batch right away so a full rebake takes
        // exactly RES / LAYERS_PER_FRAME frames end to end.
        self.encode_clipmap_bake_slices(job, encoder);
    }

    /// Fullscreen-lag fix — encode this frame's slice batch of the
    /// in-flight clipmap bake. On the final batch, copy the staging
    /// volume over the live clipmap and flip the origin — the copy is
    /// encoded before this frame's probe traces, so the swap is atomic
    /// from the tracer's point of view.
    pub(super) fn encode_clipmap_bake_slices(
        &mut self,
        mut job: SdfClipmapBakeJob,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let res = SCENE_SDF_CLIPMAP_RES;
        let layers = SCENE_SDF_CLIPMAP_LAYERS_PER_FRAME.min(res - job.next_z);
        let params = SdfBakeParams {
            aabb_min: job.aabb_min,
            aabb_max: job.aabb_max,
            counts: [SCENE_SDF_CLIPMAP_BIN_CELLS, res, job.next_z, 0],
        };
        self.queue.write_buffer(&job.uniform, 0, bytemuck::bytes_of(&params));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scene_sdf_clipmap_bake_slice"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sdf_clipmap_bake_pipeline);
        pass.set_bind_group(0, &job.bind_group, &[]);
        pass.dispatch_workgroups(res / 4, res / 4, layers / 4);
        drop(pass);

        job.next_z += layers;
        if job.next_z >= res {
            encoder.copy_texture_to_texture(
                self.scene_sdf_clipmap_staging_tex.as_image_copy(),
                self.scene_sdf_clipmap_tex.as_image_copy(),
                wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: res,
                },
            );
            self.scene_sdf_clipmap_origin = job.origin;
            self.scene_sdf_clipmap_built = true;
        } else {
            self.sdf_clipmap_job = Some(job);
        }
    }

    /// Ticket 014 V6/V7/V12/V13 — invalidate WSRC cascades on
    /// camera travel OR meaningful lighting change. V13 runs the
    /// V12 hysteresis checks per cascade, so a sun rotation or
    /// camera shift only rebakes the affected cascade(s). Typical
    /// pattern: camera moves 10 m → near cascade (1.875 m cell,
    /// ~0.47 m threshold) rebakes every few frames, mid cascade
    /// (7.5 m cell, ~1.9 m threshold) rebakes occasionally, far
    /// cascade (31 m cell, ~7.8 m threshold) stays cached for
    /// much longer.
    pub(super) fn maybe_invalidate_wsrc(&mut self) {
        let cam = self.current_camera_world_pos();
        let ld = self.lighting_uniforms.light_dir;
        let lc = self.lighting_uniforms.light_color;
        let amb = self.lighting_uniforms.ambient;
        let cur_sun_color = [lc[0] * ld[3], lc[1] * ld[3], lc[2] * ld[3]];
        let cur_sky_color = [amb[0] * amb[3], amb[1] * amb[3], amb[2] * amb[3]];

        fn luma(c: [f32; 3]) -> f32 {
            c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
        }
        fn rel_diff(a: f32, b: f32) -> f32 {
            (a - b).abs() / a.max(b).max(1e-4)
        }

        for c in 0..WSRC_CASCADE_COUNT as usize {
            if !self.wsrc_built[c] {
                continue;
            }
            // Camera travel — per-cascade threshold scales with the
            // cascade's extent, so each cascade has its own
            // "moved enough" metric.
            let extent = WSRC_CASCADE_EXTENTS[c];
            let origin = self.wsrc_origin[c];
            let dx = cam[0] - origin[0];
            let dy = cam[1] - origin[1];
            let dz = cam[2] - origin[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let threshold = extent * WSRC_REBAKE_THRESHOLD;
            if dist_sq > threshold * threshold {
                self.wsrc_built[c] = false;
                continue;
            }

            // V12 hysteresis — angular sun + 5% relative luma.
            let last = self.wsrc_last_sun_dir[c];
            let sun_dot = ld[0] * last[0] + ld[1] * last[1] + ld[2] * last[2];
            if sun_dot < 0.99985 {
                self.wsrc_built[c] = false;
                continue;
            }
            if rel_diff(luma(cur_sun_color), luma(self.wsrc_last_sun_color[c])) > 0.05 {
                self.wsrc_built[c] = false;
                continue;
            }
            if rel_diff(luma(cur_sky_color), luma(self.wsrc_last_sky_color[c])) > 0.05 {
                self.wsrc_built[c] = false;
            }
        }
    }

    /// Ticket 014 V6 — bake the world-space radiance cache. One
    /// dispatch covers all `WSRC_GRID_RES³` probes × 64 octel texels.
    /// Cheap: per-texel work is one shadow-cascade lookup + analytic
    /// sun/sky math, roughly matching a single card-lighting pixel.
    /// Runs at most once per `WSRC_REBAKE_THRESHOLD × extent` of
    /// camera travel — same amortisation pattern as the clipmap.
    pub(super) fn bake_wsrc(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // V13 — bake only cascades that are marked not-built. Each
        // cascade snaps to its own cell grid (cell = extent / 16)
        // and writes into its own 16-slice block of the shared
        // atlas.
        if self.wsrc_built.iter().all(|b| *b) {
            return;
        }

        // V14 — pick the HW ray-traced bake when the adapter has
        // ray-query AND the TLAS is ready. The SW path stays the
        // fallback for non-RT adapters and for the early frames
        // before BLAS / TLAS have been built.
        let use_hw = self.hw_rt_enabled
            && self.wsrc_bake_hw_pipeline.is_some()
            && self.tlas.is_some()
            && self.tlas_instance_data_buffer.is_some();

        // Resolve a single set of lighting params — they're the same
        // across all cascades in one frame. Per-cascade differences
        // come from the origin + extent passed through the uniform.
        let ld = self.lighting_uniforms.light_dir;
        let inv_len = 1.0 / (ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]).sqrt().max(1e-4);
        let sun_dir_ws = [-ld[0]*inv_len, -ld[1]*inv_len, -ld[2]*inv_len, ld[3]];
        let lc = self.lighting_uniforms.light_color;
        let sun_intensity = ld[3].max(0.0);
        let sun_color = [
            lc[0] * sun_intensity,
            lc[1] * sun_intensity,
            lc[2] * sun_intensity,
            0.0,
        ];
        let amb = self.lighting_uniforms.ambient;
        let sky_intensity = amb[3].max(0.0);
        let sky_color = [
            amb[0] * sky_intensity,
            amb[1] * sky_intensity,
            amb[2] * sky_intensity,
            0.0,
        ];

        let shadows_enabled = self.shadow_map.enabled;
        let shadow_vps: [[[f32; 4]; 4]; 3] = if shadows_enabled {
            self.shadow_map.light_vps
        } else {
            [IDENTITY_MAT4; 3]
        };
        let shadow_splits = if shadows_enabled {
            let s = self.shadow_map.cascade_splits;
            [s[0], s[1], s[2], 0.0]
        } else {
            [f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.0]
        };

        // Lazy-build whichever bind group the selected path needs.
        // The two caches are independent — switching between paths
        // (e.g. if TLAS becomes available mid-session) is fine.
        if use_hw {
            if self.wsrc_bake_hw_bg_cache.is_none() {
                let tlas = self.tlas.as_ref().unwrap();
                let instance_buf = self.tlas_instance_data_buffer.as_ref().unwrap();
                self.wsrc_bake_hw_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wsrc_bake_hw_bg"),
                    layout: self.wsrc_bake_hw_layout.as_ref().unwrap(),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.wsrc_bake_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                        wgpu::BindGroupEntry { binding: 6, resource: tlas.as_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: instance_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.mesh_card_radiance_view) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                    ],
                }));
            }
        } else if self.wsrc_bake_bg_cache.is_none() {
            self.wsrc_bake_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wsrc_bake_bg"),
                layout: &self.wsrc_bake_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.wsrc_bake_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                ],
            }));
        }

        let cam = self.current_camera_world_pos();

        // At most ONE cascade per frame. Besides amortizing the cost,
        // this fixes a params-aliasing bug: all cascades share
        // `wsrc_bake_uniform`, and `queue.write_buffer` applies every
        // write before any of this frame's dispatches execute — baking
        // two cascades in one frame made both dispatches read the last
        // cascade's params (wrong extent + wrong atlas slice flag).
        let mut baked_one = false;
        for c in 0..WSRC_CASCADE_COUNT as usize {
            if self.wsrc_built[c] || baked_one {
                continue;
            }
            let extent = WSRC_CASCADE_EXTENTS[c];
            let cell = extent / WSRC_GRID_RES as f32;
            let origin = [
                (cam[0] / cell).round() * cell,
                (cam[1] / cell).round() * cell,
                (cam[2] / cell).round() * cell,
            ];
            self.wsrc_origin[c] = origin;

            let params = WsrcBakeParams {
                sun_dir: sun_dir_ws,
                sun_color,
                sky_color,
                grid: [origin[0], origin[1], origin[2], extent],
                shadow_vps,
                shadow_splits,
                flags: [
                    0.002,
                    if shadows_enabled { 1.0 } else { 0.0 },
                    c as f32,
                    0.0,
                ],
                ground_albedo: [
                    self.gi_scene_avg_albedo[0],
                    self.gi_scene_avg_albedo[1],
                    self.gi_scene_avg_albedo[2],
                    0.0,
                ],
            };
            self.queue.write_buffer(
                &self.wsrc_bake_uniform,
                0,
                bytemuck::bytes_of(&params),
            );

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(if use_hw { "wsrc_bake_hw_pass" } else { "wsrc_bake_pass" }),
                timestamp_writes: None,
            });
            if use_hw {
                pass.set_pipeline(self.wsrc_bake_hw_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, self.wsrc_bake_hw_bg_cache.as_ref().unwrap(), &[]);
            } else {
                pass.set_pipeline(&self.wsrc_bake_pipeline);
                pass.set_bind_group(0, self.wsrc_bake_bg_cache.as_ref().unwrap(), &[]);
            }
            // One workgroup per probe in this cascade (16³),
            // 10×10 threads per workgroup (padded octel).
            pass.dispatch_workgroups(WSRC_GRID_RES, WSRC_GRID_RES, WSRC_GRID_RES);
            drop(pass);

            self.wsrc_built[c] = true;
            self.wsrc_last_sun_dir[c] = ld;
            self.wsrc_last_sun_color[c] = [sun_color[0], sun_color[1], sun_color[2]];
            self.wsrc_last_sky_color[c] = [sky_color[0], sky_color[1], sky_color[2]];
            baked_one = true;
        }
    }

    /// Ticket 014 V1 — bake per-mesh unsigned distance fields via
    /// the compute pipeline. Drains `scene.pending_sdf_bakes` with a
    /// per-frame budget; expensive workload (O(voxels × triangles)
    /// per mesh), so the rate-limit keeps first-frame stutter
    /// bounded. Static scenes amortise and never re-bake.
    ///
    /// Ticket 022 — after each dispatch, encode a copy_texture_to_buffer
    /// against a fresh staging buffer and stash (hash, buffer) on
    /// `sdf_cache_writes`. The frame's main submit picks up the copies
    /// alongside the bake; `flush_sdf_cache_writes` then maps and
    /// persists each buffer to the on-disk cache so the next launch
    /// hits the load path in scene.rs and skips the bake entirely.
    pub(super) fn bake_pending_sdfs(
        &mut self,
        scene: &mut crate::scene::SceneGraph,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if !self.hw_rt_enabled || scene.pending_sdf_bakes.is_empty() {
            return;
        }
        const SDF_BAKE_MAX_PER_FRAME: usize = 8;
        let take = scene.pending_sdf_bakes.len().min(SDF_BAKE_MAX_PER_FRAME);
        let pending: Vec<f64> = scene.pending_sdf_bakes.drain(..take).collect();

        for handle in pending {
            let (sdf_tex, sdf_view, vb_ptr, ib_ptr, bmin, bmax, index_count, mesh_hash) = {
                let Some(node) = scene.nodes.get(handle) else { continue; };
                let Some(sdf_tex) = node.mesh_sdf.as_ref() else { continue; };
                let Some(sdf_view) = node.mesh_sdf_view.as_ref() else { continue; };
                let Some(vb) = node.gpu_vb.as_ref() else { continue; };
                let Some(ib) = node.gpu_ib.as_ref() else { continue; };
                (
                    sdf_tex.clone(),
                    sdf_view.clone(),
                    vb.clone(),
                    ib.clone(),
                    node.bounds_min,
                    node.bounds_max,
                    node.gpu_index_count,
                    node.mesh_hash,
                )
            };
            if index_count == 0 {
                continue;
            }
            let tri_count = index_count / 3;
            let params = SdfBakeParams {
                aabb_min: [bmin[0], bmin[1], bmin[2], 0.0],
                aabb_max: [bmax[0], bmax[1], bmax[2], 0.0],
                counts: [tri_count, MESH_SDF_RES, 0, 0],
            };
            self.queue.write_buffer(
                &self.sdf_bake_uniform,
                0,
                bytemuck::bytes_of(&params),
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sdf_bake_bg"),
                layout: &self.sdf_bake_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.sdf_bake_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: vb_ptr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ib_ptr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sdf_view) },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sdf_bake_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sdf_bake_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(MESH_SDF_RES / 4, MESH_SDF_RES / 4, MESH_SDF_RES / 4);
            }

            // Ticket 022 — schedule a readback against the freshly-baked
            // texture so the next launch can skip the bake. We only do
            // this when scene.rs computed a hash (it always does, but
            // skip defensively); padded staging size is bound by
            // wgpu's COPY_BYTES_PER_ROW alignment.
            if let Some(hash) = mesh_hash {
                let row_padded = ((MESH_SDF_RES * 4 + 255) & !255) as u64;
                let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sdf_cache_readback"),
                    size: row_padded * (MESH_SDF_RES as u64) * (MESH_SDF_RES as u64),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &sdf_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(row_padded as u32),
                            rows_per_image: Some(MESH_SDF_RES),
                        },
                    },
                    wgpu::Extent3d {
                        width: MESH_SDF_RES,
                        height: MESH_SDF_RES,
                        depth_or_array_layers: MESH_SDF_RES,
                    },
                );
                self.sdf_cache_writes.push((hash, staging));
            }
        }
    }
}
