//! Lumen-style screen-probe SSGI: probe placement, Hi-Z/HW/SDF trace,
//! temporal EMA, and octahedral resolve into ssgi_rt. Split from
//! end_frame_with_scene (2000-line file policy + render-graph migration
//! prep). When disabled, clears ssgi_rt to transparent.

use super::*;

impl Renderer {
    pub(super) fn record_ssgi_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        surf_w: u32,
        surf_h: u32,
    ) {
    // ============================================================
    // Ticket 007a: Lumen-style screen-probe SSGI.
    // place → trace (SW Hi-Z) → temporal (EMA ping-pong) → resolve.
    // Resolve writes `ssgi_rt_view` so downstream compositing is
    // unchanged. When disabled we just clear `ssgi_rt_view` to
    // transparent (same fallback shape as the old per-pixel path).
    // ============================================================
    let half_w = (surf_w / 2).max(1);
    let half_h = (surf_h / 2).max(1);
    let gw = self.probe_grid_w;
    let gh = self.probe_grid_h;
    let write_idx = self.probe_history_idx;
    let prev_idx = 1 - write_idx;

    // PT-1: while the path tracer owns the frame its output already
    // contains full GI — probe SSGI would burn ~2ms to be composited
    // over by nothing (the else-branch clear keeps compose additive-
    // safe either way).
    if self.ssgi_enabled && !self.pt_active() {
        let p00 = self.current_proj_matrix[0][0];
        let p11 = self.current_proj_matrix[1][1];
        let p20 = self.current_proj_matrix[2][0];
        let p21 = self.current_proj_matrix[2][1];
        let inv_view = mat4_invert(self.current_view_matrix);

        // ---- place ----
        let place_params = ProbePlaceParams {
            inv_view,
            proj_row01: [p00, p11, p20, p21],
            size: [half_w, half_h, gw, gh],
            params: [self.taa_frame_index as f32, PROBE_TILE_SIZE as f32, 0.0, 0.0],
        };
        self.queue.write_buffer(&self.probe_place_uniform, 0, bytemuck::bytes_of(&place_params));
        if self.probe_place_bg_cache.is_none() {
            self.probe_place_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe_place_bg"),
                layout: &self.probe_place_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.probe_place_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: self.probe_header_buffer.as_entire_binding() },
                ],
            }));
        }
        {
            let ts = profiler.compute_pass_timestamp_writes("probe_place_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_place_pass"), timestamp_writes: ts,
            });
            pass.set_pipeline(&self.probe_place_pipeline);
            pass.set_bind_group(0, self.probe_place_bg_cache.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups((gw + 7) / 8, (gh + 7) / 8, 1);
        }

        // ---- trace ----
        // Sun direction in world space — inverted because our
        // `light_dir` points from light toward the scene, while the
        // shader's NdotL expects the vector from the shading point
        // toward the light. Normalised because the shader doesn't.
        let ld = self.lighting_uniforms.light_dir;
        let sun_inv_len = 1.0 / (ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]).sqrt().max(1e-4);
        let sun_dir_ws = [
            -ld[0] * sun_inv_len,
            -ld[1] * sun_inv_len,
            -ld[2] * sun_inv_len,
            ld[3],
        ];
        // Sun colour = light_color × light intensity (ld.w). Sky
        // colour = ambient × ambient intensity (ambient.w) — a
        // crude dome irradiance, good enough for a one-bounce
        // shading estimate. Both fields are ignored by the SW
        // shader which inherits the same uniform struct layout.
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
        let trace_params = ProbeTraceParams {
            view: self.current_view_matrix,
            proj: self.current_proj_matrix,
            inv_view,
            proj_row01: [p00, p11, p20, p21],
            size: [half_w, half_h, gw, gh],
            params: [
                self.taa_frame_index as f32,
                self.ssgi_intensity,
                self.ssgi_radius,
                10.0,  // firefly luma cap
            ],
            sun_dir: sun_dir_ws,
            sun_color,
            sky_color,
            // Ticket 014 V3 — clipmap origin xyz + full extent w.
            // The SDF trace variant reads these; HW + Hi-Z ignore.
            clipmap: [
                self.scene_sdf_clipmap_origin[0],
                self.scene_sdf_clipmap_origin[1],
                self.scene_sdf_clipmap_origin[2],
                SCENE_SDF_CLIPMAP_EXTENT,
            ],
            // Ticket 014 V6/V13 — WSRC cascade cubes. `extent =
            // 0` marks an unbaked cascade; the shader's
            // `pick_cascade` helper skips those and falls through
            // to the next cascade (or returns black if none are
            // ready). First frame after startup all three are
            // unbaked → miss returns black, matching pre-V6.
            wsrc_cascades: [
                [
                    self.wsrc_origin[0][0],
                    self.wsrc_origin[0][1],
                    self.wsrc_origin[0][2],
                    if self.wsrc_built[0] { WSRC_CASCADE_EXTENTS[0] } else { 0.0 },
                ],
                [
                    self.wsrc_origin[1][0],
                    self.wsrc_origin[1][1],
                    self.wsrc_origin[1][2],
                    if self.wsrc_built[1] { WSRC_CASCADE_EXTENTS[1] } else { 0.0 },
                ],
                [
                    self.wsrc_origin[2][0],
                    self.wsrc_origin[2][1],
                    self.wsrc_origin[2][2],
                    if self.wsrc_built[2] { WSRC_CASCADE_EXTENTS[2] } else { 0.0 },
                ],
            ],
        };
        self.queue.write_buffer(&self.probe_trace_uniform, 0, bytemuck::bytes_of(&trace_params));
        // V3 — trace BG now binds the prev-frame history view at
        // binding 11. `prev_idx` ping-pongs every frame so we
        // cache both slots independently.
        if self.probe_trace_bg_cache[prev_idx].is_none() {
            self.probe_trace_bg_cache[prev_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe_trace_bg"),
                layout: &self.probe_trace_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.probe_trace_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hiz_views[1]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.hiz_views[2]) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.hiz_views[3]) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.hiz_views[4]) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                ],
            }));
        }
        // HW trace needs both the TLAS (at least one instance) and
        // the instance-data buffer to exist. Fall back to SDF or
        // Hi-Z when either is missing on an HW-enabled adapter
        // (e.g. first frame before the scene has loaded any
        // geometry).
        let use_hw = self.hw_rt_enabled
            && self.probe_trace_hw_pipeline.is_some()
            && self.tlas.is_some()
            && self.tlas_instance_data_buffer.is_some();
        // Ticket 014 V3/V4 — pick SDF sphere-trace over Hi-Z when
        // the scene clipmap is baked AND the instance-data buffer
        // is ready (needed for broad-phase textured hit sampling
        // added in V4). Otherwise fall through to Hi-Z. HW still
        // wins over both when the feature was granted.
        let use_sdf = !use_hw
            && self.scene_sdf_clipmap_built
            && self.tlas_instance_data_buffer.is_some();
        // Log the backend once (and again if it changes, e.g. clipmap
        // finishing its first bake promotes hiz → sdf). Nothing else in
        // the engine reveals which tier actually runs.
        let backend = if use_hw { "hw-ray-query" } else if use_sdf { "sdf-clipmap" } else { "hiz-screen" };
        if self.ssgi_backend_logged != Some(backend) {
            self.ssgi_backend_logged = Some(backend);
            eprintln!("bloom: ssgi trace backend = {}", backend);
        }

        if use_hw {
            // Build the HW bind group lazily. V3 uses a per-
            // prev_idx slot since the prev-frame history view
            // ping-pongs each frame.
            if self.probe_trace_hw_bg_cache[prev_idx].is_none() {
                let tlas = self.tlas.as_ref().unwrap();
                self.probe_trace_hw_bg_cache[prev_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe_trace_hw_bg"),
                    layout: self.probe_trace_hw_layout.as_ref().unwrap(),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.probe_trace_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: tlas.as_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.tlas_instance_data_buffer.as_ref().unwrap().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                        // Ticket 013 V2: the HW trace samples the
                        // *radiance* atlas (pre-lit by card_light_pass)
                        // at hit, not the raw albedo atlas.
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.mesh_card_radiance_view) },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                        // V7/V10 — WSRC atlas + linear sampler.
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.wsrc_atlas_sampler) },
                        // V3 — prev-frame probe history.
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                    ],
                }));
            }
            let ts = profiler.compute_pass_timestamp_writes("probe_trace_hw_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_trace_hw_pass"), timestamp_writes: ts,
            });
            pass.set_pipeline(self.probe_trace_hw_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.probe_trace_hw_bg_cache[prev_idx].as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(gw, gh, 1);
        } else if use_sdf {
            // Ticket 014 V3 — SW SDF sphere-trace path.
            // V3 (ticket 016) uses a per-prev_idx slot for the
            // prev-frame history binding.
            if self.probe_trace_sdf_bg_cache[prev_idx].is_none() {
                let nf_samp = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("clipmap_nonfiltering_sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                });
                let instance_buf = self.tlas_instance_data_buffer.as_ref()
                    .expect("V4: instance_data buffer must exist before SDF dispatch");
                self.probe_trace_sdf_bg_cache[prev_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe_trace_sdf_bg"),
                    layout: &self.probe_trace_sdf_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.probe_trace_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.scene_sdf_clipmap_view) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&nf_samp) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                        wgpu::BindGroupEntry { binding: 5, resource: instance_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.mesh_card_radiance_view) },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                        // V6/V10 — WSRC atlas + linear sampler.
                        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.wsrc_atlas_view) },
                        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.wsrc_atlas_sampler) },
                        // V3 — prev-frame probe history.
                        wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                    ],
                }));
            }
            let ts = profiler.compute_pass_timestamp_writes("probe_trace_sdf_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_trace_sdf_pass"), timestamp_writes: ts,
            });
            pass.set_pipeline(&self.probe_trace_sdf_pipeline);
            pass.set_bind_group(0, self.probe_trace_sdf_bg_cache[prev_idx].as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(gw, gh, 1);
        } else {
            let ts = profiler.compute_pass_timestamp_writes("probe_trace_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_trace_pass"), timestamp_writes: ts,
            });
            pass.set_pipeline(&self.probe_trace_pipeline);
            pass.set_bind_group(0, self.probe_trace_bg_cache[prev_idx].as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(gw, gh, 1);
        }

        // ---- temporal (EMA) ----
        // First frame forces alpha=1 so the history is seeded from
        // the current trace rather than blending against a zero clear.
        let force_refresh = if self.taa_frame_index == 0 { 1.0_f32 } else { 0.0_f32 };
        let temporal_params = ProbeTemporalParams {
            params: [0.25, force_refresh, gw as f32, gh as f32],
        };
        self.queue.write_buffer(&self.probe_temporal_uniform, 0, bytemuck::bytes_of(&temporal_params));
        // Bind group indexed by write_idx: each direction of the
        // ping-pong (read prev, write write) gets its own cached BG.
        if self.probe_temporal_bg_cache[write_idx].is_none() {
            self.probe_temporal_bg_cache[write_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe_temporal_bg"),
                layout: &self.probe_temporal_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.probe_temporal_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.probe_trace_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[prev_idx]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[write_idx]) },
                ],
            }));
        }
        {
            let ts = profiler.compute_pass_timestamp_writes("probe_temporal_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_temporal_pass"), timestamp_writes: ts,
            });
            pass.set_pipeline(&self.probe_temporal_pipeline);
            pass.set_bind_group(0, self.probe_temporal_bg_cache[write_idx].as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(gw, gh, 1);
        }

        // ---- resolve ----
        let resolve_params = ProbeResolveParams {
            inv_view,
            proj_row01: [p00, p11, p20, p21],
            size: [half_w, half_h, gw, gh],
            params: [PROBE_TILE_SIZE as f32, 1.0, 0.0, 0.0],
        };
        self.queue.write_buffer(&self.probe_resolve_uniform, 0, bytemuck::bytes_of(&resolve_params));
        if self.probe_resolve_bg_cache[write_idx].is_none() {
            self.probe_resolve_bg_cache[write_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe_resolve_bg"),
                layout: &self.probe_resolve_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.probe_resolve_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.probe_header_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.probe_history_views[write_idx]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                ],
            }));
        }
        let ts = profiler.pass_timestamp_writes("probe_resolve_pass");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("probe_resolve_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ssgi_rt_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.probe_resolve_pipeline);
        pass.set_bind_group(0, self.probe_resolve_bg_cache[write_idx].as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    } else {
        // SSGI disabled — clear the resolve target so downstream
        // composite reads contribute zero.
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssgi_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ssgi_rt_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(pass);
    }
    }
}
