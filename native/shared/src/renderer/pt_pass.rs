//! PT-1 — progressive path-trace megakernel dispatch
//! (docs/pt/PT-1-progressive-megakernel.md). Replaces the lit opaque
//! scene colour in hdr_rt when path tracing is active; sky pixels are
//! left untouched so the raster sky/clouds survive, and translucency
//! still composites on top afterwards.

use super::*;

impl Renderer {
    pub(super) fn record_pt_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        surf_w: u32,
        surf_h: u32,
    ) {
        if !self.pt_active() {
            // Leaving PT (or never entering it) invalidates history so
            // re-enabling starts a fresh accumulation, not a stale one.
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
            return;
        }
        // Same readiness gate as the HW probe trace: first frames before
        // any geometry is committed have no TLAS / instance data yet.
        if self.pt_pipeline.is_none()
            || self.tlas.is_none()
            || self.tlas_instance_data_buffer.is_none()
            || self.pt_geo_vertex_buffer.is_none()
            || self.pt_geo_index_buffer.is_none()
        {
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
            return;
        }
        // PT-2 — a grown texture store means the baked view array is
        // stale; rebuild so new textures become visible to hit shading.
        if self.pt_texture_arrays_enabled && self.pt_bg_texture_count != self.textures.len() {
            self.pt_tex_bg = None;
        }

        // ---- accumulation validity ----
        // Any camera motion beyond epsilon restarts progressive
        // accumulation (mode 1). Mode 2 (realtime) ignores the reset —
        // its EMA is designed to absorb motion — but tracking prev_vp
        // costs nothing and keeps one code path.
        //
        // Compared UNJITTERED: current_vp_matrix carries the TAA Halton
        // nudge (~1e-3 in the proj Z-coupling slots), which would read
        // as motion every frame and pin the accumulator at 1 sample.
        // The jittered inv_vp still goes to the kernel — primary rays
        // must match the jittered G-buffer depth, and accumulating
        // across jitters is free anti-aliasing.
        let vp_unjittered = mat4_multiply(
            self.current_proj_matrix_unjittered,
            self.current_view_matrix,
        );
        let mut moved = false;
        for r in 0..4 {
            for c in 0..4 {
                if (vp_unjittered[r][c] - self.pt_prev_vp[r][c]).abs() > 1e-5 {
                    moved = true;
                }
            }
        }
        if moved && self.pt_mode == 1 {
            self.pt_accum_count = 0;
        }
        // PT-3 — the uniform needs LAST frame's VP for history
        // reprojection; stash it before the tracker is overwritten.
        let prev_vp_for_reproject = self.pt_prev_vp;
        self.pt_prev_vp = vp_unjittered;
        // Geometry changed under the accumulated image (door opened,
        // enemy died) → PROGRESSIVE history is a lie, restart. Realtime
        // must NOT reset here: tlas_version bumps on every node
        // transform — during gameplay that is every single frame, which
        // silently pinned the SVGF history at 1 sample (found via the
        // debug-20 history-length view; the frozen-seed era masked it).
        // Mode 2's per-tap depth validation already rejects exactly the
        // texels whose surface actually changed.
        let mut tlas_reset = false;
        if self.tlas_built_version != self.pt_last_tlas_version {
            self.pt_last_tlas_version = self.tlas_built_version;
            if self.pt_mode == 1 {
                self.pt_accum_count = 0;
                tlas_reset = true;
            }
        }
        // Progressive mode + camera in motion OR scene churn: the raster
        // frame stays on screen (kernel write threshold) and any sample
        // traced now is discarded by next frame's reset — skip the
        // dispatch entirely. During combat the TLAS bumps every frame
        // (enemy transforms), so without the tlas_reset arm progressive
        // paid the full-res trace cost while displaying raster.
        if self.pt_mode == 1 && (moved || tlas_reset) {
            self.pt_wrote_frame = false;
            return;
        }

        // ---- trace grid ----
        // Realtime mode traces at half resolution (4x fewer rays) and
        // joint-bilaterally upsamples in the final à-trous pass; the
        // 2x2 sample phase rotates per frame so the temporal EMA
        // integrates full-res coverage over 4 frames. Progressive mode
        // stays full-res.
        // The realtime trace grid is capped at ~0.5 Mpx (960x540) so
        // raising the raster render scale sharpens the image without
        // multiplying the ray budget — the upsampler handles arbitrary
        // trace-to-full ratios. Progressive stays uncapped: quality is
        // its entire point.
        let (trace_w, trace_h) = if self.pt_mode >= 2 {
            (surf_w.div_ceil(2).min(960), surf_h.div_ceil(2).min(540))
        } else {
            (surf_w, surf_h)
        };
        // Phase pinned to (0,0): rotating it makes each trace texel
        // sample a different full-res pixel every frame, and on
        // depth-chaotic surfaces (grass) the history validation then
        // rejects almost every frame — texels never accumulate past
        // 1 spp and read as white speckle. A consistent owner pixel
        // keeps history valid; the upsample covers the other three.
        let _phase = [0u32, 0u32];

        // ---- accumulation buffers (vec4<f32> per pixel, ping-pong) ----
        // Sized to the TRACE grid; a mode switch changes the size and
        // recreates (which also resets accumulation — correct, the two
        // modes' buffer contents are not interchangeable).
        let needed = (trace_w as u64) * (trace_h as u64) * 16;
        let recreate = match &self.pt_accum_buffers[0] {
            Some(b) => b.size() != needed,
            None => true,
        };
        if recreate {
            for (i, slot) in self.pt_accum_buffers.iter_mut().enumerate() {
                *slot = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 { "pt_accum_a" } else { "pt_accum_b" }),
                    size: needed,
                    // COPY_SRC: the debug-16 numeric readback copies a
                    // window of this buffer to a staging buffer.
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
            }
            // SVGF moments side-channel (mu1, mu2, history length, raw
            // depth), ping-pong with the accum pair. wgpu zero-inits,
            // and pt_accum_count = 0 marks the whole history invalid.
            for (i, slot) in self.pt_moments_buffers.iter_mut().enumerate() {
                *slot = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 { "pt_moments_a" } else { "pt_moments_b" }),
                    size: needed,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            // PT-4 — ReSTIR reservoirs (light idx, W, M, target pdf).
            // Zero-init M = 0 marks every reservoir empty.
            for (i, slot) in self.pt_resv_buffers.iter_mut().enumerate() {
                *slot = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 { "pt_resv_a" } else { "pt_resv_b" }),
                    size: needed,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            // COPY_SRC: the first à-trous iteration's output is copied
            // back over the accum buffer as next frame's colour history
            // (SVGF feeds back the once-filtered signal).
            self.pt_atrous_scratch = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_atrous_scratch"),
                size: needed,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.pt_atrous_scratch2 = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_atrous_scratch2"),
                size: needed,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            self.pt_bg = [None, None];
            self.pt_atrous_bgs = [[None, None, None, None, None, None], [None, None, None, None, None, None]];
            self.pt_accum_count = 0;
            self.pt_accum_idx = 0;
        }

        // ---- uniforms ----
        // Sun / sky derivation matches record_ssgi_passes exactly so PT
        // brightness lines up with the raster + GI frame it replaces.
        let ld = self.lighting_uniforms.light_dir;
        let sun_inv_len = 1.0 / (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]).sqrt().max(1e-4);
        let sun_intensity = ld[3].max(0.0);
        let lc = self.lighting_uniforms.light_color;
        let amb = self.lighting_uniforms.ambient;
        let sky_intensity = amb[3].max(0.0);

        let light_count = (self.lighting_uniforms.point_light_count[0] as usize).min(16);
        let mut lights = [[0.0f32; 4]; 32];
        for i in 0..light_count {
            let pl = &self.lighting_uniforms.point_lights[i];
            lights[i * 2] = pl.position; // xyz + range
            lights[i * 2 + 1] = pl.color; // rgb + intensity
        }

        let max_bounces = if self.pt_mode == 2 { 2.0 } else { 8.0 };
        let cam = self.current_camera_pos;
        // current_inv_vp_matrix is stored transposed relative to what
        // WGSL's `M * v` needs (the composed VP inherits mat4_multiply's
        // convention; its inverse lands transposed). Upload the
        // transpose so the kernel's unprojection is the real inverse —
        // without this every ray collapses to one degenerate bundle and
        // the whole path trace silently hits garbage (found via numeric
        // readback; see docs/pt/PT-2 notes).
        let m = &self.current_inv_vp_matrix;
        let inv_vp_t = [
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ];
        // The reprojection VP uploads RAW — the opposite of inv_vp. The
        // two matrix conventions coexist: mat4_invert outputs land
        // transposed relative to WGSL's M*v (hence inv_vp's transpose
        // above), while mat4_multiply products are already in M*v
        // layout — the shadow cascade VPs upload raw for the same
        // reason. Transposing this one collapsed every reprojection
        // into a ~40-texel band at screen centre (debug-23 dump), so
        // history never matched under camera motion — invisible in the
        // frozen-seed era, which is why it survived since PT-3 M1.
        let prev_vp_t = prev_vp_for_reproject;
        let params = PtParamsCpu {
            inv_vp: inv_vp_t,
            prev_vp: prev_vp_t,
            cam_pos: [cam[0], cam[1], cam[2], 0.0],
            sun_dir: [
                -ld[0] * sun_inv_len,
                -ld[1] * sun_inv_len,
                -ld[2] * sun_inv_len,
                0.0,
            ],
            sun_color: [
                lc[0] * sun_intensity,
                lc[1] * sun_intensity,
                lc[2] * sun_intensity,
                0.0,
            ],
            sky_color: [
                amb[0] * sky_intensity,
                amb[1] * sky_intensity,
                amb[2] * sky_intensity,
                0.0,
            ],
            // size.z: PT's OWN frame counter, not taa_frame_index —
            // the TAA index freezes when TAA is disabled (settings,
            // headless tests), which froze the sample sequence and
            // silently stopped progressive accumulation from ever
            // converging (found by the pt_progressive golden).
            size: [trace_w, trace_h, self.pt_frame_index, self.pt_accum_count],
            cfg: [
                self.pt_mode as f32,
                max_bounces,
                light_count as f32,
                self.pt_debug,
            ],
            // ext.z: hybrid sun — realtime mode samples the raster
            // shadow cascades instead of tracing sun rays (crisp
            // noise-free direct shadows). Progressive keeps traced sun
            // for reference-quality penumbra.
            ext: [
                surf_w,
                surf_h,
                if self.pt_mode >= 2 && self.shadow_map.enabled { 1 } else { 0 },
                // PT-4 experimental flag (BLOOM_PT_RESTIR=1), realtime only.
                if self.pt_restir && self.pt_mode >= 2 { 1 } else { 0 },
            ],
            // RAW upload, unlike inv_vp: the shadow VPs are consumed as
            // M*v by every existing WGSL user (scene shader, WSRC
            // bake), so they are already stored in WGSL column layout.
            // Verified empirically via debug 18 — transposing them
            // black-shadows the whole frame.
            shadow_vps: self.shadow_map.light_vps,
            lights,
        };
        self.queue.write_buffer(&self.pt_uniform_buffer, 0, bytemuck::bytes_of(&params));

        // ---- bind groups (lazy; nulled on resize / TLAS or instance
        // buffer recreation). Two ping-pong variants: bg[i] reads accum
        // buffer i (binding 8) and writes buffer 1-i (binding 13).
        for i in 0..2 {
            if self.pt_bg[i].is_some() {
                continue;
            }
            let tlas = self.tlas.as_ref().unwrap();
            let entries = vec![
                    wgpu::BindGroupEntry { binding: 0, resource: self.pt_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: tlas.as_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.tlas_instance_data_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.material_rt_view) },
                    // Raw albedo atlas, NOT the pre-lit radiance atlas the
                    // GI probe trace uses — PT computes its own lighting at
                    // hits; radiance would double-count.
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.mesh_card_atlas_view) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.mesh_card_atlas_sampler) },
                    wgpu::BindGroupEntry { binding: 8, resource: self.pt_accum_buffers[i].as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                    wgpu::BindGroupEntry { binding: 10, resource: self.pt_geo_vertex_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: self.pt_geo_index_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: self.pt_accum_buffers[1 - i].as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[0]) },
                    wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[1]) },
                    wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&self.shadow_map.depth_views[2]) },
                    wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.shadow_map.sampler) },
                    // SVGF moments: read prev (paired with accum read
                    // side), write out (paired with the write side).
                    wgpu::BindGroupEntry { binding: 18, resource: self.pt_moments_buffers[i].as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 19, resource: self.pt_moments_buffers[1 - i].as_ref().unwrap().as_entire_binding() },
                    // PT-4 ReSTIR reservoirs, same ping-pong pairing.
                    wgpu::BindGroupEntry { binding: 20, resource: self.pt_resv_buffers[i].as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 21, resource: self.pt_resv_buffers[1 - i].as_ref().unwrap().as_entire_binding() },
                    // PT-7 — velocity MRT (written by hdr_scene, which
                    // runs before the PT node every frame).
                    wgpu::BindGroupEntry { binding: 22, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
            ];
            self.pt_bg[i] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pt_bg"),
                layout: self.pt_layout.as_ref().unwrap(),
                entries: &entries,
            }));
        }
        // PT-2 — group 1: the texture binding array. Real store views
        // first, white (slot 0) padding to the fixed layout count. The
        // bind group holds refs, so the temporary views live with it.
        if self.pt_texture_arrays_enabled && self.pt_tex_bg.is_none() {
            let n = self.textures.len().min(PT_MAX_TEXTURES);
            let tex_views: Vec<wgpu::TextureView> = (0..n.max(1))
                .map(|i| self.textures[i.min(self.textures.len() - 1)]
                    .create_view(&wgpu::TextureViewDescriptor::default()))
                .collect();
            let tex_view_refs: Vec<&wgpu::TextureView> = (0..PT_MAX_TEXTURES)
                .map(|i| &tex_views[if i < n { i } else { 0 }])
                .collect();
            self.pt_tex_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pt_tex_bg"),
                layout: self.pt_tex_layout.as_ref().unwrap(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&tex_view_refs),
                }],
            }));
            self.pt_bg_texture_count = self.textures.len();
        }

        // ---- dispatch ----
        {
            let ts = profiler.compute_pass_timestamp_writes("pt_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt_pass"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(self.pt_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.pt_bg[self.pt_accum_idx].as_ref().unwrap(), &[]);
            if self.pt_texture_arrays_enabled {
                pass.set_bind_group(1, self.pt_tex_bg.as_ref().unwrap(), &[]);
            }
            pass.dispatch_workgroups((trace_w + 7) / 8, (trace_h + 7) / 8, 1);
        }
        // This frame wrote into buffers[1 - idx]; it becomes next
        // frame's read side.
        let written_idx = 1 - self.pt_accum_idx;
        self.pt_accum_idx = written_idx;

        // ---- PT-3b: SVGF wavelet filter (realtime mode only) ----
        // Four variance-guided à-trous iterations on the trace grid
        // (steps 1/2/4/8), then the full-res upsample+modulate pass.
        // After iteration 1 the once-filtered signal is copied back
        // over the accum buffer: SVGF feeds the first wavelet output
        // into next frame's colour history (moments stay raw). This is
        // what makes the temporal loop stable at 1 spp — raw history
        // carries every spike forward, once-filtered history does not.
        // Progressive mode converges on its own and writes hdr
        // directly from the kernel.
        if self.pt_mode >= 2
            && self.pt_debug == 0.0
            && self.pt_atrous_mid_pipeline.is_some()
            && self.pt_atrous_scratch.is_some()
        {
            // p.y = 1.0 flags the FIRST iteration: it may substitute a
            // spatial variance estimate where the history is young.
            for (i, step) in [1.0f32, 2.0, 4.0, 8.0, 16.0, 1.0].iter().enumerate() {
                let first = if i == 0 { 1.0f32 } else { 0.0 };
                let p = [
                    [*step, first, trace_w as f32, trace_h as f32],
                    [surf_w as f32, surf_h as f32, 0.0, 0.0],
                ];
                self.queue.write_buffer(&self.pt_atrous_params_bufs[i], 0, bytemuck::bytes_of(&p));
            }

            if self.pt_atrous_bgs[written_idx][0].is_none() {
                let scratch = self.pt_atrous_scratch.as_ref().unwrap();
                let scratch2 = self.pt_atrous_scratch2.as_ref().unwrap();
                let accum_w = self.pt_accum_buffers[written_idx].as_ref().unwrap();
                let moments_w = self.pt_moments_buffers[written_idx].as_ref().unwrap();
                // Stage src → dst chain: accum→s1, then the scratches
                // ping-pong; the final upsample reads the last-written
                // scratch. cs_final never writes dst; it gets whichever
                // scratch is not its src (RO+RW of one buffer in a
                // single group fails validation).
                let chain: [(&wgpu::Buffer, &wgpu::Buffer); 6] = [
                    (accum_w, scratch),
                    (scratch, scratch2),
                    (scratch2, scratch),
                    (scratch, scratch2),
                    (scratch2, scratch),
                    (scratch, scratch2),
                ];
                for (i, (src, dst)) in chain.iter().enumerate() {
                    self.pt_atrous_bgs[written_idx][i] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pt_atrous_bg"),
                        layout: self.pt_atrous_layout.as_ref().unwrap(),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: self.pt_atrous_params_bufs[i].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view) },
                            wgpu::BindGroupEntry { binding: 6, resource: moments_w.as_entire_binding() },
                        ],
                    }));
                }
            }

            {
                let ts = profiler.compute_pass_timestamp_writes("pt_atrous");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pt_atrous"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(self.pt_atrous_mid_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, self.pt_atrous_bgs[written_idx][0].as_ref().unwrap(), &[]);
                pass.dispatch_workgroups((trace_w + 7) / 8, (trace_h + 7) / 8, 1);
            }
            // History feedback: the pass split makes the copy legal
            // (buffer copies cannot live inside a compute pass).
            encoder.copy_buffer_to_buffer(
                self.pt_atrous_scratch.as_ref().unwrap(),
                0,
                self.pt_accum_buffers[written_idx].as_ref().unwrap(),
                0,
                needed,
            );
            {
                let ts = profiler.compute_pass_timestamp_writes("pt_atrous2");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pt_atrous2"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(self.pt_atrous_mid_pipeline.as_ref().unwrap());
                for i in 1..5 {
                    pass.set_bind_group(0, self.pt_atrous_bgs[written_idx][i].as_ref().unwrap(), &[]);
                    pass.dispatch_workgroups((trace_w + 7) / 8, (trace_h + 7) / 8, 1);
                }
                pass.set_pipeline(self.pt_atrous_final_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, self.pt_atrous_bgs[written_idx][5].as_ref().unwrap(), &[]);
                pass.dispatch_workgroups((surf_w + 7) / 8, (surf_h + 7) / 8, 1);
            }
        }
        // Mirrors the kernel's write threshold: mode 1 leaves the raster
        // frame on screen until 8 samples exist (u.size.w carried the
        // pre-increment count), so SSGI/SSR must keep running for those
        // frames — the gates downstream check pt_owns_frame().
        self.pt_wrote_frame = self.pt_mode >= 2 || self.pt_accum_count >= 8;
        self.pt_accum_count = self.pt_accum_count.saturating_add(1);
        self.pt_frame_index = self.pt_frame_index.wrapping_add(1);

        // ---- debug 16: numeric readback of traced intersections ----
        // Copies a window of the accum buffer (center of frame) into a
        // staging buffer each frame; the previous frame's copy is mapped
        // (blocking) and dumped to pt_trace_dump.txt once.
        if (self.pt_debug == 16.0
            || self.pt_debug == 17.0
            || self.pt_debug == 19.0
            || self.pt_debug == 22.0
            || self.pt_debug == 23.0)
            && self.pt_accum_count > 30
            && !self.pt_dump_written
        {
            // Offsets in TRACE-grid units — the accum buffers are
            // half-res in realtime mode.
            let dump_pixels: u64 = (trace_w as u64).min(4096);
            let row = (trace_h / 2) as u64;
            let offset = row * trace_w as u64 * 16;
            if self.pt_readback_buffer.is_none() {
                self.pt_readback_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pt_readback"),
                    size: dump_pixels * 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                encoder.copy_buffer_to_buffer(
                    // written_idx == pt_accum_idx here (already flipped):
                    // the buffer this frame's dispatch wrote.
                    self.pt_accum_buffers[self.pt_accum_idx].as_ref().unwrap(),
                    offset,
                    self.pt_readback_buffer.as_ref().unwrap(),
                    0,
                    dump_pixels * 16,
                );
            } else {
                // Previous frame's copy has been submitted; map it now.
                let buf = self.pt_readback_buffer.as_ref().unwrap();
                let slice = buf.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                let data = slice.get_mapped_range();
                let vals: &[[f32; 4]] = bytemuck::cast_slice(&data);
                let mut out = String::new();
                out.push_str(&format!(
                    "middle row, {} pixels, mode {}\n",
                    vals.len(),
                    self.pt_debug
                ));
                // Every 64th pixel across the full row. Field meaning:
                // 16 = t / id / prim / kind; 17 = p0.xyz / raw depth.
                for (i, v) in vals.iter().enumerate().step_by(64) {
                    out.push_str(&format!(
                        "col {i}: {:.4} {:.4} {:.4} {:.6}\n",
                        v[0], v[1], v[2], v[3]
                    ));
                }
                // Also dump the CPU-side uniform inputs for comparison,
                // plus the unprojection computed in BOTH multiply
                // conventions. Whichever matches the GPU dump is what
                // the shader effectively computed; the other (if sane)
                // is the fix.
                let ndc = [0.0f32, 0.0, 0.998647, 1.0];
                let m = &self.current_inv_vp_matrix;
                let mut h_col = [0.0f32; 4]; // h_i = sum_c m[c][i] * ndc[c]
                let mut h_row = [0.0f32; 4]; // h_i = sum_c m[i][c] * ndc[c]
                for i in 0..4 {
                    for c in 0..4 {
                        h_col[i] += m[c][i] * ndc[c];
                        h_row[i] += m[i][c] * ndc[c];
                    }
                }
                out.push_str(&format!(
                    "cpu cam_pos = {:?}\n\
                     unproject as columns: h={:?} p={:?}\n\
                     unproject transposed: h={:?} p={:?}\n",
                    self.current_camera_pos,
                    h_col,
                    [h_col[0] / h_col[3], h_col[1] / h_col[3], h_col[2] / h_col[3]],
                    h_row,
                    [h_row[0] / h_row[3], h_row[1] / h_row[3], h_row[2] / h_row[3]],
                ));
                // Distinct instance-id count over the whole row.
                let mut ids: Vec<i64> = vals
                    .iter()
                    .filter(|v| v[3] != 0.0)
                    .map(|v| v[1] as i64)
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                let misses = vals.iter().filter(|v| v[3] == 0.0).count();
                out.push_str(&format!("distinct hit ids: {:?}\n", ids));
                out.push_str(&format!("misses: {}\n", misses));
                drop(data);
                buf.unmap();
                let _ = std::fs::write("pt_trace_dump.txt", out);
                self.pt_dump_written = true;
            }
        }
    }
}
