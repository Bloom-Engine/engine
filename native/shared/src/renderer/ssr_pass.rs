//! Screen-space reflections: stochastic ray march + temporal denoiser.
//! Split from end_frame_with_scene (2000-line file policy + render-graph
//! migration prep). Both entry points no-op when `ssr_enabled` is false.

use super::{Renderer, SsrParams, SsrTemporalParams};

impl Renderer {
    /// Quarter-res stochastic SSR ray march (GGX-sampled directions,
    /// jittered starts; temporal accumulation makes it converge).
    pub(super) fn record_ssr_march(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
    ) {
    // ============================================================
    // SSR: view-space ray march of the depth buffer + HDR sample.
    // ============================================================
    // PT-1: skipped while the path tracer owns the frame — it marches
    // the raster-lit HDR, which PT has already overwritten.
    if self.ssr_enabled && !self.pt_active() {
        let inv_proj = self.current_inv_proj_matrix;
        // EN-021 — view→world rotation for the env-miss fallback: the
        // transpose of the view matrix's 3×3 (rigid view ⇒ inverse
        // rotation = transpose). Column j of the inverse is row j of
        // the view rotation.
        let v = self.current_view_matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let sp = SsrParams {
            inv_proj,
            proj: self.current_proj_matrix,
            // n_steps lowered from 32 → 8 for stochastic SSR: the
            // GGX-sampled ray direction + jittered start offset +
            // temporal accumulation over 4–8 frames fills in the
            // gaps that any single-frame coarse march leaves behind.
            // Thickness tolerance grows proportionally with
            // step_size so the relative-error reject heuristic
            // still works with the larger strides.
            params: [self.ssr_strength, 8.0, 8.0, self.taa_frame_index as f32],
            inv_view_rot,
            // Env max LOD 6.0 matches the material path's roughness×6
            // mip ramp; intensity rides lighting camera_pos.w exactly
            // like sample_env does.
            params2: [6.0, self.lighting_uniforms.camera_pos[3], 0.0, 0.0],
        };
        self.queue.write_buffer(&self.ssr_uniform_buffer, 0, bytemuck::bytes_of(&sp));

        if self.ssr_bg_cache.is_none() {
            // EN-021 — env panorama (or the 1×1 default) for the miss
            // fallback. The cache is invalidated wherever the lighting
            // bind group swaps env sources.
            let env_view = self
                .sky_texture
                .as_ref()
                .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
                .unwrap_or_else(|| {
                    self._scene_env_default_texture
                        .create_view(&wgpu::TextureViewDescriptor::default())
                });
            self.ssr_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssr_bg"),
                layout: &self.ssr_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ssr_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.material_rt_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&env_view) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                ],
            }));
        }
        let bg = self.ssr_bg_cache.as_ref().unwrap();
        let ssr_ts = profiler.pass_timestamp_writes("ssr_pass");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssr_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ssr_rt_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: ssr_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.ssr_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    } else {
        // SSR disabled — clear the RT so TAA's read returns 0
        // (transparent black). One-time clear is cheaper than a
        // full clear+pipeline switch every frame.
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssr_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ssr_rt_view,
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

    /// SSR temporal denoiser: 3x3 pre-filter + neighborhood-clamped
    /// history blend; compose reads ssr_history[cur].
    pub(super) fn record_ssr_temporal(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) {
    // ============================================================
    // SSR temporal denoiser: blend the noisy single-ray SSR with
    // the reprojected previous history so 4–8 frames of GGX-sampled
    // rays converge to a smooth reflection. 3×3 pre-filter of the
    // noisy current frame + neighborhood clamp of reprojected
    // history. Compose then reads ssr_history[cur] instead of
    // ssr_rt.
    // ============================================================
    // PT-1: same gate as the march — no fresh rays, nothing to blend.
    if self.ssr_enabled && !self.pt_active() {
        let prev_idx = 1 - self.ssr_history_idx;
        let cur_idx = self.ssr_history_idx;

        // First frame: alpha=1 so we initialize history from the
        // current noisy frame rather than blending with zeros.
        let alpha = if self.taa_frame_index == 0 { 1.0_f32 } else { 0.1_f32 };
        let tp = SsrTemporalParams {
            params: [alpha, 0.0, 0.0, 0.0],
        };
        self.queue.write_buffer(&self.ssr_temporal_uniform_buffer, 0, bytemuck::bytes_of(&tp));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr_temporal_bg"),
            layout: &self.ssr_temporal_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ssr_temporal_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ssr_rt_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.ssr_history_views[prev_idx]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssr_temporal_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ssr_history_views[cur_idx],
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
        pass.set_pipeline(&self.ssr_temporal_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    }
}
