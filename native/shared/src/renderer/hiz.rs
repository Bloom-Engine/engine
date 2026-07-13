//! Hi-Z (hierarchical linear-depth) pyramid build: a linearize pass
//! into mip 0 followed by min-downsample passes. Consumed by SSAO/SSR
//! ray-march acceleration and (via the max-reduce in occlusion.rs) the
//! occlusion-culling grid. Split out of renderer/mod.rs (2000-line file
//! policy); pipelines and the mip chain stay fields on [`Renderer`].

use super::formats::HIZ_MIP_COUNT;
use super::formats::halton;
use super::{HizDownsampleParams, HizLinearizeParams, SsaoBlurParams, SsaoParams};
use super::Renderer;

impl Renderer {
    pub(super) fn record_hiz_chain(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        half_w: u32,
        half_h: u32,
        p22: f32,
        p32: f32,
    ) {
        // --- Hi-Z build: linearize depth into mip 0 -----------------
        let lin_params = HizLinearizeParams {
            params: [1.0 / half_w as f32, 1.0 / half_h as f32, p22, p32],
            size: [half_w, half_h, 0, 0],
        };
        self.queue.write_buffer(&self.hiz_linearize_uniform_buffer, 0, bytemuck::bytes_of(&lin_params));
        if self.hiz_linearize_bg_cache.is_none() {
            self.hiz_linearize_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hiz_linearize_bg"),
                layout: &self.hiz_linearize_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.hiz_linearize_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                ],
            }));
        }
        {
            let bg = self.hiz_linearize_bg_cache.as_ref().unwrap();
            let ts = profiler.compute_pass_timestamp_writes("hiz_linearize_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_linearize_pass"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.hiz_linearize_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((half_w + 7) / 8, (half_h + 7) / 8, 1);
        }

        // --- Hi-Z build: downsample mip i -> mip i+1 ----------------
        for i in 0..(HIZ_MIP_COUNT - 1) as usize {
            let dst_w = (half_w >> (i + 1)).max(1);
            let dst_h = (half_h >> (i + 1)).max(1);
            let ds_params = HizDownsampleParams {
                size: [dst_w, dst_h, 0, 0],
            };
            self.queue.write_buffer(&self.hiz_downsample_uniform_buffers[i], 0, bytemuck::bytes_of(&ds_params));
            if self.hiz_downsample_bg_cache[i].is_none() {
                self.hiz_downsample_bg_cache[i] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hiz_downsample_bg"),
                    layout: &self.hiz_downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.hiz_downsample_uniform_buffers[i].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hiz_views[i]) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.hiz_views[i + 1]) },
                    ],
                }));
            }
            let bg = self.hiz_downsample_bg_cache[i].as_ref().unwrap();
            let ts_label: &'static str = match i {
                0 => "hiz_downsample_pass_1",
                1 => "hiz_downsample_pass_2",
                2 => "hiz_downsample_pass_3",
                _ => "hiz_downsample_pass_4",
            };
            let ts = profiler.compute_pass_timestamp_writes(ts_label);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(ts_label),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.hiz_downsample_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
        }

    }
}

impl Renderer {
    /// SSAO bilateral blur (depth-guided, edge-preserving) when SSAO is
    /// on, or a white-clear of the blur RT when it's off so the
    /// composite samples "no occlusion". Split from
    /// end_frame_with_scene (2000-line file policy).
    pub(super) fn record_ssao_blur(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        surf_w: u32,
        surf_h: u32,
    ) {
    // ============================================================
    // SSAO bilateral blur: smooth the noisy GTAO output while
    // preserving depth edges (depth-guided bilateral filter).
    // Reads ssao_rt → writes ssao_blur_rt.
    // ============================================================
    // PT: when the path tracer owns the frame it computes real
    // occlusion by tracing — screen-space AO on top double-darkens
    // every crevice. Route compose to "no occlusion" (white clear
    // below) for those frames.
    if self.ssao_enabled && !self.pt_owns_frame() {
        // texel_size is the size of one SSAO RT texel (half-res).
        let ao_w = (surf_w / 2).max(1) as f32;
        let ao_h = (surf_h / 2).max(1) as f32;
        let bp = SsaoBlurParams {
            params: [1.0 / ao_w, 1.0 / ao_h, 0.05, 0.0],
        };
        self.queue.write_buffer(&self.ssao_blur_uniform_buffer, 0, bytemuck::bytes_of(&bp));

        if self.ssao_blur_bg_cache.is_none() {
            self.ssao_blur_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao_blur_bg"),
                layout: &self.ssao_blur_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ssao_blur_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ssao_rt_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                ],
            }));
        }
        let bg = self.ssao_blur_bg_cache.as_ref().unwrap();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao_blur_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ssao_blur_rt_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.ssao_blur_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    } else {
        // SSAO disabled — clear the blur RT to WHITE so the
        // composite pass samples "no occlusion". Cheaper than a
        // full blur pass; the clear is the only GPU work.
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao_blur_disabled_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ssao_blur_rt_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    }
}

impl Renderer {
    /// GTAO compute dispatch (half-res, Hi-Z-accelerated, temporal EMA
    /// ping-pong). Caller guards on `ssao_enabled` and passes the
    /// projection terms. Split from end_frame_with_scene.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_gtao(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        half_w: u32,
        half_h: u32,
        p00: f32,
        p11: f32,
        p20: f32,
        p21: f32,
    ) {
        let p22 = self.current_proj_matrix[2][2];
        let p32 = self.current_proj_matrix[3][2];
        // --- SSAO (compute GTAO, samples Hi-Z pyramid) --------------
        let ld = self.lighting_uniforms.light_dir;
        let v = &self.current_view_matrix;
        let light_dir_vs = [
            v[0][0]*ld[0] + v[1][0]*ld[1] + v[2][0]*ld[2],
            v[0][1]*ld[0] + v[1][1]*ld[1] + v[2][1]*ld[2],
            v[0][2]*ld[0] + v[1][2]*ld[1] + v[2][2]*ld[2],
            0.0,
        ];
        // Temporal accumulation: ping-pong history textures.
        // `write_idx` is the current-frame output; `read_idx` the
        // previous frame's result. First 4 frames force alpha=1
        // so the initial clear never contaminates the signal.
        let write_idx = self.ssao_history_idx;
        let read_idx = 1 - write_idx;
        let frame_phase = self.ssao_history_frame % 4;
        let force_refresh = if self.ssao_history_frame < 4 { 1u32 } else { 0u32 };
        // 4-frame EMA: alpha = 1/4 = 0.25 gives equal weight to
        // each of the 4 phases at steady state.
        let alpha = 0.25_f32;
        // Halton-5 rotation: uncorrelated with TAA's base-2/3 jitter
        // so the two noise patterns don't resonate.
        let halton5 = halton(self.ssao_history_frame + 1, 5);
        let sp = SsaoParams {
            params: [
                1.0 / half_w as f32,
                1.0 / half_h as f32,
                self.ssao_radius,
                self.ssao_strength,
            ],
            proj_row01: [p00, p11, p20, p21],
            proj_z: [p22, p32, 1.0 / p00, 1.0 / p11],
            light_dir_vs,
            size: [half_w, half_h, frame_phase, force_refresh],
            temporal: [alpha, halton5, 0.0, 0.0],
        };
        self.queue.write_buffer(&self.ssao_uniform_buffer, 0, bytemuck::bytes_of(&sp));

        if self.ssao_bg_cache[write_idx].is_none() {
            self.ssao_bg_cache[write_idx] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao_bg"),
                layout: &self.ssao_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ssao_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.ssao_rt_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.hiz_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.hiz_views[1]) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.hiz_views[2]) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.hiz_views[3]) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.hiz_views[4]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.velocity_rt_view) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.ssao_history_views[read_idx]) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.ssao_history_views[write_idx]) },
                ],
            }));
        }
        let bg = self.ssao_bg_cache[write_idx].as_ref().unwrap();

        let ssao_ts = profiler.compute_pass_timestamp_writes("ssao_pass");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssao_pass"),
            timestamp_writes: ssao_ts,
        });
        pass.set_pipeline(&self.ssao_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups((half_w + 7) / 8, (half_h + 7) / 8, 1);

        // Flip ping-pong indices for the next frame.
        self.ssao_history_idx = read_idx;
        self.ssao_history_frame = self.ssao_history_frame.wrapping_add(1);
    }
}
