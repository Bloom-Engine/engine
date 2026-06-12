//! Post-FX chain passes split from end_frame_with_scene (2000-line file
//! policy + render-graph migration prep). Starts with the bloom chain;
//! the rest of the tail (compose/upscale/TAA/DoF/blur/SSS) migrates here
//! cluster by cluster.

use super::*;

impl Renderer {
    /// Bloom: progressive downsample (Karis-thresholded first tap)
    /// followed by additive upsample back up the chain. No-op (clears
    /// nothing) when disabled — compose skips the bloom sample.
    pub(super) fn record_bloom_chain(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        surf_w: u32,
        surf_h: u32,
    ) {
    // ============================================================
    // Bloom: progressive downsample (Karis-thresholded first tap)
    // followed by additive upsample back up the chain.
    // ============================================================
    if self.bloom_enabled {
    let mip_dims: Vec<(u32, u32)> = (0..BLOOM_MIP_COUNT)
        .map(|i| (
            ((surf_w / 2) >> i).max(1),
            ((surf_h / 2) >> i).max(1),
        ))
        .collect();

    // Build per-pass bind groups + uniform writes. Each downsample
    // reads the previous mip (or hdr_rt for the first) and writes
    // to the current mip. Each upsample reads mip i+1 and blends
    // additively into mip i.
    let bloom_filter_radius = 1.0_f32; // upsample tent radius

    // Downsample chain: mip 0 reads HDR, mips 1..N read previous mip.
    for i in 0..BLOOM_MIP_COUNT as usize {
        let (src_view, src_w, src_h, threshold_pass) = if i == 0 {
            (&self.hdr_rt_view, surf_w as f32, surf_h as f32, true)
        } else {
            let prev = &self.bloom_mip_views[i - 1];
            let (pw, ph) = mip_dims[i - 1];
            (prev, pw as f32, ph as f32, false)
        };

        let bp = BloomParams {
            params: [1.0 / src_w, 1.0 / src_h, bloom_filter_radius, 1.0],
        };
        self.queue.write_buffer(&self.bloom_uniform_buffer, 0, bytemuck::bytes_of(&bp));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_downsample_bg"),
            layout: &self.bloom_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bloom_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
            ],
        });

        let bloom_ts = profiler.pass_timestamp_writes("bloom_pass");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom_downsample_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.bloom_mip_views[i],
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: bloom_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pl = if threshold_pass {
            &self.bloom_pipeline_threshold_downsample
        } else {
            &self.bloom_pipeline_downsample
        };
        pass.set_pipeline(pl);
        // Force the viewport to this mip's actual size — wgpu's
        // auto-viewport derives from the surface config, not the
        // mip-view attachment, so without this the bloom pass
        // writes into a fraction of the mip and leaves the rest
        // uninitialized.
        let (mw, mh) = mip_dims[i];
        pass.set_viewport(0.0, 0.0, mw as f32, mh as f32, 0.0, 1.0);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    // Upsample chain: blend mip i+1 additively into mip i for
    // i = N-2..0. Final mip 0 ends up with the full bloom result.
    for i in (0..(BLOOM_MIP_COUNT as usize - 1)).rev() {
        let src_view = &self.bloom_mip_views[i + 1];
        let (sw, sh) = mip_dims[i + 1];

        let bp = BloomParams {
            params: [1.0 / sw as f32, 1.0 / sh as f32, bloom_filter_radius, 0.0],
        };
        self.queue.write_buffer(&self.bloom_uniform_buffer, 0, bytemuck::bytes_of(&bp));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_upsample_bg"),
            layout: &self.bloom_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bloom_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
            ],
        });

        let bloom_up_ts = profiler.pass_timestamp_writes("bloom_pass");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom_upsample_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.bloom_mip_views[i],
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // Load — additive blend on top of what
                    // downsample wrote.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: bloom_up_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.bloom_pipeline_upsample);
        // Same viewport fix as the downsample loop above — without
        // this the upsample tents only cover a sub-region of the
        // destination mip.
        let (mw, mh) = mip_dims[i];
        pass.set_viewport(0.0, 0.0, mw as f32, mh as f32, 0.0, 1.0);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    } // end if self.bloom_enabled
    }
}
