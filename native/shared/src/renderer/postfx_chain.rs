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

impl Renderer {
    /// Scene compose: merge HDR + SSR + SSGI*albedo + bloom + fog + sun
    /// shafts into composed_rt. Runs unconditionally so the TAA-on path
    /// (TAA consumes this) and the TAA-off path (composite consumes it)
    /// see the same atmospherics.
    pub(super) fn record_scene_compose(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Composite input views (were locals in end_frame_with_scene).
        let ssr_composite_view = if self.ssr_enabled {
            &self.ssr_history_views[self.ssr_history_idx]
        } else {
            &self.ssr_rt_view
        };
        let ssgi_composite_view = &self.ssgi_rt_view;
    // ============================================================
    // Scene-compose pass: merge HDR + SSR + SSGI*albedo + bloom
    // + fog + sun shafts into composed_rt. Runs unconditionally
    // so both the TAA-on path (TAA consumes this) and the
    // TAA-off path (composite consumes this) get the same
    // atmospherics + post-effects.
    // ============================================================
    let inv_vp_current = self.current_inv_vp_matrix;
    // Sun shaft screen-space position. Project a point far along
    // the sun direction through the current VP. If behind the
    // camera (clip.w ≤ 0), the sun is off-screen → disable.
    let sun_dir = self.lighting_uniforms.light_dir;
    let sun_world = [sun_dir[0] * 1000.0, sun_dir[1] * 1000.0, sun_dir[2] * 1000.0, 1.0];
    let clip = mat4_mul_vec4(&self.current_vp_matrix, &sun_world);
    let (sun_uv, shaft_strength_eff) = if clip[3] > 0.0 {
        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];
        let u = ndc_x * 0.5 + 0.5;
        let v = 1.0 - (ndc_y * 0.5 + 0.5);
        // Allow off-screen suns to still cast shafts that streak
        // in from the edge — clamp to a small margin beyond ±[0,1]
        // rather than disabling outright.
        let off = u < -1.0 || u > 2.0 || v < -1.0 || v > 2.0;
        if off { ([0.0, 0.0], 0.0) } else { ([u, v], self.sun_shaft_strength) }
    } else {
        ([0.0, 0.0], 0.0)
    };
    // When bloom_enabled is false we skip the downsample/upsample
    // chain entirely; forcing the composite's bloom multiplier to
    // 0 here means stale bloom_mip_views[0] contents contribute
    // nothing visually.
    let effective_bloom_intensity = if self.bloom_enabled { self.bloom_intensity } else { 0.0 };
    let cp = SceneComposeParams {
        // misc.y = procedural-sky aerial-perspective on/off flag.
        // The scene_compose shader reads this to decide between
        // the legacy 16-step fog march and the V2 3D LUT sample.
        misc: [
            effective_bloom_intensity,
            if self.procedural_sky_enabled { 1.0 } else { 0.0 },
            AERIAL_MAX_DIST_KM,
            0.0,
        ],
        inv_vp: inv_vp_current,
        fog_color_density: [
            self.fog_color[0], self.fog_color[1], self.fog_color[2], self.fog_density,
        ],
        fog_params: [self.fog_height_ref, self.fog_height_falloff, 0.0, 0.0],
        sun_shaft_uv_strength: [
            sun_uv[0], sun_uv[1], shaft_strength_eff, self.sun_shaft_decay,
        ],
        sun_shaft_color: [
            self.sun_shaft_color[0], self.sun_shaft_color[1], self.sun_shaft_color[2], 0.0,
        ],
    };
    self.queue.write_buffer(&self.scene_compose_uniform_buffer, 0, bytemuck::bytes_of(&cp));
    {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_compose_bg"),
            layout: &self.scene_compose_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.scene_compose_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(ssr_composite_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(ssgi_composite_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.bloom_mip_views[0]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.composite_sampler) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                // EN-005 V2 — always bound; shader gates use on `misc.y`.
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&self.aerial_perspective_view) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::Sampler(&self.aerial_perspective_sampler) },
            ],
        });
        // NOTE: GPU timestamp deliberately not requested on this pass.
        // Empirically (sponza, Metal) the reported delta was ~249 ms
        // for what should be a sub-millisecond fullscreen pass. Likely
        // the end-of-pass write is synchronized to a later barrier
        // and includes idle time. CPU-side timing via the enclosing
        // `post_fx` phase captures the cost adequately.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene_compose_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.composed_rt_view,
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
        pass.set_pipeline(&self.scene_compose_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    }
}
