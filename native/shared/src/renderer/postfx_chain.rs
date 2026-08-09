//! Post-FX chain passes split from end_frame_with_scene (2000-line file
//! policy + render-graph migration prep). Starts with the bloom chain;
//! the rest of the tail (compose/upscale/TAA/DoF/blur/SSS) migrates here
//! cluster by cluster.

use super::*;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(super) enum CompositeSource {
    Hdr,
    Upscale,
    Taa0,
    Taa1,
    DepthOfField,
    MotionBlur,
    SubsurfaceScattering,
    ContrastAdaptiveSharpen,
}

impl CompositeSource {
    pub(super) const COUNT: usize = 8;

    pub(super) const fn bind_group_cache_index(self, exposure_idx: usize) -> usize {
        self as usize * 2 + exposure_idx
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(super) enum SsrCompositeSource {
    Fallback,
    History0,
    History1,
}

impl SsrCompositeSource {
    pub(super) const COUNT: usize = 3;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(super) enum PostFxSource {
    Hdr,
    Composed,
    Upscale,
    Taa0,
    Taa1,
    DepthOfField,
    MotionBlur,
    SubsurfaceScattering,
}

impl PostFxSource {
    pub(super) const COUNT: usize = 8;
}

#[inline]
fn bloom_threshold(auto_exposure: bool, manual_exposure: f32) -> f32 {
    if auto_exposure {
        2.5
    } else {
        2.5 / manual_exposure.max(0.05)
    }
}

#[inline]
fn taa_current_weight(history_valid: bool, frame_index: u32, render_scale: f32) -> f32 {
    if !history_valid || frame_index < 4 {
        1.0
    } else {
        let s2 = render_scale * render_scale;
        0.05 + 0.05 * s2
    }
}

fn reactive_taa_cache_key(plan_id: u64, rebuild_epoch: u64) -> (u64, u64) {
    (plan_id, rebuild_epoch)
}

#[inline]
fn exposure_update_rate(history_valid: bool, authored_rate: f32) -> f32 {
    if history_valid {
        authored_rate
    } else {
        -1.0
    }
}

#[inline]
fn bloom_mip_extent(width: u32, height: u32, mip_index: usize) -> (u32, u32) {
    (
        ((width / 2) >> mip_index).max(1),
        ((height / 2) >> mip_index).max(1),
    )
}

impl Renderer {
    /// Build the stable bloom pass bindings after the mip chain is created or
    /// resized. Each pass needs its own uniform buffer because all CPU writes
    /// happen before the shared command encoder is submitted.
    pub(super) fn rebuild_bloom_pass_resources(&mut self) {
        let downsample_count = BLOOM_MIP_COUNT as usize;
        let upsample_count = downsample_count.saturating_sub(1);
        let (render_width, render_height) = self.render_extent();
        let filter_radius = 1.0_f32;
        let threshold = bloom_threshold(self.auto_exposure, self.manual_exposure);

        // These texel sizes are immutable until the next resize. Initialize
        // the buffers directly rather than overwriting all eleven every
        // frame; reusing an in-flight Metal buffer can otherwise serialize
        // the resource scheduler.
        self.bloom_downsample_param_buffers = (0..downsample_count)
            .map(|i| {
                let (source_width, source_height) = if i == 0 {
                    (render_width, render_height)
                } else {
                    bloom_mip_extent(render_width, render_height, i - 1)
                };
                let params = BloomParams {
                    params: [
                        1.0 / source_width as f32,
                        1.0 / source_height as f32,
                        filter_radius,
                        threshold,
                    ],
                };
                let usage = if i == 0 {
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                } else {
                    wgpu::BufferUsages::UNIFORM
                };
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("bloom_downsample_params"),
                        contents: bytemuck::bytes_of(&params),
                        usage,
                    })
            })
            .collect();
        self.bloom_upsample_param_buffers = (0..upsample_count)
            .map(|i| {
                let (source_width, source_height) =
                    bloom_mip_extent(render_width, render_height, i + 1);
                let params = BloomParams {
                    params: [
                        1.0 / source_width as f32,
                        1.0 / source_height as f32,
                        filter_radius,
                        0.0,
                    ],
                };
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("bloom_upsample_params"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            })
            .collect();
        self.bloom_threshold_written = threshold;

        self.bloom_downsample_bind_groups = (0..downsample_count)
            .map(|i| {
                let src_view = if i == 0 {
                    &self.hdr_rt_view
                } else {
                    &self.bloom_mip_views[i - 1]
                };
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bloom_downsample_bg"),
                    layout: &self.bloom_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.bloom_downsample_param_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                    ],
                })
            })
            .collect();
        self.bloom_upsample_bind_groups = (0..upsample_count)
            .map(|i| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bloom_upsample_bg"),
                    layout: &self.bloom_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.bloom_upsample_param_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.bloom_mip_views[i + 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                    ],
                })
            })
            .collect();
    }
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
            let bloom_filter_radius = 1.0_f32; // upsample tent radius
                                               // Texel sizes are fixed until resize. Only the threshold in the
                                               // first pass can change at runtime, and only when exposure does.
            let threshold = bloom_threshold(self.auto_exposure, self.manual_exposure);
            if threshold.to_bits() != self.bloom_threshold_written.to_bits() {
                let params = BloomParams {
                    params: [
                        1.0 / surf_w as f32,
                        1.0 / surf_h as f32,
                        bloom_filter_radius,
                        threshold,
                    ],
                };
                self.queue.write_buffer(
                    &self.bloom_downsample_param_buffers[0],
                    0,
                    bytemuck::bytes_of(&params),
                );
                self.bloom_threshold_written = threshold;
            }

            // Downsample chain: mip 0 reads HDR, mips 1..N read previous mip.
            for i in 0..BLOOM_MIP_COUNT as usize {
                let threshold_pass = i == 0;

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
                let (mw, mh) = bloom_mip_extent(surf_w, surf_h, i);
                pass.set_viewport(0.0, 0.0, mw as f32, mh as f32, 0.0, 1.0);
                pass.set_bind_group(0, &self.bloom_downsample_bind_groups[i], &[]);
                pass.draw(0..3, 0..1);
            }

            // Upsample chain: blend mip i+1 additively into mip i for
            // i = N-2..0. Final mip 0 ends up with the full bloom result.
            for i in (0..(BLOOM_MIP_COUNT as usize - 1)).rev() {
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
                let (mw, mh) = bloom_mip_extent(surf_w, surf_h, i);
                pass.set_viewport(0.0, 0.0, mw as f32, mh as f32, 0.0, 1.0);
                pass.set_bind_group(0, &self.bloom_upsample_bind_groups[i], &[]);
                pass.draw(0..3, 0..1);
            }
        } // end if self.bloom_enabled
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bloom_mip_extent, bloom_threshold, exposure_update_rate, reactive_taa_cache_key,
        taa_current_weight, CompositeSource, PostFxSource, SsrCompositeSource,
    };

    #[test]
    fn bloom_mip_extent_matches_half_resolution_chain_and_never_reaches_zero() {
        assert_eq!(bloom_mip_extent(512, 256, 0), (256, 128));
        assert_eq!(bloom_mip_extent(512, 256, 3), (32, 16));
        assert_eq!(bloom_mip_extent(3, 1, 0), (1, 1));
        assert_eq!(bloom_mip_extent(3, 1, 8), (1, 1));
    }

    #[test]
    fn bloom_threshold_tracks_manual_exposure_and_clamps_black() {
        assert_eq!(bloom_threshold(false, 1.0), 2.5);
        assert_eq!(bloom_threshold(false, 2.0), 1.25);
        assert_eq!(bloom_threshold(false, 0.0), 50.0);
        assert_eq!(bloom_threshold(false, -1.0), 50.0);
    }

    #[test]
    fn bloom_threshold_stays_in_pre_exposed_units_for_auto_exposure() {
        assert_eq!(bloom_threshold(true, 0.0), 2.5);
        assert_eq!(bloom_threshold(true, 8.0), 2.5);
    }

    #[test]
    fn composite_cache_key_covers_every_source_and_exposure_slot() {
        let sources = [
            CompositeSource::Hdr,
            CompositeSource::Upscale,
            CompositeSource::Taa0,
            CompositeSource::Taa1,
            CompositeSource::DepthOfField,
            CompositeSource::MotionBlur,
            CompositeSource::SubsurfaceScattering,
            CompositeSource::ContrastAdaptiveSharpen,
        ];
        let mut keys = Vec::new();
        for source in sources {
            for exposure_idx in 0..2 {
                keys.push(source.bind_group_cache_index(exposure_idx));
            }
        }
        keys.sort_unstable();
        keys.dedup();

        assert_eq!(keys, (0..CompositeSource::COUNT * 2).collect::<Vec<_>>());
    }

    #[test]
    fn scene_compose_cache_has_one_slot_for_every_ssr_source() {
        let sources = [
            SsrCompositeSource::Fallback,
            SsrCompositeSource::History0,
            SsrCompositeSource::History1,
        ];
        let mut keys: Vec<_> = sources.into_iter().map(|source| source as usize).collect();
        keys.sort_unstable();
        keys.dedup();

        assert_eq!(keys, (0..SsrCompositeSource::COUNT).collect::<Vec<_>>());
    }

    #[test]
    fn reactive_taa_cache_key_includes_plan_and_pool_generation() {
        let key = reactive_taa_cache_key(41, 7);
        assert_ne!(key, reactive_taa_cache_key(42, 7));
        assert_ne!(key, reactive_taa_cache_key(41, 8));
    }

    #[test]
    fn postfx_source_keys_are_dense_and_unique() {
        let sources = [
            PostFxSource::Hdr,
            PostFxSource::Composed,
            PostFxSource::Upscale,
            PostFxSource::Taa0,
            PostFxSource::Taa1,
            PostFxSource::DepthOfField,
            PostFxSource::MotionBlur,
            PostFxSource::SubsurfaceScattering,
        ];
        let mut keys: Vec<_> = sources.into_iter().map(|source| source as usize).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys, (0..PostFxSource::COUNT).collect::<Vec<_>>());
    }

    #[test]
    fn invalid_taa_history_is_replaced_without_changing_steady_weights() {
        assert_eq!(taa_current_weight(false, 99, 0.5), 1.0);
        assert_eq!(taa_current_weight(true, 3, 1.0), 1.0);
        assert_eq!(taa_current_weight(true, 4, 0.5), 0.0625);
        assert_eq!(taa_current_weight(true, 4, 1.0), 0.1);
    }

    #[test]
    fn invalid_exposure_history_seeds_once_then_keeps_authored_adaptation() {
        assert_eq!(exposure_update_rate(false, 0.0), -1.0);
        assert_eq!(exposure_update_rate(false, 0.05), -1.0);
        assert_eq!(exposure_update_rate(true, 0.05), 0.05);
    }
}

impl Renderer {
    /// Scene compose: merge HDR + SSR + SSGI*albedo + bloom + fog + sun
    /// shafts into composed_rt. Runs unconditionally so the TAA-on path
    /// (TAA consumes this) and the TAA-off path (composite consumes it)
    /// see the same atmospherics.
    pub(super) fn record_scene_compose(&mut self, encoder: &mut wgpu::CommandEncoder) {
        // Composite input views (were locals in end_frame_with_scene).
        // PT-1: while path tracing, the SSR passes are skipped entirely,
        // so history is stale — route compose to ssr_rt, which the march
        // else-branch keeps cleared to transparent black.
        let ssr_composite_source = self.ssr_composite_source();
        // ============================================================
        // Scene-compose pass: merge HDR + SSR + SSGI*albedo + bloom
        // + fog + sun shafts into composed_rt. Runs unconditionally
        // so both the TAA-on path (TAA consumes this) and the
        // TAA-off path (composite consumes this) get the same
        // atmospherics + post-effects.
        // ============================================================
        // `mat4_invert` returns the CPU-side inverse transposed relative to
        // WGSL's `matrix * vector` convention. Upload the shader-facing form;
        // otherwise fog reconstructs a projectively distorted world position.
        let inv_vp_current = mat4_transpose(self.current_inv_vp_matrix);
        // Sun shaft screen-space position. Project a point far along
        // the sun direction through the current VP. If behind the
        // camera (clip.w ≤ 0), the sun is off-screen → disable.
        let sun_dir = self.lighting_uniforms.light_dir;
        let sun_world = [
            sun_dir[0] * 1000.0,
            sun_dir[1] * 1000.0,
            sun_dir[2] * 1000.0,
            1.0,
        ];
        let clip = mat4_mul_vec4(&self.current_vp_matrix, &sun_world);
        let (sun_uv, shaft_strength_eff) = if clip[3] > 0.0 {
            let ndc_x = clip[0] / clip[3];
            let ndc_y = clip[1] / clip[3];
            let u = ndc_x * 0.5 + 0.5;
            let v = 1.0 - (ndc_y * 0.5 + 0.5);
            // Allow off-screen suns to still cast shafts that streak
            // in from the edge — clamp to a small margin beyond ±[0,1]
            // rather than disabling outright.
            // `RangeInclusive::contains` also rejects NaN. Invalid projection
            // input therefore disables shafts instead of forwarding NaN UVs.
            let off = !(-1.0..=2.0).contains(&u) || !(-1.0..=2.0).contains(&v);
            if off {
                ([0.0, 0.0], 0.0)
            } else {
                ([u, v], self.sun_shaft_strength)
            }
        } else {
            ([0.0, 0.0], 0.0)
        };
        // When bloom_enabled is false we skip the downsample/upsample
        // chain entirely; forcing the composite's bloom multiplier to
        // 0 here means stale bloom_mip_views[0] contents contribute
        // nothing visually.
        let effective_bloom_intensity = if self.bloom_enabled {
            self.bloom_intensity
        } else {
            0.0
        };
        let cp = SceneComposeParams {
            // misc.y = procedural-sky aerial-perspective on/off flag.
            // The scene_compose shader reads this to decide between
            // the legacy 16-step fog march and the V2 3D LUT sample.
            misc: [
                effective_bloom_intensity,
                if self.procedural_sky_enabled {
                    1.0
                } else {
                    0.0
                },
                AERIAL_MAX_DIST_KM,
                0.0,
            ],
            inv_vp: inv_vp_current,
            fog_color_density: [
                self.fog_color[0],
                self.fog_color[1],
                self.fog_color[2],
                self.fog_density,
            ],
            fog_params: [self.fog_height_ref, self.fog_height_falloff, 0.0, 0.0],
            sun_shaft_uv_strength: [
                sun_uv[0],
                sun_uv[1],
                shaft_strength_eff,
                self.sun_shaft_decay,
            ],
            sun_shaft_color: [
                self.sun_shaft_color[0],
                self.sun_shaft_color[1],
                self.sun_shaft_color[2],
                0.0,
            ],
        };
        self.queue.write_buffer(
            &self.scene_compose_uniform_buffer,
            0,
            bytemuck::bytes_of(&cp),
        );
        {
            let cache_index = ssr_composite_source as usize;
            if self.scene_compose_bind_group_cache[cache_index].is_none() {
                self.frame_resource_stats
                    .created_bind_group(frame_resource_stats::BindGroupCreationSite::SceneCompose);
                let ssr_composite_view = self.ssr_composite_view_for(ssr_composite_source);
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scene_compose_bg"),
                    layout: &self.scene_compose_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.scene_compose_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(ssr_composite_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.ssgi_rt_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&self.bloom_mip_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: wgpu::BindingResource::TextureView(&self.depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler),
                        },
                        // EN-005 V2 — always bound; shader gates use on `misc.y`.
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: wgpu::BindingResource::TextureView(
                                &self.aerial_perspective_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: wgpu::BindingResource::Sampler(
                                &self.aerial_perspective_sampler,
                            ),
                        },
                    ],
                });
                self.scene_compose_bind_group_cache[cache_index] = Some(bg);
            }
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
            pass.set_bind_group(
                0,
                self.scene_compose_bind_group_cache[cache_index]
                    .as_ref()
                    .expect("scene-compose bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }
    }

    fn ssr_composite_source(&self) -> SsrCompositeSource {
        if self.ssr_enabled && !self.pt_owns_frame() {
            if self.ssr_history_idx == 0 {
                SsrCompositeSource::History0
            } else {
                SsrCompositeSource::History1
            }
        } else {
            SsrCompositeSource::Fallback
        }
    }

    fn ssr_composite_view_for(&self, source: SsrCompositeSource) -> &wgpu::TextureView {
        match source {
            SsrCompositeSource::Fallback => &self.ssr_rt_view,
            SsrCompositeSource::History0 => &self.ssr_history_views[0],
            SsrCompositeSource::History1 => &self.ssr_history_views[1],
        }
    }
}

impl Renderer {
    /// Post-FX tail: upscale (when sub-res and TAA is off), TAA, DoF,
    /// motion blur, SSS, and CAS — each stage reads the output of the
    /// last enabled stage before it. The internal `pre_*_view`
    /// selections encode that chain; `composite_source_view` re-derives
    /// the final link for the composite pass.
    pub(super) fn record_postfx_tail(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
    ) {
        // ============================================================
        // Upscale pass: render-res composed_rt → full-surface upscale_rt.
        // Engages only when render_scale < 1.0 AND TAA is off — when
        // TAA runs it does its own Catmull-Rom upscale. Downstream
        // post-FX (DoF/MB/SSS/composite) read upscale_rt instead of
        // hdr_rt in this case so the chain operates at full surface
        // resolution.
        // ============================================================
        if self.render_scale < 0.999 && !self.taa_enabled {
            let up = UpscaleParams {
                params: [self.upscale_mode as f32, 0.0, 0.0, 0.0],
            };
            self.queue
                .write_buffer(&self.upscale_uniform_buffer, 0, bytemuck::bytes_of(&up));
            if self.upscale_bind_group_cache.is_none() {
                self.frame_resource_stats
                    .created_bind_group(frame_resource_stats::BindGroupCreationSite::Upscale);
                self.upscale_bind_group_cache =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("upscale_bg"),
                        layout: &self.upscale_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.upscale_uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.composed_rt_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                        ],
                    }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("upscale_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.upscale_rt_view,
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
            pass.set_pipeline(&self.upscale_pipeline);
            pass.set_bind_group(
                0,
                self.upscale_bind_group_cache
                    .as_ref()
                    .expect("upscale bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // TAA pass: reprojection + neighborhood clamp on composed_rt.
        // Skipped when TAA is off — composite reads composed_rt
        // directly and gets the same composed / fog / shafts output.
        // ============================================================
        if self.taa_enabled {
            let pt_owned = self.pt_owns_frame();
            if self.taa_history_pt_owned != pt_owned {
                // Progressive PT can begin or stop owning frames without a
                // mode change. Never blend across the raster/PT radiance seam.
                self.taa_history_pt_owned = pt_owned;
                self.taa_history_valid = false;
                self.taa_current_idx = 0;
            }
        }
        let taa_dst_idx = self.taa_current_idx;
        let taa_src_idx = 1 - self.taa_current_idx;

        if self.taa_enabled {
            // Effective temporal window scales with per-pixel sample
            // density (~render_scale²). At 0.5 → 0.0625 (~16-frame
            // window, close to the prior 0.05/20-frame); at 1.0 →
            // 0.10 (~10-frame), matching native TAA.
            let alpha = taa_current_weight(
                self.taa_history_valid,
                self.taa_frame_index,
                self.render_scale,
            );
            // yz = the current jitter as a composed-UV offset. Content shifts by
            // -jitter_ndc through the GL-convention perspective divide (w = -z),
            // so the rendered position of a feature is uv + (-0.5*jx, +0.5*jy)
            // (v axis flips). Empirically arbitrated by the variance rig: the
            // wrong sign DOUBLES the effective jitter and the numbers scream.
            let tp = TaaParams {
                params: [
                    alpha,
                    -0.5 * self.current_jitter_ndc[0],
                    0.5 * self.current_jitter_ndc[1],
                    if self.current_proj_matrix[3][3].abs() < 0.5 {
                        1.0
                    } else {
                        0.0
                    },
                ],
                // Match the already-qualified path-tracing inverse upload.
                // The raw CPU inverse makes world.xyz survive the homogeneous
                // divide by accident, but destroys world_h.w and therefore
                // writes zero into geometric depth history on perspective
                // cameras, rejecting every history sample on the next frame.
                inv_vp: mat4_transpose(self.current_inv_vp_matrix),
                prev_vp: self.prev_vp_matrix,
            };
            self.queue
                .write_buffer(&self.taa_uniform_buffer, 0, bytemuck::bytes_of(&tp));

            if self.temporal_reactive_active {
                self.ensure_taa_reactive_resources();
            }
            let bg = if self.temporal_reactive_active {
                let (plan_id, reactive) = {
                    let plan = self
                        .last_frame_plan
                        .as_ref()
                        .expect("reactive TAA has an active frame plan");
                    (
                        plan.plan_id,
                        plan.resource("transparency-reactive")
                            .expect("reactive topology declares its coverage input")
                            .id,
                    )
                };
                let cache_key = reactive_taa_cache_key(plan_id, self.transient_pool.rebuild_epoch);
                if self.taa_reactive_bind_group_cache_keys[taa_src_idx] != Some(cache_key) {
                    let reactive_view = self
                        .transient_pool
                        .compiled_view(plan_id, reactive)
                        .expect("reactive coverage is materialized before post-FX");
                    self.frame_resource_stats.created_bind_group(
                        frame_resource_stats::BindGroupCreationSite::TaaReactive,
                    );
                    self.taa_reactive_bind_group_cache[taa_src_idx] = Some(
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("taa_reactive_bg"),
                            layout: self
                                .taa_reactive_layout
                                .as_ref()
                                .expect("reactive TAA layout initialized"),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.taa_uniform_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.composed_rt_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.composite_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.taa_views[taa_src_idx],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.composite_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.ssao_depth_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.velocity_rt_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 8,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.composite_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 9,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.taa_depth_history_views[taa_src_idx],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 10,
                                    resource: wgpu::BindingResource::TextureView(reactive_view),
                                },
                            ],
                        }),
                    );
                    self.taa_reactive_bind_group_cache_keys[taa_src_idx] = Some(cache_key);
                }
                self.taa_reactive_bind_group_cache[taa_src_idx]
                    .as_ref()
                    .expect("reactive TAA bind group was initialized")
                    .clone()
            } else {
                if self.taa_bind_group_cache[taa_src_idx].is_none() {
                    self.frame_resource_stats
                        .created_bind_group(frame_resource_stats::BindGroupCreationSite::Taa);
                    self.taa_bind_group_cache[taa_src_idx] =
                        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("taa_bg"),
                            layout: &self.taa_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.taa_uniform_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.composed_rt_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.composite_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.taa_views[taa_src_idx],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.composite_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.ssao_depth_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.velocity_rt_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 8,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.composite_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 9,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.taa_depth_history_views[taa_src_idx],
                                    ),
                                },
                            ],
                        }));
                }
                self.taa_bind_group_cache[taa_src_idx]
                    .as_ref()
                    .expect("ordinary TAA bind group was initialized")
                    .clone()
            };
            let taa_ts = profiler.pass_timestamp_writes("taa_pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.taa_views[taa_dst_idx],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.taa_depth_history_views[taa_dst_idx],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 10_000.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: taa_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.temporal_reactive_active {
                pass.set_pipeline(
                    self.taa_reactive_pipeline
                        .as_ref()
                        .expect("reactive TAA pipeline initialized"),
                );
            } else {
                pass.set_pipeline(&self.taa_pipeline);
            }
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            #[cfg(not(target_arch = "wasm32"))]
            if self.pending_quality_capture_dir.is_some() {
                self.record_taa_diagnostics(encoder, &bg, self.temporal_reactive_active);
            }
            self.taa_history_valid = true;
            self.taa_history_written = true;
        }

        // ============================================================
        // DoF pass: variable-radius Poisson disc blur driven by CoC
        // Reads TAA output / upscale_rt / hdr_rt + depth → dof_rt
        // ============================================================
        let pre_dof_source = if self.taa_enabled {
            if taa_dst_idx == 0 {
                PostFxSource::Taa0
            } else {
                PostFxSource::Taa1
            }
        } else if self.render_scale < 0.999 {
            PostFxSource::Upscale
        } else {
            PostFxSource::Hdr
        };

        if self.dof_enabled && self.dof_aperture > 0.0 {
            let inv_proj = self.current_inv_proj_matrix;
            let dp = DofParams {
                params: [
                    self.dof_focus_distance,
                    self.dof_aperture,
                    self.dof_max_blur,
                    0.0,
                ],
                inv_proj,
            };
            self.queue
                .write_buffer(&self.dof_uniform_buffer, 0, bytemuck::bytes_of(&dp));

            let cache_index = pre_dof_source as usize;
            if self.dof_bind_group_cache[cache_index].is_none() {
                self.frame_resource_stats
                    .created_bind_group(frame_resource_stats::BindGroupCreationSite::DepthOfField);
                let pre_dof_view = self.postfx_source_view_for(pre_dof_source);
                self.dof_bind_group_cache[cache_index] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("dof_bg"),
                        layout: &self.dof_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.dof_uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(pre_dof_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler),
                            },
                        ],
                    }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dof_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.dof_rt_view,
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
            pass.set_pipeline(&self.dof_pipeline);
            pass.set_bind_group(
                0,
                self.dof_bind_group_cache[cache_index]
                    .as_ref()
                    .expect("depth-of-field bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // Motion blur pass: 8-tap directional blur along velocity
        // Reads upstream color + velocity_rt → motion_blur_rt
        // ============================================================
        let pre_mblur_source = if self.dof_enabled && self.dof_aperture > 0.0 {
            PostFxSource::DepthOfField
        } else if self.taa_enabled {
            if taa_dst_idx == 0 {
                PostFxSource::Taa0
            } else {
                PostFxSource::Taa1
            }
        } else if self.render_scale < 0.999 {
            PostFxSource::Upscale
        } else {
            PostFxSource::Hdr
        };

        if self.motion_blur_enabled && self.motion_blur_strength > 0.0 {
            let mbp = MotionBlurParams {
                params: [
                    self.motion_blur_strength,
                    self.motion_blur_max_blur,
                    0.0,
                    0.0,
                ],
            };
            self.queue.write_buffer(
                &self.motion_blur_uniform_buffer,
                0,
                bytemuck::bytes_of(&mbp),
            );

            let cache_index = pre_mblur_source as usize;
            if self.motion_blur_bind_group_cache[cache_index].is_none() {
                self.frame_resource_stats
                    .created_bind_group(frame_resource_stats::BindGroupCreationSite::MotionBlur);
                let pre_mblur_view = self.postfx_source_view_for(pre_mblur_source);
                self.motion_blur_bind_group_cache[cache_index] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("motion_blur_bg"),
                        layout: &self.motion_blur_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.motion_blur_uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(pre_mblur_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.velocity_rt_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                        ],
                    }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.motion_blur_rt_view,
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
            pass.set_pipeline(&self.motion_blur_pipeline);
            pass.set_bind_group(
                0,
                self.motion_blur_bind_group_cache[cache_index]
                    .as_ref()
                    .expect("motion-blur bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // SSS pass: chromatic disc blur (skin / wax / leaves)
        // Reads upstream color + depth → sss_rt.
        // Runs after motion blur so it applies to the fully composited
        // motion state, not to individual geometry.
        // ============================================================
        let pre_sss_source = if self.motion_blur_enabled && self.motion_blur_strength > 0.0 {
            PostFxSource::MotionBlur
        } else if self.dof_enabled && self.dof_aperture > 0.0 {
            PostFxSource::DepthOfField
        } else if self.taa_enabled {
            if taa_dst_idx == 0 {
                PostFxSource::Taa0
            } else {
                PostFxSource::Taa1
            }
        } else if self.render_scale < 0.999 {
            PostFxSource::Upscale
        } else {
            PostFxSource::Hdr
        };

        if self.sss_enabled && self.sss_strength > 0.0 {
            let sp = SssParams {
                params: [self.sss_strength, self.sss_width, 500.0, 0.0],
            };
            self.queue
                .write_buffer(&self.sss_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            let cache_index = pre_sss_source as usize;
            if self.sss_bind_group_cache[cache_index].is_none() {
                self.frame_resource_stats.created_bind_group(
                    frame_resource_stats::BindGroupCreationSite::SubsurfaceScattering,
                );
                let pre_sss_view = self.postfx_source_view_for(pre_sss_source);
                self.sss_bind_group_cache[cache_index] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sss_bg"),
                        layout: &self.sss_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.sss_uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(pre_sss_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler),
                            },
                        ],
                    }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sss_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.sss_rt_view,
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
            pass.set_pipeline(&self.sss_pipeline);
            pass.set_bind_group(
                0,
                self.sss_bind_group_cache[cache_index]
                    .as_ref()
                    .expect("subsurface-scattering bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }

        // ============================================================
        // RCAS sharpen pass: contrast-adaptive 5-tap cross. Reads the
        // same texture composite would otherwise sample (sss/mb/dof/
        // taa/upscale/composed) and writes cas_rt. Off by default —
        // gated on cas_strength > 0.
        // ============================================================
        let cas_input_source = if self.sss_enabled && self.sss_strength > 0.0 {
            PostFxSource::SubsurfaceScattering
        } else if self.motion_blur_enabled && self.motion_blur_strength > 0.0 {
            PostFxSource::MotionBlur
        } else if self.dof_enabled && self.dof_aperture > 0.0 {
            PostFxSource::DepthOfField
        } else if self.taa_enabled {
            if taa_dst_idx == 0 {
                PostFxSource::Taa0
            } else {
                PostFxSource::Taa1
            }
        } else if self.render_scale < 0.999 {
            PostFxSource::Upscale
        } else {
            // TAA off, native res: composed_rt is already full-surface
            // and carries SSR / SSGI / bloom / fog / shafts.
            PostFxSource::Composed
        };

        if self.cas_strength > 0.0 {
            let cp = RcasParams {
                params: [self.cas_strength, 0.0, 0.0, 0.0],
            };
            self.queue
                .write_buffer(&self.cas_uniform_buffer, 0, bytemuck::bytes_of(&cp));
            let cache_index = cas_input_source as usize;
            if self.cas_bind_group_cache[cache_index].is_none() {
                self.frame_resource_stats.created_bind_group(
                    frame_resource_stats::BindGroupCreationSite::ContrastAdaptiveSharpen,
                );
                let cas_input_view = self.postfx_source_view_for(cas_input_source);
                self.cas_bind_group_cache[cache_index] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("cas_bg"),
                        layout: &self.cas_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.cas_uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(cas_input_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                        ],
                    }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cas_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.cas_rt_view,
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
            pass.set_pipeline(&self.cas_pipeline);
            pass.set_bind_group(
                0,
                self.cas_bind_group_cache[cache_index]
                    .as_ref()
                    .expect("contrast-adaptive-sharpen bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }
    }

    /// The view the composite pass reads: the output of the LAST enabled
    /// stage in chain order CAS > SSS > motion blur > DoF > TAA >
    /// upscale > raw HDR. Must mirror the `pre_*_view` cascade in
    /// record_postfx_tail.
    pub(super) fn composite_source(&self) -> CompositeSource {
        if self.cas_strength > 0.0 {
            CompositeSource::ContrastAdaptiveSharpen
        } else if self.sss_enabled && self.sss_strength > 0.0 {
            CompositeSource::SubsurfaceScattering
        } else if self.motion_blur_enabled && self.motion_blur_strength > 0.0 {
            CompositeSource::MotionBlur
        } else if self.dof_enabled && self.dof_aperture > 0.0 {
            CompositeSource::DepthOfField
        } else if self.taa_enabled {
            if self.taa_current_idx == 0 {
                CompositeSource::Taa0
            } else {
                CompositeSource::Taa1
            }
        } else if self.render_scale < 0.999 {
            CompositeSource::Upscale
        } else {
            CompositeSource::Hdr
        }
    }

    pub(super) fn composite_source_view_for(&self, source: CompositeSource) -> &wgpu::TextureView {
        match source {
            CompositeSource::Hdr => &self.hdr_rt_view,
            CompositeSource::Upscale => &self.upscale_rt_view,
            CompositeSource::Taa0 => &self.taa_views[0],
            CompositeSource::Taa1 => &self.taa_views[1],
            CompositeSource::DepthOfField => &self.dof_rt_view,
            CompositeSource::MotionBlur => &self.motion_blur_rt_view,
            CompositeSource::SubsurfaceScattering => &self.sss_rt_view,
            CompositeSource::ContrastAdaptiveSharpen => &self.cas_rt_view,
        }
    }

    fn postfx_source_view_for(&self, source: PostFxSource) -> &wgpu::TextureView {
        match source {
            PostFxSource::Hdr => &self.hdr_rt_view,
            PostFxSource::Composed => &self.composed_rt_view,
            PostFxSource::Upscale => &self.upscale_rt_view,
            PostFxSource::Taa0 => &self.taa_views[0],
            PostFxSource::Taa1 => &self.taa_views[1],
            PostFxSource::DepthOfField => &self.dof_rt_view,
            PostFxSource::MotionBlur => &self.motion_blur_rt_view,
            PostFxSource::SubsurfaceScattering => &self.sss_rt_view,
        }
    }
}

impl Renderer {
    /// Manual exposure multiplier used while auto-exposure is disabled.
    pub fn set_manual_exposure(&mut self, value: f32) {
        self.manual_exposure = value.max(0.0);
    }

    /// Toggle auto-exposure. The first enabled frame measures and seeds the
    /// current scene instead of adapting from a value frozen while disabled.
    pub fn set_auto_exposure(&mut self, enabled: bool) {
        if self.auto_exposure != enabled {
            self.auto_exposure = enabled;
            self.exposure_current_idx = 0;
            self.exposure_history_valid = false;
            self.exposure_history_written = false;
        }
    }

    /// Auto-exposure target scene key (average luminance to drive toward).
    pub fn set_auto_exposure_key(&mut self, key: f32) {
        self.auto_exposure_key = key.clamp(0.01, 1.0);
    }

    /// Auto-exposure smoothing rate per frame. Invalid history always seeds
    /// once; this authored rate applies to all subsequent adaptation.
    pub fn set_auto_exposure_rate(&mut self, rate: f32) {
        self.auto_exposure_rate = rate.clamp(0.0, 1.0);
    }

    /// Auto-exposure measure + adapt pass into the dst slot of the
    /// ping-pong exposure texture. No-op when auto_exposure is off (the
    /// composite keeps reading the stale texture, which manual_exposure
    /// bypasses). The caller owns the src/dst indices because the
    /// composite binds the same dst view.
    pub(super) fn record_auto_exposure(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        exposure_src_idx: usize,
        exposure_dst_idx: usize,
    ) {
        // The luminance source is whatever the composite will read.
        if self.auto_exposure {
            let composite_source = self.composite_source();
            let cache_index = composite_source.bind_group_cache_index(exposure_src_idx);
            let ep = ExposureParams {
                params: [
                    self.auto_exposure_key,
                    exposure_update_rate(self.exposure_history_valid, self.auto_exposure_rate),
                    // Wide clamp — without SSGI, Sponza's shadowed
                    // corridors have ~7× less average luma than its
                    // sunlit courtyard, so exposure needs to span
                    // the same range to keep perceived brightness
                    // stable across rotations.
                    0.1,
                    10.0,
                ],
            };
            self.queue
                .write_buffer(&self.exposure_uniform_buffer, 0, bytemuck::bytes_of(&ep));

            if self.exposure_bind_group_cache[cache_index].is_none() {
                self.frame_resource_stats
                    .created_bind_group(frame_resource_stats::BindGroupCreationSite::AutoExposure);
                let composite_src_view = self.composite_source_view_for(composite_source);
                self.exposure_bind_group_cache[cache_index] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("exposure_bg"),
                        layout: &self.exposure_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.exposure_uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(composite_src_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.exposure_views[exposure_src_idx],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                        ],
                    }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("exposure_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.exposure_views[exposure_dst_idx],
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
            pass.set_pipeline(&self.exposure_pipeline);
            pass.set_bind_group(
                0,
                self.exposure_bind_group_cache[cache_index]
                    .as_ref()
                    .expect("auto-exposure bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
            drop(pass);
            self.exposure_history_valid = true;
            self.exposure_history_written = true;
        }
    }
}
