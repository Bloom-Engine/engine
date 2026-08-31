//! Screen-space reflections: stochastic ray march + temporal denoiser.
//! Split from end_frame_with_scene (2000-line file policy + render-graph
//! migration prep). Both entry points no-op when `ssr_enabled` is false.

use super::{Renderer, SsrParams, SsrTemporalParams};

fn ssr_temporal_alpha(history_valid: bool) -> f32 {
    if history_valid {
        0.1
    } else {
        1.0
    }
}

impl Renderer {
    /// Toggle SSR on/off. SSR contributes nothing in scenes with
    /// no on-screen geometry to reflect (e.g., single object
    /// against sky) — turning it off there saves a fullscreen pass.
    pub fn set_ssr_enabled(&mut self, enabled: bool) {
        if self.ssr_enabled != enabled {
            self.ssr_enabled = enabled;
            self.ssr_history_idx = 0;
            self.ssr_history_valid = false;
        }
    }

    /// SSR strength multiplier (0 = off, 0.5 = default, 1+ = strong).
    /// Changing the radiance multiplier invalidates incompatible history.
    pub fn set_ssr_strength(&mut self, strength: f32) {
        let strength = strength.max(0.0);
        if self.ssr_strength != strength {
            self.ssr_strength = strength;
            self.ssr_history_idx = 0;
            self.ssr_history_valid = false;
        }
    }

    fn create_ssr_bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        iridescence_metadata: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        // EN-021 — SSR misses must sample the same environment that owns the
        // scene's split-sum IBL. Procedural-sky mode previously kept binding
        // `sky_texture` (the last panorama loaded), so hit/miss ownership
        // changes during camera motion also changed the reflected world and
        // appeared as a moving bright patch on rough floors.
        let panorama_or_default_view;
        let env_view = if self.lighting_bg_is_procedural {
            &self.procedural_sky_equirect_full_view
        } else {
            panorama_or_default_view = self
                .sky_texture
                .as_ref()
                .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
                .unwrap_or_else(|| {
                    self._scene_env_default_texture
                        .create_view(&wgpu::TextureViewDescriptor::default())
                });
            &panorama_or_default_view
        };
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.ssr_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&self.depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&self.material_rt_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(env_view),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
            },
        ];
        if let Some(metadata) = iridescence_metadata {
            entries.push(wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::TextureView(metadata),
            });
        }
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

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
        if self.ssr_enabled && !self.pt_owns_frame() {
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
            self.queue
                .write_buffer(&self.ssr_uniform_buffer, 0, bytemuck::bytes_of(&sp));

            if self.iridescence_ssr_active {
                if self.ssr_layered_bg_cache.is_none() {
                    let resources = self
                        .scene_iridescence_ssr_resources
                        .as_ref()
                        .expect("active iridescence SSR has lazy resources");
                    self.ssr_layered_bg_cache = Some(self.create_ssr_bind_group(
                        "iridescence_ssr_bg",
                        &resources.ssr_layout,
                        Some(&resources.metadata_view),
                    ));
                }
            } else if self.ssr_bg_cache.is_none() {
                self.ssr_bg_cache =
                    Some(self.create_ssr_bind_group("ssr_bg", &self.ssr_layout, None));
            }
            let (pipeline, bg) = if self.iridescence_ssr_active {
                let resources = self
                    .scene_iridescence_ssr_resources
                    .as_ref()
                    .expect("active iridescence SSR has lazy resources");
                (
                    &resources.ssr_pipeline,
                    self.ssr_layered_bg_cache
                        .as_ref()
                        .expect("layered SSR bind group initialized"),
                )
            } else {
                (
                    &self.ssr_pipeline,
                    self.ssr_bg_cache
                        .as_ref()
                        .expect("base SSR bind group initialized"),
                )
            };
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
            pass.set_pipeline(pipeline);
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
    pub(super) fn record_ssr_temporal(&mut self, encoder: &mut wgpu::CommandEncoder) {
        // ============================================================
        // SSR temporal denoiser: blend the noisy single-ray SSR with
        // the reprojected previous history so 4–8 frames of GGX-sampled
        // rays converge to a smooth reflection. 3×3 pre-filter of the
        // noisy current frame + neighborhood clamp of reprojected
        // history. Compose then reads ssr_history[cur] instead of
        // ssr_rt.
        // ============================================================
        // PT-1: same gate as the march — no fresh rays, nothing to blend.
        if self.ssr_enabled && !self.pt_owns_frame() {
            let prev_idx = 1 - self.ssr_history_idx;
            let cur_idx = self.ssr_history_idx;

            // Explicit validity owns initialization. TAA's frame counter is
            // unrelated to SSR lifetime: SSR can be toggled, resized, or
            // suspended by PT long after TAA frame zero.
            let alpha = ssr_temporal_alpha(self.ssr_history_valid);
            let tp = SsrTemporalParams {
                params: [
                    alpha,
                    if self.current_proj_matrix[3][3].abs() < 0.5 {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                ],
                inv_vp: super::mat4_transpose(self.current_inv_vp_matrix),
                prev_vp: self.prev_vp_matrix,
            };
            self.queue.write_buffer(
                &self.ssr_temporal_uniform_buffer,
                0,
                bytemuck::bytes_of(&tp),
            );

            if self.ssr_temporal_bind_group_cache[prev_idx].is_none() {
                self.frame_resource_stats.created_bind_group(
                    super::frame_resource_stats::BindGroupCreationSite::SsrTemporal,
                );
                self.ssr_temporal_bind_group_cache[prev_idx] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ssr_temporal_bg"),
                        layout: &self.ssr_temporal_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.ssr_temporal_uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.ssr_rt_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.ssr_history_views[prev_idx],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.velocity_rt_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(&self.depth_view),
                            },
                        ],
                    }));
            }
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
            pass.set_bind_group(
                0,
                self.ssr_temporal_bind_group_cache[prev_idx]
                    .as_ref()
                    .expect("SSR temporal bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
            drop(pass);
            #[cfg(not(target_arch = "wasm32"))]
            if self.pending_quality_capture_dir.is_some() {
                let diagnostic_bg = self.ssr_temporal_bind_group_cache[prev_idx]
                    .as_ref()
                    .expect("SSR temporal bind group was initialized")
                    .clone();
                self.record_ssr_temporal_diagnostics(encoder, &diagnostic_bg);
            }
            self.ssr_history_valid = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ssr_temporal_alpha;

    #[test]
    fn invalid_ssr_history_is_replaced_before_temporal_blending() {
        assert_eq!(ssr_temporal_alpha(false), 1.0);
        assert_eq!(ssr_temporal_alpha(true), 0.1);
    }
}
