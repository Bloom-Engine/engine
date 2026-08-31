//! Lumen-style screen-probe SSGI: probe placement, Hi-Z/HW/SDF trace,
//! temporal phase accumulation, and octahedral resolve into ssgi_rt. Split from
//! end_frame_with_scene (2000-line file policy + render-graph migration
//! prep). When disabled, clears ssgi_rt to transparent.

use super::*;

// Average a finite ring of sixteen complete diffuse estimates while the
// SSGI-owned angular sequence rotates its 32 directions. This preserves 512
// effective samples without additional ray queries and becomes stationary
// after one cycle; surface rejection prevents stale transport at disocclusion.
const SSGI_TEMPORAL_PHASE_WEIGHT: f32 = 0.0625;
const SSGI_TEMPORAL_OUTPUT_WEIGHT: f32 = 0.125;

fn probe_inverse_view_for_wgsl(view: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    mat4_transpose(mat4_invert(view))
}

impl Renderer {
    /// Toggle SSGI (screen-space global illumination) on/off. Off means no
    /// probe work; either transition invalidates radiance from the old route.
    pub fn set_ssgi_enabled(&mut self, enabled: bool) {
        if self.ssgi_enabled != enabled {
            self.ssgi_enabled = enabled;
            self.probe_history_idx = 0;
            self.probe_frame_index = 0;
            self.probe_history_valid = false;
        }
    }

    /// SSGI intensity multiplier (0 = off, 0.5 = default, 1+ = strong).
    /// Controls the brightness of indirect bounce light.
    pub fn set_ssgi_intensity(&mut self, intensity: f32) {
        let intensity = intensity.max(0.0);
        if self.ssgi_intensity != intensity {
            self.ssgi_intensity = intensity;
            self.probe_history_idx = 0;
            self.probe_frame_index = 0;
            self.probe_history_valid = false;
        }
    }

    /// SSGI max march distance in view-space meters (default 20).
    /// Tune to the scene scale: small for tight rooms, large for
    /// open-world interiors.
    pub fn set_ssgi_radius(&mut self, radius: f32) {
        let radius = radius.max(0.1);
        if self.ssgi_radius != radius {
            self.ssgi_radius = radius;
            self.probe_history_idx = 0;
            self.probe_frame_index = 0;
            self.probe_history_valid = false;
        }
    }

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
        // safe either way). pt_owns_frame, not pt_active: progressive mode
        // shows raster frames until accumulation warms up, and those still
        // want SSGI.
        if self.ssgi_enabled && !self.pt_owns_frame() {
            // Coherent card lighting fades in over ~48 frames once the bake
            // is complete, and drops out immediately if streaming resumes.
            self.card_light_coherent_ramp = if self.card_light_coherent {
                (self.card_light_coherent_ramp + 1.0 / 48.0).min(1.0)
            } else {
                0.0
            };
            let p00 = self.current_proj_matrix[0][0];
            let p11 = self.current_proj_matrix[1][1];
            let p20 = self.current_proj_matrix[2][0];
            let p21 = self.current_proj_matrix[2][1];
            let camera_moving = postfx_chain::taa_camera_moving(
                &self.current_view_matrix,
                &self.prev_view_matrix,
                &self.current_proj_matrix_unjittered,
                &self.prev_proj_matrix_unjittered,
            );
            // The dense 8x8 lattice already supplies four receiver sites in
            // each former probe footprint. Keep those sites centered during
            // motion: shifting the whole screen lattice every frame changes
            // the measured world receivers and exposes the sampling pattern
            // that temporal resolve is trying to anchor to the scene.
            let probe_lattice_jitter_active = false;
            // `mat4_invert` lands transposed relative to WGSL's `M * v`
            // convention (the path tracer has the same upload boundary).
            // Uploading it raw mirrored screen-probe world positions across
            // the camera: right-hand facade probes traced from the distant
            // red awnings, then resolve projected that radiance back onto the
            // facade. Convert once here; place, HW/SDF trace and resolve all
            // share this exact uniform value.
            let inv_view = probe_inverse_view_for_wgsl(self.current_view_matrix);

            // ---- place ----
            let place_params = ProbePlaceParams {
                inv_view,
                proj_row01: [p00, p11, p20, p21],
                size: [half_w, half_h, gw, gh],
                params: [
                    (self.probe_frame_index & 4095) as f32,
                    PROBE_TILE_SIZE as f32,
                    if probe_lattice_jitter_active {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                ],
            };
            self.queue.write_buffer(
                &self.probe_place_uniform,
                0,
                bytemuck::bytes_of(&place_params),
            );
            if self.probe_place_bg_cache.is_none() {
                self.probe_place_bg_cache =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("probe_place_bg"),
                        layout: &self.probe_place_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.probe_place_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.probe_header_buffer.as_entire_binding(),
                            },
                        ],
                    }));
            }
            {
                let ts = profiler.compute_pass_timestamp_writes("probe_place_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_place_pass"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_place_pipeline);
                pass.set_bind_group(0, self.probe_place_bg_cache.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups((gw + 7) / 8, (gh + 7) / 8, 1);
            }

            // ---- trace ----
            // Sun direction in world space. Bloom's public directional-light
            // convention is surface-to-light (the same vector consumed by
            // the raster PBR and shadow passes), so preserve its sign here.
            // Normalise because the probe shaders do not.
            let ld = self.lighting_uniforms.light_dir;
            let sun_inv_len = 1.0
                / (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2])
                    .sqrt()
                    .max(1e-4);
            let sun_dir_ws = [
                ld[0] * sun_inv_len,
                ld[1] * sun_inv_len,
                ld[2] * sun_inv_len,
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
            let shadows_enabled = self.shadow_map.enabled;
            let shadow_vps = if shadows_enabled {
                self.shadow_map.light_vps
            } else {
                [IDENTITY_MAT4; 3]
            };
            let shadow_splits = if shadows_enabled {
                let splits = self.shadow_map.cascade_splits;
                [splits[0], splits[1], splits[2], 0.0]
            } else {
                [f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.0]
            };
            let trace_params = ProbeTraceParams {
                view: self.current_view_matrix,
                proj: self.current_proj_matrix,
                inv_view,
                proj_row01: [p00, p11, p20, p21],
                size: [half_w, half_h, gw, gh],
                params: [
                    (self.probe_frame_index & 4095) as f32,
                    self.ssgi_intensity,
                    self.ssgi_radius,
                    10.0, // firefly luma cap
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
                        if self.wsrc_built[0] {
                            WSRC_CASCADE_EXTENTS[0]
                        } else {
                            0.0
                        },
                    ],
                    [
                        self.wsrc_origin[1][0],
                        self.wsrc_origin[1][1],
                        self.wsrc_origin[1][2],
                        if self.wsrc_built[1] {
                            WSRC_CASCADE_EXTENTS[1]
                        } else {
                            0.0
                        },
                    ],
                    [
                        self.wsrc_origin[2][0],
                        self.wsrc_origin[2][1],
                        self.wsrc_origin[2][2],
                        if self.wsrc_built[2] {
                            WSRC_CASCADE_EXTENTS[2]
                        } else {
                            0.0
                        },
                    ],
                ],
                shadow_vps,
                shadow_splits,
                shadow_params: [
                    0.002,
                    if shadows_enabled { 1.0 } else { 0.0 },
                    // Fade the coherent card-lighting term in over ~0.8 s
                    // instead of switching the instant the final BLAS lands.
                    // The binary flip made the sun-lit bounce (most visibly
                    // Bistro's red awnings onto the façade above them) appear
                    // as a sudden delayed light change ~20 s into a session.
                    self.card_light_coherent_ramp,
                    0.0,
                ],
            };
            self.queue.write_buffer(
                &self.probe_trace_uniform,
                0,
                bytemuck::bytes_of(&trace_params),
            );
            // V3 — trace BG now binds the prev-frame history view at
            // binding 11. `prev_idx` ping-pongs every frame so we
            // cache both slots independently.
            if self.probe_trace_bg_cache[prev_idx].is_none() {
                self.probe_trace_bg_cache[prev_idx] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("probe_trace_bg"),
                        layout: &self.probe_trace_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.probe_trace_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.probe_header_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.hiz_views[1]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.hiz_views[2]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(&self.hiz_views[3]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(&self.hiz_views[4]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.probe_trace_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 11,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.probe_history_views[prev_idx],
                                ),
                            },
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
            let use_sdf =
                !use_hw && self.scene_sdf_clipmap_built && self.tlas_instance_data_buffer.is_some();
            // Log the backend once (and again if it changes, e.g. clipmap
            // finishing its first bake promotes hiz → sdf). Nothing else in
            // the engine reveals which tier actually runs.
            let backend = if use_hw {
                "hw-ray-query"
            } else if use_sdf {
                "sdf-clipmap"
            } else {
                "hiz-screen"
            };
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
                    self.probe_trace_hw_bg_cache[prev_idx] = Some(
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("probe_trace_hw_bg"),
                            layout: self.probe_trace_hw_layout.as_ref().unwrap(),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.probe_trace_uniform.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: self.probe_header_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: tlas.as_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: self
                                        .tlas_instance_data_buffer
                                        .as_ref()
                                        .unwrap()
                                        .as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.probe_trace_view,
                                    ),
                                },
                                // Hardware GI shades from camera-independent
                                // card material data and traces its own sun
                                // visibility. The CSM-lit atlas is reserved
                                // for the software fallback.
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.mesh_card_atlas_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.mesh_card_atlas_sampler,
                                    ),
                                },
                                // V7/V10 — WSRC atlas + linear sampler.
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.wsrc_atlas_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 8,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.wsrc_atlas_sampler,
                                    ),
                                },
                                // V3 — prev-frame probe history.
                                wgpu::BindGroupEntry {
                                    binding: 9,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.probe_history_views[prev_idx],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 10,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.shadow_map.depth_views[0],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 11,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.shadow_map.depth_views[1],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 12,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.shadow_map.depth_views[2],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 13,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.shadow_map.sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 14,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.mesh_card_emissive_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 15,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.mesh_card_radiance_view,
                                    ),
                                },
                            ],
                        }),
                    );
                }
                let ts = profiler.compute_pass_timestamp_writes("probe_trace_hw_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_trace_hw_pass"),
                    timestamp_writes: ts,
                });
                let pipeline = if self.transparent_gi_active {
                    self.probe_trace_hw_transparent_pipeline
                        .as_ref()
                        .expect("active transparent GI has a lazy HW specialization")
                } else {
                    self.probe_trace_hw_pipeline.as_ref().unwrap()
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(
                    0,
                    self.probe_trace_hw_bg_cache[prev_idx].as_ref().unwrap(),
                    &[],
                );
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
                    let instance_buf = self
                        .tlas_instance_data_buffer
                        .as_ref()
                        .expect("V4: instance_data buffer must exist before SDF dispatch");
                    self.probe_trace_sdf_bg_cache[prev_idx] =
                        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("probe_trace_sdf_bg"),
                            layout: &self.probe_trace_sdf_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.probe_trace_uniform.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: self.probe_header_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.scene_sdf_clipmap_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(&nf_samp),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.probe_trace_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: instance_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.mesh_card_radiance_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.mesh_card_atlas_sampler,
                                    ),
                                },
                                // V6/V10 — WSRC atlas + linear sampler.
                                wgpu::BindGroupEntry {
                                    binding: 8,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.wsrc_atlas_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 9,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.wsrc_atlas_sampler,
                                    ),
                                },
                                // V3 — prev-frame probe history.
                                wgpu::BindGroupEntry {
                                    binding: 10,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.probe_history_views[prev_idx],
                                    ),
                                },
                            ],
                        }));
                }
                let ts = profiler.compute_pass_timestamp_writes("probe_trace_sdf_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_trace_sdf_pass"),
                    timestamp_writes: ts,
                });
                let pipeline = if self.transparent_gi_active {
                    self.probe_trace_sdf_transparent_pipeline
                        .as_ref()
                        .expect("active transparent GI has a lazy SDF specialization")
                } else {
                    &self.probe_trace_sdf_pipeline
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(
                    0,
                    self.probe_trace_sdf_bg_cache[prev_idx].as_ref().unwrap(),
                    &[],
                );
                pass.dispatch_workgroups(gw, gh, 1);
            } else {
                let ts = profiler.compute_pass_timestamp_writes("probe_trace_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_trace_pass"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_trace_pipeline);
                pass.set_bind_group(
                    0,
                    self.probe_trace_bg_cache[prev_idx].as_ref().unwrap(),
                    &[],
                );
                pass.dispatch_workgroups(gw, gh, 1);
            }

            // ---- temporal (EMA) ----
            // Invalid SSGI history is replaced independently of the TAA
            // counter: the effects have different enable/ownership lifetimes.
            let force_refresh =
                if !self.probe_history_valid || self.transparent_gi_force_probe_refresh {
                    1.0_f32
                } else {
                    0.0_f32
                };
            self.transparent_gi_force_probe_refresh = false;
            let (
                probe_header_bytes,
                probe_world_cache_offset,
                probe_world_cache_bytes,
                cache_capacity,
            ) = probe_storage_buffer_layout(gw * gh, self.hw_rt_enabled);
            let mut cache_signature = FNV_OFFSET;
            cache_signature = fnv1a_bytes(cache_signature, &self.tlas_built_version.to_le_bytes());
            cache_signature =
                fnv1a_bytes(cache_signature, &self.card_light_input_hash.to_le_bytes());
            cache_signature = fnv1a_bytes(
                cache_signature,
                bytemuck::bytes_of(&[self.ssgi_intensity, self.ssgi_radius]),
            );
            let cache_signature = ((cache_signature >> 32) as u32 ^ cache_signature as u32).max(1);
            let cache_signature_changed = self.probe_world_cache_signature != cache_signature;
            if use_hw && (force_refresh > 0.5 || cache_signature_changed) {
                // A camera cut, feature transition, or lighting-mode reset
                // invalidates both screen history and the persistent lookup.
                // This is a buffer clear command, not a render/compute pass.
                encoder.clear_buffer(
                    &self.probe_header_buffer,
                    probe_world_cache_offset,
                    Some(probe_world_cache_bytes),
                );
            }
            if use_hw {
                self.probe_world_cache_signature = cache_signature;
            }
            // A hardware scene is not cacheable while BLAS/card admission is
            // still changing its hit-lighting field. Software paths do not
            // have that delayed handoff.
            let cache_writes_allowed =
                use_hw && self.card_light_coherent && self.card_light_coherent_ramp >= 0.999;
            let temporal_params = ProbeTemporalParams {
                params: [
                    SSGI_TEMPORAL_PHASE_WEIGHT,
                    force_refresh,
                    gw as f32,
                    gh as f32,
                ],
                size: [half_w as f32, half_h as f32, PROBE_TILE_SIZE as f32, p00],
                confidence: [
                    if use_hw { 1.0 } else { 0.0 },
                    (self.probe_frame_index & 15) as f32,
                    SSGI_TEMPORAL_OUTPUT_WEIGHT,
                    if probe_lattice_jitter_active {
                        1.0
                    } else {
                        0.0
                    },
                ],
                world_cache: [
                    if use_hw { cache_capacity } else { 0 },
                    cache_signature,
                    self.probe_frame_index,
                    u32::from(cache_writes_allowed),
                ],
            };
            self.queue.write_buffer(
                &self.probe_temporal_uniform,
                0,
                bytemuck::bytes_of(&temporal_params),
            );
            // Bind group indexed by write_idx: each direction of the
            // ping-pong (read prev, write write) gets its own cached BG.
            if self.probe_temporal_bg_cache[write_idx].is_none() {
                self.probe_temporal_bg_cache[write_idx] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("probe_temporal_bg"),
                        layout: &self.probe_temporal_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.probe_temporal_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.probe_trace_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.probe_history_views[prev_idx],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.probe_history_views[write_idx],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &self.probe_header_buffer,
                                    offset: 0,
                                    size: std::num::NonZeroU64::new(probe_header_bytes),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.velocity_rt_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &self.probe_header_buffer,
                                    offset: probe_world_cache_offset,
                                    size: std::num::NonZeroU64::new(probe_world_cache_bytes),
                                }),
                            },
                        ],
                    }));
            }
            {
                let ts = profiler.compute_pass_timestamp_writes("probe_temporal_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_temporal_pass"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_temporal_pipeline);
                pass.set_bind_group(
                    0,
                    self.probe_temporal_bg_cache[write_idx].as_ref().unwrap(),
                    &[],
                );
                pass.dispatch_workgroups(gw, gh, 1);
            }
            {
                let ts = profiler.compute_pass_timestamp_writes("probe_spatial_pass");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("probe_spatial_pass"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.probe_spatial_pipeline);
                pass.set_bind_group(
                    0,
                    self.probe_temporal_bg_cache[write_idx].as_ref().unwrap(),
                    &[],
                );
                pass.dispatch_workgroups(gw.div_ceil(8), gh.div_ceil(8), 1);
            }
            self.probe_history_valid = true;

            // ---- resolve ----
            // Progressive HW scene admission changes the cache signature once
            // per newly admitted BLAS. That is a radiance update, not a
            // receiver-history discontinuity: hard-resetting the per-pixel
            // resolve on every one of those frames exposes the screen-space
            // probe rows whenever the camera moves. Follow the probe-domain
            // temporal window (1/8 current) while the ray scene grows. True
            // feature/camera-cut invalidation still requests a full refresh.
            let resolve_history_current_floor = if force_refresh > 0.5 {
                1.0
            } else if use_hw && cache_signature_changed {
                SSGI_TEMPORAL_OUTPUT_WEIGHT
            } else {
                0.0
            };
            let resolve_params = ProbeResolveParams {
                inv_view,
                prev_view: self.prev_view_matrix,
                proj_row01: [p00, p11, p20, p21],
                size: [half_w, half_h, gw, gh],
                params: [
                    PROBE_TILE_SIZE as f32,
                    1.0,
                    resolve_history_current_floor,
                    if camera_moving { 1.0 } else { 0.0 },
                ],
                temporal: [
                    (self.probe_frame_index & 15) as f32,
                    if probe_lattice_jitter_active {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                ],
            };
            self.queue.write_buffer(
                &self.probe_resolve_uniform,
                0,
                bytemuck::bytes_of(&resolve_params),
            );
            #[cfg(not(target_arch = "wasm32"))]
            if self.pending_quality_capture_dir.is_some() {
                self.record_ssgi_temporal_diagnostics(encoder, write_idx, gw, gh, half_w, half_h);
                self.record_ssgi_resolve_support_diagnostic(encoder, prev_idx);
            }
            if self.probe_resolve_bg_cache[write_idx].is_none() {
                self.probe_resolve_bg_cache[write_idx] =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("probe_resolve_bg"),
                        layout: &self.probe_resolve_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.probe_resolve_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.probe_header_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.probe_history_views[write_idx],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.ssgi_rt_views[prev_idx],
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
                                resource: wgpu::BindingResource::TextureView(
                                    &self.ssgi_rt_views[write_idx],
                                ),
                            },
                        ],
                    }));
            }
            let ts = profiler.compute_pass_timestamp_writes("probe_resolve_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_resolve_pass"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.probe_resolve_pipeline);
            pass.set_bind_group(
                0,
                self.probe_resolve_bg_cache[write_idx].as_ref().unwrap(),
                &[],
            );
            pass.dispatch_workgroups(half_w.div_ceil(8), half_h.div_ceil(8), 1);
        } else {
            // Suppressed frames do not produce probe history. This also
            // catches progressive PT changing ownership without a mode change.
            self.probe_history_valid = false;
            // SSGI disabled — clear the resolve target so downstream
            // composite reads contribute zero.
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssgi_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssgi_rt_views[write_idx],
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

#[cfg(test)]
mod tests {
    use super::{
        probe_inverse_view_for_wgsl, SSGI_TEMPORAL_OUTPUT_WEIGHT, SSGI_TEMPORAL_PHASE_WEIGHT,
    };
    use crate::renderer::{mat4_look_at, mat4_mul_vec4};

    #[test]
    fn screen_probe_integral_converges_over_sixteen_angular_phases() {
        assert_eq!(SSGI_TEMPORAL_PHASE_WEIGHT, 0.0625);
        assert_eq!(SSGI_TEMPORAL_OUTPUT_WEIGHT, 0.125);
    }

    #[test]
    fn probe_inverse_view_upload_reconstructs_the_original_world_point() {
        let view = mat4_look_at(
            [-4.526_873, 1.544, 6.502_634],
            [62.65, -1.156, -67.57],
            [0.0, 1.0, 0.0],
        );
        let world = [4.486_212, 4.024_019, 2.742_565, 1.0];
        let view_point = mat4_mul_vec4(&view, &world);
        let inverse_for_wgsl = probe_inverse_view_for_wgsl(view);
        let reconstructed = mat4_mul_vec4(&inverse_for_wgsl, &view_point);
        for axis in 0..4 {
            assert!(
                (reconstructed[axis] - world[axis]).abs() < 0.000_1,
                "axis {axis}: reconstructed={} expected={}",
                reconstructed[axis],
                world[axis],
            );
        }
    }
}
