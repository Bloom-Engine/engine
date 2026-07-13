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
            return;
        }
        // Same readiness gate as the HW probe trace: first frames before
        // any geometry is committed have no TLAS / instance data yet.
        if self.pt_pipeline.is_none()
            || self.tlas.is_none()
            || self.tlas_instance_data_buffer.is_none()
        {
            self.pt_accum_count = 0;
            return;
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
        self.pt_prev_vp = vp_unjittered;
        // Geometry changed under the accumulated image (door opened,
        // enemy died) → history is a lie, restart.
        if self.tlas_built_version != self.pt_last_tlas_version {
            self.pt_last_tlas_version = self.tlas_built_version;
            self.pt_accum_count = 0;
        }

        // ---- accumulation buffer (vec4<f32> per pixel) ----
        let needed = (surf_w as u64) * (surf_h as u64) * 16;
        let recreate = match &self.pt_accum_buffer {
            Some(b) => b.size() != needed,
            None => true,
        };
        if recreate {
            self.pt_accum_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_accum"),
                size: needed,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.pt_bg = None;
            self.pt_accum_count = 0;
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
        let params = PtParamsCpu {
            inv_vp: self.current_inv_vp_matrix,
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
            size: [surf_w, surf_h, self.taa_frame_index, self.pt_accum_count],
            cfg: [
                self.pt_mode as f32,
                max_bounces,
                light_count as f32,
                self.pt_debug,
            ],
            lights,
        };
        self.queue.write_buffer(&self.pt_uniform_buffer, 0, bytemuck::bytes_of(&params));

        // ---- bind group (lazy; nulled on resize / TLAS or instance
        // buffer recreation) ----
        if self.pt_bg.is_none() {
            let tlas = self.tlas.as_ref().unwrap();
            self.pt_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pt_bg"),
                layout: self.pt_layout.as_ref().unwrap(),
                entries: &[
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
                    wgpu::BindGroupEntry { binding: 8, resource: self.pt_accum_buffer.as_ref().unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.hdr_rt_view) },
                ],
            }));
        }

        // ---- dispatch ----
        {
            let ts = profiler.compute_pass_timestamp_writes("pt_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pt_pass"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(self.pt_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.pt_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups((surf_w + 7) / 8, (surf_h + 7) / 8, 1);
        }
        self.pt_accum_count = self.pt_accum_count.saturating_add(1);
    }
}
