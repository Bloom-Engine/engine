//! Public material-system API implemented on [`Renderer`].
//!
//! Kept separate from the render-loop implementation so material compilation,
//! instance submission, authoring controls, and per-frame ABI synchronization
//! can evolve without growing `renderer/mod.rs`.

use super::*;

impl Renderer {
    /// Read-only capability, limit, residency, and diagnostic report for the
    /// material indirection backend selected from the actual wgpu device.
    pub fn material_binding_report_json(&self) -> String {
        self.material_system.indirection.report_json()
    }

    /// Debug override for qualification. 0=auto, 1=Tier C, 2=Tier B, 3=Tier A.
    /// The indirection layer rejects requests above the adapter's detected tier.
    pub fn set_material_binding_tier_override(&mut self, code: u32) -> bool {
        let accepted = self
            .material_system
            .indirection
            .set_tier_override(&self.device, code);
        self.material_system
            .indirection
            .flush(&self.device, &self.queue);
        accepted
    }

    /// Stable GPU-facing ID corresponding to a legacy material handle.
    pub fn material_id(
        &self,
        handle: material_system::MaterialHandle,
    ) -> material_indirection::MaterialId {
        handle
            .checked_sub(1)
            .and_then(|index| self.material_system.material_ids.get(index as usize))
            .copied()
            .unwrap_or(material_indirection::MaterialId::FALLBACK)
    }

    /// Stable GPU-facing ID corresponding to a legacy renderer texture index.
    pub fn texture_id(&self, texture_index: u32) -> material_indirection::TextureId {
        self.global_texture_ids
            .get(texture_index as usize)
            .copied()
            .unwrap_or(material_indirection::TextureId::FALLBACK)
    }

    /// Compile a material from user-supplied WGSL source. Returns a
    /// handle to use with `submit_material_draw`. The source may
    /// `#include "material_abi.wgsl"` and any `common/*.wgsl` header.
    pub fn compile_material(
        &mut self,
        wgsl_source: &str,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        self.compile_material_with_options(
            wgsl_source,
            material_pipeline::FragmentProfile::Opaque,
            material_pipeline::Bucket::Opaque,
            false,
            false,
        )
    }

    /// Phase 4a — full-control material compile. Games that want a
    /// translucent / refractive / additive material (or a non-default
    /// bucket) call this directly. Plain `compile_material` is a
    /// convenience for Opaque + no scene reads.
    ///
    /// `wants_instancing` adds a per-instance vertex buffer layout at
    /// slot 1 (EN-001). Materials compiled with it must be drawn via
    /// `submit_material_draw_instanced` + a buffer from
    /// `create_instance_buffer`.
    pub fn compile_material_with_options(
        &mut self,
        wgsl_source: &str,
        profile: material_pipeline::FragmentProfile,
        bucket: material_pipeline::Bucket,
        reads_scene: bool,
        wants_instancing: bool,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        self.material_system.compile(
            &self.device,
            wgsl_source,
            profile,
            bucket,
            reads_scene,
            wants_instancing,
            formats::HDR_FORMAT,
            formats::MATERIAL_FORMAT,
            formats::VELOCITY_FORMAT,
            wgpu::TextureFormat::Rgba8Unorm,
            formats::DEPTH_FORMAT,
        )
    }

    /// EN-001 — compile a material that opts into the standard per-instance
    /// vertex layout (Opaque profile + Opaque bucket + wants_instancing).
    /// Pair with `create_instance_buffer` + `submit_material_draw_instanced`.
    /// The game shader's VertexInput must declare the instance attribute
    /// locations (see `material_abi.wgsl` for the layout).
    pub fn compile_material_instanced(
        &mut self,
        wgsl_source: &str,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        self.compile_material_instanced_bucket(wgsl_source, 0, false)
    }

    /// EN-026/027 — instanced compile into a chosen bucket.
    ///
    /// The original instanced path was hardcoded to Opaque, which is right for
    /// grass and wrong for the two things that most want instancing: particles
    /// (additive, thousands of quads) and decals (cutout, alpha-tested against
    /// the atlas). `bucket`: 0 = opaque, 1 = cutout, 2 = additive,
    /// 3 = transparent.
    ///
    /// `reads_scene` binds the scene colour/depth snapshot group. Soft
    /// particles NEED it — a billboard that intersects the ground shows a hard
    /// straight seam otherwise, which is the single biggest tell that a "puff"
    /// is a flat card — and without this flag the group is absent from the
    /// pipeline layout and the shader fails validation at create time.
    pub fn compile_material_instanced_bucket(
        &mut self,
        wgsl_source: &str,
        bucket: u32,
        reads_scene: bool,
    ) -> Result<material_system::MaterialHandle, material_pipeline::MaterialCompileError> {
        let (profile, bucket) = match bucket {
            1 => (
                material_pipeline::FragmentProfile::Opaque,
                material_pipeline::Bucket::Cutout,
            ),
            2 => (
                material_pipeline::FragmentProfile::Translucent,
                material_pipeline::Bucket::Additive,
            ),
            3 => (
                material_pipeline::FragmentProfile::Translucent,
                material_pipeline::Bucket::Transparent,
            ),
            _ => (
                material_pipeline::FragmentProfile::Opaque,
                material_pipeline::Bucket::Opaque,
            ),
        };
        self.material_system.compile(
            &self.device,
            wgsl_source,
            profile,
            bucket,
            reads_scene,
            true,
            formats::HDR_FORMAT,
            formats::MATERIAL_FORMAT,
            formats::VELOCITY_FORMAT,
            wgpu::TextureFormat::Rgba8Unorm,
            formats::DEPTH_FORMAT,
        )
    }

    /// EN-001 — upload a CPU-side per-instance buffer to GPU memory.
    /// `raw` is laid out as 9 floats per instance (pos.xyz, rot_y,
    /// scale, tint.rgba). Returns a handle for use with
    /// `submit_material_draw_instanced`.
    pub fn create_instance_buffer(&mut self, raw: &[f32], count: u32) -> u32 {
        self.material_system
            .create_instance_buffer(&self.device, &self.queue, raw, count)
    }

    /// EN-001 — release the GPU memory backing an instance buffer.
    /// Safe to call with handle 0 or stale handles (no-op).
    pub fn destroy_instance_buffer(&mut self, handle: u32) {
        self.material_system.destroy_instance_buffer(handle);
    }

    /// EN-017 V2 — append a fullscreen WGSL post-pass to the stack.
    /// Compiles the shader, lazily allocates ping-pong LDR
    /// intermediates as the stack grows, and pushes onto the stack.
    /// Returns the 1-based handle of the newly added pass on success
    /// (so callers can treat 0 as "compile failed"), or Err on
    /// shader-compile failure; the existing stack is left intact.
    ///
    /// The fragment shader sees `scene_color_tex` (LDR, post-tonemap)
    /// + `scene_depth_tex` at `@group(0)` — see
    /// `post_pass::POST_PASS_PRELUDE` for the exact ABI.
    ///
    /// Stack order matters: the first added pass runs first, the
    /// next sees the first's output, and so on. The last pass writes
    /// the swapchain.
    pub fn add_post_pass(
        &mut self,
        wgsl_source: &str,
    ) -> Result<u32, post_pass::PostPassCompileError> {
        let pipeline = post_pass::compile_post_pass(&self.device, wgsl_source, self.output_format)?;

        if self.composite_ldr_rt_a.is_none() {
            let (texture, view) = post_pass::create_composite_ldr_rt(
                &self.device,
                self.surface_config.width,
                self.surface_config.height,
                self.output_format,
            );
            self.composite_ldr_rt_a = Some(texture);
            self.composite_ldr_rt_a_view = Some(view);
        }
        if self.post_passes.len() + 1 >= 2 && self.composite_ldr_rt_b.is_none() {
            let (texture, view) = post_pass::create_composite_ldr_rt(
                &self.device,
                self.surface_config.width,
                self.surface_config.height,
                self.output_format,
            );
            self.composite_ldr_rt_b = Some(texture);
            self.composite_ldr_rt_b_view = Some(view);
        }

        self.post_passes.push(pipeline);
        Ok(self.post_passes.len() as u32)
    }

    /// EN-017 V2 — wipe the post-pass stack. The composite output
    /// goes directly to the swapchain again (zero post-pass cost).
    /// LDR intermediates stay allocated to avoid churn when toggled.
    pub fn clear_all_post_passes(&mut self) {
        self.post_passes.clear();
    }

    /// EN-017 V1 backward-compat — replace the entire stack with a
    /// single post-pass.
    pub fn set_post_pass(
        &mut self,
        wgsl_source: &str,
    ) -> Result<(), post_pass::PostPassCompileError> {
        self.clear_all_post_passes();
        self.add_post_pass(wgsl_source)?;
        Ok(())
    }

    /// EN-017 V1 backward-compat — clear the post-pass stack.
    pub fn clear_post_pass(&mut self) {
        self.clear_all_post_passes();
    }

    /// Phase 6 — compile a material from a WGSL file on disk and
    /// register the path with the hot-reload watcher.
    pub fn compile_material_from_file(
        &mut self,
        path: &std::path::Path,
        profile: material_pipeline::FragmentProfile,
        bucket: material_pipeline::Bucket,
        reads_scene: bool,
    ) -> Result<material_system::MaterialHandle, String> {
        let canonical =
            std::fs::canonicalize(path).map_err(|e| format!("canonicalize {path:?}: {e}"))?;
        let source =
            std::fs::read_to_string(&canonical).map_err(|e| format!("read {canonical:?}: {e}"))?;
        let handle = self
            .compile_material_with_options(&source, profile, bucket, reads_scene, false)
            .map_err(|e| format!("compile {canonical:?}: {e:?}"))?;
        self.material_hot_reload.register(
            handle,
            hot_reload::FileMaterialDesc {
                path: canonical,
                profile,
                bucket,
                reads_scene,
                wants_instancing: false,
            },
        );
        Ok(handle)
    }

    /// Drain pending hot-reload events and rebuild affected pipelines.
    /// Failures retain the previous live pipeline.
    pub fn poll_material_hot_reload(&mut self) {
        let pending = self.material_hot_reload.drain_pending();
        for (handle, desc) in pending {
            let source = match std::fs::read_to_string(&desc.path) {
                Ok(source) => source,
                Err(error) => {
                    eprintln!("[hot_reload] read {:?} failed: {error}", desc.path);
                    continue;
                }
            };
            match self.material_system.compile(
                &self.device,
                &source,
                desc.profile,
                desc.bucket,
                desc.reads_scene,
                desc.wants_instancing,
                formats::HDR_FORMAT,
                formats::MATERIAL_FORMAT,
                formats::VELOCITY_FORMAT,
                wgpu::TextureFormat::Rgba8Unorm,
                formats::DEPTH_FORMAT,
            ) {
                Ok(new_handle) => {
                    let new_idx = (new_handle - 1) as usize;
                    let old_idx = (handle - 1) as usize;
                    if let Some(pipeline) = self
                        .material_system
                        .pipelines
                        .get_mut(new_idx)
                        .and_then(|slot| slot.take())
                    {
                        if let Some(slot) = self.material_system.pipelines.get_mut(old_idx) {
                            *slot = Some(pipeline);
                        }
                    }
                    eprintln!("[hot_reload] reloaded {:?} (handle {handle})", desc.path);
                }
                Err(error) => {
                    eprintln!(
                        "[hot_reload] compile {:?} failed: {error:?} — keeping previous",
                        desc.path
                    );
                }
            }
        }
    }

    /// Submit a material draw against a cached mesh.
    pub fn submit_material_draw(
        &mut self,
        material: material_system::MaterialHandle,
        mesh_handle: u64,
        mesh_idx: usize,
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
    ) {
        let model = mat4_multiply(
            mat4_translate(IDENTITY_MAT4, position),
            mat4_scale(IDENTITY_MAT4, [scale, scale, scale]),
        );
        let mvp = mat4_multiply(self.current_vp_matrix, model);
        self.material_system.submit_draw(
            &self.device,
            &self.queue,
            &self.joint_buffer,
            material,
            mesh_handle,
            mesh_idx,
            mvp,
            model,
            mvp,
            tint,
            [0, 0, 0, 0],
        );
    }

    /// EN-001 — submit an instanced material draw.
    pub fn submit_material_draw_instanced(
        &mut self,
        material: material_system::MaterialHandle,
        mesh_handle: u64,
        mesh_idx: usize,
        instance_buffer: u32,
        instance_count: u32,
    ) {
        let model = IDENTITY_MAT4;
        let mvp = self.current_vp_matrix;
        self.material_system.submit_draw_instanced(
            &self.device,
            &self.queue,
            &self.joint_buffer,
            material,
            mesh_handle,
            mesh_idx,
            instance_buffer,
            instance_count,
            mvp,
            model,
            mvp,
            [1.0, 1.0, 1.0, 1.0],
            [0, 0, 0, 0],
        );
    }

    /// Create a planar reflection probe and return its 1-based handle.
    pub fn create_planar_reflection(
        &mut self,
        plane_y: f32,
        normal: [f32; 3],
        resolution: u32,
    ) -> u32 {
        let resolution = if resolution == 0 {
            (self.surface_config.width / 2).max(16)
        } else {
            resolution
        };
        let probe = planar_reflection::PlanarReflectionProbe::new(
            &self.device,
            plane_y,
            normal,
            resolution,
        );
        let view_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planar_probe_per_view"),
            size: std::mem::size_of::<material_system::PerViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.planar_probes.push(Some(probe));
        self.planar_probe_view_buffers.push(Some(view_buffer));
        self.planar_probe_view_bgs.push(None);
        self.planar_probes.len() as u32
    }

    /// Link a material handle to a planar reflection probe.
    pub fn set_material_reflection_probe(&mut self, material: u32, probe: u32) {
        if material == 0 {
            return;
        }
        if probe == 0 {
            let view = self.material_system.default_black_view.clone();
            if let Err(error) =
                self.material_system
                    .set_reflection_probe(&self.device, material, 0, &view)
            {
                eprintln!("[planar_reflection] unlink failed for material {material}: {error}");
            }
            return;
        }
        let idx = probe as usize - 1;
        let probe_view = match self.planar_probes.get(idx).and_then(|item| item.as_ref()) {
            Some(probe) => probe.color_view.clone(),
            None => {
                eprintln!("[planar_reflection] unknown probe handle {probe}");
                return;
            }
        };
        if let Err(error) =
            self.material_system
                .set_reflection_probe(&self.device, material, probe, &probe_view)
        {
            eprintln!("[planar_reflection] set failed for material {material}: {error}");
        }
    }

    /// Create a texture array from RGBA8 source layers.
    pub fn create_texture_array(&mut self, layers: &[(&[u8], u32, u32)]) -> u32 {
        self.material_system
            .create_texture_array(&self.device, &self.queue, layers)
    }

    /// Create a texture array with explicit format and mip control.
    pub fn create_texture_array_ex(
        &mut self,
        layers: &[(&[u8], u32, u32)],
        format: u32,
        mip_levels: u32,
    ) -> u32 {
        self.material_system.create_texture_array_ex(
            &self.device,
            &self.queue,
            layers,
            format,
            mip_levels,
        )
    }

    /// Link an albedo, normal, or metallic-roughness texture array.
    pub fn set_material_texture_array(&mut self, material: u32, slot: u32, array: u32) {
        let probe_view = match self
            .material_system
            .material_reflection_probe_handle(material)
        {
            Some(probe) if probe != 0 => {
                let idx = probe as usize - 1;
                self.planar_probes
                    .get(idx)
                    .and_then(|item| item.as_ref())
                    .map(|probe| probe.color_view.clone())
                    .unwrap_or_else(|| self.material_system.default_black_view.clone())
            }
            _ => self.material_system.default_black_view.clone(),
        };
        self.material_system.set_material_texture_array(
            &self.device,
            material,
            slot,
            array,
            &probe_view,
        );
    }

    /// Set the material shading model.
    pub fn set_material_shading_model(&mut self, material: u32, model: u32) {
        let probe_view = self.resolve_probe_view_for_material(material);
        if let Err(error) = self.material_system.set_material_shading_model(
            &self.device,
            &self.queue,
            material,
            model,
            &probe_view,
        ) {
            eprintln!("[foliage] set_material_shading_model failed: {error}");
        }
    }

    /// Set foliage transmission and wrap-light parameters.
    pub fn set_material_foliage(
        &mut self,
        material: u32,
        trans_color: [f32; 3],
        trans_amount: f32,
        wrap_factor: f32,
    ) {
        let probe_view = self.resolve_probe_view_for_material(material);
        if let Err(error) = self.material_system.set_material_foliage(
            &self.device,
            &self.queue,
            material,
            trans_color,
            trans_amount,
            wrap_factor,
            &probe_view,
        ) {
            eprintln!("[foliage] set_material_foliage failed: {error}");
        }
    }

    fn resolve_probe_view_for_material(&self, material: u32) -> wgpu::TextureView {
        match self
            .material_system
            .material_reflection_probe_handle(material)
        {
            Some(probe) if probe != 0 => {
                let idx = probe as usize - 1;
                self.planar_probes
                    .get(idx)
                    .and_then(|item| item.as_ref())
                    .map(|probe| probe.color_view.clone())
                    .unwrap_or_else(|| self.material_system.default_black_view.clone())
            }
            _ => self.material_system.default_black_view.clone(),
        }
    }

    /// Set whether a material is rendered into planar-reflection probes.
    pub fn set_material_probe_visible(&mut self, material: u32, visible: bool) {
        self.material_system.set_probe_visible(material, visible);
    }

    /// Synchronize material PerFrame and PerView uniforms.
    pub fn material_system_begin_frame(&mut self, time_seconds: f32, delta_time: f32) {
        self.lighting_uniforms.wind = [self.wind[0], self.wind[1], self.wind[2], time_seconds];
        self.lighting_uniforms.cloud = self.cloud_params;
        self.lighting_uniforms.frame_misc = [delta_time, 0.0, 0.0, 0.0];
        let screen_w = self.surface_config.width as f32;
        let screen_h = self.surface_config.height as f32;
        let (render_w, render_h) = self.render_extent();
        let per_frame = material_system::PerFrameUniforms {
            time: time_seconds,
            delta_time,
            frame_index: self.taa_frame_index as u32,
            _pad0: 0,
            screen_resolution: [screen_w, screen_h],
            render_resolution: [render_w as f32, render_h as f32],
            taa_jitter: [0.0, 0.0],
            _pad1: [0.0, 0.0],
            wind: self.wind,
            cloud: self.cloud_params,
        };
        let per_view = material_system::PerViewUniforms {
            view: self.current_view_matrix,
            proj: self.current_proj_matrix,
            view_proj: self.current_vp_matrix,
            prev_view_proj: self.velocity_ref_vp,
            inv_proj: self.current_inv_proj_matrix,
            camera_pos: [
                self.current_camera_pos[0],
                self.current_camera_pos[1],
                self.current_camera_pos[2],
                self.lighting_uniforms.camera_pos[3],
            ],
            camera_dir: [0.0, 0.0, -1.0, 70.0_f32.to_radians()],
            ambient: self.lighting_uniforms.ambient,
            fog: [
                self.fog_color[0],
                self.fog_color[1],
                self.fog_color[2],
                self.fog_density,
            ],
            sun_dir: self.lighting_uniforms.light_dir,
            sun_color: self.lighting_uniforms.light_color,
            dir_light_count: self.lighting_uniforms.dir_light_count,
            dir_lights: std::array::from_fn(|index| material_system::PerViewDirLight {
                direction: self.lighting_uniforms.dir_lights[index].direction,
                color: self.lighting_uniforms.dir_lights[index].color,
            }),
            point_light_count: self.lighting_uniforms.point_light_count,
            point_lights: std::array::from_fn(|index| material_system::PerViewPointLight {
                position: self.lighting_uniforms.point_lights[index].position,
                color: self.lighting_uniforms.point_lights[index].color,
            }),
            shadow_splits: self.lighting_uniforms.shadow_cascade_splits,
            shadow_view: self.lighting_uniforms.shadow_view_matrix,
            shadow_cascades: self.lighting_uniforms.shadow_cascade_vps,
        };
        self.material_system.update_frame_uniforms(
            &self.device,
            &self.queue,
            &per_frame,
            &per_view,
        );
    }
}
