//! Model draw submission: cached-model draw commands and the
//! immediate-mode mesh path (incl. GPU-skinning joint-pose plumbing).
//! Split from renderer/mod.rs (2000-line file policy — same pattern as
//! shadow_pass.rs).

use super::*;

/// Previous transform for one ordinary cached-model instance. Calls are paired
/// by stable submission slot, with the handle preventing a newly spawned
/// different model from inheriting the departed instance's velocity.
#[derive(Clone, Copy)]
struct CachedModelMotionEntry {
    handle_bits: u64,
    model: [[f32; 4]; 4],
}

#[derive(Default)]
pub(super) struct CachedModelMotionHistory {
    // CPU metadata only: feeds the existing prev_mvp uniform and adds no GPU
    // allocation or pass. Skinned draws own keyed palette history instead.
    previous: Vec<CachedModelMotionEntry>,
    current: Vec<CachedModelMotionEntry>,
    slot: usize,
}

pub(super) fn cached_model_subset_is_canonical<I>(mesh_count: usize, mesh_indices: I) -> bool
where
    I: Iterator<Item = usize>,
{
    let mut previous = None;
    for mesh_index in mesh_indices {
        if mesh_index >= mesh_count || previous.is_some_and(|value| mesh_index <= value) {
            return false;
        }
        previous = Some(mesh_index);
    }
    true
}

impl Renderer {
    pub(super) fn begin_skin_motion_frame(&mut self) {
        self.skin_motion_epoch = self.skin_motion_epoch.wrapping_add(1);
        std::mem::swap(
            &mut self.skin_unkeyed_previous,
            &mut self.skin_unkeyed_current,
        );
        self.skin_unkeyed_previous_count = self.skin_unkeyed_slot;
        self.skin_unkeyed_slot = 0;
    }

    pub(super) fn begin_cached_model_frame(&mut self) {
        self.model_draw_commands.clear();
        std::mem::swap(
            &mut self.cached_model_motion.previous,
            &mut self.cached_model_motion.current,
        );
        self.cached_model_motion.current.clear();
        self.cached_model_motion.slot = 0;
    }

    pub(super) fn reset_model_motion_history(&mut self) {
        self.material_system.reset_motion_history();
        self.cached_model_motion.previous.clear();
        self.skin_unkeyed_previous.clear();
        self.skin_unkeyed_current.clear();
        self.skin_unkeyed_previous_count = 0;
        self.skin_unkeyed_slot = 0;
        self.skin_prev_palettes.clear();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn cached_model_motion_stats(&self) -> (usize, usize) {
        let history = &self.cached_model_motion;
        let bytes = (history.previous.capacity() + history.current.capacity())
            * std::mem::size_of::<CachedModelMotionEntry>();
        (history.current.len(), bytes)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn unkeyed_skin_motion_stats(&self) -> (usize, usize) {
        let palettes = [&self.skin_unkeyed_previous, &self.skin_unkeyed_current];
        let outer = palettes
            .iter()
            .map(|values| {
                values
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Vec<[[f32; 4]; 4]>>())
            })
            .sum::<usize>();
        let inner = palettes
            .iter()
            .flat_map(|values| values.iter())
            .map(|palette| {
                palette
                    .capacity()
                    .saturating_mul(std::mem::size_of::<[[f32; 4]; 4]>())
            })
            .sum::<usize>();
        (self.skin_unkeyed_slot, outer.saturating_add(inner))
    }

    /// Set a single joint matrix for testing (joint_index 0-127).
    pub fn set_joint_test(&mut self, joint_index: usize, angle: f32) {
        if joint_index >= 128 {
            return;
        }
        let c = angle.cos();
        let s = angle.sin();
        let mat: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0],
            [0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.queue.write_buffer(
            &self.joint_buffer,
            (joint_index * 64) as u64,
            bytemuck::cast_slice(&mat),
        );
    }

    /// Stage a skin pose using stable submission order as its identity.
    pub fn set_joint_matrices(&mut self, matrices: &[[[f32; 4]; 4]]) {
        let slot = self.skin_unkeyed_slot;
        self.skin_unkeyed_slot += 1;
        let previous = self
            .skin_unkeyed_previous
            .get(slot)
            .filter(|palette| {
                slot < self.skin_unkeyed_previous_count && palette.len() == matrices.len()
            })
            .cloned()
            .unwrap_or_else(|| matrices.to_vec());
        if let Some(current) = self.skin_unkeyed_current.get_mut(slot) {
            current.clear();
            current.extend_from_slice(matrices);
        } else {
            self.skin_unkeyed_current.push(matrices.to_vec());
        }
        self.pending_skin_groups_prev.push(previous);
        self.pending_skin_groups.push(matrices.to_vec());
    }

    pub fn set_model_skin_scale(&mut self, scale: f32) {
        self.model_skin_scale = scale;
    }

    /// `key` pairs this pose with the same model's pose in the immediately
    /// preceding frame. World placement is baked into each matrix, so the
    /// previous palette captures both skeletal deformation and locomotion.
    pub fn set_joint_matrices_scaled(
        &mut self,
        key: u64,
        matrices: &[[[f32; 4]; 4]],
        scale: f32,
        position: [f32; 3],
        rot_sin: f32,
        rot_cos: f32,
    ) {
        let mut scaled = Vec::with_capacity(matrices.len());
        for matrix in matrices {
            let mut transformed = *matrix;
            for col in 0..4 {
                transformed[col][0] *= scale;
                transformed[col][1] *= scale;
                transformed[col][2] *= scale;
            }
            for col in 0..4 {
                let x = transformed[col][0];
                let z = transformed[col][2];
                transformed[col][0] = rot_cos * x + rot_sin * z;
                transformed[col][2] = -rot_sin * x + rot_cos * z;
            }
            transformed[3][0] += position[0];
            transformed[3][1] += position[1];
            transformed[3][2] += position[2];
            scaled.push(transformed);
        }

        let previous = match self.skin_prev_palettes.get(&key) {
            Some((epoch, palette))
                if epoch.wrapping_add(1) == self.skin_motion_epoch
                    && palette.len() == scaled.len() =>
            {
                palette.clone()
            }
            _ => scaled.clone(),
        };
        self.pending_skin_groups_prev.push(previous);
        self.skin_prev_palettes
            .insert(key, (self.skin_motion_epoch, scaled.clone()));
        self.pending_skin_groups.push(scaled);
    }

    /// Pair an ordinary cached instance with the same submission slot from the
    /// prior frame. A missing/mismatched slot is a spawn and therefore seeds
    /// with the current transform (zero object velocity).
    fn track_cached_model_motion(
        &mut self,
        handle_bits: u64,
        model: [[f32; 4]; 4],
    ) -> [[f32; 4]; 4] {
        let slot = self.cached_model_motion.slot;
        self.cached_model_motion.slot += 1;
        let previous = self
            .cached_model_motion
            .previous
            .get(slot)
            .filter(|entry| entry.handle_bits == handle_bits)
            .map(|entry| entry.model)
            .unwrap_or(model);
        self.cached_model_motion
            .current
            .push(CachedModelMotionEntry { handle_bits, model });
        previous
    }

    /// Record a cached model draw command. The actual rendering happens in end_frame().
    pub fn draw_model_cached(
        &mut self,
        handle_bits: u64,
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
    ) {
        self.has_blend_model_draws |= self.model_blended.contains(&handle_bits);
        self.has_layered_blend_model_draws |= self.model_layered_blended.contains(&handle_bits);
        self.has_refractive_model_draws |=
            self.imported_refraction_enabled && self.model_refractive.contains(&handle_bits);
        let mesh_count = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes.len(),
            _ => return,
        };
        // Foliage wind amount for this model (0 = not a plant). Rides in misc.z.
        let foliage = self.foliage_wind.get(&handle_bits).copied().unwrap_or(0.0);
        let model_matrix = mat4_multiply(
            mat4_translate(IDENTITY_MAT4, position),
            mat4_scale(IDENTITY_MAT4, [scale, scale, scale]),
        );
        let previous_model = self.track_cached_model_motion(handle_bits, model_matrix);

        for mesh_idx in 0..mesh_count {
            let source_transform = self
                .model_gpu_cache
                .get(&handle_bits)
                .and_then(Option::as_ref)
                .map(|meshes| meshes[mesh_idx].source_transform)
                .unwrap_or(IDENTITY_MAT4);
            let mesh_model = mat4_multiply(model_matrix, source_transform);
            let previous_mesh_model = mat4_multiply(previous_model, source_transform);
            let model_mvp = mat4_multiply(self.current_vp_matrix, mesh_model);
            let previous_mvp = mat4_multiply(self.velocity_ref_vp, previous_mesh_model);
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;

            // Grow uniform pool if needed
            self.ensure_model_uniform_slot(slot);

            // Stage uniform for this draw (flushed in one write at end-frame)
            self.stage_model_uniform(
                slot,
                &Uniforms3D {
                    mvp: model_mvp,
                    model: mesh_model,
                    prev_mvp: previous_mvp,
                    model_tint: tint,
                    misc: [0.0, 0.0, foliage, 0.0],
                },
            );

            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
                model: mesh_model,
                tint,
                skinned: false,
                joint_offset: 0.0,
                bounds_override: None,
            });
        }
    }

    /// Record only a canonical subset of one cached model. This is the
    /// established renderer path for primitives that a mixed `.bgeo` archive
    /// explicitly retained as compatibility geometry. The ordinary
    /// `draw_model_cached` path remains unchanged and allocation-free.
    ///
    /// Returns `false` without recording anything when the cache is absent or
    /// the indices are out of range, duplicated, or unordered. An empty subset
    /// is a successful no-op.
    pub fn draw_model_cached_meshes(
        &mut self,
        handle_bits: u64,
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
        mesh_indices: &[usize],
    ) -> bool {
        self.draw_model_cached_mesh_subset(
            handle_bits,
            position,
            scale,
            tint,
            mesh_indices.iter().copied(),
        )
    }

    /// Full-transform form of `draw_model_cached_meshes` for rotated, pitched,
    /// or otherwise authored placements.
    pub fn draw_model_cached_meshes_transform(
        &mut self,
        handle_bits: u64,
        model_matrix: [[f32; 4]; 4],
        tint: [f32; 4],
        mesh_indices: &[usize],
    ) -> bool {
        self.draw_model_cached_mesh_subset_transform(
            handle_bits,
            model_matrix,
            tint,
            mesh_indices.iter().copied(),
        )
    }

    /// Allocation-free bridge from the model/archive ownership route to the
    /// compatibility renderer. Virtual placements from the same route are
    /// submitted separately through `submit_virtual_geometry`. A compact
    /// compatibility cache that contains skin weights consumes the staged
    /// joint palette through the established cached-skinned path; a full-model
    /// subset cannot do that without also drawing virtual-owned primitives and
    /// therefore fails closed.
    #[cfg(feature = "models3d")]
    pub fn draw_model_cached_compatibility(
        &mut self,
        handle_bits: u64,
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
        route: &crate::models::ModelVirtualGeometryRoute,
    ) -> bool {
        if let Some(source_indices) = self.model_virtual_compatibility_sources.get(&handle_bits) {
            if source_indices.len() != route.compatibility_placements.len()
                || source_indices.iter().copied().ne(route
                    .compatibility_placements
                    .iter()
                    .map(|placement| placement.model_mesh_index))
            {
                return false;
            }
            if source_indices.is_empty() {
                return true;
            }
            if self.is_model_skinned(handle_bits) {
                self.draw_model_cached_skinned(handle_bits, position, scale, tint);
            } else {
                self.draw_model_cached(handle_bits, position, scale, tint);
            }
            return true;
        }
        if self.is_model_skinned(handle_bits) {
            return false;
        }
        self.draw_model_cached_mesh_subset(
            handle_bits,
            position,
            scale,
            tint,
            route
                .compatibility_placements
                .iter()
                .map(|placement| placement.model_mesh_index),
        )
    }

    /// Full-transform form of `draw_model_cached_compatibility`.
    /// Skinned palettes already own world placement, so callers must use the
    /// position/scale form after staging the palette instead of supplying a
    /// second arbitrary transform here.
    #[cfg(feature = "models3d")]
    pub fn draw_model_cached_compatibility_transform(
        &mut self,
        handle_bits: u64,
        model_matrix: [[f32; 4]; 4],
        tint: [f32; 4],
        route: &crate::models::ModelVirtualGeometryRoute,
    ) -> bool {
        if let Some(source_indices) = self.model_virtual_compatibility_sources.get(&handle_bits) {
            if source_indices.len() != route.compatibility_placements.len()
                || source_indices.iter().copied().ne(route
                    .compatibility_placements
                    .iter()
                    .map(|placement| placement.model_mesh_index))
            {
                return false;
            }
            if source_indices.is_empty() {
                return true;
            }
            if self.is_model_skinned(handle_bits) {
                return false;
            }
            self.draw_model_cached_transform(handle_bits, model_matrix, tint);
            return true;
        }
        if self.is_model_skinned(handle_bits) {
            return false;
        }
        self.draw_model_cached_mesh_subset_transform(
            handle_bits,
            model_matrix,
            tint,
            route
                .compatibility_placements
                .iter()
                .map(|placement| placement.model_mesh_index),
        )
    }

    fn draw_model_cached_mesh_subset<I>(
        &mut self,
        handle_bits: u64,
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
        mesh_indices: I,
    ) -> bool
    where
        I: Iterator<Item = usize> + Clone,
    {
        let model_matrix = mat4_multiply(
            mat4_translate(IDENTITY_MAT4, position),
            mat4_scale(IDENTITY_MAT4, [scale, scale, scale]),
        );
        self.draw_model_cached_mesh_subset_transform(handle_bits, model_matrix, tint, mesh_indices)
    }

    fn draw_model_cached_mesh_subset_transform<I>(
        &mut self,
        handle_bits: u64,
        model_matrix: [[f32; 4]; 4],
        tint: [f32; 4],
        mesh_indices: I,
    ) -> bool
    where
        I: Iterator<Item = usize> + Clone,
    {
        let mesh_count = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes.len(),
            _ => return false,
        };
        if !cached_model_subset_is_canonical(mesh_count, mesh_indices.clone()) {
            return false;
        }
        if mesh_indices.clone().next().is_none() {
            return true;
        }

        self.has_blend_model_draws |= self.model_blended.contains(&handle_bits);
        self.has_layered_blend_model_draws |= self.model_layered_blended.contains(&handle_bits);
        self.has_refractive_model_draws |=
            self.imported_refraction_enabled && self.model_refractive.contains(&handle_bits);
        let foliage = self.foliage_wind.get(&handle_bits).copied().unwrap_or(0.0);
        let previous_model = self.track_cached_model_motion(handle_bits, model_matrix);

        for mesh_idx in mesh_indices {
            let source_transform = self
                .model_gpu_cache
                .get(&handle_bits)
                .and_then(Option::as_ref)
                .map(|meshes| meshes[mesh_idx].source_transform)
                .unwrap_or(IDENTITY_MAT4);
            let mesh_model = mat4_multiply(model_matrix, source_transform);
            let previous_mesh_model = mat4_multiply(previous_model, source_transform);
            let model_mvp = mat4_multiply(self.current_vp_matrix, mesh_model);
            let previous_mvp = mat4_multiply(self.velocity_ref_vp, previous_mesh_model);
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;
            self.ensure_model_uniform_slot(slot);
            self.stage_model_uniform(
                slot,
                &Uniforms3D {
                    mvp: model_mvp,
                    model: mesh_model,
                    prev_mvp: previous_mvp,
                    model_tint: tint,
                    misc: [0.0, 0.0, foliage, 0.0],
                },
            );
            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
                model: mesh_model,
                tint,
                skinned: false,
                joint_offset: 0.0,
                bounds_override: None,
            });
        }
        true
    }

    /// `draw_model_cached` with a Y-axis rotation folded into the model
    /// matrix (translate ∘ rotY ∘ scale). Backs `bloom_draw_model_rotated`
    /// for static models — previously that FFI only had the immediate-mode
    /// vertex path, which bypasses the scene pipeline entirely (no alpha
    /// cutout, no normal/MR maps, no foliage wind or transmission, no
    /// cutout shadows) and re-transforms every vertex on the CPU each
    /// frame. Alpha-cutout foliage drawn through it rendered its cards'
    /// transparent texels as opaque.
    pub fn draw_model_cached_rotated(
        &mut self,
        handle_bits: u64,
        position: [f32; 3],
        scale: f32,
        rot_y: f32,
        tint: [f32; 4],
    ) {
        self.has_blend_model_draws |= self.model_blended.contains(&handle_bits);
        self.has_layered_blend_model_draws |= self.model_layered_blended.contains(&handle_bits);
        self.has_refractive_model_draws |=
            self.imported_refraction_enabled && self.model_refractive.contains(&handle_bits);
        let mesh_count = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes.len(),
            _ => return,
        };

        let foliage = self.foliage_wind.get(&handle_bits).copied().unwrap_or(0.0);
        let (s, c) = rot_y.sin_cos();
        // Column-major rotY (matches mat4_translate / mat4_scale layout).
        let rot: [[f32; 4]; 4] = [
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let model_matrix = mat4_multiply(
            mat4_translate(IDENTITY_MAT4, position),
            mat4_multiply(rot, mat4_scale(IDENTITY_MAT4, [scale, scale, scale])),
        );
        let previous_model = self.track_cached_model_motion(handle_bits, model_matrix);

        for mesh_idx in 0..mesh_count {
            let source_transform = self
                .model_gpu_cache
                .get(&handle_bits)
                .and_then(Option::as_ref)
                .map(|meshes| meshes[mesh_idx].source_transform)
                .unwrap_or(IDENTITY_MAT4);
            let mesh_model = mat4_multiply(model_matrix, source_transform);
            let previous_mesh_model = mat4_multiply(previous_model, source_transform);
            let model_mvp = mat4_multiply(self.current_vp_matrix, mesh_model);
            let previous_mvp = mat4_multiply(self.velocity_ref_vp, previous_mesh_model);
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;
            self.ensure_model_uniform_slot(slot);

            self.stage_model_uniform(
                slot,
                &Uniforms3D {
                    mvp: model_mvp,
                    model: mesh_model,
                    prev_mvp: previous_mvp,
                    model_tint: tint,
                    misc: [0.0, 0.0, foliage, 0.0],
                },
            );

            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
                model: mesh_model,
                tint,
                skinned: false,
                joint_offset: 0.0,
                bounds_override: None,
            });
        }
    }

    /// EN-039 — `draw_model_cached` taking the model matrix WHOLE, instead of
    /// composing it from a position/scale/yaw. `draw_model_cached_rotated` above
    /// can only express a Y rotation, so an immediate-mode caller could not
    /// pitch or roll: the shooter's weapon stays level while the camera looks up
    /// or down, and the same wall hits any held prop, thrown object or debris
    /// chunk. Everything downstream (alpha cutout, normal/MR maps, foliage wind,
    /// cutout shadows, planar reflections) is identical — this only skips the
    /// composition step and trusts the caller's matrix.
    pub fn draw_model_cached_transform(
        &mut self,
        handle_bits: u64,
        model_matrix: [[f32; 4]; 4],
        tint: [f32; 4],
    ) {
        self.has_blend_model_draws |= self.model_blended.contains(&handle_bits);
        self.has_layered_blend_model_draws |= self.model_layered_blended.contains(&handle_bits);
        self.has_refractive_model_draws |=
            self.imported_refraction_enabled && self.model_refractive.contains(&handle_bits);
        let mesh_count = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes.len(),
            _ => return,
        };

        let foliage = self.foliage_wind.get(&handle_bits).copied().unwrap_or(0.0);
        let previous_model = self.track_cached_model_motion(handle_bits, model_matrix);

        for mesh_idx in 0..mesh_count {
            let source_transform = self
                .model_gpu_cache
                .get(&handle_bits)
                .and_then(Option::as_ref)
                .map(|meshes| meshes[mesh_idx].source_transform)
                .unwrap_or(IDENTITY_MAT4);
            let mesh_model = mat4_multiply(model_matrix, source_transform);
            let previous_mesh_model = mat4_multiply(previous_model, source_transform);
            let model_mvp = mat4_multiply(self.current_vp_matrix, mesh_model);
            let previous_mvp = mat4_multiply(self.velocity_ref_vp, previous_mesh_model);
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;
            self.ensure_model_uniform_slot(slot);

            self.stage_model_uniform(
                slot,
                &Uniforms3D {
                    mvp: model_mvp,
                    model: mesh_model,
                    prev_mvp: previous_mvp,
                    model_tint: tint,
                    misc: [0.0, 0.0, foliage, 0.0],
                },
            );

            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
                model: mesh_model,
                tint,
                skinned: false,
                joint_offset: 0.0,
                bounds_override: None,
            });
        }
    }

    /// Cached-path draw for a SKINNED model: the bind-pose VB/IB stay
    /// GPU-resident (raw joint indices) and the scene VS skins from the
    /// shared joint buffer — replacing the immediate path's per-frame
    /// CPU re-transform + whole-batch vertex re-upload. Consumes the
    /// staged pose ONCE for the whole model and shares its joint-buffer
    /// offset across every primitive via the per-draw uniform's misc.x
    /// (per-primitive pops starved multi-primitive models onto offset 0
    /// — the marauder/tyrant blob; see `take_staged_skin_offset`).
    ///
    /// Rotation-less by design: joint matrices already bake the model's
    /// world orientation (same contract as the old immediate path).
    pub fn draw_model_cached_skinned(
        &mut self,
        handle_bits: u64,
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
    ) {
        self.has_blend_model_draws |= self.model_blended.contains(&handle_bits);
        self.has_layered_blend_model_draws |= self.model_layered_blended.contains(&handle_bits);
        self.has_refractive_model_draws |=
            self.imported_refraction_enabled && self.model_refractive.contains(&handle_bits);
        // Union of the cached meshes' local AABBs = the model's rest-pose
        // AABB (sentinel min > max when every mesh is empty).
        let (mesh_count, rest_min, rest_max) = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => {
                let mut rmin = [f32::MAX; 3];
                let mut rmax = [f32::MIN; 3];
                for mesh in meshes.iter() {
                    if mesh.local_min[0] > mesh.local_max[0] {
                        continue;
                    }
                    for a in 0..3 {
                        if mesh.local_min[a] < rmin[a] {
                            rmin[a] = mesh.local_min[a];
                        }
                        if mesh.local_max[a] > rmax[a] {
                            rmax[a] = mesh.local_max[a];
                        }
                    }
                }
                (meshes.len(), rmin, rmax)
            }
            _ => return,
        };
        if mesh_count == 0 {
            return;
        }

        let joint_offset = self.take_staged_skin_offset().unwrap_or(0.0);

        // Model matrix places the rare rigid (weightless) verts; weighted
        // verts get their world placement from the joint matrices.
        let model_matrix = mat4_multiply(
            mat4_translate(IDENTITY_MAT4, position),
            mat4_scale(IDENTITY_MAT4, [scale, scale, scale]),
        );

        // Rigorous world AABB: a skinned vertex is a convex blend of its
        // per-joint transforms of ONE rest position, so the union of every
        // joint matrix applied to the rest AABB bounds all weighted verts
        // (same argument as `finish_model_segment`). Union in the plain
        // model-matrix transform too so rigid verts are covered. The
        // model's joints occupy frame_joint_data[joint_offset..len] right
        // now: its group was staged just above and the next model's group
        // hasn't been staged yet.
        let (mut wmin, mut wmax) = super::transform_aabb(&model_matrix, rest_min, rest_max);
        let jstart = (joint_offset.max(0.0) as usize).min(self.frame_joint_data.len());
        for j in jstart..self.frame_joint_data.len() {
            let (m0, m1) = super::transform_aabb(&self.frame_joint_data[j], rest_min, rest_max);
            for a in 0..3 {
                if m0[a] < wmin[a] {
                    wmin[a] = m0[a];
                }
                if m1[a] > wmax[a] {
                    wmax[a] = m1[a];
                }
            }
        }
        // Sentinel (min > max) → no bounds; the passes treat the draw as
        // uncullable rather than culling it away.
        let bounds_override = if wmin[0] <= wmax[0] {
            Some((wmin, wmax))
        } else {
            None
        };

        let vp = self.current_vp_matrix;
        for mesh_idx in 0..mesh_count {
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;
            self.ensure_model_uniform_slot(slot);

            // Skinned draws: mvp is the BARE view-projection — the
            // joint matrices bake world placement, so the VS goes
            // joint-space → world → clip without a model term.
            // prev_mvp is the EN-022 velocity-reference VP (previous
            // frame, jitter-cancelled): with the previous palette in
            // the VS this yields REAL skeletal+locomotion velocity
            // (it used to be the current VP -> exactly zero velocity,
            // which is why enemies ghosted under TAA/TSR).
            self.stage_model_uniform(
                slot,
                &Uniforms3D {
                    mvp: vp,
                    model: model_matrix,
                    prev_mvp: self.velocity_ref_vp,
                    model_tint: tint,
                    misc: [joint_offset, 1.0, 0.0, 0.0],
                },
            );

            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
                model: model_matrix,
                tint,
                skinned: true,
                joint_offset,
                bounds_override,
            });

            // PT-6 — register the mesh for the dynamic TLAS path: a
            // compute pass skins it into the PT megabuffer and a
            // per-frame BLAS build makes it visible to every ray
            // (traced shadows, bounce light, reflections).
            if self.hw_rt_enabled {
                let (wmin, wmax) = bounds_override.unwrap_or(([0.0; 3], [-1.0; 3]));
                self.pt_dynamic_draws.push(super::PtDynamicDraw {
                    cache_handle: handle_bits,
                    mesh_idx,
                    joint_offset,
                    model: model_matrix,
                    wmin,
                    wmax,
                });
            }
        }
    }

    /// Bind group for one 256 B slot of the pooled model uniform buffer.
    pub(super) fn model_uniform_bg_for_slot(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        pool: &wgpu::Buffer,
        slot: usize,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model_uniform_bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: pool,
                    offset: (slot * MODEL_UNIFORM_STRIDE) as u64,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<Uniforms3D>() as u64),
                }),
            }],
        })
    }

    /// Pack a draw's uniforms into the CPU scratch; `flush_model_uniforms`
    /// uploads the whole used range in one `write_buffer` at end-frame.
    fn stage_model_uniform(&mut self, slot: usize, uniforms: &Uniforms3D) {
        self.ensure_model_uniform_slot(slot);
        let off = slot * MODEL_UNIFORM_STRIDE;
        if self.model_uniform_scratch.len() < off + MODEL_UNIFORM_STRIDE {
            self.model_uniform_scratch
                .resize(off + MODEL_UNIFORM_STRIDE, 0);
        }
        self.model_uniform_scratch[off..off + std::mem::size_of::<Uniforms3D>()]
            .copy_from_slice(bytemuck::bytes_of(uniforms));
    }

    /// Upload every staged cached-model uniform in one write. Called once
    /// per frame from the end-frame paths, before passes execute (queued
    /// writes land at submit, ahead of all encoded passes).
    pub(super) fn flush_model_uniforms(&mut self) {
        if self.next_model_uniform_slot == 0 {
            return;
        }
        let used = (self.next_model_uniform_slot * MODEL_UNIFORM_STRIDE)
            .min(self.model_uniform_scratch.len());
        if used > 0 {
            self.queue.write_buffer(
                &self.model_uniform_pool,
                0,
                &self.model_uniform_scratch[..used],
            );
        }
    }

    fn ensure_model_uniform_slot(&mut self, slot: usize) {
        if slot < self.model_uniform_pool_capacity {
            return;
        }
        // Grow by doubling: new pool + rebuild every slot bind group
        // (rare — only when a scene exceeds the previous high-water mark).
        let new_cap = (slot + 1)
            .next_power_of_two()
            .max(self.model_uniform_pool_capacity * 2);
        self.model_uniform_pool = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model_uniform_pool"),
            size: (new_cap * MODEL_UNIFORM_STRIDE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.model_uniform_bind_groups = (0..new_cap)
            .map(|s| {
                Self::model_uniform_bg_for_slot(
                    &self.device,
                    &self.uniform_3d_layout,
                    &self.model_uniform_pool,
                    s,
                )
            })
            .collect();
        self.model_uniform_pool_capacity = new_cap;
    }

    pub fn draw_model_mesh(
        &mut self,
        vertices: &[Vertex3D],
        indices: &[u32],
        position: [f32; 3],
        scale: f32,
    ) {
        self.draw_model_mesh_tinted(vertices, indices, position, scale, [1.0, 1.0, 1.0, 1.0], 0);
    }

    /// True if any vertex carries skin weights (same test the cache and
    /// the per-vertex draw loops use).
    fn mesh_has_skin(vertices: &[Vertex3D]) -> bool {
        vertices
            .iter()
            .any(|v| v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01)
    }

    /// Pop the next staged skin pose (FIFO) and pack it into the frame
    /// joint accumulator, returning the base slot offset its matrices
    /// were packed at. `None` when nothing is staged.
    ///
    /// drawModel-level callers must call this ONCE per model and share
    /// the offset across every primitive: when each primitive popped its
    /// own entry, the second primitive of a multi-primitive skinned model
    /// (marauder, tyrant) found the FIFO empty, fell back to offset 0 and
    /// got skinned by whatever pose lives at the start of the joint
    /// buffer — the player's — rendering as a giant mangled blob glued to
    /// the player's position.
    pub fn take_staged_skin_offset(&mut self) -> Option<f32> {
        if self.pending_skin_groups.is_empty() {
            return None;
        }
        let group = self.pending_skin_groups.remove(0);
        // PT-7 — the prev-palette queue moves in lockstep so both
        // arenas assign the SAME offset to this draw. A length
        // mismatch (or an unpaired group from a legacy caller) falls
        // back to the current pose = zero skeletal velocity.
        let prev_group = if !self.pending_skin_groups_prev.is_empty() {
            let p = self.pending_skin_groups_prev.remove(0);
            if p.len() == group.len() {
                p
            } else {
                group.clone()
            }
        } else {
            group.clone()
        };
        let start = self.frame_joint_data.len();
        // Cap at the 1024-slot buffer. Overflowing poses land at offset 0,
        // which at least avoids an out-of-range read — the model will look
        // mis-posed but not corrupt memory.
        if start + group.len() <= 1024 {
            self.frame_joint_data.extend_from_slice(&group);
            self.frame_joint_data_prev.extend_from_slice(&prev_group);
            Some(start as f32)
        } else {
            Some(0.0)
        }
    }

    /// Same as `draw_model_mesh_tinted` but applies a Y-axis rotation
    /// (radians) to the mesh local space before scale + translate.
    /// Skinned meshes ignore the rotation here — pose joints already
    /// drive their orientation. CPU-side baking mirrors the unrotated
    /// path so callers can mix rotated and unrotated draws freely
    /// without extra GPU state.
    pub fn draw_model_mesh_tinted_rotated(
        &mut self,
        vertices: &[Vertex3D],
        indices: &[u32],
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
        texture_idx: u32,
        rot_y: f32,
    ) {
        // Mirror the joint-pose plumbing in the non-rotated path so a
        // skinned mesh drawn here still consumes its pending pose.
        let joint_offset = if Self::mesh_has_skin(vertices) {
            self.take_staged_skin_offset()
        } else {
            None
        };
        self.draw_model_mesh_tinted_rotated_with_joints(
            vertices,
            indices,
            position,
            scale,
            tint,
            texture_idx,
            rot_y,
            joint_offset,
        );
    }

    /// Rotated-path body with an explicit joint-buffer offset. Callers
    /// that draw multiple primitives of ONE skinned model pop the staged
    /// pose once (`take_staged_skin_offset`) and pass the same offset to
    /// every primitive.
    pub fn draw_model_mesh_tinted_rotated_with_joints(
        &mut self,
        vertices: &[Vertex3D],
        indices: &[u32],
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
        texture_idx: u32,
        rot_y: f32,
        joint_offset: Option<f32>,
    ) {
        // Own bounded segment (even if the texture matches) so the shadow
        // pass can cull + cache this draw independently of neighbours.
        self.push_draw_call_3d(texture_idx, true);
        let joint_offset: f32 = joint_offset.unwrap_or(0.0);

        let cos_y = rot_y.cos();
        let sin_y = rot_y.sin();
        let base = self.vertices_3d.len() as u32;
        let mut seg = SegBounds::new();
        for v in vertices {
            let is_skinned = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01;
            let pos = if is_skinned {
                v.position
            } else {
                // Rotate local-space position around Y, then scale + translate.
                let lx = v.position[0];
                let ly = v.position[1];
                let lz = v.position[2];
                let rx = cos_y * lx + sin_y * lz;
                let rz = -sin_y * lx + cos_y * lz;
                [
                    rx * scale + position[0],
                    ly * scale + position[1],
                    rz * scale + position[2],
                ]
            };
            // Rotate the surface normal too so lighting matches the new
            // orientation. Y-axis rotation leaves normal.y untouched.
            let n = v.normal;
            let normal = if is_skinned {
                n
            } else {
                [
                    cos_y * n[0] + sin_y * n[2],
                    n[1],
                    -sin_y * n[0] + cos_y * n[2],
                ]
            };
            // Rotate tangent.xyz the same way; preserve handedness in w.
            let t = v.tangent;
            let tangent = if is_skinned {
                t
            } else {
                [
                    cos_y * t[0] + sin_y * t[2],
                    t[1],
                    -sin_y * t[0] + cos_y * t[2],
                    t[3],
                ]
            };
            let joints_out = if is_skinned {
                [
                    v.joints[0] + joint_offset,
                    v.joints[1] + joint_offset,
                    v.joints[2] + joint_offset,
                    v.joints[3] + joint_offset,
                ]
            } else {
                v.joints
            };
            seg.note(is_skinned, if is_skinned { v.position } else { pos });
            self.vertices_3d.push(Vertex3D {
                position: pos,
                normal,
                color: [
                    v.color[0] * tint[0],
                    v.color[1] * tint[1],
                    v.color[2] * tint[2],
                    v.color[3] * tint[3],
                ],
                uv: v.uv,
                joints: joints_out,
                weights: v.weights,
                tangent,
            });
        }
        for &idx in indices {
            self.indices_3d.push(base + idx);
        }
        self.finish_model_segment(seg, joint_offset);
    }

    pub fn draw_model_mesh_tinted(
        &mut self,
        vertices: &[Vertex3D],
        indices: &[u32],
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
        texture_idx: u32,
    ) {
        // If this mesh is skinned, consume the next pending pose
        // (FIFO) and pack its matrices into the frame accumulator at
        // the current cursor. Each vertex's joint indices then get
        // shifted by that cursor so the shader samples this mesh's
        // slice of the shared joint buffer. With a 1024-slot buffer,
        // multiple skinned models can coexist in one frame.
        let joint_offset = if Self::mesh_has_skin(vertices) {
            self.take_staged_skin_offset()
        } else {
            None
        };
        self.draw_model_mesh_tinted_with_joints(
            vertices,
            indices,
            position,
            scale,
            tint,
            texture_idx,
            joint_offset,
        );
    }

    /// Body of `draw_model_mesh_tinted` with an explicit joint-buffer
    /// offset. Callers that draw multiple primitives of ONE skinned model
    /// pop the staged pose once (`take_staged_skin_offset`) and pass the
    /// same offset to every primitive.
    pub fn draw_model_mesh_tinted_with_joints(
        &mut self,
        vertices: &[Vertex3D],
        indices: &[u32],
        position: [f32; 3],
        scale: f32,
        tint: [f32; 4],
        texture_idx: u32,
        joint_offset: Option<f32>,
    ) {
        // Own bounded segment — see the rotated variant.
        self.push_draw_call_3d(texture_idx, true);
        let joint_offset: f32 = joint_offset.unwrap_or(0.0);

        let base = self.vertices_3d.len() as u32;
        let mut seg = SegBounds::new();
        for v in vertices {
            // Check if vertex is skinned (has non-zero weights)
            let is_skinned = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01;
            let pos = if is_skinned {
                // Skinned: pass raw bind-pose positions — joint matrices handle transform
                v.position
            } else {
                // Unskinned: apply CPU-side position + scale
                [
                    v.position[0] * scale + position[0],
                    v.position[1] * scale + position[1],
                    v.position[2] * scale + position[2],
                ]
            };
            let joints_out = if is_skinned {
                [
                    v.joints[0] + joint_offset,
                    v.joints[1] + joint_offset,
                    v.joints[2] + joint_offset,
                    v.joints[3] + joint_offset,
                ]
            } else {
                v.joints
            };
            seg.note(is_skinned, if is_skinned { v.position } else { pos });
            self.vertices_3d.push(Vertex3D {
                position: pos,
                normal: v.normal,
                color: [
                    v.color[0] * tint[0],
                    v.color[1] * tint[1],
                    v.color[2] * tint[2],
                    v.color[3] * tint[3],
                ],
                uv: v.uv,
                joints: joints_out,
                weights: v.weights,
                tangent: v.tangent,
            });
        }
        for &idx in indices {
            self.indices_3d.push(base + idx);
        }
        self.finish_model_segment(seg, joint_offset);
    }

    /// Fold a model draw's accumulated bounds into its (freshly-pushed)
    /// segment. Skinned content is bounded by the union of its joint
    /// matrices applied to the rest-pose AABB — rigorous, because a
    /// skinned vertex is a convex combination of its per-joint
    /// transforms of one rest position, and a convex combination of
    /// points inside a set of AABBs stays inside their union AABB.
    fn finish_model_segment(&mut self, seg: SegBounds, joint_offset: f32) {
        let mut wmin = seg.world_min;
        let mut wmax = seg.world_max;
        if seg.any_skinned && seg.rest_min[0] <= seg.rest_max[0] {
            // The model's joints occupy frame_joint_data[joint_offset..len]
            // at this point: its group was staged immediately before its
            // primitives and the next model's group hasn't been staged yet.
            let jstart = (joint_offset.max(0.0) as usize).min(self.frame_joint_data.len());
            for j in jstart..self.frame_joint_data.len() {
                let (m0, m1) =
                    super::transform_aabb(&self.frame_joint_data[j], seg.rest_min, seg.rest_max);
                for a in 0..3 {
                    if m0[a] < wmin[a] {
                        wmin[a] = m0[a];
                    }
                    if m1[a] > wmax[a] {
                        wmax[a] = m1[a];
                    }
                }
            }
        }
        let call = self
            .draw_calls_3d
            .last_mut()
            .expect("finish_model_segment: segment was pushed at fn start");
        call.has_skinned = seg.any_skinned;
        call.content_hash = seg.hash;
        // Sentinel stays (min > max) when nothing contributed bounds —
        // the shadow pass then treats the segment as uncullable.
        if wmin[0] <= wmax[0] {
            call.wmin = wmin;
            call.wmax = wmax;
        }
    }
}

#[cfg(test)]
mod subset_tests {
    use super::cached_model_subset_is_canonical;

    #[test]
    fn cached_model_subset_preflight_is_atomic_and_canonical() {
        assert!(cached_model_subset_is_canonical(5, [].into_iter()));
        assert!(cached_model_subset_is_canonical(5, [0, 2, 4].into_iter()));
        assert!(!cached_model_subset_is_canonical(5, [0, 5].into_iter()));
        assert!(!cached_model_subset_is_canonical(5, [2, 2].into_iter()));
        assert!(!cached_model_subset_is_canonical(5, [3, 1].into_iter()));
    }
}

/// Per-model-draw bounds accumulator: world AABB + content hash for the
/// non-skinned verts (already world-space at append time), rest-pose
/// AABB for the skinned ones (their world placement lives in the joint
/// matrices — see `finish_model_segment`).
struct SegBounds {
    world_min: [f32; 3],
    world_max: [f32; 3],
    rest_min: [f32; 3],
    rest_max: [f32; 3],
    any_skinned: bool,
    hash: u64,
}

impl SegBounds {
    fn new() -> Self {
        SegBounds {
            world_min: [f32::MAX; 3],
            world_max: [f32::MIN; 3],
            rest_min: [f32::MAX; 3],
            rest_max: [f32::MIN; 3],
            any_skinned: false,
            hash: super::types::FNV_OFFSET,
        }
    }

    #[inline]
    fn note(&mut self, is_skinned: bool, pos: [f32; 3]) {
        if is_skinned {
            self.any_skinned = true;
            for a in 0..3 {
                if pos[a] < self.rest_min[a] {
                    self.rest_min[a] = pos[a];
                }
                if pos[a] > self.rest_max[a] {
                    self.rest_max[a] = pos[a];
                }
            }
        } else {
            for a in 0..3 {
                if pos[a] < self.world_min[a] {
                    self.world_min[a] = pos[a];
                }
                if pos[a] > self.world_max[a] {
                    self.world_max[a] = pos[a];
                }
            }
            self.hash = super::types::fnv1a_bytes(self.hash, bytemuck::bytes_of(&pos));
        }
    }
}
