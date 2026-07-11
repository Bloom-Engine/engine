//! Model draw submission: cached-model draw commands and the
//! immediate-mode mesh path (incl. GPU-skinning joint-pose plumbing).
//! Split from renderer/mod.rs (2000-line file policy — same pattern as
//! shadow_pass.rs).

use super::*;

impl Renderer {
    /// Record a cached model draw command. The actual rendering happens in end_frame().
    pub fn draw_model_cached(&mut self, handle_bits: u64, position: [f32; 3], scale: f32, tint: [f32; 4]) {
        let mesh_count = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes.len(),
            _ => return,
        };

        for mesh_idx in 0..mesh_count {
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;

            // Grow uniform pool if needed
            self.ensure_model_uniform_slot(slot);

            // Compute model MVP: VP * translate(position) * scale(s)
            let model_matrix = mat4_multiply(
                mat4_translate(IDENTITY_MAT4, position),
                mat4_scale(IDENTITY_MAT4, [scale, scale, scale]),
            );
            let model_mvp = mat4_multiply(self.current_vp_matrix, model_matrix);

            // Write uniform for this draw
            self.queue.write_buffer(
                &self.model_uniform_buffers[slot],
                0,
                bytemuck::bytes_of(&Uniforms3D { mvp: model_mvp, model: model_matrix, prev_mvp: model_mvp, model_tint: tint }),
            );

            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
                model: model_matrix,
            });
        }
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
        let mesh_count = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes.len(),
            _ => return,
        };

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

        for mesh_idx in 0..mesh_count {
            let slot = self.next_model_uniform_slot;
            self.next_model_uniform_slot += 1;
            self.ensure_model_uniform_slot(slot);

            let model_mvp = mat4_multiply(self.current_vp_matrix, model_matrix);
            self.queue.write_buffer(
                &self.model_uniform_buffers[slot],
                0,
                bytemuck::bytes_of(&Uniforms3D { mvp: model_mvp, model: model_matrix, prev_mvp: model_mvp, model_tint: tint }),
            );

            self.model_draw_commands.push(CachedModelDraw {
                uniform_slot: slot,
                cache_handle: handle_bits,
                mesh_idx,
                model: model_matrix,
            });
        }
    }

    fn ensure_model_uniform_slot(&mut self, slot: usize) {
        while self.model_uniform_buffers.len() <= slot {
            let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("model_uniform"),
                contents: bytemuck::bytes_of(&Uniforms3D { mvp: IDENTITY_MAT4, model: IDENTITY_MAT4, prev_mvp: IDENTITY_MAT4, model_tint: [1.0; 4] }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("model_uniform_bg"),
                layout: &self.uniform_3d_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            self.model_uniform_buffers.push(buf);
            self.model_uniform_bind_groups.push(bg);
        }
    }

    pub fn draw_model_mesh(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32) {
        self.draw_model_mesh_tinted(vertices, indices, position, scale, [1.0, 1.0, 1.0, 1.0], 0);
    }

    /// True if any vertex carries skin weights (same test the cache and
    /// the per-vertex draw loops use).
    fn mesh_has_skin(vertices: &[Vertex3D]) -> bool {
        vertices.iter().any(|v|
            v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01)
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
        let start = self.frame_joint_data.len();
        // Cap at the 1024-slot buffer. Overflowing poses land at offset 0,
        // which at least avoids an out-of-range read — the model will look
        // mis-posed but not corrupt memory.
        if start + group.len() <= 1024 {
            self.frame_joint_data.extend_from_slice(&group);
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
    pub fn draw_model_mesh_tinted_rotated(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32, tint: [f32; 4], texture_idx: u32, rot_y: f32) {
        // Mirror the joint-pose plumbing in the non-rotated path so a
        // skinned mesh drawn here still consumes its pending pose.
        let joint_offset = if Self::mesh_has_skin(vertices) {
            self.take_staged_skin_offset()
        } else {
            None
        };
        self.draw_model_mesh_tinted_rotated_with_joints(
            vertices, indices, position, scale, tint, texture_idx, rot_y, joint_offset);
    }

    /// Rotated-path body with an explicit joint-buffer offset. Callers
    /// that draw multiple primitives of ONE skinned model pop the staged
    /// pose once (`take_staged_skin_offset`) and pass the same offset to
    /// every primitive.
    pub fn draw_model_mesh_tinted_rotated_with_joints(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32, tint: [f32; 4], texture_idx: u32, rot_y: f32, joint_offset: Option<f32>) {
        self.ensure_draw_state_3d(texture_idx);
        let joint_offset: f32 = joint_offset.unwrap_or(0.0);

        let cos_y = rot_y.cos();
        let sin_y = rot_y.sin();
        let base = self.vertices_3d.len() as u32;
        for v in vertices {
            let is_skinned = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01;
            let pos = if is_skinned {
                v.position
            } else {
                // Rotate local-space position around Y, then scale + translate.
                let lx = v.position[0];
                let ly = v.position[1];
                let lz = v.position[2];
                let rx =  cos_y * lx + sin_y * lz;
                let rz = -sin_y * lx + cos_y * lz;
                [rx * scale + position[0],
                 ly * scale + position[1],
                 rz * scale + position[2]]
            };
            // Rotate the surface normal too so lighting matches the new
            // orientation. Y-axis rotation leaves normal.y untouched.
            let n = v.normal;
            let normal = if is_skinned {
                n
            } else {
                [ cos_y * n[0] + sin_y * n[2],
                  n[1],
                 -sin_y * n[0] + cos_y * n[2] ]
            };
            // Rotate tangent.xyz the same way; preserve handedness in w.
            let t = v.tangent;
            let tangent = if is_skinned {
                t
            } else {
                [ cos_y * t[0] + sin_y * t[2],
                  t[1],
                 -sin_y * t[0] + cos_y * t[2],
                  t[3] ]
            };
            let joints_out = if is_skinned {
                [v.joints[0] + joint_offset,
                 v.joints[1] + joint_offset,
                 v.joints[2] + joint_offset,
                 v.joints[3] + joint_offset]
            } else {
                v.joints
            };
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
    }

    pub fn draw_model_mesh_tinted(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32, tint: [f32; 4], texture_idx: u32) {
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
            vertices, indices, position, scale, tint, texture_idx, joint_offset);
    }

    /// Body of `draw_model_mesh_tinted` with an explicit joint-buffer
    /// offset. Callers that draw multiple primitives of ONE skinned model
    /// pop the staged pose once (`take_staged_skin_offset`) and pass the
    /// same offset to every primitive.
    pub fn draw_model_mesh_tinted_with_joints(&mut self, vertices: &[Vertex3D], indices: &[u32], position: [f32; 3], scale: f32, tint: [f32; 4], texture_idx: u32, joint_offset: Option<f32>) {
        self.ensure_draw_state_3d(texture_idx);
        let joint_offset: f32 = joint_offset.unwrap_or(0.0);

        let base = self.vertices_3d.len() as u32;
        for v in vertices {
            // Check if vertex is skinned (has non-zero weights)
            let is_skinned = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3] > 0.01;
            let pos = if is_skinned {
                // Skinned: pass raw bind-pose positions — joint matrices handle transform
                v.position
            } else {
                // Unskinned: apply CPU-side position + scale
                [v.position[0] * scale + position[0],
                 v.position[1] * scale + position[1],
                 v.position[2] * scale + position[2]]
            };
            let joints_out = if is_skinned {
                [v.joints[0] + joint_offset,
                 v.joints[1] + joint_offset,
                 v.joints[2] + joint_offset,
                 v.joints[3] + joint_offset]
            } else {
                v.joints
            };
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
    }
}
