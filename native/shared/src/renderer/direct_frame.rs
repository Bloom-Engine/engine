use super::Renderer;

impl Renderer {
    pub fn end_frame(&mut self) {
        if !self.material_per_view_bg_live {
            self.refresh_material_per_view_bg();
        }
        // Flush pending joint matrices to GPU right before rendering
        self.flush_joint_matrices();
        // One pooled upload for every cached-model draw's uniforms.
        self.flush_model_uniforms();

        // Q1: If rendering to a texture, use the RT view. Otherwise use the surface.
        // We take ownership of the RT views (via Option::take) to avoid holding a
        // borrow on `self` while the rest of end_frame mutates it.
        let rt_color = self.rt_color_view.take();
        let rt_depth = self.rt_depth_view.take();
        let using_rt = rt_color.is_some();

        let surface_output = if using_rt {
            None
        } else {
            match self.acquire_frame() {
                Some(t) => Some(t),
                None => {
                    // Swapchain lost+reconfigured. Restore RT views if set.
                    self.rt_color_view = rt_color;
                    self.rt_depth_view = rt_depth;
                    return;
                }
            }
        };

        let view: wgpu::TextureView;
        let owned_depth_view: wgpu::TextureView;

        if let Some(ref rt_view) = rt_color {
            view = rt_view.clone();
            owned_depth_view = rt_depth.as_ref().unwrap().clone();
        } else {
            view = self
                .frame_texture(surface_output.as_ref().unwrap())
                .create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.output_format),
                    ..Default::default()
                });
            owned_depth_view = self
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
        }

        // Restore RT views so they persist across frames.
        self.rt_color_view = rt_color;
        self.rt_depth_view = rt_depth;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bloom_encoder"),
            });
        self.frame_resource_stats.created_command_encoder();

        // Upload 2D data to persistent GPU buffers
        let has_2d = !self.vertices_2d.is_empty();
        if has_2d {
            let vb_size = std::mem::size_of_val(self.vertices_2d.as_slice());
            let ib_size = std::mem::size_of_val(self.indices_2d.as_slice());
            self.ensure_buffer_capacity_2d(vb_size, ib_size);
            self.queue.write_buffer(
                &self.persistent_vb_2d,
                0,
                bytemuck::cast_slice(&self.vertices_2d),
            );
            self.queue.write_buffer(
                &self.persistent_ib_2d,
                0,
                bytemuck::cast_slice(&self.indices_2d),
            );
        }

        // Upload 3D data to persistent GPU buffers
        let has_3d = !self.vertices_3d.is_empty();
        if has_3d {
            let vb_size = std::mem::size_of_val(self.vertices_3d.as_slice());
            let ib_size = std::mem::size_of_val(self.indices_3d.as_slice());
            self.ensure_buffer_capacity_3d(vb_size, ib_size);
            self.queue.write_buffer(
                &self.persistent_vb_3d,
                0,
                bytemuck::cast_slice(&self.vertices_3d),
            );
            self.queue.write_buffer(
                &self.persistent_ib_3d,
                0,
                bytemuck::cast_slice(&self.indices_3d),
            );
        }

        {
            // Only attach a depth target when we're drawing 3D. pipeline_2d is
            // depth-less; on some mobile Vulkan drivers (Adreno) pairing a
            // depth-less pipeline with a pass that carries a depth attachment
            // discards all draws silently. Matches the overlay_2d pass in
            // end_frame_with_scene, which also omits depth.
            let depth_attachment = if has_3d {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &owned_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                })
            } else {
                None
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Draw 3D geometry first (with depth testing), batched by texture
            if has_3d {
                pass.set_pipeline(&self.pipeline_3d);
                pass.set_bind_group(0, &self.uniform_bind_group_3d, &[]);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);
                pass.set_vertex_buffer(0, self.persistent_vb_3d.slice(..));
                pass.set_index_buffer(self.persistent_ib_3d.slice(..), wgpu::IndexFormat::Uint32);

                if self.draw_calls_3d.is_empty() {
                    // No draw calls tracked — draw all with white texture (backward compat)
                    pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                    pass.draw_indexed(0..self.indices_3d.len() as u32, 0, 0..1);
                } else {
                    let num_calls = self.draw_calls_3d.len();
                    for i in 0..num_calls {
                        let call = &self.draw_calls_3d[i];
                        let next_start = if i + 1 < num_calls {
                            self.draw_calls_3d[i + 1].index_start
                        } else {
                            self.indices_3d.len() as u32
                        };
                        let count = next_start - call.index_start;
                        if count == 0 {
                            continue;
                        }
                        let tex_idx = call.texture_idx as usize;
                        if tex_idx < self.texture_bind_groups.len() {
                            pass.set_bind_group(2, &self.texture_bind_groups[tex_idx], &[]);
                        } else {
                            pass.set_bind_group(2, &self.texture_bind_groups[0], &[]);
                        }
                        pass.draw_indexed(call.index_start..next_start, 0, 0..1);
                    }
                }
            }

            // Draw cached models (static models with GPU-resident buffers).
            // Use the scene pipeline so PBR-style material bindings (base
            // color + normal map) apply — drawModel should behave the same
            // as attachModelToNode for PBR purposes.
            if !self.model_draw_commands.is_empty() {
                pass.set_pipeline(&self.scene_pipeline);
                pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                pass.set_bind_group(3, &self.joint_bind_group, &[]);

                for cmd in &self.model_draw_commands {
                    if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                        if cmd.mesh_idx < meshes.len() {
                            let mesh = &meshes[cmd.mesh_idx];
                            let draw = self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count);
                            pass.set_bind_group(
                                0,
                                &self.model_uniform_bind_groups[cmd.uniform_slot],
                                &[],
                            );
                            pass.set_bind_group(2, &mesh.material_bg, &[]);
                            pass.set_vertex_buffer(0, draw.vertex.slice(..));
                            pass.set_index_buffer(draw.index.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(draw.index_range(), draw.base_vertex, 0..1);
                        }
                    }
                }
            }

            // Draw 2D geometry (no depth testing, always passes)
            if has_2d {
                pass.set_pipeline(&self.pipeline_2d);
                pass.set_vertex_buffer(0, self.persistent_vb_2d.slice(..));
                pass.set_index_buffer(self.persistent_ib_2d.slice(..), wgpu::IndexFormat::Uint32);

                let num_calls = self.draw_calls_2d.len();
                for i in 0..num_calls {
                    let call = &self.draw_calls_2d[i];
                    let next_start = if i + 1 < num_calls {
                        self.draw_calls_2d[i + 1].index_start
                    } else {
                        self.indices_2d.len() as u32
                    };
                    let count = next_start - call.index_start;
                    if count == 0 {
                        continue;
                    }

                    pass.set_bind_group(
                        0,
                        &self.uniform_bind_groups[call.uniform_idx as usize],
                        &[],
                    );
                    if (call.texture_idx as usize) < self.texture_bind_groups.len() {
                        pass.set_bind_group(
                            1,
                            &self.texture_bind_groups[call.texture_idx as usize],
                            &[],
                        );
                    }
                    pass.draw_indexed(call.index_start..next_start, 0, 0..1);
                }
            }
        }

        // A direct frame still owns the normal output texture. Service a PNG
        // request after drawing, just as the scene graph's terminal capture
        // pass does. Render-target-only frames and scene diagnostics stay
        // pending for an eligible output/scene frame.
        #[cfg(not(target_arch = "wasm32"))]
        let frame_readback = if self.screenshot_requested
            && self.pending_quality_capture_dir.is_none()
            && self.pending_mrt_capture_dir.is_none()
        {
            surface_output
                .as_ref()
                .map(|output| self.record_frame_readback(&mut encoder, self.frame_texture(output)))
        } else {
            None
        };

        self.flush_lighting_uniforms();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(readback) = frame_readback {
            self.queue.submit(std::iter::once(encoder.finish()));
            self.finish_frame_readback(readback);
        } else {
            self.submit_frame_commands(encoder.finish());
        }
        #[cfg(target_arch = "wasm32")]
        self.submit_frame_commands(encoder.finish());
        if let Some(out) = surface_output {
            self.present_frame(out);
        }
        self.finish_frame_resource_stats();
    }
}
