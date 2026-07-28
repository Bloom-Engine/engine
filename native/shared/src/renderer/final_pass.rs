//! Terminal composite, game post-pass stack, and 2D overlay recording.
//!
//! Kept in one module so the compiled frame graph can execute the complete
//! visible frame without growing `renderer/mod.rs`.

use super::{CompositeParams, Renderer};

impl Renderer {
    pub(super) fn record_final_composite_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        output_view: &wgpu::TextureView,
        exposure_dst_idx: usize,
    ) {
        profiler.begin("post_fx");
        let composite_source = self.composite_source();
        let composite_cache_index = composite_source.bind_group_cache_index(exposure_dst_idx);
        let params = CompositeParams {
            params: [
                self.tonemap_kind as f32,
                if self.auto_exposure && self.exposure_history_valid {
                    1.0
                } else {
                    0.0
                },
                self.manual_exposure,
                self.auto_exposure_key,
            ],
            filmic: [
                self.chromatic_aberration,
                self.vignette_strength,
                self.vignette_softness,
                self.grain_strength,
            ],
            misc: [self.taa_frame_index as f32, self.sharpen_strength, 0.0, 0.0],
        };
        self.queue.write_buffer(
            &self.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        if self.composite_bind_group_cache[composite_cache_index].is_none() {
            self.frame_resource_stats.created_bind_group(
                super::frame_resource_stats::BindGroupCreationSite::FinalComposite,
            );
            let composite_src_view = self.composite_source_view_for(composite_source);
            let composite_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite_bg"),
                layout: &self.composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(composite_src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.composite_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &self.exposure_views[exposure_dst_idx],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.ssao_blur_rt_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                    },
                ],
            });
            self.composite_bind_group_cache[composite_cache_index] = Some(composite_bg);
        }
        let composite_target_view = if self.post_passes.is_empty() {
            output_view
        } else {
            self.composite_ldr_rt_a_view.as_ref().unwrap_or(output_view)
        };
        {
            let timestamp_writes = profiler.pass_timestamp_writes("final_composite_pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: composite_target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(
                0,
                self.composite_bind_group_cache[composite_cache_index]
                    .as_ref()
                    .expect("final composite bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }

        let pass_count = self.post_passes.len();
        for index in 0..pass_count {
            let input_view = if index % 2 == 0 {
                self.composite_ldr_rt_a_view.as_ref().unwrap_or(output_view)
            } else {
                self.composite_ldr_rt_b_view.as_ref().unwrap_or(output_view)
            };
            let is_last = index == pass_count - 1;
            let target_view = if is_last {
                output_view
            } else if index % 2 == 0 {
                self.composite_ldr_rt_b_view.as_ref().unwrap_or(output_view)
            } else {
                self.composite_ldr_rt_a_view.as_ref().unwrap_or(output_view)
            };
            let input_slot = index % 2;
            if self.post_passes[index].bind_group_cache[input_slot].is_none() {
                self.frame_resource_stats.created_bind_group(
                    super::frame_resource_stats::BindGroupCreationSite::CustomPostPass,
                );
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("post_pass_bg"),
                    layout: &self.post_passes[index].bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.post_pass_depth_sampler),
                        },
                    ],
                });
                self.post_passes[index].bind_group_cache[input_slot] = Some(bind_group);
            }
            let post_pass = &self.post_passes[index];
            let timestamp_writes = profiler.pass_timestamp_writes("post_pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_post_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&post_pass.pipeline);
            pass.set_bind_group(
                0,
                post_pass.bind_group_cache[input_slot]
                    .as_ref()
                    .expect("custom post-pass bind group was initialized"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }
        profiler.end("post_fx");
    }

    pub(super) fn record_overlay_2d_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        output_view: &wgpu::TextureView,
    ) {
        profiler.begin("overlay_2d");
        if self.vertices_2d.is_empty() {
            profiler.end("overlay_2d");
            return;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_2d_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline_2d);
            pass.set_vertex_buffer(0, self.persistent_vb_2d.slice(..));
            pass.set_index_buffer(self.persistent_ib_2d.slice(..), wgpu::IndexFormat::Uint32);

            let index_count = self.indices_2d.len() as u32;
            for (index, call) in self.draw_calls_2d.iter().enumerate() {
                let next_start = self
                    .draw_calls_2d
                    .get(index + 1)
                    .map_or(index_count, |next| next.index_start);
                if next_start == call.index_start {
                    continue;
                }
                pass.set_bind_group(0, &self.uniform_bind_groups[call.uniform_idx as usize], &[]);
                if let Some(texture) = self.texture_bind_groups.get(call.texture_idx as usize) {
                    pass.set_bind_group(1, texture, &[]);
                }
                pass.draw_indexed(call.index_start..next_start, 0, 0..1);
            }
        }
        profiler.end("overlay_2d");
    }
}
