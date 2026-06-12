//! Hi-Z (hierarchical linear-depth) pyramid build: a linearize pass
//! into mip 0 followed by min-downsample passes. Consumed by SSAO/SSR
//! ray-march acceleration and (via the max-reduce in occlusion.rs) the
//! occlusion-culling grid. Split out of renderer/mod.rs (2000-line file
//! policy); pipelines and the mip chain stay fields on [`Renderer`].

use super::formats::HIZ_MIP_COUNT;
use super::{HizDownsampleParams, HizLinearizeParams};
use super::Renderer;

impl Renderer {
    pub(super) fn record_hiz_chain(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        half_w: u32,
        half_h: u32,
        p22: f32,
        p32: f32,
    ) {
        // --- Hi-Z build: linearize depth into mip 0 -----------------
        let lin_params = HizLinearizeParams {
            params: [1.0 / half_w as f32, 1.0 / half_h as f32, p22, p32],
            size: [half_w, half_h, 0, 0],
        };
        self.queue.write_buffer(&self.hiz_linearize_uniform_buffer, 0, bytemuck::bytes_of(&lin_params));
        if self.hiz_linearize_bg_cache.is_none() {
            self.hiz_linearize_bg_cache = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hiz_linearize_bg"),
                layout: &self.hiz_linearize_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.hiz_linearize_uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssao_depth_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hiz_views[0]) },
                ],
            }));
        }
        {
            let bg = self.hiz_linearize_bg_cache.as_ref().unwrap();
            let ts = profiler.compute_pass_timestamp_writes("hiz_linearize_pass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_linearize_pass"),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.hiz_linearize_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((half_w + 7) / 8, (half_h + 7) / 8, 1);
        }

        // --- Hi-Z build: downsample mip i -> mip i+1 ----------------
        for i in 0..(HIZ_MIP_COUNT - 1) as usize {
            let dst_w = (half_w >> (i + 1)).max(1);
            let dst_h = (half_h >> (i + 1)).max(1);
            let ds_params = HizDownsampleParams {
                size: [dst_w, dst_h, 0, 0],
            };
            self.queue.write_buffer(&self.hiz_downsample_uniform_buffers[i], 0, bytemuck::bytes_of(&ds_params));
            if self.hiz_downsample_bg_cache[i].is_none() {
                self.hiz_downsample_bg_cache[i] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hiz_downsample_bg"),
                    layout: &self.hiz_downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.hiz_downsample_uniform_buffers[i].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hiz_views[i]) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.hiz_views[i + 1]) },
                    ],
                }));
            }
            let bg = self.hiz_downsample_bg_cache[i].as_ref().unwrap();
            let ts_label: &'static str = match i {
                0 => "hiz_downsample_pass_1",
                1 => "hiz_downsample_pass_2",
                2 => "hiz_downsample_pass_3",
                _ => "hiz_downsample_pass_4",
            };
            let ts = profiler.compute_pass_timestamp_writes(ts_label);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(ts_label),
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.hiz_downsample_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
        }

    }
}
