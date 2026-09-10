//! Analytic surface reconstruction through the production probe placement pass.

use super::*;
use wgpu::util::DeviceExt;

#[test]
fn ssgi_probe_receivers_stay_on_their_plane_at_every_resolution() {
    let Some(eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let renderer = &eng.renderer;
    let device = &renderer.device;
    let queue = &renderer.queue;
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    let mut checked = 0;
    for (width, height) in [(128_u32, 128_u32), (640, 360), (641, 361), (1920, 1080)] {
        // Analytic plane: dot((0, 0.6, 0.8), position) == -2, with unit
        // projection diagonals and an identity view. Every depth sample is
        // evaluated at its own texel center, independently of probe placement.
        let depth: Vec<f32> = (0..width * height)
            .map(|index| {
                let ndc_y = 1.0 - 2.0 * ((index / width) as f32 + 0.5) / height as f32;
                2.0 / (0.8 - 0.6 * ndc_y)
            })
            .collect();
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("analytic_probe_plane"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&depth),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            extent,
        );
        let view = texture.create_view(&Default::default());
        let grid_w = width.div_ceil(8);
        let grid_h = height.div_ceil(8);
        let size = u64::from(grid_w * grid_h) * 112;
        let probes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("analytic_plane_probes"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("analytic_plane_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for phase in [0_f32, 5.0, 10.0, 15.0] {
            // PlaceParams: inverse view, projection diagonals/offsets, extents,
            // and phase/tile/moving. Four phases exercise fractional positions
            // in both axes, including either side of a texel center.
            let mut uniform = [0_u32; 28];
            for index in [0, 5, 10, 15, 16, 17, 26] {
                uniform[index] = 1_f32.to_bits();
            }
            uniform[20..24].copy_from_slice(&[width, height, grid_w, grid_h]);
            uniform[24] = phase.to_bits();
            uniform[25] = 8_f32.to_bits();
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("analytic_plane_params"),
                contents: bytemuck::cast_slice(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("analytic_plane_placement"),
                layout: &renderer.probe_place_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: probes.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&renderer.probe_place_pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(grid_w.div_ceil(8), grid_h.div_ceil(8), 1);
            }
            encoder.copy_buffer_to_buffer(&probes, 0, &staging, 0, size);
            queue.submit([encoder.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().expect("analytic plane readback");
            let bytes = slice.get_mapped_range();
            let values: &[f32] = bytemuck::cast_slice(&bytes);
            let mut max_plane_error = 0_f32;
            let mut max_normal_error = 0_f32;
            // Interior receivers have all three depth taps. Border taps can
            // coincide after clamping and intentionally use a fallback normal.
            for y in 1..grid_h - 1 {
                for x in 1..grid_w - 1 {
                    let offset = ((y * grid_w + x) * 28) as usize;
                    let p = &values[offset..offset + 8];
                    assert!(p.iter().all(|value| value.is_finite()));
                    assert_eq!(p[3], 1.0, "analytic plane receiver is invalid");
                    max_plane_error = max_plane_error.max((0.6 * p[1] + 0.8 * p[2] + 2.0).abs());
                    max_normal_error = max_normal_error
                        .max(p[4].abs().max((p[5] - 0.6).abs()).max((p[6] - 0.8).abs()));
                    checked += 1;
                }
            }
            eprintln!("probe plane {width}x{height} phase={phase}: position_error={max_plane_error} normal_error={max_normal_error}");
            assert!(
                max_plane_error <= 0.00001,
                "probe receiver moved off its depth plane"
            );
            assert!(
                max_normal_error <= 0.0015,
                "probe normal changed with pixel footprint"
            );
            drop(bytes);
            staging.unmap();
        }
    }
    eprintln!("analytic probe plane: {checked} receiver checks passed");
}
