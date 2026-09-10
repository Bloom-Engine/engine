use super::*;
use bloom_shared::profiler::Profiler;

fn resolved_pair(device: &wgpu::Device, queue: &wgpu::Queue, queries: &wgpu::QuerySet) -> [u64; 2] {
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp-reference-resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp-reference-readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    // Resolve the same recorded events again, after the measured submission.
    // Keep its host copy in a subsequent submission as an independent oracle.
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(queries, 0..2, &resolved, 0);
    queue.submit([encoder.finish()]);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&resolved, 0, &staging, 0, 16);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("timestamp reference completion");
    rx.recv().unwrap().expect("timestamp reference map");
    let bytes = slice.get_mapped_range();
    let pair = [
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
    ];
    drop(bytes);
    staging.unmap();
    pair
}

#[test]
fn profiler_reports_the_current_gpu_frame_from_its_first_sample() {
    let Some(eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let device = &eng.renderer.device;
    let queue = &eng.renderer.queue;
    if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        eprintln!("skip: timestamp-query feature unavailable");
        return;
    }
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("timestamp-frame-fixture"),
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut mismatches = 0;
    for generation in 0..3 {
        let mut profiler = Profiler::new();
        profiler.init_gpu(device, queue);
        profiler.set_enabled(true);
        for frame in 0..4 {
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("timestamp-frame-fixture"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: f64::from(frame) / 4.0,
                                g: 0.2,
                                b: 0.6,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    timestamp_writes: profiler.pass_timestamp_writes("current-frame"),
                    ..Default::default()
                });
            }
            profiler.resolve(&mut encoder);
            queue.submit([encoder.finish()]);
            profiler.frame_end(device);

            let pair = resolved_pair(device, queue, profiler.query_set().unwrap());
            assert!(
                pair[0] > 0 && pair[1] > pair[0],
                "invalid independent query pair: {pair:?}"
            );
            let expected_us =
                (pair[1] - pair[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1000.0;
            let reported_us = profiler.frame_history().last().unwrap().1;
            if reported_us != expected_us {
                mismatches += 1;
                eprintln!("timestamp mismatch generation={generation} frame={frame} expected_us={expected_us} reported_us={reported_us}");
            }
            let report: serde_json::Value = serde_json::from_str(&profiler.quality_report_json(
                3,
                0,
                frame + 1,
                1.0 / 60.0,
                3,
                1.0,
                1.0,
                "{}",
                "{}",
            ))
            .unwrap();
            assert_eq!(report["gpu_timing_valid"], true);
            assert_eq!(report["gpu_timing_valid_frames"], frame + 1);
        }
        // A reserved but unresolved frame cannot reuse the last valid result.
        profiler.reserve_gpu_pair("current-frame");
        profiler.frame_end(device);
        assert_eq!(profiler.snapshot()[0].2, None);
        let report: serde_json::Value = serde_json::from_str(&profiler.quality_report_json(
            3,
            0,
            5,
            1.0 / 60.0,
            3,
            1.0,
            1.0,
            "{}",
            "{}",
        ))
        .unwrap();
        assert_eq!(report["gpu_timing_valid"], false);
        assert_eq!(report["gpu_timing_valid_frames"], 4);
    }
    assert_eq!(
        mismatches, 0,
        "profiler reported stale or uninitialized GPU frame data"
    );
}
