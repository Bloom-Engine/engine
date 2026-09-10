//! Explicit raw phase captures for locating the first changing SSGI stage.

use super::*;

pub(super) fn capture(eng: &EngineState, root: &Path) {
    let renderer = &eng.renderer;
    let directory = root.join(format!("frame-{:04}", renderer.probe_frame_index));
    std::fs::create_dir_all(&directory).expect("phase diagnostics directory");
    let width = renderer.probe_grid_w;
    let height = renderer.probe_grid_h;
    let header_bytes = u64::from(width * height) * 112;
    let row_bytes = (width * 8).div_ceil(256) * 256;
    let texture_bytes = u64::from(row_bytes * height * 64);
    let device = &renderer.device;
    let staging = |label, size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let headers = staging("ssgi-phase-headers", header_bytes);
    let trace = staging("ssgi-phase-trace", texture_bytes);
    let history = staging("ssgi-phase-history", texture_bytes);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ssgi-phase-diagnostics"),
    });
    encoder.copy_buffer_to_buffer(&renderer.probe_header_buffer, 0, &headers, 0, header_bytes);
    let latest_history = 1 - renderer.probe_history_idx;
    for (texture, buffer) in [
        (&renderer.probe_trace_tex, &trace),
        (&renderer.probe_history_textures[latest_history], &history),
    ] {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 64,
            },
        );
    }
    renderer.queue.submit([encoder.finish()]);
    for (name, buffer) in [
        ("headers", &headers),
        ("trace", &trace),
        ("history", &history),
    ] {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("phase map result");
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("phase readback completion");
        rx.recv().expect("phase map callback").expect("phase map");
        std::fs::write(
            directory.join(format!("{name}.bin")),
            &slice.get_mapped_range(),
        )
        .expect("phase buffer capture");
        buffer.unmap();
    }
    let metadata = serde_json::json!({
        "frame_index_after_submit": renderer.probe_frame_index,
        "traced_phase": renderer.probe_frame_index.wrapping_sub(1) & 15,
        "width": width, "height": height, "layers": 64,
        "bytes_per_row": row_bytes, "header_bytes": 112,
        "latest_history": latest_history,
    });
    std::fs::write(
        directory.join("layout.json"),
        serde_json::to_vec_pretty(&metadata).expect("phase metadata"),
    )
    .expect("phase metadata capture");
}
