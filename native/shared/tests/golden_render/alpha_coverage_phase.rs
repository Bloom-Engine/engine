use super::*;
use wgpu::util::DeviceExt;

fn phase_function(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let begin = source.find("fn mask_coverage_threshold(").unwrap();
    let end = begin + source[begin..].find("\n}\n").unwrap() + 3;
    source[begin..end].replace("mask_coverage_threshold", name)
}

#[test]
fn cutout_phase_preserves_the_approved_grid_on_scene_and_shadow_paths() {
    let Some(eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let device = &eng.renderer.device;
    let queue = &eng.renderer.queue;
    // UV, LOD, padding, texture dimensions, approved Bayer rank, padding.
    // Fixed reference points cover power-of-two and odd extents, UV wrapping,
    // fractional LOD, one-texel axes, and shifts beyond the integer bit width.
    let cases: [[f32; 8]; 12] = [
        [0.25, 0.25, 1.0, 0.0, 1024.0, 1024.0, 5.0, 0.0],
        [0.25, 0.5, 2.0, 0.0, 1024.0, 512.0, 5.0, 0.0],
        [0.25, 0.25, 1.0, 0.0, 513.0, 257.0, 0.0, 0.0],
        [0.75, 0.75, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.25, 0.25, 1000.0, 0.0, 1024.0, 1024.0, 0.0, 0.0],
        [-0.75, 1.25, 1.0, 0.0, 1024.0, 1024.0, 5.0, 0.0],
        [0.25, 0.25, 1.99, 0.0, 1024.0, 1024.0, 5.0, 0.0],
        [0.25, 0.25, 0.0, 0.0, 1024.0, 1024.0, 5.0, 0.0],
        [0.5, 0.5, 1.0, 0.0, 8.0, 8.0, 4.0, 0.0],
        [0.5, 0.5, 2.0, 0.0, 8.0, 8.0, 0.0, 0.0],
        [0.5, 0.5, 3.0, 0.0, 193.0, 65.0, 0.0, 0.0],
        [0.25, 0.25, 1.0, 0.0, 1.0, 1024.0, 15.0, 0.0],
    ];
    let source = format!(
        "{}\n{}\n{}",
        phase_function(
            include_str!("../../src/renderer/shaders/core.rs"),
            "scene_phase"
        ),
        phase_function(include_str!("../../src/shadows.rs"), "shadow_phase"),
        r#"
struct Case { coordinate: vec4<f32>, extent: vec4<f32> };
@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@compute @workgroup_size(64)
fn check_phase(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&cases)) { return; }
    let c = cases[id.x];
    output[id.x] = vec2<f32>(
        scene_phase(c.coordinate.xy, vec2<u32>(c.extent.xy), c.coordinate.z),
        shadow_phase(c.coordinate.xy, vec2<u32>(c.extent.xy), c.coordinate.z),
    );
}
"#,
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cutout_phase_reference"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cutout_phase_reference"),
        layout: None,
        module: &shader,
        entry_point: Some("check_phase"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutout_phase_cases"),
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let size = cases.len() as u64 * 8;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cutout_phase_output"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cutout_phase_readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutout_phase_reference"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
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
    rx.recv().unwrap().expect("cutout phase readback");
    let bytes = slice.get_mapped_range();
    for (index, case) in cases.iter().enumerate() {
        let expected = (case[6] + 0.5) / 16.0;
        for (path, offset) in [("scene", index * 8), ("shadow", index * 8 + 4)] {
            let actual = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert_eq!(actual, expected, "{path} phase case {index}: {case:?}");
        }
    }
    drop(bytes);
    staging.unmap();
    eprintln!(
        "cutout phase: {} approved reference points pass on both paths",
        cases.len()
    );
}
