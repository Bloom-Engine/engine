use super::super::{gpu_driven::GpuDrawRecord, Uniforms3D, Vertex3D};
use super::*;
use wgpu::util::DeviceExt;

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "{actual} != {expected}"
    );
}

#[test]
fn packed_record_is_exactly_one_rg32uint_texel() {
    assert_eq!(std::mem::size_of::<VisibilityRecord>(), 8);
    assert_eq!(std::mem::align_of::<VisibilityRecord>(), 4);
    assert_eq!(VISIBILITY_BYTES_PER_PIXEL, 8);
    assert_eq!(VISIBILITY_FORMAT, wgpu::TextureFormat::Rg32Uint);
    assert_eq!(target_bytes(1_920, 1_080), Some(16_588_800));
    assert_eq!(target_bytes(u32::MAX, u32::MAX), None);

    let report = contract_json();
    assert!(report.starts_with("{\"format\":\"rg32uint\""));
    assert!(report.contains("\"native_1080p_bytes\":16588800"));
    assert!(report.contains("\"required_feature\":\"primitive-index\""));
    assert!(report.contains("\"vertex_stride_bytes\":96"));
    assert!(report.contains("\"shipping_enabled\":false"));
}

#[test]
fn ids_and_face_orientation_round_trip_without_background_collision() {
    for (draw, primitive, front) in [
        (0, 0, false),
        (17, 42, true),
        (DRAW_INDEX_MASK, PRIMITIVE_ID_MASK, false),
    ] {
        let encoded = VisibilityRecord::encode(draw, primitive, front).unwrap();
        assert_eq!(encoded.decode(), Some((draw, primitive, front)));
    }
    assert_eq!(VisibilityRecord::BACKGROUND.decode(), None);
    assert_eq!(VisibilityRecord::encode(INVALID_DRAW_ID, 0, true), None);
    assert_eq!(VisibilityRecord::encode(VIRTUAL_DRAW_BIT, 0, true), None);
    assert_eq!(VisibilityRecord::encode(0, FRONT_FACE_BIT, true), None);
}

#[test]
fn perspective_reconstruction_matches_vertices_and_known_depth_weighting() {
    let clip = [
        [-1.0, -1.0, 0.2, 1.0],
        [2.0, -2.0, 0.4, 2.0],
        [0.0, 4.0, 0.8, 4.0],
    ];
    for (point, expected) in [
        ([-1.0, -1.0], [1.0, 0.0, 0.0]),
        ([1.0, -1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0], [0.0, 0.0, 1.0]),
    ] {
        let actual = perspective_barycentrics(point, clip).unwrap();
        for lane in 0..3 {
            assert_close(actual[lane], expected[lane]);
        }
    }

    let center = perspective_barycentrics([0.0, -1.0 / 3.0], clip).unwrap();
    assert_close(center[0], 4.0 / 7.0);
    assert_close(center[1], 2.0 / 7.0);
    assert_close(center[2], 1.0 / 7.0);
    assert_close(center.iter().sum(), 1.0);
}

#[test]
fn shared_reconstruction_header_parses_and_keeps_the_cpu_abi_constants() {
    wgpu::naga::front::wgsl::parse_str(RECONSTRUCTION_WGSL)
        .unwrap_or_else(|error| panic!("visibility reconstruction WGSL failed: {error:?}"));
    wgpu::naga::front::wgsl::parse_str(GEOMETRY_WGSL)
        .unwrap_or_else(|error| panic!("visibility geometry WGSL failed: {error:?}"));
    assert!(
        RECONSTRUCTION_WGSL.contains("const BLOOM_VISIBILITY_FRONT_FACE_BIT: u32 = 0x80000000u")
    );
    assert!(RECONSTRUCTION_WGSL.contains("fn bloom_perspective_barycentrics("));
    assert!(GEOMETRY_WGSL.contains("const BLOOM_VERTEX3D_WORDS: u32 = 24u"));
    assert_eq!(std::mem::size_of::<Vertex3D>(), 96);
}

#[test]
fn runtime_modes_request_only_the_explicit_optional_feature() {
    assert_eq!(parse_runtime_mode(None), RuntimeMode::Off);
    assert_eq!(parse_runtime_mode(Some("off")), RuntimeMode::Off);
    assert_eq!(parse_runtime_mode(Some("validate")), RuntimeMode::Validate);
    assert_eq!(parse_runtime_mode(Some("DEBUG")), RuntimeMode::Debug);
    assert_eq!(parse_runtime_mode(Some("pbr")), RuntimeMode::Shade);
    assert!(RuntimeMode::Shade.shades());

    let supported = wgpu::Features::PRIMITIVE_INDEX | wgpu::Features::TIMESTAMP_QUERY;
    let mut required = wgpu::Features::empty();
    request_feature_for_mode(RuntimeMode::Off, supported, &mut required);
    assert!(required.is_empty());
    request_feature_for_mode(RuntimeMode::Validate, supported, &mut required);
    assert_eq!(required, wgpu::Features::PRIMITIVE_INDEX);
    request_feature_for_mode(RuntimeMode::Shade, supported, &mut required);
    assert_eq!(required, wgpu::Features::PRIMITIVE_INDEX);

    let mut unsupported = wgpu::Features::empty();
    request_feature_for_mode(
        RuntimeMode::Debug,
        wgpu::Features::TIMESTAMP_QUERY,
        &mut unsupported,
    );
    assert!(unsupported.is_empty());

    let disabled = VisibilityBufferRuntime::disabled(RuntimeMode::Off, "not-requested");
    assert!(!disabled.enabled());
    assert!(disabled.resources.is_none());
    assert!(disabled.report_json().contains("\"allocated_bytes\":0"));
}

#[test]
fn runtime_raster_reconstruction_and_overlay_shaders_parse() {
    let generated =
        super::super::gpu_driven::make_gpu_scene_shader(super::super::shaders::SCENE_SHADER);
    let raster = make_visibility_raster_shader(&generated);
    let depth = super::super::visibility_shading::make_visibility_depth_shader(&generated);
    wgpu::naga::front::wgsl::parse_str(&raster)
        .unwrap_or_else(|error| panic!("visibility runtime raster WGSL failed: {error:?}"));
    wgpu::naga::front::wgsl::parse_str(&depth)
        .unwrap_or_else(|error| panic!("visibility depth WGSL failed: {error:?}"));
    wgpu::naga::front::wgsl::parse_str(RUNTIME_RECONSTRUCT_WGSL)
        .unwrap_or_else(|error| panic!("visibility runtime reconstruction WGSL failed: {error:?}"));
    wgpu::naga::front::wgsl::parse_str(DEBUG_OVERLAY_WGSL)
        .unwrap_or_else(|error| panic!("visibility debug overlay WGSL failed: {error:?}"));
    assert!(raster.starts_with("enable primitive_index;"));
    assert!(raster.contains("out.draw_id = draw_index"));
    assert!(raster.contains("(in.draw_flags & 2u) == 0u"));
    assert!(depth.contains("return vec2<u32>(0xffffffffu, 0xffffffffu)"));
    assert!(RUNTIME_RECONSTRUCT_WGSL.contains("visibility_vertex_count()"));
    assert!(RUNTIME_RECONSTRUCT_WGSL.contains("@group(0) @binding(6)"));
}

#[test]
fn large_vertex_arenas_split_at_aligned_storage_binding_boundaries() {
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = 128 * 1024 * 1024;
    limits.min_storage_buffer_offset_alignment = 256;
    let vertex_stride = std::mem::size_of::<super::super::Vertex3D>() as u64;
    let used_bytes = 1_738_262 * vertex_stride;
    let ranges = visibility_vertex_binding_ranges(&limits, 256 * 1024 * 1024, used_bytes);
    assert_eq!(ranges[0], (0, 134_217_216));
    assert_eq!(ranges[1], (134_217_216, used_bytes - 134_217_216));
    assert_eq!(ranges[2], (0, vertex_stride));
    assert!(ranges
        .iter()
        .all(|&(_, size)| size <= u64::from(limits.max_storage_buffer_binding_size)));
    assert!(ranges
        .iter()
        .all(|&(offset, _)| offset % u64::from(limits.min_storage_buffer_offset_alignment) == 0));
}

#[cfg(not(target_arch = "wasm32"))]
fn try_device(required_features: wgpu::Features) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    if !adapter.features().contains(required_features) {
        eprintln!("adapter lacks required visibility-oracle features");
        return None;
    }
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("visibility_buffer_oracle_device"),
        required_features,
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    receiver
        .recv()
        .expect("visibility readback callback dropped")
        .expect("visibility readback mapping failed");
    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_raster_ids_faces_and_reconstruction_match_the_cpu_oracle() {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 16;
    const VISIBILITY_ROW_BYTES: u32 = 256;
    const BARYCENTRIC_ROW_BYTES: u32 = WIDTH * 16;
    let Some((device, queue)) = try_device(wgpu::Features::PRIMITIVE_INDEX) else {
        eprintln!("no GPU adapter — skipping visibility raster oracle");
        return;
    };

    let clip = [
        [-0.9, -0.8, 0.5, 1.0],
        [-0.2, -1.6, 1.0, 2.0],
        [-2.0, 3.2, 2.0, 4.0],
        [0.1, -0.8, 0.5, 1.0],
        [2.0, 3.2, 2.0, 4.0],
        [1.8, -1.6, 1.0, 2.0],
    ];
    let shader_source = format!(
        "enable primitive_index;\n\
             {RECONSTRUCTION_WGSL}\n\
             struct VertexOut {{ @builtin(position) position: vec4<f32>, }};\n\
             struct FragmentOut {{\n\
               @location(0) visibility: vec2<u32>,\n\
               @location(1) barycentrics: vec4<f32>,\n\
             }};\n\
             fn clip_position(index: u32) -> vec4<f32> {{\n\
               var positions = array<vec4<f32>, 6>(\n\
                 vec4<f32>(-0.9, -0.8, 0.5, 1.0),\n\
                 vec4<f32>(-0.2, -1.6, 1.0, 2.0),\n\
                 vec4<f32>(-2.0, 3.2, 2.0, 4.0),\n\
                 vec4<f32>(0.1, -0.8, 0.5, 1.0),\n\
                 vec4<f32>(2.0, 3.2, 2.0, 4.0),\n\
                 vec4<f32>(1.8, -1.6, 1.0, 2.0),\n\
               );\n\
               return positions[index];\n\
             }}\n\
             @vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {{\n\
               var out: VertexOut;\n\
               out.position = clip_position(index);\n\
               return out;\n\
             }}\n\
             @fragment fn fs_main(\n\
               in: VertexOut,\n\
               @builtin(primitive_index) primitive_id: u32,\n\
               @builtin(front_facing) front_facing: bool,\n\
             ) -> FragmentOut {{\n\
               let first = primitive_id * 3u;\n\
               let point_ndc = vec2<f32>(\n\
                 in.position.x / {WIDTH}.0 * 2.0 - 1.0,\n\
                 1.0 - in.position.y / {HEIGHT}.0 * 2.0,\n\
               );\n\
               let barycentrics = bloom_perspective_barycentrics(\n\
                 point_ndc,\n\
                 clip_position(first),\n\
                 clip_position(first + 1u),\n\
                 clip_position(first + 2u),\n\
               );\n\
               var out: FragmentOut;\n\
               out.visibility = bloom_encode_visibility(7u, primitive_id, front_facing);\n\
               out.barycentrics = vec4<f32>(barycentrics, 1.0);\n\
               return out;\n\
             }}"
    );
    wgpu::naga::front::wgsl::parse_str(&shader_source)
        .unwrap_or_else(|error| panic!("visibility raster oracle WGSL failed: {error:?}"));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("visibility_buffer_oracle_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("visibility_buffer_oracle_pipeline_layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("visibility_buffer_oracle_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: VISIBILITY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let make_target = |label, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let visibility = make_target("visibility_buffer_oracle_ids", VISIBILITY_FORMAT);
    let barycentrics = make_target(
        "visibility_buffer_oracle_barycentrics",
        wgpu::TextureFormat::Rgba32Float,
    );
    let visibility_view = visibility.create_view(&Default::default());
    let barycentric_view = barycentrics.create_view(&Default::default());
    let visibility_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("visibility_buffer_oracle_id_readback"),
        size: (VISIBILITY_ROW_BYTES * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let barycentric_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("visibility_buffer_oracle_barycentric_readback"),
        size: (BARYCENTRIC_ROW_BYTES * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("visibility_buffer_oracle_encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("visibility_buffer_oracle_pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &visibility_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: u32::MAX as f64,
                            g: u32::MAX as f64,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &barycentric_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..6, 0..1);
    }
    for (texture, buffer, bytes_per_row) in [
        (&visibility, &visibility_readback, VISIBILITY_ROW_BYTES),
        (&barycentrics, &barycentric_readback, BARYCENTRIC_ROW_BYTES),
    ] {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit(std::iter::once(encoder.finish()));

    let id_bytes = readback(&device, &visibility_readback);
    let barycentric_bytes = readback(&device, &barycentric_readback);
    let mut primitive_pixels = [0usize; 2];
    let mut primitive_faces = [None; 2];
    let mut background_pixels = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let id_offset = (y * VISIBILITY_ROW_BYTES + x * 8) as usize;
            let words: &[u32] = bytemuck::cast_slice(&id_bytes[id_offset..id_offset + 8]);
            let record = VisibilityRecord {
                draw_id: words[0],
                primitive_and_face: words[1],
            };
            let Some((draw_id, primitive_id, front_facing)) = record.decode() else {
                background_pixels += 1;
                continue;
            };
            assert_eq!(draw_id, 7);
            assert!(primitive_id < 2);
            let primitive = primitive_id as usize;
            primitive_pixels[primitive] += 1;
            match primitive_faces[primitive] {
                Some(expected) => assert_eq!(front_facing, expected),
                None => primitive_faces[primitive] = Some(front_facing),
            }

            let point_ndc = [
                (x as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / HEIGHT as f32 * 2.0,
            ];
            let first = primitive * 3;
            let expected = perspective_barycentrics(
                point_ndc,
                [clip[first], clip[first + 1], clip[first + 2]],
            )
            .unwrap();
            let bary_offset = (y * BARYCENTRIC_ROW_BYTES + x * 16) as usize;
            let actual: &[f32] =
                bytemuck::cast_slice(&barycentric_bytes[bary_offset..bary_offset + 16]);
            for lane in 0..3 {
                assert!(
                    (actual[lane] - expected[lane]).abs() <= 2.0e-5,
                    "pixel ({x},{y}) primitive {primitive}: GPU {:?}, CPU {:?}",
                    &actual[..3],
                    expected,
                );
            }
            assert_close(actual[0] + actual[1] + actual[2], 1.0);
        }
    }
    assert!(background_pixels > 0, "clear sentinel was not preserved");
    assert!(primitive_pixels.iter().all(|pixels| *pixels > 0));
    assert_ne!(
        primitive_faces[0], primitive_faces[1],
        "opposite winding must preserve distinct front-face bits"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_pulls_shared_geometry_and_reconstructs_every_vertex_lane() {
    const OUTPUT_WORDS: usize = 100;
    let Some((device, queue)) = try_device(wgpu::Features::empty()) else {
        eprintln!("no GPU adapter — skipping shared-geometry oracle");
        return;
    };

    let padding = Vertex3D {
        position: [-99.0; 3],
        normal: [-98.0; 3],
        color: [-97.0; 4],
        uv: [-96.0; 2],
        joints: [-95.0; 4],
        weights: [-94.0; 4],
        tangent: [-93.0; 4],
    };
    let vertices = [
        padding,
        Vertex3D {
            position: [-0.9, -0.8, 1.0],
            normal: [0.1, 0.2, 0.3],
            color: [0.4, 0.5, 0.6, 0.7],
            uv: [0.8, 0.9],
            joints: [1.0, 2.0, 3.0, 4.0],
            weights: [0.1, 0.2, 0.3, 0.4],
            tangent: [0.7, 0.2, 0.1, -1.0],
        },
        Vertex3D {
            position: [-0.2, -1.6, 2.0],
            normal: [1.1, 1.2, 1.3],
            color: [1.4, 1.5, 1.6, 1.7],
            uv: [1.8, 1.9],
            joints: [5.0, 6.0, 7.0, 8.0],
            weights: [0.4, 0.3, 0.2, 0.1],
            tangent: [0.1, 0.6, 0.3, 1.0],
        },
        Vertex3D {
            position: [-2.0, 3.2, 4.0],
            normal: [2.1, 2.2, 2.3],
            color: [2.4, 2.5, 2.6, 2.7],
            uv: [2.8, 2.9],
            joints: [9.0, 10.0, 11.0, 12.0],
            weights: [0.25, 0.25, 0.25, 0.25],
            tangent: [0.4, 0.2, 0.8, -1.0],
        },
    ];
    let mvp = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5, 0.0],
    ];
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let draw = GpuDrawRecord {
        uniforms: Uniforms3D {
            mvp,
            model: identity,
            prev_mvp: identity,
            model_tint: [1.0; 4],
            misc: [0.0; 4],
        },
        bounds_min: [-2.0, -1.6, 1.0, 0.0],
        bounds_max: [-0.2, 3.2, 4.0, 0.0],
        draw: [3, 3, 1_i32 as u32, 1_234],
    };
    let indices = [91u32, 92, 93, 0, 1, 2];
    let point_ndc = [-0.45f32, -0.1, 0.0, 0.0];
    let visibility_record = VisibilityRecord::encode(0, 0, true).unwrap();

    let shader_source = [
        RECONSTRUCTION_WGSL,
        GEOMETRY_WGSL,
        r#"
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
    misc: vec4<f32>,
};
struct GpuDrawRecord {
    uniforms: Uniforms3D,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    draw: vec4<u32>,
};
struct VertexTable { records: array<BloomPackedVertex3D>, };
struct IndexTable { values: array<u32>, };
struct DrawTable { records: array<GpuDrawRecord>, };
struct OutputTable { words: array<u32>, };

@group(0) @binding(0) var visibility_texture: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> vertices: VertexTable;
@group(0) @binding(2) var<storage, read> indices: IndexTable;
@group(0) @binding(3) var<storage, read> draws: DrawTable;
@group(0) @binding(4) var<uniform> point_ndc: vec4<f32>;
@group(0) @binding(5) var<storage, read_write> output: OutputTable;

fn write_vertex(offset: u32, vertex: BloomVertex3D) {
    output.words[offset + 0u] = bitcast<u32>(vertex.position.x);
    output.words[offset + 1u] = bitcast<u32>(vertex.position.y);
    output.words[offset + 2u] = bitcast<u32>(vertex.position.z);
    output.words[offset + 3u] = bitcast<u32>(vertex.normal.x);
    output.words[offset + 4u] = bitcast<u32>(vertex.normal.y);
    output.words[offset + 5u] = bitcast<u32>(vertex.normal.z);
    output.words[offset + 6u] = bitcast<u32>(vertex.color.x);
    output.words[offset + 7u] = bitcast<u32>(vertex.color.y);
    output.words[offset + 8u] = bitcast<u32>(vertex.color.z);
    output.words[offset + 9u] = bitcast<u32>(vertex.color.w);
    output.words[offset + 10u] = bitcast<u32>(vertex.uv.x);
    output.words[offset + 11u] = bitcast<u32>(vertex.uv.y);
    output.words[offset + 12u] = bitcast<u32>(vertex.joints.x);
    output.words[offset + 13u] = bitcast<u32>(vertex.joints.y);
    output.words[offset + 14u] = bitcast<u32>(vertex.joints.z);
    output.words[offset + 15u] = bitcast<u32>(vertex.joints.w);
    output.words[offset + 16u] = bitcast<u32>(vertex.weights.x);
    output.words[offset + 17u] = bitcast<u32>(vertex.weights.y);
    output.words[offset + 18u] = bitcast<u32>(vertex.weights.z);
    output.words[offset + 19u] = bitcast<u32>(vertex.weights.w);
    output.words[offset + 20u] = bitcast<u32>(vertex.tangent.x);
    output.words[offset + 21u] = bitcast<u32>(vertex.tangent.y);
    output.words[offset + 22u] = bitcast<u32>(vertex.tangent.z);
    output.words[offset + 23u] = bitcast<u32>(vertex.tangent.w);
}

@compute @workgroup_size(1)
fn cs_main() {
    let raw_visibility = textureLoad(visibility_texture, vec2<i32>(0, 0), 0).xy;
    if (!bloom_visibility_valid(raw_visibility)) {
        output.words[96] = BLOOM_VISIBILITY_INVALID_DRAW_ID;
        return;
    }
    let visibility = bloom_decode_visibility(raw_visibility);
    let draw = draws.records[visibility.draw_id];
    let first_index = draw.draw.y + visibility.primitive_id * 3u;
    let base_vertex = bitcast<i32>(draw.draw.z);
    let index0 = u32(i32(indices.values[first_index]) + base_vertex);
    let index1 = u32(i32(indices.values[first_index + 1u]) + base_vertex);
    let index2 = u32(i32(indices.values[first_index + 2u]) + base_vertex);
    let vertex0 = bloom_decode_vertex3d(vertices.records[index0]);
    let vertex1 = bloom_decode_vertex3d(vertices.records[index1]);
    let vertex2 = bloom_decode_vertex3d(vertices.records[index2]);
    write_vertex(0u, vertex0);
    write_vertex(24u, vertex1);
    write_vertex(48u, vertex2);

    let clip0 = draw.uniforms.mvp * vec4<f32>(vertex0.position, 1.0);
    let clip1 = draw.uniforms.mvp * vec4<f32>(vertex1.position, 1.0);
    let clip2 = draw.uniforms.mvp * vec4<f32>(vertex2.position, 1.0);
    let bary = bloom_perspective_barycentrics(point_ndc.xy, clip0, clip1, clip2);
    let interpolated = BloomVertex3D(
        bloom_interpolate3(vertex0.position, vertex1.position, vertex2.position, bary),
        bloom_interpolate3(vertex0.normal, vertex1.normal, vertex2.normal, bary),
        bloom_interpolate4(vertex0.color, vertex1.color, vertex2.color, bary),
        bloom_interpolate2(vertex0.uv, vertex1.uv, vertex2.uv, bary),
        bloom_interpolate4(vertex0.joints, vertex1.joints, vertex2.joints, bary),
        bloom_interpolate4(vertex0.weights, vertex1.weights, vertex2.weights, bary),
        bloom_interpolate4(vertex0.tangent, vertex1.tangent, vertex2.tangent, bary),
    );
    write_vertex(72u, interpolated);
    output.words[96] = visibility.draw_id;
    output.words[97] = visibility.primitive_id;
    output.words[98] = select(0u, 1u, visibility.front_facing);
    output.words[99] = draw.draw.w;
}
"#,
    ]
    .concat();
    wgpu::naga::front::wgsl::parse_str(&shader_source)
        .unwrap_or_else(|error| panic!("shared-geometry oracle WGSL failed: {error:?}"));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("visibility_shared_geometry_oracle_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("visibility_shared_geometry_oracle_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let visibility = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("visibility_shared_geometry_oracle_ids"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: VISIBILITY_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &visibility,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::bytes_of(&visibility_record),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(VISIBILITY_BYTES_PER_PIXEL as u32),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let make_storage = |label, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let vertex_buffer = make_storage(
        "visibility_shared_geometry_oracle_vertices",
        bytemuck::cast_slice(&vertices),
    );
    let index_buffer = make_storage(
        "visibility_shared_geometry_oracle_indices",
        bytemuck::cast_slice(&indices),
    );
    let draw_buffer = make_storage(
        "visibility_shared_geometry_oracle_draws",
        bytemuck::bytes_of(&draw),
    );
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("visibility_shared_geometry_oracle_point"),
        contents: bytemuck::cast_slice(&point_ndc),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("visibility_shared_geometry_oracle_output"),
        size: (OUTPUT_WORDS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("visibility_shared_geometry_oracle_readback"),
        size: (OUTPUT_WORDS * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let visibility_view = visibility.create_view(&Default::default());
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("visibility_shared_geometry_oracle_bind_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&visibility_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: draw_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: point_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("visibility_shared_geometry_oracle_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("visibility_shared_geometry_oracle_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &readback_buffer,
        0,
        (OUTPUT_WORDS * 4) as u64,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let output_bytes = readback(&device, &readback_buffer);
    let output: &[u32] = bytemuck::cast_slice(&output_bytes);
    let expected_raw: &[u32] = bytemuck::cast_slice(&vertices[1..]);
    assert_eq!(&output[..72], expected_raw);

    let clip = [
        [-0.9, -0.8, 0.5, 1.0],
        [-0.2, -1.6, 0.5, 2.0],
        [-2.0, 3.2, 0.5, 4.0],
    ];
    let bary = perspective_barycentrics([point_ndc[0], point_ndc[1]], clip).unwrap();
    let source: &[f32] = bytemuck::cast_slice(&vertices[1..]);
    for lane in 0..24 {
        let expected =
            source[lane] * bary[0] + source[24 + lane] * bary[1] + source[48 + lane] * bary[2];
        let actual = f32::from_bits(output[72 + lane]);
        assert!(
            (actual - expected).abs() <= 2.0e-5,
            "interpolated Vertex3D lane {lane}: GPU {actual}, CPU {expected}"
        );
    }
    assert_eq!(&output[96..100], &[0, 0, 1, 1_234]);
}
