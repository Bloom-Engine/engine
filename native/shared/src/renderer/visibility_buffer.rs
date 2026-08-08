//! Packed visibility-buffer contract for the #27 qualification path.
//!
//! This module does not enable a shipping render path. It locks the 8-byte
//! target ABI and reconstruction math that an opt-in A/B implementation will
//! use. The existing forward MRT remains authoritative until total frame cost
//! and image parity pass on every required capability tier.

pub(crate) const VISIBILITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
pub(crate) const VISIBILITY_BYTES_PER_PIXEL: u64 = 8;
pub(crate) const INVALID_DRAW_ID: u32 = u32::MAX;
pub(crate) const FRONT_FACE_BIT: u32 = 1 << 31;
pub(crate) const PRIMITIVE_ID_MASK: u32 = FRONT_FACE_BIT - 1;

/// One visibility-buffer texel. The second word reserves its high bit for the
/// rasterized face orientation and leaves 31 bits for the primitive index.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VisibilityRecord {
    pub draw_id: u32,
    pub primitive_and_face: u32,
}

impl VisibilityRecord {
    pub(crate) const BACKGROUND: Self = Self {
        draw_id: INVALID_DRAW_ID,
        primitive_and_face: u32::MAX,
    };

    pub(crate) const fn encode(
        draw_id: u32,
        primitive_id: u32,
        front_facing: bool,
    ) -> Option<Self> {
        if draw_id == INVALID_DRAW_ID || primitive_id > PRIMITIVE_ID_MASK {
            return None;
        }
        Some(Self {
            draw_id,
            primitive_and_face: primitive_id | if front_facing { FRONT_FACE_BIT } else { 0 },
        })
    }

    pub(crate) const fn decode(self) -> Option<(u32, u32, bool)> {
        if self.draw_id == INVALID_DRAW_ID {
            return None;
        }
        Some((
            self.draw_id,
            self.primitive_and_face & PRIMITIVE_ID_MASK,
            (self.primitive_and_face & FRONT_FACE_BIT) != 0,
        ))
    }
}

/// Exact allocation size of the packed visibility target, excluding backend
/// row/heap alignment that must be reported separately by the runtime A/B.
pub(crate) const fn target_bytes(width: u32, height: u32) -> Option<u64> {
    match (width as u64).checked_mul(height as u64) {
        Some(pixels) => pixels.checked_mul(VISIBILITY_BYTES_PER_PIXEL),
        None => None,
    }
}

/// Stable machine-readable contract included in renderer diagnostics even
/// while the experimental path is disabled.
pub(crate) fn contract_json() -> String {
    let format_name = match VISIBILITY_FORMAT {
        wgpu::TextureFormat::Rg32Uint => "rg32uint",
        _ => "invalid",
    };
    let background = VisibilityRecord::BACKGROUND;
    let max_record = VisibilityRecord::encode(0, PRIMITIVE_ID_MASK, true)
        .expect("the visibility ABI maximum must remain encodable");
    debug_assert_eq!(background.decode(), None);
    debug_assert_eq!(max_record.decode(), Some((0, PRIMITIVE_ID_MASK, true)));
    format!(
        concat!(
            "{{\"format\":\"{}\",\"bytes_per_pixel\":{},",
            "\"invalid_draw_id\":{},\"primitive_bits\":31,",
            "\"front_face_bits\":1,\"shipping_enabled\":false,",
            "\"native_1080p_bytes\":{},\"reconstruction_wgsl_bytes\":{},",
            "\"activation\":\"opt-in A/B qualification required\"}}"
        ),
        format_name,
        VISIBILITY_BYTES_PER_PIXEL,
        INVALID_DRAW_ID,
        target_bytes(1_920, 1_080).expect("1080p visibility allocation is bounded"),
        RECONSTRUCTION_WGSL.len(),
    )
}

pub(crate) const RECONSTRUCTION_WGSL: &str =
    include_str!("../../shaders/visibility_buffer/reconstruct.wgsl");

#[cfg(test)]
fn screen_barycentrics(point: [f32; 2], triangle: [[f32; 2]; 3]) -> Option<[f32; 3]> {
    let edge = |a: [f32; 2], b: [f32; 2], p: [f32; 2]| {
        (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    };
    let area = edge(triangle[1], triangle[2], triangle[0]);
    if area.abs() <= 1.0e-12 {
        return None;
    }
    Some([
        edge(triangle[1], triangle[2], point) / area,
        edge(triangle[2], triangle[0], point) / area,
        edge(triangle[0], triangle[1], point) / area,
    ])
}

#[cfg(test)]
fn perspective_barycentrics(point: [f32; 2], clip: [[f32; 4]; 3]) -> Option<[f32; 3]> {
    if clip.iter().any(|vertex| vertex[3].abs() <= 1.0e-12) {
        return None;
    }
    let ndc = [
        [clip[0][0] / clip[0][3], clip[0][1] / clip[0][3]],
        [clip[1][0] / clip[1][3], clip[1][1] / clip[1][3]],
        [clip[2][0] / clip[2][3], clip[2][1] / clip[2][3]],
    ];
    let linear = screen_barycentrics(point, ndc)?;
    let weighted = [
        linear[0] / clip[0][3],
        linear[1] / clip[1][3],
        linear[2] / clip[2][3],
    ];
    let sum = weighted[0] + weighted[1] + weighted[2];
    if sum.abs() <= 1.0e-12 {
        return None;
    }
    Some([weighted[0] / sum, weighted[1] / sum, weighted[2] / sum])
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(report.contains("\"shipping_enabled\":false"));
    }

    #[test]
    fn ids_and_face_orientation_round_trip_without_background_collision() {
        for (draw, primitive, front) in [
            (0, 0, false),
            (17, 42, true),
            (u32::MAX - 1, PRIMITIVE_ID_MASK, false),
        ] {
            let encoded = VisibilityRecord::encode(draw, primitive, front).unwrap();
            assert_eq!(encoded.decode(), Some((draw, primitive, front)));
        }
        assert_eq!(VisibilityRecord::BACKGROUND.decode(), None);
        assert_eq!(VisibilityRecord::encode(INVALID_DRAW_ID, 0, true), None);
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
        assert!(RECONSTRUCTION_WGSL
            .contains("const BLOOM_VISIBILITY_FRONT_FACE_BIT: u32 = 0x80000000u"));
        assert!(RECONSTRUCTION_WGSL.contains("fn bloom_perspective_barycentrics("));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        if !adapter.features().contains(wgpu::Features::PRIMITIVE_INDEX) {
            eprintln!("adapter lacks PRIMITIVE_INDEX — skipping visibility raster oracle");
            return None;
        }
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("visibility_buffer_oracle_device"),
            required_features: wgpu::Features::PRIMITIVE_INDEX,
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
        let Some((device, queue)) = try_device() else {
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
}
