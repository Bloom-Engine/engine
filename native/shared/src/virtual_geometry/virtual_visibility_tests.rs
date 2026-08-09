use super::*;
use crate::renderer::visibility_buffer::{
    VisibilityDraw, VisibilityRecord, INVALID_DRAW_ID, VISIBILITY_FORMAT,
};

#[test]
fn raw_virtual_clusters_rasterize_namespaced_visibility_ids_on_the_real_gpu() {
    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 16;
    const ROW_BYTES: u32 = 256;

    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no GPU adapter — skipping virtual visibility raster oracle");
        return;
    };
    let required = wgpu::Features::PRIMITIVE_INDEX | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    if !device.features().contains(required) {
        eprintln!("adapter lacks primitive-index/indirect-first-instance — skipping oracle");
        return;
    }

    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let emitter = GpuVirtualDrawEmitter::new(&device, &selector).unwrap();
    let raster = GpuVirtualVisibilityRaster::new(&device, &pool, &selector, &emitter).unwrap();
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    raster
        .prepare_frame(
            &queue,
            GpuVirtualVisibilityFrame::new(identity, identity).unwrap(),
        )
        .unwrap();

    let visibility = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual_visibility_oracle_ids"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: VISIBILITY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual_visibility_oracle_depth"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::renderer::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let readback_source = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("virtual_visibility_oracle_copy"),
        size: u64::from(ROW_BYTES * HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let visibility_view = visibility.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_visibility_oracle_encoder"),
    });
    selector
        .record(
            &queue,
            &mut encoder,
            &pool,
            &[GpuVirtualInstance::identity(mesh, 901)],
            traversal_view(50.0),
        )
        .unwrap();
    emitter.record(&queue, &mut encoder, &selector).unwrap();
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("virtual_visibility_oracle_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &visibility_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(u32::MAX),
                        g: f64::from(u32::MAX),
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        raster.draw_fixed_for_test(&mut pass, &emitter, 4).unwrap();
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &visibility,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_source,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let bytes = read_gpu_buffer(
        &device,
        &queue,
        &readback_source,
        u64::from(ROW_BYTES * HEIGHT),
    );
    let mut covered = 0usize;
    let mut background = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = (y * ROW_BYTES + x * 8) as usize;
            let words: &[u32] = bytemuck::cast_slice(&bytes[offset..offset + 8]);
            let record = VisibilityRecord {
                draw_id: words[0],
                primitive_and_face: words[1],
            };
            if record.draw_id == INVALID_DRAW_ID {
                background += 1;
                continue;
            }
            let Some((VisibilityDraw::Virtual(draw_index), primitive, _)) = record.decode_draw()
            else {
                panic!("virtual raster emitted a compatibility or invalid visibility ID");
            };
            assert!(draw_index < 4);
            assert_eq!(primitive, 0);
            covered += 1;
        }
    }
    assert!(covered > 0);
    assert!(background > 0);
    assert_eq!(
        raster.counted_submission_supported(),
        device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
    );
}
