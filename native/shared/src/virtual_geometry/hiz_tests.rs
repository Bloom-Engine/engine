use super::*;
use crate::virtual_geometry::VirtualGeometryHiZFrame;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn previous_hiz_culls_only_stable_captured_instances_and_rejects_camera_motion() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping virtual Hi-Z oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    bind_test_materials(&mut pool, &queue, mesh);
    for (frame, group) in [(2, 2), (3, 3), (4, 4), (5, 6)] {
        pool.begin_frame(frame);
        pool.make_group_resident(&queue, mesh, group).unwrap();
    }
    let mut selector =
        GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let model = [
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0],
        [-0.05, -0.05, -10.0, 1.0],
    ];
    let instance = GpuVirtualInstance::new(mesh, 19, model).unwrap();
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let source = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual_hiz_uniform_occluder"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&vec![2.0f32; 64 * 64]),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(64 * 4),
            rows_per_image: Some(64),
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    let captured = VirtualGeometryHiZFrame {
        frame_index: 1,
        view_projection: identity,
        view: identity,
        render_extent: (128, 128),
        camera_cut: false,
    };
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_hiz_capture_oracle"),
    });
    selector.record_previous_hiz_capture(
        &device,
        &queue,
        &mut encoder,
        &source.create_view(&wgpu::TextureViewDescriptor::default()),
        (64, 64),
        captured,
        &[instance],
    );
    queue.submit(std::iter::once(encoder.finish()));
    selector.after_submit_previous_hiz();

    let current = VirtualGeometryHiZFrame {
        frame_index: 2,
        ..captured
    };
    assert!(selector.previous_hiz_history_valid(current));
    assert!(selector.previous_hiz_contains(instance));
    assert!(
        !selector.previous_hiz_history_valid(VirtualGeometryHiZFrame {
            camera_cut: true,
            ..current
        })
    );
    assert!(
        !selector.previous_hiz_history_valid(VirtualGeometryHiZFrame {
            frame_index: 3,
            ..current
        })
    );
    assert!(
        !selector.previous_hiz_history_valid(VirtualGeometryHiZFrame {
            render_extent: (256, 128),
            ..current
        })
    );
    let mut eligible = instance;
    eligible.set_previous_hiz_eligible(true);
    let counters = run_hiz_traversal(&device, &queue, &pool, &selector, eligible, current);
    assert_eq!(counters.selected_count, 0);
    assert_eq!(counters.occlusion_culled_groups, 2);
    assert_eq!(counters.occlusion_uncertain_groups, 0);

    let mut moved = current;
    moved.view_projection[3][0] = 0.1;
    let moved_counters = run_hiz_traversal(&device, &queue, &pool, &selector, eligible, moved);
    assert_eq!(moved_counters.occlusion_culled_groups, 0);
    assert!(moved_counters.occlusion_uncertain_groups >= 2);
    assert_eq!(moved_counters.selected_count, 4);

    let new_instance = GpuVirtualInstance::new(mesh, 20, model).unwrap();
    let new_counters = run_hiz_traversal(&device, &queue, &pool, &selector, new_instance, current);
    assert_eq!(new_counters.occlusion_culled_groups, 0);
    assert!(new_counters.occlusion_uncertain_groups >= 2);
    assert_eq!(new_counters.selected_count, 4);

    selector.invalidate_previous_hiz(false);
    assert!(!selector.previous_hiz_history_valid(current));
}

#[cfg(not(target_arch = "wasm32"))]
fn run_hiz_traversal(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pool: &GpuVirtualGeometryPool,
    selector: &GpuVirtualHierarchySelector,
    instance: GpuVirtualInstance,
    frame: VirtualGeometryHiZFrame,
) -> GpuVirtualTraversalCounters {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_hiz_traversal_oracle"),
    });
    selector
        .record_with_previous_hiz(
            queue,
            &mut encoder,
            pool,
            &[instance],
            traversal_view(0.0),
            frame,
        )
        .unwrap();
    queue.submit(std::iter::once(encoder.finish()));
    bytemuck::pod_read_unaligned(&read_gpu_buffer(
        device,
        queue,
        selector.counter_buffer(),
        std::mem::size_of::<GpuVirtualTraversalCounters>() as u64,
    ))
}
