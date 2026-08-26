use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn asynchronous_feedback_streams_missing_groups_without_hiding_ancestors() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping async feedback oracle");
        return;
    };
    let mut pool_config = gpu_config(5);
    pool_config.max_upload_bytes_per_frame = u64::from(MIN_PAGE_BYTES);
    pool_config.max_upload_pages_per_frame = 1;
    let mut pool = GpuVirtualGeometryPool::new(&device, pool_config).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    bind_test_materials(&mut pool, &queue, mesh);
    pool.begin_frame(2);
    pool.make_group_resident(&queue, mesh, 2).unwrap();
    pool.begin_frame(3);
    pool.make_group_resident(&queue, mesh, 3).unwrap();

    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let mut streamer = GpuVirtualPageStreamer::new(
        &device,
        &selector,
        GpuVirtualStreamingConfig {
            max_readback_requests: 8,
            max_pending_groups: 16,
            max_group_attempts_per_frame: 8,
        },
    )
    .unwrap();
    let instances = [GpuVirtualInstance::identity(mesh, 73)];
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_async_feedback_test_encoder"),
    });
    selector
        .record(
            &queue,
            &mut encoder,
            &pool,
            &instances,
            traversal_view(50.0),
        )
        .unwrap();
    assert!(streamer.record(&mut encoder, &selector));
    queue.submit(std::iter::once(encoder.finish()));
    streamer.after_submit();
    assert_eq!(streamer.telemetry().in_flight_readbacks, 1);

    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    streamer.poll(&device);
    let feedback = streamer.telemetry();
    assert_eq!(feedback.captures_completed, 1);
    assert_eq!(feedback.attempted_requests, 2);
    assert_eq!(feedback.copied_requests, 2);
    assert_eq!(feedback.pending_groups, 2);
    assert!(feedback.last_visible_groups > 0);
    assert_eq!(feedback.last_occlusion_culled_groups, 0);
    assert_eq!(feedback.last_occlusion_uncertain_groups, 0);

    pool.begin_frame(4);
    streamer.service(&mut pool, &queue);
    let first_stream = streamer.telemetry();
    assert_eq!(first_stream.pending_groups, 1);
    assert_eq!(first_stream.groups_resolved, 1);
    assert_eq!(first_stream.uploaded_pages, 1);
    assert!(first_stream.budget_stalls >= 1);

    pool.begin_frame(5);
    streamer.service(&mut pool, &queue);
    let streamed = streamer.telemetry();
    assert_eq!(streamed.pending_groups, 0);
    assert_eq!(streamed.groups_resolved, 2);
    assert_eq!(streamed.uploaded_pages, 2);
    assert!(pool
        .is_page_resident(VirtualPageId {
            mesh,
            page_index: 3,
        })
        .unwrap());
    assert!(pool
        .is_page_resident(VirtualPageId {
            mesh,
            page_index: 4,
        })
        .unwrap());

    let (selected, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(50.0),
    );
    assert_eq!(
        selected
            .iter()
            .map(|record| record.cluster_table_index)
            .collect::<Vec<_>>(),
        [4, 5, 6, 7]
    );
    assert!(requests.is_empty());
    assert_eq!(counters.fallback_groups, 0);
}
