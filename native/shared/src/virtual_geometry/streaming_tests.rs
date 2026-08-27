use super::*;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(target_arch = "wasm32"))]
static NEXT_STREAM_FILE: AtomicU64 = AtomicU64::new(1);

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
            max_io_requests: 8,
            max_io_bytes: 8 * u64::from(MIN_PAGE_BYTES),
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

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn file_backed_feedback_reads_only_requested_pages_off_thread() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping file streaming oracle");
        return;
    };
    let memory_asset = hierarchy_asset(hierarchy_archive());
    let bytes = memory_asset.file_bytes().unwrap().to_vec();
    let file_hash = sha256(&bytes);
    let path = std::env::temp_dir().join(format!(
        "bloom-vg-stream-{}-{}.bgeo",
        std::process::id(),
        NEXT_STREAM_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, &bytes).unwrap();
    let file_asset = Arc::new(
        VirtualGeometryAsset::from_indexed_file_bytes(
            path.clone(),
            bytes,
            ArtifactIdentity {
                bytes: memory_asset.artifact_bytes(),
                format_version: memory_asset.archive().format_version,
                file_sha256: file_hash,
                payload_sha256: memory_asset.archive().payload_sha256,
                source_sha256: memory_asset.archive().source_sha256,
            },
        )
        .unwrap(),
    );
    assert!(file_asset.is_file_backed());
    assert!(file_asset.page_bytes(0).is_some());
    assert!(file_asset.page_bytes(1).is_none());

    let mut pool_config = gpu_config(5);
    pool_config.max_upload_bytes_per_frame = 8 * u64::from(MIN_PAGE_BYTES);
    pool_config.max_upload_pages_per_frame = 8;
    let mut pool = GpuVirtualGeometryPool::new(&device, pool_config).unwrap();
    let mesh = pool.register_mesh(&queue, file_asset).unwrap();
    bind_test_materials(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let mut streamer = GpuVirtualPageStreamer::new(
        &device,
        &selector,
        GpuVirtualStreamingConfig {
            max_readback_requests: 8,
            max_pending_groups: 16,
            max_group_attempts_per_frame: 8,
            max_io_requests: 8,
            max_io_bytes: 8 * u64::from(MIN_PAGE_BYTES),
        },
    )
    .unwrap();
    let instances = [GpuVirtualInstance::identity(mesh, 79)];
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_file_feedback_test_encoder"),
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
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    streamer.poll(&device);
    assert!(streamer.telemetry().pending_groups > 0);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut frame = 2;
    while streamer.telemetry().pending_groups != 0 {
        pool.begin_frame(frame);
        streamer.service(&mut pool, &queue);
        frame += 1;
        assert!(
            std::time::Instant::now() < deadline,
            "file-backed page worker timed out"
        );
        std::thread::yield_now();
    }
    let telemetry = streamer.telemetry();
    assert!(telemetry.io_requests > 0);
    assert_eq!(telemetry.io_completions, telemetry.io_requests);
    assert_eq!(telemetry.io_failures, 0);
    assert_eq!(telemetry.in_flight_io_groups, 0);
    assert_eq!(telemetry.ready_io_groups, 0);
    assert_eq!(telemetry.reserved_io_bytes, 0);
    assert_eq!(telemetry.io_budget_bytes, 8 * u64::from(MIN_PAGE_BYTES));
    assert!(telemetry.peak_reserved_io_bytes > 0);
    assert!(telemetry.peak_reserved_io_bytes <= telemetry.io_budget_bytes);
    assert_eq!(telemetry.io_bytes_read, telemetry.uploaded_bytes);
    assert!(telemetry.uploaded_pages > 0);
    assert!(pool
        .is_page_resident(VirtualPageId {
            mesh,
            page_index: 0,
        })
        .unwrap());
    std::fs::remove_file(path).unwrap();
}
