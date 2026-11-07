use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_pool_enforces_frame_upload_and_eviction_limits_without_partial_pages() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping virtual-geometry budget oracle");
        return;
    };
    let asset = hierarchy_asset(hierarchy_archive());
    let mut upload_config = gpu_config(3);
    upload_config.max_upload_pages_per_frame = 1;
    let mut upload_limited = GpuVirtualGeometryPool::new(&device, upload_config).unwrap();
    let mesh = upload_limited
        .register_mesh(&queue, Arc::clone(&asset))
        .unwrap();
    upload_limited.begin_frame(2);
    upload_limited.make_group_resident(&queue, mesh, 2).unwrap();
    let before_denial = upload_limited.telemetry();
    assert!(matches!(
        upload_limited.make_group_resident(&queue, mesh, 3),
        Err(VirtualGeometryGpuError::UploadBudgetExceeded { .. })
    ));
    let after_denial = upload_limited.telemetry();
    assert_eq!(
        after_denial.frame_upload_pages,
        before_denial.frame_upload_pages
    );
    assert_eq!(
        after_denial.frame_upload_bytes,
        before_denial.frame_upload_bytes
    );
    assert_eq!(after_denial.resident_pages, before_denial.resident_pages);
    assert_eq!(
        after_denial.denied_uploads,
        before_denial.denied_uploads + 1
    );
    assert!(!upload_limited
        .is_page_resident(VirtualPageId {
            mesh,
            page_index: 2,
        })
        .unwrap());

    let mut byte_config = gpu_config(3);
    byte_config.max_upload_bytes_per_frame = u64::from(MIN_PAGE_BYTES);
    let mut byte_limited = GpuVirtualGeometryPool::new(&device, byte_config).unwrap();
    let mesh = byte_limited
        .register_mesh(&queue, hierarchy_asset(large_intermediate_page_archive()))
        .unwrap();
    byte_limited.make_group_resident(&queue, mesh, 2).unwrap();
    let before_denial = byte_limited.telemetry();
    assert_eq!(before_denial.frame_upload_bytes, 3_920);
    assert!(matches!(
        byte_limited.make_group_resident(&queue, mesh, 3),
        Err(VirtualGeometryGpuError::UploadBudgetExceeded {
            requested_bytes: 224,
            remaining_bytes: 176,
            ..
        })
    ));
    let after_denial = byte_limited.telemetry();
    assert_eq!(
        after_denial.frame_upload_bytes,
        before_denial.frame_upload_bytes
    );
    assert_eq!(after_denial.resident_pages, before_denial.resident_pages);
    assert_eq!(
        after_denial.denied_uploads,
        before_denial.denied_uploads + 1
    );

    let mut eviction_config = gpu_config(3);
    eviction_config.max_evictions_per_frame = 0;
    let mut eviction_limited = GpuVirtualGeometryPool::new(&device, eviction_config).unwrap();
    let mesh = eviction_limited.register_mesh(&queue, asset).unwrap();
    eviction_limited.begin_frame(2);
    eviction_limited
        .make_group_resident(&queue, mesh, 2)
        .unwrap();
    eviction_limited
        .make_group_resident(&queue, mesh, 3)
        .unwrap();
    let before_denial = eviction_limited.telemetry();
    assert_eq!(
        eviction_limited
            .make_group_resident(&queue, mesh, 4)
            .unwrap_err(),
        VirtualGeometryGpuError::EvictionBudgetExceeded
    );
    let after_denial = eviction_limited.telemetry();
    assert_eq!(after_denial.resident_pages, before_denial.resident_pages);
    assert_eq!(after_denial.frame_evictions, 0);
    assert_eq!(
        after_denial.denied_uploads,
        before_denial.denied_uploads + 1
    );
    assert!(!eviction_limited
        .is_page_resident(VirtualPageId {
            mesh,
            page_index: 3,
        })
        .unwrap());
}
