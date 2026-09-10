use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_matches_cpu_across_lod_and_frustum_decisions() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping hierarchy selector oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let instances = [GpuVirtualInstance::identity(mesh, 17)];

    let (leaf, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(50.0),
    );
    assert_eq!(
        leaf.iter()
            .map(|record| record.cluster_table_index)
            .collect::<Vec<_>>(),
        [4, 5, 6, 7]
    );
    assert!(requests.is_empty());
    assert_eq!(counters.refined_groups, 4);
    assert_eq!(counters.fallback_groups, 0);

    let (middle, _, _) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(150.0),
    );
    assert_eq!(
        middle
            .iter()
            .map(|record| record.cluster_table_index)
            .collect::<Vec<_>>(),
        [2, 3]
    );

    let (coarse, _, _) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(250.0),
    );
    assert_eq!(
        coarse
            .iter()
            .map(|record| record.cluster_table_index)
            .collect::<Vec<_>>(),
        [0, 1]
    );

    let mut outside = traversal_view(50.0);
    outside.frustum_planes[0] = [1.0, 0.0, 0.0, -100.0];
    let (culled, requests, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &instances, outside);
    assert!(culled.is_empty());
    assert!(requests.is_empty());
    assert_eq!(counters.frustum_culled_groups, 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_follows_every_branched_child_range() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping branched hierarchy oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(branching_hierarchy_archive()))
        .unwrap();
    assert_ne!(
        pool.cluster_entry(mesh, 0).unwrap().identity[3]
            & super::gpu_pool::GPU_VIRTUAL_CLUSTER_UNIFORM_CHILD_RANGE,
        0,
        "the root group should retain the uniform-child fast path"
    );
    assert_eq!(
        pool.cluster_entry(mesh, 2).unwrap().identity[3]
            & super::gpu_pool::GPU_VIRTUAL_CLUSTER_UNIFORM_CHILD_RANGE,
        0,
        "the genuinely branched group must take the complete child scan"
    );
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    pool.make_group_resident(&queue, mesh, 8).unwrap();
    assert_eq!(
        pool.protect_group_pages(mesh, 6, 1.0f32.to_bits()).unwrap(),
        4,
        "leaf feedback must retain its leaf, complete branched parent, and root pages"
    );
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let (selected, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &[GpuVirtualInstance::identity(mesh, 171)],
        traversal_view(50.0),
    );

    assert_eq!(
        selected
            .iter()
            .map(|record| record.cluster_table_index)
            .collect::<Vec<_>>(),
        [6, 7, 8, 9]
    );
    assert!(requests.is_empty());
    assert_eq!(counters.refined_groups, 2);
    assert_eq!(counters.fallback_groups, 0);
    assert_eq!(counters.invalid_records, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_refines_atomic_groups_that_straddle_frustum_planes() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping group-frustum oracle");
        return;
    };
    let mut archive = hierarchy_archive();
    for root in &mut archive.clusters[0..2] {
        root.first_child = 2;
        root.child_count = 2;
    }
    for middle in &mut archive.clusters[2..4] {
        middle.parent = 0;
        middle.parent_count = 2;
        middle.first_child = 4;
        middle.child_count = 4;
    }
    for leaf in &mut archive.clusters[4..8] {
        leaf.parent = 2;
        leaf.parent_count = 2;
    }
    archive.clusters[2].aabb_min = [-2.0, 0.0, 0.0];
    archive.clusters[2].aabb_max = [-1.0, 1.0, 1.0];
    archive.clusters[2].sphere_center = [-1.5, 0.5, 0.5];
    archive.clusters[3].aabb_min = [2.0, 0.0, 0.0];
    archive.clusters[3].aabb_max = [3.0, 1.0, 1.0];
    archive.clusters[3].sphere_center = [2.5, 0.5, 0.5];

    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(archive))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let instances = [GpuVirtualInstance::identity(mesh, 19)];
    let mut view = traversal_view(50.0);
    view.frustum_planes = [[0.0, 0.0, 0.0, 1.0]; 6];
    view.frustum_planes[0] = [1.0, 0.0, 0.0, 0.0];
    view.frustum_planes[1] = [-1.0, 0.0, 0.0, 1.0];

    let (selected, requests, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &instances, view);
    assert_eq!(
        selected
            .iter()
            .map(|record| record.cluster_table_index)
            .collect::<Vec<_>>(),
        [4, 5, 6, 7]
    );
    assert!(requests.is_empty());
    assert_eq!(counters.frustum_culled_groups, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_fails_open_laterally_for_near_clipped_groups() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping near-clip frustum oracle");
        return;
    };
    let mut archive = hierarchy_archive();
    for cluster in &mut archive.clusters {
        cluster.aabb_min = [0.02, -0.005, -0.03];
        cluster.aabb_max = [0.03, 0.005, 0.0];
        cluster.sphere_center = [0.025, 0.0, -0.015];
        cluster.sphere_radius = 0.0175;
    }

    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(archive))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let view_projection =
        crate::renderer::mat4_perspective(60.0_f32.to_radians(), 1.0, 0.01, 1_000.0);
    let view = VirtualGeometryView {
        frustum_planes: crate::scene::extract_frustum_planes(&view_projection),
        view_projection,
        camera_position: [0.0; 3],
        projection_scale: 100.0,
        target_error_pixels: 1.0,
    };

    let near_clipped = [GpuVirtualInstance::identity(mesh, 20)];
    let (selected, requests, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &near_clipped, view);
    assert_eq!(
        selected
            .iter()
            .map(|record| record.cluster_table_index)
            .collect::<Vec<_>>(),
        [4, 5, 6, 7]
    );
    assert!(requests.is_empty());
    assert_eq!(counters.frustum_culled_groups, 0);

    let mut far_offscreen_model = crate::renderer::IDENTITY_MAT4;
    far_offscreen_model[3][0] = 5.0;
    far_offscreen_model[3][2] = -2.0;
    let far_offscreen = [GpuVirtualInstance::new(mesh, 21, far_offscreen_model).unwrap()];
    let (selected, requests, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &far_offscreen, view);
    assert!(selected.is_empty());
    assert!(requests.is_empty());
    assert_eq!(counters.frustum_culled_groups, 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_preserves_near_clipped_priority_order() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping near-clip priority oracle");
        return;
    };
    let mut archive = hierarchy_archive();
    for cluster in &mut archive.clusters {
        cluster.aabb_min = [0.02, -0.005, -0.03];
        cluster.aabb_max = [0.03, 0.005, 0.0];
        cluster.sphere_center = [0.025, 0.0, -0.015];
        cluster.sphere_radius = 0.0175;
    }
    archive.clusters[2].geometric_error = 1.0;
    archive.clusters[3].geometric_error = 2.0;

    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(3)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(archive))
        .unwrap();
    bind_test_materials(&mut pool, &queue, mesh);
    pool.begin_frame(2);
    pool.make_group_resident(&queue, mesh, 2).unwrap();
    pool.make_group_resident(&queue, mesh, 3).unwrap();
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let view_projection =
        crate::renderer::mat4_perspective(60.0_f32.to_radians(), 1.0, 0.01, 1_000.0);
    let view = VirtualGeometryView {
        frustum_planes: crate::scene::extract_frustum_planes(&view_projection),
        view_projection,
        camera_position: [0.0; 3],
        projection_scale: 100.0,
        target_error_pixels: 1.0,
    };

    let (_, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &[GpuVirtualInstance::identity(mesh, 22)],
        view,
    );
    assert_eq!(requests.len(), 2);
    let priorities = requests
        .iter()
        .map(|request| (request.page_index, f32::from_bits(request.priority_bits)))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert!(priorities[&3].is_finite());
    assert!(priorities[&4].is_finite());
    assert!(priorities[&4] > priorities[&3]);
    assert_eq!(counters.fallback_groups, 2);
    assert_eq!(counters.frustum_culled_groups, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_keeps_resident_ancestors_and_requests_missing_pages() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping hierarchy fallback oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(3)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    bind_test_materials(&mut pool, &queue, mesh);
    pool.begin_frame(2);
    pool.make_group_resident(&queue, mesh, 2).unwrap();
    pool.make_group_resident(&queue, mesh, 3).unwrap();
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let instances = [GpuVirtualInstance::identity(mesh, 23)];

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
        [2, 3]
    );
    assert_eq!(
        requests
            .iter()
            .map(|request| request.page_index)
            .collect::<Vec<_>>(),
        [3, 4]
    );
    assert_eq!(counters.refined_groups, 2);
    assert_eq!(counters.fallback_groups, 2);
    assert_eq!(counters.missing_current_pages, 0);
    assert_eq!(counters.invalid_records, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_keeps_request_and_residency_priorities_consistent() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping priority consistency oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    bind_test_materials(&mut pool, &queue, mesh);
    pool.begin_frame(2);
    pool.make_group_resident(&queue, mesh, 2).unwrap();
    pool.make_group_resident(&queue, mesh, 3).unwrap();
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let instances = [GpuVirtualInstance::identity(mesh, 24)];

    let (_, requests, _) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(50.0),
    );
    let requested_priorities = requests
        .iter()
        .map(|request| (request.source_cluster, request.priority_bits))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(requested_priorities.len(), 2);

    pool.begin_frame(3);
    pool.make_group_resident(&queue, mesh, 4).unwrap();
    pool.make_group_resident(&queue, mesh, 6).unwrap();
    let (_, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(50.0),
    );
    assert!(requests.is_empty());
    let page_use_bytes = read_gpu_buffer(
        &device,
        &queue,
        selector.page_use_buffer(),
        selector.page_use_buffer().size(),
    );
    let page_uses = decode_records::<GpuVirtualPageUse>(
        &page_use_bytes,
        counters
            .page_use_count
            .min(selector.config().max_page_requests) as usize,
    );
    let resident_priorities = page_uses
        .iter()
        .filter(|page_use| requested_priorities.contains_key(&page_use.source_cluster))
        .map(|page_use| (page_use.source_cluster, page_use.priority_bits))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(resident_priorities, requested_priorities);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_reports_bounded_output_overflow_without_overwriting() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping hierarchy overflow oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(
        &device,
        &pool,
        GpuVirtualTraversalConfig {
            max_instances: 1,
            max_selected_clusters: 2,
            max_page_requests: 1,
        },
    )
    .unwrap();
    let instances = [GpuVirtualInstance::identity(mesh, 29)];
    let (selected, requests, counters) = run_traversal(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(50.0),
    );
    assert_eq!(selected.len(), 2);
    assert!(selected
        .iter()
        .all(|record| (4..=7).contains(&record.cluster_table_index)));
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 4);
    assert_eq!(counters.selected_overflow, 2);
    assert_eq!(counters.request_overflow, 0);

    let mut partial_pool = GpuVirtualGeometryPool::new(&device, gpu_config(3)).unwrap();
    let partial_mesh = partial_pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    bind_test_materials(&mut partial_pool, &queue, partial_mesh);
    partial_pool.begin_frame(2);
    partial_pool
        .make_group_resident(&queue, partial_mesh, 2)
        .unwrap();
    partial_pool
        .make_group_resident(&queue, partial_mesh, 3)
        .unwrap();
    let partial_selector = GpuVirtualHierarchySelector::new(
        &device,
        &partial_pool,
        GpuVirtualTraversalConfig {
            max_instances: 1,
            max_selected_clusters: 4,
            max_page_requests: 1,
        },
    )
    .unwrap();
    let partial_instances = [GpuVirtualInstance::identity(partial_mesh, 31)];
    let (_, requests, counters) = run_traversal(
        &device,
        &queue,
        &partial_pool,
        &partial_selector,
        &partial_instances,
        traversal_view(50.0),
    );
    assert_eq!(requests.len(), 1);
    assert!([3, 4].contains(&requests[0].page_index));
    assert_eq!(counters.page_request_count, 2);
    assert_eq!(counters.request_overflow, 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_cone_culling_is_conservative_for_transform_class() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping hierarchy cone oracle");
        return;
    };
    let mut archive = hierarchy_archive();
    for cluster in &mut archive.clusters {
        cluster.normal_cone_axis = [0.0, 0.0, 1.0];
        cluster.normal_cone_cutoff = 1.0;
    }
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(1)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(archive))
        .unwrap();
    bind_test_materials(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();

    let front = [GpuVirtualInstance::identity(mesh, 41)];
    let mut front_view = traversal_view(1_000.0);
    front_view.camera_position = [0.5, 0.5, 10.0];
    let (selected, _, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &front, front_view);
    assert_eq!(selected.len(), 2);
    assert_eq!(counters.cone_culled_clusters, 0);

    let mut back_view = front_view;
    back_view.camera_position = [0.5, 0.5, -10.0];
    let (selected, _, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &front, back_view);
    assert!(selected.is_empty());
    assert_eq!(counters.cone_culled_clusters, 2);

    let non_uniform = [GpuVirtualInstance::new(
        mesh,
        43,
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    .unwrap()];
    assert!(!non_uniform[0].cone_cull_safe());
    let (selected, _, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &non_uniform, back_view);
    assert_eq!(selected.len(), 2);
    assert_eq!(counters.cone_culled_clusters, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_is_stateless_across_camera_cuts_and_fast_instance_motion() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping hierarchy motion oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let translated = |instance_id, x| {
        GpuVirtualInstance::new(
            mesh,
            instance_id,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [x, 0.0, 0.0, 1.0],
            ],
        )
        .unwrap()
    };

    let before = [translated(51, 0.0), translated(53, 10.0)];
    let (_, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &before,
        traversal_view(50.0),
    );
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 8);
    assert_eq!(counters.selected_overflow, 0);

    let after = [translated(51, -15.0), translated(53, 25.0)];
    let mut cut_view = traversal_view(50.0);
    cut_view.camera_position = [-100.0, 80.0, -60.0];
    let (selected, requests, counters) =
        assert_traversal_matches_cpu(&device, &queue, &pool, &selector, &after, cut_view);
    assert_eq!(selected.len(), 8);
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 8);
    assert_eq!(counters.missing_current_pages, 0);
    assert_eq!(counters.invalid_records, 0);
}
