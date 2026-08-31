use super::*;

fn multi_source_mesh_hierarchy_archive() -> GeometryArchive {
    let mut archive = hierarchy_archive();
    // The two independent hierarchy trees share one archive and root page,
    // matching a glTF with two source meshes. Tree 0 is 0 -> 2 -> 4/5;
    // tree 1 is 1 -> 3 -> 6/7.
    for index in [1usize, 3, 6, 7] {
        archive.clusters[index].mesh_index = 1;
    }
    archive
}

#[test]
fn runtime_asset_reports_canonical_source_mesh_routes() {
    let mut archive = multi_source_mesh_hierarchy_archive();
    for cluster in archive
        .clusters
        .iter_mut()
        .filter(|cluster| cluster.mesh_index == 0)
    {
        cluster.flags |= FLAG_ALPHA_MASKED;
        cluster.material_index = Some(17);
    }
    let asset = hierarchy_asset(archive);
    assert_eq!(asset.source_root_span(0), Some(0..1));
    assert_eq!(asset.source_root_span(1), Some(1..2));
    assert_eq!(asset.source_root_span(7), None);
    let routes = asset.source_mesh_routes();
    assert_eq!(routes.len(), 3);
    assert_eq!(routes[0].source_mesh_index, 0);
    assert_eq!(routes[0].virtual_primitive_count, 0);
    assert!(routes[0].compatibility.is_empty());
    assert_eq!(
        routes[0].alpha_masked_compatibility,
        [VirtualGeometryAlphaMaskedRoute {
            mesh_index: 0,
            primitive_index: 0,
            material_index: Some(17),
        }]
    );
    assert_eq!(routes[1].source_mesh_index, 1);
    assert_eq!(routes[1].virtual_primitive_count, 1);
    assert!(routes[1].compatibility.is_empty());
    assert!(routes[1].alpha_masked_compatibility.is_empty());
    assert_eq!(routes[2].source_mesh_index, 7);
    assert_eq!(routes[2].virtual_primitive_count, 0);
    assert_eq!(
        routes[2].compatibility,
        [CompatibilityRecord {
            mesh_index: 7,
            primitive_index: 0,
            reason: CompatibilityReason::Skinned,
            detail: 0,
        }]
    );
    assert!(routes[2].alpha_masked_compatibility.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_filters_shared_archives_by_source_mesh_placement() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no eight-storage-buffer GPU adapter — skipping source-mesh filter oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(
            &queue,
            hierarchy_asset(multi_source_mesh_hierarchy_archive()),
        )
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();

    let source_zero =
        GpuVirtualInstance::for_source_mesh(mesh, 0, 41, crate::renderer::IDENTITY_MAT4).unwrap();
    let source_one =
        GpuVirtualInstance::for_source_mesh(mesh, 1, 43, crate::renderer::IDENTITY_MAT4).unwrap();
    assert_eq!(source_zero.source_mesh_index(), Some(0));
    assert_eq!(source_one.source_mesh_index(), Some(1));
    assert_eq!(
        GpuVirtualInstance::identity(mesh, 45).source_mesh_index(),
        None
    );

    let all_sources = GpuVirtualInstance::identity(mesh, 45);
    let mut rejected_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    assert_eq!(
        selector
            .record(
                &queue,
                &mut rejected_encoder,
                &pool,
                &[all_sources],
                traversal_view(50.0),
            )
            .unwrap_err(),
        VirtualGeometryTraversalError::SourceMeshFilterRequired { mesh }
    );

    let compatibility_only =
        GpuVirtualInstance::for_source_mesh(mesh, 7, 47, crate::renderer::IDENTITY_MAT4).unwrap();
    assert_eq!(
        selector
            .record(
                &queue,
                &mut rejected_encoder,
                &pool,
                &[compatibility_only],
                traversal_view(50.0),
            )
            .unwrap_err(),
        VirtualGeometryTraversalError::SourceMeshNotVirtual {
            mesh,
            source_mesh_index: 7,
        }
    );

    let selected_indices = |selected: Vec<GpuSelectedVirtualCluster>| {
        selected
            .into_iter()
            .map(|record| (record.instance_index, record.cluster_table_index))
            .collect::<Vec<_>>()
    };
    let (selected, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &[source_zero],
        traversal_view(50.0),
    );
    assert_eq!(selected_indices(selected), [(0, 4), (0, 5)]);
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 2);

    let (selected, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &[source_one],
        traversal_view(50.0),
    );
    assert_eq!(selected_indices(selected), [(0, 6), (0, 7)]);
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 2);

    let (mut selected, requests, counters) = assert_traversal_matches_cpu(
        &device,
        &queue,
        &pool,
        &selector,
        &[source_zero, source_one],
        traversal_view(50.0),
    );
    selected.sort_unstable();
    assert_eq!(selected_indices(selected), [(0, 4), (0, 5), (1, 6), (1, 7)]);
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 4);
}
