use super::*;

#[test]
fn selected_records_are_render_ready_across_multiple_meshes() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping multi-mesh decode oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(10)).unwrap();
    let first = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, first);

    let mut quantized = hierarchy_archive();
    quantized.format_version = QUANTIZED_VERSION;
    quantized.vertex_encoding = VertexEncoding::Quantized;
    let second = pool
        .register_mesh(&queue, hierarchy_asset(quantized))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, second);

    let mesh_entry = pool.mesh_entry(second).unwrap();
    assert!(mesh_entry.cluster_table_base > 0);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let (selected, requests, counters) = run_traversal(
        &device,
        &queue,
        &pool,
        &selector,
        &[GpuVirtualInstance::identity(second, 73)],
        traversal_view(50.0),
    );
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 4);
    assert_eq!(selected.len(), 4);
    for selection in &selected {
        assert_eq!(selection.mesh_id, second.raw());
        assert!(
            (mesh_entry.cluster_table_base + 4..=mesh_entry.cluster_table_base + 7)
                .contains(&selection.cluster_table_index)
        );
        assert!(selection.physical_page_base < pool.config().capacity_bytes as u32);
        assert_eq!(
            selection.physical_page_base % mesh_entry.page_stride_bytes,
            0
        );
        assert_eq!(selection.vertex_encoding(), mesh_entry.vertex_encoding);
        assert_eq!(
            pool.cluster_entry(
                second,
                selection.cluster_table_index - mesh_entry.cluster_table_base
            )
            .unwrap()
            .payload[3],
            second.raw()
        );
    }

    let decoded = run_virtual_decode_probe(&device, &queue, &pool, &selector, 4, 3);
    assert_decoded_test_vertices(&decoded, &selected, [1.0, 128.0 / 255.0, 64.0 / 255.0, 1.0]);
}

fn material_archive() -> GeometryArchive {
    let mut archive = hierarchy_archive();
    for cluster_index in [0, 2, 4, 5] {
        archive.clusters[cluster_index].material_index = Some(3);
    }
    for cluster_index in [1, 3, 6, 7] {
        archive.clusters[cluster_index].material_index = Some(9);
    }
    archive
}

fn translated(x: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, 0.0, 0.0, 1.0],
    ]
}

fn transform(model: [[f32; 4]; 4], position: [f32; 3]) -> [f32; 4] {
    [
        model[0][0] * position[0]
            + model[1][0] * position[1]
            + model[2][0] * position[2]
            + model[3][0],
        model[0][1] * position[0]
            + model[1][1] * position[1]
            + model[2][1] * position[2]
            + model[3][1],
        model[0][2] * position[0]
            + model[1][2] * position[1]
            + model[2][2] * position[2]
            + model[3][2],
        1.0,
    ]
}

#[test]
fn temporal_instance_state_is_finite_affine_and_preserves_the_traversal_prefix() {
    let mesh = VirtualMeshId::from_raw(1);
    let current = translated(7.0);
    let previous = translated(4.0);
    let tint = [0.25, 0.5, 0.75, 0.9];
    let instance =
        GpuVirtualInstance::with_render_state(mesh, 41, current, previous, tint).unwrap();
    assert_eq!(std::mem::size_of::<GpuVirtualInstance>(), 224);
    assert_eq!(instance.mesh_id(), mesh);
    assert_eq!(instance.instance_id(), 41);
    assert_eq!(instance.model(), current);
    assert_eq!(instance.previous_model(), previous);
    assert_eq!(instance.model_tint(), tint);

    let mut non_affine_previous = previous;
    non_affine_previous[0][3] = 1.0;
    assert!(matches!(
        GpuVirtualInstance::with_render_state(mesh, 43, current, non_affine_previous, tint),
        Err(VirtualGeometryTraversalError::InvalidInstanceTransform { instance: 43 })
    ));
    let mut invalid_tint = tint;
    invalid_tint[2] = f32::NAN;
    assert!(matches!(
        GpuVirtualInstance::with_render_state(mesh, 47, current, previous, invalid_tint),
        Err(VirtualGeometryTraversalError::InvalidInstanceTransform { instance: 47 })
    ));
}

#[test]
fn material_binding_is_atomic_and_temporal_gpu_decode_uses_dense_instances() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping temporal material oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(material_archive()))
        .unwrap();
    let before = pool.cluster_entry(mesh, 4).unwrap();
    assert_eq!(before.identity[2], 0);
    assert_eq!(pool.mesh_entry(mesh).unwrap().flags, GPU_VIRTUAL_MESH_VALID);

    let binding = |source_material_index, material_id| VirtualMaterialBinding {
        source_material_index: Some(source_material_index),
        material_id,
    };
    let unbound_selector =
        GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let mut unbound_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_unbound_material_encoder"),
    });
    assert_eq!(
        unbound_selector
            .record(
                &queue,
                &mut unbound_encoder,
                &pool,
                &[GpuVirtualInstance::identity(mesh, 499)],
                traversal_view(50.0),
            )
            .unwrap_err(),
        VirtualGeometryTraversalError::UnboundMaterials { mesh }
    );
    assert_eq!(
        pool.bind_mesh_materials(&queue, mesh, &[binding(3, 0), binding(9, 202)])
            .unwrap_err(),
        VirtualGeometryGpuError::InvalidMaterialBinding(Some(3))
    );
    assert_eq!(
        pool.bind_mesh_materials(&queue, mesh, &[binding(3, 101)])
            .unwrap_err(),
        VirtualGeometryGpuError::MissingMaterialBinding(Some(9))
    );
    assert_eq!(pool.cluster_entry(mesh, 4).unwrap(), before);
    assert_eq!(
        pool.bind_mesh_materials(&queue, mesh, &[binding(3, 101), binding(3, 102)])
            .unwrap_err(),
        VirtualGeometryGpuError::DuplicateMaterialBinding(Some(3))
    );
    assert_eq!(
        pool.bind_mesh_materials(
            &queue,
            mesh,
            &[binding(3, 101), binding(9, 202), binding(11, 303)],
        )
        .unwrap_err(),
        VirtualGeometryGpuError::UnusedMaterialBinding(Some(11))
    );
    assert_eq!(pool.cluster_entry(mesh, 4).unwrap(), before);

    pool.bind_mesh_materials(&queue, mesh, &[binding(9, 202), binding(3, 101)])
        .unwrap();
    assert_eq!(pool.cluster_entry(mesh, 4).unwrap().identity[2], 101);
    assert_eq!(pool.cluster_entry(mesh, 6).unwrap().identity[2], 202);
    assert_eq!(
        pool.mesh_entry(mesh).unwrap().flags,
        GPU_VIRTUAL_MESH_VALID | GPU_VIRTUAL_MESH_MATERIALS_BOUND
    );

    let mesh_entry = pool.mesh_entry(mesh).unwrap();
    let cluster_bytes = read_gpu_buffer(
        &device,
        &queue,
        pool.cluster_table_buffer(),
        pool.cluster_table_buffer().size(),
    );
    let gpu_clusters = decode_records::<GpuVirtualClusterEntry>(
        &cluster_bytes,
        pool.config().max_cluster_records as usize,
    );
    assert_eq!(
        gpu_clusters[(mesh_entry.cluster_table_base + 4) as usize].identity[2],
        101
    );
    assert_eq!(
        gpu_clusters[(mesh_entry.cluster_table_base + 6) as usize].identity[2],
        202
    );

    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let instances = [
        GpuVirtualInstance::with_render_state(
            mesh,
            501,
            translated(0.0),
            translated(-2.0),
            [1.0, 0.5, 0.25, 1.0],
        )
        .unwrap(),
        GpuVirtualInstance::with_render_state(
            mesh,
            777,
            translated(10.0),
            translated(8.0),
            [0.5, 1.0, 2.0, 0.75],
        )
        .unwrap(),
    ];
    let (selected, requests, counters) = run_traversal(
        &device,
        &queue,
        &pool,
        &selector,
        &instances,
        traversal_view(50.0),
    );
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 8);
    assert_eq!(selected.len(), 8);
    assert_eq!(
        selected
            .iter()
            .filter(|selection| selection.instance_index == 0)
            .count(),
        4
    );
    assert_eq!(
        selected
            .iter()
            .filter(|selection| selection.instance_index == 1)
            .count(),
        4
    );
    for selection in &selected {
        let expected = if [4, 5].contains(&selection.cluster_table_index) {
            101
        } else {
            202
        };
        assert_eq!(selection.material_id, expected);
    }

    let decoded = run_virtual_decode_probe(&device, &queue, &pool, &selector, 8, 3);
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let raw_color = [1.0, 0.5, 0.25, 1.0];
    for (selected_index, selection) in selected.iter().enumerate() {
        let instance = instances[selection.instance_index as usize];
        for (corner, position) in positions.into_iter().enumerate() {
            let vertex = decoded[selected_index * 3 + corner];
            assert_f32x4_close(vertex.current_world, transform(instance.model(), position));
            assert_f32x4_close(
                vertex.previous_world,
                transform(instance.previous_model(), position),
            );
            let tint = instance.model_tint();
            assert_f32x4_close(
                vertex.tinted_color,
                [
                    raw_color[0] * tint[0],
                    raw_color[1] * tint[1],
                    raw_color[2] * tint[2],
                    raw_color[3] * tint[3],
                ],
            );
            assert_f32x4_close(vertex.world_normal, [0.0, 0.0, 1.0, 0.0]);
        }
    }
}
