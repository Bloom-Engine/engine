#[cfg(not(target_arch = "wasm32"))]
use super::decode::VIRTUAL_GEOMETRY_DECODE_WGSL;
#[cfg(not(target_arch = "wasm32"))]
use super::traversal::select_cpu_reference;
use super::*;
use bloom_geometry_format::{
    hex_hash, sha256, CompatibilityReason, GeometryArchive, PageRecord, CLUSTER_RECORD_BYTES,
    COMPATIBILITY_RECORD_BYTES, ENDIAN_TAG, FLAG_COARSE_ROOT, HEADER_BYTES, MAGIC, MIN_PAGE_BYTES,
    NO_RELATION, PAGE_RECORD_BYTES, QUANTIZED_VERSION, VERSION,
};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
const VIRTUAL_GEOMETRY_DECODE_PROBE_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/decode_probe.wgsl");

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i16(bytes: &mut Vec<u8>, value: i16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32x3(bytes: &mut Vec<u8>, value: [f32; 3]) {
    for component in value {
        push_f32(bytes, component);
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn metadata_only_archive() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_BYTES);
    bytes.extend_from_slice(&MAGIC);
    push_u32(&mut bytes, VERSION);
    push_u32(&mut bytes, HEADER_BYTES as u32);
    push_u32(&mut bytes, ENDIAN_TAG);
    push_u32(&mut bytes, 0);
    bytes.extend_from_slice(&sha256(b"source"));
    bytes.extend_from_slice(&sha256(&[]));
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    for _ in 0..4 {
        push_u64(&mut bytes, HEADER_BYTES as u64);
    }
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, HEADER_BYTES as u64);
    push_u32(&mut bytes, MIN_PAGE_BYTES);
    push_u32(&mut bytes, 0);
    assert_eq!(bytes.len(), HEADER_BYTES);
    bytes
}

fn cluster(
    page_index: u32,
    lod_level: u32,
    parent: u32,
    first_child: u32,
    child_count: u32,
    coarse_root: bool,
) -> ClusterRecord {
    ClusterRecord {
        mesh_index: 0,
        primitive_index: 0,
        material_index: None,
        flags: if coarse_root { FLAG_COARSE_ROOT } else { 0 },
        page_index,
        vertex_count: 3,
        triangle_count: 1,
        lod_level,
        vertex_offset: 0,
        index_offset: 216,
        aabb_min: [0.0; 3],
        aabb_max: [1.0; 3],
        sphere_center: [0.5; 3],
        sphere_radius: 1.0,
        normal_cone_axis: [0.0, 0.0, 1.0],
        normal_cone_cutoff: -1.0,
        geometric_error: lod_level as f32,
        parent,
        parent_count: u32::from(parent != NO_RELATION),
        first_child,
        child_count,
        vertex_stride: 72,
    }
}

fn page(payload_offset: u64, first_cluster: u32, cluster_count: u32) -> PageRecord {
    PageRecord {
        payload_offset,
        payload_bytes: 100,
        first_cluster,
        cluster_count,
        sha256: [0; 32],
    }
}

fn hierarchy_archive() -> GeometryArchive {
    GeometryArchive {
        format_version: VERSION,
        vertex_encoding: VertexEncoding::Float32,
        source_sha256: [1; 32],
        payload_sha256: [2; 32],
        page_budget_bytes: MIN_PAGE_BYTES,
        file_payload_offset: 0,
        clusters: vec![
            cluster(0, 2, NO_RELATION, 2, 1, true),
            cluster(0, 2, NO_RELATION, 3, 1, true),
            cluster(1, 1, 0, 4, 2, false),
            cluster(2, 1, 1, 6, 2, false),
            cluster(3, 0, 2, NO_RELATION, 0, false),
            cluster(3, 0, 2, NO_RELATION, 0, false),
            cluster(4, 0, 3, NO_RELATION, 0, false),
            cluster(4, 0, 3, NO_RELATION, 0, false),
        ],
        pages: vec![
            page(0, 0, 2),
            page(100, 2, 1),
            page(200, 3, 1),
            page(300, 4, 2),
            page(400, 6, 2),
        ],
        compatibility: vec![CompatibilityRecord {
            mesh_index: 7,
            primitive_index: 0,
            reason: CompatibilityReason::Skinned,
            detail: 0,
        }],
    }
}

fn hierarchy_asset(archive: GeometryArchive) -> Arc<VirtualGeometryAsset> {
    Arc::new(VirtualGeometryAsset::from_bytes(encode_archive(archive)).unwrap())
}

fn split_leaf_group_archive() -> GeometryArchive {
    let mut archive = hierarchy_archive();
    archive.clusters[5].page_index = 4;
    archive.clusters[6].page_index = 5;
    archive.clusters[7].page_index = 5;
    archive.pages = vec![
        page(0, 0, 2),
        page(100, 2, 1),
        page(200, 3, 1),
        page(300, 4, 1),
        page(400, 5, 1),
        page(500, 6, 2),
    ];
    archive
}

fn large_intermediate_page_archive() -> GeometryArchive {
    let mut archive = hierarchy_archive();
    archive.clusters[2].vertex_count = 48;
    archive
}

fn encode_archive(mut archive: GeometryArchive) -> Vec<u8> {
    let mut payload = Vec::new();
    for (page_index, page) in archive.pages.iter_mut().enumerate() {
        let page_start = payload.len();
        let cluster_start = page.first_cluster as usize;
        let cluster_end = cluster_start + page.cluster_count as usize;
        for cluster in &mut archive.clusters[cluster_start..cluster_end] {
            cluster.page_index = page_index as u32;
            cluster.vertex_offset = payload.len() as u64;
            cluster.vertex_stride = archive.vertex_encoding.stride();
            for vertex_index in 0..cluster.vertex_count {
                let position = match vertex_index % 3 {
                    0 => [0.0, 0.0, 0.0],
                    1 => [1.0, 0.0, 0.0],
                    _ => [0.0, 1.0, 0.0],
                };
                encode_test_vertex(&mut payload, position, archive.vertex_encoding);
            }
            cluster.index_offset = payload.len() as u64;
            payload.extend_from_slice(&[0, 1, 2]);
            payload.resize(align_up(payload.len(), 16), 0);
        }
        page.payload_offset = page_start as u64;
        page.payload_bytes = (payload.len() - page_start) as u32;
        page.sha256 = sha256(&payload[page_start..]);
    }
    archive.payload_sha256 = sha256(&payload);

    let page_table_offset = HEADER_BYTES + archive.clusters.len() * CLUSTER_RECORD_BYTES;
    let compatibility_table_offset = page_table_offset + archive.pages.len() * PAGE_RECORD_BYTES;
    let payload_offset = align_up(
        compatibility_table_offset + archive.compatibility.len() * COMPATIBILITY_RECORD_BYTES,
        16,
    );
    let file_bytes = payload_offset + payload.len();
    let mut bytes = Vec::with_capacity(file_bytes);
    bytes.extend_from_slice(&MAGIC);
    push_u32(&mut bytes, archive.format_version);
    push_u32(&mut bytes, HEADER_BYTES as u32);
    push_u32(&mut bytes, ENDIAN_TAG);
    push_u32(&mut bytes, 0);
    bytes.extend_from_slice(&archive.source_sha256);
    bytes.extend_from_slice(&archive.payload_sha256);
    push_u32(&mut bytes, archive.clusters.len() as u32);
    push_u32(&mut bytes, archive.pages.len() as u32);
    push_u32(&mut bytes, archive.compatibility.len() as u32);
    push_u32(&mut bytes, 0);
    push_u64(&mut bytes, HEADER_BYTES as u64);
    push_u64(&mut bytes, page_table_offset as u64);
    push_u64(&mut bytes, compatibility_table_offset as u64);
    push_u64(&mut bytes, payload_offset as u64);
    push_u64(&mut bytes, payload.len() as u64);
    push_u64(&mut bytes, file_bytes as u64);
    push_u32(&mut bytes, archive.page_budget_bytes);
    push_u32(&mut bytes, 0);

    for cluster in &archive.clusters {
        push_u32(&mut bytes, cluster.mesh_index);
        push_u32(&mut bytes, cluster.primitive_index);
        push_u32(&mut bytes, cluster.material_index.unwrap_or(u32::MAX));
        push_u32(&mut bytes, cluster.flags);
        push_u32(&mut bytes, cluster.page_index);
        push_u32(&mut bytes, cluster.vertex_count);
        push_u32(&mut bytes, cluster.triangle_count);
        push_u32(&mut bytes, cluster.lod_level);
        push_u64(&mut bytes, cluster.vertex_offset);
        push_u64(&mut bytes, cluster.index_offset);
        push_f32x3(&mut bytes, cluster.aabb_min);
        push_f32x3(&mut bytes, cluster.aabb_max);
        push_f32x3(&mut bytes, cluster.sphere_center);
        push_f32(&mut bytes, cluster.sphere_radius);
        push_f32x3(&mut bytes, cluster.normal_cone_axis);
        push_f32(&mut bytes, cluster.normal_cone_cutoff);
        push_f32(&mut bytes, cluster.geometric_error);
        push_u32(&mut bytes, cluster.parent);
        push_u32(&mut bytes, cluster.first_child);
        push_u32(&mut bytes, cluster.child_count);
        push_u32(&mut bytes, cluster.vertex_stride);
        push_u32(&mut bytes, cluster.parent_count);
    }
    for page in &archive.pages {
        push_u64(&mut bytes, page.payload_offset);
        push_u32(&mut bytes, page.payload_bytes);
        push_u32(&mut bytes, page.first_cluster);
        push_u32(&mut bytes, page.cluster_count);
        push_u32(&mut bytes, 0);
        bytes.extend_from_slice(&page.sha256);
        push_u64(&mut bytes, 0);
    }
    for record in &archive.compatibility {
        push_u32(&mut bytes, record.mesh_index);
        push_u32(&mut bytes, record.primitive_index);
        push_u32(&mut bytes, record.reason as u32);
        push_u32(&mut bytes, record.detail);
    }
    bytes.resize(payload_offset, 0);
    bytes.extend_from_slice(&payload);
    assert_eq!(bytes.len(), file_bytes);
    bytes
}

fn encode_test_vertex(output: &mut Vec<u8>, position: [f32; 3], encoding: VertexEncoding) {
    match encoding {
        VertexEncoding::Float32 => {
            for value in position
                .into_iter()
                .chain([0.0, 0.0, 1.0])
                .chain([1.0, 0.0, 0.0, 1.0])
                .chain([position[0], position[1]])
                .chain([position[0] * 0.5, position[1] * 0.5])
                .chain([1.0, 0.5, 0.25, 1.0])
            {
                push_f32(output, value);
            }
        }
        VertexEncoding::Quantized => {
            for value in position {
                push_u16(output, if value == 0.0 { 0 } else { u16::MAX });
            }
            push_i16(output, 0);
            push_i16(output, 0);
            push_i16(output, i16::MAX);
            push_i16(output, 0);
            for value in [
                position[0],
                position[1],
                position[0] * 0.5,
                position[1] * 0.5,
            ] {
                push_u16(output, half::f16::from_f32(value).to_bits());
            }
            output.extend_from_slice(&[255, 128, 64, 255]);
            push_i16(output, i16::MAX);
            push_u16(output, 1);
            push_u16(output, 0);
        }
    }
}

#[test]
fn runtime_loader_rejects_corruption_and_verifies_index_identity() {
    let bytes = metadata_only_archive();
    let identity = ArtifactIdentity {
        bytes: bytes.len() as u64,
        format_version: VERSION,
        file_sha256: sha256(&bytes),
        payload_sha256: sha256(&[]),
        source_sha256: sha256(b"source"),
    };
    let asset = VirtualGeometryAsset::from_indexed_bytes(bytes.clone(), identity).unwrap();
    assert_eq!(asset.archive().format_version, VERSION);
    assert!(asset.archive().pages.is_empty());

    let mut corrupt = bytes.clone();
    corrupt[0] ^= 1;
    assert!(VirtualGeometryAsset::from_bytes(corrupt)
        .unwrap_err()
        .to_string()
        .contains("magic"));

    let mut wrong_identity = identity;
    wrong_identity.file_sha256[0] ^= 1;
    let error = VirtualGeometryAsset::from_indexed_bytes(bytes, wrong_identity).unwrap_err();
    assert!(error.to_string().contains("artifact hash mismatch"));
    assert!(error
        .to_string()
        .contains(&hex_hash(wrong_identity.file_sha256)));
}

#[test]
fn coarse_roots_are_pinned_and_fallback_is_deterministic() {
    let asset = hierarchy_asset(hierarchy_archive());
    let mut residency = VirtualGeometryResidency::new(asset, 1_120).unwrap();
    assert_eq!(residency.pinned_upload_pages(), 0..1);
    assert!(residency.is_page_resident(0));
    assert_eq!(residency.telemetry().resident_bytes, 448);

    let root = residency.resolve_cluster(4).unwrap().unwrap();
    assert_eq!(
        root.group,
        ClusterGroup {
            first_cluster: 0,
            cluster_count: 1
        }
    );
    assert_eq!(root.fallback_levels, 2);

    let mid_a = residency.make_group_resident(2).unwrap();
    assert_eq!(mid_a.upload_pages, vec![1]);
    assert!(mid_a.evict_pages.is_empty());
    let middle = residency.resolve_cluster(4).unwrap().unwrap();
    assert_eq!(
        middle.group,
        ClusterGroup {
            first_cluster: 2,
            cluster_count: 1
        }
    );
    assert_eq!(middle.fallback_levels, 1);

    let mid_b = residency.make_group_resident(3).unwrap();
    assert_eq!(mid_b.upload_pages, vec![2]);
    assert!(mid_b.evict_pages.is_empty());
    assert_eq!(residency.telemetry().resident_bytes, 896);
    assert_eq!(
        residency.resolve_cluster(4).unwrap().unwrap().group,
        ClusterGroup {
            first_cluster: 2,
            cluster_count: 1
        }
    );

    let leaves_a = residency.make_group_resident(4).unwrap();
    assert_eq!(leaves_a.upload_pages, vec![3]);
    assert_eq!(leaves_a.evict_pages, vec![2]);
    assert_eq!(leaves_a.resident_bytes, 1_120);
    let exact = residency.resolve_cluster(5).unwrap().unwrap();
    assert_eq!(
        exact.group,
        ClusterGroup {
            first_cluster: 4,
            cluster_count: 2
        }
    );
    assert_eq!(exact.fallback_levels, 0);
    assert!(residency.telemetry().resident_bytes <= residency.telemetry().budget_bytes);
}

#[test]
fn insufficient_budgets_fail_without_mutating_residency() {
    let archive = hierarchy_archive();
    let asset = hierarchy_asset(archive.clone());
    assert_eq!(
        VirtualGeometryResidency::new(Arc::clone(&asset), 447).unwrap_err(),
        ResidencyError::RootBudgetExceeded {
            required_bytes: 448,
            budget_bytes: 447,
        }
    );

    let mut residency =
        VirtualGeometryResidency::new(hierarchy_asset(split_leaf_group_archive()), 672).unwrap();
    let before = residency.telemetry();
    let error = residency.make_group_resident(4).unwrap_err();
    assert!(matches!(
        error,
        ResidencyError::GroupBudgetExceeded {
            required_bytes: 448,
            available_bytes: 224,
            ..
        }
    ));
    assert_eq!(residency.telemetry(), before);
    assert!(residency.is_page_resident(0));
    assert!(!residency.is_page_resident(3));
    assert!(!residency.is_page_resident(4));
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
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("virtual_geometry_pool_test_device"),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn try_traversal_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 7;
    let optional_indirect =
        wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("virtual_geometry_traversal_test_device"),
        required_limits: limits,
        required_features: adapter.features() & optional_indirect,
        ..Default::default()
    }))
    .ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn gpu_config(slot_count: u64) -> GpuVirtualGeometryConfig {
    GpuVirtualGeometryConfig {
        capacity_bytes: slot_count * u64::from(MIN_PAGE_BYTES),
        page_stride_bytes: MIN_PAGE_BYTES,
        max_meshes: 2,
        max_page_records: 16,
        max_cluster_records: 32,
        max_clusters_per_group: 16,
        max_hierarchy_levels: 8,
        max_upload_bytes_per_frame: 8 * u64::from(MIN_PAGE_BYTES),
        max_upload_pages_per_frame: 8,
        max_evictions_per_frame: 8,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_gpu_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    bytes: u64,
) -> Vec<u8> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("virtual_geometry_test_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_test_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = readback.slice(..);
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
        .expect("virtual-geometry readback callback dropped")
        .expect("virtual-geometry readback mapping failed");
    let mapped = slice.get_mapped_range();
    let result = mapped.to_vec();
    drop(mapped);
    readback.unmap();
    result
}

#[cfg(not(target_arch = "wasm32"))]
fn traversal_config() -> GpuVirtualTraversalConfig {
    GpuVirtualTraversalConfig {
        max_instances: 4,
        max_selected_clusters: 16,
        max_page_requests: 16,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn traversal_view(target_error_pixels: f32) -> VirtualGeometryView {
    VirtualGeometryView {
        frustum_planes: [[0.0, 0.0, 0.0, 1.0]; 6],
        view_projection: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_position: [0.5, 0.5, 10.0],
        projection_scale: 100.0,
        target_error_pixels,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_records<T: bytemuck::Pod>(bytes: &[u8], count: usize) -> Vec<T> {
    bytes
        .chunks_exact(std::mem::size_of::<T>())
        .take(count)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDecodedVirtualVertex {
    position: [f32; 4],
    normal: [f32; 4],
    tangent: [f32; 4],
    uv0_uv1: [f32; 4],
    color: [f32; 4],
    /// Selected record, cluster, corner, and page-local vertex index.
    info: [u32; 4],
}

#[cfg(not(target_arch = "wasm32"))]
const _: () = assert!(std::mem::size_of::<GpuDecodedVirtualVertex>() == 96);

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVirtualDecodeProbeParams {
    selected_count: u32,
    max_corners: u32,
    output_capacity: u32,
    reserved: u32,
}

#[cfg(not(target_arch = "wasm32"))]
fn gpu_buffer_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_virtual_decode_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pool: &GpuVirtualGeometryPool,
    selector: &GpuVirtualHierarchySelector,
    selected_count: u32,
    max_corners: u32,
) -> Vec<GpuDecodedVirtualVertex> {
    let output_count = selected_count
        .checked_mul(max_corners)
        .expect("virtual decode probe output count overflow");
    let output_bytes =
        u64::from(output_count) * std::mem::size_of::<GpuDecodedVirtualVertex>() as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("virtual_geometry_decode_probe_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("virtual_geometry_decode_probe_params"),
        size: std::mem::size_of::<GpuVirtualDecodeProbeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &params,
        0,
        bytemuck::bytes_of(&GpuVirtualDecodeProbeParams {
            selected_count,
            max_corners,
            output_capacity: output_count,
            reserved: 0,
        }),
    );

    let shader_source = [
        "@group(0) @binding(0) var<storage, read> virtual_page_words: BloomVirtualRawWords;\n",
        VIRTUAL_GEOMETRY_DECODE_WGSL,
        VIRTUAL_GEOMETRY_DECODE_PROBE_WGSL,
    ]
    .concat();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("virtual_geometry_decode_probe_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("virtual_geometry_decode_probe_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("decode_selected_corners"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("virtual_geometry_decode_probe_bind_group"),
        layout: &layout,
        entries: &[
            gpu_buffer_binding(0, pool.physical_buffer()),
            gpu_buffer_binding(1, pool.mesh_table_buffer()),
            gpu_buffer_binding(2, pool.cluster_table_buffer()),
            gpu_buffer_binding(3, selector.selected_buffer()),
            gpu_buffer_binding(4, &output),
            gpu_buffer_binding(5, &params),
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_decode_probe_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("virtual_geometry_decode_probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(max_corners.div_ceil(32), selected_count, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    decode_records(
        &read_gpu_buffer(device, queue, &output, output_bytes),
        output_count as usize,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_f32x4_close(actual: [f32; 4], expected: [f32; 4]) {
    for (component, expected) in actual.into_iter().zip(expected) {
        assert!(
            (component - expected).abs() <= 1.0e-5,
            "decoded component {component} did not match {expected}; actual vector was {actual:?}"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_decoded_test_vertices(
    decoded: &[GpuDecodedVirtualVertex],
    selected: &[GpuSelectedVirtualCluster],
    expected_color: [f32; 4],
) {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    for (selected_index, selection) in selected.iter().enumerate() {
        for (corner, position) in positions.into_iter().enumerate() {
            let vertex = decoded[selected_index * 3 + corner];
            assert_f32x4_close(
                vertex.position,
                [position[0], position[1], position[2], 1.0],
            );
            assert_f32x4_close(vertex.normal, [0.0, 0.0, 1.0, 0.0]);
            assert_f32x4_close(vertex.tangent, [1.0, 0.0, 0.0, 1.0]);
            assert_f32x4_close(
                vertex.uv0_uv1,
                [
                    position[0],
                    position[1],
                    position[0] * 0.5,
                    position[1] * 0.5,
                ],
            );
            assert_f32x4_close(vertex.color, expected_color);
            assert_eq!(
                vertex.info,
                [
                    selected_index as u32,
                    selection.cluster_index,
                    corner as u32,
                    corner as u32,
                ]
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_traversal(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pool: &GpuVirtualGeometryPool,
    selector: &GpuVirtualHierarchySelector,
    instances: &[GpuVirtualInstance],
    view: VirtualGeometryView,
) -> (
    Vec<GpuSelectedVirtualCluster>,
    Vec<GpuVirtualPageRequest>,
    GpuVirtualTraversalCounters,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_traversal_test_encoder"),
    });
    selector
        .record(queue, &mut encoder, pool, instances, view)
        .unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let counter_bytes = read_gpu_buffer(
        device,
        queue,
        selector.counter_buffer(),
        std::mem::size_of::<GpuVirtualTraversalCounters>() as u64,
    );
    let counters = bytemuck::pod_read_unaligned::<GpuVirtualTraversalCounters>(&counter_bytes);
    let selected_bytes = read_gpu_buffer(
        device,
        queue,
        selector.selected_buffer(),
        selector.selected_buffer().size(),
    );
    let request_bytes = read_gpu_buffer(
        device,
        queue,
        selector.page_request_buffer(),
        selector.page_request_buffer().size(),
    );
    let selected = decode_records(
        &selected_bytes,
        counters
            .selected_count
            .min(selector.config().max_selected_clusters) as usize,
    );
    let requests = decode_records(
        &request_bytes,
        counters
            .page_request_count
            .min(selector.config().max_page_requests) as usize,
    );
    (selected, requests, counters)
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_traversal_matches_cpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pool: &GpuVirtualGeometryPool,
    selector: &GpuVirtualHierarchySelector,
    instances: &[GpuVirtualInstance],
    view: VirtualGeometryView,
) -> (
    Vec<GpuSelectedVirtualCluster>,
    Vec<GpuVirtualPageRequest>,
    GpuVirtualTraversalCounters,
) {
    let (mut selected, mut requests, counters) =
        run_traversal(device, queue, pool, selector, instances, view);
    let mut cpu = select_cpu_reference(pool, selector.config(), instances, view).unwrap();
    selected.sort_unstable();
    requests.sort_unstable();
    cpu.selected.sort_unstable();
    cpu.requests.sort_unstable();
    assert_eq!(selected, cpu.selected);
    assert_eq!(requests, cpu.requests);
    assert_eq!(counters, cpu.counters);
    (selected, requests, counters)
}

#[cfg(not(target_arch = "wasm32"))]
fn make_hierarchy_fully_resident(
    pool: &mut GpuVirtualGeometryPool,
    queue: &wgpu::Queue,
    mesh: VirtualMeshId,
) {
    pool.begin_frame(2);
    for cluster in [2, 3, 4, 6] {
        pool.make_group_resident(queue, mesh, cluster).unwrap();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_gpu_raw_page_decode(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    archive: GeometryArchive,
    expected_color: [f32; 4],
) {
    let mut pool = GpuVirtualGeometryPool::new(device, gpu_config(5)).unwrap();
    let mesh = pool.register_mesh(queue, hierarchy_asset(archive)).unwrap();
    make_hierarchy_fully_resident(&mut pool, queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(device, &pool, traversal_config()).unwrap();
    let (selected, requests, counters) = run_traversal(
        device,
        queue,
        &pool,
        &selector,
        &[GpuVirtualInstance::identity(mesh, 69)],
        traversal_view(50.0),
    );
    assert_eq!(counters.selected_count, 4);
    assert_eq!(counters.selected_overflow, 0);
    assert_eq!(counters.invalid_records, 0);
    assert_eq!(counters.missing_current_pages, 0);
    assert!(requests.is_empty());
    assert_eq!(selected.len(), 4);
    let decoded = run_virtual_decode_probe(device, queue, &pool, &selector, 4, 3);
    assert_eq!(decoded.len(), 12);
    assert_decoded_test_vertices(&decoded, &selected, expected_color);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_raw_page_vertex_decoder_matches_float32_and_quantized_archives() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping raw-page decode oracle");
        return;
    };
    assert_gpu_raw_page_decode(&device, &queue, hierarchy_archive(), [1.0, 0.5, 0.25, 1.0]);

    let mut quantized = hierarchy_archive();
    quantized.format_version = QUANTIZED_VERSION;
    quantized.vertex_encoding = VertexEncoding::Quantized;
    assert_gpu_raw_page_decode(
        &device,
        &queue,
        quantized,
        [1.0, 128.0 / 255.0, 64.0 / 255.0, 1.0],
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn run_draw_emission(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pool: &GpuVirtualGeometryPool,
    selector: &GpuVirtualHierarchySelector,
    emitter: &GpuVirtualDrawEmitter,
    instances: &[GpuVirtualInstance],
    view: VirtualGeometryView,
) -> (
    Vec<GpuVirtualDrawIndirect>,
    GpuVirtualDrawEmissionState,
    GpuVirtualDispatchIndirect,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_draw_emission_test_encoder"),
    });
    selector
        .record(queue, &mut encoder, pool, instances, view)
        .unwrap();
    emitter.record(queue, &mut encoder, selector).unwrap();
    queue.submit(std::iter::once(encoder.finish()));
    let state_bytes = read_gpu_buffer(
        device,
        queue,
        emitter.state_buffer(),
        std::mem::size_of::<GpuVirtualDrawEmissionState>() as u64,
    );
    let state = bytemuck::pod_read_unaligned::<GpuVirtualDrawEmissionState>(&state_bytes);
    let dispatch_bytes = read_gpu_buffer(
        device,
        queue,
        emitter.dispatch_buffer(),
        std::mem::size_of::<GpuVirtualDispatchIndirect>() as u64,
    );
    let dispatch = bytemuck::pod_read_unaligned::<GpuVirtualDispatchIndirect>(&dispatch_bytes);
    let command_bytes = read_gpu_buffer(
        device,
        queue,
        emitter.command_buffer(),
        emitter.command_buffer().size(),
    );
    let commands = decode_records(&command_bytes, state.draw_count as usize);
    (commands, state, dispatch)
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_virtual_draw_emission_compacts_selected_clusters_into_exact_indirect_commands() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping virtual draw emission oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let emitter = GpuVirtualDrawEmitter::new(&device, &selector).unwrap();
    let (commands, state, dispatch) = run_draw_emission(
        &device,
        &queue,
        &pool,
        &selector,
        &emitter,
        &[GpuVirtualInstance::identity(mesh, 71)],
        traversal_view(50.0),
    );
    assert_eq!(
        dispatch,
        GpuVirtualDispatchIndirect {
            workgroups_x: 1,
            workgroups_y: 1,
            workgroups_z: 1,
        }
    );
    assert_eq!(state.draw_count, 4);
    assert_eq!(state.batch_fallback, 0);
    assert_eq!(state.selector_selected_count, 4);
    assert_eq!(state.selector_selected_overflow, 0);
    assert_eq!(state.selector_invalid_or_missing, 0);
    assert_eq!(state.emitted_triangles, 4);
    assert_eq!(state.emitted_draws, 4);
    assert_eq!(
        commands,
        (0..4)
            .map(|draw_index| GpuVirtualDrawIndirect {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: draw_index,
            })
            .collect::<Vec<_>>()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_virtual_draw_commands_execute_with_exact_first_instance_values() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping virtual indirect draw oracle");
        return;
    };
    if !device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        eprintln!("adapter lacks indirect first-instance — skipping virtual indirect draw oracle");
        return;
    }
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let emitter = GpuVirtualDrawEmitter::new(&device, &selector).unwrap();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("virtual_geometry_indirect_draw_oracle_shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) draw_id: u32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.24, -1.0),
        vec2<f32>(0.24, -1.0),
        vec2<f32>(0.0, 1.0),
    );
    let center_x = -0.75 + f32(instance_index) * 0.5;
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index] + vec2<f32>(center_x, 0.0), 0.0, 1.0);
    output.draw_id = instance_index + 1u;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<u32> {
    return vec4<u32>(input.draw_id, 0u, 0u, 255u);
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("virtual_geometry_indirect_draw_oracle_pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Uint,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual_geometry_indirect_draw_oracle_target"),
        size: wgpu::Extent3d {
            width: 4,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("virtual_geometry_indirect_draw_oracle_readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_indirect_draw_oracle_encoder"),
    });
    selector
        .record(
            &queue,
            &mut encoder,
            &pool,
            &[GpuVirtualInstance::identity(mesh, 77)],
            traversal_view(50.0),
        )
        .unwrap();
    emitter.record(&queue, &mut encoder, &selector).unwrap();
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("virtual_geometry_indirect_draw_oracle_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        if device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
        {
            pass.multi_draw_indirect_count(
                emitter.command_buffer(),
                0,
                emitter.state_buffer(),
                0,
                emitter.draw_capacity(),
            );
        } else {
            pass.multi_draw_indirect(emitter.command_buffer(), 0, 4);
        }
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    let slice = readback.slice(..);
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
        .expect("virtual indirect draw callback dropped")
        .expect("virtual indirect draw readback failed");
    let bytes = slice.get_mapped_range();
    for pixel in 0..4usize {
        assert_eq!(
            &bytes[pixel * 4..pixel * 4 + 4],
            &[(pixel + 1) as u8, 0, 0, 255],
            "indirect draw {pixel} did not rasterize its first_instance value"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_virtual_draw_emission_suppresses_the_whole_batch_on_selection_overflow() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping virtual draw fallback oracle");
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
            max_page_requests: 2,
        },
    )
    .unwrap();
    let emitter = GpuVirtualDrawEmitter::new(&device, &selector).unwrap();
    let (commands, state, dispatch) = run_draw_emission(
        &device,
        &queue,
        &pool,
        &selector,
        &emitter,
        &[GpuVirtualInstance::identity(mesh, 73)],
        traversal_view(50.0),
    );
    assert!(commands.is_empty());
    assert_eq!(
        dispatch,
        GpuVirtualDispatchIndirect {
            workgroups_x: 0,
            workgroups_y: 1,
            workgroups_z: 1,
        }
    );
    assert_eq!(state.draw_count, 0);
    assert_eq!(state.batch_fallback, 1);
    assert_eq!(state.selector_selected_count, 4);
    assert_eq!(state.selector_selected_overflow, 2);
    assert_eq!(state.emitted_triangles, 0);
    assert_eq!(state.emitted_draws, 0);

    let other_selector = GpuVirtualHierarchySelector::new(
        &device,
        &pool,
        GpuVirtualTraversalConfig {
            max_instances: 1,
            max_selected_clusters: 2,
            max_page_requests: 2,
        },
    )
    .unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_draw_emitter_identity_encoder"),
    });
    assert_eq!(
        emitter
            .record(&queue, &mut encoder, &other_selector)
            .unwrap_err(),
        VirtualGeometryDrawEmissionError::SelectorMismatch
    );

    // A bounded request overflow does not invalidate the already-resident
    // ancestor selection. It must keep drawing that complete fallback batch.
    let mut partial_pool = GpuVirtualGeometryPool::new(&device, gpu_config(3)).unwrap();
    let partial_mesh = partial_pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
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
    let partial_emitter = GpuVirtualDrawEmitter::new(&device, &partial_selector).unwrap();
    let (commands, state, _) = run_draw_emission(
        &device,
        &queue,
        &partial_pool,
        &partial_selector,
        &partial_emitter,
        &[GpuVirtualInstance::identity(partial_mesh, 79)],
        traversal_view(50.0),
    );
    assert_eq!(commands.len(), 2);
    assert_eq!(state.draw_count, 2);
    assert_eq!(state.batch_fallback, 0);
    assert_eq!(state.selector_selected_overflow, 0);
    assert_eq!(state.selector_invalid_or_missing, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_matches_cpu_across_lod_and_frustum_decisions() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping hierarchy selector oracle");
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
            .map(|record| record.cluster_index)
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
            .map(|record| record.cluster_index)
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
            .map(|record| record.cluster_index)
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
fn gpu_hierarchy_selector_keeps_resident_ancestors_and_requests_missing_pages() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping hierarchy fallback oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(3)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
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
            .map(|record| record.cluster_index)
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
fn gpu_hierarchy_selector_reports_bounded_output_overflow_without_overwriting() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping hierarchy overflow oracle");
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
        .all(|record| (4..=7).contains(&record.cluster_index)));
    assert!(requests.is_empty());
    assert_eq!(counters.selected_count, 4);
    assert_eq!(counters.selected_overflow, 2);
    assert_eq!(counters.request_overflow, 0);

    let mut partial_pool = GpuVirtualGeometryPool::new(&device, gpu_config(3)).unwrap();
    let partial_mesh = partial_pool
        .register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
        .unwrap();
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
        eprintln!("no seven-storage-buffer GPU adapter — skipping hierarchy cone oracle");
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
        eprintln!("no seven-storage-buffer GPU adapter — skipping hierarchy motion oracle");
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

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_pool_uploads_validated_pages_and_matches_its_gpu_tables() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping virtual-geometry pool oracle");
        return;
    };
    let asset = hierarchy_asset(hierarchy_archive());
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(3)).unwrap();
    assert_eq!(pool.physical_buffer().size(), 3 * u64::from(MIN_PAGE_BYTES));
    assert_eq!(
        pool.page_table_buffer().size(),
        u64::from(pool.config().max_page_records)
            * std::mem::size_of::<GpuVirtualPageEntry>() as u64
    );
    assert_eq!(
        pool.mesh_table_buffer().size(),
        u64::from(pool.config().max_meshes) * std::mem::size_of::<GpuVirtualMeshEntry>() as u64
    );
    assert_eq!(
        pool.cluster_table_buffer().size(),
        u64::from(pool.config().max_cluster_records)
            * std::mem::size_of::<GpuVirtualClusterEntry>() as u64
    );
    let mesh = pool.register_mesh(&queue, Arc::clone(&asset)).unwrap();
    assert_eq!(
        pool.telemetry().resident_slot_bytes,
        u64::from(MIN_PAGE_BYTES)
    );
    assert_eq!(pool.telemetry().pinned_pages, 1);

    pool.begin_frame(2);
    pool.make_group_resident(&queue, mesh, 2).unwrap();
    pool.make_group_resident(&queue, mesh, 3).unwrap();
    let transition = pool.make_group_resident(&queue, mesh, 4).unwrap();
    assert_eq!(transition.uploaded.len(), 1);
    assert_eq!(transition.evicted.len(), 1);
    assert_eq!(
        transition.resident_slot_bytes,
        3 * u64::from(MIN_PAGE_BYTES)
    );
    assert_eq!(
        pool.resolve_cluster(mesh, 4)
            .unwrap()
            .unwrap()
            .fallback_levels,
        0
    );

    let page = VirtualPageId {
        mesh,
        page_index: 3,
    };
    let page_entry = pool.page_entry(page).unwrap();
    assert_ne!(page_entry.slot_plus_one, 0);
    assert_eq!(page_entry.mesh_id, mesh.raw());
    let physical = read_gpu_buffer(
        &device,
        &queue,
        pool.physical_buffer(),
        pool.config().capacity_bytes,
    );
    let physical_offset =
        (page_entry.slot_plus_one as usize - 1) * pool.config().page_stride_bytes as usize;
    let expected = asset.page_bytes(3).unwrap();
    assert_eq!(
        &physical[physical_offset..physical_offset + expected.len()],
        expected
    );

    let page_table = read_gpu_buffer(
        &device,
        &queue,
        pool.page_table_buffer(),
        u64::from(pool.config().max_page_records)
            * std::mem::size_of::<GpuVirtualPageEntry>() as u64,
    );
    let gpu_pages = bytemuck::cast_slice::<u8, GpuVirtualPageEntry>(&page_table);
    let mesh_entry = pool.mesh_entry(mesh).unwrap();
    assert_eq!(
        gpu_pages[(mesh_entry.page_table_base + page.page_index) as usize],
        page_entry
    );
    let mesh_table = read_gpu_buffer(
        &device,
        &queue,
        pool.mesh_table_buffer(),
        u64::from(pool.config().max_meshes) * std::mem::size_of::<GpuVirtualMeshEntry>() as u64,
    );
    let gpu_meshes = bytemuck::cast_slice::<u8, GpuVirtualMeshEntry>(&mesh_table);
    assert_eq!(gpu_meshes[mesh.descriptor_index() as usize - 1], mesh_entry);
    let cluster_table = read_gpu_buffer(
        &device,
        &queue,
        pool.cluster_table_buffer(),
        u64::from(pool.config().max_cluster_records)
            * std::mem::size_of::<GpuVirtualClusterEntry>() as u64,
    );
    let gpu_clusters = bytemuck::cast_slice::<u8, GpuVirtualClusterEntry>(&cluster_table);
    let cluster_entry = pool.cluster_entry(mesh, 4).unwrap();
    assert_eq!(
        gpu_clusters[(mesh_entry.cluster_table_base + 4) as usize],
        cluster_entry
    );
    assert_eq!(cluster_entry.page_lod_counts, [3, 0, 3, 1]);
    assert_eq!(cluster_entry.payload, [0, 216, 72, 0]);

    let telemetry = pool.telemetry();
    assert_eq!(telemetry.capacity_bytes, 3 * u64::from(MIN_PAGE_BYTES));
    assert_eq!(telemetry.resident_slot_bytes, telemetry.capacity_bytes);
    assert_eq!(telemetry.resident_payload_bytes, 1_120);
    assert_eq!(telemetry.page_table_bytes, 256);
    assert_eq!(telemetry.mesh_table_bytes, 96);
    assert_eq!(telemetry.cluster_table_bytes, 4_096);
    assert_eq!(telemetry.total_gpu_bytes, 16_736);
    assert_eq!(telemetry.live_cluster_records, 8);
    assert_eq!(telemetry.frame_upload_pages, 3);
    assert_eq!(telemetry.frame_upload_bytes, 896);
    assert_eq!(telemetry.frame_evictions, 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_pool_cluster_table_exhaustion_is_preflight_atomic() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping virtual-geometry cluster-capacity oracle");
        return;
    };
    let mut config = gpu_config(1);
    config.max_cluster_records = 7;
    let mut pool = GpuVirtualGeometryPool::new(&device, config).unwrap();
    let before = pool.telemetry();
    assert_eq!(
        pool.register_mesh(&queue, hierarchy_asset(hierarchy_archive()))
            .unwrap_err(),
        VirtualGeometryGpuError::ClusterTableExhausted
    );
    assert_eq!(pool.telemetry(), before);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_hierarchy_selector_rejects_other_pools_and_retiring_meshes_before_dispatch() {
    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no seven-storage-buffer GPU adapter — skipping selector ownership oracle");
        return;
    };
    let asset = hierarchy_asset(hierarchy_archive());
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(1)).unwrap();
    let mesh = pool.register_mesh(&queue, Arc::clone(&asset)).unwrap();
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let mut other_pool = GpuVirtualGeometryPool::new(&device, gpu_config(1)).unwrap();
    let other_mesh = other_pool.register_mesh(&queue, asset).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_geometry_selector_ownership_encoder"),
    });
    assert_eq!(
        selector
            .record(
                &queue,
                &mut encoder,
                &other_pool,
                &[GpuVirtualInstance::identity(other_mesh, 61)],
                traversal_view(50.0),
            )
            .unwrap_err(),
        VirtualGeometryTraversalError::PoolMismatch
    );

    pool.retire_mesh(&queue, mesh).unwrap();
    assert!(matches!(
        selector.record(
            &queue,
            &mut encoder,
            &pool,
            &[GpuVirtualInstance::identity(mesh, 63)],
            traversal_view(50.0),
        ),
        Err(VirtualGeometryTraversalError::Pool(
            VirtualGeometryGpuError::RetiringMesh(id)
        )) if id == mesh
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_pool_rejects_an_atomic_group_when_only_one_slot_is_replaceable() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping virtual-geometry atomicity oracle");
        return;
    };
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(2)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(split_leaf_group_archive()))
        .unwrap();
    pool.begin_frame(2);
    let before = pool.telemetry();
    let before_a = pool
        .page_entry(VirtualPageId {
            mesh,
            page_index: 3,
        })
        .unwrap();
    let before_b = pool
        .page_entry(VirtualPageId {
            mesh,
            page_index: 4,
        })
        .unwrap();
    assert_eq!(
        pool.make_group_resident(&queue, mesh, 4).unwrap_err(),
        VirtualGeometryGpuError::PhysicalPoolExhausted {
            requested_pages: 2,
            available_pages: 1,
        }
    );
    assert_eq!(pool.telemetry(), before);
    assert_eq!(
        pool.page_entry(VirtualPageId {
            mesh,
            page_index: 3,
        })
        .unwrap(),
        before_a
    );
    assert_eq!(
        pool.page_entry(VirtualPageId {
            mesh,
            page_index: 4,
        })
        .unwrap(),
        before_b
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn retired_mesh_ids_and_slots_are_reused_only_after_gpu_completion() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping virtual-geometry retirement oracle");
        return;
    };
    let asset = hierarchy_asset(hierarchy_archive());
    let mut config = gpu_config(2);
    config.max_meshes = 1;
    let mut pool = GpuVirtualGeometryPool::new(&device, config).unwrap();
    let old = pool.register_mesh(&queue, Arc::clone(&asset)).unwrap();
    pool.retire_mesh(&queue, old).unwrap();
    assert!(matches!(
        pool.mesh_entry(old),
        Err(VirtualGeometryGpuError::RetiringMesh(id)) if id == old
    ));
    assert_eq!(pool.telemetry().retiring_slots, 1);
    assert_eq!(
        pool.register_mesh(&queue, Arc::clone(&asset)).unwrap_err(),
        VirtualGeometryGpuError::MeshTableExhausted
    );

    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    assert_eq!(pool.collect_completed(), 1);
    pool.begin_frame(2);
    let new = pool.register_mesh(&queue, asset).unwrap();
    assert_ne!(old.raw(), new.raw());
    assert!(matches!(
        pool.mesh_entry(old),
        Err(VirtualGeometryGpuError::StaleMesh(id)) if id == old
    ));
    assert_eq!(new.descriptor_index(), old.descriptor_index());
}

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
