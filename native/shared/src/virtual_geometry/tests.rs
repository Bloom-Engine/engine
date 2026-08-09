use super::*;
use bloom_geometry_format::{
    hex_hash, sha256, CompatibilityReason, GeometryArchive, PageRecord, CLUSTER_RECORD_BYTES,
    COMPATIBILITY_RECORD_BYTES, ENDIAN_TAG, FLAG_COARSE_ROOT, HEADER_BYTES, MAGIC, MIN_PAGE_BYTES,
    NO_RELATION, PAGE_RECORD_BYTES, VERSION,
};
use std::sync::Arc;

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
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

fn encode_archive(mut archive: GeometryArchive) -> Vec<u8> {
    let mut payload = Vec::new();
    for (page_index, page) in archive.pages.iter_mut().enumerate() {
        let page_start = payload.len();
        let cluster_start = page.first_cluster as usize;
        let cluster_end = cluster_start + page.cluster_count as usize;
        for cluster in &mut archive.clusters[cluster_start..cluster_end] {
            cluster.page_index = page_index as u32;
            cluster.vertex_offset = payload.len() as u64;
            for _ in 0..cluster.vertex_count {
                for _ in 0..18 {
                    push_f32(&mut payload, 0.0);
                }
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
