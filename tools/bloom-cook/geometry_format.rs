//! Versioned, little-endian cooked geometry container.
//!
//! Version 1 stores deterministic meshlets in independently hashed,
//! budget-bounded pages. Its fixed cluster record supports either the default
//! leaf-only artifact or opt-in atomic parent/child replacement groups. All
//! offsets and hierarchy relations are validated before payload access.

use crate::geometry_quantization::{self, QuantizationStats};
use crate::meshlet::Meshlet;
pub use bloom_geometry_format::*;

pub fn encode_geometry(
    meshlets: &[Meshlet],
    compatibility: &[CompatibilityRecord],
    source_sha256: [u8; 32],
    page_budget_bytes: u32,
) -> Result<Vec<u8>, String> {
    encode_geometry_with_vertex_encoding(
        meshlets,
        compatibility,
        source_sha256,
        page_budget_bytes,
        VertexEncoding::Float32,
    )
}

pub fn encode_geometry_with_vertex_encoding(
    meshlets: &[Meshlet],
    compatibility: &[CompatibilityRecord],
    source_sha256: [u8; 32],
    page_budget_bytes: u32,
    vertex_encoding: VertexEncoding,
) -> Result<Vec<u8>, String> {
    validate_page_budget(page_budget_bytes)?;

    let mut payload = Vec::new();
    let mut clusters = Vec::with_capacity(meshlets.len());
    let mut pages = Vec::<PageRecord>::new();
    let mut page_start = 0usize;
    let mut page_first_cluster = 0usize;
    let mut previous_page_class = None;

    for meshlet in meshlets {
        validate_meshlet(meshlet)?;
        let encoded_bytes = align_up(
            geometry_quantization::encoded_meshlet_bytes(meshlet, vertex_encoding),
            16,
        );
        if encoded_bytes > page_budget_bytes as usize {
            return Err(format!(
                "meshlet payload {encoded_bytes} bytes exceeds page budget {page_budget_bytes}"
            ));
        }
        let current_page_bytes = payload.len() - page_start;
        let coarse_root = meshlet.flags & FLAG_COARSE_ROOT != 0;
        // Every coarse-root page is pinned for the complete lifetime of the
        // registered mesh. Root LOD therefore does not define a different
        // residency class, and separating mesh-first roots by level wastes a
        // physical 64 KiB slot for each short run. Streamable pages retain the
        // strict one-level-per-page contract used by atomic refinement.
        let page_class = (coarse_root, if coarse_root { 0 } else { meshlet.lod_level });
        if current_page_bytes > 0
            && (previous_page_class != Some(page_class)
                || current_page_bytes
                    .checked_add(encoded_bytes)
                    .is_none_or(|bytes| bytes > page_budget_bytes as usize))
        {
            finish_page(
                &payload,
                page_start,
                page_first_cluster,
                clusters.len(),
                &mut pages,
            )?;
            page_start = payload.len();
            page_first_cluster = clusters.len();
        }
        previous_page_class = Some(page_class);

        let page_index = pages.len() as u32;
        let vertex_offset = payload.len() as u64;
        for vertex in &meshlet.vertices {
            geometry_quantization::encode_vertex(
                &mut payload,
                vertex,
                meshlet.bounds.aabb_min,
                meshlet.bounds.aabb_max,
                vertex_encoding,
            )?;
        }
        let index_offset = payload.len() as u64;
        payload.extend_from_slice(&meshlet.local_indices);
        payload.resize(align_up(payload.len(), 16), 0);

        clusters.push(ClusterRecord {
            mesh_index: meshlet.mesh_index,
            primitive_index: meshlet.primitive_index,
            material_index: meshlet.material_index,
            flags: meshlet.flags,
            page_index,
            vertex_count: meshlet.vertices.len() as u32,
            triangle_count: meshlet.triangle_count(),
            lod_level: meshlet.lod_level,
            vertex_offset,
            index_offset,
            aabb_min: meshlet.bounds.aabb_min,
            aabb_max: meshlet.bounds.aabb_max,
            sphere_center: meshlet.bounds.sphere_center,
            sphere_radius: meshlet.bounds.sphere_radius,
            normal_cone_axis: meshlet.bounds.normal_cone_axis,
            normal_cone_cutoff: meshlet.bounds.normal_cone_cutoff,
            geometric_error: meshlet.geometric_error,
            parent: meshlet.parent,
            parent_count: meshlet.parent_count,
            first_child: meshlet.first_child,
            child_count: meshlet.child_count,
            vertex_stride: vertex_encoding.stride(),
        });
    }
    finish_page(
        &payload,
        page_start,
        page_first_cluster,
        clusters.len(),
        &mut pages,
    )?;

    let cluster_table_offset = HEADER_BYTES;
    let page_table_offset = checked_table_end(
        cluster_table_offset,
        clusters.len(),
        CLUSTER_RECORD_BYTES,
        "cluster table",
    )?;
    let compatibility_table_offset = checked_table_end(
        page_table_offset,
        pages.len(),
        PAGE_RECORD_BYTES,
        "page table",
    )?;
    let compatibility_end = checked_table_end(
        compatibility_table_offset,
        compatibility.len(),
        COMPATIBILITY_RECORD_BYTES,
        "compatibility table",
    )?;
    let payload_offset = align_up(compatibility_end, 16);
    let file_bytes = payload_offset
        .checked_add(payload.len())
        .ok_or("geometry file length overflow")?;
    let payload_sha256 = sha256(&payload);

    let mut output = Vec::with_capacity(file_bytes);
    output.extend_from_slice(&MAGIC);
    push_u32(
        &mut output,
        match vertex_encoding {
            VertexEncoding::Float32 => VERSION,
            VertexEncoding::Quantized => QUANTIZED_VERSION,
        },
    );
    push_u32(&mut output, HEADER_BYTES as u32);
    push_u32(&mut output, ENDIAN_TAG);
    push_u32(&mut output, 0);
    output.extend_from_slice(&source_sha256);
    output.extend_from_slice(&payload_sha256);
    push_u32(&mut output, checked_u32(clusters.len(), "cluster count")?);
    push_u32(&mut output, checked_u32(pages.len(), "page count")?);
    push_u32(
        &mut output,
        checked_u32(compatibility.len(), "compatibility count")?,
    );
    push_u32(&mut output, 0);
    push_u64(&mut output, cluster_table_offset as u64);
    push_u64(&mut output, page_table_offset as u64);
    push_u64(&mut output, compatibility_table_offset as u64);
    push_u64(&mut output, payload_offset as u64);
    push_u64(&mut output, payload.len() as u64);
    push_u64(&mut output, file_bytes as u64);
    push_u32(&mut output, page_budget_bytes);
    push_u32(&mut output, 0);
    debug_assert_eq!(output.len(), HEADER_BYTES);

    for cluster in &clusters {
        encode_cluster_record(&mut output, cluster);
    }
    for page in &pages {
        encode_page_record(&mut output, page);
    }
    for record in compatibility {
        push_u32(&mut output, record.mesh_index);
        push_u32(&mut output, record.primitive_index);
        push_u32(&mut output, record.reason as u32);
        push_u32(&mut output, record.detail);
    }
    output.resize(payload_offset, 0);
    output.extend_from_slice(&payload);
    debug_assert_eq!(output.len(), file_bytes);

    // Exercise the strict reader before returning an artifact. A writer bug
    // cannot put an unchecked container on disk.
    decode_geometry(&output)?;
    Ok(output)
}

pub fn measure_vertex_error(
    meshlets: &[Meshlet],
    bytes: &[u8],
) -> Result<QuantizationStats, String> {
    let archive = decode_geometry(bytes)?;
    let payload = bytes
        .get(archive.payload_range())
        .ok_or("cooked geometry payload is truncated")?;
    geometry_quantization::measure(
        meshlets,
        &archive.clusters,
        payload,
        archive.vertex_encoding,
    )
}

fn validate_meshlet(meshlet: &Meshlet) -> Result<(), String> {
    if meshlet.vertices.len() < 3 || meshlet.vertices.len() > u8::MAX as usize {
        return Err(format!(
            "meshlet vertex count {} is outside 3..={}",
            meshlet.vertices.len(),
            u8::MAX
        ));
    }
    if meshlet.local_indices.is_empty() || !meshlet.local_indices.len().is_multiple_of(3) {
        return Err("meshlet indices are not a non-empty triangle list".to_string());
    }
    if meshlet
        .local_indices
        .iter()
        .any(|index| *index as usize >= meshlet.vertices.len())
    {
        return Err("meshlet local index exceeds its vertex count".to_string());
    }
    let bounds = &meshlet.bounds;
    let finite = bounds
        .aabb_min
        .iter()
        .chain(bounds.aabb_max.iter())
        .chain(bounds.sphere_center.iter())
        .chain(std::iter::once(&bounds.sphere_radius))
        .chain(bounds.normal_cone_axis.iter())
        .chain(std::iter::once(&bounds.normal_cone_cutoff))
        .chain(std::iter::once(&meshlet.geometric_error))
        .all(|component| component.is_finite());
    if !finite
        || bounds.sphere_radius < 0.0
        || !(-1.0..=1.0).contains(&bounds.normal_cone_cutoff)
        || meshlet.geometric_error < 0.0
    {
        return Err("meshlet bounds/error contain invalid values".to_string());
    }
    Ok(())
}

fn finish_page(
    payload: &[u8],
    page_start: usize,
    first_cluster: usize,
    cluster_end: usize,
    pages: &mut Vec<PageRecord>,
) -> Result<(), String> {
    if cluster_end == first_cluster {
        return Ok(());
    }
    let slice = payload
        .get(page_start..)
        .ok_or("internal page start exceeds payload")?;
    pages.push(PageRecord {
        payload_offset: page_start as u64,
        payload_bytes: checked_u32(slice.len(), "page length")?,
        first_cluster: checked_u32(first_cluster, "first page cluster")?,
        cluster_count: checked_u32(cluster_end - first_cluster, "page cluster count")?,
        sha256: sha256(slice),
    });
    Ok(())
}

fn encode_cluster_record(output: &mut Vec<u8>, record: &ClusterRecord) {
    let start = output.len();
    push_u32(output, record.mesh_index);
    push_u32(output, record.primitive_index);
    push_u32(output, record.material_index.unwrap_or(u32::MAX));
    push_u32(output, record.flags);
    push_u32(output, record.page_index);
    push_u32(output, record.vertex_count);
    push_u32(output, record.triangle_count);
    push_u32(output, record.lod_level);
    push_u64(output, record.vertex_offset);
    push_u64(output, record.index_offset);
    push_f32x3(output, record.aabb_min);
    push_f32x3(output, record.aabb_max);
    push_f32x3(output, record.sphere_center);
    push_f32(output, record.sphere_radius);
    push_f32x3(output, record.normal_cone_axis);
    push_f32(output, record.normal_cone_cutoff);
    push_f32(output, record.geometric_error);
    push_u32(output, record.parent);
    push_u32(output, record.first_child);
    push_u32(output, record.child_count);
    push_u32(output, record.vertex_stride);
    push_u32(output, record.parent_count);
    debug_assert_eq!(output.len() - start, CLUSTER_RECORD_BYTES);
}

fn encode_page_record(output: &mut Vec<u8>, record: &PageRecord) {
    let start = output.len();
    push_u64(output, record.payload_offset);
    push_u32(output, record.payload_bytes);
    push_u32(output, record.first_cluster);
    push_u32(output, record.cluster_count);
    push_u32(output, 0);
    output.extend_from_slice(&record.sha256);
    push_u64(output, 0);
    debug_assert_eq!(output.len() - start, PAGE_RECORD_BYTES);
}

fn checked_table_end(
    start: usize,
    count: usize,
    stride: usize,
    label: &str,
) -> Result<usize, String> {
    count
        .checked_mul(stride)
        .and_then(|bytes| start.checked_add(bytes))
        .ok_or_else(|| format!("{label} range overflow"))
}

fn checked_u32(value: usize, label: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{label} exceeds u32"))
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(output: &mut Vec<u8>, value: f32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_f32x3(output: &mut Vec<u8>, value: [f32; 3]) {
    for component in value {
        push_f32(output, component);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet::{
        build_leaf_meshlets, MeshletLimits, StaticPrimitive, StaticVertex, FLAG_COARSE_ROOT,
    };

    fn triangle(x: f32, primitive_index: u32) -> Meshlet {
        let vertex = |position| StaticVertex {
            position,
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv0: [0.0, 0.0],
            uv1: [0.0, 0.0],
            color: [1.0; 4],
        };
        let primitive = StaticPrimitive {
            mesh_index: 0,
            primitive_index,
            material_index: Some(2),
            double_sided: false,
            alpha_masked: false,
            vertices: vec![
                vertex([x, 0.0, 0.0]),
                vertex([x + 1.0, 0.0, 0.0]),
                vertex([x, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
        };
        build_leaf_meshlets(&primitive, MeshletLimits::default())
            .unwrap()
            .remove(0)
    }

    fn sample_archive(page_budget: u32) -> Vec<u8> {
        let meshlets = vec![triangle(0.0, 0), triangle(2.0, 1)];
        encode_geometry(
            &meshlets,
            &[CompatibilityRecord {
                mesh_index: 1,
                primitive_index: 0,
                reason: CompatibilityReason::Skinned,
                detail: 0,
            }],
            sha256(b"source"),
            page_budget,
        )
        .unwrap()
    }

    #[test]
    fn encoding_is_deterministic_and_round_trips() {
        let a = sample_archive(DEFAULT_PAGE_BYTES);
        let b = sample_archive(DEFAULT_PAGE_BYTES);
        assert_eq!(a, b);
        let decoded = decode_geometry(&a).unwrap();
        assert_eq!(decoded.clusters.len(), 2);
        assert_eq!(decoded.pages.len(), 1);
        assert_eq!(decoded.compatibility.len(), 1);
        assert_eq!(decoded.triangle_count(), 2);
        assert_eq!(decoded.format_version, VERSION);
        assert_eq!(decoded.vertex_encoding, VertexEncoding::Float32);
        assert_eq!(decoded.source_sha256, sha256(b"source"));
        assert_eq!(
            decoded.compatibility[0].reason,
            CompatibilityReason::Skinned
        );
    }

    #[test]
    fn quantized_v2_is_smaller_bounded_and_preserves_missing_tangents() {
        let vertex = |position, uv0, color, tangent| StaticVertex {
            position,
            normal: [0.25, 0.5, 0.829_156_2],
            tangent,
            uv0,
            uv1: [uv0[0] * 0.25, uv0[1] * 0.5],
            color,
        };
        let primitive = StaticPrimitive {
            mesh_index: 0,
            primitive_index: 0,
            material_index: Some(2),
            double_sided: false,
            alpha_masked: false,
            vertices: vec![
                vertex(
                    [0.0, 0.0, 0.0],
                    [0.123_45, -0.75],
                    [0.1, 0.2, 0.3, 1.0],
                    [1.0, 0.25, 0.0, -1.0],
                ),
                vertex(
                    [1.0, 0.0, 0.0],
                    [1.5, 0.333_3],
                    [0.4, 0.5, 0.6, 0.7],
                    [1.0, 0.25, 0.0, 1.0],
                ),
                vertex(
                    [0.333_333, 1.0, 0.0],
                    [-2.25, 0.875],
                    [0.8, 0.9, 1.0, 0.0],
                    [0.0; 4],
                ),
            ],
            indices: vec![0, 1, 2],
        };
        let meshlets = build_leaf_meshlets(&primitive, MeshletLimits::default()).unwrap();
        let float32 =
            encode_geometry(&meshlets, &[], sha256(b"quantized"), DEFAULT_PAGE_BYTES).unwrap();
        let quantized = encode_geometry_with_vertex_encoding(
            &meshlets,
            &[],
            sha256(b"quantized"),
            DEFAULT_PAGE_BYTES,
            VertexEncoding::Quantized,
        )
        .unwrap();
        let float32_archive = decode_geometry(&float32).unwrap();
        let quantized_archive = decode_geometry(&quantized).unwrap();
        assert_eq!(quantized_archive.format_version, QUANTIZED_VERSION);
        assert_eq!(quantized_archive.vertex_encoding, VertexEncoding::Quantized);
        assert_eq!(
            quantized_archive.clusters[0].vertex_stride,
            geometry_quantization::QUANTIZED_VERTEX_BYTES
        );
        assert!(quantized_archive.payload_bytes() < float32_archive.payload_bytes());

        let stats = measure_vertex_error(&meshlets, &quantized).unwrap();
        assert!(stats.max_position_cluster_relative_error <= 1.0 / 65_000.0);
        assert!(stats.max_normal_angular_error_degrees < 0.05);
        assert!(stats.max_tangent_angular_error_degrees < 0.05);
        assert!(stats.max_uv_absolute_error < 0.001);
        assert!(stats.max_color_absolute_error <= 1.0 / 255.0);
        assert!(stats.max_tangent_handedness_error <= 1.0 / 32_767.0);
    }

    #[test]
    fn quantized_v2_rejects_values_that_cannot_be_represented_safely() {
        let mut invalid_color = triangle(0.0, 0);
        invalid_color.vertices[0].color[0] = 1.01;
        assert!(encode_geometry_with_vertex_encoding(
            &[invalid_color],
            &[],
            sha256(b"invalid-color"),
            DEFAULT_PAGE_BYTES,
            VertexEncoding::Quantized,
        )
        .unwrap_err()
        .contains("outside 0..=1"));

        let mut invalid_uv = triangle(0.0, 0);
        invalid_uv.vertices[0].uv0[0] = 70_000.0;
        assert!(encode_geometry_with_vertex_encoding(
            &[invalid_uv],
            &[],
            sha256(b"invalid-uv"),
            DEFAULT_PAGE_BYTES,
            VertexEncoding::Quantized,
        )
        .unwrap_err()
        .contains("finite f16 range"));
    }

    #[test]
    fn compatibility_only_archive_is_valid_and_inspectable() {
        let bytes = encode_geometry(
            &[],
            &[CompatibilityRecord {
                mesh_index: 4,
                primitive_index: 2,
                reason: CompatibilityReason::AlphaBlend,
                detail: 9,
            }],
            sha256(b"transparent"),
            DEFAULT_PAGE_BYTES,
        )
        .unwrap();
        let decoded = decode_geometry(&bytes).unwrap();
        assert!(decoded.clusters.is_empty());
        assert!(decoded.pages.is_empty());
        assert_eq!(decoded.compatibility.len(), 1);
        assert_eq!(
            decoded.compatibility[0].reason,
            CompatibilityReason::AlphaBlend
        );
    }

    #[test]
    fn compatibility_partition_rejects_overlap_and_duplicate_routes() {
        let meshlets = [triangle(0.0, 0)];
        let overlap = encode_geometry(
            &meshlets,
            &[CompatibilityRecord {
                mesh_index: 0,
                primitive_index: 0,
                reason: CompatibilityReason::AlphaBlend,
                detail: 0,
            }],
            sha256(b"overlap"),
            DEFAULT_PAGE_BYTES,
        )
        .unwrap_err();
        assert!(overlap.contains("overlaps eligible mesh 0 primitive 0"));

        let duplicate = encode_geometry(
            &[],
            &[
                CompatibilityRecord {
                    mesh_index: 3,
                    primitive_index: 2,
                    reason: CompatibilityReason::Skinned,
                    detail: 0,
                },
                CompatibilityRecord {
                    mesh_index: 3,
                    primitive_index: 2,
                    reason: CompatibilityReason::MorphTargets,
                    detail: 0,
                },
            ],
            sha256(b"duplicate"),
            DEFAULT_PAGE_BYTES,
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicates mesh 3 primitive 2"));
    }

    #[test]
    fn page_budget_is_hard_and_pages_are_independently_hashed() {
        let meshlets: Vec<_> = (0..20)
            .map(|index| triangle(index as f32 * 2.0, index))
            .collect();
        let bytes = encode_geometry(&meshlets, &[], sha256(b"source"), MIN_PAGE_BYTES).unwrap();
        let decoded = decode_geometry(&bytes).unwrap();
        assert_eq!(decoded.pages.len(), 2);
        assert!(decoded.maximum_page_bytes() <= MIN_PAGE_BYTES);

        let oversized_vertex_count =
            (MIN_PAGE_BYTES as usize / StaticVertex::ENCODED_BYTES as usize) + 1;
        let mut large = triangle(0.0, 0);
        large.vertices = vec![large.vertices[0]; oversized_vertex_count];
        assert!(
            encode_geometry(&[large], &[], sha256(b"source"), MIN_PAGE_BYTES)
                .unwrap_err()
                .contains("exceeds page budget")
        );
    }

    #[test]
    fn payload_corruption_never_reaches_offsets() {
        let mut bytes = sample_archive(DEFAULT_PAGE_BYTES);
        let payload_offset = decode_geometry(&bytes).unwrap().file_payload_offset as usize;
        bytes[payload_offset] ^= 0x80;
        assert!(decode_geometry(&bytes)
            .unwrap_err()
            .contains("payload hash mismatch"));
    }

    #[test]
    fn malformed_header_ranges_are_rejected() {
        let mut bytes = sample_archive(DEFAULT_PAGE_BYTES);
        bytes[104..112].copy_from_slice(&0u64.to_le_bytes());
        assert!(decode_geometry(&bytes)
            .unwrap_err()
            .contains("non-canonical or overlapping"));

        let mut bytes = sample_archive(DEFAULT_PAGE_BYTES);
        bytes[8..12].copy_from_slice(&(QUANTIZED_VERSION + 1).to_le_bytes());
        assert!(decode_geometry(&bytes)
            .unwrap_err()
            .contains("unsupported cooked geometry version"));

        let mut bytes = sample_archive(DEFAULT_PAGE_BYTES);
        bytes[16..20].copy_from_slice(&ENDIAN_TAG.swap_bytes().to_le_bytes());
        assert!(decode_geometry(&bytes).unwrap_err().contains("endian tag"));
    }

    #[test]
    fn malformed_atomic_parent_groups_are_rejected() {
        let mut child = triangle(0.0, 0);
        child.parent = 0;
        child.parent_count = 1;
        let mut parent = child.clone();
        parent.flags |= FLAG_COARSE_ROOT;
        parent.lod_level = 1;
        parent.geometric_error = 0.25;
        parent.parent = NO_RELATION;
        parent.parent_count = 0;
        parent.first_child = 1;
        parent.child_count = 1;
        let mut bytes = encode_geometry(
            &[parent, child],
            &[],
            sha256(b"hierarchy"),
            DEFAULT_PAGE_BYTES,
        )
        .unwrap();

        let child_parent_count = HEADER_BYTES + CLUSTER_RECORD_BYTES + 124;
        bytes[child_parent_count..child_parent_count + 4].copy_from_slice(&0u32.to_le_bytes());
        assert!(decode_geometry(&bytes)
            .unwrap_err()
            .contains("parent range exceeds cluster table"));
    }

    #[test]
    fn pages_may_not_mix_hierarchy_residency_classes() {
        let mut child_a = triangle(0.0, 0);
        let mut child_b = triangle(2.0, 0);
        child_a.parent = 0;
        child_a.parent_count = 1;
        child_b.parent = 0;
        child_b.parent_count = 1;
        let mut parent = child_a.clone();
        parent.flags |= FLAG_COARSE_ROOT;
        parent.lod_level = 1;
        parent.geometric_error = 0.25;
        parent.parent = NO_RELATION;
        parent.parent_count = 0;
        parent.first_child = 1;
        parent.child_count = 2;
        let mut bytes = encode_geometry(
            &[parent, child_a, child_b],
            &[],
            sha256(b"page-classes"),
            DEFAULT_PAGE_BYTES,
        )
        .unwrap();

        let second_child_level = HEADER_BYTES + CLUSTER_RECORD_BYTES * 2 + 28;
        bytes[second_child_level..second_child_level + 4].copy_from_slice(&1u32.to_le_bytes());
        assert!(decode_geometry(&bytes)
            .unwrap_err()
            .contains("mixes streamable hierarchy levels or root residency classes"));
    }

    #[test]
    fn pinned_roots_of_different_lod_levels_share_a_page() {
        let mut root_a = triangle(0.0, 0);
        root_a.flags |= FLAG_COARSE_ROOT;
        root_a.lod_level = 1;
        root_a.geometric_error = 0.25;
        let mut root_b = triangle(2.0, 0);
        root_b.flags |= FLAG_COARSE_ROOT;
        root_b.lod_level = 3;
        root_b.geometric_error = 1.0;

        let bytes = encode_geometry(
            &[root_a, root_b],
            &[],
            sha256(b"mixed-level-roots"),
            DEFAULT_PAGE_BYTES,
        )
        .unwrap();
        let archive = decode_geometry(&bytes).unwrap();
        assert_eq!(archive.pages.len(), 1);
        assert_eq!(archive.coarse_root_page_count(), 1);
    }
}
