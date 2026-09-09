use crate::hash::{hex_hash, sha256};
use crate::types::*;
use crate::validate::{
    validate_clusters, validate_compatibility_partition, validate_page_budget, validate_pages,
};
use crate::wire::{
    align_up, checked_table_end, read_f32, read_f32x3, read_hash, read_u32, read_u64, read_usize,
};

pub fn decode_geometry(bytes: &[u8]) -> Result<GeometryArchive, String> {
    if bytes.len() < HEADER_BYTES {
        return Err(format!(
            "geometry header truncated: {} bytes, expected at least {HEADER_BYTES}",
            bytes.len()
        ));
    }
    if bytes[..8] != MAGIC {
        return Err("invalid cooked geometry magic".to_string());
    }
    let version = read_u32(bytes, 8, "version")?;
    let vertex_encoding = match version {
        VERSION => VertexEncoding::Float32,
        QUANTIZED_VERSION => VertexEncoding::Quantized,
        _ => {
            return Err(format!(
                "unsupported cooked geometry version {version}; expected {VERSION} or \
                 {QUANTIZED_VERSION}"
            ))
        }
    };
    let header_bytes = read_u32(bytes, 12, "header size")? as usize;
    if header_bytes != HEADER_BYTES {
        return Err(format!(
            "invalid cooked geometry header size {header_bytes}; expected {HEADER_BYTES}"
        ));
    }
    let endian_tag = read_u32(bytes, 16, "endian tag")?;
    if endian_tag != ENDIAN_TAG {
        return Err(format!(
            "unsupported cooked geometry endian tag 0x{endian_tag:08x}"
        ));
    }

    let source_sha256 = read_hash(bytes, 24, "source hash")?;
    let expected_payload_sha256 = read_hash(bytes, 56, "payload hash")?;
    let cluster_count = read_u32(bytes, 88, "cluster count")? as usize;
    let page_count = read_u32(bytes, 92, "page count")? as usize;
    let compatibility_count = read_u32(bytes, 96, "compatibility count")? as usize;
    let cluster_table_offset = read_usize(bytes, 104, "cluster table offset")?;
    let page_table_offset = read_usize(bytes, 112, "page table offset")?;
    let compatibility_table_offset = read_usize(bytes, 120, "compatibility table offset")?;
    let payload_offset = read_usize(bytes, 128, "payload offset")?;
    let payload_bytes = read_usize(bytes, 136, "payload length")?;
    let declared_file_bytes = read_usize(bytes, 144, "file length")?;
    let page_budget_bytes = read_u32(bytes, 152, "page budget")?;
    validate_page_budget(page_budget_bytes)?;

    if declared_file_bytes != bytes.len() {
        return Err(format!(
            "cooked geometry length mismatch: header {declared_file_bytes}, actual {}",
            bytes.len()
        ));
    }
    let expected_page_offset = checked_table_end(
        HEADER_BYTES,
        cluster_count,
        CLUSTER_RECORD_BYTES,
        "cluster table",
    )?;
    let expected_compatibility_offset = checked_table_end(
        expected_page_offset,
        page_count,
        PAGE_RECORD_BYTES,
        "page table",
    )?;
    let compatibility_end = checked_table_end(
        expected_compatibility_offset,
        compatibility_count,
        COMPATIBILITY_RECORD_BYTES,
        "compatibility table",
    )?;
    let expected_payload_offset = align_up(compatibility_end, 16);
    if cluster_table_offset != HEADER_BYTES
        || page_table_offset != expected_page_offset
        || compatibility_table_offset != expected_compatibility_offset
        || payload_offset != expected_payload_offset
    {
        return Err("cooked geometry table offsets are non-canonical or overlapping".to_string());
    }
    let payload_end = payload_offset
        .checked_add(payload_bytes)
        .ok_or("payload range overflow")?;
    if payload_end != bytes.len() {
        return Err(format!(
            "payload range ends at {payload_end}, file length is {}",
            bytes.len()
        ));
    }
    let payload = &bytes[payload_offset..payload_end];
    let payload_sha256 = sha256(payload);
    if payload_sha256 != expected_payload_sha256 {
        return Err(format!(
            "cooked geometry payload hash mismatch: expected {}, actual {}",
            hex_hash(expected_payload_sha256),
            hex_hash(payload_sha256)
        ));
    }

    let mut clusters = Vec::with_capacity(cluster_count);
    for index in 0..cluster_count {
        clusters.push(decode_cluster_record(
            bytes,
            cluster_table_offset + index * CLUSTER_RECORD_BYTES,
        )?);
    }
    let mut pages = Vec::with_capacity(page_count);
    for index in 0..page_count {
        pages.push(decode_page_record(
            bytes,
            page_table_offset + index * PAGE_RECORD_BYTES,
        )?);
    }
    let mut compatibility = Vec::with_capacity(compatibility_count);
    for index in 0..compatibility_count {
        let offset = compatibility_table_offset + index * COMPATIBILITY_RECORD_BYTES;
        compatibility.push(CompatibilityRecord {
            mesh_index: read_u32(bytes, offset, "compatibility mesh index")?,
            primitive_index: read_u32(bytes, offset + 4, "compatibility primitive index")?,
            reason: CompatibilityReason::from_code(read_u32(
                bytes,
                offset + 8,
                "compatibility reason",
            )?)?,
            detail: read_u32(bytes, offset + 12, "compatibility detail")?,
        });
    }

    validate_pages(&pages, &clusters, payload, page_budget_bytes)?;
    validate_clusters(&clusters, &pages, payload, vertex_encoding)?;
    validate_compatibility_partition(&clusters, &compatibility)?;
    Ok(GeometryArchive {
        format_version: version,
        vertex_encoding,
        source_sha256,
        payload_sha256,
        page_budget_bytes,
        file_payload_offset: payload_offset as u64,
        clusters,
        pages,
        compatibility,
    })
}

fn decode_cluster_record(bytes: &[u8], offset: usize) -> Result<ClusterRecord, String> {
    let material_index = match read_u32(bytes, offset + 8, "cluster material")? {
        u32::MAX => None,
        index => Some(index),
    };
    Ok(ClusterRecord {
        mesh_index: read_u32(bytes, offset, "cluster mesh index")?,
        primitive_index: read_u32(bytes, offset + 4, "cluster primitive index")?,
        material_index,
        flags: read_u32(bytes, offset + 12, "cluster flags")?,
        page_index: read_u32(bytes, offset + 16, "cluster page")?,
        vertex_count: read_u32(bytes, offset + 20, "cluster vertex count")?,
        triangle_count: read_u32(bytes, offset + 24, "cluster triangle count")?,
        lod_level: read_u32(bytes, offset + 28, "cluster LOD level")?,
        vertex_offset: read_u64(bytes, offset + 32, "cluster vertex offset")?,
        index_offset: read_u64(bytes, offset + 40, "cluster index offset")?,
        aabb_min: read_f32x3(bytes, offset + 48, "cluster aabb min")?,
        aabb_max: read_f32x3(bytes, offset + 60, "cluster aabb max")?,
        sphere_center: read_f32x3(bytes, offset + 72, "cluster sphere center")?,
        sphere_radius: read_f32(bytes, offset + 84, "cluster sphere radius")?,
        normal_cone_axis: read_f32x3(bytes, offset + 88, "cluster normal cone")?,
        normal_cone_cutoff: read_f32(bytes, offset + 100, "cluster normal cutoff")?,
        geometric_error: read_f32(bytes, offset + 104, "cluster geometric error")?,
        parent: read_u32(bytes, offset + 108, "cluster parent")?,
        first_child: read_u32(bytes, offset + 112, "cluster first child")?,
        child_count: read_u32(bytes, offset + 116, "cluster child count")?,
        vertex_stride: read_u32(bytes, offset + 120, "cluster vertex stride")?,
        parent_count: read_u32(bytes, offset + 124, "cluster parent count")?,
    })
}

fn decode_page_record(bytes: &[u8], offset: usize) -> Result<PageRecord, String> {
    Ok(PageRecord {
        payload_offset: read_u64(bytes, offset, "page payload offset")?,
        payload_bytes: read_u32(bytes, offset + 8, "page payload length")?,
        first_cluster: read_u32(bytes, offset + 12, "page first cluster")?,
        cluster_count: read_u32(bytes, offset + 16, "page cluster count")?,
        sha256: read_hash(bytes, offset + 24, "page hash")?,
    })
}
