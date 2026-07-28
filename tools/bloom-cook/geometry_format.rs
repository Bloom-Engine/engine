//! Versioned, little-endian cooked geometry container.
//!
//! Version 1 stores deterministic leaf meshlets in independently hashed,
//! budget-bounded pages. Hierarchy relation/error fields are present from the
//! first version so adding parent clusters does not require changing the
//! container layout. All offsets are validated before payload access.

use crate::meshlet::{Meshlet, StaticVertex, NO_RELATION};
use sha2::{Digest, Sha256};

pub const MAGIC: [u8; 8] = *b"BLMGEO1\0";
pub const VERSION: u32 = 1;
pub const ENDIAN_TAG: u32 = 0x0102_0304;
pub const HEADER_BYTES: usize = 160;
pub const CLUSTER_RECORD_BYTES: usize = 128;
pub const PAGE_RECORD_BYTES: usize = 64;
pub const COMPATIBILITY_RECORD_BYTES: usize = 16;
pub const DEFAULT_PAGE_BYTES: u32 = 64 * 1024;
pub const MIN_PAGE_BYTES: u32 = 4 * 1024;
pub const MAX_PAGE_BYTES: u32 = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CompatibilityReason {
    NonTriangleTopology = 1,
    Skinned = 2,
    MorphTargets = 3,
    AlphaBlend = 4,
}

impl CompatibilityReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::NonTriangleTopology => "non-triangle-topology",
            Self::Skinned => "skinned",
            Self::MorphTargets => "morph-targets",
            Self::AlphaBlend => "alpha-blend",
        }
    }

    fn from_code(code: u32) -> Result<Self, String> {
        match code {
            1 => Ok(Self::NonTriangleTopology),
            2 => Ok(Self::Skinned),
            3 => Ok(Self::MorphTargets),
            4 => Ok(Self::AlphaBlend),
            _ => Err(format!("unknown compatibility reason code {code}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompatibilityRecord {
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub reason: CompatibilityReason,
    /// Reason-specific detail. Alpha blend records store the material index or
    /// `u32::MAX`; topology records store the glTF mode discriminant.
    pub detail: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterRecord {
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub material_index: Option<u32>,
    pub flags: u32,
    pub page_index: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub vertex_offset: u64,
    pub index_offset: u64,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub sphere_center: [f32; 3],
    pub sphere_radius: f32,
    pub normal_cone_axis: [f32; 3],
    pub normal_cone_cutoff: f32,
    pub geometric_error: f32,
    pub parent: u32,
    pub first_child: u32,
    pub child_count: u32,
    pub vertex_stride: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PageRecord {
    pub payload_offset: u64,
    pub payload_bytes: u32,
    pub first_cluster: u32,
    pub cluster_count: u32,
    pub sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometryArchive {
    pub source_sha256: [u8; 32],
    pub payload_sha256: [u8; 32],
    pub page_budget_bytes: u32,
    pub clusters: Vec<ClusterRecord>,
    pub pages: Vec<PageRecord>,
    pub compatibility: Vec<CompatibilityRecord>,
}

impl GeometryArchive {
    pub fn payload_bytes(&self) -> u64 {
        self.pages
            .iter()
            .map(|page| page.payload_bytes as u64)
            .sum()
    }

    pub fn triangle_count(&self) -> u64 {
        self.clusters
            .iter()
            .map(|cluster| cluster.triangle_count as u64)
            .sum()
    }

    pub fn maximum_page_bytes(&self) -> u32 {
        self.pages
            .iter()
            .map(|page| page.payload_bytes)
            .max()
            .unwrap_or(0)
    }
}

pub fn encode_geometry(
    meshlets: &[Meshlet],
    compatibility: &[CompatibilityRecord],
    source_sha256: [u8; 32],
    page_budget_bytes: u32,
) -> Result<Vec<u8>, String> {
    validate_page_budget(page_budget_bytes)?;

    let mut payload = Vec::new();
    let mut clusters = Vec::with_capacity(meshlets.len());
    let mut pages = Vec::<PageRecord>::new();
    let mut page_start = 0usize;
    let mut page_first_cluster = 0usize;

    for meshlet in meshlets {
        validate_meshlet(meshlet)?;
        let encoded_bytes = align_up(meshlet.encoded_payload_bytes(), 16);
        if encoded_bytes > page_budget_bytes as usize {
            return Err(format!(
                "meshlet payload {encoded_bytes} bytes exceeds page budget {page_budget_bytes}"
            ));
        }
        let current_page_bytes = payload.len() - page_start;
        if current_page_bytes > 0
            && current_page_bytes
                .checked_add(encoded_bytes)
                .is_none_or(|bytes| bytes > page_budget_bytes as usize)
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

        let page_index = pages.len() as u32;
        let vertex_offset = payload.len() as u64;
        for vertex in &meshlet.vertices {
            encode_vertex(&mut payload, vertex);
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
            first_child: meshlet.first_child,
            child_count: meshlet.child_count,
            vertex_stride: StaticVertex::ENCODED_BYTES,
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
    push_u32(&mut output, VERSION);
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
    if version != VERSION {
        return Err(format!(
            "unsupported cooked geometry version {version}; expected {VERSION}"
        ));
    }
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
        let offset = cluster_table_offset + index * CLUSTER_RECORD_BYTES;
        clusters.push(decode_cluster_record(bytes, offset)?);
    }
    let mut pages = Vec::with_capacity(page_count);
    for index in 0..page_count {
        let offset = page_table_offset + index * PAGE_RECORD_BYTES;
        pages.push(decode_page_record(bytes, offset)?);
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
    validate_clusters(&clusters, &pages, payload)?;
    Ok(GeometryArchive {
        source_sha256,
        payload_sha256,
        page_budget_bytes,
        clusters,
        pages,
        compatibility,
    })
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn hex_hash(hash: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn validate_page_budget(page_budget_bytes: u32) -> Result<(), String> {
    if !(MIN_PAGE_BYTES..=MAX_PAGE_BYTES).contains(&page_budget_bytes)
        || !page_budget_bytes.is_power_of_two()
    {
        return Err(format!(
            "geometry page budget must be a power of two in \
             {MIN_PAGE_BYTES}..={MAX_PAGE_BYTES}, got {page_budget_bytes}"
        ));
    }
    Ok(())
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

fn encode_vertex(output: &mut Vec<u8>, vertex: &StaticVertex) {
    for value in vertex
        .position
        .iter()
        .chain(vertex.normal.iter())
        .chain(vertex.tangent.iter())
        .chain(vertex.uv0.iter())
        .chain(vertex.uv1.iter())
        .chain(vertex.color.iter())
    {
        push_f32(output, *value);
    }
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
    push_u32(output, 0);
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
    push_u32(output, 0);
    debug_assert_eq!(output.len() - start, CLUSTER_RECORD_BYTES);
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
    })
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

fn decode_page_record(bytes: &[u8], offset: usize) -> Result<PageRecord, String> {
    Ok(PageRecord {
        payload_offset: read_u64(bytes, offset, "page payload offset")?,
        payload_bytes: read_u32(bytes, offset + 8, "page payload length")?,
        first_cluster: read_u32(bytes, offset + 12, "page first cluster")?,
        cluster_count: read_u32(bytes, offset + 16, "page cluster count")?,
        sha256: read_hash(bytes, offset + 24, "page hash")?,
    })
}

fn validate_pages(
    pages: &[PageRecord],
    clusters: &[ClusterRecord],
    payload: &[u8],
    page_budget_bytes: u32,
) -> Result<(), String> {
    if pages.is_empty() {
        return if clusters.is_empty() && payload.is_empty() {
            Ok(())
        } else {
            Err("non-empty cooked geometry contains no pages".to_string())
        };
    }
    let mut expected_payload_offset = 0u64;
    let mut expected_cluster = 0u32;
    for (page_index, page) in pages.iter().enumerate() {
        if page.payload_offset != expected_payload_offset
            || page.first_cluster != expected_cluster
            || page.cluster_count == 0
        {
            return Err(format!(
                "page {page_index} has a gap, overlap, or empty cluster range"
            ));
        }
        if page.payload_bytes == 0 || page.payload_bytes > page_budget_bytes {
            return Err(format!(
                "page {page_index} length {} violates budget {page_budget_bytes}",
                page.payload_bytes
            ));
        }
        let start = usize::try_from(page.payload_offset)
            .map_err(|_| format!("page {page_index} offset exceeds host address space"))?;
        let end = start
            .checked_add(page.payload_bytes as usize)
            .ok_or_else(|| format!("page {page_index} range overflow"))?;
        let page_payload = payload
            .get(start..end)
            .ok_or_else(|| format!("page {page_index} exceeds payload"))?;
        let actual_hash = sha256(page_payload);
        if actual_hash != page.sha256 {
            return Err(format!(
                "page {page_index} hash mismatch: expected {}, actual {}",
                hex_hash(page.sha256),
                hex_hash(actual_hash)
            ));
        }
        expected_payload_offset = end as u64;
        expected_cluster = expected_cluster
            .checked_add(page.cluster_count)
            .ok_or("page cluster range overflow")?;
    }
    if expected_payload_offset != payload.len() as u64 {
        return Err("page ranges do not cover the complete payload".to_string());
    }
    if expected_cluster as usize != clusters.len() {
        return Err("page cluster ranges do not cover the cluster table".to_string());
    }
    Ok(())
}

fn validate_clusters(
    clusters: &[ClusterRecord],
    pages: &[PageRecord],
    payload: &[u8],
) -> Result<(), String> {
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        if cluster.vertex_stride != StaticVertex::ENCODED_BYTES
            || !(3..=u8::MAX as u32).contains(&cluster.vertex_count)
            || cluster.triangle_count == 0
        {
            return Err(format!("cluster {cluster_index} has invalid counts/stride"));
        }
        let page = pages
            .get(cluster.page_index as usize)
            .ok_or_else(|| format!("cluster {cluster_index} references missing page"))?;
        let first = page.first_cluster as usize;
        let end = first + page.cluster_count as usize;
        if cluster_index < first || cluster_index >= end {
            return Err(format!(
                "cluster {cluster_index} is outside its page cluster range"
            ));
        }
        let page_start = page.payload_offset;
        let page_end = page_start + page.payload_bytes as u64;
        let vertex_bytes = (cluster.vertex_count as u64)
            .checked_mul(cluster.vertex_stride as u64)
            .ok_or_else(|| format!("cluster {cluster_index} vertex range overflow"))?;
        let index_bytes = (cluster.triangle_count as u64)
            .checked_mul(3)
            .ok_or_else(|| format!("cluster {cluster_index} index range overflow"))?;
        let vertex_end = cluster
            .vertex_offset
            .checked_add(vertex_bytes)
            .ok_or_else(|| format!("cluster {cluster_index} vertex range overflow"))?;
        let index_end = cluster
            .index_offset
            .checked_add(index_bytes)
            .ok_or_else(|| format!("cluster {cluster_index} index range overflow"))?;
        if cluster.vertex_offset < page_start
            || cluster.vertex_offset % 16 != 0
            || cluster.index_offset != vertex_end
            || index_end > page_end
        {
            return Err(format!(
                "cluster {cluster_index} payload offsets exceed or overlap its page"
            ));
        }
        let vertex_start = cluster.vertex_offset as usize;
        let vertex_end = vertex_end as usize;
        let index_start = cluster.index_offset as usize;
        let index_end = index_end as usize;
        for component in payload[vertex_start..vertex_end].chunks_exact(4) {
            if !f32::from_le_bytes(component.try_into().unwrap()).is_finite() {
                return Err(format!(
                    "cluster {cluster_index} vertex payload contains NaN/Inf"
                ));
            }
        }
        if payload[index_start..index_end]
            .iter()
            .any(|index| *index as u32 >= cluster.vertex_count)
        {
            return Err(format!(
                "cluster {cluster_index} local index exceeds vertex count"
            ));
        }
        let finite_bounds = cluster
            .aabb_min
            .iter()
            .chain(cluster.aabb_max.iter())
            .chain(cluster.sphere_center.iter())
            .chain(std::iter::once(&cluster.sphere_radius))
            .chain(cluster.normal_cone_axis.iter())
            .chain(std::iter::once(&cluster.normal_cone_cutoff))
            .chain(std::iter::once(&cluster.geometric_error))
            .all(|value| value.is_finite());
        if !finite_bounds
            || cluster
                .aabb_min
                .iter()
                .zip(cluster.aabb_max)
                .any(|(min, max)| *min > max)
            || cluster.sphere_radius < 0.0
            || !(-1.0..=1.0).contains(&cluster.normal_cone_cutoff)
            || cluster.geometric_error < 0.0
        {
            return Err(format!("cluster {cluster_index} has invalid bounds/error"));
        }
        validate_relation(cluster_index, "parent", cluster.parent, clusters.len())?;
        if cluster.child_count == 0 {
            if cluster.first_child != NO_RELATION {
                return Err(format!(
                    "cluster {cluster_index} has no children but a first-child index"
                ));
            }
        } else {
            let first = cluster.first_child as usize;
            let end = first
                .checked_add(cluster.child_count as usize)
                .ok_or_else(|| format!("cluster {cluster_index} child range overflow"))?;
            if first >= clusters.len() || end > clusters.len() {
                return Err(format!(
                    "cluster {cluster_index} child range exceeds cluster table"
                ));
            }
        }
    }
    Ok(())
}

fn validate_relation(
    cluster_index: usize,
    label: &str,
    relation: u32,
    cluster_count: usize,
) -> Result<(), String> {
    if relation != NO_RELATION && relation as usize >= cluster_count {
        return Err(format!(
            "cluster {cluster_index} {label} index {relation} exceeds cluster table"
        ));
    }
    Ok(())
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

fn read_u32(bytes: &[u8], offset: usize, label: &str) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| format!("{label} is truncated"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize, label: &str) -> Result<u64, String> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| format!("{label} is truncated"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn read_usize(bytes: &[u8], offset: usize, label: &str) -> Result<usize, String> {
    let value = read_u64(bytes, offset, label)?;
    usize::try_from(value).map_err(|_| format!("{label} exceeds host address space"))
}

fn read_f32(bytes: &[u8], offset: usize, label: &str) -> Result<f32, String> {
    Ok(f32::from_bits(read_u32(bytes, offset, label)?))
}

fn read_f32x3(bytes: &[u8], offset: usize, label: &str) -> Result<[f32; 3], String> {
    Ok([
        read_f32(bytes, offset, label)?,
        read_f32(bytes, offset + 4, label)?,
        read_f32(bytes, offset + 8, label)?,
    ])
}

fn read_hash(bytes: &[u8], offset: usize, label: &str) -> Result<[u8; 32], String> {
    bytes
        .get(offset..offset + 32)
        .ok_or_else(|| format!("{label} is truncated"))?
        .try_into()
        .map_err(|_| format!("{label} has invalid length"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet::{build_leaf_meshlets, MeshletLimits, StaticPrimitive, StaticVertex};

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
        assert_eq!(decoded.source_sha256, sha256(b"source"));
        assert_eq!(
            decoded.compatibility[0].reason,
            CompatibilityReason::Skinned
        );
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
        let payload_offset = read_usize(&bytes, 128, "payload").unwrap();
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
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(decode_geometry(&bytes)
            .unwrap_err()
            .contains("unsupported cooked geometry version"));

        let mut bytes = sample_archive(DEFAULT_PAGE_BYTES);
        bytes[16..20].copy_from_slice(&ENDIAN_TAG.swap_bytes().to_le_bytes());
        assert!(decode_geometry(&bytes).unwrap_err().contains("endian tag"));
    }
}
