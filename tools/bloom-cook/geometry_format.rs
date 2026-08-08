//! Versioned, little-endian cooked geometry container.
//!
//! Version 1 stores deterministic meshlets in independently hashed,
//! budget-bounded pages. Its fixed cluster record supports either the default
//! leaf-only artifact or opt-in atomic parent/child replacement groups. All
//! offsets and hierarchy relations are validated before payload access.

use crate::geometry_quantization::{self, QuantizationStats, VertexEncoding};
use crate::meshlet::{
    Meshlet, FLAG_ALPHA_MASKED, FLAG_COARSE_ROOT, FLAG_DOUBLE_SIDED, NO_RELATION,
};
use sha2::{Digest, Sha256};

pub const MAGIC: [u8; 8] = *b"BLMGEO1\0";
pub const VERSION: u32 = 1;
pub const QUANTIZED_VERSION: u32 = 2;
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
    pub lod_level: u32,
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
    pub parent_count: u32,
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
    pub format_version: u32,
    pub vertex_encoding: VertexEncoding,
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

    pub fn coarse_root_page_count(&self) -> usize {
        self.pages
            .iter()
            .take_while(|page| {
                self.clusters[page.first_cluster as usize].flags & FLAG_COARSE_ROOT != 0
            })
            .count()
    }

    pub fn coarse_root_page_bytes(&self) -> u64 {
        self.pages
            .iter()
            .take(self.coarse_root_page_count())
            .map(|page| page.payload_bytes as u64)
            .sum()
    }
}

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
        let page_class = (meshlet.lod_level, meshlet.flags & FLAG_COARSE_ROOT != 0);
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
    validate_clusters(&clusters, &pages, payload, vertex_encoding)?;
    Ok(GeometryArchive {
        format_version: version,
        vertex_encoding,
        source_sha256,
        payload_sha256,
        page_budget_bytes,
        clusters,
        pages,
        compatibility,
    })
}

pub fn measure_vertex_error(
    meshlets: &[Meshlet],
    bytes: &[u8],
) -> Result<QuantizationStats, String> {
    let archive = decode_geometry(bytes)?;
    let payload_offset = read_usize(bytes, 128, "payload offset")?;
    let payload = bytes
        .get(payload_offset..)
        .ok_or("cooked geometry payload is truncated")?;
    geometry_quantization::measure(
        meshlets,
        &archive.clusters,
        payload,
        archive.vertex_encoding,
    )
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
    let mut reached_non_root_pages = false;
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
        let cluster_start = page.first_cluster as usize;
        let cluster_end = cluster_start + page.cluster_count as usize;
        let page_clusters = clusters
            .get(cluster_start..cluster_end)
            .ok_or_else(|| format!("page {page_index} cluster range exceeds cluster table"))?;
        let first_class = (
            page_clusters[0].lod_level,
            page_clusters[0].flags & FLAG_COARSE_ROOT != 0,
        );
        if page_clusters.iter().any(|cluster| {
            (cluster.lod_level, cluster.flags & FLAG_COARSE_ROOT != 0) != first_class
        }) {
            return Err(format!(
                "page {page_index} mixes hierarchy levels or root residency classes"
            ));
        }
        if first_class.1 {
            if reached_non_root_pages {
                return Err(format!(
                    "page {page_index} places coarse roots after streamable pages"
                ));
            }
        } else {
            reached_non_root_pages = true;
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
    vertex_encoding: VertexEncoding,
) -> Result<(), String> {
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let known_flags = FLAG_DOUBLE_SIDED | FLAG_ALPHA_MASKED | FLAG_COARSE_ROOT;
        if !(3..=u8::MAX as u32).contains(&cluster.vertex_count)
            || cluster.triangle_count == 0
            || cluster.flags & !known_flags != 0
        {
            return Err(format!(
                "cluster {cluster_index} has invalid counts, stride, or flags"
            ));
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
        let index_start = cluster.index_offset as usize;
        let index_end = index_end as usize;
        geometry_quantization::validate_cluster_vertices(
            cluster_index,
            cluster,
            payload,
            vertex_encoding,
        )?;
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
        if cluster.parent == NO_RELATION {
            if cluster.parent_count != 0 {
                return Err(format!(
                    "cluster {cluster_index} has no parent but a non-zero parent count"
                ));
            }
        } else {
            let parent_end = (cluster.parent as usize)
                .checked_add(cluster.parent_count as usize)
                .ok_or_else(|| format!("cluster {cluster_index} parent range overflow"))?;
            if cluster.parent_count == 0 || parent_end > clusters.len() {
                return Err(format!(
                    "cluster {cluster_index} parent range exceeds cluster table"
                ));
            }
        }
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
    validate_hierarchy(clusters)?;
    Ok(())
}

fn validate_hierarchy(clusters: &[ClusterRecord]) -> Result<(), String> {
    let hierarchy_present = clusters.iter().any(|cluster| {
        cluster.parent_count != 0
            || cluster.child_count != 0
            || cluster.lod_level != 0
            || cluster.flags & FLAG_COARSE_ROOT != 0
    });
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        if cluster.parent == NO_RELATION {
            if hierarchy_present && cluster.flags & FLAG_COARSE_ROOT == 0 {
                return Err(format!(
                    "hierarchy cluster {cluster_index} has no parent and is not a coarse root"
                ));
            }
            continue;
        }
        if cluster.flags & FLAG_COARSE_ROOT != 0 {
            return Err(format!(
                "cluster {cluster_index} is both a hierarchy child and coarse root"
            ));
        }
        let parent_start = cluster.parent as usize;
        let parent_end = parent_start + cluster.parent_count as usize;
        let first_parent = &clusters[parent_start];
        for (parent_index, parent) in clusters[parent_start..parent_end].iter().enumerate() {
            let parent_index = parent_start + parent_index;
            let child_start = parent.first_child as usize;
            let child_end = child_start
                .checked_add(parent.child_count as usize)
                .ok_or_else(|| format!("cluster {parent_index} child range overflow"))?;
            if !(child_start..child_end).contains(&cluster_index)
                || parent.lod_level <= cluster.lod_level
                || parent.first_child != first_parent.first_child
                || parent.child_count != first_parent.child_count
                || parent.lod_level != first_parent.lod_level
                || parent.geometric_error != first_parent.geometric_error
            {
                return Err(format!(
                    "cluster {cluster_index} has a non-reciprocal or inconsistent parent group"
                ));
            }
        }
    }
    for (parent_index, parent) in clusters.iter().enumerate() {
        if parent.child_count == 0 {
            continue;
        }
        let child_start = parent.first_child as usize;
        let child_end = child_start + parent.child_count as usize;
        let first_child = &clusters[child_start];
        let parent_start = first_child.parent as usize;
        let parent_end = parent_start
            .checked_add(first_child.parent_count as usize)
            .ok_or_else(|| format!("cluster {parent_index} sibling range overflow"))?;
        if first_child.parent == NO_RELATION
            || first_child.parent_count == 0
            || !(parent_start..parent_end).contains(&parent_index)
        {
            return Err(format!(
                "parent {parent_index} is outside its reciprocal sibling group"
            ));
        }
        for (child_index, child) in clusters[child_start..child_end].iter().enumerate() {
            let child_index = child_start + child_index;
            if child.parent != first_child.parent
                || child.parent_count != first_child.parent_count
                || child.lod_level >= parent.lod_level
                || child.mesh_index != parent.mesh_index
                || child.primitive_index != parent.primitive_index
                || child.material_index != parent.material_index
                || (child.flags & !FLAG_COARSE_ROOT) != (parent.flags & !FLAG_COARSE_ROOT)
                || parent.geometric_error < child.geometric_error
            {
                return Err(format!(
                    "parent {parent_index} and child {child_index} violate hierarchy identity/error"
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
            .contains("mixes hierarchy levels or root residency classes"));
    }
}
