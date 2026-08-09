use std::ops::Range;

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
pub const FLOAT32_VERTEX_BYTES: u32 = 72;
pub const QUANTIZED_VERTEX_BYTES: u32 = 32;
pub(crate) const QUANTIZED_TANGENT_VALID: u16 = 1;

pub const NO_RELATION: u32 = u32::MAX;
pub const FLAG_DOUBLE_SIDED: u32 = 1 << 0;
pub const FLAG_ALPHA_MASKED: u32 = 1 << 1;
pub const FLAG_COARSE_ROOT: u32 = 1 << 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VertexEncoding {
    Float32,
    Quantized,
}

impl VertexEncoding {
    pub const fn stride(self) -> u32 {
        match self {
            Self::Float32 => FLOAT32_VERTEX_BYTES,
            Self::Quantized => QUANTIZED_VERTEX_BYTES,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Quantized => "quantized32",
        }
    }
}

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

    pub(crate) fn from_code(code: u32) -> Result<Self, String> {
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
    /// Byte offset relative to the archive payload, not the file start.
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
    /// Validated file-relative start of the page payload region.
    pub file_payload_offset: u64,
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

    pub fn payload_range(&self) -> Range<usize> {
        let start = self.file_payload_offset as usize;
        start..start + self.payload_bytes() as usize
    }

    pub fn page_file_range(&self, page_index: usize) -> Option<Range<usize>> {
        let page = self.pages.get(page_index)?;
        let start = self.file_payload_offset.checked_add(page.payload_offset)? as usize;
        Some(start..start.checked_add(page.payload_bytes as usize)?)
    }
}
