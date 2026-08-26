use bloom_geometry_format::{
    decode_geometry, hex_hash, sha256, CompatibilityRecord, GeometryArchive, FLAG_ALPHA_MASKED,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

/// Identity supplied by the validated #136 asset index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactIdentity {
    pub bytes: u64,
    pub format_version: u32,
    pub file_sha256: [u8; 32],
    pub payload_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
}

/// An immutable cooked archive whose complete wire contract was validated.
#[derive(Clone, Debug)]
pub struct VirtualGeometryAsset {
    bytes: Arc<[u8]>,
    archive: Arc<GeometryArchive>,
}

/// One source glTF mesh's explicit split between virtual and compatibility
/// primitives. A production model loader uses this table to create filtered
/// virtual instances and retain ordinary draws for every listed fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometrySourceMeshRoute {
    pub source_mesh_index: u32,
    pub virtual_primitive_count: u32,
    pub compatibility: Vec<CompatibilityRecord>,
    /// MASK primitives are present in older cooked archives as clusters, but
    /// remain compatibility-owned until virtual visibility can evaluate the
    /// exact texture/sampler/cutoff contract.
    pub alpha_masked_compatibility: Vec<VirtualGeometryAlphaMaskedRoute>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualGeometryAlphaMaskedRoute {
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub material_index: Option<u32>,
}

impl VirtualGeometryAsset {
    /// Loads a directly supplied archive. The embedded payload and page hashes
    /// are verified before this returns.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, VirtualGeometryLoadError> {
        let bytes = bytes.into();
        let archive = decode_geometry(&bytes).map_err(VirtualGeometryLoadError::Format)?;
        Ok(Self {
            bytes,
            archive: Arc::new(archive),
        })
    }

    /// Loads an index-selected archive and verifies every provenance field,
    /// including the complete-file hash that is intentionally outside `.bgeo`.
    pub fn from_indexed_bytes(
        bytes: impl Into<Arc<[u8]>>,
        identity: ArtifactIdentity,
    ) -> Result<Self, VirtualGeometryLoadError> {
        let bytes = bytes.into();
        if bytes.len() as u64 != identity.bytes {
            return Err(VirtualGeometryLoadError::Identity(format!(
                "artifact length mismatch: index {}, actual {}",
                identity.bytes,
                bytes.len()
            )));
        }
        let actual_file_sha256 = sha256(&bytes);
        if actual_file_sha256 != identity.file_sha256 {
            return Err(VirtualGeometryLoadError::Identity(format!(
                "artifact hash mismatch: index {}, actual {}",
                hex_hash(identity.file_sha256),
                hex_hash(actual_file_sha256)
            )));
        }
        let asset = Self::from_bytes(bytes)?;
        let archive = asset.archive();
        if archive.format_version != identity.format_version
            || archive.payload_sha256 != identity.payload_sha256
            || archive.source_sha256 != identity.source_sha256
        {
            return Err(VirtualGeometryLoadError::Identity(
                "artifact format, payload hash, or source hash disagrees with the asset index"
                    .to_string(),
            ));
        }
        Ok(asset)
    }

    pub fn archive(&self) -> &GeometryArchive {
        &self.archive
    }

    pub fn archive_arc(&self) -> Arc<GeometryArchive> {
        Arc::clone(&self.archive)
    }

    pub fn file_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn page_file_range(&self, page_index: usize) -> Option<Range<usize>> {
        self.archive.page_file_range(page_index)
    }

    pub fn page_bytes(&self, page_index: usize) -> Option<&[u8]> {
        self.page_file_range(page_index)
            .and_then(|range| self.bytes.get(range))
    }

    /// Return a canonical source-mesh routing table. Eligible primitive
    /// identity is deduplicated across hierarchy clusters; compatibility
    /// records remain complete and retain their stable cooker reason/detail.
    pub fn source_mesh_routes(&self) -> Vec<VirtualGeometrySourceMeshRoute> {
        let mut virtual_primitives = BTreeMap::<u32, BTreeSet<u32>>::new();
        let mut alpha_masked = BTreeMap::<u32, BTreeMap<u32, Option<u32>>>::new();
        for cluster in &self.archive.clusters {
            if cluster.flags & FLAG_ALPHA_MASKED != 0 {
                alpha_masked
                    .entry(cluster.mesh_index)
                    .or_default()
                    .entry(cluster.primitive_index)
                    .or_insert(cluster.material_index);
            } else {
                virtual_primitives
                    .entry(cluster.mesh_index)
                    .or_default()
                    .insert(cluster.primitive_index);
            }
        }
        let mut compatibility = BTreeMap::<u32, Vec<CompatibilityRecord>>::new();
        for record in &self.archive.compatibility {
            compatibility
                .entry(record.mesh_index)
                .or_default()
                .push(*record);
        }
        let source_meshes = virtual_primitives
            .keys()
            .chain(compatibility.keys())
            .chain(alpha_masked.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        source_meshes
            .into_iter()
            .map(|source_mesh_index| VirtualGeometrySourceMeshRoute {
                source_mesh_index,
                virtual_primitive_count: virtual_primitives
                    .get(&source_mesh_index)
                    .map_or(0, |primitives| primitives.len() as u32),
                compatibility: compatibility.remove(&source_mesh_index).unwrap_or_default(),
                alpha_masked_compatibility: alpha_masked
                    .remove(&source_mesh_index)
                    .unwrap_or_default()
                    .into_iter()
                    .map(
                        |(primitive_index, material_index)| VirtualGeometryAlphaMaskedRoute {
                            mesh_index: source_mesh_index,
                            primitive_index,
                            material_index,
                        },
                    )
                    .collect(),
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualGeometryLoadError {
    Format(String),
    Identity(String),
}

impl fmt::Display for VirtualGeometryLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(formatter, "invalid virtual-geometry archive: {error}"),
            Self::Identity(error) => {
                write!(formatter, "virtual-geometry identity failure: {error}")
            }
        }
    }
}

impl std::error::Error for VirtualGeometryLoadError {}
