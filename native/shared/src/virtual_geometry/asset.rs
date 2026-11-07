use bloom_geometry_format::{
    decode_geometry, hex_hash, sha256, CompatibilityRecord, GeometryArchive, FLAG_ALPHA_MASKED,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
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
    backing: VirtualGeometryBacking,
    archive: Arc<GeometryArchive>,
    source_root_spans: Arc<BTreeMap<u32, Range<u32>>>,
}

#[derive(Clone, Debug)]
enum VirtualGeometryBacking {
    Memory(Arc<[u8]>),
    #[cfg(not(target_arch = "wasm32"))]
    File {
        path: Arc<PathBuf>,
        file_bytes: u64,
        root_pages: Arc<Vec<Arc<[u8]>>>,
    },
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
        let source_root_spans = source_root_spans(&archive);
        Ok(Self {
            backing: VirtualGeometryBacking::Memory(bytes),
            archive: Arc::new(archive),
            source_root_spans: Arc::new(source_root_spans),
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

    /// Convert a fully validated indexed archive into a file-backed asset.
    /// Only coarse fallback pages survive this call; the complete temporary
    /// byte vector is dropped on the loader worker before the result is polled.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn from_indexed_file_bytes(
        path: PathBuf,
        bytes: impl Into<Arc<[u8]>>,
        identity: ArtifactIdentity,
    ) -> Result<Self, VirtualGeometryLoadError> {
        let memory = Self::from_indexed_bytes(bytes, identity)?;
        let root_pages = (0..memory.archive.coarse_root_page_count())
            .map(|page_index| {
                memory
                    .page_bytes(page_index)
                    .map(Arc::<[u8]>::from)
                    .ok_or_else(|| {
                        VirtualGeometryLoadError::Identity(format!(
                            "coarse root page {page_index} is missing from the validated artifact"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            backing: VirtualGeometryBacking::File {
                path: Arc::new(path),
                file_bytes: identity.bytes,
                root_pages: Arc::new(root_pages),
            },
            archive: memory.archive,
            source_root_spans: memory.source_root_spans,
        })
    }

    pub fn archive(&self) -> &GeometryArchive {
        &self.archive
    }

    pub fn archive_arc(&self) -> Arc<GeometryArchive> {
        Arc::clone(&self.archive)
    }

    /// Complete resident archive bytes for direct-memory assets. File-backed
    /// store assets return `None` because only their coarse roots are retained.
    pub fn file_bytes(&self) -> Option<&[u8]> {
        match &self.backing {
            VirtualGeometryBacking::Memory(bytes) => Some(bytes),
            #[cfg(not(target_arch = "wasm32"))]
            VirtualGeometryBacking::File { .. } => None,
        }
    }

    pub fn artifact_bytes(&self) -> u64 {
        match &self.backing {
            VirtualGeometryBacking::Memory(bytes) => bytes.len() as u64,
            #[cfg(not(target_arch = "wasm32"))]
            VirtualGeometryBacking::File { file_bytes, .. } => *file_bytes,
        }
    }

    pub fn is_file_backed(&self) -> bool {
        match self.backing {
            VirtualGeometryBacking::Memory(_) => false,
            #[cfg(not(target_arch = "wasm32"))]
            VirtualGeometryBacking::File { .. } => true,
        }
    }

    pub fn page_file_range(&self, page_index: usize) -> Option<Range<usize>> {
        self.archive.page_file_range(page_index)
    }

    pub fn page_bytes(&self, page_index: usize) -> Option<&[u8]> {
        match &self.backing {
            VirtualGeometryBacking::Memory(bytes) => self
                .page_file_range(page_index)
                .and_then(|range| bytes.get(range)),
            #[cfg(not(target_arch = "wasm32"))]
            VirtualGeometryBacking::File { root_pages, .. } => {
                root_pages.get(page_index).map(AsRef::as_ref)
            }
        }
    }

    /// Materialize one validated page. File I/O callers must invoke this only
    /// on a worker; the method is synchronous by design and never used by the
    /// renderer's ordinary memory-backed service path.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(super) fn read_page_owned(
        &self,
        page_index: usize,
    ) -> Result<Vec<u8>, VirtualGeometryLoadError> {
        self.read_pages_owned(&[page_index as u32])?
            .remove(&(page_index as u32))
            .ok_or_else(|| {
                VirtualGeometryLoadError::Identity(format!("archive page {page_index} is missing"))
            })
    }

    /// Materialize one atomic page set with a single file open. The caller's
    /// validated hierarchy already supplies sorted, unique page indices.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn read_pages_owned(
        &self,
        page_indices: &[u32],
    ) -> Result<BTreeMap<u32, Vec<u8>>, VirtualGeometryLoadError> {
        let mut output = BTreeMap::new();
        let mut file = match &self.backing {
            VirtualGeometryBacking::Memory(_) => None,
            VirtualGeometryBacking::File {
                path, file_bytes, ..
            } => {
                let metadata = std::fs::metadata(path.as_ref()).map_err(|error| {
                    VirtualGeometryLoadError::Identity(format!(
                        "inspect file-backed artifact {}: {error}",
                        path.display()
                    ))
                })?;
                if metadata.len() != *file_bytes {
                    return Err(VirtualGeometryLoadError::Identity(format!(
                        "file-backed artifact length changed: expected {file_bytes}, actual {}",
                        metadata.len()
                    )));
                }
                Some(std::fs::File::open(path.as_ref()).map_err(|error| {
                    VirtualGeometryLoadError::Identity(format!(
                        "open file-backed artifact {}: {error}",
                        path.display()
                    ))
                })?)
            }
        };

        for page_index in page_indices {
            let index = *page_index as usize;
            let page = self.archive.pages.get(index).ok_or_else(|| {
                VirtualGeometryLoadError::Identity(format!("archive page {index} is missing"))
            })?;
            let range = self.page_file_range(index).ok_or_else(|| {
                VirtualGeometryLoadError::Identity(format!("archive page {index} range is missing"))
            })?;
            let bytes = match (&self.backing, file.as_mut()) {
                (VirtualGeometryBacking::Memory(bytes), _) => {
                    bytes.get(range).map(<[u8]>::to_vec).ok_or_else(|| {
                        VirtualGeometryLoadError::Identity(format!(
                            "archive page {index} range is missing"
                        ))
                    })?
                }
                (VirtualGeometryBacking::File { .. }, Some(file)) => {
                    file.seek(SeekFrom::Start(range.start as u64))
                        .map_err(|error| {
                            VirtualGeometryLoadError::Identity(format!(
                                "seek file-backed page {index}: {error}"
                            ))
                        })?;
                    let mut bytes = vec![0; range.len()];
                    file.read_exact(&mut bytes).map_err(|error| {
                        VirtualGeometryLoadError::Identity(format!(
                            "read file-backed page {index}: {error}"
                        ))
                    })?;
                    bytes
                }
                (VirtualGeometryBacking::File { .. }, None) => unreachable!(),
            };
            let actual = sha256(&bytes);
            if actual != page.sha256 {
                return Err(VirtualGeometryLoadError::Identity(format!(
                    "page {index} hash mismatch: expected {}, actual {}",
                    hex_hash(page.sha256),
                    hex_hash(actual)
                )));
            }
            output.insert(*page_index, bytes);
        }
        Ok(output)
    }

    /// Smallest root-table span containing every coarse root for one source
    /// mesh. Current cookers make this range exact; older archives remain
    /// correct because traversal still verifies each root's source identity.
    pub(crate) fn source_root_span(&self, source_mesh_index: u32) -> Option<Range<u32>> {
        self.source_root_spans.get(&source_mesh_index).cloned()
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

fn source_root_spans(archive: &GeometryArchive) -> BTreeMap<u32, Range<u32>> {
    let root_count = archive.pages[..archive.coarse_root_page_count()]
        .iter()
        .map(|page| page.cluster_count as usize)
        .sum::<usize>()
        .min(archive.clusters.len());
    let mut spans = BTreeMap::<u32, Range<u32>>::new();
    for (root_index, cluster) in archive.clusters[..root_count].iter().enumerate() {
        let root_index = root_index as u32;
        spans
            .entry(cluster.mesh_index)
            .and_modify(|span| span.end = root_index + 1)
            .or_insert(root_index..root_index + 1);
    }
    spans
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
