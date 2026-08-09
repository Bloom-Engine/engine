use bloom_geometry_format::{decode_geometry, hex_hash, sha256, GeometryArchive};
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
