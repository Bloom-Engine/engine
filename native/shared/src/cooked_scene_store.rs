//! Source-free indexed scene selection and validation for native shipping.
//!
//! This synchronous boundary is intended for a loading worker, not a render
//! callback. It reads only `index.json` and one immutable `.bscene` chunk;
//! source glTF, buffers, images, and cooker manifests are never consulted.

use crate::adapter_profile::AdapterAssetProfilePlan;
use crate::models_cooked_scene::{prepare_cooked_scene, PreparedCookedScene};
use bloom_scene_format::{hex_hash, sha256, VERSION as SCENE_FORMAT_VERSION};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CookedSceneProfile {
    platform: String,
    quality: String,
}

impl CookedSceneProfile {
    pub fn new(platform: &str, quality: &str) -> Result<Self, CookedSceneStoreError> {
        validate_profile_component(platform, "platform")?;
        validate_profile_component(quality, "quality")?;
        Ok(Self {
            platform: platform.to_string(),
            quality: quality.to_string(),
        })
    }

    pub fn platform(&self) -> &str {
        &self.platform
    }

    pub fn quality(&self) -> &str {
        &self.quality
    }

    pub fn label(&self) -> String {
        format!("{}/{}", self.platform, self.quality)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CookedSceneStoreRequest {
    pub logical_id: String,
    pub requested: CookedSceneProfile,
    pub fallbacks: Vec<CookedSceneProfile>,
    pub allow_unprofiled: bool,
}

impl CookedSceneStoreRequest {
    pub fn new(logical_id: impl Into<String>, requested: CookedSceneProfile) -> Self {
        Self {
            logical_id: logical_id.into(),
            requested,
            fallbacks: Vec::new(),
            allow_unprofiled: false,
        }
    }

    pub fn for_device(
        logical_id: impl Into<String>,
        quality: &str,
        device: &wgpu::Device,
    ) -> Result<Self, CookedSceneStoreError> {
        let plan = AdapterAssetProfilePlan::from_features(device.features());
        let mut request = Self::new(
            logical_id,
            CookedSceneProfile::new(plan.selected_platform(), quality)?,
        );
        if plan.has_portable_fallback() {
            request
                .fallbacks
                .push(CookedSceneProfile::new("portable", quality)?);
        }
        Ok(request)
    }

    pub fn with_fallback(mut self, fallback: CookedSceneProfile) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    pub fn allow_unprofiled(mut self, allow: bool) -> Self {
        self.allow_unprofiled = allow;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CookedSceneSelectionKind {
    Exact,
    Fallback,
    UnprofiledFallback,
}

pub struct ResolvedCookedScene {
    pub logical_id: String,
    pub requested_profile: CookedSceneProfile,
    pub selected_profile: Option<CookedSceneProfile>,
    pub selection_kind: CookedSceneSelectionKind,
    pub fallback_rank: Option<u32>,
    pub artifact_path: PathBuf,
    pub artifact_bytes: u64,
    pub artifact_sha256: [u8; 32],
    pub prepared: PreparedCookedScene,
}

impl ResolvedCookedScene {
    pub fn report_json(&self) -> String {
        serde_json::json!({
            "artifact": {
                "bytes": self.artifact_bytes,
                "path": self.artifact_path.display().to_string(),
                "sha256": hex_hash(self.artifact_sha256),
            },
            "logical_id": self.logical_id,
            "schema": "bloom-runtime-scene-selection-v1",
            "selection": {
                "fallback_rank": self.fallback_rank,
                "kind": match self.selection_kind {
                    CookedSceneSelectionKind::Exact => "exact",
                    CookedSceneSelectionKind::Fallback => "fallback",
                    CookedSceneSelectionKind::UnprofiledFallback => "unprofiled-fallback",
                },
                "requested_profile": {
                    "platform": self.requested_profile.platform(),
                    "quality": self.requested_profile.quality(),
                },
                "selected_profile": self.selected_profile.as_ref().map(|profile| serde_json::json!({
                    "platform": profile.platform(),
                    "quality": profile.quality(),
                })),
            },
        })
        .to_string()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CookedSceneStoreConfig {
    pub max_index_bytes: u64,
    pub max_artifact_bytes: u64,
}

impl Default for CookedSceneStoreConfig {
    fn default() -> Self {
        Self {
            max_index_bytes: 16 * 1024 * 1024,
            max_artifact_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

pub fn load_cooked_scene_from_store(
    store: &Path,
    request: &CookedSceneStoreRequest,
    config: CookedSceneStoreConfig,
) -> Result<ResolvedCookedScene, CookedSceneStoreError> {
    validate_request(request)?;
    if config.max_index_bytes == 0 || config.max_artifact_bytes == 0 {
        return Err(CookedSceneStoreError::InvalidRequest(
            "scene store budgets must be non-zero".to_string(),
        ));
    }
    let index_path = store.join("index.json");
    let index_bytes = read_bounded(&index_path, config.max_index_bytes, "asset index")?;
    let index: Value = serde_json::from_slice(&index_bytes).map_err(|error| {
        CookedSceneStoreError::Index(format!(
            "parse asset index {}: {error}",
            index_path.display()
        ))
    })?;
    let entries = parse_index(&index)?;
    let (entry, kind, fallback_rank) = select_entry(&entries, request)?;
    if entry.artifact_bytes > config.max_artifact_bytes {
        return Err(CookedSceneStoreError::Artifact(format!(
            "scene declares {} bytes, loader limit is {}",
            entry.artifact_bytes, config.max_artifact_bytes
        )));
    }
    validate_chunk_path(&entry.artifact_path, entry.artifact_sha256)?;
    reject_symlink_path(store, &entry.artifact_path)?;
    let artifact_path = store.join(&entry.artifact_path);
    let bytes = read_bounded(&artifact_path, config.max_artifact_bytes, "scene artifact")?;
    if bytes.len() as u64 != entry.artifact_bytes {
        return Err(CookedSceneStoreError::Artifact(format!(
            "scene byte length is {}, index declares {}",
            bytes.len(),
            entry.artifact_bytes
        )));
    }
    let actual_hash = sha256(&bytes);
    if actual_hash != entry.artifact_sha256 {
        return Err(CookedSceneStoreError::Artifact(format!(
            "scene hash is {}, index declares {}",
            hex_hash(actual_hash),
            hex_hash(entry.artifact_sha256)
        )));
    }
    let prepared = prepare_cooked_scene(&bytes).map_err(CookedSceneStoreError::Artifact)?;
    if prepared.payload_sha256() != entry.payload_sha256 {
        return Err(CookedSceneStoreError::Artifact(
            "scene payload hash does not match the index".to_string(),
        ));
    }
    let dependency_count = prepared.texture_dependencies().len() as u64;
    if dependency_count != entry.textures {
        return Err(CookedSceneStoreError::Artifact(format!(
            "scene has {dependency_count} texture dependencies, index declares {}",
            entry.textures
        )));
    }
    Ok(ResolvedCookedScene {
        logical_id: request.logical_id.clone(),
        requested_profile: request.requested.clone(),
        selected_profile: entry.profile.clone(),
        selection_kind: kind,
        fallback_rank,
        artifact_path,
        artifact_bytes: entry.artifact_bytes,
        artifact_sha256: entry.artifact_sha256,
        prepared,
    })
}

#[derive(Clone)]
struct SceneEntry {
    logical_id: String,
    profile: Option<CookedSceneProfile>,
    artifact_path: PathBuf,
    artifact_bytes: u64,
    artifact_sha256: [u8; 32],
    payload_sha256: [u8; 32],
    textures: u64,
}

fn parse_index(index: &Value) -> Result<Vec<SceneEntry>, CookedSceneStoreError> {
    let object = index
        .as_object()
        .ok_or_else(|| CookedSceneStoreError::Index("asset index is not an object".to_string()))?;
    let schema = required_string(object, "schema")?;
    if !matches!(schema, "bloom-asset-index-v1" | "bloom-asset-index-v2") {
        return Err(CookedSceneStoreError::Index(format!(
            "unsupported asset index schema {schema:?}; recook assets"
        )));
    }
    let values = object
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            CookedSceneStoreError::Index("asset index entries are missing".to_string())
        })?;
    if required_u64(object, "entry_count")? != values.len() as u64 {
        return Err(CookedSceneStoreError::Index(
            "asset index entry_count does not match entries".to_string(),
        ));
    }
    let mut identities = BTreeSet::new();
    let mut profiled = 0u64;
    let mut scenes = Vec::new();
    for value in values {
        let entry = value.as_object().ok_or_else(|| {
            CookedSceneStoreError::Index("index entry is not an object".to_string())
        })?;
        let kind = required_string(entry, "kind")?;
        if !matches!(kind, "geometry" | "texture" | "scene") {
            return Err(CookedSceneStoreError::Index(format!(
                "asset index contains unsupported asset kind {kind:?}"
            )));
        }
        let logical_id = required_string(entry, "logical_id")?.to_string();
        validate_logical_id(&logical_id)?;
        let profile = entry.get("profile").map(parse_profile).transpose()?;
        profiled += u64::from(profile.is_some());
        if !identities.insert((logical_id.clone(), profile.clone())) {
            return Err(CookedSceneStoreError::Index(
                "asset index contains a duplicate logical ID/profile".to_string(),
            ));
        }
        if kind == "scene" {
            scenes.push(parse_scene_entry(entry, logical_id, profile)?);
        }
    }
    if schema == "bloom-asset-index-v1" && profiled != 0 {
        return Err(CookedSceneStoreError::Index(
            "v1 asset index may not contain profiled entries".to_string(),
        ));
    }
    if schema == "bloom-asset-index-v2" && required_u64(object, "profiled_entry_count")? != profiled
    {
        return Err(CookedSceneStoreError::Index(
            "asset index profiled_entry_count does not match entries".to_string(),
        ));
    }
    Ok(scenes)
}

fn parse_scene_entry(
    entry: &Map<String, Value>,
    logical_id: String,
    profile: Option<CookedSceneProfile>,
) -> Result<SceneEntry, CookedSceneStoreError> {
    let artifact = entry
        .get("artifact")
        .and_then(Value::as_object)
        .ok_or_else(|| CookedSceneStoreError::Index("scene artifact is missing".to_string()))?;
    let version = required_u64(artifact, "format_version")?;
    if version != u64::from(SCENE_FORMAT_VERSION) {
        return Err(CookedSceneStoreError::Index(format!(
            "scene format {version} is incompatible; recook assets"
        )));
    }
    Ok(SceneEntry {
        logical_id,
        profile,
        artifact_path: PathBuf::from(required_string(artifact, "path")?),
        artifact_bytes: required_u64(artifact, "bytes")?,
        artifact_sha256: parse_hash(required_string(artifact, "sha256")?, "scene hash")?,
        payload_sha256: parse_hash(
            required_string(artifact, "payload_sha256")?,
            "scene payload hash",
        )?,
        textures: required_u64(artifact, "textures")?,
    })
}

fn select_entry<'a>(
    entries: &'a [SceneEntry],
    request: &CookedSceneStoreRequest,
) -> Result<(&'a SceneEntry, CookedSceneSelectionKind, Option<u32>), CookedSceneStoreError> {
    if let Some(entry) = entries.iter().find(|entry| {
        entry.logical_id == request.logical_id && entry.profile.as_ref() == Some(&request.requested)
    }) {
        return Ok((entry, CookedSceneSelectionKind::Exact, None));
    }
    for (rank, fallback) in request.fallbacks.iter().enumerate() {
        if let Some(entry) = entries.iter().find(|entry| {
            entry.logical_id == request.logical_id && entry.profile.as_ref() == Some(fallback)
        }) {
            return Ok((entry, CookedSceneSelectionKind::Fallback, Some(rank as u32)));
        }
    }
    if request.allow_unprofiled {
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.logical_id == request.logical_id && entry.profile.is_none())
        {
            return Ok((entry, CookedSceneSelectionKind::UnprofiledFallback, None));
        }
    }
    Err(CookedSceneStoreError::Missing(format!(
        "no indexed scene {:?} matches requested profile {}",
        request.logical_id,
        request.requested.label()
    )))
}

fn parse_profile(value: &Value) -> Result<CookedSceneProfile, CookedSceneStoreError> {
    let profile = value.as_object().ok_or_else(|| {
        CookedSceneStoreError::Index("asset profile is not an object".to_string())
    })?;
    if profile.len() != 2 {
        return Err(CookedSceneStoreError::Index(
            "asset profile has unknown or missing fields".to_string(),
        ));
    }
    CookedSceneProfile::new(
        required_string(profile, "platform")?,
        required_string(profile, "quality")?,
    )
}

fn validate_request(request: &CookedSceneStoreRequest) -> Result<(), CookedSceneStoreError> {
    validate_logical_id(&request.logical_id)?;
    let mut profiles = BTreeSet::new();
    profiles.insert(request.requested.clone());
    if request
        .fallbacks
        .iter()
        .any(|profile| !profiles.insert(profile.clone()))
    {
        return Err(CookedSceneStoreError::InvalidRequest(
            "scene fallback profiles must be unique".to_string(),
        ));
    }
    Ok(())
}

fn validate_chunk_path(path: &Path, hash: [u8; 32]) -> Result<(), CookedSceneStoreError> {
    let expected = PathBuf::from(format!("chunks/sha256/{}.bscene", hex_hash(hash)));
    if path != expected
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(CookedSceneStoreError::Index(format!(
            "scene chunk path {:?} is not canonical {:?}",
            path, expected
        )));
    }
    Ok(())
}

fn reject_symlink_path(store: &Path, relative: &Path) -> Result<(), CookedSceneStoreError> {
    let mut current = store.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(CookedSceneStoreError::Index(
                "scene path is not relative".to_string(),
            ));
        };
        current.push(component);
        let metadata = std::fs::symlink_metadata(&current).map_err(|error| {
            CookedSceneStoreError::Io(format!("inspect {}: {error}", current.display()))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(CookedSceneStoreError::Artifact(format!(
                "scene path {} traverses a symlink",
                current.display()
            )));
        }
    }
    Ok(())
}

fn read_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>, CookedSceneStoreError> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        CookedSceneStoreError::Io(format!("inspect {label} {}: {error}", path.display()))
    })?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err(CookedSceneStoreError::Io(format!(
            "{label} {} has {} bytes, maximum is {maximum}",
            path.display(),
            metadata.len()
        )));
    }
    std::fs::read(path).map_err(|error| {
        CookedSceneStoreError::Io(format!("read {label} {}: {error}", path.display()))
    })
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, CookedSceneStoreError> {
    object.get(field).and_then(Value::as_str).ok_or_else(|| {
        CookedSceneStoreError::Index(format!("required string field {field:?} is missing"))
    })
}

fn required_u64(object: &Map<String, Value>, field: &str) -> Result<u64, CookedSceneStoreError> {
    object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        CookedSceneStoreError::Index(format!("required integer field {field:?} is missing"))
    })
}

fn parse_hash(value: &str, label: &str) -> Result<[u8; 32], CookedSceneStoreError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(CookedSceneStoreError::Index(format!(
            "{label} is not canonical SHA-256"
        )));
    }
    let mut output = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    Ok(output)
}

fn validate_logical_id(value: &str) -> Result<(), CookedSceneStoreError> {
    if value.is_empty()
        || value.len() > 512
        || !value.is_ascii()
        || value.starts_with('/')
        || value.contains('\\')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(CookedSceneStoreError::InvalidRequest(format!(
            "invalid logical asset ID {value:?}"
        )));
    }
    Ok(())
}

fn validate_profile_component(value: &str, label: &str) -> Result<(), CookedSceneStoreError> {
    if value.is_empty()
        || value.len() > 32
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
    {
        return Err(CookedSceneStoreError::InvalidRequest(format!(
            "invalid {label} profile {value:?}"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CookedSceneStoreError {
    InvalidRequest(String),
    Missing(String),
    Index(String),
    Artifact(String),
    Io(String),
}

impl fmt::Display for CookedSceneStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(formatter, "invalid scene request: {message}"),
            Self::Missing(message) => write!(formatter, "missing cooked scene: {message}"),
            Self::Index(message) => write!(formatter, "invalid cooked-scene index: {message}"),
            Self::Artifact(message) => write!(formatter, "invalid cooked scene: {message}"),
            Self::Io(message) => write!(formatter, "cooked-scene I/O error: {message}"),
        }
    }
}

impl std::error::Error for CookedSceneStoreError {}
