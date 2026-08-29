//! Non-blocking native loading of indexed cooked DDS texture variants.
//!
//! The worker consumes only the shipping `index.json` and immutable chunk.
//! Source images and cooker manifests are not runtime dependencies. Filesystem
//! access, hashing, DDS parsing, and package validation remain off the render
//! thread; the completed DDS is uploaded through the established texture path.

use crate::adapter_profile::AdapterAssetProfilePlan;
use image_dds::ddsfile::{Dds, DxgiFormat};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Cursor;
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};

const INDEX_FILE: &str = "index.json";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CookedTextureProfile {
    platform: String,
    quality: String,
}

impl CookedTextureProfile {
    pub fn new(platform: &str, quality: &str) -> Result<Self, CookedTextureStoreError> {
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

    fn as_json(&self) -> Value {
        serde_json::json!({
            "platform": self.platform,
            "quality": self.quality,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CookedTextureRequestPolicy {
    Explicit,
    Adapter {
        runtime_platform: &'static str,
        bc_supported: bool,
        native_profile_selected: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CookedTextureStoreRequest {
    pub logical_id: String,
    pub requested: CookedTextureProfile,
    pub fallbacks: Vec<CookedTextureProfile>,
    pub allow_unprofiled: bool,
    policy: CookedTextureRequestPolicy,
}

impl CookedTextureStoreRequest {
    pub fn new(logical_id: impl Into<String>, requested: CookedTextureProfile) -> Self {
        Self {
            logical_id: logical_id.into(),
            requested,
            fallbacks: Vec::new(),
            allow_unprofiled: false,
            policy: CookedTextureRequestPolicy::Explicit,
        }
    }

    pub fn for_device(
        logical_id: impl Into<String>,
        quality: &str,
        device: &wgpu::Device,
    ) -> Result<Self, CookedTextureStoreError> {
        Self::for_runtime_features(logical_id, quality, device.features())
    }

    fn for_runtime_features(
        logical_id: impl Into<String>,
        quality: &str,
        features: wgpu::Features,
    ) -> Result<Self, CookedTextureStoreError> {
        let plan = AdapterAssetProfilePlan::from_features(features);
        let requested = CookedTextureProfile::new(plan.selected_platform(), quality)?;
        let mut request = Self::new(logical_id, requested);
        request.policy = CookedTextureRequestPolicy::Adapter {
            runtime_platform: plan.runtime_platform(),
            bc_supported: plan.bc_supported(),
            native_profile_selected: plan.native_profile_selected(),
        };
        if plan.has_portable_fallback() {
            request
                .fallbacks
                .push(CookedTextureProfile::new("portable", quality)?);
        }
        Ok(request)
    }

    pub fn with_fallback(mut self, fallback: CookedTextureProfile) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    pub fn allow_unprofiled(mut self, allow: bool) -> Self {
        self.allow_unprofiled = allow;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CookedTextureSelectionKind {
    Exact,
    Fallback,
    UnprofiledFallback,
}

impl CookedTextureSelectionKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Fallback => "fallback",
            Self::UnprofiledFallback => "unprofiled-fallback",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CookedTextureStoreSelection {
    pub kind: CookedTextureSelectionKind,
    pub requested_profile: CookedTextureProfile,
    pub selected_profile: Option<CookedTextureProfile>,
    pub fallback_rank: Option<u32>,
    pub reason: &'static str,
    pub request_policy: CookedTextureRequestPolicy,
}

impl CookedTextureStoreSelection {
    fn as_json(&self) -> Value {
        let policy = match self.request_policy {
            CookedTextureRequestPolicy::Explicit => serde_json::json!({"kind": "explicit"}),
            CookedTextureRequestPolicy::Adapter {
                runtime_platform,
                bc_supported,
                native_profile_selected,
            } => serde_json::json!({
                "bc_supported": bc_supported,
                "kind": "adapter",
                "native_profile_selected": native_profile_selected,
                "runtime_platform": runtime_platform,
            }),
        };
        serde_json::json!({
            "fallback_rank": self.fallback_rank,
            "kind": self.kind.name(),
            "policy": policy,
            "reason": self.reason,
            "requested_profile": self.requested_profile.as_json(),
            "selected_profile": self.selected_profile.as_ref().map(CookedTextureProfile::as_json),
        })
    }

    pub fn report_json(&self) -> String {
        self.as_json().to_string()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CookedTextureArtifactFormat {
    Bc7Linear,
    Bc7Srgb,
    Rgba8Linear,
    Rgba8Srgb,
    Rgba8NormalVariance,
}

impl CookedTextureArtifactFormat {
    fn parse(value: &str) -> Result<Self, CookedTextureStoreError> {
        match value {
            "bc7-rgba-unorm" => Ok(Self::Bc7Linear),
            "bc7-rgba-unorm-srgb" => Ok(Self::Bc7Srgb),
            "rgba8-unorm" => Ok(Self::Rgba8Linear),
            "rgba8-unorm-srgb" => Ok(Self::Rgba8Srgb),
            "rgba8-unorm-normal-variance" => Ok(Self::Rgba8NormalVariance),
            other => Err(CookedTextureStoreError::Index(format!(
                "unsupported cooked texture format {other:?}; recook assets"
            ))),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bc7Linear => "bc7-rgba-unorm",
            Self::Bc7Srgb => "bc7-rgba-unorm-srgb",
            Self::Rgba8Linear => "rgba8-unorm",
            Self::Rgba8Srgb => "rgba8-unorm-srgb",
            Self::Rgba8NormalVariance => "rgba8-unorm-normal-variance",
        }
    }

    pub const fn requires_bc(self) -> bool {
        matches!(self, Self::Bc7Linear | Self::Bc7Srgb)
    }

    pub const fn is_normal_map(self) -> bool {
        matches!(self, Self::Rgba8NormalVariance)
    }

    const fn dxgi(self) -> DxgiFormat {
        match self {
            Self::Bc7Linear => DxgiFormat::BC7_UNorm,
            Self::Bc7Srgb => DxgiFormat::BC7_UNorm_sRGB,
            Self::Rgba8Linear | Self::Rgba8NormalVariance => DxgiFormat::R8G8B8A8_UNorm,
            Self::Rgba8Srgb => DxgiFormat::R8G8B8A8_UNorm_sRGB,
        }
    }
}

pub struct ResolvedCookedTexture {
    pub logical_id: String,
    pub selection: CookedTextureStoreSelection,
    pub artifact_path: PathBuf,
    pub artifact_bytes: u64,
    pub artifact_sha256: [u8; 32],
    pub format: CookedTextureArtifactFormat,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    dds: Dds,
}

impl ResolvedCookedTexture {
    pub fn dds(&self) -> &Dds {
        &self.dds
    }

    pub fn report_json(&self) -> String {
        serde_json::json!({
            "artifact": {
                "bytes": self.artifact_bytes,
                "format": self.format.name(),
                "height": self.height,
                "mip_levels": self.mip_levels,
                "path": self.artifact_path.display().to_string(),
                "sha256": hex_hash(self.artifact_sha256),
                "width": self.width,
            },
            "logical_id": self.logical_id,
            "schema": "bloom-runtime-texture-selection-v1",
            "selection": self.selection.as_json(),
        })
        .to_string()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CookedTextureStoreTicket(u64);

impl CookedTextureStoreTicket {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CookedTextureStoreConfig {
    pub max_pending_requests: u32,
    pub max_index_bytes: u64,
    pub max_artifact_bytes: u64,
}

impl Default for CookedTextureStoreConfig {
    fn default() -> Self {
        Self {
            max_pending_requests: 32,
            max_index_bytes: 16 * 1024 * 1024,
            max_artifact_bytes: 1024 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CookedTextureStoreTelemetry {
    pub pending_requests: u32,
    pub queued_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub queue_full_rejections: u64,
    pub loaded_artifact_bytes: u64,
    pub exact_selections: u64,
    pub fallback_selections: u64,
    pub unprofiled_selections: u64,
}

pub struct CookedTextureStoreLoader {
    request_tx: SyncSender<WorkerRequest>,
    completion_rx: Receiver<WorkerCompletion>,
    completed: BTreeMap<CookedTextureStoreTicket, WorkerCompletion>,
    max_outstanding_requests: u32,
    next_ticket: u64,
    telemetry: CookedTextureStoreTelemetry,
}

impl CookedTextureStoreLoader {
    pub fn new(
        store: impl Into<PathBuf>,
        config: CookedTextureStoreConfig,
    ) -> Result<Self, CookedTextureStoreError> {
        if config.max_pending_requests == 0
            || config.max_index_bytes == 0
            || config.max_artifact_bytes == 0
        {
            return Err(CookedTextureStoreError::InvalidRequest(
                "store loader budgets must all be non-zero".to_string(),
            ));
        }
        let (request_tx, request_rx) =
            mpsc::sync_channel::<WorkerRequest>(config.max_pending_requests as usize);
        let (completion_tx, completion_rx) = mpsc::channel::<WorkerCompletion>();
        let store = store.into();
        std::thread::Builder::new()
            .name("bloom-texture-store".to_string())
            .spawn(move || worker_main(&store, config, request_rx, completion_tx))
            .map_err(|error| {
                CookedTextureStoreError::Io(format!("start cooked-texture worker: {error}"))
            })?;
        Ok(Self {
            request_tx,
            completion_rx,
            completed: BTreeMap::new(),
            max_outstanding_requests: config.max_pending_requests,
            next_ticket: 1,
            telemetry: CookedTextureStoreTelemetry::default(),
        })
    }

    pub fn request(
        &mut self,
        request: CookedTextureStoreRequest,
    ) -> Result<CookedTextureStoreTicket, CookedTextureStoreError> {
        validate_request(&request)?;
        self.drain_completions();
        if self
            .telemetry
            .pending_requests
            .saturating_add(self.completed.len() as u32)
            >= self.max_outstanding_requests
        {
            self.telemetry.queue_full_rejections =
                self.telemetry.queue_full_rejections.saturating_add(1);
            return Err(CookedTextureStoreError::QueueFull);
        }
        let ticket = CookedTextureStoreTicket(self.next_ticket);
        match self.request_tx.try_send(WorkerRequest { ticket, request }) {
            Ok(()) => {
                self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
                self.telemetry.queued_requests = self.telemetry.queued_requests.saturating_add(1);
                self.telemetry.pending_requests = self.telemetry.pending_requests.saturating_add(1);
                Ok(ticket)
            }
            Err(TrySendError::Full(_)) => {
                self.telemetry.queue_full_rejections =
                    self.telemetry.queue_full_rejections.saturating_add(1);
                Err(CookedTextureStoreError::QueueFull)
            }
            Err(TrySendError::Disconnected(_)) => Err(CookedTextureStoreError::WorkerStopped),
        }
    }

    pub fn poll(
        &mut self,
        ticket: CookedTextureStoreTicket,
    ) -> Option<Result<ResolvedCookedTexture, CookedTextureStoreError>> {
        self.drain_completions();
        self.completed
            .remove(&ticket)
            .map(|completion| completion.result)
    }

    pub fn telemetry(&mut self) -> CookedTextureStoreTelemetry {
        self.drain_completions();
        self.telemetry
    }

    fn drain_completions(&mut self) {
        loop {
            match self.completion_rx.try_recv() {
                Ok(completion) => {
                    self.telemetry.pending_requests =
                        self.telemetry.pending_requests.saturating_sub(1);
                    self.telemetry.completed_requests =
                        self.telemetry.completed_requests.saturating_add(1);
                    match &completion.result {
                        Ok(resolved) => {
                            self.telemetry.loaded_artifact_bytes = self
                                .telemetry
                                .loaded_artifact_bytes
                                .saturating_add(resolved.artifact_bytes);
                            match resolved.selection.kind {
                                CookedTextureSelectionKind::Exact => {
                                    self.telemetry.exact_selections =
                                        self.telemetry.exact_selections.saturating_add(1);
                                }
                                CookedTextureSelectionKind::Fallback => {
                                    self.telemetry.fallback_selections =
                                        self.telemetry.fallback_selections.saturating_add(1);
                                }
                                CookedTextureSelectionKind::UnprofiledFallback => {
                                    self.telemetry.unprofiled_selections =
                                        self.telemetry.unprofiled_selections.saturating_add(1);
                                }
                            }
                            if resolved.selection.request_policy
                                != CookedTextureRequestPolicy::Explicit
                                || resolved.selection.kind != CookedTextureSelectionKind::Exact
                            {
                                log::info!(
                                    "bloom: cooked texture selection = {}",
                                    resolved.report_json()
                                );
                            }
                        }
                        Err(_) => {
                            self.telemetry.failed_requests =
                                self.telemetry.failed_requests.saturating_add(1);
                        }
                    }
                    self.completed.insert(completion.ticket, completion);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
    }
}

struct WorkerRequest {
    ticket: CookedTextureStoreTicket,
    request: CookedTextureStoreRequest,
}

struct WorkerCompletion {
    ticket: CookedTextureStoreTicket,
    result: Result<ResolvedCookedTexture, CookedTextureStoreError>,
}

fn worker_main(
    store: &Path,
    config: CookedTextureStoreConfig,
    request_rx: Receiver<WorkerRequest>,
    completion_tx: mpsc::Sender<WorkerCompletion>,
) {
    while let Ok(request) = request_rx.recv() {
        let result = resolve_and_load(store, config, &request.request);
        if completion_tx
            .send(WorkerCompletion {
                ticket: request.ticket,
                result,
            })
            .is_err()
        {
            break;
        }
    }
}

#[derive(Clone)]
struct IndexedTextureEntry {
    logical_id: String,
    profile: Option<CookedTextureProfile>,
    artifact_path: PathBuf,
    artifact_bytes: u64,
    artifact_sha256: [u8; 32],
    format: CookedTextureArtifactFormat,
    width: u32,
    height: u32,
    mip_levels: u32,
}

fn resolve_and_load(
    store: &Path,
    config: CookedTextureStoreConfig,
    request: &CookedTextureStoreRequest,
) -> Result<ResolvedCookedTexture, CookedTextureStoreError> {
    let index_path = store.join(INDEX_FILE);
    let index_bytes = read_bounded(&index_path, config.max_index_bytes, "asset index")?;
    let index: Value = serde_json::from_slice(&index_bytes).map_err(|error| {
        CookedTextureStoreError::Index(format!(
            "parse asset index {}: {error}",
            index_path.display()
        ))
    })?;
    let entries = parse_index(&index)?;
    let (entry, selection) = select_entry(&entries, request)?;
    if entry.artifact_bytes > config.max_artifact_bytes {
        return Err(CookedTextureStoreError::Io(format!(
            "texture declares {} bytes, loader limit is {}",
            entry.artifact_bytes, config.max_artifact_bytes
        )));
    }
    validate_selection_format(&selection, entry.format)?;
    validate_chunk_path(&entry.artifact_path, entry.artifact_sha256)?;
    reject_symlink_path(store, &entry.artifact_path)?;
    let artifact_path = store.join(&entry.artifact_path);
    let bytes = read_bounded(
        &artifact_path,
        config.max_artifact_bytes,
        "texture artifact",
    )?;
    if bytes.len() as u64 != entry.artifact_bytes {
        return Err(CookedTextureStoreError::Artifact(format!(
            "texture byte length is {}, index declares {}",
            bytes.len(),
            entry.artifact_bytes
        )));
    }
    let actual_hash = sha256(&bytes);
    if actual_hash != entry.artifact_sha256 {
        return Err(CookedTextureStoreError::Artifact(format!(
            "texture hash is {}, index declares {}",
            hex_hash(actual_hash),
            hex_hash(entry.artifact_sha256)
        )));
    }
    let dds = validate_dds(&bytes, &entry)?;
    Ok(ResolvedCookedTexture {
        logical_id: request.logical_id.clone(),
        selection,
        artifact_path,
        artifact_bytes: entry.artifact_bytes,
        artifact_sha256: entry.artifact_sha256,
        format: entry.format,
        width: entry.width,
        height: entry.height,
        mip_levels: entry.mip_levels,
        dds,
    })
}

fn parse_index(index: &Value) -> Result<Vec<IndexedTextureEntry>, CookedTextureStoreError> {
    let object = index.as_object().ok_or_else(|| {
        CookedTextureStoreError::Index("asset index is not an object".to_string())
    })?;
    let schema = required_string(object, "schema")?;
    if !matches!(schema, "bloom-asset-index-v1" | "bloom-asset-index-v2") {
        return Err(CookedTextureStoreError::Index(format!(
            "unsupported asset index schema {schema:?}; recook assets"
        )));
    }
    let values = object
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            CookedTextureStoreError::Index("asset index entries are missing".to_string())
        })?;
    let declared_count = required_u64(object, "entry_count")?;
    if declared_count != values.len() as u64 {
        return Err(CookedTextureStoreError::Index(format!(
            "asset index entry_count is {declared_count}, actual {}",
            values.len()
        )));
    }

    let mut entries = Vec::new();
    let mut identities = BTreeSet::new();
    let mut profiled_entries = 0u64;
    for value in values {
        let entry = value.as_object().ok_or_else(|| {
            CookedTextureStoreError::Index("asset index entry is not an object".to_string())
        })?;
        let kind = required_string(entry, "kind")?;
        let logical_id = required_string(entry, "logical_id")?.to_string();
        validate_logical_id(&logical_id)?;
        let profile = entry.get("profile").map(parse_profile).transpose()?;
        profiled_entries += u64::from(profile.is_some());
        if !identities.insert((logical_id, profile)) {
            return Err(CookedTextureStoreError::Index(
                "asset index contains a duplicate logical ID/profile".to_string(),
            ));
        }
        match kind {
            "texture" => entries.push(parse_entry(value)?),
            "geometry" => {}
            other => {
                return Err(CookedTextureStoreError::Index(format!(
                    "asset index contains unsupported asset kind {other:?}"
                )))
            }
        }
    }
    if schema == "bloom-asset-index-v1" && profiled_entries != 0 {
        return Err(CookedTextureStoreError::Index(
            "v1 asset index may not contain profiled entries".to_string(),
        ));
    }
    if schema == "bloom-asset-index-v2"
        && required_u64(object, "profiled_entry_count")? != profiled_entries
    {
        return Err(CookedTextureStoreError::Index(
            "asset index profiled_entry_count does not match its entries".to_string(),
        ));
    }
    Ok(entries)
}

fn parse_entry(value: &Value) -> Result<IndexedTextureEntry, CookedTextureStoreError> {
    let object = value.as_object().ok_or_else(|| {
        CookedTextureStoreError::Index("asset index entry is not an object".to_string())
    })?;
    let logical_id = required_string(object, "logical_id")?.to_string();
    validate_logical_id(&logical_id)?;
    parse_hash(required_string(object, "source_sha256")?, "source hash")?;
    parse_hash(required_string(object, "build_key_sha256")?, "build key")?;
    let manifest = object
        .get("manifest")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            CookedTextureStoreError::Index("manifest identity is missing".to_string())
        })?;
    parse_hash(required_string(manifest, "sha256")?, "manifest hash")?;
    let artifact = object
        .get("artifact")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            CookedTextureStoreError::Index(format!(
                "logical texture {logical_id:?} has no artifact object"
            ))
        })?;
    let artifact_bytes = required_u64(artifact, "bytes")?;
    let width = required_u32(artifact, "width")?;
    let height = required_u32(artifact, "height")?;
    let mip_levels = required_u32(artifact, "mip_levels")?;
    if artifact_bytes == 0 || width == 0 || height == 0 || mip_levels == 0 {
        return Err(CookedTextureStoreError::Index(format!(
            "logical texture {logical_id:?} declares an empty artifact"
        )));
    }
    let maximum_mips = u32::BITS - width.max(height).leading_zeros();
    if mip_levels > maximum_mips {
        return Err(CookedTextureStoreError::Index(format!(
            "logical texture {logical_id:?} declares {mip_levels} mips, maximum is {maximum_mips}"
        )));
    }
    Ok(IndexedTextureEntry {
        logical_id,
        profile: object.get("profile").map(parse_profile).transpose()?,
        artifact_path: PathBuf::from(required_string(artifact, "path")?),
        artifact_bytes,
        artifact_sha256: parse_hash(required_string(artifact, "sha256")?, "artifact hash")?,
        format: CookedTextureArtifactFormat::parse(required_string(artifact, "format")?)?,
        width,
        height,
        mip_levels,
    })
}

fn select_entry(
    entries: &[IndexedTextureEntry],
    request: &CookedTextureStoreRequest,
) -> Result<(IndexedTextureEntry, CookedTextureStoreSelection), CookedTextureStoreError> {
    let candidates = entries
        .iter()
        .filter(|entry| entry.logical_id == request.logical_id)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(CookedTextureStoreError::Resolution(format!(
            "logical texture {:?} is not in the cooked index",
            request.logical_id
        )));
    }
    if let Some(entry) = candidates
        .iter()
        .find(|entry| entry.profile.as_ref() == Some(&request.requested))
    {
        return Ok((
            (*entry).clone(),
            selection(
                request,
                entry.profile.clone(),
                CookedTextureSelectionKind::Exact,
                None,
            ),
        ));
    }
    for (rank, fallback) in request.fallbacks.iter().enumerate() {
        if let Some(entry) = candidates
            .iter()
            .find(|entry| entry.profile.as_ref() == Some(fallback))
        {
            return Ok((
                (*entry).clone(),
                selection(
                    request,
                    entry.profile.clone(),
                    CookedTextureSelectionKind::Fallback,
                    Some(rank as u32),
                ),
            ));
        }
    }
    if request.allow_unprofiled {
        if let Some(entry) = candidates.iter().find(|entry| entry.profile.is_none()) {
            return Ok((
                (*entry).clone(),
                selection(
                    request,
                    None,
                    CookedTextureSelectionKind::UnprofiledFallback,
                    None,
                ),
            ));
        }
    }
    let mut available = candidates
        .iter()
        .filter_map(|entry| entry.profile.as_ref().map(CookedTextureProfile::label))
        .collect::<Vec<_>>();
    available.sort();
    Err(CookedTextureStoreError::Resolution(format!(
        "logical texture {:?} has no allowed variant for {}; available profiles: {}",
        request.logical_id,
        request.requested.label(),
        if available.is_empty() {
            "none".to_string()
        } else {
            available.join(", ")
        }
    )))
}

fn selection(
    request: &CookedTextureStoreRequest,
    selected_profile: Option<CookedTextureProfile>,
    kind: CookedTextureSelectionKind,
    fallback_rank: Option<u32>,
) -> CookedTextureStoreSelection {
    let reason = match (kind, request.policy) {
        (CookedTextureSelectionKind::Exact, CookedTextureRequestPolicy::Explicit) => {
            "requested-profile"
        }
        (
            CookedTextureSelectionKind::Exact,
            CookedTextureRequestPolicy::Adapter {
                native_profile_selected: true,
                ..
            },
        ) => "adapter-native-profile",
        (CookedTextureSelectionKind::Exact, CookedTextureRequestPolicy::Adapter { .. }) => {
            "adapter-portable-profile"
        }
        (CookedTextureSelectionKind::Fallback, CookedTextureRequestPolicy::Explicit) => {
            "ordered-explicit-fallback"
        }
        (CookedTextureSelectionKind::Fallback, CookedTextureRequestPolicy::Adapter { .. }) => {
            "portable-fallback-after-native-miss"
        }
        (CookedTextureSelectionKind::UnprofiledFallback, _) => "explicit-unprofiled-fallback",
    };
    CookedTextureStoreSelection {
        kind,
        requested_profile: request.requested.clone(),
        selected_profile,
        fallback_rank,
        reason,
        request_policy: request.policy,
    }
}

fn validate_request(request: &CookedTextureStoreRequest) -> Result<(), CookedTextureStoreError> {
    validate_logical_id(&request.logical_id)?;
    let mut profiles = BTreeSet::new();
    profiles.insert(request.requested.clone());
    for fallback in &request.fallbacks {
        if !profiles.insert(fallback.clone()) {
            return Err(CookedTextureStoreError::InvalidRequest(format!(
                "asset resolution profile {} is duplicated",
                fallback.label()
            )));
        }
    }
    if let CookedTextureRequestPolicy::Adapter {
        runtime_platform,
        bc_supported,
        native_profile_selected,
    } = request.policy
    {
        let expected = AdapterAssetProfilePlan::from_bc_support(bc_supported);
        let expected_fallbacks = if expected.has_portable_fallback() {
            vec![CookedTextureProfile::new(
                "portable",
                request.requested.quality(),
            )?]
        } else {
            Vec::new()
        };
        if runtime_platform != expected.runtime_platform()
            || native_profile_selected != expected.native_profile_selected()
            || request.requested.platform() != expected.selected_platform()
            || request.fallbacks != expected_fallbacks
            || request.allow_unprofiled
        {
            return Err(CookedTextureStoreError::InvalidRequest(
                "adapter-owned texture request was mutated after capability selection".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_selection_format(
    selection: &CookedTextureStoreSelection,
    format: CookedTextureArtifactFormat,
) -> Result<(), CookedTextureStoreError> {
    let CookedTextureRequestPolicy::Adapter { bc_supported, .. } = selection.request_policy else {
        return Ok(());
    };
    if format.requires_bc() && !bc_supported {
        return Err(CookedTextureStoreError::Artifact(format!(
            "adapter-owned selection chose BC format {} without accepted BC support",
            format.name()
        )));
    }
    if selection
        .selected_profile
        .as_ref()
        .is_some_and(|profile| profile.platform() == "portable")
        && format.requires_bc()
    {
        return Err(CookedTextureStoreError::Artifact(format!(
            "portable texture profile contains non-portable format {}",
            format.name()
        )));
    }
    Ok(())
}

fn validate_dds(bytes: &[u8], entry: &IndexedTextureEntry) -> Result<Dds, CookedTextureStoreError> {
    let dds = Dds::read(Cursor::new(bytes))
        .map_err(|error| CookedTextureStoreError::Artifact(format!("parse DDS: {error}")))?;
    if dds.get_dxgi_format() != Some(entry.format.dxgi()) {
        return Err(CookedTextureStoreError::Artifact(format!(
            "DDS format {:?} does not match indexed {}",
            dds.get_dxgi_format(),
            entry.format.name()
        )));
    }
    if dds.get_width() != entry.width
        || dds.get_height() != entry.height
        || dds.get_num_mipmap_levels() != entry.mip_levels
        || dds.get_depth() != 1
        || dds.get_num_array_layers() != 1
    {
        return Err(CookedTextureStoreError::Artifact(
            "DDS dimensions, mip count, depth, or layers disagree with the index".to_string(),
        ));
    }
    image_dds::Surface::from_dds(&dds).map_err(|error| {
        CookedTextureStoreError::Artifact(format!("validate DDS surface layout: {error}"))
    })?;
    Ok(dds)
}

fn parse_profile(value: &Value) -> Result<CookedTextureProfile, CookedTextureStoreError> {
    let object = value.as_object().ok_or_else(|| {
        CookedTextureStoreError::Index("asset profile is not an object".to_string())
    })?;
    CookedTextureProfile::new(
        required_string(object, "platform")?,
        required_string(object, "quality")?,
    )
}

fn validate_profile_component(value: &str, label: &str) -> Result<(), CookedTextureStoreError> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(CookedTextureStoreError::InvalidRequest(format!(
            "asset {label} {value:?} is not canonical"
        )));
    }
    Ok(())
}

fn validate_logical_id(value: &str) -> Result<(), CookedTextureStoreError> {
    if value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err(CookedTextureStoreError::InvalidRequest(format!(
            "logical asset ID {value:?} is not canonical"
        )));
    }
    Ok(())
}

fn validate_chunk_path(path: &Path, hash: [u8; 32]) -> Result<(), CookedTextureStoreError> {
    let expected = PathBuf::from("chunks")
        .join("sha256")
        .join(format!("{}.dds", hex_hash(hash)));
    if path != expected {
        return Err(CookedTextureStoreError::Index(format!(
            "texture path {:?} is non-canonical; expected {:?}",
            path, expected
        )));
    }
    Ok(())
}

fn reject_symlink_path(store: &Path, relative: &Path) -> Result<(), CookedTextureStoreError> {
    let mut current = store.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(CookedTextureStoreError::Index(
                "texture path is not relative and canonical".to_string(),
            ));
        };
        current.push(component);
        let metadata = std::fs::symlink_metadata(&current).map_err(|error| {
            CookedTextureStoreError::Io(format!("inspect {}: {error}", current.display()))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(CookedTextureStoreError::Io(format!(
                "asset store path {} is a symlink",
                current.display()
            )));
        }
    }
    Ok(())
}

fn read_bounded(
    path: &Path,
    maximum_bytes: u64,
    label: &str,
) -> Result<Vec<u8>, CookedTextureStoreError> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        CookedTextureStoreError::Io(format!("inspect {label} {}: {error}", path.display()))
    })?;
    if !metadata.is_file() {
        return Err(CookedTextureStoreError::Io(format!(
            "{label} {} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > maximum_bytes {
        return Err(CookedTextureStoreError::Io(format!(
            "{label} {} is {} bytes, limit is {maximum_bytes}",
            path.display(),
            metadata.len()
        )));
    }
    std::fs::read(path).map_err(|error| {
        CookedTextureStoreError::Io(format!("read {label} {}: {error}", path.display()))
    })
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a str, CookedTextureStoreError> {
    object.get(field).and_then(Value::as_str).ok_or_else(|| {
        CookedTextureStoreError::Index(format!("asset index field {field:?} is missing"))
    })
}

fn required_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<u64, CookedTextureStoreError> {
    object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        CookedTextureStoreError::Index(format!("asset index field {field:?} is missing"))
    })
}

fn required_u32(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<u32, CookedTextureStoreError> {
    u32::try_from(required_u64(object, field)?).map_err(|_| {
        CookedTextureStoreError::Index(format!("asset index field {field:?} exceeds u32"))
    })
}

fn parse_hash(value: &str, label: &str) -> Result<[u8; 32], CookedTextureStoreError> {
    if value.len() != 64
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CookedTextureStoreError::Index(format!(
            "{label} is not a lowercase SHA-256 digest"
        )));
    }
    let mut hash = [0u8; 32];
    for (index, byte) in hash.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            CookedTextureStoreError::Index(format!("{label} is not a SHA-256 digest"))
        })?;
    }
    Ok(hash)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hex_hash(hash: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in hash {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CookedTextureStoreError {
    InvalidRequest(String),
    Index(String),
    Resolution(String),
    Io(String),
    Artifact(String),
    Upload(String),
    QueueFull,
    WorkerStopped,
}

impl fmt::Display for CookedTextureStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(error) => write!(formatter, "invalid texture request: {error}"),
            Self::Index(error) => write!(formatter, "invalid asset index: {error}"),
            Self::Resolution(error) => write!(formatter, "texture resolution failed: {error}"),
            Self::Io(error) => write!(formatter, "texture store I/O failed: {error}"),
            Self::Artifact(error) => write!(formatter, "texture validation failed: {error}"),
            Self::Upload(error) => write!(formatter, "texture upload failed: {error}"),
            Self::QueueFull => write!(formatter, "cooked-texture store queue is full"),
            Self::WorkerStopped => write!(formatter, "cooked-texture store worker stopped"),
        }
    }
}

impl std::error::Error for CookedTextureStoreError {}

#[cfg(test)]
#[path = "cooked_texture_store_tests.rs"]
mod tests;
