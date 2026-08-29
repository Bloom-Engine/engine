//! Non-blocking native resolution of cooked virtual-geometry store entries.
//!
//! Runtime lookup deliberately consumes only the deterministic `index.json`
//! and immutable chunk it names. Source files and cooker manifests are not a
//! shipping dependency. All filesystem access, JSON parsing, hashing, and
//! archive validation happen on the worker thread; `request` and `poll` never
//! wait for storage.

use super::{ArtifactIdentity, VirtualGeometryAsset, VirtualGeometryLoadError};
use bloom_geometry_format::hex_hash;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;

const INDEX_FILE: &str = "index.json";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VirtualGeometryAssetProfile {
    platform: String,
    quality: String,
}

impl VirtualGeometryAssetProfile {
    pub fn new(platform: &str, quality: &str) -> Result<Self, VirtualGeometryStoreError> {
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
pub enum VirtualGeometryStoreRequestPolicy {
    Explicit,
    Adapter {
        runtime_platform: &'static str,
        bc_supported: bool,
        native_profile_selected: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryStoreRequest {
    pub logical_id: String,
    pub requested: VirtualGeometryAssetProfile,
    pub fallbacks: Vec<VirtualGeometryAssetProfile>,
    pub allow_unprofiled: bool,
    policy: VirtualGeometryStoreRequestPolicy,
}

impl VirtualGeometryStoreRequest {
    pub fn new(logical_id: impl Into<String>, requested: VirtualGeometryAssetProfile) -> Self {
        Self {
            logical_id: logical_id.into(),
            requested,
            fallbacks: Vec::new(),
            allow_unprofiled: false,
            policy: VirtualGeometryStoreRequestPolicy::Explicit,
        }
    }

    /// Build the canonical package request from the renderer's accepted
    /// device features. Desktop BC devices request their native platform and
    /// carry one explicit portable fallback. Devices without BC request the
    /// capability-neutral portable package directly.
    pub fn for_device(
        logical_id: impl Into<String>,
        quality: &str,
        device: &wgpu::Device,
    ) -> Result<Self, VirtualGeometryStoreError> {
        Self::for_runtime_features(logical_id, quality, device.features())
    }

    fn for_runtime_features(
        logical_id: impl Into<String>,
        quality: &str,
        features: wgpu::Features,
    ) -> Result<Self, VirtualGeometryStoreError> {
        let runtime_platform = runtime_platform_profile();
        let bc_supported = features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        let native_profile_selected = desktop_bc_profile(runtime_platform) && bc_supported;
        let selected_platform = if native_profile_selected {
            runtime_platform
        } else {
            "portable"
        };
        let requested = VirtualGeometryAssetProfile::new(selected_platform, quality)?;
        let mut request = Self::new(logical_id, requested);
        request.policy = VirtualGeometryStoreRequestPolicy::Adapter {
            runtime_platform,
            bc_supported,
            native_profile_selected,
        };
        if native_profile_selected {
            request
                .fallbacks
                .push(VirtualGeometryAssetProfile::new("portable", quality)?);
        }
        Ok(request)
    }

    pub fn with_fallback(mut self, fallback: VirtualGeometryAssetProfile) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    pub fn allow_unprofiled(mut self, allow: bool) -> Self {
        self.allow_unprofiled = allow;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualGeometrySelectionKind {
    Exact,
    Fallback,
    UnprofiledFallback,
}

impl VirtualGeometrySelectionKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Fallback => "fallback",
            Self::UnprofiledFallback => "unprofiled-fallback",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualGeometryStoreSelection {
    pub kind: VirtualGeometrySelectionKind,
    pub requested_profile: VirtualGeometryAssetProfile,
    pub selected_profile: Option<VirtualGeometryAssetProfile>,
    pub fallback_rank: Option<u32>,
    pub reason: &'static str,
    pub request_policy: VirtualGeometryStoreRequestPolicy,
}

impl VirtualGeometryStoreSelection {
    fn as_json(&self) -> Value {
        let policy = match self.request_policy {
            VirtualGeometryStoreRequestPolicy::Explicit => {
                serde_json::json!({"kind": "explicit"})
            }
            VirtualGeometryStoreRequestPolicy::Adapter {
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
            "selected_profile": self.selected_profile.as_ref().map(VirtualGeometryAssetProfile::as_json),
        })
    }

    pub fn report_json(&self) -> String {
        self.as_json().to_string()
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedVirtualGeometryAsset {
    pub logical_id: String,
    pub selection: VirtualGeometryStoreSelection,
    pub artifact_path: PathBuf,
    pub artifact_bytes: u64,
    pub asset: Arc<VirtualGeometryAsset>,
}

impl ResolvedVirtualGeometryAsset {
    pub fn report_json(&self) -> String {
        serde_json::json!({
            "artifact_bytes": self.artifact_bytes,
            "artifact_path": self.artifact_path.display().to_string(),
            "logical_id": self.logical_id,
            "schema": "bloom-runtime-asset-selection-v1",
            "selection": self.selection.as_json(),
        })
        .to_string()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct VirtualGeometryStoreTicket(u64);

impl VirtualGeometryStoreTicket {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualGeometryStoreConfig {
    pub max_pending_requests: u32,
    pub max_index_bytes: u64,
    pub max_artifact_bytes: u64,
}

impl Default for VirtualGeometryStoreConfig {
    fn default() -> Self {
        Self {
            max_pending_requests: 32,
            max_index_bytes: 16 * 1024 * 1024,
            max_artifact_bytes: 16 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VirtualGeometryStoreTelemetry {
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

/// One bounded native worker for runtime store lookup and archive validation.
///
/// The worker is intentionally independent of the renderer. A simple caller
/// can request an asset, poll it during ordinary updates, and register the
/// completed immutable asset with the existing virtual-geometry pool.
pub struct VirtualGeometryStoreLoader {
    request_tx: SyncSender<WorkerRequest>,
    completion_rx: Receiver<WorkerCompletion>,
    completed: BTreeMap<VirtualGeometryStoreTicket, WorkerCompletion>,
    max_outstanding_requests: u32,
    next_ticket: u64,
    telemetry: VirtualGeometryStoreTelemetry,
}

impl VirtualGeometryStoreLoader {
    pub fn new(
        store: impl Into<PathBuf>,
        config: VirtualGeometryStoreConfig,
    ) -> Result<Self, VirtualGeometryStoreError> {
        if config.max_pending_requests == 0
            || config.max_index_bytes == 0
            || config.max_artifact_bytes == 0
        {
            return Err(VirtualGeometryStoreError::InvalidRequest(
                "store loader budgets must all be non-zero".to_string(),
            ));
        }
        let (request_tx, request_rx) =
            mpsc::sync_channel::<WorkerRequest>(config.max_pending_requests as usize);
        let (completion_tx, completion_rx) = mpsc::channel::<WorkerCompletion>();
        let store = store.into();
        std::thread::Builder::new()
            .name("bloom-vg-store".to_string())
            .spawn(move || worker_main(&store, config, request_rx, completion_tx))
            .map_err(|error| {
                VirtualGeometryStoreError::Io(format!("start virtual-geometry worker: {error}"))
            })?;
        Ok(Self {
            request_tx,
            completion_rx,
            completed: BTreeMap::new(),
            max_outstanding_requests: config.max_pending_requests,
            next_ticket: 1,
            telemetry: VirtualGeometryStoreTelemetry::default(),
        })
    }

    /// Queue one lookup without waiting for filesystem capacity or the worker.
    pub fn request(
        &mut self,
        request: VirtualGeometryStoreRequest,
    ) -> Result<VirtualGeometryStoreTicket, VirtualGeometryStoreError> {
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
            return Err(VirtualGeometryStoreError::QueueFull);
        }
        let ticket = VirtualGeometryStoreTicket(self.next_ticket);
        let worker_request = WorkerRequest { ticket, request };
        match self.request_tx.try_send(worker_request) {
            Ok(()) => {
                self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
                self.telemetry.queued_requests = self.telemetry.queued_requests.saturating_add(1);
                self.telemetry.pending_requests = self.telemetry.pending_requests.saturating_add(1);
                Ok(ticket)
            }
            Err(TrySendError::Full(_)) => {
                self.telemetry.queue_full_rejections =
                    self.telemetry.queue_full_rejections.saturating_add(1);
                Err(VirtualGeometryStoreError::QueueFull)
            }
            Err(TrySendError::Disconnected(_)) => Err(VirtualGeometryStoreError::WorkerStopped),
        }
    }

    /// Poll one ticket without blocking. Completions for other tickets remain
    /// retained until their caller polls them.
    pub fn poll(
        &mut self,
        ticket: VirtualGeometryStoreTicket,
    ) -> Option<Result<ResolvedVirtualGeometryAsset, VirtualGeometryStoreError>> {
        self.drain_completions();
        self.completed
            .remove(&ticket)
            .map(|completion| completion.result)
    }

    pub fn telemetry(&mut self) -> VirtualGeometryStoreTelemetry {
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
                            if resolved.selection.request_policy
                                != VirtualGeometryStoreRequestPolicy::Explicit
                                || resolved.selection.kind != VirtualGeometrySelectionKind::Exact
                            {
                                log::info!(
                                    "bloom: cooked asset selection = {}",
                                    resolved.report_json()
                                );
                            }
                            self.telemetry.loaded_artifact_bytes = self
                                .telemetry
                                .loaded_artifact_bytes
                                .saturating_add(resolved.artifact_bytes);
                            match resolved.selection.kind {
                                VirtualGeometrySelectionKind::Exact => {
                                    self.telemetry.exact_selections =
                                        self.telemetry.exact_selections.saturating_add(1);
                                }
                                VirtualGeometrySelectionKind::Fallback => {
                                    self.telemetry.fallback_selections =
                                        self.telemetry.fallback_selections.saturating_add(1);
                                }
                                VirtualGeometrySelectionKind::UnprofiledFallback => {
                                    self.telemetry.unprofiled_selections =
                                        self.telemetry.unprofiled_selections.saturating_add(1);
                                }
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
    ticket: VirtualGeometryStoreTicket,
    request: VirtualGeometryStoreRequest,
}

struct WorkerCompletion {
    ticket: VirtualGeometryStoreTicket,
    result: Result<ResolvedVirtualGeometryAsset, VirtualGeometryStoreError>,
}

fn worker_main(
    store: &Path,
    config: VirtualGeometryStoreConfig,
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

fn resolve_and_load(
    store: &Path,
    config: VirtualGeometryStoreConfig,
    request: &VirtualGeometryStoreRequest,
) -> Result<ResolvedVirtualGeometryAsset, VirtualGeometryStoreError> {
    let index_path = store.join(INDEX_FILE);
    let index_bytes = read_bounded(&index_path, config.max_index_bytes, "asset index")?;
    let index: Value = serde_json::from_slice(&index_bytes).map_err(|error| {
        VirtualGeometryStoreError::Index(format!(
            "parse asset index {}: {error}",
            index_path.display()
        ))
    })?;
    let entries = parse_index(&index)?;
    let (entry, selection) = select_entry(&entries, request)?;
    if entry.identity.bytes > config.max_artifact_bytes {
        return Err(VirtualGeometryStoreError::Io(format!(
            "artifact declares {} bytes, loader limit is {}",
            entry.identity.bytes, config.max_artifact_bytes
        )));
    }
    validate_chunk_path(&entry.artifact_path, entry.identity.file_sha256)?;
    let artifact_path = store.join(&entry.artifact_path);
    reject_symlink_path(store, &entry.artifact_path)?;
    let bytes = read_bounded(
        &artifact_path,
        config.max_artifact_bytes,
        "geometry artifact",
    )?;
    let asset =
        VirtualGeometryAsset::from_indexed_file_bytes(artifact_path.clone(), bytes, entry.identity)
            .map_err(VirtualGeometryStoreError::Asset)?;
    Ok(ResolvedVirtualGeometryAsset {
        logical_id: request.logical_id.clone(),
        selection,
        artifact_path,
        artifact_bytes: entry.identity.bytes,
        asset: Arc::new(asset),
    })
}

#[derive(Clone)]
struct IndexedEntry {
    logical_id: String,
    profile: Option<VirtualGeometryAssetProfile>,
    artifact_path: PathBuf,
    identity: ArtifactIdentity,
}

fn parse_index(index: &Value) -> Result<Vec<IndexedEntry>, VirtualGeometryStoreError> {
    let object = index.as_object().ok_or_else(|| {
        VirtualGeometryStoreError::Index("asset index is not an object".to_string())
    })?;
    let schema = object
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            VirtualGeometryStoreError::Index("asset index schema is missing".to_string())
        })?;
    if !matches!(schema, "bloom-asset-index-v1" | "bloom-asset-index-v2") {
        return Err(VirtualGeometryStoreError::Index(format!(
            "unsupported asset index schema {schema:?}; recook assets"
        )));
    }
    let values = object
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            VirtualGeometryStoreError::Index("asset index entries are missing".to_string())
        })?;
    let declared_count = object
        .get("entry_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            VirtualGeometryStoreError::Index("asset index entry_count is missing".to_string())
        })?;
    if declared_count != values.len() as u64 {
        return Err(VirtualGeometryStoreError::Index(format!(
            "asset index entry_count is {declared_count}, actual {}",
            values.len()
        )));
    }

    let mut entries = Vec::with_capacity(values.len());
    let mut identities = BTreeSet::new();
    for value in values {
        let object = value.as_object().ok_or_else(|| {
            VirtualGeometryStoreError::Index("asset index entry is not an object".to_string())
        })?;
        let kind = required_string(object, "kind")?;
        let logical_id = required_string(object, "logical_id")?.to_string();
        validate_logical_id(&logical_id)?;
        let profile = object.get("profile").map(parse_profile).transpose()?;
        let key = (logical_id, profile);
        if !identities.insert(key) {
            return Err(VirtualGeometryStoreError::Index(
                "asset index contains a duplicate logical ID/profile".to_string(),
            ));
        }
        match kind {
            "geometry" => entries.push(parse_entry(value)?),
            "texture" => {}
            other => {
                return Err(VirtualGeometryStoreError::Index(format!(
                    "asset index contains unsupported asset kind {other:?}"
                )))
            }
        }
    }
    Ok(entries)
}

fn parse_entry(value: &Value) -> Result<IndexedEntry, VirtualGeometryStoreError> {
    let object = value.as_object().ok_or_else(|| {
        VirtualGeometryStoreError::Index("asset index entry is not an object".to_string())
    })?;
    let logical_id = required_string(object, "logical_id")?.to_string();
    validate_logical_id(&logical_id)?;
    if required_string(object, "kind")? != "geometry" {
        return Err(VirtualGeometryStoreError::Index(format!(
            "logical asset {logical_id:?} is not geometry"
        )));
    }
    let profile = object.get("profile").map(parse_profile).transpose()?;
    let source_sha256 = parse_hash(required_string(object, "source_sha256")?, "source hash")?;
    let artifact = object
        .get("artifact")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            VirtualGeometryStoreError::Index(format!(
                "logical asset {logical_id:?} has no artifact object"
            ))
        })?;
    let artifact_path = PathBuf::from(required_string(artifact, "path")?);
    let bytes = required_u64(artifact, "bytes")?;
    let format_version =
        u32::try_from(required_u64(artifact, "format_version")?).map_err(|_| {
            VirtualGeometryStoreError::Index("artifact format_version exceeds u32".to_string())
        })?;
    let file_sha256 = parse_hash(required_string(artifact, "sha256")?, "artifact hash")?;
    let payload_sha256 = parse_hash(required_string(artifact, "payload_sha256")?, "payload hash")?;
    Ok(IndexedEntry {
        logical_id,
        profile,
        artifact_path,
        identity: ArtifactIdentity {
            bytes,
            format_version,
            file_sha256,
            payload_sha256,
            source_sha256,
        },
    })
}

fn select_entry(
    entries: &[IndexedEntry],
    request: &VirtualGeometryStoreRequest,
) -> Result<(IndexedEntry, VirtualGeometryStoreSelection), VirtualGeometryStoreError> {
    let candidates = entries
        .iter()
        .filter(|entry| entry.logical_id == request.logical_id)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(VirtualGeometryStoreError::Resolution(format!(
            "logical asset {:?} is not in the cooked index",
            request.logical_id
        )));
    }
    if let Some(entry) = candidates
        .iter()
        .find(|entry| entry.profile.as_ref() == Some(&request.requested))
    {
        return Ok((
            (*entry).clone(),
            VirtualGeometryStoreSelection {
                kind: VirtualGeometrySelectionKind::Exact,
                requested_profile: request.requested.clone(),
                selected_profile: entry.profile.clone(),
                fallback_rank: None,
                reason: match request.policy {
                    VirtualGeometryStoreRequestPolicy::Explicit => "requested-profile",
                    VirtualGeometryStoreRequestPolicy::Adapter {
                        native_profile_selected: true,
                        ..
                    } => "adapter-native-profile",
                    VirtualGeometryStoreRequestPolicy::Adapter {
                        native_profile_selected: false,
                        ..
                    } => "adapter-portable-profile",
                },
                request_policy: request.policy,
            },
        ));
    }
    for (rank, fallback) in request.fallbacks.iter().enumerate() {
        if let Some(entry) = candidates
            .iter()
            .find(|entry| entry.profile.as_ref() == Some(fallback))
        {
            return Ok((
                (*entry).clone(),
                VirtualGeometryStoreSelection {
                    kind: VirtualGeometrySelectionKind::Fallback,
                    requested_profile: request.requested.clone(),
                    selected_profile: entry.profile.clone(),
                    fallback_rank: Some(rank as u32),
                    reason: match request.policy {
                        VirtualGeometryStoreRequestPolicy::Explicit => "ordered-explicit-fallback",
                        VirtualGeometryStoreRequestPolicy::Adapter { .. } => {
                            "portable-fallback-after-native-miss"
                        }
                    },
                    request_policy: request.policy,
                },
            ));
        }
    }
    if request.allow_unprofiled {
        if let Some(entry) = candidates.iter().find(|entry| entry.profile.is_none()) {
            return Ok((
                (*entry).clone(),
                VirtualGeometryStoreSelection {
                    kind: VirtualGeometrySelectionKind::UnprofiledFallback,
                    requested_profile: request.requested.clone(),
                    selected_profile: None,
                    fallback_rank: None,
                    reason: "explicit-unprofiled-fallback",
                    request_policy: request.policy,
                },
            ));
        }
    }
    let mut available = candidates
        .iter()
        .filter_map(|entry| entry.profile.as_ref().map(|profile| profile.label()))
        .collect::<Vec<_>>();
    available.sort();
    Err(VirtualGeometryStoreError::Resolution(format!(
        "logical asset {:?} has no allowed variant for {}; available profiles: {}",
        request.logical_id,
        request.requested.label(),
        if available.is_empty() {
            "none".to_string()
        } else {
            available.join(", ")
        }
    )))
}

fn validate_request(
    request: &VirtualGeometryStoreRequest,
) -> Result<(), VirtualGeometryStoreError> {
    validate_logical_id(&request.logical_id)?;
    let mut profiles = BTreeSet::new();
    profiles.insert(request.requested.clone());
    for fallback in &request.fallbacks {
        if !profiles.insert(fallback.clone()) {
            return Err(VirtualGeometryStoreError::InvalidRequest(format!(
                "asset resolution profile {} is duplicated",
                fallback.label()
            )));
        }
    }
    if let VirtualGeometryStoreRequestPolicy::Adapter {
        runtime_platform,
        bc_supported,
        native_profile_selected,
    } = request.policy
    {
        let expected_native = desktop_bc_profile(runtime_platform) && bc_supported;
        let expected_platform = if expected_native {
            runtime_platform
        } else {
            "portable"
        };
        let expected_fallbacks = if expected_native {
            vec![VirtualGeometryAssetProfile::new(
                "portable",
                request.requested.quality(),
            )?]
        } else {
            Vec::new()
        };
        if runtime_platform != runtime_platform_profile()
            || native_profile_selected != expected_native
            || request.requested.platform() != expected_platform
            || request.fallbacks != expected_fallbacks
            || request.allow_unprofiled
        {
            return Err(VirtualGeometryStoreError::InvalidRequest(
                "adapter-owned asset request was mutated after capability selection".to_string(),
            ));
        }
    }
    Ok(())
}

const fn runtime_platform_profile() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "ios") {
        "ios"
    } else if cfg!(target_os = "tvos") {
        "tvos"
    } else if cfg!(target_os = "visionos") {
        "visionos"
    } else {
        "portable"
    }
}

fn desktop_bc_profile(platform: &str) -> bool {
    matches!(platform, "macos" | "windows" | "linux")
}

fn validate_logical_id(value: &str) -> Result<(), VirtualGeometryStoreError> {
    if value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.split('/').any(|part| {
            part.is_empty()
                || matches!(part, "." | "..")
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err(VirtualGeometryStoreError::InvalidRequest(format!(
            "logical asset ID {value:?} is not canonical"
        )));
    }
    Ok(())
}

fn validate_profile_component(value: &str, label: &str) -> Result<(), VirtualGeometryStoreError> {
    if value.is_empty()
        || value.len() > 32
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
    {
        return Err(VirtualGeometryStoreError::InvalidRequest(format!(
            "asset {label} profile {value:?} is not canonical"
        )));
    }
    Ok(())
}

fn parse_profile(value: &Value) -> Result<VirtualGeometryAssetProfile, VirtualGeometryStoreError> {
    let object = value.as_object().ok_or_else(|| {
        VirtualGeometryStoreError::Index("asset profile is not an object".to_string())
    })?;
    if object.len() != 2 {
        return Err(VirtualGeometryStoreError::Index(
            "asset profile has unknown or missing fields".to_string(),
        ));
    }
    VirtualGeometryAssetProfile::new(
        required_string(object, "platform")?,
        required_string(object, "quality")?,
    )
    .map_err(|error| VirtualGeometryStoreError::Index(error.to_string()))
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a str, VirtualGeometryStoreError> {
    object.get(field).and_then(Value::as_str).ok_or_else(|| {
        VirtualGeometryStoreError::Index(format!("asset index field {field:?} is missing"))
    })
}

fn required_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<u64, VirtualGeometryStoreError> {
    object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        VirtualGeometryStoreError::Index(format!("asset index field {field:?} is missing"))
    })
}

#[cfg(test)]
mod request_policy_tests {
    use super::*;

    #[test]
    fn automatic_policy_selects_native_bc_or_portable_without_silent_fallback() {
        let native = VirtualGeometryStoreRequest::for_runtime_features(
            "city/bistro",
            "high",
            wgpu::Features::TEXTURE_COMPRESSION_BC,
        )
        .unwrap();
        if desktop_bc_profile(runtime_platform_profile()) {
            assert_eq!(native.requested.platform(), runtime_platform_profile());
            assert_eq!(native.fallbacks.len(), 1);
            assert_eq!(native.fallbacks[0].label(), "portable/high");
        } else {
            assert_eq!(native.requested.label(), "portable/high");
            assert!(native.fallbacks.is_empty());
        }

        let portable = VirtualGeometryStoreRequest::for_runtime_features(
            "city/bistro",
            "high",
            wgpu::Features::empty(),
        )
        .unwrap();
        assert_eq!(portable.requested.label(), "portable/high");
        assert!(portable.fallbacks.is_empty());
        assert_eq!(
            portable.policy,
            VirtualGeometryStoreRequestPolicy::Adapter {
                runtime_platform: runtime_platform_profile(),
                bc_supported: false,
                native_profile_selected: false,
            }
        );

        let entry = IndexedEntry {
            logical_id: "city/bistro".to_string(),
            profile: Some(VirtualGeometryAssetProfile::new("portable", "high").unwrap()),
            artifact_path: PathBuf::from("unused.bgeo"),
            identity: ArtifactIdentity {
                bytes: 1,
                format_version: 2,
                file_sha256: [1; 32],
                payload_sha256: [2; 32],
                source_sha256: [3; 32],
            },
        };
        let (_, selection) = select_entry(&[entry], &native).unwrap();
        if desktop_bc_profile(runtime_platform_profile()) {
            assert_eq!(selection.kind, VirtualGeometrySelectionKind::Fallback);
            assert_eq!(selection.reason, "portable-fallback-after-native-miss");
            assert_eq!(selection.fallback_rank, Some(0));
        } else {
            assert_eq!(selection.kind, VirtualGeometrySelectionKind::Exact);
            assert_eq!(selection.reason, "adapter-portable-profile");
        }
        let report: Value = serde_json::from_str(&selection.report_json()).unwrap();
        assert_eq!(report["policy"]["kind"], "adapter");
        assert_eq!(report["selected_profile"]["platform"], "portable");

        let mut mutated = portable;
        mutated.allow_unprofiled = true;
        assert!(validate_request(&mutated)
            .unwrap_err()
            .to_string()
            .contains("mutated after capability selection"));
    }
}

fn parse_hash(value: &str, label: &str) -> Result<[u8; 32], VirtualGeometryStoreError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(VirtualGeometryStoreError::Index(format!(
            "{label} is not a lowercase SHA-256 digest"
        )));
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(VirtualGeometryStoreError::Index(format!(
            "{label} is not lowercase"
        )));
    }
    let mut hash = [0u8; 32];
    for (index, byte) in hash.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            VirtualGeometryStoreError::Index(format!("{label} is not a SHA-256 digest"))
        })?;
    }
    Ok(hash)
}

fn validate_chunk_path(path: &Path, hash: [u8; 32]) -> Result<(), VirtualGeometryStoreError> {
    let expected = PathBuf::from("chunks")
        .join("sha256")
        .join(format!("{}.bgeo", hex_hash(hash)));
    if path != expected {
        return Err(VirtualGeometryStoreError::Index(format!(
            "artifact path {:?} is non-canonical; expected {:?}",
            path, expected
        )));
    }
    Ok(())
}

fn reject_symlink_path(store: &Path, relative: &Path) -> Result<(), VirtualGeometryStoreError> {
    let mut current = store.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(VirtualGeometryStoreError::Index(
                "artifact path is not relative and canonical".to_string(),
            ));
        };
        current.push(component);
        let metadata = std::fs::symlink_metadata(&current).map_err(|error| {
            VirtualGeometryStoreError::Io(format!("inspect {}: {error}", current.display()))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(VirtualGeometryStoreError::Io(format!(
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
) -> Result<Vec<u8>, VirtualGeometryStoreError> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        VirtualGeometryStoreError::Io(format!("inspect {label} {}: {error}", path.display()))
    })?;
    if !metadata.is_file() {
        return Err(VirtualGeometryStoreError::Io(format!(
            "{label} {} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > maximum_bytes {
        return Err(VirtualGeometryStoreError::Io(format!(
            "{label} {} is {} bytes, limit is {maximum_bytes}",
            path.display(),
            metadata.len()
        )));
    }
    std::fs::read(path).map_err(|error| {
        VirtualGeometryStoreError::Io(format!("read {label} {}: {error}", path.display()))
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualGeometryStoreError {
    InvalidRequest(String),
    Index(String),
    Resolution(String),
    Io(String),
    Asset(VirtualGeometryLoadError),
    QueueFull,
    WorkerStopped,
}

impl fmt::Display for VirtualGeometryStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(error) => write!(formatter, "invalid store request: {error}"),
            Self::Index(error) => write!(formatter, "invalid asset index: {error}"),
            Self::Resolution(error) => write!(formatter, "asset resolution failed: {error}"),
            Self::Io(error) => write!(formatter, "asset store I/O failed: {error}"),
            Self::Asset(error) => write!(formatter, "asset validation failed: {error}"),
            Self::QueueFull => write!(formatter, "virtual-geometry store queue is full"),
            Self::WorkerStopped => write!(formatter, "virtual-geometry store worker stopped"),
        }
    }
}

impl std::error::Error for VirtualGeometryStoreError {}
