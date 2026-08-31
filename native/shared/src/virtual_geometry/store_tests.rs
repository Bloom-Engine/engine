use super::*;
use serde_json::{json, Value};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_STORE_TEST: AtomicU64 = AtomicU64::new(1);

#[test]
fn native_store_loader_resolves_explicit_profiles_without_blocking_poll() {
    let root = temporary_store("profiles");
    let asset = hierarchy_asset(hierarchy_archive());
    let artifact = install_artifact(&root, asset.file_bytes().unwrap());
    write_index(
        &root,
        vec![
            index_entry("city/bistro", Some(("macos", "high")), &artifact),
            index_entry("city/bistro", Some(("portable", "medium")), &artifact),
            index_entry("city/bistro", None, &artifact),
        ],
    );

    let mut loader = VirtualGeometryStoreLoader::new(
        &root,
        VirtualGeometryStoreConfig {
            max_pending_requests: 4,
            ..VirtualGeometryStoreConfig::default()
        },
    )
    .unwrap();
    let exact = loader
        .request(VirtualGeometryStoreRequest::new(
            "city/bistro",
            profile("macos", "high"),
        ))
        .unwrap();
    let fallback = loader
        .request(
            VirtualGeometryStoreRequest::new("city/bistro", profile("windows", "ultra"))
                .with_fallback(profile("portable", "medium")),
        )
        .unwrap();

    let exact = wait_for(&mut loader, exact).unwrap();
    assert_eq!(exact.selection.kind, VirtualGeometrySelectionKind::Exact);
    assert_eq!(
        exact.selection.selected_profile.unwrap().label(),
        "macos/high"
    );
    assert_eq!(exact.asset.archive(), asset.archive());
    assert!(exact.asset.is_file_backed());
    assert!(exact.asset.file_bytes().is_none());
    assert_eq!(exact.asset.page_bytes(0), asset.page_bytes(0));
    assert!(exact.asset.page_bytes(3).is_none());
    assert_eq!(
        exact.asset.read_page_owned(3).unwrap(),
        asset.page_bytes(3).unwrap()
    );

    let fallback = wait_for(&mut loader, fallback).unwrap();
    assert_eq!(
        fallback.selection.kind,
        VirtualGeometrySelectionKind::Fallback
    );
    assert_eq!(fallback.selection.fallback_rank, Some(0));
    assert_eq!(
        fallback.selection.selected_profile.unwrap().label(),
        "portable/medium"
    );
    let telemetry = loader.telemetry();
    assert_eq!(telemetry.pending_requests, 0);
    assert_eq!(telemetry.completed_requests, 2);
    assert_eq!(telemetry.exact_selections, 1);
    assert_eq!(telemetry.fallback_selections, 1);
    assert_eq!(telemetry.failed_requests, 0);
    assert_eq!(
        telemetry.loaded_artifact_bytes,
        2 * asset.file_bytes().unwrap().len() as u64
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_geometry_store_ignores_other_valid_asset_kinds() {
    let root = temporary_store("mixed-kinds");
    let asset = hierarchy_asset(hierarchy_archive());
    let artifact = install_artifact(&root, asset.file_bytes().unwrap());
    let texture = json!({
        "artifact": {
            "bytes": 256,
            "format": "bc7-rgba-unorm-srgb",
            "height": 8,
            "mip_levels": 4,
            "path": format!("chunks/sha256/{}.dds", hex_hash([3; 32])),
            "sha256": hex_hash([3; 32]),
            "width": 8,
        },
        "build_key_sha256": hex_hash([5; 32]),
        "kind": "texture",
        "logical_id": "textures/cobble",
        "manifest": {
            "path": "manifests/textures/cobble.json",
            "sha256": hex_hash([9; 32]),
        },
        "source_sha256": hex_hash([13; 32]),
    });
    write_index(
        &root,
        vec![texture.clone(), index_entry("city/bistro", None, &artifact)],
    );

    let mut loader =
        VirtualGeometryStoreLoader::new(&root, VirtualGeometryStoreConfig::default()).unwrap();
    let ticket = loader
        .request(
            VirtualGeometryStoreRequest::new("city/bistro", profile("portable", "high"))
                .allow_unprofiled(true),
        )
        .unwrap();
    assert_eq!(
        wait_for(&mut loader, ticket).unwrap().asset.archive(),
        asset.archive()
    );

    drop(loader);
    let mut unknown = texture;
    unknown["kind"] = json!("unknown-future-kind");
    write_index(
        &root,
        vec![unknown, index_entry("city/bistro", None, &artifact)],
    );
    let mut loader =
        VirtualGeometryStoreLoader::new(&root, VirtualGeometryStoreConfig::default()).unwrap();
    let ticket = loader
        .request(
            VirtualGeometryStoreRequest::new("city/bistro", profile("portable", "high"))
                .allow_unprofiled(true),
        )
        .unwrap();
    assert!(matches!(
        wait_for(&mut loader, ticket),
        Err(VirtualGeometryStoreError::Index(message))
            if message.contains("unsupported asset kind")
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_store_loader_requires_opt_in_for_unprofiled_and_fails_closed() {
    let root = temporary_store("unprofiled");
    let asset = hierarchy_asset(hierarchy_archive());
    let artifact = install_artifact(&root, asset.file_bytes().unwrap());
    write_index(&root, vec![index_entry("props/bench", None, &artifact)]);
    let mut loader =
        VirtualGeometryStoreLoader::new(&root, VirtualGeometryStoreConfig::default()).unwrap();

    let denied = loader
        .request(VirtualGeometryStoreRequest::new(
            "props/bench",
            profile("macos", "high"),
        ))
        .unwrap();
    assert!(matches!(
        wait_for(&mut loader, denied),
        Err(VirtualGeometryStoreError::Resolution(_))
    ));
    let allowed = loader
        .request(
            VirtualGeometryStoreRequest::new("props/bench", profile("macos", "high"))
                .allow_unprofiled(true),
        )
        .unwrap();
    assert_eq!(
        wait_for(&mut loader, allowed).unwrap().selection.kind,
        VirtualGeometrySelectionKind::UnprofiledFallback
    );
    let telemetry = loader.telemetry();
    assert_eq!(telemetry.failed_requests, 1);
    assert_eq!(telemetry.unprofiled_selections, 1);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_store_loader_rejects_corrupt_index_selected_chunks() {
    let root = temporary_store("corrupt");
    let asset = hierarchy_asset(hierarchy_archive());
    let mut corrupt = asset.file_bytes().unwrap().to_vec();
    *corrupt.last_mut().unwrap() ^= 0x80;
    let artifact = install_artifact(&root, &corrupt);
    let mut entry = index_entry("props/corrupt", Some(("macos", "high")), &artifact);
    entry["artifact"]["payload_sha256"] = json!(hex_hash(asset.archive().payload_sha256));
    entry["source_sha256"] = json!(hex_hash(asset.archive().source_sha256));
    write_index(&root, vec![entry]);
    let mut loader =
        VirtualGeometryStoreLoader::new(&root, VirtualGeometryStoreConfig::default()).unwrap();
    let ticket = loader
        .request(VirtualGeometryStoreRequest::new(
            "props/corrupt",
            profile("macos", "high"),
        ))
        .unwrap();
    assert!(matches!(
        wait_for(&mut loader, ticket),
        Err(VirtualGeometryStoreError::Asset(
            VirtualGeometryLoadError::Format(_)
        ))
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_store_loader_bounds_unclaimed_completions() {
    let root = temporary_store("bounded");
    let asset = hierarchy_asset(hierarchy_archive());
    let artifact = install_artifact(&root, asset.file_bytes().unwrap());
    write_index(
        &root,
        vec![index_entry("props/box", Some(("macos", "high")), &artifact)],
    );
    let mut loader = VirtualGeometryStoreLoader::new(
        &root,
        VirtualGeometryStoreConfig {
            max_pending_requests: 1,
            ..VirtualGeometryStoreConfig::default()
        },
    )
    .unwrap();
    let first = loader
        .request(VirtualGeometryStoreRequest::new(
            "props/box",
            profile("macos", "high"),
        ))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while loader.telemetry().pending_requests != 0 {
        assert!(Instant::now() < deadline, "store worker timed out");
        std::thread::yield_now();
    }
    assert!(matches!(
        loader.request(VirtualGeometryStoreRequest::new(
            "props/box",
            profile("macos", "high")
        )),
        Err(VirtualGeometryStoreError::QueueFull)
    ));
    assert!(loader.poll(first).unwrap().is_ok());
    let second = loader
        .request(VirtualGeometryStoreRequest::new(
            "props/box",
            profile("macos", "high"),
        ))
        .unwrap();
    assert!(wait_for(&mut loader, second).is_ok());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn file_backed_pages_reject_damage_after_initial_validation() {
    let root = temporary_store("page-damage");
    let memory = hierarchy_asset(hierarchy_archive());
    let artifact = install_artifact(&root, memory.file_bytes().unwrap());
    write_index(
        &root,
        vec![index_entry(
            "props/arch",
            Some(("macos", "high")),
            &artifact,
        )],
    );
    let mut loader =
        VirtualGeometryStoreLoader::new(&root, VirtualGeometryStoreConfig::default()).unwrap();
    let ticket = loader
        .request(VirtualGeometryStoreRequest::new(
            "props/arch",
            profile("macos", "high"),
        ))
        .unwrap();
    let resolved = wait_for(&mut loader, ticket).unwrap();
    let range = resolved.asset.page_file_range(3).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&resolved.artifact_path)
        .unwrap();
    file.seek(SeekFrom::Start(range.start as u64)).unwrap();
    file.write_all(&[0xff]).unwrap();
    file.flush().unwrap();

    assert!(matches!(
        resolved.asset.read_page_owned(3),
        Err(VirtualGeometryLoadError::Identity(_))
    ));
    assert_eq!(resolved.asset.page_bytes(0), memory.page_bytes(0));
    std::fs::remove_dir_all(root).unwrap();
}

struct InstalledArtifact {
    bytes: u64,
    format_version: u32,
    sha256: [u8; 32],
    payload_sha256: [u8; 32],
    source_sha256: [u8; 32],
    path: String,
}

fn install_artifact(root: &Path, bytes: &[u8]) -> InstalledArtifact {
    let archive = bloom_geometry_format::decode_geometry(bytes);
    let file_hash = sha256(bytes);
    let path = format!("chunks/sha256/{}.bgeo", hex_hash(file_hash));
    std::fs::create_dir_all(root.join("chunks/sha256")).unwrap();
    std::fs::write(root.join(&path), bytes).unwrap();
    match archive {
        Ok(archive) => InstalledArtifact {
            bytes: bytes.len() as u64,
            format_version: archive.format_version,
            sha256: file_hash,
            payload_sha256: archive.payload_sha256,
            source_sha256: archive.source_sha256,
            path,
        },
        Err(_) => InstalledArtifact {
            bytes: bytes.len() as u64,
            format_version: VERSION,
            sha256: file_hash,
            payload_sha256: [0; 32],
            source_sha256: [0; 32],
            path,
        },
    }
}

fn index_entry(
    logical_id: &str,
    profile: Option<(&str, &str)>,
    artifact: &InstalledArtifact,
) -> Value {
    let mut entry = json!({
        "artifact": {
            "bytes": artifact.bytes,
            "format_version": artifact.format_version,
            "path": artifact.path,
            "payload_sha256": hex_hash(artifact.payload_sha256),
            "sha256": hex_hash(artifact.sha256),
        },
        "build_key_sha256": hex_hash([7; 32]),
        "kind": "geometry",
        "logical_id": logical_id,
        "manifest": {
            "path": format!("manifests/{logical_id}.json"),
            "sha256": hex_hash([11; 32]),
        },
        "source_sha256": hex_hash(artifact.source_sha256),
    });
    if let Some((platform, quality)) = profile {
        entry["profile"] = json!({"platform": platform, "quality": quality});
    }
    entry
}

fn write_index(root: &Path, entries: Vec<Value>) {
    let profile_count = entries
        .iter()
        .filter(|entry| entry.get("profile").is_some())
        .count();
    let mut index = json!({
        "entries": entries,
        "entry_count": entries.len(),
        "schema": if profile_count == 0 {
            "bloom-asset-index-v1"
        } else {
            "bloom-asset-index-v2"
        },
    });
    if profile_count != 0 {
        index["profiled_entry_count"] = json!(profile_count);
    }
    std::fs::write(
        root.join("index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
}

fn profile(platform: &str, quality: &str) -> VirtualGeometryAssetProfile {
    VirtualGeometryAssetProfile::new(platform, quality).unwrap()
}

fn wait_for(
    loader: &mut VirtualGeometryStoreLoader,
    ticket: VirtualGeometryStoreTicket,
) -> Result<ResolvedVirtualGeometryAsset, VirtualGeometryStoreError> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(result) = loader.poll(ticket) {
            return result;
        }
        assert!(Instant::now() < deadline, "store worker timed out");
        std::thread::yield_now();
    }
}

fn temporary_store(label: &str) -> PathBuf {
    let id = NEXT_STORE_TEST.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "bloom-virtual-store-{label}-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}
