//! Canonical index generation for a validated loose cooked-asset store.

use crate::asset_profile::{validate_profile_component, AssetProfile};
use crate::asset_store::{
    indexed_manifest_entry, indexed_manifest_entry_for_profile, validate_logical_id,
};
use crate::geometry_cook::write_atomically;
use crate::geometry_format::{hex_hash, sha256};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

const INDEX_SCHEMA: &str = "bloom-asset-index-v1";
const PROFILED_INDEX_SCHEMA: &str = "bloom-asset-index-v2";
const BUILD_REPORT_SCHEMA: &str = "bloom-asset-index-build-report-v1";
const PROFILED_BUILD_REPORT_SCHEMA: &str = "bloom-asset-index-build-report-v2";
const INSPECT_REPORT_SCHEMA: &str = "bloom-asset-index-inspect-report-v1";
const PROFILED_INSPECT_REPORT_SCHEMA: &str = "bloom-asset-index-inspect-report-v2";

pub fn build_asset_index_command(store: &Path) -> Result<String, String> {
    let built = build_index(store)?;
    let path = store.join("index.json");
    let written = if path.exists() {
        let existing = std::fs::read(&path)
            .map_err(|error| format!("read asset index {}: {error}", path.display()))?;
        if existing == built.bytes {
            false
        } else {
            write_atomically(&path, &built.bytes)?;
            true
        }
    } else {
        write_atomically(&path, &built.bytes)?;
        true
    };
    build_report(&path, &built, written)
}

pub fn inspect_asset_index_command(store: &Path) -> Result<String, String> {
    let (built, actual) = validated_index(store)?;
    let mut report = json!({
        "entries": built.entry_count,
        "index": store.join("index.json").display().to_string(),
        "index_sha256": hex_hash(sha256(&actual)),
        "referenced_bytes": built.referenced_bytes,
        "schema": if built.profiled_entry_count == 0 {
            INSPECT_REPORT_SCHEMA
        } else {
            PROFILED_INSPECT_REPORT_SCHEMA
        },
        "unique_chunk_bytes": built.unique_chunk_bytes,
        "unique_chunks": built.unique_chunks,
        "validation": "pass",
    });
    if built.profiled_entry_count != 0 {
        report["index_schema"] = json!(built.schema);
        report["profiled_entries"] = json!(built.profiled_entry_count);
    }
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize asset index inspection: {error}"))
}

pub(crate) fn validated_index_value(store: &Path) -> Result<Value, String> {
    let (_, actual) = validated_index(store)?;
    serde_json::from_slice(&actual).map_err(|error| format!("parse validated asset index: {error}"))
}

fn validated_index(store: &Path) -> Result<(BuiltIndex, Vec<u8>), String> {
    let built = build_index(store)?;
    let path = store.join("index.json");
    let actual = std::fs::read(&path)
        .map_err(|error| format!("read asset index {}: {error}", path.display()))?;
    if actual != built.bytes {
        return Err(format!(
            "asset index {} is stale, corrupt, or non-canonical",
            path.display()
        ));
    }
    Ok((built, actual))
}

struct BuiltIndex {
    bytes: Vec<u8>,
    schema: &'static str,
    entry_count: usize,
    profiled_entry_count: usize,
    unique_chunks: usize,
    referenced_bytes: u64,
    unique_chunk_bytes: u64,
}

fn build_index(store: &Path) -> Result<BuiltIndex, String> {
    let mut descriptors = Vec::new();
    let mut logical_ids = Vec::new();
    collect_logical_ids(
        &store.join("manifests"),
        &store.join("manifests"),
        &mut logical_ids,
    )?;
    descriptors.extend(logical_ids.into_iter().map(ManifestDescriptor::legacy));
    collect_profiled_manifests(store, &mut descriptors)?;
    descriptors.sort();
    if descriptors.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("asset store contains duplicate logical ID/profile entries".to_string());
    }

    let mut entries = Vec::with_capacity(descriptors.len());
    let mut unique_chunks = BTreeMap::<String, u64>::new();
    let mut referenced_bytes = 0u64;
    for descriptor in &descriptors {
        let entry = match &descriptor.profile {
            None => indexed_manifest_entry(&descriptor.logical_id, store)?,
            Some(profile) => {
                indexed_manifest_entry_for_profile(&descriptor.logical_id, store, Some(profile))?
            }
        };
        let artifact_sha256 = entry_string(&entry, "/artifact/sha256")?;
        let artifact_bytes = entry_u64(&entry, "/artifact/bytes")?;
        referenced_bytes = referenced_bytes
            .checked_add(artifact_bytes)
            .ok_or("asset index referenced-byte total overflow")?;
        if let Some(previous_bytes) =
            unique_chunks.insert(artifact_sha256.to_string(), artifact_bytes)
        {
            if previous_bytes != artifact_bytes {
                return Err(format!(
                    "chunk {artifact_sha256} has inconsistent byte lengths across manifests"
                ));
            }
        }
        entries.push(entry);
    }
    let unique_chunk_bytes = unique_chunks
        .values()
        .try_fold(0u64, |total, bytes| total.checked_add(*bytes).ok_or(()))
        .map_err(|()| "asset index unique-byte total overflow".to_string())?;
    let entry_count = entries.len();
    let profiled_entry_count = descriptors
        .iter()
        .filter(|descriptor| descriptor.profile.is_some())
        .count();
    let schema = if profiled_entry_count == 0 {
        INDEX_SCHEMA
    } else {
        PROFILED_INDEX_SCHEMA
    };
    let mut index = json!({
        "entries": entries,
        "entry_count": entry_count,
        "schema": schema,
    });
    if profiled_entry_count != 0 {
        index["profiled_entry_count"] = json!(profiled_entry_count);
    }
    let mut bytes = serde_json::to_vec_pretty(&index)
        .map_err(|error| format!("serialize asset index: {error}"))?;
    bytes.push(b'\n');
    Ok(BuiltIndex {
        bytes,
        schema,
        entry_count,
        profiled_entry_count,
        unique_chunks: unique_chunks.len(),
        referenced_bytes,
        unique_chunk_bytes,
    })
}

fn build_report(path: &Path, built: &BuiltIndex, written: bool) -> Result<String, String> {
    let mut report = json!({
        "entries": built.entry_count,
        "index": path.display().to_string(),
        "index_sha256": hex_hash(sha256(&built.bytes)),
        "referenced_bytes": built.referenced_bytes,
        "schema": if built.profiled_entry_count == 0 {
            BUILD_REPORT_SCHEMA
        } else {
            PROFILED_BUILD_REPORT_SCHEMA
        },
        "unique_chunk_bytes": built.unique_chunk_bytes,
        "unique_chunks": built.unique_chunks,
        "writes": {
            "indexes": u8::from(written),
        },
    });
    if built.profiled_entry_count != 0 {
        report["index_schema"] = json!(built.schema);
        report["profiled_entries"] = json!(built.profiled_entry_count);
    }
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize asset index build report: {error}"))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ManifestDescriptor {
    logical_id: String,
    profile: Option<AssetProfile>,
}

impl ManifestDescriptor {
    fn legacy(logical_id: String) -> Self {
        Self {
            logical_id,
            profile: None,
        }
    }
}

fn collect_profiled_manifests(
    store: &Path,
    descriptors: &mut Vec<ManifestDescriptor>,
) -> Result<(), String> {
    let root = store.join("variants");
    if !require_directory_if_present(&root, "variant tree")? {
        return Ok(());
    }
    for (platform, platform_path) in child_directories(&root)? {
        validate_profile_component(&platform, "platform")?;
        for (quality, quality_path) in child_directories(&platform_path)? {
            validate_profile_component(&quality, "quality")?;
            let profile = AssetProfile::new(&platform, &quality)?;
            let mut logical_ids = Vec::new();
            collect_logical_ids(&quality_path, &quality_path, &mut logical_ids)?;
            descriptors.extend(
                logical_ids
                    .into_iter()
                    .map(|logical_id| ManifestDescriptor {
                        logical_id,
                        profile: Some(profile.clone()),
                    }),
            );
        }
    }
    Ok(())
}

fn child_directories(root: &Path) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    require_directory_if_present(root, "variant tree")?;
    let mut entries = std::fs::read_dir(root)
        .map_err(|error| format!("read variant directory {}: {error}", root.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read variant directory {}: {error}", root.display()))?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    entries
        .into_iter()
        .map(|entry| {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|error| format!("inspect variant path {}: {error}", path.display()))?;
            if file_type.is_symlink() || !file_type.is_dir() {
                return Err(format!(
                    "variant tree level may contain only directories, found {}",
                    path.display()
                ));
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| format!("variant path {} is not UTF-8", path.display()))?;
            Ok((name, path))
        })
        .collect()
}

fn collect_logical_ids(
    root: &Path,
    directory: &Path,
    logical_ids: &mut Vec<String>,
) -> Result<(), String> {
    if !require_directory_if_present(directory, "manifest tree")? {
        return Ok(());
    }
    let mut entries = std::fs::read_dir(directory)
        .map_err(|error| format!("read manifest directory {}: {error}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read manifest directory {}: {error}", directory.display()))?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("inspect manifest path {}: {error}", path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "manifest tree may not contain symlink {}",
                path.display()
            ));
        }
        if file_type.is_dir() {
            collect_logical_ids(root, &path, logical_ids)?;
            continue;
        }
        if !file_type.is_file() || path.extension().and_then(|value| value.to_str()) != Some("json")
        {
            return Err(format!(
                "manifest tree contains unexpected path {}",
                path.display()
            ));
        }
        logical_ids.push(logical_id_from_path(root, &path)?);
    }
    Ok(())
}

fn require_directory_if_present(path: &Path, label: &str) -> Result<bool, String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("inspect {label} {}: {error}", path.display())),
    };
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} root may not be a symlink: {}",
            path.display()
        ));
    }
    if !metadata.is_dir() {
        return Err(format!(
            "{label} root is not a directory: {}",
            path.display()
        ));
    }
    Ok(true)
}

fn logical_id_from_path(root: &Path, path: &Path) -> Result<String, String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| format!("manifest path {} escapes its root", path.display()))?;
    let relative = relative
        .to_str()
        .ok_or_else(|| format!("manifest path {} is not UTF-8", path.display()))?;
    let relative = relative.replace(std::path::MAIN_SEPARATOR, "/");
    let logical_id = relative
        .strip_suffix(".json")
        .ok_or_else(|| format!("manifest path {} lacks .json suffix", path.display()))?;
    validate_logical_id(logical_id)?;
    Ok(logical_id.to_string())
}

fn entry_string<'a>(entry: &'a Value, pointer: &str) -> Result<&'a str, String> {
    entry
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("indexed asset field {pointer} is missing or not a string"))
}

fn entry_u64(entry: &Value, pointer: &str) -> Result<u64, String> {
    entry
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("indexed asset field {pointer} is missing or not an integer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_id_derivation_preserves_dotted_names_and_rejects_non_manifests() {
        assert_eq!(
            logical_id_from_path(
                Path::new("store/manifests"),
                Path::new("store/manifests/props/chair.v2.json")
            )
            .unwrap(),
            "props/chair.v2"
        );
        assert!(logical_id_from_path(
            Path::new("store/manifests"),
            Path::new("store/elsewhere/chair.json")
        )
        .is_err());
        assert!(logical_id_from_path(
            Path::new("store/manifests"),
            Path::new("store/manifests/../escape.json")
        )
        .is_err());
    }

    #[test]
    fn manifest_scan_rejects_unindexed_files() {
        let root = std::env::temp_dir().join(format!(
            "bloom-cook-index-scan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let manifests = root.join("manifests");
        std::fs::create_dir_all(&manifests).unwrap();
        std::fs::write(manifests.join("unexpected.txt"), b"not a manifest").unwrap();
        assert!(build_asset_index_command(&root)
            .unwrap_err()
            .contains("unexpected path"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn variant_scan_rejects_noncanonical_profile_directories() {
        let root = std::env::temp_dir().join(format!(
            "bloom-cook-profile-scan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("variants/MacOS/high")).unwrap();
        assert!(build_asset_index_command(&root)
            .unwrap_err()
            .contains("lowercase ASCII"));
        let _ = std::fs::remove_dir_all(root);
    }
}
