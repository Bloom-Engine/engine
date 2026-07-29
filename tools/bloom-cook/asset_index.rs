//! Canonical index generation for a validated loose cooked-asset store.

use crate::asset_store::{indexed_manifest_entry, validate_logical_id};
use crate::geometry_cook::write_atomically;
use crate::geometry_format::{hex_hash, sha256};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

const INDEX_SCHEMA: &str = "bloom-asset-index-v1";
const BUILD_REPORT_SCHEMA: &str = "bloom-asset-index-build-report-v1";
const INSPECT_REPORT_SCHEMA: &str = "bloom-asset-index-inspect-report-v1";

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
    serde_json::to_string_pretty(&json!({
        "entries": built.entry_count,
        "index": path.display().to_string(),
        "index_sha256": hex_hash(sha256(&actual)),
        "referenced_bytes": built.referenced_bytes,
        "schema": INSPECT_REPORT_SCHEMA,
        "unique_chunk_bytes": built.unique_chunk_bytes,
        "unique_chunks": built.unique_chunks,
        "validation": "pass",
    }))
    .map_err(|error| format!("serialize asset index inspection: {error}"))
}

struct BuiltIndex {
    bytes: Vec<u8>,
    entry_count: usize,
    unique_chunks: usize,
    referenced_bytes: u64,
    unique_chunk_bytes: u64,
}

fn build_index(store: &Path) -> Result<BuiltIndex, String> {
    let mut logical_ids = Vec::new();
    collect_logical_ids(
        &store.join("manifests"),
        &store.join("manifests"),
        &mut logical_ids,
    )?;
    logical_ids.sort();
    if logical_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("asset store contains duplicate logical IDs".to_string());
    }

    let mut entries = Vec::with_capacity(logical_ids.len());
    let mut unique_chunks = BTreeMap::<String, u64>::new();
    let mut referenced_bytes = 0u64;
    for logical_id in logical_ids {
        let entry = indexed_manifest_entry(&logical_id, store)?;
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
    let index = json!({
        "entries": entries,
        "entry_count": entry_count,
        "schema": INDEX_SCHEMA,
    });
    let mut bytes = serde_json::to_vec_pretty(&index)
        .map_err(|error| format!("serialize asset index: {error}"))?;
    bytes.push(b'\n');
    Ok(BuiltIndex {
        bytes,
        entry_count,
        unique_chunks: unique_chunks.len(),
        referenced_bytes,
        unique_chunk_bytes,
    })
}

fn build_report(path: &Path, built: &BuiltIndex, written: bool) -> Result<String, String> {
    serde_json::to_string_pretty(&json!({
        "entries": built.entry_count,
        "index": path.display().to_string(),
        "index_sha256": hex_hash(sha256(&built.bytes)),
        "referenced_bytes": built.referenced_bytes,
        "schema": BUILD_REPORT_SCHEMA,
        "unique_chunk_bytes": built.unique_chunk_bytes,
        "unique_chunks": built.unique_chunks,
        "writes": {
            "indexes": u8::from(written),
        },
    }))
    .map_err(|error| format!("serialize asset index build report: {error}"))
}

fn collect_logical_ids(
    root: &Path,
    directory: &Path,
    logical_ids: &mut Vec<String>,
) -> Result<(), String> {
    if !directory.exists() {
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
}
