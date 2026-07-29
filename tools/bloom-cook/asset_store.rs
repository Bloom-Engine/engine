//! Deterministic content-addressed storage for cooked geometry artifacts.
//!
//! A logical manifest is installed only after its immutable chunk has been
//! fully written and validated. Matching manifests are strict cache entries:
//! corrupt metadata or chunks fail closed instead of being treated as a miss.

use crate::geometry_cook::{
    cook_prepared_geometry, geometry_build_key_sha256, prepare_geometry, write_atomically,
    PreparedGeometry, GEOMETRY_RECIPE_VERSION,
};
use crate::geometry_format::{
    decode_geometry, hex_hash, sha256, MAX_PAGE_BYTES, MIN_PAGE_BYTES, QUANTIZED_VERSION, VERSION,
};
use crate::geometry_quantization::VertexEncoding;
use crate::meshlet::MeshletLimits;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const MANIFEST_SCHEMA: &str = "bloom-asset-manifest-v1";
const REPORT_SCHEMA: &str = "bloom-asset-build-report-v1";
const INSPECT_SCHEMA: &str = "bloom-asset-inspect-report-v1";
const CHUNK_DIRECTORY: &str = "chunks/sha256";

pub fn store_geometry_command(
    logical_id: &str,
    input: &Path,
    store: &Path,
    flags: &[String],
) -> Result<String, String> {
    validate_logical_id(logical_id)?;
    let prepared = prepare_geometry(input, flags)?;
    let build_key = hex_hash(prepared.build_key_sha256());
    let manifest_path = manifest_path(store, logical_id);

    if manifest_path.exists() {
        let manifest = read_manifest(&manifest_path)?;
        validate_manifest_identity(&manifest, logical_id)?;
        let stored_key = manifest_string(&manifest, "/build_key_sha256")?;
        validate_hex_hash(stored_key, "manifest build key")?;
        if stored_key == build_key {
            let artifact = verify_manifest_artifact(&manifest, store, Some(&prepared))?;
            return build_report(
                logical_id,
                input,
                &manifest_path,
                &build_key,
                BuildOutcome::cache_hit(),
                &artifact,
            );
        }
    }

    let source_sha256 = hex_hash(prepared.source_sha256());
    let settings = prepared.settings_json();
    let expected_format_version = prepared.expected_format_version();
    let cooked = cook_prepared_geometry(input, prepared)?;
    let archive = decode_geometry(&cooked.bytes)?;
    if archive.format_version != expected_format_version {
        return Err(format!(
            "cooked geometry format {} does not match recipe expectation {expected_format_version}",
            archive.format_version
        ));
    }

    let artifact_sha256 = hex_hash(sha256(&cooked.bytes));
    let relative_path = format!("{CHUNK_DIRECTORY}/{artifact_sha256}.bgeo");
    let artifact_path = store.join(&relative_path);
    let chunk_written = install_chunk(&artifact_path, &cooked.bytes, &artifact_sha256)?;
    let artifact = ArtifactSummary {
        relative_path,
        sha256: artifact_sha256,
        payload_sha256: hex_hash(archive.payload_sha256),
        bytes: cooked.bytes.len() as u64,
        format_version: archive.format_version,
    };
    let manifest = json!({
        "artifact": {
            "bytes": artifact.bytes,
            "format_version": artifact.format_version,
            "path": artifact.relative_path,
            "payload_sha256": artifact.payload_sha256,
            "sha256": artifact.sha256,
        },
        "build_key_sha256": build_key,
        "dependencies": [
            {
                "kind": "source-closure",
                "sha256": source_sha256,
            }
        ],
        "kind": "geometry",
        "logical_id": logical_id,
        "recipe": {
            "name": "bloom-geometry",
            "version": GEOMETRY_RECIPE_VERSION,
        },
        "schema": MANIFEST_SCHEMA,
        "settings": settings,
        "source": {
            "sha256": source_sha256,
        },
    });
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("serialize asset manifest: {error}"))?;
    manifest_bytes.push(b'\n');
    write_atomically(&manifest_path, &manifest_bytes)?;

    build_report(
        logical_id,
        input,
        &manifest_path,
        &build_key,
        BuildOutcome::cache_miss(chunk_written),
        &artifact,
    )
}

pub fn inspect_asset_command(logical_id: &str, store: &Path) -> Result<String, String> {
    validate_logical_id(logical_id)?;
    let manifest_path = manifest_path(store, logical_id);
    let manifest = read_manifest(&manifest_path)?;
    validate_manifest_identity(&manifest, logical_id)?;
    let contract = validate_manifest_contract(&manifest)?;
    let artifact = verify_manifest_artifact(&manifest, store, None)?;

    serde_json::to_string_pretty(&json!({
        "artifact": {
            "bytes": artifact.bytes,
            "format_version": artifact.format_version,
            "path": artifact.relative_path,
            "payload_sha256": artifact.payload_sha256,
            "sha256": artifact.sha256,
        },
        "build_key_sha256": contract.build_key_sha256,
        "kind": "geometry",
        "logical_id": logical_id,
        "manifest": manifest_path.display().to_string(),
        "schema": INSPECT_SCHEMA,
        "source_sha256": hex_hash(contract.source_sha256),
        "validation": "pass",
    }))
    .map_err(|error| format!("serialize asset inspection report: {error}"))
}

#[derive(Debug, Eq, PartialEq)]
struct ArtifactSummary {
    relative_path: String,
    sha256: String,
    payload_sha256: String,
    bytes: u64,
    format_version: u32,
}

struct ManifestContract {
    build_key_sha256: String,
    source_sha256: [u8; 32],
    expected_format_version: u32,
}

struct BuildOutcome {
    cache: &'static str,
    chunk_written: bool,
    manifest_written: bool,
}

impl BuildOutcome {
    const fn cache_hit() -> Self {
        Self {
            cache: "hit",
            chunk_written: false,
            manifest_written: false,
        }
    }

    const fn cache_miss(chunk_written: bool) -> Self {
        Self {
            cache: "miss",
            chunk_written,
            manifest_written: true,
        }
    }
}

fn build_report(
    logical_id: &str,
    input: &Path,
    manifest_path: &Path,
    build_key: &str,
    outcome: BuildOutcome,
    artifact: &ArtifactSummary,
) -> Result<String, String> {
    serde_json::to_string_pretty(&json!({
        "artifact": {
            "bytes": artifact.bytes,
            "format_version": artifact.format_version,
            "path": artifact.relative_path,
            "payload_sha256": artifact.payload_sha256,
            "sha256": artifact.sha256,
        },
        "build_key_sha256": build_key,
        "cache": outcome.cache,
        "input": input.display().to_string(),
        "logical_id": logical_id,
        "manifest": manifest_path.display().to_string(),
        "schema": REPORT_SCHEMA,
        "writes": {
            "chunks": u8::from(outcome.chunk_written),
            "manifests": u8::from(outcome.manifest_written),
        },
    }))
    .map_err(|error| format!("serialize asset build report: {error}"))
}

fn install_chunk(
    path: &Path,
    expected_bytes: &[u8],
    expected_sha256: &str,
) -> Result<bool, String> {
    if path.exists() {
        let existing =
            std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let actual_sha256 = hex_hash(sha256(&existing));
        if actual_sha256 != expected_sha256 || existing != expected_bytes {
            return Err(format!(
                "content-addressed chunk {} is corrupt: expected {expected_sha256}, \
                 actual {actual_sha256}",
                path.display()
            ));
        }
        decode_geometry(&existing)
            .map_err(|error| format!("validate existing chunk {}: {error}", path.display()))?;
        return Ok(false);
    }
    write_atomically(path, expected_bytes)?;
    Ok(true)
}

fn verify_manifest_artifact(
    manifest: &Value,
    store: &Path,
    expected: Option<&PreparedGeometry>,
) -> Result<ArtifactSummary, String> {
    let contract = validate_manifest_contract(manifest)?;
    let relative_path = manifest_string(manifest, "/artifact/path")?;
    let artifact_sha256 = manifest_string(manifest, "/artifact/sha256")?;
    let payload_sha256 = manifest_string(manifest, "/artifact/payload_sha256")?;
    validate_hex_hash(artifact_sha256, "artifact hash")?;
    validate_hex_hash(payload_sha256, "artifact payload hash")?;
    let canonical_path = format!("{CHUNK_DIRECTORY}/{artifact_sha256}.bgeo");
    if relative_path != canonical_path {
        return Err(format!(
            "manifest artifact path {relative_path:?} is not canonical {canonical_path:?}"
        ));
    }
    let declared_bytes = manifest_u64(manifest, "/artifact/bytes")?;
    let declared_format = manifest_u64(manifest, "/artifact/format_version")?;
    let declared_format = u32::try_from(declared_format)
        .map_err(|_| "manifest artifact format version exceeds u32".to_string())?;
    let path = store.join(relative_path);
    let bytes =
        std::fs::read(&path).map_err(|error| format!("read chunk {}: {error}", path.display()))?;
    if bytes.len() as u64 != declared_bytes {
        return Err(format!(
            "chunk {} length mismatch: manifest {declared_bytes}, actual {}",
            path.display(),
            bytes.len()
        ));
    }
    let actual_sha256 = hex_hash(sha256(&bytes));
    if actual_sha256 != artifact_sha256 {
        return Err(format!(
            "chunk {} hash mismatch: manifest {artifact_sha256}, actual {actual_sha256}",
            path.display()
        ));
    }
    let archive = decode_geometry(&bytes)
        .map_err(|error| format!("validate chunk {}: {error}", path.display()))?;
    if archive.format_version != declared_format {
        return Err(format!(
            "chunk {} format mismatch: manifest {declared_format}, actual {}",
            path.display(),
            archive.format_version
        ));
    }
    if archive.format_version != contract.expected_format_version {
        return Err(format!(
            "chunk {} format {} does not match manifest vertex format",
            path.display(),
            archive.format_version
        ));
    }
    if archive.source_sha256 != contract.source_sha256 {
        return Err(format!(
            "chunk {} source closure does not match its manifest",
            path.display()
        ));
    }
    let actual_payload_sha256 = hex_hash(archive.payload_sha256);
    if actual_payload_sha256 != payload_sha256 {
        return Err(format!(
            "chunk {} payload hash mismatch: manifest {payload_sha256}, \
             actual {actual_payload_sha256}",
            path.display()
        ));
    }

    if let Some(expected) = expected {
        if manifest.pointer("/settings") != Some(&expected.settings_json()) {
            return Err("matching build key has non-canonical geometry settings".to_string());
        }
        if contract.source_sha256 != expected.source_sha256() {
            return Err("matching build key has the wrong source closure hash".to_string());
        }
        if archive.format_version != expected.expected_format_version() {
            return Err("matching build key has the wrong geometry format version".to_string());
        }
    }

    Ok(ArtifactSummary {
        relative_path: relative_path.to_string(),
        sha256: artifact_sha256.to_string(),
        payload_sha256: payload_sha256.to_string(),
        bytes: declared_bytes,
        format_version: declared_format,
    })
}

fn validate_manifest_contract(manifest: &Value) -> Result<ManifestContract, String> {
    if manifest_string(manifest, "/recipe/name")? != "bloom-geometry"
        || manifest_u64(manifest, "/recipe/version")? != u64::from(GEOMETRY_RECIPE_VERSION)
    {
        return Err("asset manifest has an unsupported geometry recipe".to_string());
    }
    let source_sha256_text = manifest_string(manifest, "/source/sha256")?;
    let source_sha256 = parse_hex_hash(source_sha256_text, "manifest source hash")?;
    let expected_dependencies = json!([
        {
            "kind": "source-closure",
            "sha256": source_sha256_text,
        }
    ]);
    if manifest.pointer("/dependencies") != Some(&expected_dependencies) {
        return Err("asset manifest has non-canonical dependencies".to_string());
    }

    let settings = manifest
        .pointer("/settings")
        .and_then(Value::as_object)
        .ok_or("asset manifest geometry settings are missing or not an object")?;
    if settings.len() != 5 {
        return Err("asset manifest geometry settings have unknown or missing fields".to_string());
    }
    let max_vertices = manifest_u32(manifest, "/settings/max_vertices_per_meshlet")?;
    let max_triangles = manifest_u32(manifest, "/settings/max_triangles_per_meshlet")?;
    MeshletLimits {
        max_vertices,
        max_triangles,
    }
    .validate()?;
    let page_bytes = manifest_u32(manifest, "/settings/page_budget_bytes")?;
    if !(MIN_PAGE_BYTES..=MAX_PAGE_BYTES).contains(&page_bytes) || !page_bytes.is_power_of_two() {
        return Err("asset manifest geometry page budget is invalid".to_string());
    }
    let hierarchy_levels = manifest_u32(manifest, "/settings/hierarchy_levels")?;
    if hierarchy_levels > 16 {
        return Err("asset manifest geometry hierarchy level count exceeds 16".to_string());
    }
    let vertex_encoding = match manifest_string(manifest, "/settings/vertex_format")? {
        "float32" => VertexEncoding::Float32,
        "quantized32" => VertexEncoding::Quantized,
        other => {
            return Err(format!(
                "asset manifest has unknown vertex format {other:?}"
            ))
        }
    };
    let build_key_sha256 = manifest_string(manifest, "/build_key_sha256")?;
    validate_hex_hash(build_key_sha256, "manifest build key")?;
    let actual_key = hex_hash(geometry_build_key_sha256(
        source_sha256,
        max_vertices,
        max_triangles,
        page_bytes,
        hierarchy_levels,
        vertex_encoding,
    ));
    if actual_key != build_key_sha256 {
        return Err(format!(
            "asset manifest build key mismatch: declared {build_key_sha256}, actual {actual_key}"
        ));
    }
    Ok(ManifestContract {
        build_key_sha256: build_key_sha256.to_string(),
        source_sha256,
        expected_format_version: match vertex_encoding {
            VertexEncoding::Float32 => VERSION,
            VertexEncoding::Quantized => QUANTIZED_VERSION,
        },
    })
}

fn read_manifest(path: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse manifest {}: {error}", path.display()))
}

fn validate_manifest_identity(manifest: &Value, logical_id: &str) -> Result<(), String> {
    if manifest_string(manifest, "/schema")? != MANIFEST_SCHEMA {
        return Err("unsupported asset manifest schema".to_string());
    }
    if manifest_string(manifest, "/kind")? != "geometry" {
        return Err("asset manifest kind is not geometry".to_string());
    }
    if manifest_string(manifest, "/logical_id")? != logical_id {
        return Err("asset manifest logical ID does not match its path".to_string());
    }
    Ok(())
}

fn manifest_string<'a>(manifest: &'a Value, pointer: &str) -> Result<&'a str, String> {
    manifest
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("asset manifest field {pointer} is missing or not a string"))
}

fn manifest_u64(manifest: &Value, pointer: &str) -> Result<u64, String> {
    manifest
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("asset manifest field {pointer} is missing or not an integer"))
}

fn manifest_u32(manifest: &Value, pointer: &str) -> Result<u32, String> {
    let value = manifest_u64(manifest, pointer)?;
    u32::try_from(value).map_err(|_| format!("asset manifest field {pointer} exceeds u32"))
}

fn validate_hex_hash(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not a canonical lowercase SHA-256"));
    }
    Ok(())
}

fn parse_hex_hash(value: &str, label: &str) -> Result<[u8; 32], String> {
    validate_hex_hash(value, label)?;
    let mut result = [0u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| format!("{label} contains invalid hex"))?;
    }
    Ok(result)
}

fn manifest_path(store: &Path, logical_id: &str) -> PathBuf {
    store.join("manifests").join(format!("{logical_id}.json"))
}

fn validate_logical_id(logical_id: &str) -> Result<(), String> {
    if logical_id.is_empty()
        || logical_id.starts_with('/')
        || logical_id.ends_with('/')
        || logical_id.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err(format!(
            "logical asset ID {logical_id:?} must be a relative slash-separated \
             ASCII identifier without empty, dot, or parent segments"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_ids_are_relative_and_canonical() {
        for valid in ["helmet", "world/sponza-main", "props/chair.v2"] {
            validate_logical_id(valid).unwrap();
        }
        assert_ne!(
            manifest_path(Path::new("store"), "props/chair.v2"),
            manifest_path(Path::new("store"), "props/chair.v3")
        );
        for invalid in [
            "",
            "/root",
            "root/",
            "a//b",
            ".",
            "..",
            "a/../b",
            "snowman-☃",
        ] {
            assert!(validate_logical_id(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn geometry_store_is_deterministic_incremental_and_fail_closed() {
        let root = temporary_root("store");
        let input = root.join("triangle.glb");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&input, minimal_triangle_glb()).unwrap();
        let flags = vec![
            "--hierarchy-levels".to_string(),
            "2".to_string(),
            "--vertex-format".to_string(),
            "quantized32".to_string(),
        ];

        let first = store_geometry_command("tests/triangle", &input, &root, &flags).unwrap();
        let first: Value = serde_json::from_str(&first).unwrap();
        assert_eq!(first["cache"], "miss");
        assert_eq!(first["writes"]["chunks"], 1);
        assert_eq!(first["writes"]["manifests"], 1);
        let manifest_path = root.join("manifests/tests/triangle.json");
        let manifest_before = std::fs::read(&manifest_path).unwrap();
        let inspection = inspect_asset_command("tests/triangle", &root).unwrap();
        let inspection: Value = serde_json::from_str(&inspection).unwrap();
        assert_eq!(inspection["validation"], "pass");

        let mut tampered_manifest: Value = serde_json::from_slice(&manifest_before).unwrap();
        tampered_manifest["settings"]["hierarchy_levels"] = json!(3);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&tampered_manifest).unwrap(),
        )
        .unwrap();
        assert!(inspect_asset_command("tests/triangle", &root)
            .unwrap_err()
            .contains("build key mismatch"));
        std::fs::write(&manifest_path, &manifest_before).unwrap();

        let second = store_geometry_command("tests/triangle", &input, &root, &flags).unwrap();
        let second: Value = serde_json::from_str(&second).unwrap();
        assert_eq!(second["cache"], "hit");
        assert_eq!(second["writes"]["chunks"], 0);
        assert_eq!(second["writes"]["manifests"], 0);
        assert_eq!(std::fs::read(&manifest_path).unwrap(), manifest_before);
        assert_eq!(second["build_key_sha256"], first["build_key_sha256"]);
        assert_eq!(second["artifact"]["sha256"], first["artifact"]["sha256"]);

        let other = store_geometry_command("tests/triangle-copy", &input, &root, &flags).unwrap();
        let other: Value = serde_json::from_str(&other).unwrap();
        assert_eq!(other["cache"], "miss");
        assert_eq!(other["writes"]["chunks"], 0);
        assert_eq!(other["writes"]["manifests"], 1);
        let other_manifest_path = root.join("manifests/tests/triangle-copy.json");
        let other_manifest_before = std::fs::read(&other_manifest_path).unwrap();

        let changed = store_geometry_command("tests/triangle", &input, &root, &[]).unwrap();
        let changed: Value = serde_json::from_str(&changed).unwrap();
        assert_eq!(changed["cache"], "miss");
        assert_ne!(changed["build_key_sha256"], first["build_key_sha256"]);
        assert_ne!(changed["artifact"]["sha256"], first["artifact"]["sha256"]);
        assert_eq!(
            std::fs::read(&other_manifest_path).unwrap(),
            other_manifest_before
        );

        let changed_manifest = read_manifest(&manifest_path).unwrap();
        let chunk = root.join(manifest_string(&changed_manifest, "/artifact/path").unwrap());
        let mut bytes = std::fs::read(&chunk).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        std::fs::write(&chunk, bytes).unwrap();
        assert!(store_geometry_command("tests/triangle", &input, &root, &[])
            .unwrap_err()
            .contains("hash mismatch"));

        let _ = std::fs::remove_dir_all(root);
    }

    fn temporary_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "bloom-cook-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn minimal_triangle_glb() -> Vec<u8> {
        let mut binary = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0u16, 1, 2] {
            binary.extend_from_slice(&index.to_le_bytes());
        }
        binary.resize(binary.len().div_ceil(4) * 4, 0);
        let json = format!(
            r#"{{
                "asset":{{"version":"2.0"}},
                "buffers":[{{"byteLength":{}}}],
                "bufferViews":[
                    {{"buffer":0,"byteOffset":0,"byteLength":36}},
                    {{"buffer":0,"byteOffset":36,"byteLength":6}}
                ],
                "accessors":[
                    {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
                      "min":[0,0,0],"max":[1,1,0]}},
                    {{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}
                ],
                "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],
                "nodes":[{{"mesh":0}}],
                "scenes":[{{"nodes":[0]}}],
                "scene":0
            }}"#,
            binary.len()
        );
        let mut json_bytes = json.into_bytes();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let total_length = 12 + 8 + json_bytes.len() + 8 + binary.len();
        let mut glb = Vec::with_capacity(total_length);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4e4f_534au32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004e_4942u32.to_le_bytes());
        glb.extend_from_slice(&binary);
        glb
    }
}
