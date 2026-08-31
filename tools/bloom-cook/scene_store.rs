//! Deterministic content-addressed storage for source-free scene archives.

use crate::asset_profile::AssetProfile;
use crate::asset_store::{
    manifest_path_for_profile, manifest_profile, manifest_string, manifest_u64, read_manifest,
    validate_hex_hash, validate_logical_id, validate_manifest_identity, CHUNK_DIRECTORY,
    MANIFEST_SCHEMA, PROFILED_MANIFEST_SCHEMA,
};
use crate::geometry_cook::write_atomically;
use crate::geometry_format::{hex_hash, sha256};
use crate::scene_cook::{prepare_scene, SCENE_RECIPE_VERSION};
use crate::texture_store::store_prepared_texture;
use bloom_scene_format::{decode_scene, encode_scene, VERSION as SCENE_FORMAT_VERSION};
use serde_json::{json, Value};
use std::path::Path;

pub(crate) fn store_scene_command(
    logical_id: &str,
    input: &Path,
    store: &Path,
    flags: &[String],
) -> Result<String, String> {
    validate_logical_id(logical_id)?;
    let (profile, remaining) = AssetProfile::split_optional_flags(flags)?;
    if let Some(option) = remaining.first() {
        return Err(format!("unknown scene-store option {option:?}"));
    }
    let prepared = prepare_scene(input, logical_id)?;
    let texture_count = prepared.textures.len();
    let mut texture_cache_hits = 0usize;
    let mut texture_writes = 0usize;
    for (dependency, texture) in prepared.archive.textures.iter().zip(prepared.textures) {
        let report = store_prepared_texture(
            &dependency.logical_id,
            input,
            store,
            profile.clone(),
            texture,
        )?;
        let report: Value = serde_json::from_str(&report)
            .map_err(|error| format!("parse texture build report: {error}"))?;
        texture_cache_hits += usize::from(report["cache"] == "hit");
        texture_writes += report["writes"]["chunks"].as_u64().unwrap_or(0) as usize
            + report["writes"]["manifests"].as_u64().unwrap_or(0) as usize;
    }

    let bytes = encode_scene(&prepared.archive)?;
    let decoded = decode_scene(&bytes)?;
    let artifact_sha256 = hex_hash(sha256(&bytes));
    let payload_sha256 = hex_hash(decoded.payload_sha256);
    let build_key = hex_hash(scene_build_key(decoded.payload_sha256, profile.as_ref()));
    let manifest_path = manifest_path_for_profile(store, logical_id, profile.as_ref());
    if manifest_path.exists() {
        let manifest = read_manifest(&manifest_path)?;
        let kind = validate_manifest_identity(&manifest, logical_id, profile.as_ref())?;
        if kind != "scene" {
            return Err(format!(
                "asset manifest kind is {kind:?}, expected \"scene\""
            ));
        }
        if manifest_string(&manifest, "/build_key_sha256")? == build_key {
            let artifact = verify_scene_manifest(&manifest, store)?;
            return build_report(
                logical_id,
                input,
                &manifest_path,
                &build_key,
                profile.as_ref(),
                "hit",
                false,
                false,
                texture_count,
                texture_cache_hits,
                texture_writes,
                &prepared.sanitation,
                &artifact,
            );
        }
    }

    let relative_path = format!("{CHUNK_DIRECTORY}/{artifact_sha256}.bscene");
    let artifact_path = store.join(&relative_path);
    let chunk_written = install_scene_chunk(&artifact_path, &bytes, &artifact_sha256)?;
    let artifact = SceneArtifact {
        relative_path,
        sha256: artifact_sha256,
        payload_sha256,
        bytes: bytes.len() as u64,
        format_version: SCENE_FORMAT_VERSION,
        primitives: prepared.archive.primitives.len() as u64,
        placements: prepared.archive.placements.len() as u64,
        textures: texture_count as u64,
        animation_clips: prepared
            .archive
            .animation
            .as_ref()
            .map_or(0, |animation| animation.clips.len() as u64),
        joints: prepared
            .archive
            .animation
            .as_ref()
            .and_then(|animation| animation.skeleton.as_ref())
            .map_or(0, |skeleton| skeleton.joints.len() as u64),
    };
    let source_sha256 = hex_hash(prepared.archive.source_geometry_sha256);
    let dependencies = dependency_json(&prepared.archive, &source_sha256);
    let mut manifest = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": build_key,
        "dependencies": dependencies,
        "diagnostics": diagnostics_json(&prepared.archive.diagnostics),
        "kind": "scene",
        "logical_id": logical_id,
        "recipe": {
            "name": "bloom-scene",
            "version": SCENE_RECIPE_VERSION,
        },
        "schema": MANIFEST_SCHEMA,
        "source": {
            "sha256": source_sha256,
        },
    });
    if let Some(profile) = &profile {
        manifest["schema"] = json!(PROFILED_MANIFEST_SCHEMA);
        manifest["profile"] = profile.as_json();
    }
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("serialize scene manifest: {error}"))?;
    manifest_bytes.push(b'\n');
    write_atomically(&manifest_path, &manifest_bytes)?;
    build_report(
        logical_id,
        input,
        &manifest_path,
        &build_key,
        profile.as_ref(),
        "miss",
        chunk_written,
        true,
        texture_count,
        texture_cache_hits,
        texture_writes,
        &prepared.sanitation,
        &artifact,
    )
}

pub(crate) fn inspect_scene_manifest(
    logical_id: &str,
    store: &Path,
    profile: Option<&AssetProfile>,
    manifest_path: &Path,
    manifest: &Value,
) -> Result<String, String> {
    let artifact = verify_scene_manifest(manifest, store)?;
    let mut report = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": manifest_string(manifest, "/build_key_sha256")?,
        "kind": "scene",
        "logical_id": logical_id,
        "manifest": manifest_path.display().to_string(),
        "schema": "bloom-scene-inspect-report-v1",
        "source_sha256": manifest_string(manifest, "/source/sha256")?,
        "validation": "pass",
    });
    if let Some(profile) = profile {
        report["profile"] = profile.as_json();
    }
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize scene inspection report: {error}"))
}

pub(crate) fn indexed_scene_manifest_entry(
    logical_id: &str,
    store: &Path,
    profile: Option<&AssetProfile>,
    manifest_path: &Path,
    manifest_bytes: &[u8],
    manifest: &Value,
) -> Result<Value, String> {
    let artifact = verify_scene_manifest(manifest, store)?;
    let relative_manifest = manifest_path
        .strip_prefix(store)
        .map_err(|_| "scene manifest path escaped the store".to_string())?
        .to_string_lossy()
        .replace('\\', "/");
    let mut entry = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": manifest_string(manifest, "/build_key_sha256")?,
        "kind": "scene",
        "logical_id": logical_id,
        "manifest": {
            "path": relative_manifest,
            "sha256": hex_hash(sha256(manifest_bytes)),
        },
        "source_sha256": manifest_string(manifest, "/source/sha256")?,
    });
    if let Some(profile) = profile {
        entry["profile"] = profile.as_json();
    }
    Ok(entry)
}

fn verify_scene_manifest(manifest: &Value, store: &Path) -> Result<SceneArtifact, String> {
    if manifest_string(manifest, "/recipe/name")? != "bloom-scene"
        || manifest_u64(manifest, "/recipe/version")? != u64::from(SCENE_RECIPE_VERSION)
    {
        return Err("asset manifest has an unsupported scene recipe".to_string());
    }
    let source_sha256 = manifest_string(manifest, "/source/sha256")?;
    validate_hex_hash(source_sha256, "scene source hash")?;
    let artifact_sha256 = manifest_string(manifest, "/artifact/sha256")?;
    let payload_sha256 = manifest_string(manifest, "/artifact/payload_sha256")?;
    validate_hex_hash(artifact_sha256, "scene artifact hash")?;
    validate_hex_hash(payload_sha256, "scene payload hash")?;
    let relative_path = manifest_string(manifest, "/artifact/path")?;
    let canonical_path = format!("{CHUNK_DIRECTORY}/{artifact_sha256}.bscene");
    if relative_path != canonical_path {
        return Err(format!(
            "scene artifact path {relative_path:?} is not canonical {canonical_path:?}"
        ));
    }
    let path = store.join(relative_path);
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|error| format!("inspect scene chunk {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "scene chunk {} is not a regular file",
            path.display()
        ));
    }
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("read scene chunk {}: {error}", path.display()))?;
    if bytes.len() as u64 != manifest_u64(manifest, "/artifact/bytes")? {
        return Err(format!("scene chunk {} length mismatch", path.display()));
    }
    if hex_hash(sha256(&bytes)) != artifact_sha256 {
        return Err(format!("scene chunk {} hash mismatch", path.display()));
    }
    let decoded = decode_scene(&bytes)
        .map_err(|error| format!("validate scene chunk {}: {error}", path.display()))?;
    if hex_hash(decoded.payload_sha256) != payload_sha256 {
        return Err(format!(
            "scene chunk {} payload hash mismatch",
            path.display()
        ));
    }
    if hex_hash(decoded.archive.source_geometry_sha256) != source_sha256 {
        return Err(format!(
            "scene chunk {} source hash mismatch",
            path.display()
        ));
    }
    let expected_dependencies = dependency_json(&decoded.archive, source_sha256);
    if manifest.pointer("/dependencies") != Some(&expected_dependencies) {
        return Err("scene manifest has non-canonical dependencies".to_string());
    }
    if manifest.pointer("/diagnostics") != Some(&diagnostics_json(&decoded.archive.diagnostics)) {
        return Err("scene manifest has non-canonical diagnostics".to_string());
    }
    let profile = manifest_profile(manifest)?;
    let actual_key = hex_hash(scene_build_key(decoded.payload_sha256, profile.as_ref()));
    let declared_key = manifest_string(manifest, "/build_key_sha256")?;
    validate_hex_hash(declared_key, "scene build key")?;
    if actual_key != declared_key {
        return Err(format!(
            "scene build key mismatch: declared {declared_key}, actual {actual_key}"
        ));
    }
    let artifact = SceneArtifact {
        relative_path: relative_path.to_string(),
        sha256: artifact_sha256.to_string(),
        payload_sha256: payload_sha256.to_string(),
        bytes: bytes.len() as u64,
        format_version: SCENE_FORMAT_VERSION,
        primitives: decoded.archive.primitives.len() as u64,
        placements: decoded.archive.placements.len() as u64,
        textures: decoded.archive.textures.len() as u64,
        animation_clips: decoded
            .archive
            .animation
            .as_ref()
            .map_or(0, |animation| animation.clips.len() as u64),
        joints: decoded
            .archive
            .animation
            .as_ref()
            .and_then(|animation| animation.skeleton.as_ref())
            .map_or(0, |skeleton| skeleton.joints.len() as u64),
    };
    if manifest.pointer("/artifact") != Some(&artifact.as_json()) {
        return Err("scene manifest artifact metadata is non-canonical".to_string());
    }
    Ok(artifact)
}

fn dependency_json(archive: &bloom_scene_format::SceneArchive, source_sha256: &str) -> Value {
    let mut dependencies = vec![json!({
        "kind": "source-closure",
        "sha256": source_sha256,
    })];
    dependencies.extend(archive.textures.iter().map(|texture| {
        json!({
            "kind": "texture",
            "logical_id": texture.logical_id,
            "sha256": hex_hash(texture.source_sha256),
        })
    }));
    Value::Array(dependencies)
}

fn diagnostics_json(diagnostics: &bloom_scene_format::SceneDiagnostics) -> Value {
    json!({
        "dropped_placements": diagnostics.dropped_placements,
        "dropped_primitives": diagnostics.dropped_primitives,
        "dropped_triangles": diagnostics.dropped_triangles,
        "non_finite_attribute_vertices": diagnostics.non_finite_attribute_vertices,
        "non_finite_position_vertices": diagnostics.non_finite_position_vertices,
    })
}

fn scene_build_key(payload_sha256: [u8; 32], profile: Option<&AssetProfile>) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(128);
    bytes.extend_from_slice(b"bloom-scene-recipe\0");
    bytes.extend_from_slice(&SCENE_RECIPE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&payload_sha256);
    if let Some(profile) = profile {
        bytes.extend_from_slice(profile.platform().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(profile.quality().as_bytes());
    }
    sha256(&bytes)
}

fn install_scene_chunk(
    path: &Path,
    expected: &[u8],
    expected_sha256: &str,
) -> Result<bool, String> {
    if path.exists() {
        let existing = std::fs::read(path)
            .map_err(|error| format!("read scene chunk {}: {error}", path.display()))?;
        if existing != expected || hex_hash(sha256(&existing)) != expected_sha256 {
            return Err(format!(
                "content-addressed scene chunk {} is corrupt",
                path.display()
            ));
        }
        decode_scene(&existing)?;
        return Ok(false);
    }
    write_atomically(path, expected)?;
    Ok(true)
}

#[derive(Debug)]
struct SceneArtifact {
    relative_path: String,
    sha256: String,
    payload_sha256: String,
    bytes: u64,
    format_version: u32,
    primitives: u64,
    placements: u64,
    textures: u64,
    animation_clips: u64,
    joints: u64,
}

impl SceneArtifact {
    fn as_json(&self) -> Value {
        json!({
            "animation_clips": self.animation_clips,
            "bytes": self.bytes,
            "format_version": self.format_version,
            "joints": self.joints,
            "path": self.relative_path,
            "payload_sha256": self.payload_sha256,
            "placements": self.placements,
            "primitives": self.primitives,
            "sha256": self.sha256,
            "textures": self.textures,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    logical_id: &str,
    input: &Path,
    manifest_path: &Path,
    build_key: &str,
    profile: Option<&AssetProfile>,
    cache: &str,
    chunk_written: bool,
    manifest_written: bool,
    texture_count: usize,
    texture_cache_hits: usize,
    texture_writes: usize,
    sanitation: &crate::scene_cook::SceneSanitation,
    artifact: &SceneArtifact,
) -> Result<String, String> {
    let mut report = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": build_key,
        "cache": cache,
        "input": input.display().to_string(),
        "logical_id": logical_id,
        "manifest": manifest_path.display().to_string(),
        "diagnostics": {
            "dropped_placements": sanitation.dropped_placements,
            "dropped_primitives": sanitation.dropped_primitives,
            "dropped_triangles": sanitation.dropped_triangles,
            "non_finite_attribute_vertices": sanitation.non_finite_attribute_vertices,
            "non_finite_position_vertices": sanitation.non_finite_position_vertices,
        },
        "schema": "bloom-scene-build-report-v1",
        "textures": {
            "cache_hits": texture_cache_hits,
            "count": texture_count,
            "writes": texture_writes,
        },
        "writes": {
            "chunks": u8::from(chunk_written),
            "manifests": u8::from(manifest_written),
        },
    });
    if let Some(profile) = profile {
        report["profile"] = profile.as_json();
    }
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize scene build report: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloom_shared::cooked_scene_store::{
        load_cooked_scene_from_store, CookedSceneProfile, CookedSceneStoreConfig,
        CookedSceneStoreRequest,
    };
    use std::path::PathBuf;

    #[test]
    fn scene_store_is_deterministic_incremental_and_source_free() {
        let first_root = temporary_root("scene-first");
        let second_root = temporary_root("scene-second");
        std::fs::create_dir_all(&first_root).unwrap();
        std::fs::create_dir_all(&second_root).unwrap();
        let source = first_root.join("triangle.glb");
        let second_source = second_root.join("triangle.glb");
        let glb = minimal_triangle_glb();
        std::fs::write(&source, &glb).unwrap();
        std::fs::write(&second_source, &glb).unwrap();
        let flags = profile_flags();

        let first = store_scene_command("tests/triangle", &source, &first_root, &flags).unwrap();
        let first: Value = serde_json::from_str(&first).unwrap();
        assert_eq!(first["cache"], "miss");
        assert_eq!(first["artifact"]["primitives"], 1);
        assert_eq!(first["artifact"]["placements"], 1);
        assert_eq!(first["artifact"]["textures"], 0);

        let repeated = store_scene_command("tests/triangle", &source, &first_root, &flags).unwrap();
        let repeated: Value = serde_json::from_str(&repeated).unwrap();
        assert_eq!(repeated["cache"], "hit");
        assert_eq!(repeated["writes"]["chunks"], 0);
        assert_eq!(repeated["writes"]["manifests"], 0);

        let second =
            store_scene_command("tests/triangle", &second_source, &second_root, &flags).unwrap();
        let second: Value = serde_json::from_str(&second).unwrap();
        assert_eq!(first["artifact"]["sha256"], second["artifact"]["sha256"]);
        assert_eq!(
            first["artifact"]["payload_sha256"],
            second["artifact"]["payload_sha256"]
        );

        crate::asset_index::build_asset_index_command(&first_root).unwrap();
        crate::asset_index::inspect_asset_index_command(&first_root).unwrap();
        let profile = CookedSceneProfile::new("portable", "high").unwrap();
        let request = CookedSceneStoreRequest::new("tests/triangle", profile);
        let resolved =
            load_cooked_scene_from_store(&first_root, &request, CookedSceneStoreConfig::default())
                .unwrap();
        assert!(resolved.prepared.texture_dependencies().is_empty());
        let cooked = resolved.prepared.finish(&[]).unwrap();
        assert_eq!(cooked.model.meshes.len(), 1);
        assert_eq!(cooked.model.mesh_transforms.len(), 1);

        let relative = first["artifact"]["path"].as_str().unwrap();
        let chunk = first_root.join(relative);
        let mut damaged = std::fs::read(&chunk).unwrap();
        *damaged.last_mut().unwrap() ^= 0x40;
        std::fs::write(&chunk, damaged).unwrap();
        let error =
            load_cooked_scene_from_store(&first_root, &request, CookedSceneStoreConfig::default())
                .err()
                .unwrap()
                .to_string();
        assert!(error.contains("hash"), "unexpected error: {error}");

        std::fs::remove_dir_all(first_root).unwrap();
        std::fs::remove_dir_all(second_root).unwrap();
    }

    fn profile_flags() -> Vec<String> {
        vec![
            "--platform".to_string(),
            "portable".to_string(),
            "--quality".to_string(),
            "high".to_string(),
        ]
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
