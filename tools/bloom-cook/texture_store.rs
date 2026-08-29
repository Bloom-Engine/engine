//! Deterministic content-addressed storage for cooked BC7 textures.

use crate::asset_profile::AssetProfile;
use crate::asset_store::{
    manifest_path_for_profile, manifest_profile, manifest_string, manifest_u64, parse_hex_hash,
    read_manifest, validate_hex_hash, validate_logical_id, validate_manifest_identity,
    CHUNK_DIRECTORY, INSPECT_SCHEMA, MANIFEST_SCHEMA, PROFILED_INSPECT_SCHEMA,
    PROFILED_MANIFEST_SCHEMA, PROFILED_REPORT_SCHEMA, REPORT_SCHEMA,
};
use crate::geometry_cook::write_atomically;
use crate::geometry_format::{hex_hash, sha256};
use crate::texture_cook::{
    cook_prepared_texture, PreparedTexture, TextureSettings, TEXTURE_RECIPE_VERSION,
};
use image_dds::ddsfile::{Dds, DxgiFormat};
use serde_json::{json, Value};
use std::io::Cursor;
use std::path::Path;

pub(crate) fn store_texture_command(
    logical_id: &str,
    input: &Path,
    store: &Path,
    flags: &[String],
) -> Result<String, String> {
    validate_logical_id(logical_id)?;
    let (profile, texture_flags) = AssetProfile::split_optional_flags(flags)?;
    let settings = TextureSettings::parse(texture_flags.iter().map(String::as_str))?;
    let prepared = PreparedTexture::read(input, settings)?;
    let build_key = hex_hash(build_key_for_profile(
        settings.build_key_sha256(prepared.source_sha256),
        profile.as_ref(),
    ));
    let manifest_path = manifest_path_for_profile(store, logical_id, profile.as_ref());

    if manifest_path.exists() {
        let manifest = read_manifest(&manifest_path)?;
        let kind = validate_manifest_identity(&manifest, logical_id, profile.as_ref())?;
        if kind != "texture" {
            return Err(format!(
                "asset manifest kind is {kind:?}, expected \"texture\""
            ));
        }
        let stored_key = manifest_string(&manifest, "/build_key_sha256")?;
        validate_hex_hash(stored_key, "manifest build key")?;
        if stored_key == build_key {
            let artifact = verify_texture_manifest_artifact(&manifest, store, Some(&prepared))?;
            return texture_build_report(
                logical_id,
                input,
                &manifest_path,
                &build_key,
                profile.as_ref(),
                TextureBuildOutcome::cache_hit(),
                &artifact,
            );
        }
    }

    let cooked = cook_prepared_texture(input, &prepared)?;
    let metadata = validate_dds(&cooked.bytes, settings)?;
    if metadata.width != cooked.width
        || metadata.height != cooked.height
        || metadata.mip_levels != cooked.mip_levels
    {
        return Err("serialized DDS metadata does not match the texture encoder".to_string());
    }
    let artifact_sha256 = hex_hash(sha256(&cooked.bytes));
    let relative_path = format!("{CHUNK_DIRECTORY}/{artifact_sha256}.dds");
    let artifact_path = store.join(&relative_path);
    let chunk_written =
        install_texture_chunk(&artifact_path, &cooked.bytes, &artifact_sha256, settings)?;
    let artifact = TextureArtifactSummary {
        relative_path,
        sha256: artifact_sha256,
        bytes: cooked.bytes.len() as u64,
        format: settings.format_name().to_string(),
        width: metadata.width,
        height: metadata.height,
        mip_levels: metadata.mip_levels,
    };
    let source_sha256 = hex_hash(prepared.source_sha256);
    let mut manifest = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": build_key,
        "dependencies": [
            {
                "kind": "source-file",
                "sha256": source_sha256,
            }
        ],
        "kind": "texture",
        "logical_id": logical_id,
        "recipe": {
            "name": "bloom-texture",
            "version": TEXTURE_RECIPE_VERSION,
        },
        "schema": MANIFEST_SCHEMA,
        "settings": settings.as_json(),
        "source": {
            "sha256": source_sha256,
        },
    });
    if let Some(profile) = &profile {
        manifest["schema"] = json!(PROFILED_MANIFEST_SCHEMA);
        manifest["profile"] = profile.as_json();
    }
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("serialize asset manifest: {error}"))?;
    manifest_bytes.push(b'\n');
    write_atomically(&manifest_path, &manifest_bytes)?;

    texture_build_report(
        logical_id,
        input,
        &manifest_path,
        &build_key,
        profile.as_ref(),
        TextureBuildOutcome::cache_miss(chunk_written),
        &artifact,
    )
}

pub(crate) fn inspect_texture_manifest(
    logical_id: &str,
    store: &Path,
    profile: Option<&AssetProfile>,
    manifest_path: &Path,
    manifest: &Value,
) -> Result<String, String> {
    let contract = validate_texture_manifest_contract(manifest)?;
    let artifact = verify_texture_manifest_artifact(manifest, store, None)?;
    let mut report = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": contract.build_key_sha256,
        "kind": "texture",
        "logical_id": logical_id,
        "manifest": manifest_path.display().to_string(),
        "schema": if profile.is_some() { PROFILED_INSPECT_SCHEMA } else { INSPECT_SCHEMA },
        "source_sha256": hex_hash(contract.source_sha256),
        "validation": "pass",
    });
    if let Some(profile) = profile {
        report["profile"] = profile.as_json();
    }
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize asset inspection report: {error}"))
}

pub(crate) fn indexed_texture_manifest_entry(
    logical_id: &str,
    store: &Path,
    profile: Option<&AssetProfile>,
    manifest_path: &Path,
    manifest_bytes: &[u8],
    manifest: &Value,
) -> Result<Value, String> {
    let contract = validate_texture_manifest_contract(manifest)?;
    let artifact = verify_texture_manifest_artifact(manifest, store, None)?;
    let relative_manifest = manifest_path
        .strip_prefix(store)
        .map_err(|_| "asset manifest path escaped the store".to_string())?
        .to_string_lossy()
        .replace('\\', "/");
    let mut entry = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": contract.build_key_sha256,
        "kind": "texture",
        "logical_id": logical_id,
        "manifest": {
            "path": relative_manifest,
            "sha256": hex_hash(sha256(manifest_bytes)),
        },
        "source_sha256": hex_hash(contract.source_sha256),
    });
    if let Some(profile) = profile {
        entry["profile"] = profile.as_json();
    }
    Ok(entry)
}

#[derive(Debug, Eq, PartialEq)]
struct TextureArtifactSummary {
    relative_path: String,
    sha256: String,
    bytes: u64,
    format: String,
    width: u32,
    height: u32,
    mip_levels: u32,
}

impl TextureArtifactSummary {
    fn as_json(&self) -> Value {
        json!({
            "bytes": self.bytes,
            "format": self.format,
            "height": self.height,
            "mip_levels": self.mip_levels,
            "path": self.relative_path,
            "sha256": self.sha256,
            "width": self.width,
        })
    }
}

struct TextureManifestContract {
    build_key_sha256: String,
    source_sha256: [u8; 32],
    settings: TextureSettings,
}

struct TextureBuildOutcome {
    cache: &'static str,
    chunk_written: bool,
    manifest_written: bool,
}

impl TextureBuildOutcome {
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

fn texture_build_report(
    logical_id: &str,
    input: &Path,
    manifest_path: &Path,
    build_key: &str,
    profile: Option<&AssetProfile>,
    outcome: TextureBuildOutcome,
    artifact: &TextureArtifactSummary,
) -> Result<String, String> {
    let mut report = json!({
        "artifact": artifact.as_json(),
        "build_key_sha256": build_key,
        "cache": outcome.cache,
        "input": input.display().to_string(),
        "kind": "texture",
        "logical_id": logical_id,
        "manifest": manifest_path.display().to_string(),
        "schema": if profile.is_some() { PROFILED_REPORT_SCHEMA } else { REPORT_SCHEMA },
        "writes": {
            "chunks": u8::from(outcome.chunk_written),
            "manifests": u8::from(outcome.manifest_written),
        },
    });
    if let Some(profile) = profile {
        report["profile"] = profile.as_json();
    }
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize asset build report: {error}"))
}

fn install_texture_chunk(
    path: &Path,
    expected_bytes: &[u8],
    expected_sha256: &str,
    settings: TextureSettings,
) -> Result<bool, String> {
    if path.exists() {
        let existing =
            std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let actual_sha256 = hex_hash(sha256(&existing));
        if actual_sha256 != expected_sha256 || existing != expected_bytes {
            return Err(format!(
                "content-addressed chunk {} is corrupt: expected {expected_sha256}, actual {actual_sha256}",
                path.display()
            ));
        }
        validate_dds(&existing, settings)
            .map_err(|error| format!("validate existing chunk {}: {error}", path.display()))?;
        return Ok(false);
    }
    write_atomically(path, expected_bytes)?;
    Ok(true)
}

fn verify_texture_manifest_artifact(
    manifest: &Value,
    store: &Path,
    expected: Option<&PreparedTexture>,
) -> Result<TextureArtifactSummary, String> {
    let contract = validate_texture_manifest_contract(manifest)?;
    let relative_path = manifest_string(manifest, "/artifact/path")?;
    let artifact_sha256 = manifest_string(manifest, "/artifact/sha256")?;
    validate_hex_hash(artifact_sha256, "artifact hash")?;
    let canonical_path = format!("{CHUNK_DIRECTORY}/{artifact_sha256}.dds");
    if relative_path != canonical_path {
        return Err(format!(
            "manifest artifact path {relative_path:?} is not canonical {canonical_path:?}"
        ));
    }
    let declared_bytes = manifest_u64(manifest, "/artifact/bytes")?;
    let declared_width = manifest_u32(manifest, "/artifact/width")?;
    let declared_height = manifest_u32(manifest, "/artifact/height")?;
    let declared_mips = manifest_u32(manifest, "/artifact/mip_levels")?;
    let declared_format = manifest_string(manifest, "/artifact/format")?;
    if declared_format != contract.settings.format_name() {
        return Err("texture artifact format does not match its settings".to_string());
    }
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
    let metadata = validate_dds(&bytes, contract.settings)
        .map_err(|error| format!("validate chunk {}: {error}", path.display()))?;
    if (metadata.width, metadata.height, metadata.mip_levels)
        != (declared_width, declared_height, declared_mips)
    {
        return Err(format!(
            "chunk {} DDS metadata does not match its manifest",
            path.display()
        ));
    }
    if let Some(expected) = expected {
        if contract.source_sha256 != expected.source_sha256 {
            return Err("matching build key has the wrong source-file hash".to_string());
        }
        if contract.settings != expected.settings {
            return Err("matching build key has non-canonical texture settings".to_string());
        }
    }
    Ok(TextureArtifactSummary {
        relative_path: relative_path.to_string(),
        sha256: artifact_sha256.to_string(),
        bytes: declared_bytes,
        format: declared_format.to_string(),
        width: declared_width,
        height: declared_height,
        mip_levels: declared_mips,
    })
}

fn validate_texture_manifest_contract(manifest: &Value) -> Result<TextureManifestContract, String> {
    if manifest_string(manifest, "/recipe/name")? != "bloom-texture"
        || manifest_u64(manifest, "/recipe/version")? != u64::from(TEXTURE_RECIPE_VERSION)
    {
        return Err("asset manifest has an unsupported texture recipe".to_string());
    }
    let source_sha256_text = manifest_string(manifest, "/source/sha256")?;
    let source_sha256 = parse_hex_hash(source_sha256_text, "manifest source hash")?;
    let expected_dependencies = json!([
        {
            "kind": "source-file",
            "sha256": source_sha256_text,
        }
    ]);
    if manifest.pointer("/dependencies") != Some(&expected_dependencies) {
        return Err("asset manifest has non-canonical dependencies".to_string());
    }
    let settings = TextureSettings::from_manifest(
        manifest
            .pointer("/settings")
            .ok_or("asset manifest texture settings are missing")?,
    )?;
    let build_key_sha256 = manifest_string(manifest, "/build_key_sha256")?;
    validate_hex_hash(build_key_sha256, "manifest build key")?;
    let base_key = settings.build_key_sha256(source_sha256);
    let profile = manifest_profile(manifest)?;
    let actual_key = hex_hash(build_key_for_profile(base_key, profile.as_ref()));
    if actual_key != build_key_sha256 {
        return Err(format!(
            "asset manifest build key mismatch: declared {build_key_sha256}, actual {actual_key}"
        ));
    }
    Ok(TextureManifestContract {
        build_key_sha256: build_key_sha256.to_string(),
        source_sha256,
        settings,
    })
}

fn build_key_for_profile(base_key_sha256: [u8; 32], profile: Option<&AssetProfile>) -> [u8; 32] {
    let Some(profile) = profile else {
        return base_key_sha256;
    };
    let mut bytes = Vec::with_capacity(96);
    bytes.extend_from_slice(b"bloom-profiled-texture-recipe\0");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&base_key_sha256);
    bytes.extend_from_slice(&(profile.platform().len() as u32).to_le_bytes());
    bytes.extend_from_slice(profile.platform().as_bytes());
    bytes.extend_from_slice(&(profile.quality().len() as u32).to_le_bytes());
    bytes.extend_from_slice(profile.quality().as_bytes());
    sha256(&bytes)
}

struct TextureMetadata {
    width: u32,
    height: u32,
    mip_levels: u32,
}

fn validate_dds(bytes: &[u8], settings: TextureSettings) -> Result<TextureMetadata, String> {
    let dds = Dds::read(Cursor::new(bytes)).map_err(|error| format!("parse DDS: {error}"))?;
    let expected_format = if settings.format_name() == "bc7-rgba-unorm" {
        DxgiFormat::BC7_UNorm
    } else {
        DxgiFormat::BC7_UNorm_sRGB
    };
    if dds.get_dxgi_format() != Some(expected_format) {
        return Err(format!(
            "DDS format {:?} does not match expected {expected_format:?}",
            dds.get_dxgi_format()
        ));
    }
    if dds.get_width() == 0
        || dds.get_height() == 0
        || dds.get_num_mipmap_levels() == 0
        || dds.get_depth() != 1
        || dds.get_num_array_layers() != 1
    {
        return Err("DDS must be a non-empty 2D single-layer texture".to_string());
    }
    Ok(TextureMetadata {
        width: dds.get_width(),
        height: dds.get_height(),
        mip_levels: dds.get_num_mipmap_levels(),
    })
}

fn manifest_u32(manifest: &Value, pointer: &str) -> Result<u32, String> {
    let value = manifest_u64(manifest, pointer)?;
    u32::try_from(value).map_err(|_| format!("asset manifest field {pointer} exceeds u32"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn texture_store_rebuilds_only_changed_texture_and_package_index() {
        let root = temporary_root("texture-incremental");
        std::fs::create_dir_all(&root).unwrap();
        let first_source = root.join("first.png");
        let second_source = root.join("second.png");
        let mesh_source = root.join("triangle.glb");
        write_test_texture(&first_source, 11);
        write_test_texture(&second_source, 29);
        std::fs::write(&mesh_source, minimal_triangle_glb()).unwrap();

        let first = store_texture_command("textures/first", &first_source, &root, &[]).unwrap();
        let second = store_texture_command("textures/second", &second_source, &root, &[]).unwrap();
        crate::asset_store::store_geometry_command("meshes/triangle", &mesh_source, &root, &[])
            .unwrap();
        let first: Value = serde_json::from_str(&first).unwrap();
        let second: Value = serde_json::from_str(&second).unwrap();
        assert_eq!(first["cache"], "miss");
        assert_eq!(first["writes"]["chunks"], 1);
        assert_eq!(first["writes"]["manifests"], 1);
        assert_eq!(second["cache"], "miss");

        let first_manifest_path = root.join("manifests/textures/first.json");
        let second_manifest_path = root.join("manifests/textures/second.json");
        let mesh_manifest_path = root.join("manifests/meshes/triangle.json");
        let first_manifest_before = std::fs::read(&first_manifest_path).unwrap();
        let second_manifest_before = std::fs::read(&second_manifest_path).unwrap();
        let mesh_manifest_before = std::fs::read(&mesh_manifest_path).unwrap();
        let second_manifest: Value = serde_json::from_slice(&second_manifest_before).unwrap();
        let second_chunk_path = root.join(
            second_manifest
                .pointer("/artifact/path")
                .and_then(Value::as_str)
                .unwrap(),
        );
        let second_chunk_before = std::fs::read(&second_chunk_path).unwrap();
        let mesh_manifest: Value = serde_json::from_slice(&mesh_manifest_before).unwrap();
        let mesh_chunk_path = root.join(
            mesh_manifest
                .pointer("/artifact/path")
                .and_then(Value::as_str)
                .unwrap(),
        );
        let mesh_chunk_before = std::fs::read(&mesh_chunk_path).unwrap();

        let index = crate::asset_index::build_asset_index_command(&root).unwrap();
        let index: Value = serde_json::from_str(&index).unwrap();
        assert_eq!(index["entries"], 3);
        assert_eq!(index["unique_chunks"], 3);
        assert_eq!(index["writes"]["indexes"], 1);
        let index_before = std::fs::read(root.join("index.json")).unwrap();
        let index_document: Value = serde_json::from_slice(&index_before).unwrap();
        let kinds = index_document["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["kind"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(kinds, ["geometry", "texture", "texture"]);

        let unchanged = store_texture_command("textures/first", &first_source, &root, &[]).unwrap();
        let unchanged: Value = serde_json::from_str(&unchanged).unwrap();
        assert_eq!(unchanged["cache"], "hit");
        assert_eq!(unchanged["writes"]["chunks"], 0);
        assert_eq!(unchanged["writes"]["manifests"], 0);
        assert_eq!(
            std::fs::read(&first_manifest_path).unwrap(),
            first_manifest_before
        );
        assert_eq!(
            std::fs::read(root.join("index.json")).unwrap(),
            index_before
        );

        write_test_texture(&first_source, 47);
        let changed = store_texture_command("textures/first", &first_source, &root, &[]).unwrap();
        let changed: Value = serde_json::from_str(&changed).unwrap();
        assert_eq!(changed["cache"], "miss");
        assert_eq!(changed["writes"]["chunks"], 1);
        assert_eq!(changed["writes"]["manifests"], 1);
        assert_ne!(changed["build_key_sha256"], first["build_key_sha256"]);
        assert_ne!(changed["artifact"]["sha256"], first["artifact"]["sha256"]);
        assert_eq!(
            std::fs::read(&second_manifest_path).unwrap(),
            second_manifest_before
        );
        assert_eq!(
            std::fs::read(&second_chunk_path).unwrap(),
            second_chunk_before
        );
        assert_eq!(
            std::fs::read(&mesh_manifest_path).unwrap(),
            mesh_manifest_before
        );
        assert_eq!(std::fs::read(&mesh_chunk_path).unwrap(), mesh_chunk_before);
        assert_eq!(
            std::fs::read(root.join("index.json")).unwrap(),
            index_before
        );
        assert!(crate::asset_index::inspect_asset_index_command(&root)
            .unwrap_err()
            .contains("stale"));

        let rebuilt = crate::asset_index::build_asset_index_command(&root).unwrap();
        let rebuilt: Value = serde_json::from_str(&rebuilt).unwrap();
        assert_eq!(rebuilt["entries"], 3);
        assert_eq!(rebuilt["writes"]["indexes"], 1);
        assert_ne!(
            std::fs::read(root.join("index.json")).unwrap(),
            index_before
        );
        assert_eq!(
            std::fs::read(&second_manifest_path).unwrap(),
            second_manifest_before
        );
        assert_eq!(
            std::fs::read(&second_chunk_path).unwrap(),
            second_chunk_before
        );
        assert_eq!(
            std::fs::read(&mesh_manifest_path).unwrap(),
            mesh_manifest_before
        );
        assert_eq!(std::fs::read(&mesh_chunk_path).unwrap(), mesh_chunk_before);
        let inspection =
            crate::asset_store::inspect_asset_command("textures/first", &root, &[]).unwrap();
        let inspection: Value = serde_json::from_str(&inspection).unwrap();
        assert_eq!(inspection["kind"], "texture");
        assert_eq!(inspection["validation"], "pass");

        let changed_manifest = read_manifest(&first_manifest_path).unwrap();
        let changed_chunk =
            root.join(manifest_string(&changed_manifest, "/artifact/path").unwrap());
        let mut corrupt = std::fs::read(&changed_chunk).unwrap();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0x80;
        std::fs::write(&changed_chunk, corrupt).unwrap();
        assert!(
            crate::asset_store::inspect_asset_command("textures/first", &root, &[])
                .unwrap_err()
                .contains("hash mismatch")
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profiled_texture_keys_include_semantics_and_profile_but_deduplicate_bytes() {
        let root = temporary_root("texture-profile");
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("mask.png");
        write_test_texture(&source, 7);
        let linear_flags = vec![
            "--linear".to_string(),
            "--platform".to_string(),
            "portable".to_string(),
            "--quality".to_string(),
            "high".to_string(),
        ];
        let normal_flags = vec![
            "--normal".to_string(),
            "--platform".to_string(),
            "portable".to_string(),
            "--quality".to_string(),
            "high".to_string(),
        ];
        let linear =
            store_texture_command("textures/mask-linear", &source, &root, &linear_flags).unwrap();
        let normal =
            store_texture_command("textures/mask-normal", &source, &root, &normal_flags).unwrap();
        let linear: Value = serde_json::from_str(&linear).unwrap();
        let normal: Value = serde_json::from_str(&normal).unwrap();
        assert_ne!(linear["build_key_sha256"], normal["build_key_sha256"]);
        assert_eq!(linear["artifact"]["sha256"], normal["artifact"]["sha256"]);
        assert_eq!(normal["writes"]["chunks"], 0);

        let inspect_flags = vec![
            "--platform".to_string(),
            "portable".to_string(),
            "--quality".to_string(),
            "high".to_string(),
        ];
        let inspection = crate::asset_store::inspect_asset_command(
            "textures/mask-normal",
            &root,
            &inspect_flags,
        )
        .unwrap();
        let inspection: Value = serde_json::from_str(&inspection).unwrap();
        assert_eq!(inspection["profile"]["platform"], "portable");
        assert_eq!(inspection["artifact"]["format"], "bc7-rgba-unorm");

        std::fs::remove_dir_all(root).unwrap();
    }

    fn write_test_texture(path: &Path, seed: u8) {
        let image = image::RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba([
                seed.wrapping_add((x * 17) as u8),
                seed.wrapping_add((y * 29) as u8),
                seed.wrapping_add(((x + y) * 11) as u8),
                255,
            ])
        });
        image.save(path).unwrap();
    }

    fn minimal_triangle_glb() -> Vec<u8> {
        let mut binary = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
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
}
