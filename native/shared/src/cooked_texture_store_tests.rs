use super::*;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_TEXTURE_STORE_TEST: AtomicU64 = AtomicU64::new(1);

#[test]
fn indexed_texture_loader_selects_adapter_profile_and_reports_deliberate_fallback() {
    let root = temporary_store("profiles");
    let portable = install_dds(
        &root,
        &dds_bytes(image_dds::ImageFormat::Rgba8UnormSrgb),
        CookedTextureArtifactFormat::Rgba8Srgb,
    );
    let native = install_dds(
        &root,
        &dds_bytes(image_dds::ImageFormat::BC7RgbaUnormSrgb),
        CookedTextureArtifactFormat::Bc7Srgb,
    );
    let bc_plan = AdapterAssetProfilePlan::from_bc_support(true);
    let native_platform = bc_plan.runtime_platform();
    write_index(
        &root,
        vec![
            index_entry("textures/facade", Some(("portable", "high")), &portable),
            index_entry("textures/facade", Some((native_platform, "high")), &native),
            index_entry("textures/fallback", Some(("portable", "high")), &portable),
        ],
    );
    let mut loader = CookedTextureStoreLoader::new(
        &root,
        CookedTextureStoreConfig {
            max_pending_requests: 4,
            ..CookedTextureStoreConfig::default()
        },
    )
    .unwrap();

    let portable_ticket = loader
        .request(
            CookedTextureStoreRequest::for_runtime_features(
                "textures/facade",
                "high",
                wgpu::Features::empty(),
            )
            .unwrap(),
        )
        .unwrap();
    let portable_result = wait_for(&mut loader, portable_ticket).unwrap();
    assert_eq!(
        portable_result.format,
        CookedTextureArtifactFormat::Rgba8Srgb
    );
    assert_eq!(portable_result.selection.reason, "adapter-portable-profile");
    assert_eq!(
        portable_result.selection.kind,
        CookedTextureSelectionKind::Exact
    );

    let native_ticket = loader
        .request(
            CookedTextureStoreRequest::for_runtime_features(
                "textures/facade",
                "high",
                wgpu::Features::TEXTURE_COMPRESSION_BC,
            )
            .unwrap(),
        )
        .unwrap();
    let native_result = wait_for(&mut loader, native_ticket).unwrap();
    if bc_plan.native_profile_selected() {
        assert_eq!(native_result.format, CookedTextureArtifactFormat::Bc7Srgb);
        assert_eq!(native_result.selection.reason, "adapter-native-profile");
    } else {
        assert_eq!(native_result.format, CookedTextureArtifactFormat::Rgba8Srgb);
        assert_eq!(native_result.selection.reason, "adapter-portable-profile");
    }

    let fallback_ticket = loader
        .request(
            CookedTextureStoreRequest::for_runtime_features(
                "textures/fallback",
                "high",
                wgpu::Features::TEXTURE_COMPRESSION_BC,
            )
            .unwrap(),
        )
        .unwrap();
    let fallback = wait_for(&mut loader, fallback_ticket).unwrap();
    if bc_plan.native_profile_selected() {
        assert_eq!(
            fallback.selection.kind,
            CookedTextureSelectionKind::Fallback
        );
        assert_eq!(fallback.selection.fallback_rank, Some(0));
        assert_eq!(
            fallback.selection.reason,
            "portable-fallback-after-native-miss"
        );
    } else {
        assert_eq!(fallback.selection.kind, CookedTextureSelectionKind::Exact);
    }
    let report: Value = serde_json::from_str(&fallback.report_json()).unwrap();
    assert_eq!(report["schema"], "bloom-runtime-texture-selection-v1");
    assert_eq!(report["selection"]["policy"]["kind"], "adapter");
    assert_eq!(report["artifact"]["format"], "rgba8-unorm-srgb");

    let telemetry = loader.telemetry();
    assert_eq!(telemetry.completed_requests, 3);
    assert_eq!(telemetry.failed_requests, 0);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn indexed_texture_loader_rejects_hash_damage_and_bc_portable_artifacts() {
    let root = temporary_store("damage");
    let rgba_bytes = dds_bytes(image_dds::ImageFormat::Rgba8UnormSrgb);
    let portable = install_dds(&root, &rgba_bytes, CookedTextureArtifactFormat::Rgba8Srgb);
    write_index(
        &root,
        vec![index_entry(
            "textures/damaged",
            Some(("portable", "high")),
            &portable,
        )],
    );
    let artifact_path = root.join(&portable.path);
    let mut damaged = rgba_bytes;
    *damaged.last_mut().unwrap() ^= 0x40;
    std::fs::write(&artifact_path, damaged).unwrap();

    let mut loader =
        CookedTextureStoreLoader::new(&root, CookedTextureStoreConfig::default()).unwrap();
    let ticket = loader
        .request(
            CookedTextureStoreRequest::for_runtime_features(
                "textures/damaged",
                "high",
                wgpu::Features::empty(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        wait_for(&mut loader, ticket),
        Err(CookedTextureStoreError::Artifact(message)) if message.contains("texture hash")
    ));
    drop(loader);

    let bc = install_dds(
        &root,
        &dds_bytes(image_dds::ImageFormat::BC7RgbaUnormSrgb),
        CookedTextureArtifactFormat::Bc7Srgb,
    );
    write_index(
        &root,
        vec![index_entry(
            "textures/nonportable",
            Some(("portable", "high")),
            &bc,
        )],
    );
    let mut loader =
        CookedTextureStoreLoader::new(&root, CookedTextureStoreConfig::default()).unwrap();
    let ticket = loader
        .request(
            CookedTextureStoreRequest::for_runtime_features(
                "textures/nonportable",
                "high",
                wgpu::Features::empty(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        wait_for(&mut loader, ticket),
        Err(CookedTextureStoreError::Artifact(message))
            if message.contains("without accepted BC support")
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn resolved_portable_texture_uploads_without_accepted_bc_support() {
    let Some((device, queue)) = try_device(false) else {
        return;
    };
    let root = temporary_store("gpu-upload");
    let portable = install_dds(
        &root,
        &dds_bytes(image_dds::ImageFormat::Rgba8UnormSrgb),
        CookedTextureArtifactFormat::Rgba8Srgb,
    );
    write_index(
        &root,
        vec![index_entry(
            "textures/gpu",
            Some(("portable", "high")),
            &portable,
        )],
    );
    let mut renderer = crate::Renderer::new_headless(device, queue, 16, 16);
    let request = renderer
        .cooked_texture_store_request("textures/gpu", "high")
        .unwrap();
    assert_eq!(request.requested.platform(), "portable");
    let mut loader =
        CookedTextureStoreLoader::new(&root, CookedTextureStoreConfig::default()).unwrap();
    let ticket = loader.request(request).unwrap();
    let resolved = wait_for(&mut loader, ticket).unwrap();

    let mut textures = crate::TextureManager::new();
    let handle = textures
        .load_resolved_cooked_texture(&mut renderer, &resolved)
        .unwrap();
    let loaded = textures.get(handle).unwrap();
    assert_eq!((loaded.width, loaded.height), (8, 8));
    textures.unload_texture(handle, &mut renderer);
    assert!(textures.get(handle).is_none());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn resolved_native_bc_texture_uploads_when_feature_is_accepted() {
    let plan = AdapterAssetProfilePlan::from_bc_support(true);
    if !plan.native_profile_selected() {
        return;
    }
    let Some((device, queue)) = try_device(true) else {
        return;
    };
    let root = temporary_store("gpu-bc-upload");
    let portable = install_dds(
        &root,
        &dds_bytes(image_dds::ImageFormat::Rgba8UnormSrgb),
        CookedTextureArtifactFormat::Rgba8Srgb,
    );
    let native = install_dds(
        &root,
        &dds_bytes(image_dds::ImageFormat::BC7RgbaUnormSrgb),
        CookedTextureArtifactFormat::Bc7Srgb,
    );
    write_index(
        &root,
        vec![
            index_entry("textures/gpu-bc", Some(("portable", "high")), &portable),
            index_entry(
                "textures/gpu-bc",
                Some((plan.runtime_platform(), "high")),
                &native,
            ),
            index_entry(
                "textures/gpu-fallback",
                Some(("portable", "high")),
                &portable,
            ),
        ],
    );

    let mut renderer = crate::Renderer::new_headless(device, queue, 16, 16);
    let request = renderer
        .cooked_texture_store_request("textures/gpu-bc", "high")
        .unwrap();
    assert_eq!(request.requested.platform(), plan.runtime_platform());
    let mut loader =
        CookedTextureStoreLoader::new(&root, CookedTextureStoreConfig::default()).unwrap();
    let ticket = loader.request(request).unwrap();
    let resolved = wait_for(&mut loader, ticket).unwrap();
    assert_eq!(resolved.format, CookedTextureArtifactFormat::Bc7Srgb);
    assert_eq!(resolved.selection.reason, "adapter-native-profile");

    let mut textures = crate::TextureManager::new();
    let handle = textures
        .load_resolved_cooked_texture(&mut renderer, &resolved)
        .unwrap();
    assert_eq!(
        textures
            .get(handle)
            .map(|texture| (texture.width, texture.height)),
        Some((8, 8))
    );
    textures.unload_texture(handle, &mut renderer);

    let fallback_request = renderer
        .cooked_texture_store_request("textures/gpu-fallback", "high")
        .unwrap();
    let fallback_ticket = loader.request(fallback_request).unwrap();
    let fallback = wait_for(&mut loader, fallback_ticket).unwrap();
    assert_eq!(fallback.format, CookedTextureArtifactFormat::Rgba8Srgb);
    assert_eq!(
        fallback.selection.kind,
        CookedTextureSelectionKind::Fallback
    );
    assert_eq!(fallback.selection.fallback_rank, Some(0));
    assert_eq!(
        fallback.selection.reason,
        "portable-fallback-after-native-miss"
    );
    let fallback_handle = textures
        .load_resolved_cooked_texture(&mut renderer, &fallback)
        .unwrap();
    textures.unload_texture(fallback_handle, &mut renderer);
    std::fs::remove_dir_all(root).unwrap();
}

struct InstalledTexture {
    bytes: u64,
    sha256: [u8; 32],
    path: String,
    format: CookedTextureArtifactFormat,
    width: u32,
    height: u32,
    mip_levels: u32,
}

fn dds_bytes(format: image_dds::ImageFormat) -> Vec<u8> {
    let dxgi = match format {
        image_dds::ImageFormat::Rgba8UnormSrgb => DxgiFormat::R8G8B8A8_UNorm_sRGB,
        image_dds::ImageFormat::BC7RgbaUnormSrgb => DxgiFormat::BC7_UNorm_sRGB,
        other => panic!("unsupported test DDS format {other:?}"),
    };
    let mut dds = Dds::new_dxgi(image_dds::ddsfile::NewDxgiParams {
        height: 8,
        width: 8,
        depth: None,
        format: dxgi,
        mipmap_levels: Some(4),
        array_layers: Some(1),
        caps2: None,
        is_cubemap: false,
        resource_dimension: image_dds::ddsfile::D3D10ResourceDimension::Texture2D,
        alpha_mode: image_dds::ddsfile::AlphaMode::Straight,
    })
    .unwrap();
    for (index, byte) in dds.data.iter_mut().enumerate() {
        *byte = index.wrapping_mul(29).wrapping_add(17) as u8;
    }
    let mut bytes = Vec::new();
    dds.write(&mut bytes).unwrap();
    bytes
}

fn install_dds(root: &Path, bytes: &[u8], format: CookedTextureArtifactFormat) -> InstalledTexture {
    let hash = sha256(bytes);
    let path = format!("chunks/sha256/{}.dds", hex_hash(hash));
    std::fs::create_dir_all(root.join("chunks/sha256")).unwrap();
    std::fs::write(root.join(&path), bytes).unwrap();
    let dds = Dds::read(Cursor::new(bytes)).unwrap();
    InstalledTexture {
        bytes: bytes.len() as u64,
        sha256: hash,
        path,
        format,
        width: dds.get_width(),
        height: dds.get_height(),
        mip_levels: dds.get_num_mipmap_levels(),
    }
}

fn index_entry(
    logical_id: &str,
    profile: Option<(&str, &str)>,
    artifact: &InstalledTexture,
) -> Value {
    let mut entry = json!({
        "artifact": {
            "bytes": artifact.bytes,
            "format": artifact.format.name(),
            "height": artifact.height,
            "mip_levels": artifact.mip_levels,
            "path": artifact.path,
            "sha256": hex_hash(artifact.sha256),
            "width": artifact.width,
        },
        "build_key_sha256": hex_hash([7; 32]),
        "kind": "texture",
        "logical_id": logical_id,
        "manifest": {
            "path": format!("manifests/{logical_id}.json"),
            "sha256": hex_hash([11; 32]),
        },
        "source_sha256": hex_hash([13; 32]),
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
    let index = json!({
        "entries": entries,
        "entry_count": entries.len(),
        "profiled_entry_count": profile_count,
        "schema": "bloom-asset-index-v2",
    });
    std::fs::write(
        root.join("index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
}

fn wait_for(
    loader: &mut CookedTextureStoreLoader,
    ticket: CookedTextureStoreTicket,
) -> Result<ResolvedCookedTexture, CookedTextureStoreError> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(result) = loader.poll(ticket) {
            return result;
        }
        assert!(Instant::now() < deadline, "texture store worker timed out");
        std::thread::yield_now();
    }
}

fn temporary_store(label: &str) -> PathBuf {
    let id = NEXT_TEXTURE_STORE_TEST.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "bloom-texture-store-{label}-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn try_device(require_bc: bool) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return None;
    }
    if require_bc
        && !adapter
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
    {
        return None;
    }
    let mut required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    if require_bc {
        required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
    }
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cooked_texture_portable_upload_test"),
        required_features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .ok()
}
