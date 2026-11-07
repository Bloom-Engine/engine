use bloom_shared::cooked_scene_store::{
    load_cooked_scene_from_store, CookedSceneProfile, CookedSceneStoreConfig,
    CookedSceneStoreRequest,
};
use bloom_shared::cooked_texture_store::{
    CookedTextureProfile, CookedTextureStoreConfig, CookedTextureStoreLoader,
    CookedTextureStoreRequest,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(
            "usage: cooked_scene_store_smoke <store> <logical-id> <platform> <quality>".into(),
        );
    }
    let store = Path::new(&args[1]);
    let requested = CookedSceneProfile::new(&args[3], &args[4])?;
    let request = CookedSceneStoreRequest::new(&args[2], requested);
    let resolved =
        load_cooked_scene_from_store(store, &request, CookedSceneStoreConfig::default())?;
    let scene_report: serde_json::Value = serde_json::from_str(&resolved.report_json())?;
    let texture_profile = match &resolved.selected_profile {
        Some(profile) => Some(CookedTextureProfile::new(
            profile.platform(),
            profile.quality(),
        )?),
        None => None,
    };
    let mut texture_loader =
        CookedTextureStoreLoader::new(store, CookedTextureStoreConfig::default())?;
    let mut texture_handles = Vec::with_capacity(resolved.prepared.texture_dependencies().len());
    let mut texture_bytes = 0u64;
    for (index, dependency) in resolved.prepared.texture_dependencies().iter().enumerate() {
        let mut request = CookedTextureStoreRequest::new(
            &dependency.logical_id,
            texture_profile
                .clone()
                .unwrap_or(CookedTextureProfile::new("portable", &args[4])?),
        );
        if texture_profile.is_none() {
            request = request.allow_unprofiled(true);
        }
        let ticket = texture_loader.request(request)?;
        let texture = loop {
            if let Some(result) = texture_loader.poll(ticket) {
                break result?;
            }
            std::thread::yield_now();
        };
        texture_bytes = texture_bytes.saturating_add(texture.artifact_bytes);
        texture_handles.push(u32::try_from(index + 1)?);
    }
    let cooked = resolved.prepared.finish(&texture_handles)?;
    let unique_primitives = cooked
        .model
        .meshes
        .iter()
        .map(|mesh| Arc::as_ptr(mesh) as usize)
        .collect::<BTreeSet<_>>()
        .len();
    let report = serde_json::json!({
        "animation_clips": cooked.animation.as_ref().map_or(0, |value| value.animations.len()),
        "joints": cooked.animation.as_ref().and_then(|value| value.skeleton.as_ref()).map_or(0, |value| value.joints.len()),
        "placements": cooked.model.meshes.len(),
        "schema": "bloom-runtime-scene-smoke-v1",
        "scene": scene_report,
        "source_gltf_reads": 0,
        "texture_artifact_bytes": texture_bytes,
        "texture_dependencies": texture_handles.len(),
        "unique_primitives": unique_primitives,
        "validation": "pass",
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
