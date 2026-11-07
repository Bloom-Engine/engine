//! Manual end-to-end qualification for an indexed cooked texture package.

use bloom_shared::cooked_texture_store::{
    CookedTextureStoreConfig, CookedTextureStoreLoader, CookedTextureStoreTicket,
    ResolvedCookedTexture,
};
use bloom_shared::{Renderer, TextureManager};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let store = PathBuf::from(
        args.next()
            .ok_or("usage: cooked_texture_store_smoke STORE LOGICAL_ID [QUALITY]")?,
    );
    let logical_id = args
        .next()
        .ok_or("usage: cooked_texture_store_smoke STORE LOGICAL_ID [QUALITY]")?;
    let quality = args.next().unwrap_or_else(|| "high".to_string());
    if args.next().is_some() {
        return Err("usage: cooked_texture_store_smoke STORE LOGICAL_ID [QUALITY]".into());
    }

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let accepted_optional =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_COMPRESSION_BC;
    let required_features = adapter.features() & accepted_optional;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cooked_texture_store_smoke"),
        required_features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))?;
    let mut renderer = Renderer::new_headless(device, queue, 16, 16);
    let request = renderer.cooked_texture_store_request(logical_id, &quality)?;
    let mut loader = CookedTextureStoreLoader::new(store, CookedTextureStoreConfig::default())?;
    let ticket = loader.request(request)?;
    let resolved = wait_for(&mut loader, ticket)?;
    let mut textures = TextureManager::new();
    let handle = textures.load_resolved_cooked_texture(&mut renderer, &resolved)?;
    let texture = textures
        .get(handle)
        .ok_or("uploaded texture handle was not registered")?;
    if (texture.width, texture.height) != (resolved.width, resolved.height) {
        return Err("uploaded texture dimensions disagree with the selected artifact".into());
    }
    println!("{}", resolved.report_json());
    textures.unload_texture(handle, &mut renderer);
    Ok(())
}

fn wait_for(
    loader: &mut CookedTextureStoreLoader,
    ticket: CookedTextureStoreTicket,
) -> Result<ResolvedCookedTexture, Box<dyn std::error::Error>> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(result) = loader.poll(ticket) {
            return result.map_err(Into::into);
        }
        if Instant::now() >= deadline {
            return Err("cooked texture store worker timed out".into());
        }
        std::thread::yield_now();
    }
}
