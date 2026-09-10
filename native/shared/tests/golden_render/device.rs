//! Raster test device ownership and mandatory hardware qualification.
//!
//! Keep a device alive across renderer lifetimes, as the PT oracle already does.
//! Creating a new native instance/device for every capture can exhaust backend
//! resources halfway through the corpus. Each renderer still owns fresh scene,
//! history, and render-graph state. Memory accounting uses an isolated instance.

use super::{EngineState, Renderer, H, W};
use std::sync::OnceLock;

struct RasterContext {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

static RASTER_DEVICE: OnceLock<Result<Option<RasterContext>, String>> = OnceLock::new();

pub(super) fn requested_backends() -> wgpu::Backends {
    let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
    assert!(
        !backends.is_empty(),
        "WGPU_BACKEND selects no valid backend"
    );
    backends
}

fn hardware_required() -> bool {
    ["BLOOM_REQUIRE_GPU", "BLOOM_REQUIRE_RAY_QUERY"]
        .iter()
        .any(|name| {
            std::env::var(name)
                .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        })
}

fn create_context() -> Result<Option<RasterContext>, String> {
    let backends = requested_backends();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let adapter = if let Some(index) = adapters
        .iter()
        .position(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
    {
        Some(adapters.swap_remove(index))
    } else {
        // Some Metal headless configurations enumerate no devices but still
        // provide a physical adapter through request_adapter.
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()
            .filter(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
    };
    let Some(adapter) = adapter else {
        return Ok(None);
    };
    let info = adapter.get_info();
    let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|error| {
        format!(
            "raster adapter '{}' ({:?}) exists but device creation failed: {error}",
            info.name, info.backend
        )
    })?;
    eprintln!(
        "Raster golden adapter: {} ({:?}, {:?}), driver={} {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
    Ok(Some(RasterContext {
        instance,
        device,
        queue,
    }))
}

fn fresh_engine(context: &RasterContext) -> EngineState {
    let renderer = Renderer::new_headless(context.device.clone(), context.queue.clone(), W, H);
    let mut eng = EngineState::new(renderer);
    eng.renderer.set_taa_enabled(false);
    eng.renderer.set_render_scale(1.0);
    eng
}

fn unavailable() -> Option<EngineState> {
    assert!(
        !hardware_required(),
        "hardware qualification requires a non-CPU raster adapter for {:?}",
        requested_backends()
    );
    None
}

pub(super) fn try_engine() -> Option<EngineState> {
    match RASTER_DEVICE.get_or_init(create_context) {
        Ok(Some(context)) => Some(fresh_engine(context)),
        Ok(None) => unavailable(),
        Err(error) => panic!("{error}"),
    }
}

pub(super) fn try_isolated_engine() -> Option<(EngineState, wgpu::Instance)> {
    match create_context() {
        Ok(Some(context)) => Some((fresh_engine(&context), context.instance)),
        Ok(None) => {
            unavailable();
            None
        }
        Err(error) => panic!("{error}"),
    }
}
