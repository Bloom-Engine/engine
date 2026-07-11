//! Shared host-surface attach path (PerryTS/perry#5519).
//!
//! Factors the wgpu bring-up — instance → surface → adapter → device →
//! swapchain config → [`Renderer`] → [`EngineState`] — that every
//! platform's `bloom_init_window` duplicates near-verbatim into one
//! helper, so a host application that already owns a native render
//! surface (e.g. Perry UI's `BloomView`: an `NSView`/`UIView`/
//! `GtkWidget`/`ANativeWindow`/`HWND`) can hand it to the engine instead
//! of letting the engine create its own window.
//!
//! Each platform crate exposes a thin `bloom_attach_native(handle, w, h)`
//! FFI that turns the host pointer into the platform's
//! [`wgpu::SurfaceTargetUnsafe`] and calls [`attach_engine`]. The only
//! per-platform deltas — backend bitmask, the raw-handle variant, and the
//! swapchain format policy — are parameters here; the ~120 lines of
//! adapter / feature / limit / device negotiation live in one place.

use crate::engine::EngineState;
use crate::renderer::Renderer;

/// Minimal blocking executor for wgpu's async adapter/device requests.
/// The platform crates each carry a private copy of this (`bloom_init_
/// window` predates this shared module); kept here so the attach path has
/// no extra dependency on `pollster`.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(Box::new(future)) };
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// How [`attach_engine`] picks the swapchain texture format.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FormatPreference {
    /// Prefer an sRGB-capable format (Apple Metal / desktop default —
    /// the renderer writes linear color and relies on the swapchain for
    /// the sRGB encode).
    Srgb,
    /// Prefer a *non*-sRGB format, falling back to the first reported
    /// (tvOS / visionOS: those backends double-encode if handed an sRGB
    /// swapchain, so the renderer does the encode itself).
    NonSrgb,
    /// Take the adapter's first reported format unchanged. GL / some
    /// mobile surfaces don't expose an sRGB variant and fail to
    /// configure if one is forced (Linux / Windows).
    First,
}

/// Inputs to [`attach_engine`]. Sizes are split into *logical* (the
/// points / DIPs the engine reasons in) and *physical* (the backing
/// pixels the swapchain allocates) so HiDPI hosts pass both; non-HiDPI
/// hosts pass equal values.
pub struct AttachParams {
    /// Backends to instantiate (e.g. `wgpu::Backends::METAL`, or
    /// `VULKAN | GL` on Linux/Android).
    pub backends: wgpu::Backends,
    pub logical_w: u32,
    pub logical_h: u32,
    pub physical_w: u32,
    pub physical_h: u32,
    pub format: FormatPreference,
}

/// Build a fully-configured [`EngineState`] that renders into a
/// host-owned surface. This is the GPU half of `bloom_init_window` with
/// the windowing half removed: the caller supplies the surface target,
/// we own the instance / adapter / device / swapchain and the engine.
///
/// Returns `Err` with a human-readable reason instead of panicking, so a
/// host that attaches to a not-yet-realized view can surface the failure
/// rather than abort the process.
///
/// # Safety
/// `target` must reference a live native view / window / layer / surface
/// that outlives the returned [`EngineState`]; the host owns it and must
/// not free it while the engine renders.
pub unsafe fn attach_engine(
    target: wgpu::SurfaceTargetUnsafe,
    params: AttachParams,
) -> Result<EngineState, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: params.backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let surface = instance
        .create_surface_unsafe(target)
        .map_err(|e| format!("create_surface failed: {e}"))?;

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| format!("no compatible adapter: {e}"))?;

    // Optional device features, requested only when the adapter offers
    // them (mirrors bloom_init_window): GPU-timestamp profiling, BC
    // texture compression, and HW ray query for the GI probe path.
    let supported = adapter.features();
    let mut required_features = wgpu::Features::empty();
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if supported.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
    }
    let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let rt_mask = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    if !force_sw_gi && supported.contains(rt_mask) {
        required_features |= rt_mask;
    }
    let experimental_features = if required_features.intersects(rt_mask) {
        // wgpu 29 requires this explicit opt-in token for EXPERIMENTAL_*
        // features. Apple-Silicon Metal ray query has been stable since
        // wgpu v25, so the documented UB risk is acceptable here.
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };

    // The material ABI declares 5 bind groups; wgpu defaults to 4. Every
    // real backend supports >= 7.
    let adapter_limits = adapter.limits();
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_bind_groups = 5;

    // A user material's fragment stage binds more sampled textures than
    // wgpu's default permits — 19 against a default cap of 16, once the
    // scene/GI inputs sit alongside the material's own maps.
    //
    // Only the fallback below ever picked up the adapter's real limits, and
    // it runs solely when request_device *fails*. On macOS the default
    // request succeeds at 16, so the shortfall surfaced much later, as an
    // abort inside create_pipeline_layout('user_material') — the shooter
    // could not open on macOS at all. (iOS escaped it by luck: its first
    // request fails on an unrelated limit, so it always retried with the
    // adapter's limits and got the headroom as a side effect.)
    //
    // Ask for what the adapter actually offers on the limits the material
    // system leans on, never dropping below wgpu's defaults.
    required_limits.max_sampled_textures_per_shader_stage = required_limits
        .max_sampled_textures_per_shader_stage
        .max(adapter_limits.max_sampled_textures_per_shader_stage);
    required_limits.max_samplers_per_shader_stage = required_limits
        .max_samplers_per_shader_stage
        .max(adapter_limits.max_samplers_per_shader_stage);

    if required_features.intersects(rt_mask) {
        required_limits =
            required_limits.using_minimum_supported_acceleration_structure_values();
    }

    let device_desc = wgpu::DeviceDescriptor {
        label: Some("bloom_device"),
        required_features,
        required_limits: required_limits.clone(),
        experimental_features,
        ..Default::default()
    };

    // Some constrained mobile GPUs (e.g. A18) report a feature/limit set
    // they then refuse at device-create time. Retry once with the
    // adapter's own reported limits + no optional features before giving
    // up — matches the iOS init path's fallback.
    let (device, queue) = match block_on(adapter.request_device(&device_desc)) {
        Ok(pair) => pair,
        Err(first) => {
            let fallback = wgpu::DeviceDescriptor {
                label: Some("bloom_device_fallback"),
                required_features: wgpu::Features::empty(),
                required_limits: {
                    let mut l = adapter.limits();
                    l.max_bind_groups = l.max_bind_groups.max(5);
                    l
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                ..Default::default()
            };
            block_on(adapter.request_device(&fallback)).map_err(|second| {
                format!("request_device failed: {first}; fallback: {second}")
            })?
        }
    };

    let surface_caps = surface.get_capabilities(&adapter);
    if surface_caps.formats.is_empty() {
        return Err("surface reports no supported formats".to_string());
    }
    let format = match params.format {
        FormatPreference::Srgb => surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]),
        FormatPreference::NonSrgb => surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]),
        FormatPreference::First => surface_caps.formats[0],
    };

    let physical_w = params.physical_w.max(1);
    let physical_h = params.physical_h.max(1);
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format,
        width: physical_w,
        height: physical_h,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let renderer = Renderer::new(
        device,
        queue,
        surface,
        surface_config,
        params.logical_w.max(1),
        params.logical_h.max(1),
    );
    Ok(EngineState::new(renderer))
}
