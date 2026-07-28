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
fn request_device(
    instance: &wgpu::Instance,
    compatible_surface: Option<&wgpu::Surface<'_>>,
) -> Result<
    (
        wgpu::Adapter,
        wgpu::Device,
        wgpu::Queue,
        crate::renderer::device_negotiation::DeviceNegotiationReport,
    ),
    String,
> {
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface,
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| format!("no compatible adapter: {e}"))?;

    let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let negotiated = block_on(
        crate::renderer::device_negotiation::request_device_with_fallback(
            &adapter,
            crate::renderer::device_negotiation::DeviceRequestOptions {
                allow_ray_query: !force_sw_gi,
            },
        ),
    )?;
    eprintln!(
        "bloom: renderer device negotiation = {}",
        negotiated.report.report_json()
    );
    Ok((
        adapter,
        negotiated.device,
        negotiated.queue,
        negotiated.report,
    ))
}

/// Build the same production renderer without a presentation surface.
/// Used only by explicit batch/headless hosts, which need exact pixels and
/// must not depend on window-system DPI or drawable availability.
pub fn attach_headless_engine(
    backends: wgpu::Backends,
    width: u32,
    height: u32,
) -> Result<EngineState, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let (_adapter, device, queue, negotiation) = request_device(&instance, None)?;
    let mut renderer = Renderer::new_headless(device, queue, width.max(1), height.max(1));
    renderer.set_device_negotiation_report(negotiation.report_json());
    Ok(EngineState::new(renderer))
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

    let (adapter, device, queue, negotiation) = request_device(&instance, Some(&surface))?;

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

    let mut renderer = Renderer::new(
        device,
        queue,
        surface,
        surface_config,
        params.logical_w.max(1),
        params.logical_h.max(1),
    );
    renderer.set_device_negotiation_report(negotiation.report_json());
    Ok(EngineState::new(renderer))
}
