//! Render-target sizing invariant.
//!
//! The pass chain computes viewports and compute dispatches from
//! `render_extent()` (surface * render_scale) and writes into the
//! depth/HDR/G-buffer targets. Those two must agree from the very first
//! frame.
//!
//! The golden suite cannot cover this: its harness calls
//! `set_taa_enabled(false)` immediately after construction, which
//! internally resizes and so always repairs the invariant before
//! anything renders. A real host does not — every `attach_engine`
//! platform (macOS/iOS/tvOS/Linux/Android) boots and then only resizes
//! when the OS-reported window size *changes*, which is false on frame
//! 1. That left those platforms rendering against targets sized to the
//! full surface while the passes ran at half extent: post-FX silently
//! dropped out, and the depth-snapshot copy failed validation outright
//! whenever a scene-reading translucent material was in view.

use bloom_shared::renderer::Renderer;

fn try_renderer() -> Option<Renderer> {
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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some(Renderer::new_headless(device, queue, 256, 256))
}

/// Straight out of the constructor — no resize, no TAA toggle, which is
/// exactly what a freshly attached host looks like on frame 1.
#[test]
fn render_targets_match_render_extent_at_construction() {
    let Some(r) = try_renderer() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    assert_eq!(
        r.render_target_extent(),
        r.render_extent(),
        "render targets disagree with render_extent straight out of the \
         constructor — the boot path is not honouring render_scale"
    );
}

/// The invariant must survive the scale/TAA setters and an explicit
/// resize, not just hold at construction.
#[test]
fn render_targets_track_render_extent_through_changes() {
    let Some(mut r) = try_renderer() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    r.set_render_scale(1.0);
    assert_eq!(r.render_target_extent(), r.render_extent(), "after set_render_scale(1.0)");

    r.set_render_scale(0.5);
    assert_eq!(r.render_target_extent(), r.render_extent(), "after set_render_scale(0.5)");

    r.resize(640, 480, 640, 480);
    assert_eq!(r.render_target_extent(), r.render_extent(), "after resize");

    r.set_taa_enabled(false);
    assert_eq!(r.render_target_extent(), r.render_extent(), "after set_taa_enabled(false)");
}
