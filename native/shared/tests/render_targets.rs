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
use bloom_shared::scene::SceneGraph;
use bloom_shared::profiler::Profiler;

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

/// The deferred frame path must honor the render-target override — the
/// regression this pins: until 2026-07-17, `end_frame_with_scene` acquired
/// the surface unconditionally and ignored `rt_color_view`, so
/// render-to-texture silently no-oped for every 3D app (the editor's
/// thumbnails drew as nothing; even a bare clear never reached the texture).
///
/// The test shape makes the old behavior impossible to miss: headless there
/// is no surface, so without the override branch the frame early-returns and
/// the RT stays all zeros. With the fix, the frame's final passes write the
/// RT — we read it back and require the clear color to have arrived.
#[test]
fn deferred_frame_writes_render_target_override() {
    let Some(mut r) = try_renderer() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    const W: u32 = 64;
    const H: u32 = 64;
    let (_bg_idx, tex_idx) = r.create_render_texture(W, H);

    // Arm the override exactly the way bloom_begin_texture_mode does.
    let color_view = r
        .get_texture_ref(tex_idx)
        .expect("render texture registered")
        .create_view(&wgpu::TextureViewDescriptor::default());
    let depth_tex = r.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_rt_depth"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    r.rt_depth_view = Some(depth_tex.create_view(&wgpu::TextureViewDescriptor::default()));
    r.rt_depth_texture = Some(depth_tex);
    r.rt_color_view = Some(color_view);
    r.rt_width = W;
    r.rt_height = H;

    // A distinctive clear so "the frame arrived" is unambiguous.
    r.set_clear_color(255.0, 0.0, 255.0, 255.0);

    let mut scene = SceneGraph::new();
    let mut profiler = Profiler::new();
    r.begin_frame();
    r.end_frame_with_scene(&mut scene, &mut profiler);

    // Clear the override (endTextureMode).
    r.end_texture_mode();

    // Read the RT back and look at the center pixel.
    let tex = r.get_texture_ref(tex_idx).expect("render texture registered");
    let bpr = (W * 4 + 255) & !255;
    let buf = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_rt_readback"),
        size: (bpr * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    r.queue.submit(std::iter::once(enc.finish()));

    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = r.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().expect("map callback").expect("map ok");

    let data = slice.get_mapped_range();
    let center = ((H / 2) * bpr + (W / 2) * 4) as usize;
    let px = [data[center], data[center + 1], data[center + 2], data[center + 3]];
    drop(data);

    // Channel order differs by surface format (RGBA vs BGRA) and the clear
    // rides through tonemapping — so the contract is deliberately loose:
    // the pixel must be MAGENTA-ISH (both outer channels well above the
    // middle one), which an untouched all-zeros texture can never satisfy.
    assert!(
        px[0] > 60 && px[2] > 60 && px[1] < px[0] / 2 && px[1] < px[2] / 2,
        "render target did not receive the deferred frame (center pixel {:?}) — \
         end_frame_with_scene is ignoring rt_color_view again",
        px
    );
}
