//! Opt-in #27 runtime smoke.
//!
//! This is its own integration-test process because device feature selection
//! reads BLOOM_VISIBILITY_BUFFER exactly once at startup. The test exercises
//! production device negotiation, shared scene geometry, the depth-equal ID
//! raster, full-screen reconstruction, and the debug composition pass.

use bloom_shared::renderer::Vertex3D;

#[test]
fn opt_in_runtime_raster_and_reconstruction_execute_on_real_gpu() {
    // This test binary contains one test, so mutating its process environment
    // cannot race another test. Production hosts set the same value before
    // attaching the engine.
    unsafe { std::env::set_var("BLOOM_VISIBILITY_BUFFER", "debug") };
    unsafe { std::env::set_var("BLOOM_SKIP_SKY", "1") };
    let mut engine =
        match bloom_shared::attach::attach_headless_engine(wgpu::Backends::PRIMARY, 128, 128) {
            Ok(engine) => engine,
            Err(error) if error.contains("no compatible adapter") => {
                eprintln!("skip: no native GPU adapter ({error})");
                return;
            }
            Err(error) => panic!("production renderer device negotiation failed: {error}"),
        };
    engine.renderer.set_render_scale(1.0);
    engine.renderer.set_taa_enabled(false);

    let initial: serde_json::Value =
        serde_json::from_str(&engine.renderer.renderer_capability_report_json())
            .expect("initial capability report is JSON");
    let runtime = &initial["runtime_support"]["gpu_driven"]["visibility_buffer_runtime"];
    if runtime["enabled"] != true {
        eprintln!(
            "skip: requested visibility runtime unavailable ({})",
            runtime["disabled_reason"]
        );
        return;
    }

    let vertices = vec![
        Vertex3D {
            position: [-0.35, -0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.25, 0.8, 0.4, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.35, -0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.25, 0.8, 0.4, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.0, 0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.25, 0.8, 0.4, 1.0],
            uv: [0.5, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
    ];
    let indices = vec![0, 1, 2];
    let mut nodes = Vec::with_capacity(32);
    for index in 0..32 {
        let node = engine.scene.create_node();
        nodes.push(node);
        engine
            .scene
            .update_geometry(node, vertices.clone(), indices.clone());
        let column = (index % 8) as f32;
        let row = (index / 8) as f32;
        engine
            .scene
            .set_trs(node, (column - 3.5) * 0.7, (row - 1.5) * 0.7, 0.0, 0.0, 1.0);
    }

    engine.begin_frame();
    engine.renderer.set_clear_color(0.02, 0.03, 0.05, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(255.0, 255.0, 255.0, 1.0);
    engine.renderer.screenshot_requested = true;
    engine.end_frame();

    let (_, _, mut rgba) = engine
        .renderer
        .screenshot_data
        .take()
        .expect("debug visibility frame produced a screenshot");
    if matches!(
        engine.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    let reconstructed_normal_pixels = rgba
        .chunks_exact(4)
        .filter(|pixel| {
            pixel[2] > pixel[0].saturating_add(20)
                && pixel[2] > pixel[1].saturating_add(20)
                && pixel[3] > 200
        })
        .count();
    assert!(
        reconstructed_normal_pixels > 256,
        "debug overlay exposed no meaningful reconstructed-normal coverage"
    );

    let report: serde_json::Value =
        serde_json::from_str(&engine.renderer.renderer_capability_report_json())
            .expect("post-frame capability report is JSON");
    let runtime = &report["runtime_support"]["gpu_driven"]["visibility_buffer_runtime"];
    assert_eq!(runtime["enabled"], true);
    assert_eq!(runtime["forward_authoritative"], true);
    assert_eq!(runtime["debug_overlay"], true);
    assert!(runtime["eligible_draws"]
        .as_u64()
        .is_some_and(|value| value >= 32));
    assert_eq!(runtime["compatibility_draws"], 0);
    assert_eq!(runtime["width"], 128);
    assert_eq!(runtime["height"], 128);
    assert_eq!(runtime["allocated_bytes"], 128 * 128 * 16);
    assert_eq!(runtime["frame_recorded"], true);

    // A steady frame must reuse the diagnostic textures and bind group. The
    // first frame was a screenshot frame and is deliberately excluded from
    // the engine's steady-state resource snapshot.
    engine.begin_frame();
    engine.renderer.set_clear_color(0.02, 0.03, 0.05, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(255.0, 255.0, 255.0, 1.0);
    engine.end_frame();

    let steady: serde_json::Value =
        serde_json::from_str(&engine.renderer.quality_runtime_paths_json())
            .expect("steady-state runtime report is JSON");
    assert_eq!(
        steady["steady_state_resources"]["bind_group_creations"]["sites"]["visibility_buffer"],
        0
    );
    assert_eq!(
        steady["steady_state_resources"]["transient_physical_creations"]["textures"],
        0
    );

    // If an eligible frame is followed by one without eligible geometry, the
    // overlay must not replay the previous reconstruction.
    for node in nodes {
        engine.scene.set_visible(node, false);
    }
    engine.begin_frame();
    engine.renderer.set_clear_color(0.08, 0.02, 0.02, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.screenshot_requested = true;
    engine.end_frame();

    let (_, _, mut empty_rgba) = engine
        .renderer
        .screenshot_data
        .take()
        .expect("empty visibility frame produced a screenshot");
    if matches!(
        engine.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in empty_rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    let stale_normal_pixels = empty_rgba
        .chunks_exact(4)
        .filter(|pixel| {
            pixel[2] > pixel[0].saturating_add(20)
                && pixel[2] > pixel[1].saturating_add(20)
                && pixel[3] > 200
        })
        .count();
    assert_eq!(stale_normal_pixels, 0, "stale reconstructed-normal overlay");

    let empty_report: serde_json::Value =
        serde_json::from_str(&engine.renderer.renderer_capability_report_json())
            .expect("empty-frame capability report is JSON");
    let empty_runtime = &empty_report["runtime_support"]["gpu_driven"]["visibility_buffer_runtime"];
    assert_eq!(empty_runtime["frame_recorded"], false);
    assert_eq!(empty_runtime["eligible_draws"], 0);
}
