//! Opt-in #27 full-PBR visibility shading smoke.
//!
//! This test has its own process because device feature selection reads the
//! runtime mode once. Eligible forward fragments are suppressed in `shade`
//! mode, so visible green coverage proves the reconstructed fragment inputs
//! reached Bloom's authoritative PBR evaluator and all four MRT outputs.

use bloom_shared::{models::MaterialLayeredPbr, renderer::Vertex3D};

#[test]
fn opt_in_visibility_shading_replaces_eligible_forward_pixels() {
    unsafe { std::env::set_var("BLOOM_VISIBILITY_BUFFER", "shade") };
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
            "skip: requested visibility shading unavailable ({})",
            runtime["disabled_reason"]
        );
        return;
    }

    let vertices = vec![
        Vertex3D {
            position: [-0.35, -0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.1, 0.9, 0.2, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.35, -0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.1, 0.9, 0.2, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.0, 0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.1, 0.9, 0.2, 1.0],
            uv: [0.5, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
    ];
    for index in 0..32 {
        let node = engine.scene.create_node();
        engine
            .scene
            .update_geometry(node, vertices.clone(), vec![0, 1, 2]);
        let column = (index % 8) as f32;
        let row = (index / 8) as f32;
        engine
            .scene
            .set_trs(node, (column - 3.5) * 0.7, (row - 1.5) * 0.7, 0.0, 0.0, 1.0);
    }
    let compatibility = engine.scene.create_node();
    engine
        .scene
        .update_geometry(compatibility, vertices.clone(), vec![0, 1, 2]);
    engine.scene.set_trs(compatibility, 0.0, 1.8, 0.0, 0.0, 1.0);
    engine
        .scene
        .set_material_color(compatibility, 0.8, 0.15, 0.7, 1.0);
    engine.scene.set_material_layered_pbr(
        compatibility,
        MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 0.8,
            clearcoat_roughness_factor: 0.25,
            ..Default::default()
        },
    );

    engine.begin_frame();
    engine.renderer.set_clear_color(0.08, 0.01, 0.01, 1.0);
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
        .expect("visibility-shaded frame produced a screenshot");
    if matches!(
        engine.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    let green_pixels = rgba
        .chunks_exact(4)
        .filter(|pixel| {
            pixel[1] > pixel[0].saturating_add(30)
                && pixel[1] > pixel[2].saturating_add(30)
                && pixel[3] > 200
        })
        .count();
    assert!(
        green_pixels > 256,
        "eligible forward pixels were suppressed but visibility PBR produced no coverage"
    );

    let report: serde_json::Value =
        serde_json::from_str(&engine.renderer.renderer_capability_report_json())
            .expect("post-frame capability report is JSON");
    let runtime = &report["runtime_support"]["gpu_driven"]["visibility_buffer_runtime"];
    assert_eq!(runtime["requested_mode"], "shade");
    assert_eq!(runtime["pbr_shading"], true);
    assert_eq!(runtime["forward_authoritative"], false);
    assert_eq!(
        runtime["composition"],
        "visibility-eligible+forward-compatibility"
    );
    assert!(runtime["eligible_draws"]
        .as_u64()
        .is_some_and(|value| value >= 32));
    assert_eq!(runtime["compatibility_draws"], 1);
    assert_eq!(runtime["allocated_bytes"], 128 * 128 * 8);

    engine.begin_frame();
    engine.renderer.set_clear_color(0.08, 0.01, 0.01, 1.0);
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
}
