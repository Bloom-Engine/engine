//! Process-isolated forward/visibility image parity for #27.
//!
//! Runtime mode is selected once during device negotiation, so the parent test
//! launches this test binary twice and compares byte-level screenshots from
//! identical scenes. This exercises the real forward-suppression/composition
//! path rather than comparing two standalone shader helpers.

use bloom_shared::{models::MaterialLayeredPbr, renderer::Vertex3D};
use std::path::Path;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 128;
const CHILD_OUTPUT: &str = "BLOOM_VISIBILITY_PARITY_CHILD_OUTPUT";

fn render_scene(path: &Path) {
    unsafe { std::env::set_var("BLOOM_SKIP_SKY", "1") };
    let mut engine = match bloom_shared::attach::attach_headless_engine(
        wgpu::Backends::PRIMARY,
        WIDTH,
        HEIGHT,
    ) {
        Ok(engine) => engine,
        Err(error) if error.contains("no compatible adapter") => {
            eprintln!("skip: no native GPU adapter ({error})");
            std::fs::write(path, []).expect("write skipped parity marker");
            return;
        }
        Err(error) => panic!("production renderer device negotiation failed: {error}"),
    };
    engine.renderer.set_render_scale(1.0);
    engine.renderer.set_taa_enabled(false);

    let vertices = vec![
        Vertex3D {
            position: [-0.38, -0.34, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.75, 0.35, 0.12, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.38, -0.34, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.12, 0.78, 0.25, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.0, 0.38, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.18, 0.3, 0.9, 1.0],
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
        engine.scene.set_trs(
            node,
            (column - 3.5) * 0.72,
            (row - 1.5) * 0.72,
            0.0,
            0.0,
            1.0,
        );
        engine
            .scene
            .set_material_pbr(node, 0.12 + row * 0.22, 0.05 + column * 0.12);
    }
    let compatibility = engine.scene.create_node();
    engine
        .scene
        .update_geometry(compatibility, vertices, vec![0, 1, 2]);
    engine
        .scene
        .set_trs(compatibility, 0.0, 1.75, 0.0, 0.0, 1.0);
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
    engine.renderer.set_clear_color(0.035, 0.02, 0.055, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(210.0, 225.0, 255.0, 0.55);
    engine.renderer.screenshot_requested = true;
    engine.end_frame();

    let (_, _, mut rgba) = engine
        .renderer
        .screenshot_data
        .take()
        .expect("parity frame produced a screenshot");
    if matches!(
        engine.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    std::fs::write(path, rgba).expect("write parity screenshot bytes");
}

#[test]
fn visibility_shading_child_capture() {
    let Ok(output) = std::env::var(CHILD_OUTPUT) else {
        return;
    };
    render_scene(Path::new(&output));
}

fn run_child(mode: &str, output: &Path) {
    let status = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("visibility_shading_child_capture")
        .arg("--nocapture")
        .env(CHILD_OUTPUT, output)
        .env("BLOOM_VISIBILITY_BUFFER", mode)
        .status()
        .expect("launch isolated visibility parity child");
    assert!(status.success(), "{mode} parity child failed");
}

#[test]
fn visibility_shading_matches_forward_reference() {
    if std::env::var_os(CHILD_OUTPUT).is_some() {
        return;
    }
    let base = std::env::temp_dir().join(format!("bloom-visibility-parity-{}", std::process::id()));
    let forward_path = base.with_extension("forward.rgba");
    let visibility_path = base.with_extension("visibility.rgba");
    run_child("off", &forward_path);
    run_child("shade", &visibility_path);
    let forward = std::fs::read(&forward_path).expect("read forward parity screenshot");
    let visibility = std::fs::read(&visibility_path).expect("read visibility parity screenshot");
    let _ = std::fs::remove_file(forward_path);
    let _ = std::fs::remove_file(visibility_path);
    if forward.is_empty() || visibility.is_empty() {
        eprintln!("skip: parity child had no native GPU adapter");
        return;
    }
    assert_eq!(forward.len(), (WIDTH * HEIGHT * 4) as usize);
    assert_eq!(visibility.len(), forward.len());

    let mut changed_channels = 0usize;
    let mut total_delta = 0u64;
    let mut max_delta = 0u8;
    for (&expected, &actual) in forward.iter().zip(&visibility) {
        let delta = expected.abs_diff(actual);
        changed_channels += usize::from(delta != 0);
        total_delta += u64::from(delta);
        max_delta = max_delta.max(delta);
    }
    let mean_delta = total_delta as f64 / forward.len() as f64;
    eprintln!(
        "visibility parity changed_channels={changed_channels} max_delta={max_delta} mean_delta={mean_delta:.8}"
    );
    // Manual perspective reconstruction can round one code value differently
    // from fixed-function interpolation. Keep that exception both tiny and
    // sparse: no channel may exceed one LSB, under 0.5% may differ at all, and
    // the image-wide mean must stay below 0.005/255.
    assert!(max_delta <= 1, "visibility shading exceeded one-LSB parity");
    assert!(
        changed_channels * 200 <= forward.len(),
        "visibility shading changed at least 0.5% of output channels"
    );
    assert!(
        mean_delta <= 0.005,
        "visibility shading exceeded the sub-LSB mean-error gate"
    );
}
