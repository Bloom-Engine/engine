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
const CHILD_MRT_DIR: &str = "BLOOM_VISIBILITY_PARITY_CHILD_MRT_DIR";

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
    let mut eligible_nodes = Vec::with_capacity(32);
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
        eligible_nodes.push((node, column, row));
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

    // Seed retained transform history. The captured frame moves both the
    // visibility-eligible population and a forward-only layered draw, so the
    // raw velocity target proves reconstruction parity rather than merely
    // comparing an all-zero static attachment.
    engine.begin_frame();
    engine.renderer.set_clear_color(0.035, 0.02, 0.055, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(210.0, 225.0, 255.0, 0.55);
    engine.end_frame();

    for (node, column, row) in eligible_nodes {
        engine.scene.set_trs(
            node,
            (column - 3.5) * 0.72 + 0.035 + row * 0.004,
            (row - 1.5) * 0.72 - 0.018,
            0.0,
            0.0,
            1.0,
        );
    }
    engine
        .scene
        .set_trs(compatibility, 0.026, 1.73, 0.0, 0.0, 1.0);

    engine.begin_frame();
    engine.renderer.set_clear_color(0.035, 0.02, 0.055, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(210.0, 225.0, 255.0, 0.55);
    engine.renderer.pending_mrt_capture_dir =
        Some(std::env::var(CHILD_MRT_DIR).expect("MRT parity child capture directory is set"));
    engine.renderer.screenshot_requested = true;
    engine.end_frame();
    assert!(
        engine.renderer.pending_mrt_capture_dir.is_none(),
        "MRT qualification request must be one-shot"
    );

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

fn run_child(mode: &str, output: &Path, mrt_dir: &Path) {
    let status = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("visibility_shading_child_capture")
        .arg("--nocapture")
        .env(CHILD_OUTPUT, output)
        .env(CHILD_MRT_DIR, mrt_dir)
        .env("BLOOM_VISIBILITY_BUFFER", mode)
        .status()
        .expect("launch isolated visibility parity child");
    assert!(status.success(), "{mode} parity child failed");
}

fn read_attachment(directory: &Path, name: &str, bytes_per_pixel: usize) -> Vec<u8> {
    let bytes = std::fs::read(directory.join(format!("{name}.raw")))
        .unwrap_or_else(|error| panic!("read {name} MRT attachment: {error}"));
    assert_eq!(
        bytes.len(),
        WIDTH as usize * HEIGHT as usize * bytes_per_pixel
    );
    bytes
}

fn assert_unorm_parity(label: &str, forward: &[u8], visibility: &[u8]) {
    assert_eq!(visibility.len(), forward.len());
    let mut changed = 0usize;
    let mut max_delta = 0u8;
    let mut total_delta = 0u64;
    for (&expected, &actual) in forward.iter().zip(visibility) {
        let delta = expected.abs_diff(actual);
        changed += usize::from(delta != 0);
        max_delta = max_delta.max(delta);
        total_delta += u64::from(delta);
    }
    let mean_delta = total_delta as f64 / forward.len().max(1) as f64;
    eprintln!(
        "visibility MRT {label} changed_components={changed}/{} max_delta={max_delta} mean_delta={mean_delta:.9}",
        forward.len()
    );
    assert!(max_delta <= 1, "{label} exceeded one UNORM code value");
    assert!(
        changed * 200 <= forward.len(),
        "{label} changed at least 0.5% of components"
    );
}

fn half_values(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .chunks_exact(2)
        .map(|value| half::f16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
}

fn assert_half_parity(
    label: &str,
    forward: &[u8],
    visibility: &[u8],
    max_abs_gate: f32,
    mean_abs_gate: f64,
) {
    assert_eq!(visibility.len(), forward.len());
    let mut changed = 0usize;
    let mut max_abs = 0.0f32;
    let mut total_abs = 0.0f64;
    let mut count = 0usize;
    for (expected, actual) in half_values(forward).zip(half_values(visibility)) {
        assert!(
            expected.is_finite() && actual.is_finite(),
            "{label} contains non-finite values"
        );
        let delta = (expected - actual).abs();
        changed += usize::from(delta != 0.0);
        max_abs = max_abs.max(delta);
        total_abs += f64::from(delta);
        count += 1;
    }
    let mean_abs = total_abs / count.max(1) as f64;
    eprintln!(
        "visibility MRT {label} changed_components={changed}/{count} max_abs={max_abs:.9} mean_abs={mean_abs:.12}"
    );
    assert!(
        max_abs <= max_abs_gate,
        "{label} exceeded absolute-error gate"
    );
    assert!(
        mean_abs <= mean_abs_gate,
        "{label} exceeded mean-error gate"
    );
}

#[test]
fn visibility_shading_matches_forward_reference() {
    if std::env::var_os(CHILD_OUTPUT).is_some() {
        return;
    }
    let base = std::env::temp_dir().join(format!("bloom-visibility-parity-{}", std::process::id()));
    let forward_path = base.with_extension("forward.rgba");
    let visibility_path = base.with_extension("visibility.rgba");
    let forward_mrt = base.with_extension("forward-mrt");
    let visibility_mrt = base.with_extension("visibility-mrt");
    run_child("off", &forward_path, &forward_mrt);
    run_child("shade", &visibility_path, &visibility_mrt);
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

    for directory in [&forward_mrt, &visibility_mrt] {
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("scene-mrt.json")).expect("read MRT manifest"),
        )
        .expect("MRT manifest is JSON");
        assert_eq!(manifest["schema"], "bloom-mrt-capture-v1");
        assert_eq!(manifest["width"], WIDTH);
        assert_eq!(manifest["height"], HEIGHT);
        assert_eq!(manifest["attachments"].as_array().map(Vec::len), Some(4));
    }

    let forward_hdr = read_attachment(&forward_mrt, "hdr-scene", 8);
    let visibility_hdr = read_attachment(&visibility_mrt, "hdr-scene", 8);
    assert_half_parity("hdr-scene", &forward_hdr, &visibility_hdr, 0.01, 0.000_01);
    let forward_material = read_attachment(&forward_mrt, "material-properties", 2);
    let visibility_material = read_attachment(&visibility_mrt, "material-properties", 2);
    assert_unorm_parity(
        "material-properties",
        &forward_material,
        &visibility_material,
    );
    let forward_velocity = read_attachment(&forward_mrt, "motion-vectors", 4);
    let visibility_velocity = read_attachment(&visibility_mrt, "motion-vectors", 4);
    let moving_components = half_values(&forward_velocity)
        .filter(|value| value.abs() > 1.0e-5)
        .count();
    assert!(
        moving_components > 256,
        "velocity parity scene did not exercise moving retained geometry"
    );
    assert_half_parity(
        "motion-vectors",
        &forward_velocity,
        &visibility_velocity,
        0.001,
        0.000_001,
    );
    let forward_albedo = read_attachment(&forward_mrt, "albedo", 4);
    let visibility_albedo = read_attachment(&visibility_mrt, "albedo", 4);
    assert_unorm_parity("albedo", &forward_albedo, &visibility_albedo);

    let _ = std::fs::remove_dir_all(forward_mrt);
    let _ = std::fs::remove_dir_all(visibility_mrt);
}
