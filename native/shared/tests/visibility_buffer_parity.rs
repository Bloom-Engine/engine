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
const COLUMN_SPACING: f32 = 0.55;
const ROW_SPACING: f32 = 0.8;
const CHILD_OUTPUT: &str = "BLOOM_VISIBILITY_PARITY_CHILD_OUTPUT";
const CHILD_MRT_DIR: &str = "BLOOM_VISIBILITY_PARITY_CHILD_MRT_DIR";

fn layered_compatibility_materials() -> [MaterialLayeredPbr; 6] {
    let material = |lobe_mask| {
        MaterialLayeredPbr::from_authoring_factors(
            lobe_mask,
            0.82,
            0.24,
            1.0,
            0.42,
            [0.65, 0.9, 1.15],
            1.82,
            [0.62, 0.16, 0.04],
            0.38,
            0.86,
            0.61,
            0.92,
            1.34,
            120.0,
            430.0,
        )
    };
    [
        material(MaterialLayeredPbr::SPECULAR_IOR_LOBE),
        material(MaterialLayeredPbr::CLEARCOAT_LOBE),
        material(MaterialLayeredPbr::SHEEN_LOBE),
        material(MaterialLayeredPbr::ANISOTROPY_LOBE),
        material(MaterialLayeredPbr::IRIDESCENCE_LOBE),
        material(
            MaterialLayeredPbr::CLEARCOAT_LOBE
                | MaterialLayeredPbr::SPECULAR_IOR_LOBE
                | MaterialLayeredPbr::SHEEN_LOBE
                | MaterialLayeredPbr::ANISOTROPY_LOBE
                | MaterialLayeredPbr::IRIDESCENCE_LOBE,
        ),
    ]
}

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
            position: [-0.22, -0.34, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.75, 0.35, 0.12, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.22, -0.34, 0.0],
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
    let mut textured_vertices = vertices.clone();
    for vertex in &mut textured_vertices {
        vertex.uv[0] *= 6.0;
        vertex.uv[1] *= 6.0;
    }
    let mut textured_no_tangent = textured_vertices.clone();
    for vertex in &mut textured_no_tangent {
        vertex.tangent = [0.0; 4];
    }

    const MATERIAL_TEXTURE_SIZE: u32 = 8;
    let mut base_texels =
        Vec::with_capacity((MATERIAL_TEXTURE_SIZE * MATERIAL_TEXTURE_SIZE * 4) as usize);
    let mut normal_texels = Vec::with_capacity(base_texels.capacity());
    let mut mr_texels = Vec::with_capacity(base_texels.capacity());
    let mut emissive_texels = Vec::with_capacity(base_texels.capacity());
    for y in 0..MATERIAL_TEXTURE_SIZE {
        for x in 0..MATERIAL_TEXTURE_SIZE {
            let checker = (x + y) & 1;
            base_texels.extend_from_slice(&[
                35 + x as u8 * 24,
                45 + y as u8 * 22,
                if checker == 0 { 210 } else { 75 },
                255,
            ]);
            normal_texels.extend_from_slice(&[
                if checker == 0 { 92 } else { 164 },
                if (x / 2 + y) & 1 == 0 { 105 } else { 151 },
                244,
                255,
            ]);
            mr_texels.extend_from_slice(&[255, 40 + y as u8 * 25, 30 + x as u8 * 28, 255]);
            emissive_texels.extend_from_slice(&[
                if checker == 0 { 28 } else { 4 },
                if checker == 0 { 8 } else { 24 },
                12,
                255,
            ]);
        }
    }
    let base_texture = engine.renderer.register_texture_kind(
        MATERIAL_TEXTURE_SIZE,
        MATERIAL_TEXTURE_SIZE,
        &base_texels,
        false,
    );
    let normal_texture = engine.renderer.register_texture_kind(
        MATERIAL_TEXTURE_SIZE,
        MATERIAL_TEXTURE_SIZE,
        &normal_texels,
        true,
    );
    let mr_texture = engine.renderer.register_texture_kind(
        MATERIAL_TEXTURE_SIZE,
        MATERIAL_TEXTURE_SIZE,
        &mr_texels,
        false,
    );
    let emissive_texture = engine.renderer.register_texture_kind(
        MATERIAL_TEXTURE_SIZE,
        MATERIAL_TEXTURE_SIZE,
        &emissive_texels,
        false,
    );
    let mut eligible_nodes = Vec::with_capacity(32);
    for index in 0..32 {
        let node = engine.scene.create_node();
        let node_vertices = if index % 4 == 0 {
            textured_no_tangent.clone()
        } else if index % 2 == 0 {
            textured_vertices.clone()
        } else {
            vertices.clone()
        };
        engine
            .scene
            .update_geometry(node, node_vertices, vec![0, 1, 2]);
        let column = (index % 8) as f32;
        let row = (index / 8) as f32;
        engine.scene.set_trs(
            node,
            (column - 3.5) * COLUMN_SPACING,
            (row - 1.5) * ROW_SPACING,
            0.0,
            0.0,
            1.0,
        );
        engine
            .scene
            .set_material_pbr(node, 0.12 + row * 0.22, 0.05 + column * 0.12);
        if index % 2 == 0 {
            engine.scene.set_material_texture(node, base_texture);
            engine
                .scene
                .set_material_normal_texture(node, normal_texture);
            engine
                .scene
                .set_material_metallic_roughness_texture(node, mr_texture);
            engine
                .scene
                .set_material_emissive_texture(node, emissive_texture);
            engine
                .scene
                .set_material_emissive_factor(node, 0.7, 0.4, 0.55);
        }
        eligible_nodes.push((node, column, row));
    }
    // Place one forward-only surface for every scalar layered lobe plus the
    // combined material just in front of six final-row visibility triangles.
    // Shade mode records the underlying visibility IDs in the prepass, then
    // compatibility rendering replaces their depth. The fullscreen
    // visibility pass must preserve the final owner and every layered result.
    let layered_materials = layered_compatibility_materials();
    assert!(layered_materials[0].specular_authored && layered_materials[0].ior_authored);
    assert!(layered_materials[1].clearcoat_authored);
    assert!(layered_materials[2].sheen_authored);
    assert!(layered_materials[3].anisotropy_authored);
    assert!(layered_materials[4].iridescence_authored);
    assert!(
        layered_materials[5].clearcoat_authored
            && layered_materials[5].specular_authored
            && layered_materials[5].sheen_authored
            && layered_materials[5].anisotropy_authored
            && layered_materials[5].iridescence_authored
    );
    let mut compatibility_nodes = Vec::with_capacity(layered_materials.len());
    for (index, material) in layered_materials.into_iter().enumerate() {
        let node = engine.scene.create_node();
        engine
            .scene
            .update_geometry(node, vertices.clone(), vec![0, 1, 2]);
        let column = index as f32 + 1.0;
        engine.scene.set_trs(
            node,
            (column - 3.5) * COLUMN_SPACING,
            1.5 * ROW_SPACING,
            0.05,
            0.0,
            1.0,
        );
        engine.scene.set_material_color(
            node,
            0.35 + index as f32 * 0.075,
            0.18 + index as f32 * 0.045,
            0.75 - index as f32 * 0.07,
            1.0,
        );
        engine.scene.set_material_pbr(node, 0.38, 0.2);
        engine.scene.set_material_layered_pbr(node, material);
        compatibility_nodes.push((node, column));
    }
    let mask_texture = engine.renderer.register_texture_kind(
        2,
        2,
        &[
            245, 210, 45, 255, 30, 150, 60, 0, 30, 150, 60, 0, 245, 210, 45, 255,
        ],
        false,
    );
    let cutout = engine.scene.create_node();
    engine
        .scene
        .update_geometry(cutout, vertices, vec![0, 1, 2]);
    // The MASK triangle overlaps the first eligible triangle in the final
    // row. Opaque mask texels own final depth; discarded texels must continue
    // to reveal visibility-shaded geometry underneath.
    engine.scene.set_trs(
        cutout,
        -3.5 * COLUMN_SPACING,
        1.5 * ROW_SPACING,
        0.05,
        0.0,
        1.0,
    );
    engine.scene.set_material_texture(cutout, mask_texture);
    engine.scene.set_material_alpha_cutoff(cutout, 0.5);

    // Seed retained transform history. The captured frame moves both the
    // visibility-eligible population and a forward-only layered draw, so the
    // raw velocity target proves reconstruction parity rather than merely
    // comparing an all-zero static attachment.
    engine.begin_frame();
    engine.renderer.set_clear_color(0.035, 0.02, 0.055, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(210.0, 225.0, 255.0, 0.25);
    engine
        .renderer
        .set_directional_light(0.35, 0.42, 1.0, 255.0, 235.0, 215.0, 1.35);
    engine.end_frame();

    for (node, column, row) in eligible_nodes {
        engine.scene.set_trs(
            node,
            (column - 3.5) * COLUMN_SPACING + 0.035 + row * 0.004,
            (row - 1.5) * ROW_SPACING - 0.018,
            0.0,
            0.0,
            1.0,
        );
    }
    let final_row_x_motion = 0.035 + 3.0 * 0.004;
    for (node, column) in compatibility_nodes {
        engine.scene.set_trs(
            node,
            (column - 3.5) * COLUMN_SPACING + final_row_x_motion,
            1.5 * ROW_SPACING - 0.018,
            0.05,
            0.0,
            1.0,
        );
    }
    engine.scene.set_trs(
        cutout,
        -3.5 * COLUMN_SPACING + final_row_x_motion,
        1.5 * ROW_SPACING - 0.018,
        0.05,
        0.0,
        1.0,
    );

    engine.begin_frame();
    engine.renderer.set_clear_color(0.035, 0.02, 0.055, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(210.0, 225.0, 255.0, 0.25);
    engine
        .renderer
        .set_directional_light(0.35, 0.42, 1.0, 255.0, 235.0, 215.0, 1.35);
    let mrt_capture_dir =
        std::env::var(CHILD_MRT_DIR).expect("MRT parity child capture directory is set");
    engine.renderer.pending_mrt_capture_dir = Some(mrt_capture_dir.clone());
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
    std::fs::write(
        Path::new(&mrt_capture_dir).join("runtime-paths.json"),
        engine.renderer.quality_runtime_paths_json(),
    )
    .expect("write parity runtime paths");
    std::fs::write(
        Path::new(&mrt_capture_dir).join("capability-report.json"),
        engine.renderer.renderer_capability_report_json(),
    )
    .expect("write parity capability report");
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

        let paths: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("runtime-paths.json"))
                .expect("read layered runtime paths"),
        )
        .expect("layered runtime paths are JSON");
        let layered = &paths["layered_pbr"];
        let granted = layered["granted_sampled_textures_per_stage"]
            .as_u64()
            .expect("layered path reports the granted sampled-texture limit");
        let required = layered["scene_specialization_required_sampled_textures"]
            .as_u64()
            .expect("layered path reports its specialization requirement");
        let specialization_available = layered["scene_specialization_available"]
            .as_bool()
            .expect("layered path reports specialization availability");
        assert_eq!(
            specialization_available,
            granted >= required,
            "layered specialization selection must match the negotiated texture limit"
        );
        assert_eq!(
            layered["scene_specialization_initialized"], specialization_available,
            "the optional layered scene path must initialize exactly when supported"
        );
        // The sheen LUT belongs to the optional layered specialization and is
        // therefore lazy on adapters whose sampled-texture budget cannot fit
        // that path.  Compatibility draws must still be retained below, but a
        // constrained adapter must not be required to allocate an unreachable
        // specialization resource merely to satisfy this qualification test.
        assert_eq!(
            layered["sheen_lut_initialized"], specialization_available,
            "the sheen LUT must follow the availability of its scene specialization"
        );

        if directory == &visibility_mrt {
            let capabilities: serde_json::Value = serde_json::from_slice(
                &std::fs::read(directory.join("capability-report.json"))
                    .expect("read visibility capability report"),
            )
            .expect("visibility capability report is JSON");
            let gpu_driven = &capabilities["runtime_support"]["gpu_driven"];
            let visibility_runtime = &gpu_driven["visibility_buffer_runtime"];
            assert_eq!(visibility_runtime["requested_mode"], "shade");
            if visibility_runtime["enabled"] == true {
                assert_eq!(visibility_runtime["pbr_shading"], true);
                assert_eq!(visibility_runtime["forward_authoritative"], false);
                assert_eq!(
                    visibility_runtime["composition"],
                    "visibility-eligible+forward-compatibility"
                );
                assert!(visibility_runtime["eligible_draws"]
                    .as_u64()
                    .is_some_and(|draws| draws >= 32));
                assert!(visibility_runtime["compatibility_draws"]
                    .as_u64()
                    .is_some_and(|draws| draws >= 6));
                assert_eq!(visibility_runtime["frame_recorded"], true);
            } else {
                // Visibility shading is optional on adapters lacking primitive
                // indices or Tier-A material indirection. In that state there
                // is no routed draw list, so both counters must remain zero;
                // the byte/MRT comparisons below qualify the explicit,
                // forward-authoritative fallback instead.
                let reason = visibility_runtime["disabled_reason"]
                    .as_str()
                    .expect("disabled visibility runtime reports a reason");
                assert!(
                    matches!(
                        reason,
                        "primitive-index-unavailable"
                            | "gpu-driven-unavailable"
                            | "tier-a-materials-unavailable"
                    ),
                    "unexpected visibility fallback reason: {reason}"
                );
                assert_eq!(visibility_runtime["pbr_shading"], false);
                assert_eq!(visibility_runtime["forward_authoritative"], true);
                assert_eq!(visibility_runtime["composition"], "forward-authoritative");
                assert_eq!(visibility_runtime["eligible_draws"], 0);
                assert_eq!(visibility_runtime["compatibility_draws"], 0);
                assert_eq!(visibility_runtime["allocated_bytes"], 0);
                assert_eq!(visibility_runtime["frame_recorded"], false);
                if reason == "gpu-driven-unavailable" {
                    assert_eq!(gpu_driven["enabled"], false);
                }
            }
        }
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
