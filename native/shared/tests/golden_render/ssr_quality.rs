use super::*;

fn transformed_box(
    eng: &mut EngineState,
    position: [f32; 3],
    size: [f32; 3],
    color: [f32; 4],
    roughness: f32,
    metalness: f32,
    emissive: [f32; 3],
) {
    let (vertices, indices) = cube_verts(0.5, color);
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_transform(
        node,
        [
            [size[0], 0.0, 0.0, 0.0],
            [0.0, size[1], 0.0, 0.0],
            [0.0, 0.0, size[2], 0.0],
            [position[0], position[1], position[2], 1.0],
        ],
    );
    eng.scene.set_material_pbr(node, roughness, metalness);
    eng.scene
        .set_material_color(node, color[0], color[1], color[2], color[3]);
    eng.scene
        .set_material_emissive_factor(node, emissive[0], emissive[1], emissive[2]);
}

fn luma(pixel: &image::Rgb<u8>) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

fn isolated_hot_pixels(image: &image::RgbImage) -> usize {
    let mut isolated = 0;
    for y in 1..image.height() - 1 {
        for x in 1..image.width() - 1 {
            let center = luma(image.get_pixel(x, y));
            if center < 180.0 {
                continue;
            }
            let mut dark_neighbors = 0;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let neighbor =
                        luma(image.get_pixel(x.wrapping_add_signed(ox), y.wrapping_add_signed(oy)));
                    if neighbor + 70.0 < center {
                        dark_neighbors += 1;
                    }
                }
            }
            isolated += usize::from(dark_neighbors >= 6);
        }
    }
    isolated
}

#[test]
fn dark_interior_ssr_rejects_fireflies_and_preserves_smooth_reflections() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    r.set_taa_enabled(true);
    r.set_render_scale(1.0);
    r.set_ssao_enabled(false);
    r.set_ssgi_enabled(false);
    r.set_ssr_enabled(true);
    r.set_ssr_strength(1.0);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_motion_blur_enabled(false);
    r.set_shadows_enabled(false);

    let concrete = [0.075, 0.085, 0.105, 1.0];
    transformed_box(
        &mut eng,
        [0.0, -0.05, 0.0],
        [8.0, 0.1, 10.0],
        concrete,
        0.68,
        0.0,
        [0.0; 3],
    );
    transformed_box(
        &mut eng,
        [0.0, 3.05, 0.0],
        [8.0, 0.1, 10.0],
        concrete,
        0.68,
        0.0,
        [0.0; 3],
    );
    transformed_box(
        &mut eng,
        [0.0, 1.5, -4.0],
        [8.0, 3.0, 0.1],
        concrete,
        0.68,
        0.0,
        [0.0; 3],
    );
    for x in [-4.0, 4.0] {
        transformed_box(
            &mut eng,
            [x, 1.5, 0.0],
            [0.1, 3.0, 10.0],
            concrete,
            0.68,
            0.0,
            [0.0; 3],
        );
    }
    transformed_box(
        &mut eng,
        [0.0, 1.65, -3.9],
        [2.2, 1.5, 0.08],
        [1.0, 0.92, 0.72, 1.0],
        0.25,
        0.0,
        [2000.0, 1600.0, 900.0],
    );
    // On-screen bright geometry above the floor is the traced-hit negative
    // control. The back-wall opening alone can legitimately miss because SSR
    // cannot reflect content that leaves the viewport.
    transformed_box(
        &mut eng,
        [-0.6, 1.2, -1.3],
        [3.2, 2.4, 0.8],
        [1.0, 0.32, 0.08, 1.0],
        0.2,
        0.0,
        [80.0, 6.0, 0.5],
    );
    transformed_box(
        &mut eng,
        [0.6, 0.02, -0.8],
        [4.8, 0.08, 4.0],
        [0.32, 0.36, 0.42, 1.0],
        0.08,
        1.0,
        [0.0; 3],
    );

    let draw = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(1.0, 1.0, 2.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 4.5, 0.0, 0.6, -0.8, 0.0, 1.0, 0.0, 55.0, 0.0);
        r.set_ambient_light(10.0, 12.0, 18.0, 0.08);
        r.add_directional_light(-0.3, -1.0, -0.2, 0.3, 0.36, 0.5, 0.18);
    };
    let capture = |eng: &mut EngineState| render(eng, 1, draw).2;

    eng.renderer.reset_temporal_history();
    for _ in 0..24 {
        capture(&mut eng);
    }
    let directory = std::env::temp_dir().join(format!("bloom-ssr-interior-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let ssr_enabled = capture(&mut eng);
    let ssr = image::open(directory.join("ssr.png"))
        .expect("SSR qualification capture did not emit its logical graph resource")
        .to_rgb8();
    let active = ssr.pixels().filter(|pixel| luma(pixel) > 10.0).count();
    let hot = ssr.pixels().filter(|pixel| luma(pixel) > 220.0).count();
    let isolated = isolated_hot_pixels(&ssr);
    let metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("ssr.metrics.json"))
            .expect("SSR qualification capture did not emit raw HDR metrics"),
    )
    .unwrap();
    let max_luminance = metrics["max_luminance"].as_f64().unwrap();
    let p999_luminance = metrics["p999_luminance"].as_f64().unwrap();
    let raw_isolated = metrics["isolated_local_outliers"].as_u64().unwrap();
    let non_finite = metrics["non_finite_pixels"].as_u64().unwrap();
    let hit_alpha_pixels = metrics["hit_alpha_pixels"].as_u64().unwrap();
    let raw_metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("ssr-raw.metrics.json"))
            .expect("SSR qualification capture did not emit raw-march HDR metrics"),
    )
    .unwrap();
    let raw_hit_alpha_pixels = raw_metrics["hit_alpha_pixels"].as_u64().unwrap();
    let raw_march_max = raw_metrics["max_luminance"].as_f64().unwrap();
    eprintln!(
        "temporal-corpus dark-interior-ssr active={active} hot={hot} isolated={isolated} \
         raw_max={max_luminance:.4} raw_p999={p999_luminance:.4} \
         raw_isolated={raw_isolated} hit_alpha={hit_alpha_pixels} \
         march_max={raw_march_max:.4} march_hits={raw_hit_alpha_pixels} \
         non_finite={non_finite}"
    );
    assert!(
        active >= 100,
        "negative control produced no SSR reflections"
    );
    assert_eq!(non_finite, 0, "SSR history retained NaN/Inf radiance");
    assert!(
        max_luminance <= 8.01 && raw_march_max <= 8.01,
        "SSR radiance exceeded the documented firefly bound"
    );
    assert_eq!(
        raw_isolated, 0,
        "SSR history contains isolated high-radiance fireflies"
    );

    eng.renderer.set_ssr_enabled(false);
    eng.renderer.reset_temporal_history();
    for _ in 0..16 {
        capture(&mut eng);
    }
    let ssr_disabled = capture(&mut eng);
    let reflection_delta = calculate_diff_metrics(&ssr_disabled, &ssr_enabled, W, H).mean_rgb;
    eprintln!("temporal-corpus dark-interior-ssr reflection_delta={reflection_delta:.4}");
    assert!(
        reflection_delta >= 0.05,
        "smooth metal negative control did not retain a visible SSR contribution"
    );
    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"ssr_diagnostic_persistent_bytes\":0"));
    assert!(paths.contains("\"ssr_diagnostic_capture_passes\":0"));

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept dark-interior SSR diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}
