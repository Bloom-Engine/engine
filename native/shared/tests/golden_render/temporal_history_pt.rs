use super::*;

#[test]
fn path_tracing_mode_transitions_reset_incompatible_history() {
    let _rt_guard = lock_rt_goldens();
    let (mut eng, _) = match try_engine_rt() {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            skip_rt_golden("pt_history_lifetime", "no-non-cpu-ray-query-adapter");
            return;
        }
        Err(err) => panic!("{err}"),
    };
    build_pt_scene(&mut eng);

    eng.renderer.set_path_tracing(2);
    let _ = render(&mut eng, 1, draw_pt_static_frame);
    assert!(eng.renderer.path_tracing_sample_count() > 0);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ray_scene_preparation\":\"ssgi+pt\""));

    eng.renderer.set_path_tracing(1);
    assert_eq!(eng.renderer.path_tracing_sample_count(), 0);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"pt_samples\":0,\"pt_index\":0"));

    eng.renderer.set_path_tracing(0);
    assert_eq!(eng.renderer.path_tracing_sample_count(), 0);
}

#[test]
fn realtime_path_tracing_capture_exposes_svgf_history_without_normal_frame_resources() {
    let _rt_guard = lock_rt_goldens();
    let (mut eng, _) = match try_engine_rt() {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            skip_rt_golden("pt_temporal_capture", "no-non-cpu-ray-query-adapter");
            return;
        }
        Err(err) => panic!("{err}"),
    };
    build_pt_scene(&mut eng);
    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(false);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_path_tracing(2);
    r.set_path_tracing_debug_view(0);
    r.set_path_tracing_seed(0);
    r.reset_path_tracing_history(0);

    let mut frame = 0u32;
    let _ = render(&mut eng, 24, |eng| {
        draw_pt_motion_frame(eng, frame);
        frame += 1;
    });
    let samples_before_capture = eng.renderer.path_tracing_sample_count();
    assert!(
        samples_before_capture >= 8,
        "realtime PT reached only {samples_before_capture} history frames before capture"
    );
    let normal_paths = eng.renderer.quality_runtime_paths_json();
    assert!(normal_paths.contains("\"ray_scene_preparation\":\"pt\""));
    assert!(normal_paths.contains("\"pt_diagnostic_persistent_bytes\":0"));
    assert!(normal_paths.contains("\"pt_diagnostic_resources_live\":false"));

    let directory =
        std::env::temp_dir().join(format!("bloom-pt-diagnostics-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    eng.begin_frame();
    draw_pt_motion_frame(&mut eng, frame);
    eng.end_frame();
    assert!(
        eng.renderer.path_tracing_sample_count() > samples_before_capture,
        "qualification frame did not execute the realtime PT pass"
    );

    let reasons = image::open(directory.join("pt-rejection-reason.png"))
        .expect("PT capture did not emit temporal rejection reasons")
        .to_rgb8();
    let accepted = reasons
        .pixels()
        .filter(|pixel| {
            (pixel[0] < 40 && pixel[1] > 140 && pixel[2] < 60) || (pixel[0] < 40 && pixel[2] > 200)
        })
        .count();
    let motion = image::open(directory.join("pt-motion.png"))
        .expect("PT capture did not emit motion vectors")
        .to_rgb8();
    let reprojection = image::open(directory.join("pt-reprojected-uv.png"))
        .expect("PT capture did not emit reprojected UVs")
        .to_rgb8();
    let valid_reprojection = reprojection.pixels().filter(|pixel| pixel[2] > 200).count();
    let confidence = image::open(directory.join("pt-temporal-confidence.png"))
        .expect("PT capture did not emit temporal confidence")
        .to_rgb8();
    let accumulated = confidence
        .pixels()
        .filter(|pixel| pixel[1] > 16 && pixel[2] > 16)
        .count();
    let metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("hdr-scene.metrics.json"))
            .expect("PT capture did not emit raw HDR metrics"),
    )
    .unwrap();
    let non_finite = metrics["non_finite_pixels"].as_u64().unwrap();
    let max_luminance = metrics["max_luminance"].as_f64().unwrap();
    eprintln!(
        "temporal-corpus pt-svgf accepted={accepted} valid_reprojection={valid_reprojection} \
         accumulated={accumulated} non_finite={non_finite} max_luma={max_luminance:.4} total={}",
        reasons.width() * reasons.height()
    );
    assert!(
        accepted >= 100 && valid_reprojection >= 100 && accumulated >= 100,
        "settled realtime PT exposed no accepted, reprojected, accumulated history"
    );
    assert_eq!(non_finite, 0, "realtime PT emitted non-finite HDR radiance");
    assert!(max_luminance > 0.0001, "realtime PT produced no radiance");
    assert_eq!(reasons.dimensions(), motion.dimensions());
    assert_eq!(reasons.dimensions(), reprojection.dimensions());
    assert_eq!(reasons.dimensions(), confidence.dimensions());

    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"pt_diagnostic_persistent_bytes\":0"));
    assert!(paths.contains("\"pt_diagnostic_capture_passes\":1"));
    assert!(paths.contains("\"pt_diagnostic_resources_live\":false"));
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept PT diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn realtime_path_tracing_rigid_motion_bounds_trails_and_keeps_history() {
    fn transform(x: f32, angle: f32) -> [[f32; 4]; 4] {
        let (sin, cos) = angle.sin_cos();
        [
            [cos, 0.0, -sin, 0.0],
            [0.0, 1.4, 0.0, 0.0],
            [sin, 0.0, cos, 0.0],
            [x, 1.0, -0.4, 1.0],
        ]
    }

    let _rt_guard = lock_rt_goldens();
    let (mut eng, _) = match try_engine_rt() {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            skip_rt_golden("pt_rigid_motion", "no-non-cpu-ray-query-adapter");
            return;
        }
        Err(err) => panic!("{err}"),
    };
    build_pt_scene(&mut eng);
    let (vertices, indices) = cube_verts(0.7, [0.95, 0.06, 0.02, 1.0]);
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_material_pbr(node, 0.15, 0.3);
    eng.scene.set_material_color(node, 0.95, 0.06, 0.02, 1.0);
    eng.scene.set_transform(node, transform(-2.0, -0.65));

    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(false);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_path_tracing(2);
    r.set_path_tracing_debug_view(0);
    r.set_path_tracing_seed(0);
    r.reset_path_tracing_history(0);
    let _ = render(&mut eng, 24, draw_pt_static_frame);
    let old_pose = render(&mut eng, 1, draw_pt_static_frame).2;

    eng.scene.set_transform(node, transform(2.0, 0.8));
    let directory =
        std::env::temp_dir().join(format!("bloom-pt-rigid-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(render(&mut eng, 1, draw_pt_static_frame).2);
    }
    evaluate_motion_recovery("pt-rigid", &old_pose, &frames);

    let motion = image::open(directory.join("pt-motion.png"))
        .expect("moving PT capture did not emit motion vectors")
        .to_rgb8();
    let moving = motion.pixels().filter(|pixel| pixel[2] > 16).count();
    let reasons = image::open(directory.join("pt-rejection-reason.png"))
        .expect("moving PT capture did not emit rejection reasons")
        .to_rgb8();
    let mut motion_history = 0usize;
    let mut motion_rejected = 0usize;
    let mut motion_flip = 0usize;
    for (motion, reason) in motion.pixels().zip(reasons.pixels()) {
        if motion[2] <= 16 {
            continue;
        }
        motion_history +=
            usize::from(reason[0] < 40 && reason[1] > 40 && reason[1] < 100 && reason[2] > 220);
        motion_rejected +=
            usize::from(reason[0] > 220 && reason[1] < 40 && (reason[2] > 160 || reason[2] < 40));
        motion_flip += usize::from(reason[0] < 40 && reason[1] > 200 && reason[2] > 220);
    }
    let classified_motion = motion_history + motion_rejected + motion_flip;
    eprintln!(
        "temporal-corpus pt-rigid moving={moving} retained={motion_history} \
         rejected={motion_rejected} footprint_flip={motion_flip} total={}",
        motion.width() * motion.height()
    );
    assert!(moving >= 100, "rigid PT motion wrote no velocity coverage");
    assert!(
        motion_history >= 25,
        "overlapping rigid PT motion retained no reprojected history"
    );
    assert!(
        classified_motion * 10 >= moving * 9,
        "moving PT texels were neither retained nor explicitly rejected"
    );
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept PT rigid-motion diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn realtime_path_tracing_lighting_changes_converge_without_reset_or_lag() {
    fn evaluate_lighting(label: &str, previous: &[u8], frames: &[Vec<u8>]) {
        let stable = average_rgba(&frames[12..]);
        let change = calculate_diff_metrics(previous, &stable, W, H);
        let recovery = frames[..13]
            .iter()
            .map(|frame| calculate_diff_metrics(&stable, frame, W, H))
            .collect::<Vec<_>>();
        let stable_flicker = frames[12..]
            .iter()
            .map(|frame| calculate_diff_metrics(&stable, frame, W, H).mean_rgb)
            .sum::<f64>()
            / (frames.len() - 12) as f64;
        eprintln!(
            "temporal-corpus {label} change_mean={:.4} initial_mean={:.4} \
             frame4_mean={:.4} frame8_mean={:.4} frame12_outliers={:.4}% \
             stable_flicker={stable_flicker:.4}",
            change.mean_rgb,
            recovery[0].mean_rgb,
            recovery[4].mean_rgb,
            recovery[8].mean_rgb,
            recovery[12].outlier_pixel_fraction * 100.0,
        );
        assert!(
            change.mean_rgb >= 1.0 && change.outlier_pixel_fraction >= 0.01,
            "{label} negative control did not produce a visible lighting change"
        );
        assert!(
            recovery[8].mean_rgb <= recovery[0].mean_rgb * 0.65 + 0.25,
            "{label} retained stale lighting beyond eight frames"
        );
        assert!(
            recovery[12].outlier_pixel_fraction <= 0.02,
            "{label} retained coherent stale lighting after twelve frames"
        );
        assert!(
            stable_flicker <= 2.0,
            "{label} did not settle to a stable stochastic estimate"
        );
    }

    let _rt_guard = lock_rt_goldens();
    let (mut eng, _) = match try_engine_rt() {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            skip_rt_golden("pt_lighting_change", "no-non-cpu-ray-query-adapter");
            return;
        }
        Err(err) => panic!("{err}"),
    };
    build_pt_scene(&mut eng);
    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(false);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_path_tracing(2);
    r.set_path_tracing_debug_view(0);
    r.set_path_tracing_seed(0);
    r.reset_path_tracing_history(0);
    let draw = |eng: &mut EngineState, bright: bool| {
        draw_pt_static_frame(eng);
        eng.renderer.set_directional_light(
            0.5,
            1.0,
            0.3,
            255.0,
            242.25,
            229.5,
            if bright { 2.4 } else { 0.15 },
        );
    };
    let capture_state =
        |eng: &mut EngineState, bright: bool| render(eng, 1, |eng| draw(eng, bright)).2;

    let _ = render(&mut eng, 24, |eng| draw(eng, false));
    let dark = capture_state(&mut eng, false);
    let before_bright = eng.renderer.path_tracing_sample_count();
    let mut bright_frames = Vec::new();
    for _ in 0..24 {
        bright_frames.push(capture_state(&mut eng, true));
    }
    assert_eq!(
        eng.renderer.path_tracing_sample_count(),
        before_bright + 24,
        "lighting change reset realtime PT history"
    );
    evaluate_lighting("pt-light-on", &dark, &bright_frames);

    let bright = average_rgba(&bright_frames[12..]);
    let before_dark = eng.renderer.path_tracing_sample_count();
    let mut dark_frames = Vec::new();
    for _ in 0..24 {
        dark_frames.push(capture_state(&mut eng, false));
    }
    assert_eq!(
        eng.renderer.path_tracing_sample_count(),
        before_dark + 24,
        "lighting removal reset realtime PT history"
    );
    evaluate_lighting("pt-light-off", &bright, &dark_frames);
}

#[test]
fn realtime_path_tracing_resets_are_byte_exact_fresh_seeds() {
    let _rt_guard = lock_rt_goldens();
    let (mut eng, _) = match try_engine_rt() {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            skip_rt_golden("pt_reset_seed", "no-non-cpu-ray-query-adapter");
            return;
        }
        Err(err) => panic!("{err}"),
    };
    build_pt_scene(&mut eng);
    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(false);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_path_tracing(2);
    r.set_path_tracing_debug_view(0);
    r.set_path_tracing_seed(0);
    let draw = |eng: &mut EngineState, camera: [f32; 3]| {
        let r = &mut eng.renderer;
        r.set_clear_color(0.05, 0.07, 0.1, 1.0);
        r.begin_mode_3d(
            camera[0], camera[1], camera[2], 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 50.0, 0.0,
        );
        r.set_directional_light(0.5, 1.0, 0.3, 255.0, 242.25, 229.5, 1.2);
    };
    let capture =
        |eng: &mut EngineState, camera: [f32; 3]| render(eng, 1, |eng| draw(eng, camera)).2;
    let camera_a = [-5.5, 3.2, 5.0];
    let camera_b = [5.0, 4.0, 7.0];

    // Drain shared card/TLAS warm-up before establishing the seed oracle.
    let _ = render(&mut eng, 8, |eng| draw(eng, camera_b));
    eng.renderer.reset_temporal_history();
    let fresh_b = capture(&mut eng, camera_b);
    assert_eq!(eng.renderer.path_tracing_sample_count(), 1);

    let _ = render(&mut eng, 16, |eng| draw(eng, camera_a));
    eng.renderer.reset_temporal_history();
    let directory =
        std::env::temp_dir().join(format!("bloom-pt-reset-seed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let cut_b = capture(&mut eng, camera_b);
    let cut_metrics = calculate_diff_metrics(&fresh_b, &cut_b, W, H);
    assert_eq!(
        cut_metrics.max_diff, 0,
        "explicit PT reset retained pixels from the prior camera"
    );
    assert_eq!(eng.renderer.path_tracing_sample_count(), 1);
    let reasons = image::open(directory.join("pt-rejection-reason.png"))
        .expect("PT reset capture did not emit rejection reasons")
        .to_rgb8();
    let non_seed = reasons
        .pixels()
        .filter(|pixel| pixel[0].abs_diff(pixel[1]) > 2 || pixel[1].abs_diff(pixel[2]) > 2)
        .count();
    assert_eq!(
        non_seed, 0,
        "fresh PT history was not entirely classified as seed/sky"
    );

    let _ = render(&mut eng, 16, |eng| draw(eng, camera_a));
    eng.renderer.set_path_tracing(0);
    let _ = capture(&mut eng, camera_a);
    eng.renderer.set_path_tracing(2);
    let toggled_b = capture(&mut eng, camera_b);
    let toggle_metrics = calculate_diff_metrics(&fresh_b, &toggled_b, W, H);
    eprintln!(
        "temporal-corpus pt-reset cut_max={} toggle_max={} non_seed={non_seed}",
        cut_metrics.max_diff, toggle_metrics.max_diff,
    );
    assert_eq!(
        toggle_metrics.max_diff, 0,
        "PT off/on transition retained pixels from the prior ownership epoch"
    );
    assert_eq!(eng.renderer.path_tracing_sample_count(), 1);
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept PT reset diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}
