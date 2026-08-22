use super::*;

#[path = "ssr_quality.rs"]
mod ssr_quality;
#[path = "temporal_history_helpers.rs"]
mod temporal_history_helpers;
pub(super) use temporal_history_helpers::{
    average_rgba, configure_taa_motion_corpus, evaluate_motion_recovery, severe_pixel_fraction,
};

#[test]
fn ssr_history_lifetime_is_independent_from_the_taa_frame_counter() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_taa_enabled(true);

    let draw_frame = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(13.0, 18.0, 26.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.95, 0.9, 1.2);
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
        r.draw_sphere(0.0, 0.75, 0.0, 0.75, 220.0, 228.0, 240.0, 255.0);
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_frame(eng);
            eng.end_frame();
        }
    };

    advance(&mut eng, 3);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":true"));

    eng.renderer.set_ssr_enabled(false);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
    advance(&mut eng, 2);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));

    eng.renderer.set_ssr_enabled(true);
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":true"));

    eng.renderer.set_ssr_strength(0.75);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":true"));

    eng.renderer.set_path_tracing(1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
    eng.renderer.set_path_tracing(0);
    eng.renderer.set_render_scale(0.75);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
}

#[test]
fn ssgi_probe_history_tracks_only_frames_that_write_it() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_taa_enabled(false);
    let draw_frame = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(13.0, 18.0, 26.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.95, 0.9, 1.2);
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
        r.draw_sphere(0.0, 0.75, 0.0, 0.75, 220.0, 228.0, 240.0, 255.0);
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_frame(eng);
            eng.end_frame();
        }
    };

    advance(&mut eng, 1);
    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"ssgi_probe_valid\":true"));
    let first_probe_frame = eng.renderer.probe_frame_index;
    advance(&mut eng, 2);
    assert_eq!(
        eng.renderer.probe_frame_index,
        first_probe_frame + 2,
        "SSGI angular sampling must advance while TAA is disabled"
    );
    eng.renderer.set_taa_enabled(true);
    eng.renderer.set_taa_enabled(false);
    assert_eq!(
        eng.renderer.probe_frame_index,
        first_probe_frame + 2,
        "TAA toggles must not reset the independent SSGI sequence"
    );

    eng.renderer.set_ssgi_enabled(false);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":false,\"ssgi_probe_index\":0"));
    advance(&mut eng, 2);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":false,\"ssgi_probe_index\":0"));

    eng.renderer.set_ssgi_enabled(true);
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":true"));

    eng.renderer.set_ssgi_intensity(0.75);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":false,\"ssgi_probe_index\":0"));
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":true"));

    eng.renderer.set_ssgi_radius(12.0);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":false,\"ssgi_probe_index\":0"));
    eng.renderer.set_path_tracing(1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":false,\"ssgi_probe_index\":0"));
    eng.renderer.set_path_tracing(0);
    eng.renderer.set_render_scale(0.75);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_probe_valid\":false,\"ssgi_probe_index\":0"));
}

#[test]
fn taa_history_lifetime_is_explicit_across_toggles_and_resize() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_taa_enabled(true);
    let draw_frame = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(13.0, 18.0, 26.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_frame(eng);
            eng.end_frame();
        }
    };

    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"taa_valid\":true"));

    if eng.renderer.pt_supported() {
        eng.renderer.set_path_tracing(2);
        advance(&mut eng, 1);
        assert!(eng
            .renderer
            .quality_runtime_paths_json()
            .contains("\"taa_valid\":true,\"taa_index\":1,\"taa_pt_owned\":true"));
        eng.renderer.set_path_tracing(0);
        advance(&mut eng, 1);
        assert!(eng
            .renderer
            .quality_runtime_paths_json()
            .contains("\"taa_pt_owned\":false"));
    }

    eng.renderer.set_taa_enabled(false);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"taa_valid\":false,\"taa_index\":0"));
    advance(&mut eng, 2);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"taa_valid\":false,\"taa_index\":0"));

    eng.renderer.set_taa_enabled(true);
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"taa_valid\":true"));

    eng.renderer.set_render_scale(0.75);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"taa_valid\":false,\"taa_index\":0"));
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"taa_valid\":true"));
}

#[test]
fn exposure_history_seeds_each_enable_epoch_without_advancing_while_off() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let draw_frame = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(13.0, 18.0, 26.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_frame(eng);
            eng.end_frame();
        }
    };

    eng.renderer.set_auto_exposure(true);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"exposure_valid\":false,\"exposure_index\":0"));
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"exposure_valid\":true,\"exposure_index\":1"));

    eng.renderer.set_auto_exposure(false);
    advance(&mut eng, 2);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"exposure_valid\":false,\"exposure_index\":0"));

    eng.renderer.set_auto_exposure_rate(0.0);
    eng.renderer.set_auto_exposure(true);
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"exposure_valid\":true,\"exposure_index\":1"));
}

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

#[test]
fn common_camera_cut_reset_invalidates_every_temporal_owner() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_taa_enabled(true);
    eng.renderer.set_ssao_enabled(true);
    eng.renderer.set_ssr_enabled(true);
    eng.renderer.set_ssgi_enabled(true);
    eng.renderer.set_auto_exposure(true);

    let draw_frame = |eng: &mut EngineState, fov: f32| {
        let r = &mut eng.renderer;
        r.set_clear_color(13.0, 18.0, 26.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, fov, 0.0);
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
    };
    for _ in 0..2 {
        eng.begin_frame();
        draw_frame(&mut eng, 45.0);
        eng.end_frame();
    }
    let before = eng.renderer.quality_runtime_paths_json();
    assert!(before.contains("\"ssr_valid\":true"));
    assert!(before.contains("\"ssgi_probe_valid\":true"));
    assert!(before.contains("\"taa_valid\":true"));
    assert!(before.contains("\"exposure_valid\":true"));

    eng.renderer.reset_temporal_history();
    let reset = eng.renderer.quality_runtime_paths_json();
    assert!(reset.contains("\"ssr_valid\":false"));
    assert!(reset.contains("\"ssgi_probe_valid\":false"));
    assert!(reset.contains("\"taa_valid\":false"));
    assert!(reset.contains("\"exposure_valid\":false"));
    assert!(reset.contains("\"pt_samples\":0,\"pt_index\":0"));
    assert!(reset.contains("\"ssao_frames\":0,\"ssao_index\":0"));
    assert!(reset.contains("\"camera_cut_pending\":true,\"camera_cut_active\":false"));

    eng.begin_frame();
    draw_frame(&mut eng, 70.0);
    eng.end_frame();
    let after = eng.renderer.quality_runtime_paths_json();
    assert!(after.contains("\"camera_cut_pending\":false,\"camera_cut_active\":true"));
    assert!(after.contains("\"taa_valid\":true"));
    assert!(after.contains("\"ssr_valid\":true"));
    assert!(after.contains("\"ssgi_probe_valid\":true"));
    assert!(after.contains("\"exposure_valid\":true"));
}

#[test]
fn taa_capture_emits_per_pixel_diagnostics_without_retaining_resources() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_taa_enabled(true);
    eng.renderer.set_ssao_enabled(false);
    eng.renderer.set_ssr_enabled(false);
    eng.renderer.set_ssgi_enabled(false);
    eng.renderer.set_bloom_enabled(false);
    eng.renderer.set_shadows_enabled(false);
    let (reactive_vertices, reactive_indices) = cube_verts(0.7, [0.1, 0.8, 1.0, 0.95]);
    let reactive_node = eng.scene.create_node();
    eng.scene
        .update_geometry(reactive_node, reactive_vertices, reactive_indices);
    eng.scene.set_transform(
        reactive_node,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.25, 0.0, 1.0],
        ],
    );
    eng.scene
        .set_material_gltf_alpha(reactive_node, MaterialAlphaMode::Blend, 0.0, false);
    eng.scene
        .set_material_color(reactive_node, 0.1, 0.8, 1.0, 0.95);
    let draw_frame = |eng: &mut EngineState, camera_x: f32| {
        let r = &mut eng.renderer;
        r.set_clear_color(13.0, 18.0, 26.0, 255.0);
        r.begin_mode_3d(
            4.0 + camera_x,
            3.0,
            6.0,
            camera_x,
            0.5,
            0.0,
            0.0,
            1.0,
            0.0,
            45.0,
            0.0,
        );
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
        r.draw_sphere(0.0, 0.75, 0.0, 0.75, 220.0, 228.0, 240.0, 255.0);
    };
    for _ in 0..6 {
        eng.begin_frame();
        draw_frame(&mut eng, 0.0);
        eng.end_frame();
    }

    let directory =
        std::env::temp_dir().join(format!("bloom-taa-diagnostics-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    eng.begin_frame();
    draw_frame(&mut eng, 0.25);
    eng.end_frame();

    for name in [
        "taa-rejection-reason",
        "taa-motion",
        "taa-reprojected-uv",
        "taa-temporal-confidence",
        "taa-reactive-history",
        "taa-history-policy",
    ] {
        let path = directory.join(format!("{name}.png"));
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("missing diagnostic {path:?}: {error}"));
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(
            bytes.len() > 64,
            "diagnostic {path:?} is unexpectedly empty"
        );
        if name == "taa-rejection-reason" {
            let pixels = image::open(&path).unwrap().to_rgb8();
            let first = pixels.get_pixel(0, 0);
            assert!(
                pixels.pixels().any(|pixel| pixel != first),
                "rejection map must distinguish at least two per-pixel outcomes"
            );
            let palette = [
                [64u8, 64, 64],
                [255, 13, 5],
                [0, 230, 255],
                [255, 0, 204],
                [255, 191, 0],
                [13, 64, 255],
                [13, 166, 26],
            ];
            let mut counts = [0usize; 7];
            for pixel in pixels.pixels() {
                let nearest = palette
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, color)| {
                        (0..3)
                            .map(|channel| {
                                let delta = i32::from(pixel[channel]) - i32::from(color[channel]);
                                delta * delta
                            })
                            .sum::<i32>()
                    })
                    .unwrap()
                    .0;
                counts[nearest] += 1;
            }
            let pixel_count = u64::from(pixels.width()) * u64::from(pixels.height());
            let non_accepted = 1.0 - counts[6] as f64 / pixel_count as f64;
            eprintln!(
                "temporal-corpus rejection_ratio={:.4}% reasons={counts:?}",
                non_accepted * 100.0
            );
            assert!(
                counts[1] > 0,
                "motion capture must expose off-screen history"
            );
            assert!(
                counts[2] > 0,
                "transparent coverage must appear as a reactive rejection"
            );
            assert!(
                counts[4] > 0,
                "motion capture must exercise history clamping"
            );
        } else if name == "taa-temporal-confidence" {
            let pixels = image::open(&path).unwrap().to_rgb8();
            let growing = pixels
                .pixels()
                .filter(|pixel| pixel[1] > pixel[0].saturating_add(2))
                .count();
            let reset = pixels
                .pixels()
                .filter(|pixel| pixel[0] > 2 && pixel[1].saturating_add(2) < pixel[0])
                .count();
            let retaining = pixels.pixels().filter(|pixel| pixel[2] > 2).count();
            eprintln!(
                "temporal-corpus confidence_growing={growing} confidence_reset={reset} \
                 retaining_history={retaining}"
            );
            assert!(
                growing > 0,
                "compatible pixels must grow persistent confidence"
            );
            assert!(
                reset > 0,
                "rejected pixels must reset persistent confidence"
            );
            assert!(
                retaining > 0,
                "accepted pixels must retain temporal history"
            );
        } else if name == "taa-reactive-history" {
            let pixels = image::open(&path).unwrap().to_rgb8();
            let current = pixels.pixels().filter(|pixel| pixel[0] > 8).count();
            let history = pixels.pixels().filter(|pixel| pixel[1] > 8).count();
            let union = pixels.pixels().filter(|pixel| pixel[2] > 8).count();
            eprintln!(
                "temporal-corpus reactive_current={current} reactive_history={history} \
                 reactive_union={union}"
            );
            assert!(
                current > 0 && history > 0 && union >= current.max(history),
                "reactive diagnostics must expose current, history, and union coverage"
            );
        } else if name == "taa-history-policy" {
            let pixels = image::open(&path).unwrap().to_rgb8();
            let clamped = pixels.pixels().filter(|pixel| pixel[0] > 2).count();
            let current_weighted = pixels.pixels().filter(|pixel| pixel[1] > 2).count();
            let rejected = pixels.pixels().filter(|pixel| pixel[2] > 2).count();
            let minimum_current_weight = pixels.pixels().map(|pixel| pixel[1]).min().unwrap();
            let maximum_current_weight = pixels.pixels().map(|pixel| pixel[1]).max().unwrap();
            eprintln!(
                "temporal-corpus history_clamped={clamped} \
                 history_current_weighted={current_weighted} history_rejected={rejected} \
                 current_weight_range={minimum_current_weight}..={maximum_current_weight}"
            );
            assert!(
                clamped > 0
                    && current_weighted > 0
                    && rejected > 0
                    && minimum_current_weight < maximum_current_weight,
                "history-policy diagnostics must expose clamp, blend, and rejection decisions"
            );
        }
    }
    assert!(eng.renderer.pending_quality_capture_dir.is_none());
    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"diagnostic_persistent_bytes\":0"));
    assert!(paths.contains("\"diagnostic_capture_passes\":1"));
    assert!(paths.contains("\"diagnostic_resources_live\":false"));
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept TAA diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn camera_motion_sequence_bounds_ghosting_flicker_and_cut_residue() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let draw_pose = |eng: &mut EngineState, angle: f32, fov: f32| {
        let radius = 7.2;
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 10.0, 18.0, 255.0);
        r.begin_mode_3d(
            angle.sin() * radius,
            2.6,
            angle.cos() * radius,
            0.0,
            0.7,
            0.0,
            0.0,
            1.0,
            0.0,
            fov,
            0.0,
        );
        r.add_directional_light(-0.4, -1.0, -0.2, 1.0, 0.95, 0.88, 2.0);
        r.add_point_light(0.0, 2.5, -1.5, 8.0, 1.0, 0.15, 0.05, 7.0);
        r.draw_plane(0.0, 0.0, 0.0, 14.0, 14.0, 35.0, 42.0, 55.0, 255.0);
        r.draw_cube(-1.5, 0.9, 0.2, 1.8, 1.8, 1.8, 240.0, 42.0, 35.0, 255.0);
        r.draw_cube(1.2, 1.5, -1.2, 1.2, 3.0, 1.2, 25.0, 210.0, 245.0, 255.0);
        r.draw_sphere(0.3, 0.8, 1.5, 0.8, 245.0, 220.0, 35.0, 255.0);
    };
    let advance = |eng: &mut EngineState, frames: u32, angle: f32, fov: f32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_pose(eng, angle, fov);
            eng.end_frame();
        }
    };
    let capture = |eng: &mut EngineState, angle: f32, fov: f32| {
        render(eng, 1, |eng| draw_pose(eng, angle, fov)).2
    };

    let old_angle = -0.55;
    let new_angle = 0.65;
    eng.renderer.reset_temporal_history();
    let fresh_new_pose = capture(&mut eng, new_angle, 58.0);
    advance(&mut eng, 8, old_angle, 42.0);
    eng.renderer.reset_temporal_history();
    let cut_new_pose = capture(&mut eng, new_angle, 58.0);
    let cut_metrics = calculate_diff_metrics(&fresh_new_pose, &cut_new_pose, W, H);
    assert_eq!(
        cut_metrics.max_diff, 0,
        "an explicit camera cut retained pixels from the prior camera"
    );

    advance(&mut eng, 8, old_angle, 42.0);
    let mut fast_rotation = Vec::new();
    for _ in 0..24 {
        fast_rotation.push(capture(&mut eng, new_angle, 42.0));
    }
    let stable_reference = average_rgba(&fast_rotation[8..]);
    let convergence = fast_rotation[..8]
        .iter()
        .map(|frame| calculate_diff_metrics(&stable_reference, frame, W, H))
        .collect::<Vec<_>>();
    let severe_trails = fast_rotation[..8]
        .iter()
        .map(|frame| severe_pixel_fraction(&stable_reference, frame))
        .collect::<Vec<_>>();
    for (index, metrics) in convergence.iter().enumerate() {
        eprintln!(
            "temporal-corpus fast-rotation frame={index} mean_rgb={:.4} \
             outliers={:.4}% severe_trail={:.4}% ssim={:.6}",
            metrics.mean_rgb,
            metrics.outlier_pixel_fraction * 100.0,
            severe_trails[index] * 100.0,
            metrics.ssim,
        );
    }
    assert!(
        convergence[4].mean_rgb <= convergence[0].mean_rgb * 0.6 + 0.25,
        "fast-rotation history did not converge within four recovery frames"
    );
    assert!(
        convergence[4].outlier_pixel_fraction <= 0.02,
        "fast-rotation ghost trail exceeded 2% of pixels after four frames"
    );
    let ghost_trail_frames = severe_trails
        .iter()
        .enumerate()
        .find(|(index, _)| {
            severe_trails[*index..]
                .iter()
                .all(|fraction| *fraction <= 0.005)
        })
        .map(|(index, _)| index)
        .unwrap_or(severe_trails.len());
    eprintln!("temporal-corpus ghost_trail_frames={ghost_trail_frames}");
    assert!(
        ghost_trail_frames <= 4,
        "severe fast-rotation trail persisted beyond four frames"
    );
    let stable_metrics = fast_rotation[8..]
        .iter()
        .map(|frame| calculate_diff_metrics(&stable_reference, frame, W, H))
        .collect::<Vec<_>>();
    let stable_mean = stable_metrics
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .sum::<f64>()
        / stable_metrics.len() as f64;
    eprintln!("temporal-corpus stable-reference mean_flicker={stable_mean:.4}");
    assert!(
        stable_mean <= 2.0,
        "settled TAA jitter cycle did not converge to a stable estimate"
    );

    eng.renderer.reset_temporal_history();
    advance(&mut eng, 8, 0.0, 50.0);
    let mut slow_pan = Vec::new();
    for frame in 0..8 {
        slow_pan.push(capture(&mut eng, frame as f32 * 0.0025, 50.0));
    }
    let slow_deltas = slow_pan
        .windows(2)
        .map(|pair| calculate_diff_metrics(&pair[0], &pair[1], W, H))
        .collect::<Vec<_>>();
    let mean_flicker = slow_deltas
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .sum::<f64>()
        / slow_deltas.len() as f64;
    let max_outliers = slow_deltas
        .iter()
        .map(|metrics| metrics.outlier_pixel_fraction)
        .fold(0.0f64, f64::max);
    eprintln!(
        "temporal-corpus slow-pan mean_flicker={mean_flicker:.4} \
         max_outliers={:.4}%",
        max_outliers * 100.0,
    );
    assert!(
        mean_flicker <= 4.0,
        "slow-pan temporal flicker is unbounded"
    );
    assert!(
        max_outliers <= 0.03,
        "slow-pan coherent flicker exceeded 3% of pixels"
    );
}

#[test]
fn retained_rigid_and_reactive_motion_sequences_bound_trails() {
    fn transform(x: f32, angle: f32) -> [[f32; 4]; 4] {
        let (sin, cos) = angle.sin_cos();
        [
            [cos, 0.0, -sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [sin, 0.0, cos, 0.0],
            [x, 1.0, 0.0, 1.0],
        ]
    }
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);
    eng.renderer.set_transparency_composition_mode(0);

    let (vertices, indices) = cube_verts(0.9, [0.95, 0.08, 0.04, 1.0]);
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_material_pbr(node, 0.2, 0.35);
    eng.scene.set_material_color(node, 0.95, 0.08, 0.04, 1.0);

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.88, 2.2);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.1, -1.8, 5.0, 3.2, 0.35, 30.0, 170.0, 235.0, 255.0);
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_scene(eng);
            eng.end_frame();
        }
    };
    let capture = |eng: &mut EngineState| render(eng, 1, draw_scene).2;
    let run_motion = |eng: &mut EngineState, node: f64| {
        eng.scene.set_transform(node, transform(-1.6, -0.7));
        eng.renderer.reset_temporal_history();
        advance(eng, 8);
        let old_pose = capture(eng);
        eng.scene.set_transform(node, transform(1.6, 0.9));
        let mut frames = Vec::new();
        for _ in 0..24 {
            frames.push(capture(eng));
        }
        (old_pose, frames)
    };

    let (opaque_old, opaque) = run_motion(&mut eng, node);
    evaluate_motion_recovery("rigid-opaque", &opaque_old, &opaque);

    eng.scene
        .set_material_gltf_alpha(node, MaterialAlphaMode::Blend, 0.0, false);
    eng.scene.set_material_color(node, 0.1, 0.8, 1.0, 0.95);
    let (reactive_old, reactive) = run_motion(&mut eng, node);
    evaluate_motion_recovery("rigid-reactive", &reactive_old, &reactive);
    assert!(
        eng.renderer
            .quality_runtime_paths_json()
            .contains("\"temporal_reactive\":{\"enabled\":true,\"active\":true"),
        "transparent retained motion did not select reactive TAA coverage"
    );
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("reactive motion telemetry is valid JSON");
    assert_eq!(
        paths["steady_state_resources"]["bind_group_creations"]["sites"]["taa_reactive"].as_u64(),
        Some(0),
        "warmed reactive TAA must reuse its plan/generation/history-specific bind group"
    );
    assert_eq!(
        paths["steady_state_resources"]["graph_compiles"].as_u64(),
        Some(0),
        "reactive topology must not recompile after warm-up"
    );
    assert_eq!(
        paths["steady_state_resources"]["transient_physical_creations"]["textures"].as_u64(),
        Some(0),
        "reactive graph textures must be reused after warm-up"
    );

    eng.renderer.resize(320, 192, 320, 192);
    advance(&mut eng, 3);
    eng.renderer.resize(W, H, W, H);
    advance(&mut eng, 4);
    let resized_paths: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("resized reactive motion telemetry is valid JSON");
    assert_eq!(
        resized_paths["steady_state_resources"]["bind_group_creations"]["sites"]["taa_reactive"]
            .as_u64(),
        Some(0),
        "reactive TAA must rebuild for resize generation then return to zero churn"
    );
    assert_eq!(
        resized_paths["steady_state_resources"]["graph_compiles"].as_u64(),
        Some(0),
        "settled resize generation must return to cached topology"
    );
    assert_eq!(
        resized_paths["steady_state_resources"]["transient_physical_creations"]["textures"]
            .as_u64(),
        Some(0),
        "settled resize generation must reuse graph textures"
    );
}

#[test]
fn immediate_primitive_motion_writes_velocity_and_bounds_trails() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let draw_pose = |eng: &mut EngineState, moving_x: f64| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.88, 2.2);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.1, -1.8, 5.0, 3.2, 0.35, 30.0, 170.0, 235.0, 255.0);
        r.draw_cube(
            moving_x, 0.9, 0.35, 1.35, 1.8, 1.35, 245.0, 45.0, 25.0, 255.0,
        );
    };
    let capture_pose = |eng: &mut EngineState, x| render(eng, 1, |eng| draw_pose(eng, x)).2;

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture_pose(&mut eng, -1.6);
    }
    let old_pose = capture_pose(&mut eng, -1.6);
    let directory =
        std::env::temp_dir().join(format!("bloom-immediate-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture_pose(&mut eng, 1.6));
    }

    let motion = image::open(directory.join("taa-motion.png"))
        .expect("immediate primitive motion did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus immediate moving_pixels={moving_pixels}");
    assert!(
        moving_pixels >= 300,
        "moving immediate cube wrote no meaningful velocity"
    );
    evaluate_motion_recovery("immediate-cube", &old_pose, &frames);

    let paths: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json()).unwrap();
    assert_eq!(
        paths["temporal_history"]["immediate_motion_entries"].as_u64(),
        Some(3)
    );
    assert_eq!(
        paths["temporal_history"]["immediate_motion_gpu_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(
        paths["temporal_history"]["immediate_motion_passes"].as_u64(),
        Some(0)
    );
    let cpu_capacity = paths["temporal_history"]["immediate_motion_cpu_capacity_bytes"]
        .as_u64()
        .unwrap();
    eprintln!("temporal-corpus immediate history_cpu_capacity={cpu_capacity} bytes");
    assert!(
        cpu_capacity <= 4096,
        "three immediate primitives retained excessive history: {cpu_capacity} bytes"
    );
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept immediate motion diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn instanced_particle_reactive_opt_in_bounds_trails_without_taxing_opt_out() {
    const HANDLE: u64 = 0x7AA5_C011;
    const PARTICLE_SHADER_PREFIX: &str = r#"
#include "material_abi.wgsl"

struct ParticleInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
  @location(3) uv: vec2<f32>,
  @location(4) joints: vec4<f32>,
  @location(5) weights: vec4<f32>,
  @location(6) tangent: vec4<f32>,
  @location(7) instance_pos: vec3<f32>,
  @location(8) instance_rot_y: f32,
  @location(9) instance_scale: f32,
  @location(10) instance_tint: vec4<f32>,
};

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) tint: vec4<f32>,
};

@vertex
fn vs_main(in: ParticleInput) -> VsOut {
  var out: VsOut;
  let world = in.position * in.instance_scale + in.instance_pos;
  out.clip_position = view.view_proj * vec4<f32>(world, 1.0);
  out.tint = in.color * in.instance_tint;
  return out;
}

fn particle_color(in: VsOut) -> vec4<f32> {
  return vec4<f32>(in.tint.rgb * 3.0, in.tint.a);
}

@fragment
fn fs_main(in: VsOut) -> TranslucentOut {
  var out: TranslucentOut;
  out.hdr = particle_color(in);
  return out;
}
"#;
    const PARTICLE_REACTIVE_SUFFIX: &str = r#"

@fragment
fn fs_reactive(in: VsOut) -> ReactiveTranslucentOut {
  var out: ReactiveTranslucentOut;
  out.hdr = particle_color(in);
  out.reactive = in.tint.a;
  return out;
}
"#;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);
    eng.renderer.set_transparency_composition_mode(0);

    let vertex = |position| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [1.0, 0.13, 0.025, 1.0],
        uv: [0.0; 2],
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices: vec![
                vertex([-0.55, -0.55, 0.0]),
                vertex([0.55, -0.55, 0.0]),
                vertex([0.55, 0.55, 0.0]),
                vertex([-0.55, 0.55, 0.0]),
            ],
            secondary_tex_coords: None,
            indices: vec![0, 1, 2, 0, 2, 3],
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Blend,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: true,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));

    let ordinary_material = eng
        .renderer
        .compile_material_instanced_bucket(PARTICLE_SHADER_PREFIX, 2, false)
        .expect("ordinary instanced particle material compiles");
    let reactive_source = format!("{PARTICLE_SHADER_PREFIX}{PARTICLE_REACTIVE_SUFFIX}");
    let reactive_material = eng
        .renderer
        .compile_material_instanced_bucket(&reactive_source, 2, false)
        .expect("reactive instanced particle material compiles");
    let old_buffer = eng
        .renderer
        .create_instance_buffer(&[-1.35, 1.15, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1);
    let new_buffer = eng
        .renderer
        .create_instance_buffer(&[1.35, 1.15, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1);

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.0, 6.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.2, -1.7, 5.0, 3.2, 0.3, 30.0, 150.0, 220.0, 255.0);
    };
    let capture_particle = |eng: &mut EngineState, material, instance_buffer| -> Vec<u8> {
        render(eng, 1, |eng| {
            draw_scene(eng);
            eng.renderer
                .submit_material_draw_instanced(material, HANDLE, 0, instance_buffer, 1);
        })
        .2
    };
    let mut run_sequence = |material, label: &str, capture_reactive: bool| {
        eng.renderer.reset_temporal_history();
        for _ in 0..8 {
            capture_particle(&mut eng, material, old_buffer);
        }
        let old_pose = capture_particle(&mut eng, material, old_buffer);
        let directory = std::env::temp_dir().join(format!("bloom-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        if capture_reactive {
            eng.renderer.pending_quality_capture_dir =
                Some(directory.to_string_lossy().into_owned());
        }
        let mut frames = Vec::new();
        for _ in 0..24 {
            frames.push(capture_particle(&mut eng, material, new_buffer));
        }
        evaluate_motion_recovery(label, &old_pose, &frames);
        let paths: serde_json::Value =
            serde_json::from_str(&eng.renderer.quality_runtime_paths_json()).unwrap();
        assert_eq!(
            paths["temporal_reactive"]["active"].as_bool(),
            Some(capture_reactive),
            "{label} selected the wrong temporal-reactive topology"
        );
        if capture_reactive {
            let reasons = image::open(directory.join("taa-rejection-reason.png"))
                .expect("reactive particle capture did not emit rejection reasons")
                .to_rgb8();
            let reactive_pixels = reasons
                .pixels()
                .filter(|pixel| {
                    i32::from(pixel[0]).pow(2)
                        + (i32::from(pixel[1]) - 230).pow(2)
                        + (i32::from(pixel[2]) - 255).pow(2)
                        < 80_i32.pow(2)
                })
                .count();
            eprintln!("temporal-corpus {label} reactive_pixels={reactive_pixels}");
            assert!(
                reactive_pixels >= 100,
                "authored particle coverage did not reach TAA rejection"
            );
        }
        if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
            eprintln!("kept {label} diagnostics at {directory:?}");
        } else {
            let _ = std::fs::remove_dir_all(directory);
        }
    };

    run_sequence(reactive_material, "reactive-particle", true);
    run_sequence(ordinary_material, "ordinary-particle-control", false);
    eng.renderer.destroy_instance_buffer(old_buffer);
    eng.renderer.destroy_instance_buffer(new_buffer);
}

#[test]
fn legacy_skinned_motion_uses_staged_previous_palette_and_bounds_trails() {
    const PALETTE_KEY: u64 = 0x7AA5_2001;
    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn palette(bend: f32) -> [[[f32; 4]; 4]; 2] {
        let (sin, cos) = bend.sin_cos();
        [
            IDENTITY,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, sin, 0.0],
                [0.0, -sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    }

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let (mut vertices, indices) = cube_verts(0.9, [0.95, 0.12, 0.035, 1.0]);
    for vertex in &mut vertices {
        let upper = vertex.position[1] > 0.0;
        vertex.joints = if upper {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0; 4]
        };
        vertex.weights = [1.0, 0.0, 0.0, 0.0];
    }

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.88, 2.2);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.1, -1.8, 5.0, 3.2, 0.35, 30.0, 170.0, 235.0, 255.0);
    };
    let capture_pose = |eng: &mut EngineState, x: f32, bend: f32, facing: f32| {
        render(eng, 1, |eng| {
            draw_scene(eng);
            let (rot_sin, rot_cos) = facing.sin_cos();
            eng.renderer.set_joint_matrices_scaled(
                PALETTE_KEY,
                &palette(bend),
                1.0,
                [x, 1.0, 0.0],
                rot_sin,
                rot_cos,
            );
            eng.renderer
                .draw_model_mesh(&vertices, &indices, [0.0; 3], 1.0);
        })
        .2
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture_pose(&mut eng, -1.5, -0.55, -0.45);
    }
    let old_pose = capture_pose(&mut eng, -1.5, -0.55, -0.45);
    let directory =
        std::env::temp_dir().join(format!("bloom-legacy-skin-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture_pose(&mut eng, 1.5, 0.75, 0.6));
    }

    let motion = image::open(directory.join("taa-motion.png"))
        .expect("legacy skinned capture did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus legacy-skinned moving_pixels={moving_pixels}");
    assert!(
        moving_pixels >= 250,
        "legacy skinned pose wrote no meaningful velocity"
    );
    evaluate_motion_recovery("legacy-skinned", &old_pose, &frames);

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept legacy-skinned diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn cached_skinned_motion_sequence_bounds_animation_trails() {
    const HANDLE: u64 = 0x7AA5_0001;
    const PALETTE_KEY: u64 = 0x7AA5_1001;
    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn palette(bend: f32) -> [[[f32; 4]; 4]; 2] {
        let (sin, cos) = bend.sin_cos();
        [
            IDENTITY,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, sin, 0.0],
                [0.0, -sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    }

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let (mut vertices, indices) = cube_verts(0.9, [0.95, 0.12, 0.035, 1.0]);
    for vertex in &mut vertices {
        let upper = vertex.position[1] > 0.0;
        vertex.joints = if upper {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0; 4]
        };
        vertex.weights = [1.0, 0.0, 0.0, 0.0];
    }
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices,
            secondary_tex_coords: None,
            indices,
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.15,
            roughness_factor: 0.32,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));
    assert!(
        eng.renderer.is_model_skinned(HANDLE),
        "weighted temporal test mesh did not select the cached skinned path"
    );

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.88, 2.2);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.1, -1.8, 5.0, 3.2, 0.35, 30.0, 170.0, 235.0, 255.0);
    };
    let draw_pose = |eng: &mut EngineState, x: f32, bend: f32, facing: f32| {
        draw_scene(eng);
        let (rot_sin, rot_cos) = facing.sin_cos();
        eng.renderer.set_joint_matrices_scaled(
            PALETTE_KEY,
            &palette(bend),
            1.0,
            [x, 1.0, 0.0],
            rot_sin,
            rot_cos,
        );
        eng.renderer
            .draw_model_cached_skinned(HANDLE, [0.0; 3], 1.0, [1.0; 4]);
    };
    let capture_pose = |eng: &mut EngineState, x: f32, bend: f32, facing: f32| {
        render(eng, 1, |eng| draw_pose(eng, x, bend, facing)).2
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture_pose(&mut eng, -1.5, -0.55, -0.45);
    }
    let old_pose = capture_pose(&mut eng, -1.5, -0.55, -0.45);
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture_pose(&mut eng, 1.5, 0.75, 0.6));
    }
    evaluate_motion_recovery("cached-skinned", &old_pose, &frames);
}

#[test]
fn cached_alpha_tested_card_motion_writes_velocity_and_bounds_trails() {
    const HANDLE: u64 = 0x7AA5_F011;
    const TEX_SIZE: u32 = 64;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let mut pixels = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let dx = (x as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let dy = (y as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let ellipse = dx * dx / 0.82f32.powi(2) + dy * dy / 0.96f32.powi(2) < 1.0;
            let serrated_edge = ((y / 4 + x / 7) & 1) == 0 || dx.abs() < 0.68;
            let vein_gap = (x as i32 - TEX_SIZE as i32 / 2).abs() == 5 && y % 9 < 6;
            let opaque = ellipse && serrated_edge && !vein_gap;
            pixels.extend_from_slice(if opaque {
                &[30, 205, 55, 255]
            } else {
                &[0, 0, 0, 0]
            });
        }
    }
    let texture = eng.renderer.register_texture_kind_with_alpha_coverage(
        TEX_SIZE,
        TEX_SIZE,
        &pixels,
        false,
        Some(0.5),
    );
    let vertex = |position, uv| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [1.0; 4],
        uv,
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    let vertices = vec![
        vertex([-1.0, 0.0, 0.0], [0.0, 1.0]),
        vertex([1.0, 0.0, 0.0], [1.0, 1.0]),
        vertex([1.0, 2.8, 0.0], [1.0, 0.0]),
        vertex([-1.0, 2.8, 0.0], [0.0, 0.0]),
    ];
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices,
            secondary_tex_coords: None,
            indices: vec![0, 1, 2, 0, 2, 3],
            texture_idx: Some(texture),
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 0.72,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Mask,
            alpha_cutoff: 0.5,
            alpha_coverage_mips: true,
            double_sided: true,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 25.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 1.3, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.35, -1.0, -0.2, 0.9, 1.0, 0.8, 2.0);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 38.0, 45.0, 55.0, 255.0);
        r.draw_cube(0.0, 1.5, -1.1, 6.0, 3.4, 0.25, 35.0, 80.0, 125.0, 255.0);
    };
    let capture_pose = |eng: &mut EngineState, x: f32, angle: f32| {
        render(eng, 1, |eng| {
            draw_scene(eng);
            eng.renderer
                .draw_model_cached_rotated(HANDLE, [x, 0.0, 0.0], 1.0, angle, [1.0; 4]);
        })
        .2
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture_pose(&mut eng, -1.35, -0.35);
    }
    let old_pose = capture_pose(&mut eng, -1.35, -0.35);
    let directory =
        std::env::temp_dir().join(format!("bloom-foliage-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture_pose(&mut eng, 1.35, 0.5));
    }

    let motion = image::open(directory.join("taa-motion.png"))
        .expect("alpha-tested motion capture did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus alpha-tested moving_pixels={moving_pixels}");
    assert!(
        moving_pixels >= 250,
        "cached alpha-tested object motion wrote no meaningful velocity"
    );
    evaluate_motion_recovery("alpha-tested-card", &old_pose, &frames);

    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"cached_model_motion_entries\":1"));
    assert!(paths.contains("\"cached_model_motion_gpu_bytes\":0"));
    assert!(paths.contains("\"cached_model_motion_passes\":0"));
    let paths: serde_json::Value = serde_json::from_str(&paths).unwrap();
    let cpu_capacity = paths["temporal_history"]["cached_model_motion_cpu_capacity_bytes"]
        .as_u64()
        .unwrap();
    eprintln!("temporal-corpus cached-model history_cpu_capacity={cpu_capacity} bytes");
    assert!(
        cpu_capacity <= 1024,
        "one cached instance retained excessive transform history: {cpu_capacity} bytes"
    );
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept alpha-tested motion diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn emissive_light_switches_converge_without_radiance_trails() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let (vertices, indices) = cube_verts(0.65, [0.18, 0.035, 0.01, 1.0]);
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_transform(
        node,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
    );
    eng.scene.set_material_pbr(node, 0.24, 0.0);

    let draw_state = |eng: &mut EngineState, enabled: bool| {
        eng.scene.set_material_emissive_factor(
            node,
            if enabled { 7.0 } else { 0.0 },
            if enabled { 0.8 } else { 0.0 },
            if enabled { 0.08 } else { 0.0 },
        );
        let r = &mut eng.renderer;
        r.set_clear_color(5.0, 7.0, 14.0, 255.0);
        r.begin_mode_3d(4.2, 2.8, 6.2, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 0.55, 0.65, 0.9, 0.55);
        if enabled {
            r.add_point_light(0.0, 1.25, 0.2, 4.5, 1.0, 0.11, 0.02, 8.0);
        }
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(-1.4, 0.65, 0.0, 0.8, 1.3, 0.8, 105.0, 115.0, 135.0, 255.0);
        r.draw_cube(1.4, 0.65, 0.0, 0.8, 1.3, 0.8, 105.0, 115.0, 135.0, 255.0);
    };
    let capture_state =
        |eng: &mut EngineState, enabled| render(eng, 1, |eng| draw_state(eng, enabled)).2;
    let run_switch = |eng: &mut EngineState, from: bool, to: bool| {
        eng.renderer.reset_temporal_history();
        for _ in 0..8 {
            capture_state(eng, from);
        }
        let old_state = capture_state(eng, from);
        let mut frames = Vec::new();
        for _ in 0..24 {
            frames.push(capture_state(eng, to));
        }
        (old_state, frames)
    };

    let (off_state, on_frames) = run_switch(&mut eng, false, true);
    evaluate_motion_recovery("emissive-on", &off_state, &on_frames);
    let (on_state, off_frames) = run_switch(&mut eng, true, false);
    evaluate_motion_recovery("emissive-off", &on_state, &off_frames);
}

#[test]
fn render_scale_and_resize_steps_seed_without_prior_frame_residue() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 24.0, 255.0);
        r.begin_mode_3d(4.0, 2.8, 6.0, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 49.0, 0.0);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.9, 0.75, 2.0);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 35.0, 42.0, 58.0, 255.0);
        for x in -2..=2 {
            r.draw_cube(
                x as f64 * 0.75,
                0.45,
                (x & 1) as f64 * 0.45,
                0.42,
                0.9,
                0.42,
                210.0,
                55.0 + (x + 2) as f64 * 35.0,
                35.0,
                255.0,
            );
        }
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_scene(eng);
            eng.end_frame();
        }
    };
    let capture = |eng: &mut EngineState| render(eng, 1, draw_scene);

    eng.renderer.set_render_scale(0.5);
    eng.renderer.reset_temporal_history();
    let (_, _, fresh_half_scale) = capture(&mut eng);
    eng.renderer.set_render_scale(1.0);
    advance(&mut eng, 8);
    eng.renderer.set_render_scale(0.5);
    let (_, _, stepped_half_scale) = capture(&mut eng);
    let scale_metrics = calculate_diff_metrics(&fresh_half_scale, &stepped_half_scale, W, H);
    assert_eq!(
        scale_metrics.max_diff, 0,
        "render-scale step blended pixels from the incompatible full-scale history"
    );

    eng.renderer.set_render_scale(1.0);
    eng.renderer.reset_temporal_history();
    let (_, _, fresh_native_size) = capture(&mut eng);
    eng.renderer.resize(320, 192, 320, 192);
    advance(&mut eng, 3);
    eng.renderer.resize(W, H, W, H);
    let (width, height, returned_native_size) = capture(&mut eng);
    assert_eq!((width, height), (W, H));
    let resize_metrics = calculate_diff_metrics(&fresh_native_size, &returned_native_size, W, H);
    eprintln!("temporal-corpus resize metrics={resize_metrics:?}");
    assert!(
        resize_metrics.mean_rgb <= 0.5
            && resize_metrics.outlier_pixel_fraction == 0.0
            && resize_metrics.max_diff <= 32,
        "window resize restored a coherent image from destroyed prior-size history: \
         {resize_metrics:?}"
    );
    eprintln!(
        "temporal-corpus scale_step_max={} resize_step_max={}",
        scale_metrics.max_diff, resize_metrics.max_diff,
    );
}
