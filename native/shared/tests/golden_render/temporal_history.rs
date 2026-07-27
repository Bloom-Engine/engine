use super::*;

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
                counts[4] > 0,
                "motion capture must exercise history clamping"
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
    fn severe_pixel_fraction(reference: &[u8], candidate: &[u8]) -> f64 {
        let severe = reference
            .chunks_exact(4)
            .zip(candidate.chunks_exact(4))
            .filter(|(a, b)| (0..3).any(|channel| a[channel].abs_diff(b[channel]) > 64))
            .count();
        severe as f64 / (reference.len() / 4) as f64
    }

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    r.set_taa_enabled(true);
    r.set_render_scale(1.0);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(false);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_motion_blur_enabled(false);
    r.set_shadows_enabled(false);

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
    let mut stable_sum = vec![0u32; fast_rotation[0].len()];
    for frame in &fast_rotation[8..] {
        for (sum, value) in stable_sum.iter_mut().zip(frame) {
            *sum += u32::from(*value);
        }
    }
    let stable_reference = stable_sum
        .into_iter()
        .map(|sum| ((sum + 8) / 16) as u8)
        .collect::<Vec<_>>();
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
