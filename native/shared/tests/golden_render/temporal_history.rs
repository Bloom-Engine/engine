use super::*;

fn severe_pixel_fraction(reference: &[u8], candidate: &[u8]) -> f64 {
    let severe = reference
        .chunks_exact(4)
        .zip(candidate.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|channel| a[channel].abs_diff(b[channel]) > 64))
        .count();
    severe as f64 / (reference.len() / 4) as f64
}

fn average_rgba(frames: &[Vec<u8>]) -> Vec<u8> {
    assert!(!frames.is_empty());
    let mut sum = vec![0u32; frames[0].len()];
    for frame in frames {
        for (sum, value) in sum.iter_mut().zip(frame) {
            *sum += u32::from(*value);
        }
    }
    let count = frames.len() as u32;
    sum.into_iter()
        .map(|sum| ((sum + count / 2) / count) as u8)
        .collect()
}

fn evaluate_motion_recovery(label: &str, old_pose: &[u8], frames: &[Vec<u8>]) {
    let stable = average_rgba(&frames[8..]);
    let movement = calculate_diff_metrics(old_pose, &stable, W, H);
    let recovery = frames[..8]
        .iter()
        .map(|frame| calculate_diff_metrics(&stable, frame, W, H))
        .collect::<Vec<_>>();
    let severe = frames[..8]
        .iter()
        .map(|frame| severe_pixel_fraction(&stable, frame))
        .collect::<Vec<_>>();
    let trail_frames = severe
        .iter()
        .enumerate()
        .find(|(index, _)| severe[*index..].iter().all(|fraction| *fraction <= 0.005))
        .map(|(index, _)| index)
        .unwrap_or(severe.len());
    let stable_mean = frames[8..]
        .iter()
        .map(|frame| calculate_diff_metrics(&stable, frame, W, H).mean_rgb)
        .sum::<f64>()
        / (frames.len() - 8) as f64;
    eprintln!(
        "temporal-corpus {label} movement_mean={:.4} initial_mean={:.4} frame4_mean={:.4} \
         frame4_outliers={:.4}% trail_frames={trail_frames} \
         stable_flicker={stable_mean:.4}",
        movement.mean_rgb,
        recovery[0].mean_rgb,
        recovery[4].mean_rgb,
        recovery[4].outlier_pixel_fraction * 100.0,
    );
    assert!(
        trail_frames <= 4,
        "{label} left severe motion trails beyond four frames"
    );
    assert!(
        movement.mean_rgb >= 1.0 && movement.outlier_pixel_fraction >= 0.01,
        "{label} negative control did not produce visible object motion"
    );
    assert!(
        recovery[4].outlier_pixel_fraction <= 0.02,
        "{label} coherent trail covered over 2% after four frames"
    );
    assert!(
        stable_mean <= 2.0,
        "{label} did not settle to a stable jitter-cycle estimate"
    );
}

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
    r.set_transparency_composition_mode(0);

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
    let r = &mut eng.renderer;
    r.set_taa_enabled(true);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(false);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_motion_blur_enabled(false);
    r.set_shadows_enabled(false);

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
