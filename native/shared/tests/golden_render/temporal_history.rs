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
