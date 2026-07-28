use super::super::*;

#[test]
fn ssgi_hiz_immediate_scene_produces_finite_indirect_radiance() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    if std::env::var_os("BLOOM_SSGI_PROFILE_HD").is_some() {
        r.resize(1280, 720, 1280, 720);
    }
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(true);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_shadows_enabled(false);

    let draw = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, 1.8);
        r.draw_cube(0.0, -0.1, 0.0, 12.0, 0.2, 12.0, 90.0, 96.0, 107.0, 255.0);
        r.draw_cube(0.0, 2.0, -3.0, 8.0, 4.0, 0.2, 230.0, 166.0, 31.0, 255.0);
        r.draw_cube(-1.1, 1.0, 0.0, 1.8, 2.0, 1.8, 230.0, 45.0, 25.0, 255.0);
        r.draw_sphere(1.1, 0.9, -0.8, 0.9, 30.0, 110.0, 240.0, 255.0);
    };
    let capture = |eng: &mut EngineState| {
        eng.begin_frame();
        draw(eng);
        eng.end_frame();
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..24 {
        capture(&mut eng);
    }
    eng.profiler.set_enabled(true);
    for _ in 0..120 {
        capture(&mut eng);
    }
    let probe_gpu_us = eng
        .profiler
        .snapshot()
        .into_iter()
        .filter_map(|(label, _, gpu)| label.starts_with("probe_").then_some((label, gpu?)))
        .collect::<Vec<_>>();
    let probe_total_gpu_us = probe_gpu_us.iter().map(|(_, gpu)| gpu).sum::<f64>();
    eng.profiler.set_enabled(false);
    let directory = std::env::temp_dir().join(format!("bloom-ssgi-hiz-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng);

    let confidence = image::open(directory.join("ssgi-temporal-confidence.png"))
        .expect("Hi-Z SSGI capture did not emit temporal confidence")
        .to_rgb8();
    let current = confidence.pixels().filter(|pixel| pixel[1] > 0).count();
    let retained = confidence.pixels().filter(|pixel| pixel[2] > 16).count();
    let metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("ssgi.metrics.json"))
            .expect("Hi-Z SSGI capture did not emit resolved HDR metrics"),
    )
    .unwrap();
    let non_finite = metrics["non_finite_pixels"].as_u64().unwrap();
    let max_luminance = metrics["max_luminance"].as_f64().unwrap();
    let paths = eng.renderer.quality_runtime_paths_json();
    eprintln!(
        "hiz-corpus ssgi current={current} retained={retained} non_finite={non_finite} \
         max_luma={max_luminance:.6} probe_total_gpu_us={probe_total_gpu_us:.3} \
         probe_gpu_us={probe_gpu_us:?} paths={paths}"
    );
    assert!(
        paths.contains("\"ssgi_trace_backend\":\"hiz-screen\""),
        "immediate-only SSGI scene escaped the Hi-Z backend: {paths}"
    );
    assert_eq!(non_finite, 0, "Hi-Z SSGI emitted non-finite radiance");
    assert!(
        current >= 100 && retained >= 100 && max_luminance > 0.0001,
        "Hi-Z SSGI produced no usable indirect radiance: current={current}, \
         retained={retained}, max_luma={max_luminance:.6}"
    );

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept Hi-Z SSGI diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn ssgi_capture_exposes_probe_history_without_normal_frame_resources() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(true);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_shadows_enabled(false);
    super::transformed_box(
        &mut eng,
        [0.0, -0.1, 0.0],
        [12.0, 0.2, 12.0],
        [0.35, 0.38, 0.42, 1.0],
        0.8,
        0.0,
        [0.0; 3],
    );
    super::transformed_box(
        &mut eng,
        [0.0, 2.0, -3.0],
        [8.0, 4.0, 0.2],
        [0.9, 0.65, 0.12, 1.0],
        0.7,
        0.0,
        [2.5, 1.2, 0.15],
    );

    let draw = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, 1.8);
        r.draw_cube(-1.1, 1.0, 0.0, 1.8, 2.0, 1.8, 230.0, 45.0, 25.0, 255.0);
        r.draw_sphere(1.1, 0.9, -0.8, 0.9, 30.0, 110.0, 240.0, 255.0);
    };
    let capture = |eng: &mut EngineState| {
        eng.begin_frame();
        draw(eng);
        eng.end_frame();
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..24 {
        capture(&mut eng);
    }
    let directory =
        std::env::temp_dir().join(format!("bloom-ssgi-diagnostics-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng);

    let reasons = image::open(directory.join("ssgi-rejection-reason.png"))
        .expect("SSGI capture did not emit probe-domain temporal reasons")
        .to_rgb8();
    let accepted = reasons
        .pixels()
        .filter(|pixel| pixel[0] < 40 && pixel[1] > 140 && pixel[2] < 60)
        .count();
    let refreshed = reasons
        .pixels()
        .filter(|pixel| pixel[0] > 220 && pixel[1] < 40 && pixel[2] > 160)
        .count();
    let confidence = image::open(directory.join("ssgi-temporal-confidence.png"))
        .expect("SSGI capture did not emit probe-domain temporal confidence")
        .to_rgb8();
    let retained = confidence.pixels().filter(|pixel| pixel[2] > 16).count();
    let current = confidence.pixels().filter(|pixel| pixel[1] > 0).count();
    let metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("ssgi.metrics.json"))
            .expect("SSGI capture did not emit resolved HDR metrics"),
    )
    .unwrap();
    let non_finite = metrics["non_finite_pixels"].as_u64().unwrap();
    let max_luminance = metrics["max_luminance"].as_f64().unwrap();
    eprintln!(
        "temporal-corpus ssgi-probes accepted={accepted} refreshed={refreshed} current={current} \
         retained={retained} non_finite={non_finite} max_luma={max_luminance:.4} total={}",
        reasons.width() * reasons.height()
    );
    assert!(
        accepted >= 100 && current >= 100 && retained >= 100,
        "settled SSGI probes exposed no current radiance or retained history"
    );
    assert_eq!(non_finite, 0, "SSGI resolve emitted non-finite radiance");
    assert!(
        max_luminance > 0.0001,
        "SSGI probe resolve produced no indirect radiance"
    );
    assert_eq!(
        reasons.dimensions(),
        confidence.dimensions(),
        "SSGI reason/confidence atlases describe different probe domains"
    );

    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"ray_scene_preparation\":\"ssgi\""));
    assert!(paths.contains("\"ssgi_diagnostic_persistent_bytes\":0"));
    assert!(paths.contains("\"ssgi_diagnostic_capture_passes\":1"));
    assert!(paths.contains("\"ssgi_diagnostic_resources_live\":false"));
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept SSGI diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}
