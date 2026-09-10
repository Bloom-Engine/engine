//! A complete stationary angular cycle must still respond to changed lighting.

use super::*;

#[test]
fn software_ssgi_settles_each_phase_and_recovers_after_light_changes() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let renderer = &mut eng.renderer;
    renderer.set_taa_enabled(false);
    renderer.set_ssao_enabled(false);
    renderer.set_ssr_enabled(false);
    renderer.set_ssgi_enabled(true);
    renderer.set_bloom_enabled(false);
    renderer.set_auto_exposure(false);
    renderer.set_shadows_enabled(false);

    let root = std::env::temp_dir().join(format!(
        "bloom-ssgi-light-cycle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let frame = |eng: &mut EngineState, light: f32, output: Option<&Path>| {
        eng.begin_frame();
        let r = &mut eng.renderer;
        r.pending_quality_capture_dir = output.map(|path| path.to_string_lossy().into_owned());
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, light);
        r.draw_cube(0.0, -0.1, 0.0, 12.0, 0.2, 12.0, 90.0, 96.0, 107.0, 255.0);
        r.draw_cube(0.0, 2.0, -3.0, 8.0, 4.0, 0.2, 230.0, 166.0, 31.0, 255.0);
        r.draw_cube(-1.1, 1.0, 0.0, 1.8, 2.0, 1.8, 230.0, 45.0, 25.0, 255.0);
        r.draw_sphere(1.1, 0.9, -0.8, 0.9, 30.0, 110.0, 240.0, 255.0);
        eng.end_frame();
    };
    let capture = |eng: &mut EngineState, light: f32, label: &str| {
        let directory = root.join(label);
        frame(eng, light, Some(&directory));
        let metrics: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("ssgi.metrics.json")).expect("SSGI HDR metrics"),
        )
        .unwrap();
        assert_eq!(metrics["non_finite_pixels"], 0);
        image::open(directory.join("ssgi.png"))
            .expect("resolved SSGI capture")
            .to_rgba8()
    };

    // Exercise both directions without resetting history between them. At 32
    // frames every angular phase has observed the changed light twice; the
    // following complete cycle must agree with an independently seeded epoch.
    let mut settled_states = Vec::new();
    for (label, light) in [("bright", 1.8), ("dim", 0.0), ("bright-again", 1.8)] {
        for _ in 0..32 {
            frame(&mut eng, light, None);
        }
        let settled = capture(&mut eng, light, &format!("{label}-settled"));
        for phase in 0..16 {
            let current = capture(&mut eng, light, &format!("{label}-phase-{phase:02}"));
            assert!(
                settled.as_raw() == current.as_raw(),
                "{label} resolved SSGI changes at stationary angular phase {phase}"
            );
        }
        settled_states.push(settled);
    }
    let change = calculate_diff_metrics(
        settled_states[0].as_raw(),
        settled_states[1].as_raw(),
        settled_states[0].width(),
        settled_states[0].height(),
    );
    assert!(
        change.mean_rgb > 1.0,
        "lighting change did not affect indirect radiance: {change:?}"
    );
    assert!(
        settled_states[0] == settled_states[2],
        "returning light retained stale indirect radiance"
    );

    eng.renderer.reset_temporal_history();
    for _ in 0..64 {
        frame(&mut eng, 1.8, None);
    }
    let fresh = capture(&mut eng, 1.8, "fresh-reference");
    assert!(
        settled_states[2] == fresh,
        "light recovery differs from a fresh converged history"
    );
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"ssgi_trace_backend\":\"hiz-screen\""));
    eprintln!("ssgi-light-cycle stationary_frames=48 light_change={change:?}");
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept software SSGI lighting diagnostics at {root:?}");
    } else {
        std::fs::remove_dir_all(root).expect("remove SSGI lighting captures");
    }
}
