use super::super::*;

fn glass_transform(x: f32, angle: f32) -> [[f32; 4]; 4] {
    let (sin, cos) = angle.sin_cos();
    [
        [cos, 0.0, -sin, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sin, 0.0, cos, 0.0],
        [x, 1.0, 0.0, 1.0],
    ]
}

fn evaluate_reactive_motion_recovery(label: &str, old_pose: &[u8], frames: &[Vec<u8>]) {
    assert_eq!(
        frames.len(),
        24,
        "reactive recovery expects one and a half 16-sample jitter cycles"
    );
    let stable = average_rgba(&frames[8..]);
    let movement = calculate_diff_metrics(old_pose, &stable, W, H);
    // Fully reactive transmission intentionally consumes the current refracted
    // result immediately. Compare equal Halton phases one 16-frame cycle apart
    // so legitimate subpixel coverage changes are not misclassified as trails.
    let recovery = frames[..8]
        .iter()
        .zip(&frames[16..])
        .map(|(frame, phase_match)| calculate_diff_metrics(phase_match, frame, W, H))
        .collect::<Vec<_>>();
    let severe = frames[..8]
        .iter()
        .zip(&frames[16..])
        .map(|(frame, phase_match)| severe_pixel_fraction(phase_match, frame))
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
        "temporal-corpus {label} movement_mean={:.4} initial_phase_mean={:.4} \
         frame4_phase_mean={:.4} frame4_phase_outliers={:.4}% \
         trail_frames={trail_frames} stable_flicker={stable_mean:.4}",
        movement.mean_rgb,
        recovery[0].mean_rgb,
        recovery[4].mean_rgb,
        recovery[4].outlier_pixel_fraction * 100.0,
    );
    assert!(
        movement.mean_rgb >= 1.0 && movement.outlier_pixel_fraction >= 0.01,
        "{label} negative control did not produce visible object motion"
    );
    assert!(
        trail_frames <= 4,
        "{label} left severe motion trails beyond four frames"
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
fn moving_physical_refraction_writes_velocity_reactive_coverage_and_no_trail() {
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

    let (vertices, indices) = cube_verts(0.8, [1.0; 4]);
    let glass = eng.scene.create_node();
    eng.scene.update_geometry(glass, vertices, indices);
    eng.scene.set_material_pbr(glass, 0.08, 0.0);
    eng.scene.set_material_transmission(
        glass,
        MaterialTransmission {
            authored: true,
            factor: 1.0,
            ior_authored: true,
            ior: 1.52,
            volume_authored: true,
            thickness_factor: 0.9,
            attenuation_distance: 0.7,
            attenuation_color: [0.08, 0.75, 1.0],
            thickness_source: MaterialThicknessSource::Authored,
            ..Default::default()
        },
    );

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 22.0, 255.0);
        r.begin_mode_3d(0.0, 2.0, 6.2, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.9, 0.75, 1.8);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 35.0, 42.0, 58.0, 255.0);
        for y in 0..3 {
            for x in -3..=3 {
                let alternate = (x + y) & 1 == 0;
                r.draw_cube(
                    x as f64 * 0.72,
                    0.42 + y as f64 * 0.72,
                    -1.8,
                    0.62,
                    0.62,
                    0.3,
                    if alternate { 235.0 } else { 30.0 },
                    if alternate { 55.0 } else { 175.0 },
                    if alternate { 25.0 } else { 235.0 },
                    255.0,
                );
            }
        }
    };
    let capture = |eng: &mut EngineState| render(eng, 1, draw_scene).2;

    eng.scene
        .set_transform(glass, glass_transform(-1.45, -0.45));
    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture(&mut eng);
    }
    let old_pose = capture(&mut eng);

    eng.scene.set_transform(glass, glass_transform(1.45, 0.55));
    let directory =
        std::env::temp_dir().join(format!("bloom-refraction-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture(&mut eng));
    }

    let motion = image::open(directory.join("taa-motion.png"))
        .expect("refractive motion capture did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    let reasons = image::open(directory.join("taa-rejection-reason.png"))
        .expect("refractive motion capture did not emit TAA rejection reasons")
        .to_rgb8();
    let reactive_pixels = reasons
        .pixels()
        .filter(|pixel| pixel[0] < 40 && pixel[1] > 180 && pixel[2] > 210)
        .count();
    eprintln!(
        "temporal-corpus refractive moving_pixels={moving_pixels} \
         reactive_pixels={reactive_pixels}"
    );
    assert!(
        moving_pixels >= 250,
        "moving refractive geometry wrote no meaningful velocity"
    );
    assert!(
        reactive_pixels >= 250,
        "moving physical refraction wrote no reactive TAA coverage"
    );
    assert!(
        reactive_pixels >= moving_pixels * 18 / 10,
        "reactive rejection did not protect both the current and newly \
         revealed refractive footprints: moving_pixels={moving_pixels}, \
         reactive_pixels={reactive_pixels}"
    );
    evaluate_reactive_motion_recovery("physical-refraction", &old_pose, &frames);
    assert!(
        eng.renderer
            .quality_runtime_paths_json()
            .contains("\"temporal_reactive\":{\"enabled\":true,\"active\":true"),
        "physical refraction did not keep the reactive composition path active"
    );

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept refractive motion diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}
