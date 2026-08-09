use super::*;

fn luma(byte_pixel: &[u8]) -> i32 {
    (i32::from(byte_pixel[0]) * 54 + i32::from(byte_pixel[1]) * 183 + i32::from(byte_pixel[2]) * 19)
        / 256
}

/// Mean absolute 4-neighbor Laplacian. Unlike total edge contrast, this rises
/// when the same visible edge is resolved over fewer output pixels.
fn detail_energy(rgba: &[u8]) -> f64 {
    let mut total = 0u64;
    let mut samples = 0u64;
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            let pixel = |x: u32, y: u32| {
                let index = ((y * W + x) * 4) as usize;
                luma(&rgba[index..index + 4])
            };
            let center = pixel(x, y);
            let laplacian =
                center * 4 - pixel(x - 1, y) - pixel(x + 1, y) - pixel(x, y - 1) - pixel(x, y + 1);
            total += u64::from(laplacian.unsigned_abs());
            samples += 1;
        }
    }
    total as f64 / samples as f64
}

fn configure_reconstruction_scene(renderer: &mut Renderer) {
    renderer.set_ssao_enabled(false);
    renderer.set_ssr_enabled(false);
    renderer.set_ssgi_enabled(false);
    renderer.set_bloom_enabled(false);
    renderer.set_motion_blur_enabled(false);
    renderer.set_sss_enabled(false);
    renderer.set_shadows_enabled(false);
}

fn capture_preset_frames(
    eng: &mut EngineState,
    preset: u32,
    legacy_scale: Option<f32>,
    frames: u32,
) -> Vec<u8> {
    eng.renderer.apply_quality_preset(preset);
    if let Some(scale) = legacy_scale {
        eng.renderer.set_render_scale(scale);
    }
    configure_reconstruction_scene(&mut eng.renderer);
    eng.renderer.reset_temporal_history();
    render(eng, frames, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(5.0, 7.0, 12.0, 255.0);
        r.begin_mode_3d(4.4, 3.5, 6.0, 0.0, 0.45, 0.0, 0.0, 1.0, 0.0, 51.0, 0.0);
        r.set_ambient_light(255.0, 255.0, 255.0, 0.82);
        r.draw_grid(48, 0.16);
        for column in -8..=8 {
            let bright = column & 1 == 0;
            r.draw_cube(
                f64::from(column) * 0.19,
                0.55,
                -0.35 + f64::from(column & 3) * 0.08,
                0.075,
                1.10,
                0.075,
                if bright { 238.0 } else { 35.0 },
                if bright { 226.0 } else { 55.0 },
                if bright { 205.0 } else { 85.0 },
                255.0,
            );
        }
    })
    .2
}

fn capture_preset(eng: &mut EngineState, preset: u32, legacy_scale: Option<f32>) -> Vec<u8> {
    capture_preset_frames(eng, preset, legacy_scale, 24)
}

fn capture_sharpen_scene(eng: &mut EngineState, strength: f32) -> Vec<u8> {
    eng.renderer.apply_quality_preset(4);
    configure_reconstruction_scene(&mut eng.renderer);
    eng.renderer.set_taa_enabled(false);
    eng.renderer.set_sharpen_strength(strength);
    eng.renderer.reset_temporal_history();
    render(eng, 2, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(5.0, 7.0, 12.0, 255.0);
        r.begin_mode_3d(4.4, 3.5, 6.0, 0.0, 0.45, 0.0, 0.0, 1.0, 0.0, 51.0, 0.0);
        r.set_ambient_light(255.0, 255.0, 255.0, 0.82);
        r.draw_grid(48, 0.16);
        for column in -8..=8 {
            let bright = column & 1 == 0;
            r.draw_cube(
                f64::from(column) * 0.19,
                0.55,
                -0.35 + f64::from(column & 3) * 0.08,
                0.075,
                1.10,
                0.075,
                if bright { 238.0 } else { 35.0 },
                if bright { 226.0 } else { 55.0 },
                if bright { 205.0 } else { 85.0 },
                255.0,
            );
        }
    })
    .2
}

#[test]
fn default_and_ultra_presets_resolve_more_detail_than_legacy_half_scale() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    let legacy_half = capture_preset(&mut eng, 2, Some(0.5));
    let default_seed = capture_preset_frames(&mut eng, 2, None, 1);
    let default_medium = capture_preset(&mut eng, 2, None);
    let default_paths: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("default reconstruction telemetry is valid JSON");
    let ultra = capture_preset(&mut eng, 4, None);
    let legacy_energy = detail_energy(&legacy_half);
    let default_energy = detail_energy(&default_medium);
    let default_seed_energy = detail_energy(&default_seed);
    let ultra_energy = detail_energy(&ultra);
    let legacy_to_native = calculate_diff_metrics(&ultra, &legacy_half, W, H);
    let default_to_native = calculate_diff_metrics(&ultra, &default_medium, W, H);
    eprintln!(
        "quality-preset detail_energy legacy_half={legacy_energy:.4} \
         default_seed={default_seed_energy:.4} default_075={default_energy:.4} \
         ultra_native={ultra_energy:.4} \
         legacy_native_mean={:.4} default_native_mean={:.4}",
        legacy_to_native.mean_rgb, default_to_native.mean_rgb,
    );

    // Gate the two reconstruction tiers independently. Native Ultra keeps a
    // static current sample on the output pixel, which intentionally raises
    // its detail ceiling without changing the established 0.75 path. A ratio
    // between those tiers would therefore turn a native-only improvement into
    // a false 0.75 regression. These floors retain margin for cross-adapter
    // raster differences while rejecting the measured pre-fix native softness
    // and any loss from the accepted 0.75 baseline.
    assert!(
        ultra_energy >= 2.60,
        "native Ultra detail regressed below the output-aligned TAA floor: \
         native={ultra_energy:.4}"
    );
    assert!(
        default_energy >= 2.10,
        "0.75 settled detail regressed below its accepted reconstruction floor: \
         default={default_energy:.4}"
    );
    // Temporal AA should reduce the seeded frame's aliased Laplacian energy,
    // but must not keep diffusing a static surface. The zero-velocity
    // prev-jitter regression measured 72.9%; trusting zero velocity reached
    // 81.2%, and the source-footprint color-change policy reaches at least
    // 83% without accepting stale geometry history.
    assert!(
        default_energy >= default_seed_energy * 0.83,
        "static TAA accumulation lost more detail than the bounded AA window permits: \
         seed={default_seed_energy:.4}, settled={default_energy:.4}"
    );
    assert!(
        default_to_native.mean_rgb < legacy_to_native.mean_rgb * 0.55,
        "0.75 default was not materially closer to native Ultra than legacy 0.5: \
         default={default_to_native:?}, legacy={legacy_to_native:?}"
    );
    let reconstruction = &default_paths["temporal_reconstruction"];
    assert_eq!(reconstruction["enabled"].as_bool(), Some(true));
    assert_eq!(
        reconstruction["mode"].as_str(),
        Some("source-footprint-temporal")
    );
    assert_eq!(reconstruction["render_scale"].as_f64(), Some(0.75));
    assert_eq!(
        reconstruction["statistics_footprint_input_pixels"].as_f64(),
        Some(0.8)
    );
    assert!(
        reconstruction["input_extent"][0].as_u64() < reconstruction["output_extent"][0].as_u64(),
        "fractional reconstruction telemetry did not expose distinct extents: {reconstruction}"
    );
    assert_eq!(
        reconstruction["additional_persistent_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(reconstruction["additional_graph_passes"].as_u64(), Some(0));
}

#[test]
fn composite_sharpen_preserves_detail_without_inking_silhouettes() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    let unsharpened = capture_sharpen_scene(&mut eng, 0.0);
    let sharpened = capture_sharpen_scene(&mut eng, 0.5);
    let mut silhouette_delta = 0u64;
    let mut silhouette_samples = 0u64;
    let mut material_changes = 0u64;
    let mut material_samples = 0u64;
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            let sample_luma = |rgba: &[u8], x: u32, y: u32| {
                let index = ((y * W + x) * 4) as usize;
                luma(&rgba[index..index + 4])
            };
            let center = sample_luma(&unsharpened, x, y);
            let neighbors = [
                sample_luma(&unsharpened, x - 1, y),
                sample_luma(&unsharpened, x + 1, y),
                sample_luma(&unsharpened, x, y - 1),
                sample_luma(&unsharpened, x, y + 1),
            ];
            let minimum = neighbors.iter().copied().fold(center, i32::min);
            let maximum = neighbors.iter().copied().fold(center, i32::max);
            let local_span = maximum - minimum;
            let sharpened_luma = sample_luma(&sharpened, x, y);
            let delta = center.abs_diff(sharpened_luma);
            if local_span >= 80 {
                silhouette_delta += u64::from(delta);
                silhouette_samples += 1;
            } else if (8..=48).contains(&local_span) {
                material_samples += 1;
                if delta >= 1 {
                    material_changes += 1;
                }
            }
        }
    }
    let silhouette_mean = silhouette_delta as f64 / silhouette_samples.max(1) as f64;
    let material_change_ratio = material_changes as f64 / material_samples.max(1) as f64;
    eprintln!(
        "quality-preset sharpen silhouette_mean_delta={silhouette_mean:.4} \
         silhouette_samples={silhouette_samples} material_change_ratio={material_change_ratio:.4} \
         material_samples={material_samples}"
    );
    assert!(
        silhouette_samples >= 100,
        "fixture exposed too few hard edges"
    );
    assert!(
        silhouette_mean <= 1.0,
        "composite sharpen drew a visible contour along hard silhouettes"
    );
    assert!(
        material_change_ratio >= 0.02,
        "edge suppression disabled sharpening on ordinary material detail"
    );
}
