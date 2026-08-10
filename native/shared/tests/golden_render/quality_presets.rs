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

fn draw_reconstruction_scene(eng: &mut EngineState) {
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
    render(eng, frames, draw_reconstruction_scene).2
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
    render(eng, 2, draw_reconstruction_scene).2
}

fn downsample_box_2x(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    assert_eq!(width % 2, 0);
    assert_eq!(height % 2, 0);
    assert_eq!(rgba.len(), (width * height * 4) as usize);
    let output_width = width / 2;
    let output_height = height / 2;
    let mut output = vec![0u8; (output_width * output_height * 4) as usize];
    for y in 0..output_height {
        for x in 0..output_width {
            for channel in 0..4 {
                let mut sum = 0u16;
                for oy in 0..2 {
                    for ox in 0..2 {
                        let index = ((((y * 2 + oy) * width + x * 2 + ox) * 4) + channel) as usize;
                        sum += u16::from(rgba[index]);
                    }
                }
                output[((y * output_width + x) * 4 + channel) as usize] = ((sum + 2) / 4) as u8;
            }
        }
    }
    output
}

fn capture_native_unsharpened(eng: &mut EngineState, taa: bool, frames: u32) -> Vec<u8> {
    eng.renderer.apply_quality_preset(4);
    eng.renderer.set_render_scale(1.0);
    configure_reconstruction_scene(&mut eng.renderer);
    eng.renderer.set_taa_enabled(taa);
    eng.renderer.set_sharpen_strength(0.0);
    eng.renderer.reset_temporal_history();
    render(eng, frames, draw_reconstruction_scene).2
}

fn capture_fractional_unsharpened(eng: &mut EngineState, frames: u32) -> Vec<u8> {
    eng.renderer.apply_quality_preset(2);
    eng.renderer.set_render_scale(0.75);
    configure_reconstruction_scene(&mut eng.renderer);
    eng.renderer.set_taa_enabled(true);
    eng.renderer.set_sharpen_strength(0.0);
    eng.renderer.reset_temporal_history();
    render(eng, frames, draw_reconstruction_scene).2
}

#[test]
fn native_temporal_reconstruction_tracks_supersampled_reference() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    eng.renderer.resize(W * 2, H * 2, W * 2, H * 2);
    let supersampled = capture_native_unsharpened(&mut eng, false, 2);
    let reference = downsample_box_2x(&supersampled, W * 2, H * 2);
    eng.renderer.resize(W, H, W, H);
    let no_taa = capture_native_unsharpened(&mut eng, false, 2);
    let display_history = capture_native_unsharpened(&mut eng, true, 24);
    let no_taa_metrics = calculate_diff_metrics(&reference, &no_taa, W, H);
    let display_metrics = calculate_diff_metrics(&reference, &display_history, W, H);
    eprintln!("native-reference no_taa={no_taa_metrics:?} display_history={display_metrics:?}");

    assert!(
        display_metrics.ssim >= no_taa_metrics.ssim,
        "settled native temporal reconstruction diverged from the supersampled reference: \
         no_taa={no_taa_metrics:?}, temporal={display_metrics:?}"
    );
    assert!(
        display_metrics.mean_rgb <= no_taa_metrics.mean_rgb * 0.80,
        "native temporal reconstruction did not materially improve on the aliased single frame: \
         no_taa={no_taa_metrics:?}, temporal={display_metrics:?}"
    );
}

#[test]
fn fractional_temporal_reconstruction_tracks_supersampled_reference() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    eng.renderer.resize(W * 2, H * 2, W * 2, H * 2);
    let supersampled = capture_native_unsharpened(&mut eng, false, 2);
    let reference = downsample_box_2x(&supersampled, W * 2, H * 2);
    eng.renderer.resize(W, H, W, H);
    let fractional = capture_fractional_unsharpened(&mut eng, 24);
    let metrics = calculate_diff_metrics(&reference, &fractional, W, H);
    eprintln!("fractional-reference temporal={metrics:?}");

    assert!(
        metrics.ssim >= 0.9902,
        "fractional temporal reconstruction diverged from the supersampled reference: {metrics:?}"
    );
    assert!(
        metrics.mean_rgb <= 0.65,
        "fractional temporal reconstruction exceeded the supersampled-reference RGB gate: {metrics:?}"
    );
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
    let ultra_seed = capture_preset_frames(&mut eng, 4, None, 1);
    let ultra = capture_preset(&mut eng, 4, None);
    let ultra_paths: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("Ultra reconstruction telemetry is valid JSON");
    let ultra_no_taa = capture_sharpen_scene(&mut eng, 0.85);
    let legacy_energy = detail_energy(&legacy_half);
    let default_energy = detail_energy(&default_medium);
    let default_seed_energy = detail_energy(&default_seed);
    let ultra_seed_energy = detail_energy(&ultra_seed);
    let ultra_energy = detail_energy(&ultra);
    let ultra_no_taa_energy = detail_energy(&ultra_no_taa);
    let legacy_to_native = calculate_diff_metrics(&ultra, &legacy_half, W, H);
    let default_to_native = calculate_diff_metrics(&ultra, &default_medium, W, H);
    eprintln!(
        "quality-preset detail_energy legacy_half={legacy_energy:.4} \
         default_seed={default_seed_energy:.4} default_075={default_energy:.4} \
         ultra_seed={ultra_seed_energy:.4} ultra_native={ultra_energy:.4} \
         ultra_no_taa={ultra_no_taa_energy:.4} \
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
        ultra_energy >= 3.10,
        "native Ultra detail regressed below the output-aligned TAA floor: \
         native={ultra_energy:.4}"
    );
    // Native TAA is an anti-aliasing resolve, not an upscale. It may trade a
    // bounded amount of single-frame aliased Laplacian energy for stability,
    // but must not turn the native reference into a soft target that makes a
    // fractional reconstruction look deceptively close. Keep both the seeded
    // and TAA-off references visible in this gate.
    assert!(
        ultra_energy >= ultra_seed_energy * 0.80,
        "native temporal accumulation fell below the accepted seeded-detail baseline: \
         seed={ultra_seed_energy:.4}, settled={ultra_energy:.4}"
    );
    assert!(
        ultra_energy >= ultra_no_taa_energy * 0.80,
        "native TAA fell below the accepted same-frame no-TAA detail baseline: \
         no_taa={ultra_no_taa_energy:.4}, settled={ultra_energy:.4}"
    );
    assert!(
        default_energy >= 2.20,
        "0.75 settled detail regressed below its accepted reconstruction floor: \
         default={default_energy:.4}"
    );
    // Temporal AA should reduce the seeded frame's aliased Laplacian energy,
    // but must not keep diffusing a static surface. The zero-velocity
    // prev-jitter regression measured 72.9%; trusting zero velocity reached
    // 81.2%, and the source-footprint color-change policy reaches at least
    // 86% without accepting stale geometry history. This also protects the
    // bounded stationary reconstruction residual from being silently removed.
    assert!(
        default_energy >= default_seed_energy * 0.86,
        "static TAA accumulation lost more detail than the bounded AA window permits: \
         seed={default_seed_energy:.4}, settled={default_energy:.4}"
    );
    assert!(
        // Native is intentionally a sharper target now, so this comparison
        // is a tier-ordering guard rather than a demand that the unchanged
        // 0.75 path converge toward the former soft native image. Its own
        // absolute and seeded-detail floors above remain authoritative.
        default_to_native.mean_rgb < legacy_to_native.mean_rgb * 0.65,
        "0.75 default was not materially closer to native Ultra than legacy 0.5: \
         default={default_to_native:?}, legacy={legacy_to_native:?}"
    );
    let reconstruction = &default_paths["temporal_reconstruction"];
    assert_eq!(reconstruction["enabled"].as_bool(), Some(true));
    assert_eq!(
        reconstruction["mode"].as_str(),
        Some("source-footprint-temporal")
    );
    assert_eq!(
        reconstruction["history_filter"].as_str(),
        Some("camera-motion-phase-compressed-linear")
    );
    assert_eq!(reconstruction["history_filter_samples"].as_u64(), Some(1));
    assert_eq!(
        reconstruction["stationary_reconstruction_detail_strength"].as_f64(),
        Some(0.2)
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_detail_clamp"].as_f64(),
        Some(0.08)
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_additional_samples"].as_u64(),
        Some(0)
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_motion_gated"].as_bool(),
        Some(true)
    );
    assert_eq!(
        reconstruction["output_detail_filter"].as_str(),
        Some("depth-aware-local-luma-hull")
    );
    assert_eq!(reconstruction["output_detail_strength"].as_f64(), Some(0.4));
    assert_eq!(
        reconstruction["output_detail_depth_samples"].as_u64(),
        Some(1)
    );
    assert_eq!(
        reconstruction["output_detail_additional_persistent_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(
        reconstruction["output_detail_additional_graph_passes"].as_u64(),
        Some(0)
    );
    assert_eq!(reconstruction["camera_moving"].as_bool(), Some(false));
    assert_eq!(reconstruction["render_scale"].as_f64(), Some(0.75));
    assert_eq!(reconstruction["jitter_spread"].as_f64(), Some(1.0));
    assert_eq!(
        ultra_paths["temporal_reconstruction"]["jitter_spread"].as_f64(),
        Some(0.5)
    );
    assert_eq!(
        reconstruction["statistics_footprint_input_pixels"].as_f64(),
        Some(0.75)
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
