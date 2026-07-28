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

fn capture_preset(eng: &mut EngineState, preset: u32, legacy_scale: Option<f32>) -> Vec<u8> {
    eng.renderer.apply_quality_preset(preset);
    if let Some(scale) = legacy_scale {
        eng.renderer.set_render_scale(scale);
    }
    configure_reconstruction_scene(&mut eng.renderer);
    eng.renderer.reset_temporal_history();
    render(eng, 24, |eng| {
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
    let default_medium = capture_preset(&mut eng, 2, None);
    let ultra = capture_preset(&mut eng, 4, None);
    let legacy_energy = detail_energy(&legacy_half);
    let default_energy = detail_energy(&default_medium);
    let ultra_energy = detail_energy(&ultra);
    let legacy_to_native = calculate_diff_metrics(&ultra, &legacy_half, W, H);
    let default_to_native = calculate_diff_metrics(&ultra, &default_medium, W, H);
    eprintln!(
        "quality-preset detail_energy legacy_half={legacy_energy:.4} \
         default_075={default_energy:.4} ultra_native={ultra_energy:.4} \
         legacy_native_mean={:.4} default_native_mean={:.4}",
        legacy_to_native.mean_rgb, default_to_native.mean_rgb,
    );

    assert!(
        default_energy >= legacy_energy * 1.02,
        "0.75 default did not resolve meaningfully more detail than legacy 0.5: \
         {default_energy:.4} vs {legacy_energy:.4}"
    );
    assert!(
        default_to_native.mean_rgb < legacy_to_native.mean_rgb * 0.90,
        "0.75 default was not materially closer to native Ultra than legacy 0.5: \
         default={default_to_native:?}, legacy={legacy_to_native:?}"
    );
}
