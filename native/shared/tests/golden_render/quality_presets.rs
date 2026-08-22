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

const GLOSSY_DETAIL_HANDLE: u64 = 0x7AA5_1490;

/// A compact stand-in for Bistro's painted scooter/bottle materials. The
/// curved surface exercises the imported specular-glossiness workflow while
/// the one/two-texel markings expose temporal loss that the untextured cube
/// fixture cannot see.
fn install_glossy_detail_fixture(eng: &mut EngineState) {
    const TEXTURE_SIZE: u32 = 256;
    const COLUMNS: u32 = 32;
    const ROWS: u32 = 16;

    let mut base_color = Vec::with_capacity((TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize);
    let mut normal = Vec::with_capacity(base_color.capacity());
    let mut specular_gloss = Vec::with_capacity(base_color.capacity());
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            let label = (54..=202).contains(&x) && (84..=171).contains(&y);
            let label_border = label && (!(58..=198).contains(&x) || !(88..=167).contains(&y));
            let fine_print =
                label && (96..=160).contains(&y) && ((y - 96) % 9 <= 1) && (72..=188).contains(&x);
            let vertical_trim = x % 31 <= 1;
            let paint = if label_border {
                [18, 22, 31]
            } else if fine_print {
                [28, 38, 58]
            } else if label {
                [222, 208, 166]
            } else if vertical_trim {
                [215, 225, 242]
            } else {
                // Slight authored paint variation prevents a uniform-color
                // panel from hiding phase-dependent texture loss.
                let grain = ((x.wrapping_mul(13) ^ y.wrapping_mul(29)) & 7) as u8;
                [12 + grain, 52 + grain, 154 + grain * 2]
            };
            base_color.extend_from_slice(&[paint[0], paint[1], paint[2], 255]);

            let nx = (((x as f32 * 0.31).sin() * 5.0) + 128.0).round() as u8;
            let ny = (((y as f32 * 0.27).cos() * 4.0) + 128.0).round() as u8;
            normal.extend_from_slice(&[nx, ny, 255, 255]);

            let gloss_variation = if vertical_trim { 238 } else { 247 };
            specular_gloss.extend_from_slice(&[190, 190, 190, gloss_variation]);
        }
    }

    let base_color =
        eng.renderer
            .register_texture_kind(TEXTURE_SIZE, TEXTURE_SIZE, &base_color, false);
    let normal = eng
        .renderer
        .register_texture_kind(TEXTURE_SIZE, TEXTURE_SIZE, &normal, true);
    let specular_gloss =
        eng.renderer
            .register_texture_kind(TEXTURE_SIZE, TEXTURE_SIZE, &specular_gloss, false);

    let mut vertices = Vec::with_capacity(((COLUMNS + 1) * (ROWS + 1)) as usize);
    for row in 0..=ROWS {
        let v = row as f32 / ROWS as f32;
        let y = 1.55 - v * 3.10;
        for column in 0..=COLUMNS {
            let u = column as f32 / COLUMNS as f32;
            let x = (u * 2.0 - 1.0) * 2.75;
            let curve_x = x / 2.75;
            let z = -0.34 * curve_x * curve_x;
            let dz_dx = -0.68 * x / (2.75 * 2.75);
            let inv_len = (1.0 + dz_dx * dz_dx).sqrt().recip();
            vertices.push(Vertex3D {
                position: [x, y, z],
                normal: [-dz_dx * inv_len, 0.0, inv_len],
                color: [1.0; 4],
                uv: [u, v],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [inv_len, 0.0, dz_dx * inv_len, 1.0],
            });
        }
    }
    let mut indices = Vec::with_capacity((COLUMNS * ROWS * 6) as usize);
    let stride = COLUMNS + 1;
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let a = row * stride + column;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    assert!(eng.renderer.cache_model_if_static(
        GLOSSY_DETAIL_HANDLE,
        &[MeshData {
            vertices,
            secondary_tex_coords: None,
            indices,
            texture_idx: Some(base_color),
            normal_texture_idx: Some(normal),
            metallic_roughness_texture_idx: Some(specular_gloss),
            specular_glossiness_factor: Some([0.73, 0.73, 0.73, 0.95]),
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));
}

fn draw_glossy_detail_fixture(eng: &mut EngineState, camera_x: f32) {
    let r = &mut eng.renderer;
    r.set_clear_color(4.0, 5.0, 8.0, 255.0);
    r.begin_mode_3d(
        camera_x,
        0.05,
        6.3,
        camera_x * 0.12,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        43.0,
        0.0,
    );
    r.set_ambient_light(185.0, 198.0, 225.0, 0.20);
    r.add_directional_light(-0.45, -0.35, -1.0, 1.0, 0.92, 0.78, 3.2);
    r.add_point_light(-1.8, 1.6, 2.8, 9.0, 0.55, 0.72, 1.0, 9.0);
    r.draw_model_cached(GLOSSY_DETAIL_HANDLE, [0.0; 3], 1.0, [1.0; 4]);
}

fn configure_glossy_detail_capture(renderer: &mut Renderer, taa: bool, render_scale: f32) {
    renderer.apply_quality_preset(4);
    renderer.set_render_scale(render_scale);
    configure_reconstruction_scene(renderer);
    // The production Bistro path keeps directional shadows enabled. Material
    // mip policy currently shares the per-view cascade split upload, so this
    // fixture must exercise the same path while we qualify that ownership.
    renderer.set_shadows_enabled(true);
    renderer.set_taa_enabled(taa);
    renderer.set_sharpen_strength(0.0);
    renderer.set_auto_exposure(false);
    renderer.set_manual_exposure(1.0);
    renderer.reset_temporal_history();
}

fn capture_glossy_detail(
    eng: &mut EngineState,
    taa: bool,
    render_scale: f32,
    frames: u32,
    camera_x: f32,
) -> Vec<u8> {
    configure_glossy_detail_capture(&mut eng.renderer, taa, render_scale);
    render(eng, frames, |eng| draw_glossy_detail_fixture(eng, camera_x)).2
}

fn profile_fractional_taa(eng: &mut EngineState, frames: u32) -> f64 {
    eng.renderer.resize(1600, 900, 1600, 900);
    configure_glossy_detail_capture(&mut eng.renderer, true, 0.75);
    for _ in 0..24 {
        eng.begin_frame();
        draw_glossy_detail_fixture(eng, 0.0);
        eng.end_frame();
    }

    eng.profiler.set_enabled(true);
    for _ in 0..frames.max(1) {
        eng.begin_frame();
        draw_glossy_detail_fixture(eng, 0.0);
        eng.end_frame();
    }
    let taa_gpu_us = eng
        .profiler
        .snapshot()
        .into_iter()
        .find_map(|(label, _, gpu)| (label == "taa_pass").then_some(gpu?))
        .unwrap_or(0.0);
    eng.profiler.set_enabled(false);
    taa_gpu_us
}

fn temporal_derivative_error(
    previous_reference: &[u8],
    reference: &[u8],
    previous_candidate: &[u8],
    candidate: &[u8],
) -> f64 {
    let mut total = 0u64;
    let mut samples = 0u64;
    for (((previous_reference, reference), previous_candidate), candidate) in previous_reference
        .chunks_exact(4)
        .zip(reference.chunks_exact(4))
        .zip(previous_candidate.chunks_exact(4))
        .zip(candidate.chunks_exact(4))
    {
        for channel in 0..3 {
            let reference_delta =
                i16::from(reference[channel]) - i16::from(previous_reference[channel]);
            let candidate_delta =
                i16::from(candidate[channel]) - i16::from(previous_candidate[channel]);
            total += u64::from(reference_delta.abs_diff(candidate_delta));
            samples += 1;
        }
    }
    total as f64 / samples as f64
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
    assert!(
        display_metrics.rmse_luminance <= no_taa_metrics.rmse_luminance
            && display_metrics.mean_oklab_delta <= no_taa_metrics.mean_oklab_delta
            && display_metrics.mean_edge_delta <= no_taa_metrics.mean_edge_delta,
        "native temporal reconstruction regressed a perceptual reference metric: \
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
fn glossy_textured_temporal_reconstruction_tracks_supersampled_reference() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    install_glossy_detail_fixture(&mut eng);

    eng.renderer.resize(W * 2, H * 2, W * 2, H * 2);
    let supersampled = capture_glossy_detail(&mut eng, false, 1.0, 2, 0.0);
    let reference = downsample_box_2x(&supersampled, W * 2, H * 2);

    eng.renderer.resize(W, H, W, H);
    let no_taa = capture_glossy_detail(&mut eng, false, 1.0, 2, 0.0);
    let native_taa = capture_glossy_detail(&mut eng, true, 1.0, 24, 0.0);
    let fractional_no_taa = capture_glossy_detail(&mut eng, false, 0.75, 2, 0.0);
    let fractional_seed = capture_glossy_detail(&mut eng, true, 0.75, 1, 0.0);
    let fractional_taa = capture_glossy_detail(&mut eng, true, 0.75, 24, 0.0);

    if let Some(directory) = std::env::var_os("BLOOM_KEEP_GLOSSY_TEMPORAL") {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).expect("create glossy temporal artifact directory");
        for (name, pixels) in [
            ("reference.png", &reference),
            ("native-no-taa.png", &no_taa),
            ("native-taa.png", &native_taa),
            ("fractional-no-taa.png", &fractional_no_taa),
            ("fractional-seed.png", &fractional_seed),
            ("fractional-settled.png", &fractional_taa),
        ] {
            image::save_buffer(directory.join(name), pixels, W, H, image::ColorType::Rgba8)
                .expect("write glossy temporal artifact");
        }
    }

    let no_taa_metrics = calculate_diff_metrics(&reference, &no_taa, W, H);
    let native_metrics = calculate_diff_metrics(&reference, &native_taa, W, H);
    let fractional_no_taa_metrics = calculate_diff_metrics(&reference, &fractional_no_taa, W, H);
    let fractional_seed_metrics = calculate_diff_metrics(&reference, &fractional_seed, W, H);
    let fractional_metrics = calculate_diff_metrics(&reference, &fractional_taa, W, H);
    let reference_detail = detail_energy(&reference);
    let no_taa_detail = detail_energy(&no_taa);
    let native_detail = detail_energy(&native_taa);
    let fractional_no_taa_detail = detail_energy(&fractional_no_taa);
    let fractional_seed_detail = detail_energy(&fractional_seed);
    let fractional_detail = detail_energy(&fractional_taa);
    eprintln!(
        "glossy-detail reference={reference_detail:.4} no_taa={no_taa_detail:.4} \
         native={native_detail:.4} fractional_no_taa={fractional_no_taa_detail:.4} \
         fractional_seed={fractional_seed_detail:.4} fractional={fractional_detail:.4} \
         no_taa_metrics={no_taa_metrics:?} native_metrics={native_metrics:?} \
         fractional_no_taa_metrics={fractional_no_taa_metrics:?} \
         fractional_seed_metrics={fractional_seed_metrics:?} \
         fractional_metrics={fractional_metrics:?}"
    );

    // A real anti-aliasing resolve must improve the authored glossy material
    // against a supersampled reference, not merely produce more raw edge
    // energy. This is the failure mode hidden by the untextured primitive rig.
    assert!(
        native_metrics.ssim >= no_taa_metrics.ssim,
        "native TAA made the glossy textured material less reference-like: \
         no_taa={no_taa_metrics:?}, temporal={native_metrics:?}"
    );
    assert!(
        native_metrics.mean_rgb <= no_taa_metrics.mean_rgb,
        "native TAA increased glossy textured material error: \
         no_taa={no_taa_metrics:?}, temporal={native_metrics:?}"
    );
    assert!(
        native_metrics.rmse_luminance <= no_taa_metrics.rmse_luminance
            && native_metrics.mean_oklab_delta <= no_taa_metrics.mean_oklab_delta
            && native_metrics.mean_edge_delta <= no_taa_metrics.mean_edge_delta,
        "native TAA regressed glossy luminance, colour, or edge fidelity: \
         no_taa={no_taa_metrics:?}, temporal={native_metrics:?}"
    );
    assert!(
        native_detail >= reference_detail * 0.72,
        "native TAA erased too much reference material detail: \
         reference={reference_detail:.4}, temporal={native_detail:.4}"
    );
    assert!(
        fractional_metrics.ssim >= 0.972,
        "fractional reconstruction diverged on glossy authored detail: \
         {fractional_metrics:?}"
    );
    assert!(
        fractional_detail >= reference_detail * 0.76,
        "fractional reconstruction erased glossy authored detail: \
         reference={reference_detail:.4}, temporal={fractional_detail:.4}"
    );
    assert!(
        fractional_metrics.mean_rgb <= 2.20,
        "fractional glossy material error exceeded the qualified baseline: \
         {fractional_metrics:?}"
    );
    if let Ok(frames) = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES") {
        let frames = frames.parse::<u32>().expect("profile frame count");
        let taa_gpu_us = profile_fractional_taa(&mut eng, frames);
        eprintln!("fractional-reconstruction taa_gpu_us={taa_gpu_us:.3} frames={frames}");
    }
}

fn glossy_slow_pan_metrics(render_scale: f32) -> Option<(f64, f64, f64, f64, Vec<f64>)> {
    const FRAMES: usize = 12;
    const CAMERA_STEP: f32 = 0.004;

    let mut eng = try_engine()?;
    install_glossy_detail_fixture(&mut eng);

    eng.renderer.resize(W * 2, H * 2, W * 2, H * 2);
    configure_glossy_detail_capture(&mut eng.renderer, false, 1.0);
    let mut references = Vec::with_capacity(FRAMES);
    for frame in 0..FRAMES {
        let camera_x = frame as f32 * CAMERA_STEP;
        let supersampled = render(&mut eng, 1, |eng| draw_glossy_detail_fixture(eng, camera_x)).2;
        references.push(downsample_box_2x(&supersampled, W * 2, H * 2));
    }

    eng.renderer.resize(W, H, W, H);
    configure_glossy_detail_capture(&mut eng.renderer, true, render_scale);
    let _ = render(&mut eng, 24, |eng| draw_glossy_detail_fixture(eng, 0.0));
    let mut candidates = Vec::with_capacity(FRAMES);
    for frame in 0..FRAMES {
        let camera_x = frame as f32 * CAMERA_STEP;
        candidates.push(render(&mut eng, 1, |eng| draw_glossy_detail_fixture(eng, camera_x)).2);
    }

    let mut mean_rgb = 0.0;
    let mut mean_ssim = 0.0;
    let mut minimum_ssim = 1.0f64;
    for (reference, candidate) in references.iter().zip(&candidates) {
        let metrics = calculate_diff_metrics(reference, candidate, W, H);
        mean_rgb += metrics.mean_rgb;
        mean_ssim += metrics.ssim;
        minimum_ssim = minimum_ssim.min(metrics.ssim);
    }
    mean_rgb /= FRAMES as f64;
    mean_ssim /= FRAMES as f64;
    let derivative_errors = references
        .windows(2)
        .zip(candidates.windows(2))
        .map(|(reference, candidate)| {
            temporal_derivative_error(&reference[0], &reference[1], &candidate[0], &candidate[1])
        })
        .collect::<Vec<_>>();
    let derivative_error = derivative_errors.iter().sum::<f64>() / (FRAMES - 1) as f64;
    eprintln!(
        "glossy-slow-pan scale={render_scale:.2} mean_rgb={mean_rgb:.6} \
         mean_ssim={mean_ssim:.6} \
         minimum_ssim={minimum_ssim:.6} derivative_error={derivative_error:.6} \
         derivative_frames={derivative_errors:?}"
    );

    Some((
        mean_rgb,
        mean_ssim,
        minimum_ssim,
        derivative_error,
        derivative_errors,
    ))
}

#[test]
fn fractional_glossy_slow_pan_tracks_supersampled_motion() {
    let Some((_, mean_ssim, minimum_ssim, derivative_error, _)) = glossy_slow_pan_metrics(0.75)
    else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    assert!(
        mean_ssim >= 0.973 && minimum_ssim >= 0.965,
        "fractional glossy slow pan diverged from supersampled motion: \
         mean_ssim={mean_ssim:.6}, minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        derivative_error <= 0.145,
        "fractional glossy slow pan added excessive temporal variation: \
         derivative_error={derivative_error:.6}"
    );
}

#[test]
fn native_glossy_slow_pan_tracks_supersampled_motion() {
    let Some((mean_rgb, mean_ssim, minimum_ssim, derivative_error, _)) =
        glossy_slow_pan_metrics(1.0)
    else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    assert!(
        mean_ssim >= 0.985 && minimum_ssim >= 0.980,
        "native glossy slow pan diverged from supersampled motion: \
         mean_ssim={mean_ssim:.6}, minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        mean_rgb <= 0.85,
        "native glossy slow pan exceeded the supersampled-reference RGB gate: \
         mean_rgb={mean_rgb:.6}"
    );
    assert!(
        derivative_error <= 0.30,
        "native glossy slow pan added excessive temporal variation: \
         derivative_error={derivative_error:.6}"
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
        reconstruction["source_filter"].as_str(),
        Some("exact-separable-catmull-rom")
    );
    assert_eq!(reconstruction["source_filter_samples"].as_u64(), Some(9));
    assert_eq!(
        reconstruction["statistics_filter"].as_str(),
        Some("variance-corrected-cross")
    );
    assert_eq!(
        reconstruction["statistics_filter_samples"].as_u64(),
        Some(5)
    );
    assert_eq!(reconstruction["composed_source_samples"].as_u64(), Some(14));
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
