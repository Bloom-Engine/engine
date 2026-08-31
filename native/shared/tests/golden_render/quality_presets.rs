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

fn profile_taa_reconstruction(eng: &mut EngineState, frames: u32, render_scale: f32) -> f64 {
    eng.renderer.resize(1600, 900, 1600, 900);
    configure_glossy_detail_capture(&mut eng.renderer, true, render_scale);
    let camera_step = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP")
        .ok()
        .map(|value| value.parse::<f32>().expect("profile camera step"))
        .unwrap_or(0.0);
    for frame in 0..24 {
        eng.begin_frame();
        draw_glossy_detail_fixture(eng, frame as f32 * camera_step);
        eng.end_frame();
    }

    eng.profiler.set_enabled(true);
    // Profiler snapshots retain the latest 120 frames. Sample every complete
    // rolling window and average those windows so a long qualification run
    // measures its full duration instead of reporting only its final, noisy
    // 120-frame tail.
    let frames = frames.max(1);
    let mut taa_gpu_windows = Vec::new();
    for frame in 0..frames {
        eng.begin_frame();
        draw_glossy_detail_fixture(eng, (frame + 24) as f32 * camera_step);
        eng.end_frame();
        if (frame + 1) % 120 == 0 || frame + 1 == frames {
            if let Some(taa_gpu_us) = eng
                .profiler
                .snapshot()
                .into_iter()
                .find_map(|(label, _, gpu)| (label == "taa_pass").then_some(gpu?))
            {
                taa_gpu_windows.push(taa_gpu_us);
            }
        }
    }
    eng.profiler.set_enabled(false);
    assert!(
        !taa_gpu_windows.is_empty(),
        "TAA profiling produced no GPU timestamp windows"
    );
    let mean = taa_gpu_windows.iter().sum::<f64>() / taa_gpu_windows.len() as f64;
    let sample_variance = if taa_gpu_windows.len() > 1 {
        taa_gpu_windows
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (taa_gpu_windows.len() - 1) as f64
    } else {
        0.0
    };
    let standard_error = sample_variance.sqrt() / (taa_gpu_windows.len() as f64).sqrt();
    eprintln!(
        "fractional-reconstruction profile_windows={} camera_step={camera_step:.6} \
         standard_error_us={standard_error:.3}",
        taa_gpu_windows.len(),
    );
    mean
}

#[derive(Clone, Copy, Debug)]
struct ReconstructionThroughputProfile {
    wall_mean_us: f64,
    wall_p50_us: f64,
    wall_p95_us: f64,
    cpu_mean_us: f64,
    cpu_p50_us: f64,
    cpu_p95_us: f64,
    gpu_mean_us: f64,
    gpu_p50_us: f64,
    gpu_p95_us: f64,
    taa_gpu_mean_us: f64,
}

fn throughput_stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    assert!(
        !samples.is_empty(),
        "throughput profile produced no samples"
    );
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[((samples.len() - 1) as f64 * 0.50).floor() as usize];
    let p95 = samples[((samples.len() - 1) as f64 * 0.95).ceil() as usize];
    (mean, p50, p95)
}

fn profile_reconstruction_throughput(
    eng: &mut EngineState,
    frames: u32,
    render_scale: f32,
) -> ReconstructionThroughputProfile {
    const WINDOW_FRAMES: u32 = 120;
    assert!(
        frames >= WINDOW_FRAMES && frames.is_multiple_of(WINDOW_FRAMES),
        "throughput frames must be a positive multiple of {WINDOW_FRAMES}"
    );
    eng.renderer.resize(1600, 900, 1600, 900);
    configure_glossy_detail_capture(&mut eng.renderer, true, render_scale);
    let camera_step = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP")
        .ok()
        .map(|value| value.parse::<f32>().expect("profile camera step"))
        .unwrap_or(0.002);
    for frame in 0..120 {
        eng.begin_frame();
        draw_glossy_detail_fixture(eng, frame as f32 * camera_step);
        eng.end_frame();
    }

    eng.profiler.set_enabled(true);
    let mut wall_samples = Vec::with_capacity(frames as usize);
    let mut cpu_samples = Vec::with_capacity(frames as usize);
    let mut gpu_samples = Vec::with_capacity(frames as usize);
    let mut taa_gpu_windows = Vec::with_capacity((frames / WINDOW_FRAMES) as usize);
    for frame in 0..frames {
        let started = std::time::Instant::now();
        eng.begin_frame();
        draw_glossy_detail_fixture(eng, (frame + 120) as f32 * camera_step);
        eng.end_frame();
        wall_samples.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        if (frame + 1).is_multiple_of(WINDOW_FRAMES) {
            let history = eng.profiler.frame_history();
            assert_eq!(
                history.len(),
                WINDOW_FRAMES as usize,
                "profiler did not retain one complete throughput window"
            );
            cpu_samples.extend(history.iter().map(|(cpu, _)| *cpu));
            gpu_samples.extend(history.iter().map(|(_, gpu)| *gpu));
            let taa_gpu_us = eng
                .profiler
                .snapshot()
                .into_iter()
                .find_map(|(label, _, gpu)| (label == "taa_pass").then_some(gpu?))
                .expect("throughput profile produced no TAA timestamp");
            taa_gpu_windows.push(taa_gpu_us);
        }
    }
    eng.profiler.set_enabled(false);

    let (wall_mean_us, wall_p50_us, wall_p95_us) = throughput_stats(wall_samples);
    let (cpu_mean_us, cpu_p50_us, cpu_p95_us) = throughput_stats(cpu_samples);
    let (gpu_mean_us, gpu_p50_us, gpu_p95_us) = throughput_stats(gpu_samples);
    let taa_gpu_mean_us = taa_gpu_windows.iter().sum::<f64>() / taa_gpu_windows.len() as f64;
    ReconstructionThroughputProfile {
        wall_mean_us,
        wall_p50_us,
        wall_p95_us,
        cpu_mean_us,
        cpu_p50_us,
        cpu_p95_us,
        gpu_mean_us,
        gpu_p50_us,
        gpu_p95_us,
        taa_gpu_mean_us,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    assert!(!values.is_empty(), "median requires at least one sample");
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn throughput_profile_json(profile: ReconstructionThroughputProfile) -> serde_json::Value {
    serde_json::json!({
        "wall_mean_us": profile.wall_mean_us,
        "wall_p50_us": profile.wall_p50_us,
        "wall_p95_us": profile.wall_p95_us,
        "cpu_mean_us": profile.cpu_mean_us,
        "cpu_p50_us": profile.cpu_p50_us,
        "cpu_p95_us": profile.cpu_p95_us,
        "gpu_mean_us": profile.gpu_mean_us,
        "gpu_p50_us": profile.gpu_p50_us,
        "gpu_p95_us": profile.gpu_p95_us,
        "taa_gpu_mean_us": profile.taa_gpu_mean_us,
    })
}

#[test]
#[ignore = "explicit long-running GPU profiling gate"]
fn profile_fractional_taa_reconstruction() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    install_glossy_detail_fixture(&mut eng);
    let frames = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES")
        .ok()
        .map(|value| value.parse::<u32>().expect("profile frame count"))
        .unwrap_or(1200);
    let render_scale = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_RENDER_SCALE")
        .ok()
        .map(|value| value.parse::<f32>().expect("profile render scale"))
        .unwrap_or(0.75);
    let taa_gpu_us = profile_taa_reconstruction(&mut eng, frames, render_scale);
    eprintln!(
        "fractional-reconstruction taa_gpu_us={taa_gpu_us:.3} \
         frames={frames} render_scale={render_scale:.2}"
    );
}

#[test]
#[ignore = "explicit long-running native-vs-fractional throughput gate"]
fn profile_fractional_taa_native_advantage() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    install_glossy_detail_fixture(&mut eng);
    let frames = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES")
        .ok()
        .map(|value| value.parse::<u32>().expect("profile frame count"))
        .unwrap_or(600);
    let pairs = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_PAIRS")
        .ok()
        .map(|value| value.parse::<u32>().expect("profile pair count"))
        .unwrap_or(3)
        .max(1);
    let minimum_advantage = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_MIN_ADVANTAGE")
        .ok()
        .map(|value| value.parse::<f64>().expect("minimum advantage"))
        .unwrap_or(0.05);
    let mut fractional = Vec::with_capacity(pairs as usize);
    let mut native = Vec::with_capacity(pairs as usize);
    for pair in 0..pairs {
        let order = if pair % 2 == 0 {
            [("fractional", 0.75), ("native", 1.0)]
        } else {
            [("native", 1.0), ("fractional", 0.75)]
        };
        for (label, scale) in order {
            let profile = profile_reconstruction_throughput(&mut eng, frames, scale);
            eprintln!(
                "fractional-native-throughput pair={pair} label={label} scale={scale:.2} \
                 wall_mean_us={:.3} wall_p50_us={:.3} wall_p95_us={:.3} \
                 cpu_mean_us={:.3} cpu_p50_us={:.3} cpu_p95_us={:.3} \
                 gpu_mean_us={:.3} gpu_p50_us={:.3} gpu_p95_us={:.3} \
                 taa_gpu_mean_us={:.3}",
                profile.wall_mean_us,
                profile.wall_p50_us,
                profile.wall_p95_us,
                profile.cpu_mean_us,
                profile.cpu_p50_us,
                profile.cpu_p95_us,
                profile.gpu_mean_us,
                profile.gpu_p50_us,
                profile.gpu_p95_us,
                profile.taa_gpu_mean_us,
            );
            if label == "fractional" {
                fractional.push(profile);
            } else {
                native.push(profile);
            }
        }
    }

    let fractional_wall = median(fractional.iter().map(|run| run.wall_mean_us).collect());
    let native_wall = median(native.iter().map(|run| run.wall_mean_us).collect());
    let fractional_gpu = median(fractional.iter().map(|run| run.gpu_mean_us).collect());
    let native_gpu = median(native.iter().map(|run| run.gpu_mean_us).collect());
    let wall_advantage = 1.0 - fractional_wall / native_wall;
    let gpu_advantage = 1.0 - fractional_gpu / native_gpu;
    let passed = wall_advantage >= minimum_advantage && gpu_advantage >= minimum_advantage;
    eprintln!(
        "fractional-native-throughput-summary pairs={pairs} frames_per_run={frames} \
         fractional_wall_median_us={fractional_wall:.3} native_wall_median_us={native_wall:.3} \
         wall_advantage={wall_advantage:.6} fractional_gpu_median_us={fractional_gpu:.3} \
         native_gpu_median_us={native_gpu:.3} gpu_advantage={gpu_advantage:.6} \
         minimum_advantage={minimum_advantage:.6}"
    );
    if let Ok(output) = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_OUT") {
        let output = std::path::PathBuf::from(output);
        std::fs::create_dir_all(&output).expect("create fractional throughput output");
        let adapter: serde_json::Value =
            serde_json::from_str(&eng.renderer.quality_adapter_json()).expect("adapter JSON");
        let runtime: serde_json::Value =
            serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
                .expect("runtime paths JSON");
        let camera_step = std::env::var("BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP")
            .ok()
            .map(|value| value.parse::<f64>().expect("profile camera step"))
            .unwrap_or(0.002);
        let result = serde_json::json!({
            "schema": "bloom-fractional-native-throughput-v1",
            "git_commit": git_commit(),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "adapter": adapter,
            "configuration": {
                "output_extent": [1600, 900],
                "fractional_render_scale": 0.75,
                "native_render_scale": 1.0,
                "warmup_frames_per_run": 120,
                "measured_frames_per_run": frames,
                "pairs": pairs,
                "camera_step": camera_step,
                "taa": true,
                "quality_preset": 4,
                "profiler_gpu_readback_serialized": true,
            },
            "limits": {
                "min_wall_advantage": minimum_advantage,
                "min_gpu_advantage": minimum_advantage,
            },
            "fractional_runs": fractional
                .iter()
                .copied()
                .map(throughput_profile_json)
                .collect::<Vec<_>>(),
            "native_runs": native
                .iter()
                .copied()
                .map(throughput_profile_json)
                .collect::<Vec<_>>(),
            "summary": {
                "fractional_wall_median_us": fractional_wall,
                "native_wall_median_us": native_wall,
                "wall_advantage": wall_advantage,
                "fractional_gpu_median_us": fractional_gpu,
                "native_gpu_median_us": native_gpu,
                "gpu_advantage": gpu_advantage,
            },
            "steady_state_resources": runtime["steady_state_resources"].clone(),
            "passed": passed,
        });
        std::fs::write(
            output.join("result.json"),
            serde_json::to_string_pretty(&result).expect("serialize throughput result") + "\n",
        )
        .expect("write fractional throughput result");
        let summary = format!(
            "# Fractional 0.75 vs native 1.0 throughput\n\n\
             - status: **{}**\n\
             - adapter: `{}` / `{}`\n\
             - pairs: `{pairs}` × `{frames}` measured frames after 120 warm-up frames\n\
             - end-to-end median: `{fractional_wall:.3} us` vs `{native_wall:.3} us` ({:.2}% advantage)\n\
             - timestamped GPU median: `{fractional_gpu:.3} us` vs `{native_gpu:.3} us` ({:.2}% advantage)\n\
             - required advantage: `{:.2}%`\n",
            if passed { "pass" } else { "fail" },
            result["adapter"]["name"].as_str().unwrap_or("unknown"),
            result["adapter"]["backend"].as_str().unwrap_or("unknown"),
            wall_advantage * 100.0,
            gpu_advantage * 100.0,
            minimum_advantage * 100.0,
        );
        std::fs::write(output.join("summary.md"), summary)
            .expect("write fractional throughput summary");
    }
    assert!(
        wall_advantage >= minimum_advantage,
        "fractional 0.75 reconstruction did not preserve its end-to-end performance advantage: \
         fractional={fractional_wall:.3}us native={native_wall:.3}us \
         advantage={wall_advantage:.3} required={minimum_advantage:.3}"
    );
    assert!(
        gpu_advantage >= minimum_advantage,
        "fractional 0.75 reconstruction did not preserve its GPU performance advantage: \
         fractional={fractional_gpu:.3}us native={native_gpu:.3}us \
         advantage={gpu_advantage:.3} required={minimum_advantage:.3}"
    );
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
        let taa_gpu_us = profile_taa_reconstruction(&mut eng, frames, 0.75);
        eprintln!("fractional-reconstruction taa_gpu_us={taa_gpu_us:.3} frames={frames}");
    }
}

#[test]
fn half_scale_glossy_temporal_reconstruction_tracks_supersampled_reference() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    install_glossy_detail_fixture(&mut eng);

    eng.renderer.resize(W * 2, H * 2, W * 2, H * 2);
    let supersampled = capture_glossy_detail(&mut eng, false, 1.0, 2, 0.0);
    let reference = downsample_box_2x(&supersampled, W * 2, H * 2);

    eng.renderer.resize(W, H, W, H);
    let no_taa = capture_glossy_detail(&mut eng, false, 0.5, 2, 0.0);
    let temporal = capture_glossy_detail(&mut eng, true, 0.5, 24, 0.0);
    let no_taa_metrics = calculate_diff_metrics(&reference, &no_taa, W, H);
    let temporal_metrics = calculate_diff_metrics(&reference, &temporal, W, H);
    let reference_detail = detail_energy(&reference);
    let no_taa_detail = detail_energy(&no_taa);
    let temporal_detail = detail_energy(&temporal);
    eprintln!(
        "half-glossy reference={reference_detail:.4} no_taa={no_taa_detail:.4} \
         temporal={temporal_detail:.4} no_taa_metrics={no_taa_metrics:?} \
         temporal_metrics={temporal_metrics:?}"
    );

    assert!(
        temporal_metrics.ssim >= no_taa_metrics.ssim
            && temporal_metrics.mean_rgb <= no_taa_metrics.mean_rgb,
        "half-scale temporal reconstruction did not improve the aliased single frame: \
         no_taa={no_taa_metrics:?}, temporal={temporal_metrics:?}"
    );
    assert!(
        temporal_metrics.ssim >= 0.922
            && temporal_metrics.mean_rgb <= 2.13
            && temporal_metrics.rmse_luminance <= 0.0249
            && temporal_metrics.mean_oklab_delta <= 0.0083
            && temporal_metrics.mean_edge_delta <= 0.0149,
        "half-scale glossy reconstruction exceeded its qualified reference envelope: \
         {temporal_metrics:?}"
    );
    assert!(
        temporal_detail >= reference_detail * 0.54,
        "half-scale reconstruction erased too much authored glossy detail: \
         reference={reference_detail:.4}, temporal={temporal_detail:.4}"
    );
    if let Ok(frames) = std::env::var("BLOOM_PROFILE_HALF_TAA_FRAMES") {
        let frames = frames.parse::<u32>().expect("profile frame count");
        let taa_gpu_us = profile_taa_reconstruction(&mut eng, frames, 0.5);
        eprintln!("half-reconstruction taa_gpu_us={taa_gpu_us:.3} frames={frames}");
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

fn thin_feature_slow_pan_metrics() -> Option<(f64, f64, f64, f64, Vec<f64>)> {
    const FRAMES: usize = 16;
    const CAMERA_STEP: f32 = 0.002;

    let mut eng = try_engine()?;
    let mut checker = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let value = if (x + y) & 1 == 0 { 245 } else { 8 };
            checker.extend_from_slice(&[value, value, value, 255]);
        }
    }
    let texture = eng.renderer.register_texture_no_mips(64, 64, &checker);
    let vertices = [
        ([-3.0, -1.7, 0.0], [0.0, 2.0]),
        ([3.0, -1.7, 0.0], [4.0, 2.0]),
        ([3.0, 1.7, 0.0], [4.0, 0.0]),
        ([-3.0, 1.7, 0.0], [0.0, 0.0]),
    ]
    .into_iter()
    .map(|(position, uv)| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [1.0; 4],
        uv,
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    })
    .collect();
    let receiver = eng.scene.create_node();
    eng.scene
        .update_geometry(receiver, vertices, vec![0, 1, 2, 0, 2, 3]);
    eng.scene.set_material_texture(receiver, texture);
    eng.scene.set_material_pbr(receiver, 1.0, 0.0);

    let draw = |eng: &mut EngineState, camera_x: f32| {
        let r = &mut eng.renderer;
        r.set_clear_color(4.0, 5.0, 8.0, 255.0);
        r.begin_mode_3d(
            camera_x,
            0.0,
            5.2,
            camera_x * 0.12,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            42.0,
            0.0,
        );
        r.set_ambient_light(255.0, 255.0, 255.0, 1.0);
    };

    eng.renderer.resize(W * 2, H * 2, W * 2, H * 2);
    configure_glossy_detail_capture(&mut eng.renderer, false, 1.0);
    let references = (0..FRAMES)
        .map(|frame| {
            let camera_x = frame as f32 * CAMERA_STEP;
            let supersampled = render(&mut eng, 1, |eng| draw(eng, camera_x)).2;
            downsample_box_2x(&supersampled, W * 2, H * 2)
        })
        .collect::<Vec<_>>();

    eng.renderer.resize(W, H, W, H);
    configure_glossy_detail_capture(&mut eng.renderer, true, 0.75);
    let _ = render(&mut eng, 32, |eng| draw(eng, 0.0));
    let candidates = (0..FRAMES)
        .map(|frame| {
            let camera_x = frame as f32 * CAMERA_STEP;
            render(&mut eng, 1, |eng| draw(eng, camera_x)).2
        })
        .collect::<Vec<_>>();

    let metrics = references
        .iter()
        .zip(&candidates)
        .map(|(reference, candidate)| calculate_diff_metrics(reference, candidate, W, H))
        .collect::<Vec<_>>();
    let mean_rgb = metrics.iter().map(|metrics| metrics.mean_rgb).sum::<f64>() / FRAMES as f64;
    let mean_ssim = metrics.iter().map(|metrics| metrics.ssim).sum::<f64>() / FRAMES as f64;
    let minimum_ssim = metrics
        .iter()
        .map(|metrics| metrics.ssim)
        .fold(1.0f64, f64::min);
    let derivative_errors = references
        .windows(2)
        .zip(candidates.windows(2))
        .map(|(reference, candidate)| {
            temporal_derivative_error(&reference[0], &reference[1], &candidate[0], &candidate[1])
        })
        .collect::<Vec<_>>();
    let derivative_error = derivative_errors.iter().sum::<f64>() / (FRAMES - 1) as f64;
    eprintln!(
        "thin-feature-slow-pan mean_rgb={mean_rgb:.6} mean_ssim={mean_ssim:.6} \
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

fn crop_rgba(
    rgba: &[u8],
    width: u32,
    x0: u32,
    y0: u32,
    crop_width: u32,
    crop_height: u32,
) -> Vec<u8> {
    let mut crop = Vec::with_capacity((crop_width * crop_height * 4) as usize);
    for y in y0..y0 + crop_height {
        let start = ((y * width + x0) * 4) as usize;
        let end = start + (crop_width * 4) as usize;
        crop.extend_from_slice(&rgba[start..end]);
    }
    crop
}

fn coplanar_material_boundary_metrics() -> Option<(f64, f64, f64, f64)> {
    const FRAMES: usize = 12;
    const CAMERA_STEP: f32 = 0.03;
    const CROP_X: u32 = 112;
    const CROP_Y: u32 = 48;
    const CROP_W: u32 = 32;
    const CROP_H: u32 = 160;

    let mut eng = try_engine()?;
    let mut surface_detail = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let value = if ((x / 2) + (y / 2)) & 1 == 0 {
                210
            } else {
                150
            };
            surface_detail.extend_from_slice(&[value, value, value, 255]);
        }
    }
    let surface_detail = eng
        .renderer
        .register_texture_no_mips(64, 64, &surface_detail);
    let make_panel = |min_x: f32, max_x: f32| {
        [
            ([min_x, -2.2, 0.0], [0.0, 1.0]),
            ([max_x, -2.2, 0.0], [1.0, 1.0]),
            ([max_x, 2.2, 0.0], [1.0, 0.0]),
            ([min_x, 2.2, 0.0], [0.0, 0.0]),
        ]
        .into_iter()
        .map(|(position, uv)| Vertex3D {
            position,
            normal: [0.0, 0.0, 1.0],
            color: [1.0; 4],
            uv,
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [1.0, 0.0, 0.0, 1.0],
        })
        .collect::<Vec<_>>()
    };
    let glossy = eng.scene.create_node();
    eng.scene
        .update_geometry(glossy, make_panel(-3.2, 0.0), vec![0, 1, 2, 0, 2, 3]);
    eng.scene.set_material_color(glossy, 0.42, 0.48, 0.56, 1.0);
    eng.scene.set_material_texture(glossy, surface_detail);
    eng.scene.set_material_pbr(glossy, 0.08, 0.0);
    let rough = eng.scene.create_node();
    eng.scene
        .update_geometry(rough, make_panel(0.0, 3.2), vec![0, 1, 2, 0, 2, 3]);
    eng.scene.set_material_color(rough, 0.42, 0.48, 0.56, 1.0);
    eng.scene.set_material_texture(rough, surface_detail);
    eng.scene.set_material_pbr(rough, 0.92, 0.0);

    let draw = |eng: &mut EngineState, camera_x: f32| {
        let r = &mut eng.renderer;
        r.set_clear_color(4.0, 5.0, 8.0, 255.0);
        r.begin_mode_3d(
            camera_x,
            0.0,
            5.0,
            camera_x * 0.10,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            42.0,
            0.0,
        );
        r.set_ambient_light(175.0, 185.0, 205.0, 0.16);
        r.add_directional_light(-0.55, -0.2, -1.0, 1.0, 0.94, 0.82, 3.8);
        r.add_point_light(-0.6, 0.8, 2.2, 8.0, 0.65, 0.78, 1.0, 10.0);
    };
    let configure = |renderer: &mut Renderer, taa: bool, render_scale: f32| {
        renderer.apply_quality_preset(4);
        renderer.set_render_scale(render_scale);
        configure_reconstruction_scene(renderer);
        renderer.set_taa_enabled(taa);
        renderer.set_sharpen_strength(0.0);
        renderer.set_auto_exposure(false);
        renderer.set_manual_exposure(1.0);
        renderer.reset_temporal_history();
    };

    eng.renderer.resize(W * 2, H * 2, W * 2, H * 2);
    configure(&mut eng.renderer, false, 1.0);
    let references = (0..FRAMES)
        .map(|frame| {
            let camera_x = frame as f32 * CAMERA_STEP;
            let supersampled = render(&mut eng, 1, |eng| draw(eng, camera_x)).2;
            let reference = downsample_box_2x(&supersampled, W * 2, H * 2);
            crop_rgba(&reference, W, CROP_X, CROP_Y, CROP_W, CROP_H)
        })
        .collect::<Vec<_>>();

    eng.renderer.resize(W, H, W, H);
    configure(&mut eng.renderer, true, 0.75);
    let _ = render(&mut eng, 24, |eng| draw(eng, 0.0));
    let candidates = (0..FRAMES)
        .map(|frame| {
            let camera_x = frame as f32 * CAMERA_STEP;
            let frame = render(&mut eng, 1, |eng| draw(eng, camera_x)).2;
            crop_rgba(&frame, W, CROP_X, CROP_Y, CROP_W, CROP_H)
        })
        .collect::<Vec<_>>();

    let metrics = references
        .iter()
        .zip(&candidates)
        .map(|(reference, candidate)| calculate_diff_metrics(reference, candidate, CROP_W, CROP_H))
        .collect::<Vec<_>>();
    let mean_rgb = metrics.iter().map(|metrics| metrics.mean_rgb).sum::<f64>() / FRAMES as f64;
    let mean_ssim = metrics.iter().map(|metrics| metrics.ssim).sum::<f64>() / FRAMES as f64;
    let minimum_ssim = metrics
        .iter()
        .map(|metrics| metrics.ssim)
        .fold(1.0f64, f64::min);
    let derivative_error = references
        .windows(2)
        .zip(candidates.windows(2))
        .map(|(reference, candidate)| {
            temporal_derivative_error(&reference[0], &reference[1], &candidate[0], &candidate[1])
        })
        .sum::<f64>()
        / (FRAMES - 1) as f64;
    eprintln!(
        "coplanar-material-boundary mean_rgb={mean_rgb:.6} mean_ssim={mean_ssim:.6} \
         minimum_ssim={minimum_ssim:.6} derivative_error={derivative_error:.6} \
         frame_metrics={metrics:?}"
    );
    Some((mean_rgb, mean_ssim, minimum_ssim, derivative_error))
}

#[test]
fn fractional_coplanar_material_boundary_tracks_supersampled_motion() {
    let Some((mean_rgb, mean_ssim, minimum_ssim, derivative_error)) =
        coplanar_material_boundary_metrics()
    else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    // The two panels are exactly coplanar and share their base texture/color,
    // but sit at opposite ends of the perceptual-roughness range. Depth and
    // broad color alone therefore cannot identify the boundary. Keep both
    // native-reference fidelity and temporal response bounded so a future
    // discriminator cannot earn a lower still error by retaining stale detail.
    // The bounds also preserve the moderate-motion detail refresh that keeps
    // a valid high-frequency history from accumulating source-phase lag.
    assert!(
        mean_rgb <= 1.42 && mean_ssim >= 0.985 && minimum_ssim >= 0.97,
        "fractional coplanar material boundary diverged from supersampled motion: \
         mean_rgb={mean_rgb:.6}, mean_ssim={mean_ssim:.6}, \
         minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        derivative_error <= 0.44,
        "fractional coplanar material boundary added excessive temporal variation: \
         derivative_error={derivative_error:.6}"
    );
}

#[test]
fn fractional_thin_features_bound_motion_error_without_reference_lag() {
    let Some((mean_rgb, mean_ssim, minimum_ssim, derivative_error, _)) =
        thin_feature_slow_pan_metrics()
    else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    // This fixture deliberately presents an unfiltered one-texel checker at
    // fractional resolution while the camera moves. It is much harsher than
    // the authored glossy corpus: the absolute similarity score is therefore
    // a non-regression envelope, not a claim that every source texel is
    // reconstructable. Gate reference fidelity and temporal variation
    // independently. A persistent thin-feature lock can lower derivative
    // error by retaining stale samples; the RGB and SSIM bounds catch that
    // ghosting/lag trade instead of accepting it as improved stability.
    assert!(
        mean_rgb <= 11.00 && mean_ssim >= 0.660 && minimum_ssim >= 0.600,
        "fractional thin-feature motion lagged its supersampled reference: \
         mean_rgb={mean_rgb:.6}, mean_ssim={mean_ssim:.6}, \
         minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        derivative_error <= 1.05,
        "fractional thin-feature motion added excessive temporal variation: \
         derivative_error={derivative_error:.6}"
    );
}

#[test]
fn fractional_glossy_slow_pan_tracks_supersampled_motion() {
    let Some((mean_rgb, mean_ssim, minimum_ssim, derivative_error, _)) =
        glossy_slow_pan_metrics(0.75)
    else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    assert!(
        mean_rgb <= 1.09 && mean_ssim >= 0.9786 && minimum_ssim >= 0.9740,
        "fractional glossy slow pan diverged from supersampled motion: \
         mean_rgb={mean_rgb:.6}, mean_ssim={mean_ssim:.6}, \
         minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        derivative_error <= 0.122,
        "fractional glossy slow pan added excessive temporal variation: \
         derivative_error={derivative_error:.6}"
    );
}

#[test]
fn half_scale_glossy_slow_pan_tracks_supersampled_motion() {
    let Some((mean_rgb, mean_ssim, minimum_ssim, derivative_error, _)) =
        glossy_slow_pan_metrics(0.5)
    else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    assert!(
        mean_rgb <= 2.10 && mean_ssim >= 0.925 && minimum_ssim >= 0.913,
        "half-scale glossy slow pan diverged from supersampled motion: \
         mean_rgb={mean_rgb:.6}, mean_ssim={mean_ssim:.6}, \
         minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        derivative_error <= 0.20,
        "half-scale glossy slow pan added excessive temporal variation: \
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
    let legacy_paths: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("half-scale reconstruction telemetry is valid JSON");
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
        Some("approximate-separable-lanczos2")
    );
    assert_eq!(reconstruction["source_filter_samples"].as_u64(), Some(9));
    assert_eq!(
        reconstruction["statistics_filter"].as_str(),
        Some("reused-lanczos-cross")
    );
    assert_eq!(
        reconstruction["statistics_filter_samples"].as_u64(),
        Some(5)
    );
    assert_eq!(
        reconstruction["statistics_additional_samples"].as_u64(),
        Some(0)
    );
    assert_eq!(reconstruction["composed_source_samples"].as_u64(), Some(9));
    assert_eq!(
        reconstruction["bootstrap_source_filter"].as_str(),
        Some("exact-separable-catmull-rom")
    );
    assert_eq!(
        reconstruction["bootstrap_source_filter_samples"].as_u64(),
        Some(9)
    );
    assert_eq!(
        reconstruction["bootstrap_statistics_additional_samples"].as_u64(),
        Some(5)
    );
    assert_eq!(
        reconstruction["bootstrap_composed_source_samples"].as_u64(),
        Some(14)
    );
    assert_eq!(
        reconstruction["history_filter"].as_str(),
        Some("camera-motion-phase-compressed-linear")
    );
    assert_eq!(reconstruction["history_filter_samples"].as_u64(), Some(1));
    let half_reconstruction = &legacy_paths["temporal_reconstruction"];
    assert_eq!(
        half_reconstruction["source_filter"].as_str(),
        Some("approximate-radial-lanczos2")
    );
    assert_eq!(
        half_reconstruction["source_filter_samples"].as_u64(),
        Some(5)
    );
    assert_eq!(
        half_reconstruction["statistics_additional_samples"].as_u64(),
        Some(4)
    );
    assert_eq!(
        half_reconstruction["composed_source_samples"].as_u64(),
        Some(9)
    );
    let native_reconstruction = &ultra_paths["temporal_reconstruction"];
    assert_eq!(
        native_reconstruction["source_filter"].as_str(),
        Some("exact-separable-catmull-rom")
    );
    assert_eq!(
        native_reconstruction["statistics_additional_samples"].as_u64(),
        Some(5)
    );
    assert_eq!(
        native_reconstruction["composed_source_samples"].as_u64(),
        Some(14)
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_detail_strength"].as_f64(),
        Some(0.2)
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_detail_clamp"].as_f64(),
        Some(0.08)
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_detail_policy"].as_str(),
        Some("alpha-weighted-high-scale")
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_detail_alpha_gain"].as_f64(),
        Some(3.0)
    );
    assert_eq!(
        reconstruction["stationary_reconstruction_detail_min_scale"].as_f64(),
        Some(0.75)
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
        reconstruction["camera_motion_source_phase"].as_str(),
        Some("jitter-aligned-input-depth")
    );
    assert_eq!(
        reconstruction["camera_motion_source_phase_min_scale"].as_f64(),
        Some(0.75)
    );
    assert_eq!(
        reconstruction["camera_motion_source_phase_max_scale"].as_f64(),
        Some(0.95)
    );
    assert_eq!(
        reconstruction["camera_motion_source_phase_active"].as_bool(),
        Some(false)
    );
    assert_eq!(
        reconstruction["camera_motion_reconstruction_detail_strength"].as_f64(),
        Some(0.02)
    );
    assert_eq!(
        reconstruction["camera_motion_reconstruction_detail_policy"].as_str(),
        Some("detail-lock-weighted")
    );
    assert_eq!(
        reconstruction["camera_motion_reconstruction_detail_locked_strength"].as_f64(),
        Some(0.06)
    );
    assert_eq!(
        reconstruction["camera_motion_reconstruction_detail_classifier"].as_str(),
        Some("fractional-luma-variance-lock")
    );
    assert_eq!(
        reconstruction["camera_motion_reconstruction_additional_samples"].as_u64(),
        Some(0)
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
