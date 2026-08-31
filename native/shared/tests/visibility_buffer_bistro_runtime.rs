//! Environment-gated full-Bistro visibility-buffer qualification for #27.
//!
//! Run with:
//! BLOOM_VISIBILITY_BISTRO_SCENE=/path/to/bistrox.gltf \
//! cargo test --release --test visibility_buffer_bistro_runtime \
//!   --features models3d -- --nocapture
//!
//! Runtime and shadow modes are process-global startup selections. The parent
//! therefore launches one child for every forward/visibility and CSM/VSM pair,
//! drives the same warmup and camera-motion corpus, and compares final output.

use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const START_X: f32 = -3.2720;
const START_Y: f32 = 1.544;
const START_Z: f32 = 7.2358;
const START_YAW: f32 = -0.344;
const END_X: f32 = START_X + 3.748_170_4;
const END_Z: f32 = START_Z + 4.685_213;
const MOTION_STEPS: u32 = 30;
const MOTION_SAMPLE_INTERVAL: u32 = 5;
const RAY_SCENE_SETTLE_FRAMES: usize = 108;
const EXPECTED_PLACEMENTS: usize = 2_909;

const CHILD_PROFILE: &str = "BLOOM_VISIBILITY_BISTRO_CHILD_PROFILE";
const DIAGNOSTICS: &str = "BLOOM_VISIBILITY_BISTRO_DIAGNOSTICS";

#[derive(Clone, Copy, Debug)]
struct ImageMetrics {
    mean_rgb: f64,
    root_mean_square_rgb: f64,
    percentile_99_rgb: u8,
    max_rgb: u8,
    ssim: f64,
}

#[test]
fn full_bistro_visibility_matches_forward_with_screen_effects_and_shadows() {
    let Some(scene_path) = std::env::var_os("BLOOM_VISIBILITY_BISTRO_SCENE").map(PathBuf::from)
    else {
        eprintln!("skip: BLOOM_VISIBILITY_BISTRO_SCENE is not set");
        return;
    };
    let directory = std::env::var_os(DIAGNOSTICS)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("bloom-visibility-bistro-{}", std::process::id()))
        });

    if let Some(profile) = std::env::var_os(CHILD_PROFILE) {
        std::fs::create_dir_all(&directory).expect("create Bistro visibility child output");
        run_child_capture(&scene_path, &directory, &profile.to_string_lossy());
        return;
    }

    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("create Bistro visibility diagnostics");
    let executable = std::env::current_exe().expect("resolve Bistro visibility test executable");
    for shadow in ["csm", "vsm"] {
        for mode in ["off", "shade"] {
            let profile = format!("{shadow}-{mode}");
            let mut command = std::process::Command::new(&executable);
            command
                .args([
                    "--exact",
                    "full_bistro_visibility_matches_forward_with_screen_effects_and_shadows",
                    "--nocapture",
                ])
                .env(CHILD_PROFILE, &profile)
                .env(DIAGNOSTICS, &directory)
                .env("BLOOM_VISIBILITY_BUFFER", mode)
                .env("BLOOM_SKIP_SKY", "1");
            if shadow == "vsm" {
                command.env("BLOOM_VSM", "1");
            } else {
                command.env_remove("BLOOM_VSM");
            }
            let status = command
                .status()
                .unwrap_or_else(|error| panic!("launch {profile} Bistro child: {error}"));
            assert!(status.success(), "{profile} Bistro child failed");
        }
    }

    let mut all_metrics = Vec::new();
    for shadow in ["csm", "vsm"] {
        validate_runtime_reports(&directory, shadow, "off");
        validate_runtime_reports(&directory, shadow, "shade");
        for sample in 0..=MOTION_STEPS / MOTION_SAMPLE_INTERVAL {
            let step = sample * MOTION_SAMPLE_INTERVAL;
            let forward = load_rgba(&capture_path(&directory, shadow, "off", step));
            let visibility = load_rgba(&capture_path(&directory, shadow, "shade", step));
            let metrics = image_metrics(&forward, &visibility, WIDTH, HEIGHT);
            eprintln!(
                "visibility-bistro shadow={shadow} step={step:02} \
                 mean={:.9} rms={:.9} p99={} max={} ssim={:.9}",
                metrics.mean_rgb,
                metrics.root_mean_square_rgb,
                metrics.percentile_99_rgb,
                metrics.max_rgb,
                metrics.ssim,
            );
            let within_gate = if shadow == "csm" {
                metrics.ssim >= 0.999
                    && metrics.mean_rgb <= 0.1
                    && metrics.root_mean_square_rgb <= 0.5
                    && metrics.percentile_99_rgb <= 2
                    && metrics.max_rgb <= 32
            } else {
                // Separate processes can populate a newly exposed VSM page on
                // adjacent motion frames. Bound that high-contrast, one-pixel
                // silhouette variance without weakening the deterministic CSM
                // oracle or permitting a broad material/geometry mismatch.
                metrics.ssim >= 0.99
                    && metrics.mean_rgb <= 0.5
                    && metrics.root_mean_square_rgb <= 5.0
                    && metrics.percentile_99_rgb <= 16
                    && metrics.max_rgb <= 96
            };
            assert!(
                within_gate,
                "visibility Bistro diverged from forward at {shadow} step {step}: {metrics:?}"
            );
            all_metrics.push((shadow, step, metrics));
        }
    }
    std::fs::write(directory.join("metrics.txt"), format!("{all_metrics:#?}\n"))
        .expect("write Bistro visibility metrics");

    if std::env::var_os("BLOOM_KEEP_VISIBILITY_BISTRO_DIAGNOSTICS").is_some() {
        eprintln!(
            "kept full-Bistro visibility diagnostics at {}",
            directory.display()
        );
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

fn run_child_capture(scene_path: &Path, directory: &Path, profile: &str) {
    let (shadow, mode) = profile
        .split_once('-')
        .unwrap_or_else(|| panic!("invalid Bistro visibility child profile {profile}"));
    assert!(matches!(shadow, "csm" | "vsm"));
    assert!(matches!(mode, "off" | "shade"));

    let mut engine =
        bloom_shared::attach::attach_headless_engine(wgpu::Backends::PRIMARY, WIDTH, HEIGHT)
            .unwrap_or_else(|error| panic!("{profile} Bistro device setup failed: {error}"));
    configure(&mut engine.renderer);
    // Occlusion consumes asynchronous previous-frame readback. Different A/B
    // pass costs can make the same result become CPU-visible one frame apart,
    // changing draw admission before shading is compared. Occlusion owns a
    // separate qualification; this oracle isolates the visibility path.
    engine.renderer.occlusion.set_enabled(false);
    let model_handle = load_model(&mut engine, scene_path);
    attach_model_placements(&mut engine, model_handle);

    // Hardware SSGI admits one retained BLAS per frame. Comparing partially
    // admitted ray scenes produces large, process-timing-dependent interior
    // lighting deltas that have nothing to do with visibility shading.
    let admission_frames = engine.scene.pending_blas_builds.len();
    let minimum_warmup_frames = admission_frames + RAY_SCENE_SETTLE_FRAMES;
    let warmup_frames = std::env::var("BLOOM_VISIBILITY_BISTRO_WARMUP_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(minimum_warmup_frames)
        .max(minimum_warmup_frames);
    eprintln!(
        "visibility-bistro profile={profile} placements={EXPECTED_PLACEMENTS} \
         admission_frames={admission_frames} warmup_frames={warmup_frames}"
    );
    for _ in 0..warmup_frames {
        retained_frame(&mut engine, START_X, START_Z, START_YAW, false);
    }
    let start = retained_frame(&mut engine, START_X, START_Z, START_YAW, true)
        .expect("capture full-Bistro start frame");
    save_rgba(&capture_path(directory, shadow, mode, 0), &start);

    for step in 1..=MOTION_STEPS {
        let t = step as f32 / MOTION_STEPS as f32;
        let x = START_X + (END_X - START_X) * t;
        let z = START_Z + (END_Z - START_Z) * t;
        let capture = step % MOTION_SAMPLE_INTERVAL == 0;
        if std::env::var("BLOOM_VISIBILITY_BISTRO_COMPONENT_CAPTURE_STEP")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            == Some(step)
        {
            engine.renderer.pending_quality_capture_dir = Some(
                directory
                    .join(format!("{profile}-components-{step:02}"))
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        if let Some(frame) = retained_frame(&mut engine, x, z, START_YAW, capture) {
            save_rgba(&capture_path(directory, shadow, mode, step), &frame);
        }
    }

    std::fs::write(
        report_path(directory, shadow, mode, "capabilities"),
        engine.renderer.renderer_capability_report_json(),
    )
    .expect("write Bistro visibility capability report");
    std::fs::write(
        report_path(directory, shadow, mode, "runtime-paths"),
        engine.renderer.quality_runtime_paths_json(),
    )
    .expect("write Bistro visibility runtime paths");
}

fn validate_runtime_reports(directory: &Path, shadow: &str, mode: &str) {
    let capabilities: serde_json::Value = serde_json::from_slice(
        &std::fs::read(report_path(directory, shadow, mode, "capabilities"))
            .expect("read Bistro visibility capability report"),
    )
    .expect("Bistro visibility capability report is JSON");
    let paths: serde_json::Value = serde_json::from_slice(
        &std::fs::read(report_path(directory, shadow, mode, "runtime-paths"))
            .expect("read Bistro visibility runtime paths"),
    )
    .expect("Bistro visibility runtime paths are JSON");

    assert_eq!(paths["ray_scene_preparation"], "ssgi");
    assert_eq!(paths["temporal_history"]["ssr_valid"], true);
    assert_eq!(paths["temporal_history"]["ssgi_probe_valid"], true);
    assert_eq!(paths["temporal_history"]["taa_valid"], true);
    assert_eq!(paths["temporal_reconstruction"]["enabled"], true);
    let trace_backend = paths["ssgi_trace_backend"]
        .as_str()
        .expect("SSGI trace backend is reported");
    assert!(
        matches!(trace_backend, "hw-ray-query" | "software-fallback"),
        "SSGI did not execute a settled trace backend: {trace_backend}"
    );

    let visibility = &capabilities["runtime_support"]["gpu_driven"]["visibility_buffer_runtime"];
    assert_eq!(visibility["requested_mode"], mode);
    if mode == "shade" {
        assert_eq!(visibility["enabled"], true);
        assert_eq!(visibility["pbr_shading"], true);
        assert_eq!(visibility["forward_authoritative"], false);
        assert!(visibility["eligible_draws"]
            .as_u64()
            .is_some_and(|draws| draws >= 1_000));
        assert!(visibility["compatibility_draws"]
            .as_u64()
            .is_some_and(|draws| draws > 0));
    } else {
        assert_eq!(visibility["enabled"], false);
        assert_eq!(visibility["forward_authoritative"], true);
    }

    let virtual_shadows = &capabilities["runtime_support"]["virtual_shadows"];
    if shadow == "vsm" {
        assert_eq!(virtual_shadows["requested"], true);
        assert_eq!(virtual_shadows["enabled"], true);
        assert_eq!(virtual_shadows["active"], true);
        // A settled VSM cache commonly renders zero new pages on the final
        // captured frame. Residency plus receiver demand/cache hits prove the
        // virtual path is active without conflating stability with inactivity.
        assert!(virtual_shadows["resident"]
            .as_u64()
            .is_some_and(|pages| pages > 0));
        assert!(virtual_shadows["requested_pages"]
            .as_u64()
            .is_some_and(|pages| pages > 0));
        assert!(virtual_shadows["cache_hits"]
            .as_u64()
            .is_some_and(|pages| pages > 0));
    } else {
        assert_eq!(virtual_shadows["requested"], false);
        assert_eq!(virtual_shadows["enabled"], false);
        assert_eq!(virtual_shadows["active"], false);
    }
}

fn configure(renderer: &mut Renderer) {
    renderer.apply_quality_preset(4);
    renderer.set_render_scale(1.0);
    renderer.set_taa_enabled(effect_enabled("BLOOM_VISIBILITY_BISTRO_TAA"));
    renderer.set_ssao_enabled(effect_enabled("BLOOM_VISIBILITY_BISTRO_SSAO"));
    renderer.set_ssr_enabled(effect_enabled("BLOOM_VISIBILITY_BISTRO_SSR"));
    renderer.set_ssgi_enabled(effect_enabled("BLOOM_VISIBILITY_BISTRO_SSGI"));
    renderer.set_ssgi_radius(2.3);
    renderer.set_ssgi_intensity(1.1);
    renderer.set_bloom_enabled(effect_enabled("BLOOM_VISIBILITY_BISTRO_BLOOM"));
    renderer.set_motion_blur_enabled(false);
    renderer.set_sss_enabled(false);
    renderer.set_sharpen_strength(0.0);
    renderer.set_auto_exposure(false);
    renderer.set_manual_exposure(1.0);
    renderer.set_shadows_enabled(true);
}

fn effect_enabled(variable: &str) -> bool {
    std::env::var_os(variable).is_none_or(|value| value != "0")
}

fn load_model(engine: &mut EngineState, scene_path: &Path) -> f64 {
    let bytes = std::fs::read(scene_path).expect("read full Bistro glTF");
    let handle = engine.models.load_model_with_textures_from_source_path(
        &bytes,
        scene_path,
        &mut engine.renderer,
    );
    assert!(handle > 0.0, "load full Bistro glTF");
    let model = engine.models.get(handle).expect("loaded full Bistro model");
    assert_eq!(
        model.meshes.len(),
        EXPECTED_PLACEMENTS,
        "unexpected full Bistro placement corpus"
    );
    handle
}

fn attach_model_placements(engine: &mut EngineState, model_handle: f64) {
    let placements = {
        let model = engine
            .models
            .get(model_handle)
            .expect("loaded full Bistro model");
        model
            .meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| {
                (
                    Arc::clone(mesh),
                    model.mesh_transform(index),
                    model.mesh_cast_shadow(index),
                )
            })
            .collect::<Vec<_>>()
    };
    for (mesh, source_transform, cast_shadow) in placements {
        let node = engine.scene.create_node();
        let mut transmission = mesh.transmission;
        let axis_length = |column: usize| {
            let x = source_transform[column][0];
            let y = source_transform[column][1];
            let z = source_transform[column][2];
            (x * x + y * y + z * z).sqrt()
        };
        transmission.baked_thickness_scale *=
            (axis_length(0) + axis_length(1) + axis_length(2)) / 3.0;

        engine
            .scene
            .update_shared_model_geometry(node, Arc::clone(&mesh), source_transform);
        engine.scene.set_cast_shadow(node, cast_shadow);
        if let Some(texture) = mesh.texture_idx {
            engine.scene.set_material_texture(node, texture);
        }
        if let Some(texture) = mesh.normal_texture_idx {
            engine.scene.set_material_normal_texture(node, texture);
        }
        if let Some(texture) = mesh.metallic_roughness_texture_idx {
            engine
                .scene
                .set_material_metallic_roughness_texture(node, texture);
        }
        engine
            .scene
            .set_material_specular_glossiness_factor(node, mesh.specular_glossiness_factor);
        if let Some(texture) = mesh.emissive_texture_idx {
            engine.scene.set_material_emissive_texture(node, texture);
        }
        engine.scene.set_material_emissive_factor(
            node,
            mesh.emissive_factor[0],
            mesh.emissive_factor[1],
            mesh.emissive_factor[2],
        );
        engine
            .scene
            .set_material_pbr(node, mesh.roughness_factor, mesh.metallic_factor);
        engine.scene.set_material_gltf_alpha(
            node,
            mesh.alpha_mode,
            mesh.alpha_cutoff,
            mesh.double_sided,
        );
        engine
            .scene
            .set_material_alpha_coverage_mips(node, mesh.alpha_coverage_mips);
        engine.scene.set_material_transmission(node, transmission);
        engine
            .scene
            .set_material_layered_pbr(node, mesh.layered_pbr);
    }
}

fn retained_frame(
    engine: &mut EngineState,
    x: f32,
    z: f32,
    yaw: f32,
    screenshot: bool,
) -> Option<Vec<u8>> {
    engine.begin_frame();
    begin_camera(&mut engine.renderer, x, z, yaw);
    engine.renderer.screenshot_requested = screenshot;
    engine.end_frame();
    screenshot.then(|| take_screenshot(&mut engine.renderer))
}

fn begin_camera(renderer: &mut Renderer, x: f32, z: f32, yaw: f32) {
    let forward_x = -yaw.sin();
    let forward_z = -yaw.cos();
    renderer.set_clear_color(5.0, 7.0, 12.0, 255.0);
    renderer.begin_mode_3d(
        x,
        START_Y,
        z,
        x + forward_x * 100.0,
        START_Y,
        z + forward_z * 100.0,
        0.0,
        1.0,
        0.0,
        60.0,
        0.0,
    );
    renderer.set_ambient_light(255.0, 245.0, 232.0, 0.06);
    renderer.set_directional_light(0.59732, 0.79653, -0.0935387, 255.0, 212.0, 177.0, 2.60);
}

fn take_screenshot(renderer: &mut Renderer) -> Vec<u8> {
    let (_, _, mut rgba) = renderer
        .screenshot_data
        .take()
        .expect("renderer produced a screenshot");
    if matches!(
        renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    rgba
}

fn capture_path(directory: &Path, shadow: &str, mode: &str, step: u32) -> PathBuf {
    directory.join(format!("{shadow}-{mode}-{step:02}.png"))
}

fn report_path(directory: &Path, shadow: &str, mode: &str, report: &str) -> PathBuf {
    directory.join(format!("{shadow}-{mode}-{report}.json"))
}

fn save_rgba(path: &Path, pixels: &[u8]) {
    image::save_buffer(path, pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)
        .expect("write Bistro visibility capture");
}

fn load_rgba(path: &Path) -> Vec<u8> {
    let image = image::open(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .to_rgba8();
    assert_eq!(image.dimensions(), (WIDTH, HEIGHT));
    image.into_raw()
}

fn luminance(pixel: &[u8]) -> f64 {
    (0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2]))
        / 255.0
}

fn image_metrics(reference: &[u8], candidate: &[u8], width: u32, height: u32) -> ImageMetrics {
    const WINDOW: usize = 8;
    const C1: f64 = 0.0001;
    const C2: f64 = 0.0009;
    assert_eq!(reference.len(), candidate.len());

    let mut deltas = Vec::with_capacity((width * height * 3) as usize);
    let mut squared_delta = 0.0f64;
    for (expected, actual) in reference.chunks_exact(4).zip(candidate.chunks_exact(4)) {
        for channel in 0..3 {
            let delta = expected[channel].abs_diff(actual[channel]);
            deltas.push(delta);
            squared_delta += f64::from(delta) * f64::from(delta);
        }
    }
    deltas.sort_unstable();
    let count = deltas.len().max(1);
    let total_delta = deltas.iter().map(|&delta| u64::from(delta)).sum::<u64>();
    let percentile_99_index = ((count - 1) * 99) / 100;

    let width = width as usize;
    let height = height as usize;
    let mut ssim = 0.0;
    let mut windows = 0usize;
    for y0 in (0..=height - WINDOW).step_by(WINDOW) {
        for x0 in (0..=width - WINDOW).step_by(WINDOW) {
            let mut mean_a = 0.0;
            let mut mean_b = 0.0;
            for y in y0..y0 + WINDOW {
                for x in x0..x0 + WINDOW {
                    let index = (y * width + x) * 4;
                    mean_a += luminance(&reference[index..index + 4]);
                    mean_b += luminance(&candidate[index..index + 4]);
                }
            }
            let samples = (WINDOW * WINDOW) as f64;
            mean_a /= samples;
            mean_b /= samples;
            let mut variance_a = 0.0;
            let mut variance_b = 0.0;
            let mut covariance = 0.0;
            for y in y0..y0 + WINDOW {
                for x in x0..x0 + WINDOW {
                    let index = (y * width + x) * 4;
                    let a = luminance(&reference[index..index + 4]) - mean_a;
                    let b = luminance(&candidate[index..index + 4]) - mean_b;
                    variance_a += a * a;
                    variance_b += b * b;
                    covariance += a * b;
                }
            }
            variance_a /= samples;
            variance_b /= samples;
            covariance /= samples;
            ssim += ((2.0 * mean_a * mean_b + C1) * (2.0 * covariance + C2))
                / ((mean_a * mean_a + mean_b * mean_b + C1) * (variance_a + variance_b + C2));
            windows += 1;
        }
    }

    ImageMetrics {
        mean_rgb: total_delta as f64 / count as f64,
        root_mean_square_rgb: (squared_delta / count as f64).sqrt(),
        percentile_99_rgb: deltas[percentile_99_index],
        max_rgb: deltas.last().copied().unwrap_or(0),
        ssim: ssim / windows as f64,
    }
}
