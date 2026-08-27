//! Environment-gated 10M-source-triangle virtual-geometry stress gate.
//!
//! Run with:
//! BLOOM_VIRTUAL_STRESS_SCENE=/path/stress-10m.gltf \
//! BLOOM_VIRTUAL_STRESS_ARCHIVE=/path/stress-10m.bgeo \
//! cargo test --release --test virtual_geometry_stress_runtime \
//!   --features models3d -- --nocapture

use bloom_shared::engine::EngineState;
use bloom_shared::renderer::{Renderer, IDENTITY_MAT4};
use bloom_shared::virtual_geometry::{
    GpuVirtualGeometryConfig, GpuVirtualTraversalConfig, VirtualGeometryAsset,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const MINIMUM_SOURCE_TRIANGLES: u64 = 10_000_000;
const POOL_BYTES: u64 = 64 * 1024 * 1024;
const CAMERA_CENTER: [f32; 3] = [61.25, 0.0, 61.25];
const MAXIMUM_WALL_FRAME_MEAN_MS: f64 = 1000.0 / 60.0;
const MAXIMUM_GPU_FRAME_MEAN_MS: f64 = 8.0;
const MAXIMUM_GPU_FRAME_P95_MS: f64 = 12.0;
const MAXIMUM_SELECTOR_GPU_MEAN_MS: f64 = 3.0;
const MAXIMUM_DRAW_EMISSION_GPU_MEAN_MS: f64 = 0.5;

#[test]
fn virtual_geometry_renders_ten_million_source_triangles_with_fixed_residency() {
    let Some(scene_path) = std::env::var_os("BLOOM_VIRTUAL_STRESS_SCENE").map(PathBuf::from) else {
        eprintln!("skip: BLOOM_VIRTUAL_STRESS_SCENE is not set");
        return;
    };
    let Some(archive_path) = std::env::var_os("BLOOM_VIRTUAL_STRESS_ARCHIVE").map(PathBuf::from)
    else {
        eprintln!("skip: BLOOM_VIRTUAL_STRESS_ARCHIVE is not set");
        return;
    };
    let diagnostics = std::env::var_os("BLOOM_VIRTUAL_STRESS_DIAGNOSTICS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("bloom-virtual-stress-{}", std::process::id()))
        });
    std::fs::create_dir_all(&diagnostics).expect("create virtual stress diagnostics");

    unsafe {
        std::env::set_var("BLOOM_VISIBILITY_BUFFER", "shade");
        std::env::remove_var("BLOOM_GPU_DRIVEN");
        std::env::set_var("BLOOM_SKIP_SKY", "1");
    }
    let mut engine =
        bloom_shared::attach::attach_headless_engine(wgpu::Backends::PRIMARY, WIDTH, HEIGHT)
            .unwrap_or_else(|error| panic!("virtual stress device setup failed: {error}"));
    configure(&mut engine.renderer);
    let model_handle = load_model(&mut engine, &scene_path);
    let archive_bytes = std::fs::read(&archive_path).expect("read virtual stress archive");
    let asset = Arc::new(
        VirtualGeometryAsset::from_bytes(archive_bytes)
            .expect("validate 10M virtual stress archive"),
    );
    let archive = asset.archive();
    let source_triangles = archive
        .clusters
        .iter()
        .filter(|cluster| cluster.lod_level == 0)
        .map(|cluster| u64::from(cluster.triangle_count))
        .sum::<u64>();
    assert!(
        source_triangles >= MINIMUM_SOURCE_TRIANGLES,
        "stress archive contains only {source_triangles} source triangles"
    );
    assert!(
        archive.coarse_root_page_bytes() <= POOL_BYTES,
        "coarse roots alone exceed the fixed stress residency budget"
    );

    let max_page_records = u32::try_from(archive.pages.len()).expect("stress page count");
    let max_cluster_records = u32::try_from(archive.clusters.len()).expect("stress cluster count");
    let root_page_count =
        u32::try_from(archive.coarse_root_page_count()).expect("stress root page count");
    let max_clusters_per_group = archive
        .clusters
        .iter()
        .map(|cluster| cluster.parent_count.max(cluster.child_count))
        .max()
        .unwrap_or(1)
        .max(1);
    engine
        .renderer
        .enable_virtual_geometry(
            GpuVirtualGeometryConfig {
                capacity_bytes: POOL_BYTES,
                page_stride_bytes: archive.page_budget_bytes,
                max_meshes: 1,
                max_page_records,
                max_cluster_records,
                max_clusters_per_group,
                max_hierarchy_levels: 16,
                max_upload_bytes_per_frame: 8 * 1024 * 1024,
                max_upload_pages_per_frame: 128.max(root_page_count),
                max_evictions_per_frame: 128,
            },
            GpuVirtualTraversalConfig {
                max_instances: 128,
                max_selected_clusters: 262_144,
                max_page_requests: 32_768,
            },
        )
        .expect("enable fixed-budget virtual stress runtime");

    let queue = engine.renderer.queue.clone();
    let virtual_mesh = engine
        .renderer
        .virtual_geometry_pool_mut()
        .expect("virtual stress pool enabled")
        .register_mesh(&queue, Arc::clone(&asset))
        .expect("register 10M virtual stress archive");
    let (route, instances) = {
        let model = engine
            .models
            .get(model_handle)
            .expect("loaded stress model");
        let route = model
            .route_virtual_geometry(&asset)
            .expect("route exact virtual stress source closure");
        assert_eq!(route.virtual_placements.len(), 100);
        assert!(route.compatibility_placements.is_empty());
        engine
            .renderer
            .bind_model_virtual_materials(virtual_mesh, model)
            .expect("bind stress material table");
        let instances = route
            .virtual_instances(virtual_mesh, 0, IDENTITY_MAT4, IDENTITY_MAT4, [1.0; 4])
            .expect("build stress virtual instances");
        (route, instances)
    };
    assert_eq!(instances.len(), 100);

    let warmup_frames = environment_frames("BLOOM_VIRTUAL_STRESS_WARMUP_FRAMES", 180);
    for frame in 0..warmup_frames {
        render_frame(
            &mut engine,
            &instances,
            warmup_camera(frame, warmup_frames),
            false,
        );
    }

    let measured_frames = environment_frames("BLOOM_VIRTUAL_STRESS_MEASURED_FRAMES", 120);
    engine.profiler.set_enabled(true);
    let measurement_start = Instant::now();
    let mut screenshot = None;
    for frame in 0..measured_frames {
        screenshot = render_frame(
            &mut engine,
            &instances,
            measured_camera(frame, measured_frames),
            frame + 1 == measured_frames,
        );
    }
    let measurement_wall_ms = measurement_start.elapsed().as_secs_f64() * 1000.0;
    let screenshot = screenshot.expect("capture final virtual stress frame");
    assert_rendered_pixels(&screenshot);
    image::save_buffer(
        diagnostics.join("stress-frame.png"),
        &screenshot,
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
    )
    .expect("write virtual stress frame");

    let runtime_report = engine.renderer.renderer_capability_report_json();
    let runtime: serde_json::Value =
        serde_json::from_str(&runtime_report).expect("parse stress runtime report");
    let virtual_runtime = &runtime["runtime_support"]["virtual_geometry"];
    let resident_pages = virtual_runtime["resident_pages"]
        .as_u64()
        .expect("stress resident page telemetry");
    assert_eq!(
        virtual_runtime["pool_capacity_bytes"].as_u64(),
        Some(POOL_BYTES)
    );
    assert!(
        resident_pages * u64::from(archive.page_budget_bytes) <= POOL_BYTES,
        "resident physical pages exceeded the fixed pool"
    );
    for counter in [
        "last_selected_overflow",
        "last_request_overflow",
        "last_invalid_records",
        "last_depth_limit_fallbacks",
        "last_missing_current_pages",
    ] {
        assert_eq!(
            virtual_runtime[counter].as_u64(),
            Some(0),
            "virtual stress runtime reported {counter}"
        );
    }

    let adapter = engine.renderer.quality_adapter_json();
    let paths = engine.renderer.quality_runtime_paths_json();
    let profile_report = engine.profiler.quality_report_json(
        3,
        warmup_frames,
        measured_frames,
        1.0 / 60.0,
        4,
        1.0,
        measurement_wall_ms,
        &adapter,
        &paths,
    );
    let profile: serde_json::Value =
        serde_json::from_str(&profile_report).expect("parse virtual stress profile");
    assert_eq!(profile["uncapped"].as_bool(), Some(true));
    assert_eq!(profile["gpu_timestamps_available"].as_bool(), Some(true));
    let pass_gpu_mean = |label: &str| {
        let pass = profile["passes"]
            .as_array()
            .and_then(|passes| passes.iter().find(|pass| pass["label"] == label))
            .unwrap_or_else(|| panic!("missing {label} stress timing"));
        pass["gpu_mean_ms"]
            .as_f64()
            .filter(|value| *value > 0.0)
            .unwrap_or_else(|| panic!("{label} has no positive GPU timing"))
    };
    let selector_gpu_mean_ms = pass_gpu_mean("virtual_geometry_hierarchy_selection");
    let draw_emission_gpu_mean_ms = pass_gpu_mean("virtual_geometry_draw_emission");
    let gpu_frame_mean_ms = profile["gpu_frame_mean_ms"]
        .as_f64()
        .expect("stress GPU frame mean");
    let gpu_frame_p95_ms = profile["gpu_frame_p95_ms"]
        .as_f64()
        .expect("stress GPU frame p95");
    let wall_frame_mean_ms = measurement_wall_ms / f64::from(measured_frames);
    assert!(wall_frame_mean_ms <= MAXIMUM_WALL_FRAME_MEAN_MS);
    assert!(gpu_frame_mean_ms <= MAXIMUM_GPU_FRAME_MEAN_MS);
    assert!(gpu_frame_p95_ms <= MAXIMUM_GPU_FRAME_P95_MS);
    assert!(selector_gpu_mean_ms <= MAXIMUM_SELECTOR_GPU_MEAN_MS);
    assert!(draw_emission_gpu_mean_ms <= MAXIMUM_DRAW_EMISSION_GPU_MEAN_MS);

    std::fs::write(diagnostics.join("runtime-report.json"), runtime_report)
        .expect("write virtual stress runtime report");
    std::fs::write(diagnostics.join("profile-report.json"), &profile_report)
        .expect("write virtual stress profile report");
    std::fs::write(
        diagnostics.join("summary.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "bloom-virtual-geometry-stress-result-v1",
            "source_triangles": source_triangles,
            "placements": route.virtual_placements.len(),
            "archive_clusters": archive.clusters.len(),
            "archive_pages": archive.pages.len(),
            "pool_capacity_bytes": POOL_BYTES,
            "resident_pages": resident_pages,
            "measurement_wall_ms": measurement_wall_ms,
            "measured_frames": measured_frames,
            "thresholds": {
                "maximum_wall_frame_mean_ms": MAXIMUM_WALL_FRAME_MEAN_MS,
                "maximum_gpu_frame_mean_ms": MAXIMUM_GPU_FRAME_MEAN_MS,
                "maximum_gpu_frame_p95_ms": MAXIMUM_GPU_FRAME_P95_MS,
                "maximum_selector_gpu_mean_ms": MAXIMUM_SELECTOR_GPU_MEAN_MS,
                "maximum_draw_emission_gpu_mean_ms": MAXIMUM_DRAW_EMISSION_GPU_MEAN_MS,
            },
            "runtime": virtual_runtime,
            "profile": profile,
        }))
        .expect("serialize virtual stress summary"),
    )
    .expect("write virtual stress summary");
    eprintln!(
        "virtual-geometry-stress source_triangles={source_triangles} resident_pages={resident_pages} wall_ms={measurement_wall_ms:.3} diagnostics={}",
        diagnostics.display()
    );
}

fn environment_frames(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
        .max(1)
}

fn load_model(engine: &mut EngineState, scene_path: &Path) -> f64 {
    let model_bytes = std::fs::read(scene_path).expect("read virtual stress glTF");
    let handle = engine.models.load_model_with_textures_from_source_path(
        &model_bytes,
        scene_path,
        &mut engine.renderer,
    );
    assert!(handle > 0.0, "load virtual stress glTF");
    assert_eq!(
        engine
            .models
            .get(handle)
            .expect("stress model")
            .meshes
            .len(),
        100
    );
    handle
}

fn configure(renderer: &mut Renderer) {
    renderer.apply_quality_preset(4);
    renderer.set_render_scale(1.0);
    renderer.set_taa_enabled(false);
    renderer.set_ssao_enabled(false);
    renderer.set_ssr_enabled(false);
    renderer.set_ssgi_enabled(false);
    renderer.set_bloom_enabled(false);
    renderer.set_motion_blur_enabled(false);
    renderer.set_sss_enabled(false);
    renderer.set_sharpen_strength(0.0);
    renderer.set_auto_exposure(false);
    renderer.set_manual_exposure(1.0);
    renderer.set_shadows_enabled(false);
}

fn warmup_camera(frame: u32, frames: u32) -> [f32; 3] {
    let phase = frame as f32 / frames.max(1) as f32 * std::f32::consts::TAU;
    [
        CAMERA_CENTER[0] + phase.sin() * 18.0,
        95.0,
        CAMERA_CENTER[2] + 92.0 + phase.cos() * 18.0,
    ]
}

fn measured_camera(frame: u32, frames: u32) -> [f32; 3] {
    let phase = frame as f32 / frames.max(1) as f32 * std::f32::consts::TAU;
    [
        CAMERA_CENTER[0] + phase.sin() * 3.0,
        95.0,
        CAMERA_CENTER[2] + 92.0 + phase.cos() * 3.0,
    ]
}

fn render_frame(
    engine: &mut EngineState,
    instances: &[bloom_shared::virtual_geometry::GpuVirtualInstance],
    camera: [f32; 3],
    screenshot: bool,
) -> Option<Vec<u8>> {
    engine.begin_frame();
    engine.renderer.set_clear_color(6.0, 9.0, 14.0, 255.0);
    engine.renderer.begin_mode_3d(
        camera[0],
        camera[1],
        camera[2],
        CAMERA_CENTER[0],
        CAMERA_CENTER[1],
        CAMERA_CENTER[2],
        0.0,
        1.0,
        0.0,
        55.0,
        0.0,
    );
    engine.renderer.set_ambient_light(210.0, 220.0, 255.0, 0.35);
    engine
        .renderer
        .set_directional_light(0.4, 0.8, -0.2, 255.0, 244.0, 220.0, 1.5);
    engine
        .renderer
        .submit_virtual_geometry_current_view(instances, 1.0)
        .expect("submit 10M virtual stress frame");
    engine.renderer.screenshot_requested = screenshot;
    engine.end_frame();
    screenshot.then(|| {
        let (_, _, mut pixels) = engine
            .renderer
            .screenshot_data
            .take()
            .expect("virtual stress screenshot");
        if matches!(
            engine.renderer.surface_format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        pixels
    })
}

fn assert_rendered_pixels(pixels: &[u8]) {
    let bright = pixels
        .chunks_exact(4)
        .filter(|pixel| u16::from(pixel[0]) + u16::from(pixel[1]) + u16::from(pixel[2]) > 120)
        .count();
    assert!(
        bright > pixels.len() / 4 / 20,
        "10M virtual stress frame did not render enough geometry"
    );
}
