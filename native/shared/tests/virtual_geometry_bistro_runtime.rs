//! Environment-gated detailed Bistro virtual-geometry qualification.
//!
//! Run with:
//! BLOOM_BISTRO_VIRTUAL_SCENE=/path/BistroReference.gltf \
//! BLOOM_BISTRO_VIRTUAL_ARCHIVE=/path/BistroReference.bgeo \
//! cargo test --test virtual_geometry_bistro_runtime --features models3d -- --nocapture

use bloom_shared::engine::EngineState;
use bloom_shared::renderer::{Renderer, IDENTITY_MAT4};
use bloom_shared::virtual_geometry::{
    GpuVirtualGeometryConfig, GpuVirtualTraversalConfig, VirtualGeometryAsset,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const START_X: f32 = -3.2720;
const START_Z: f32 = 7.2358;
const START_YAW: f32 = -0.344;
const END_X: f32 = START_X + 3.748170285;
const END_Z: f32 = START_Z + 4.685212856;
const MOTION_STEPS: u32 = 30;
const MOTION_SAMPLE_INTERVAL: u32 = 5;

#[derive(Clone, Copy, Debug)]
struct ImageMetrics {
    mean_rgb: f64,
    ssim: f64,
    missing_geometry_fraction: f64,
    background_leak_fraction: f64,
}

#[test]
fn detailed_bistro_virtual_geometry_matches_camera_endpoint() {
    let Some(scene_path) = std::env::var_os("BLOOM_BISTRO_VIRTUAL_SCENE").map(PathBuf::from) else {
        eprintln!("skip: BLOOM_BISTRO_VIRTUAL_SCENE is not set");
        return;
    };
    let Some(archive_path) = std::env::var_os("BLOOM_BISTRO_VIRTUAL_ARCHIVE").map(PathBuf::from)
    else {
        eprintln!("skip: BLOOM_BISTRO_VIRTUAL_ARCHIVE is not set");
        return;
    };

    let directory = std::env::var_os("BLOOM_BISTRO_VIRTUAL_DIAGNOSTICS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("bloom-bistro-virtual-{}", std::process::id()))
        });
    if let Some(mode) = std::env::var_os("BLOOM_BISTRO_VIRTUAL_CHILD") {
        std::fs::create_dir_all(&directory).expect("create Bistro child output directory");
        match mode.to_string_lossy().as_ref() {
            "ordinary" => run_ordinary_child(&scene_path, &directory),
            "virtual" => run_virtual_child(&scene_path, &archive_path, &directory),
            value => panic!("unknown Bistro virtual child mode {value}"),
        }
        return;
    }

    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("create Bistro qualification directory");
    let executable = std::env::current_exe().expect("resolve Bistro qualification executable");
    for mode in ["ordinary", "virtual"] {
        let status = std::process::Command::new(&executable)
            .args([
                "--exact",
                "detailed_bistro_virtual_geometry_matches_camera_endpoint",
                "--nocapture",
            ])
            .env("BLOOM_BISTRO_VIRTUAL_CHILD", mode)
            .env("BLOOM_BISTRO_VIRTUAL_DIAGNOSTICS", &directory)
            .status()
            .unwrap_or_else(|error| panic!("launch {mode} Bistro qualification child: {error}"));
        assert!(status.success(), "{mode} Bistro qualification child failed");
    }

    let ordinary = load_rgba(&directory.join("ordinary.png"));
    let direct = load_rgba(&directory.join("virtual-direct.png"));
    let returned = load_rgba(&directory.join("virtual-returned.png"));
    let parity = image_metrics(&ordinary, &direct, WIDTH, HEIGHT);
    let motion = image_metrics(&direct, &returned, WIDTH, HEIGHT);
    let mut minimum_motion_parity_ssim = 1.0f64;
    let mut maximum_motion_parity_mean_rgb = 0.0f64;
    let mut maximum_motion_missing_geometry = 0.0f64;
    let mut maximum_motion_background_leak = 0.0f64;
    let mut minimum_path_return_ssim = 1.0f64;
    let mut maximum_path_return_mean_rgb = 0.0f64;
    let mut motion_metrics = Vec::new();
    for step in (MOTION_SAMPLE_INTERVAL..=MOTION_STEPS).step_by(MOTION_SAMPLE_INTERVAL as usize) {
        let ordinary_frame = load_rgba(&motion_path(directory.as_path(), "ordinary", step));
        let outbound = load_rgba(&motion_path(directory.as_path(), "virtual-outbound", step));
        let outbound_parity = image_metrics(&ordinary_frame, &outbound, WIDTH, HEIGHT);
        minimum_motion_parity_ssim = minimum_motion_parity_ssim.min(outbound_parity.ssim);
        maximum_motion_parity_mean_rgb =
            maximum_motion_parity_mean_rgb.max(outbound_parity.mean_rgb);
        maximum_motion_missing_geometry =
            maximum_motion_missing_geometry.max(outbound_parity.missing_geometry_fraction);
        maximum_motion_background_leak =
            maximum_motion_background_leak.max(outbound_parity.background_leak_fraction);
        let path_return = (step < MOTION_STEPS).then(|| {
            let returned_frame =
                load_rgba(&motion_path(directory.as_path(), "virtual-return", step));
            let metrics = image_metrics(&outbound, &returned_frame, WIDTH, HEIGHT);
            minimum_path_return_ssim = minimum_path_return_ssim.min(metrics.ssim);
            maximum_path_return_mean_rgb = maximum_path_return_mean_rgb.max(metrics.mean_rgb);
            metrics
        });
        motion_metrics.push((step, outbound_parity, path_return));
    }
    std::fs::write(
        directory.join("metrics.txt"),
        format!(
            "parity={parity:?}\nmotion={motion:?}\nminimum_motion_parity_ssim={minimum_motion_parity_ssim:?}\nmaximum_motion_parity_mean_rgb={maximum_motion_parity_mean_rgb:?}\nmaximum_motion_missing_geometry={maximum_motion_missing_geometry:?}\nmaximum_motion_background_leak={maximum_motion_background_leak:?}\nminimum_path_return_ssim={minimum_path_return_ssim:?}\nmaximum_path_return_mean_rgb={maximum_path_return_mean_rgb:?}\nmotion_metrics={motion_metrics:#?}\n"
        ),
    )
    .expect("write Bistro virtual metrics");
    eprintln!(
        "detailed-bistro-virtual parity={parity:?} motion={motion:?} motion_ssim_min={minimum_motion_parity_ssim:.8} motion_mean_max={maximum_motion_parity_mean_rgb:.6} motion_missing_max={maximum_motion_missing_geometry:.8} motion_background_leak_max={maximum_motion_background_leak:.8} path_ssim_min={minimum_path_return_ssim:.8} path_mean_max={maximum_path_return_mean_rgb:.6} diagnostics={}",
        directory.display()
    );
    assert!(
        parity.ssim >= 0.80
            && parity.mean_rgb <= 8.0
            && parity.missing_geometry_fraction <= 0.005
            && parity.background_leak_fraction <= 0.001,
        "virtual Bistro diverged from the ordinary material/camera reference: {parity:?}"
    );
    assert!(
        motion.ssim >= 0.985,
        "virtual Bistro failed to return to a stable camera endpoint: {motion:?}"
    );
    assert!(
        minimum_motion_parity_ssim >= 0.80
            && maximum_motion_parity_mean_rgb <= 8.0
            && maximum_motion_missing_geometry <= 0.005
            && maximum_motion_background_leak <= 0.001,
        "virtual Bistro produced a hole/flash along the fixed motion corpus: {motion_metrics:#?}"
    );
    assert!(
        minimum_path_return_ssim >= 0.985,
        "virtual Bistro motion was path-dependent at a matched camera: {motion_metrics:#?}"
    );
    if std::env::var_os("BLOOM_KEEP_BISTRO_VIRTUAL_DIAGNOSTICS").is_none() {
        let _ = std::fs::remove_dir_all(directory);
    }
}

fn run_ordinary_child(scene_path: &Path, directory: &Path) {
    unsafe {
        std::env::set_var("BLOOM_VISIBILITY_BUFFER", "off");
        std::env::set_var("BLOOM_GPU_DRIVEN", "0");
        std::env::set_var("BLOOM_SKIP_SKY", "1");
    }
    let mut engine = new_engine("ordinary");
    configure(&mut engine.renderer);
    let model_handle = load_model(&mut engine, scene_path);
    attach_model_placements(&mut engine, model_handle);
    for _ in 0..4 {
        retained_frame(&mut engine, START_X, START_Z, START_YAW, false);
    }
    let ordinary = retained_frame(&mut engine, START_X, START_Z, START_YAW, true)
        .expect("ordinary Bistro reference screenshot");
    save_rgba(&directory.join("ordinary.png"), &ordinary);
    for step in 1..=MOTION_STEPS {
        let (x, z) = motion_camera(step);
        let capture = step % MOTION_SAMPLE_INTERVAL == 0;
        let frame = retained_frame(&mut engine, x, z, START_YAW, capture);
        if let Some(frame) = frame {
            save_rgba(&motion_path(directory, "ordinary", step), &frame);
        }
    }
}

fn run_virtual_child(scene_path: &Path, archive_path: &Path, directory: &Path) {
    unsafe {
        std::env::set_var("BLOOM_VISIBILITY_BUFFER", "shade");
        std::env::remove_var("BLOOM_GPU_DRIVEN");
        std::env::set_var("BLOOM_SKIP_SKY", "1");
    }
    let mut engine = new_engine("virtual");
    configure(&mut engine.renderer);
    let model_handle = load_model(&mut engine, scene_path);
    let archive_bytes = std::fs::read(&archive_path).expect("read detailed Bistro .bgeo");
    let asset = Arc::new(
        VirtualGeometryAsset::from_bytes(archive_bytes)
            .expect("validate detailed Bistro virtual archive"),
    );
    let archive = asset.archive();
    let max_hierarchy_level = archive
        .clusters
        .iter()
        .map(|cluster| cluster.lod_level)
        .max()
        .unwrap_or(0);
    assert!(
        max_hierarchy_level < 16,
        "unexpected Bistro hierarchy depth"
    );
    let max_cluster_records = u32::try_from(archive.clusters.len()).expect("Bistro cluster count");
    let max_page_records = u32::try_from(archive.pages.len()).expect("Bistro page count");
    let root_page_count =
        u32::try_from(archive.coarse_root_page_count()).expect("Bistro root page count");
    let root_page_bytes = archive.coarse_root_page_bytes();
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
                // Native-full currently negotiates a 128 MiB storage-buffer
                // ceiling on Metal. Root residency consumes ~58 MiB, leaving
                // a fixed ~70 MiB camera-working set for streamed detail.
                capacity_bytes: 128 * 1024 * 1024,
                page_stride_bytes: archive.page_budget_bytes,
                max_meshes: 1,
                max_page_records,
                max_cluster_records,
                max_clusters_per_group,
                max_hierarchy_levels: 16,
                max_upload_bytes_per_frame: (64 * 1024 * 1024).max(root_page_bytes),
                max_upload_pages_per_frame: 1_024.max(root_page_count),
                max_evictions_per_frame: 1_024,
            },
            GpuVirtualTraversalConfig {
                max_instances: 2_048,
                max_selected_clusters: 524_288,
                max_page_requests: 65_536,
            },
        )
        .expect("enable detailed Bistro virtual geometry");

    let queue = engine.renderer.queue.clone();
    let virtual_mesh = engine
        .renderer
        .virtual_geometry_pool_mut()
        .expect("virtual pool enabled")
        .register_mesh(&queue, Arc::clone(&asset))
        .expect("register detailed Bistro virtual archive");
    let compact_handle = model_handle.to_bits() ^ 0x5647_434f_4d50_4154;
    let (route, instances) = {
        let model = engine
            .models
            .get(model_handle)
            .expect("loaded Bistro model");
        let route = model
            .route_virtual_geometry(&asset)
            .expect("exact Bistro virtual/compatibility partition");
        assert!(!route.virtual_placements.is_empty());
        engine
            .renderer
            .bind_model_virtual_materials(virtual_mesh, model)
            .expect("derive Bistro virtual material table");
        assert!(engine
            .renderer
            .cache_model_virtual_compatibility(compact_handle, model, &route,));
        let instances = route
            .virtual_instances(virtual_mesh, 0, IDENTITY_MAT4, IDENTITY_MAT4, [1.0; 4])
            .expect("build detailed Bistro virtual placements");
        eprintln!(
            "detailed-bistro-virtual placements={} compatibility={} clusters={} pages={}",
            instances.len(),
            route.compatibility_placements.len(),
            archive.clusters.len(),
            archive.pages.len(),
        );
        (route, instances)
    };

    let warmup_frames = std::env::var("BLOOM_BISTRO_VIRTUAL_WARMUP_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(180);
    for _ in 0..warmup_frames {
        virtual_frame(
            &mut engine,
            compact_handle,
            &route,
            &instances,
            START_X,
            START_Z,
            START_YAW,
            false,
        );
    }
    let direct = virtual_frame(
        &mut engine,
        compact_handle,
        &route,
        &instances,
        START_X,
        START_Z,
        START_YAW,
        true,
    )
    .expect("direct virtual Bistro screenshot");
    let direct_report = engine.renderer.renderer_capability_report_json();

    for step in 1..=MOTION_STEPS {
        let (x, z) = motion_camera(step);
        let capture = step % MOTION_SAMPLE_INTERVAL == 0;
        let frame = virtual_frame(
            &mut engine,
            compact_handle,
            &route,
            &instances,
            x,
            z,
            START_YAW,
            capture,
        );
        if let Some(frame) = frame {
            save_rgba(&motion_path(directory, "virtual-outbound", step), &frame);
            std::fs::write(
                directory.join(format!("virtual-outbound-{step:02}-report.json")),
                engine.renderer.renderer_capability_report_json(),
            )
            .expect("write outbound Bistro runtime report");
        }
    }
    let endpoint_settle_frames = std::env::var("BLOOM_BISTRO_VIRTUAL_ENDPOINT_SETTLE_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if endpoint_settle_frames != 0 {
        for _ in 0..endpoint_settle_frames {
            virtual_frame(
                &mut engine,
                compact_handle,
                &route,
                &instances,
                END_X,
                END_Z,
                START_YAW,
                false,
            );
        }
        let settled = virtual_frame(
            &mut engine,
            compact_handle,
            &route,
            &instances,
            END_X,
            END_Z,
            START_YAW,
            true,
        )
        .expect("settled endpoint virtual Bistro screenshot");
        save_rgba(&directory.join("virtual-endpoint-settled.png"), &settled);
        std::fs::write(
            directory.join("virtual-endpoint-settled-report.json"),
            engine.renderer.renderer_capability_report_json(),
        )
        .expect("write settled endpoint Bistro runtime report");
    }
    for return_step in 1..=MOTION_STEPS {
        let outbound_step = MOTION_STEPS - return_step;
        let (x, z) = motion_camera(outbound_step);
        let capture = outbound_step > 0 && outbound_step % MOTION_SAMPLE_INTERVAL == 0;
        let frame = virtual_frame(
            &mut engine,
            compact_handle,
            &route,
            &instances,
            x,
            z,
            START_YAW,
            capture,
        );
        if let Some(frame) = frame {
            save_rgba(
                &motion_path(directory, "virtual-return", outbound_step),
                &frame,
            );
        }
    }
    for _ in 0..30 {
        virtual_frame(
            &mut engine,
            compact_handle,
            &route,
            &instances,
            START_X,
            START_Z,
            START_YAW,
            false,
        );
    }
    let returned = virtual_frame(
        &mut engine,
        compact_handle,
        &route,
        &instances,
        START_X,
        START_Z,
        START_YAW,
        true,
    )
    .expect("returned virtual Bistro screenshot");

    let report = engine.renderer.renderer_capability_report_json();
    save_rgba(&directory.join("virtual-direct.png"), &direct);
    save_rgba(&directory.join("virtual-returned.png"), &returned);
    std::fs::write(directory.join("virtual-direct-report.json"), direct_report)
        .expect("write direct Bistro virtual runtime report");
    std::fs::write(directory.join("virtual-report.json"), report)
        .expect("write Bistro virtual runtime report");
}

fn new_engine(mode: &str) -> EngineState {
    bloom_shared::attach::attach_headless_engine(wgpu::Backends::PRIMARY, WIDTH, HEIGHT)
        .unwrap_or_else(|error| panic!("detailed Bistro {mode} device setup failed: {error}"))
}

fn load_model(engine: &mut EngineState, scene_path: &Path) -> f64 {
    let model_bytes = std::fs::read(scene_path).expect("read detailed Bistro glTF");
    let model_handle = engine.models.load_model_with_textures_from_source_path(
        &model_bytes,
        scene_path,
        &mut engine.renderer,
    );
    assert!(model_handle > 0.0, "load detailed Bistro glTF");
    let model = engine
        .models
        .get(model_handle)
        .expect("loaded Bistro model");
    assert_eq!(
        model.meshes.len(),
        1_176,
        "unexpected Bistro placement corpus"
    );
    model_handle
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

fn begin_camera(renderer: &mut Renderer, x: f32, z: f32, yaw: f32) {
    let forward_x = -yaw.sin();
    let forward_z = -yaw.cos();
    renderer.set_clear_color(5.0, 7.0, 12.0, 255.0);
    renderer.begin_mode_3d(
        x,
        1.544,
        z,
        x + forward_x * 100.0,
        1.544,
        z + forward_z * 100.0,
        0.0,
        1.0,
        0.0,
        60.0,
        0.0,
    );
    renderer.set_ambient_light(255.0, 245.0, 232.0, 0.18);
    renderer.set_directional_light(0.59732, 0.79653, -0.0935387, 255.0, 212.0, 177.0, 1.4);
}

fn motion_camera(step: u32) -> (f32, f32) {
    let t = step as f32 / MOTION_STEPS as f32;
    (
        START_X + (END_X - START_X) * t,
        START_Z + (END_Z - START_Z) * t,
    )
}

fn motion_path(directory: &Path, label: &str, step: u32) -> PathBuf {
    directory.join(format!("{label}-{step:02}.png"))
}

fn attach_model_placements(engine: &mut EngineState, model_handle: f64) {
    let placements = {
        let model = engine
            .models
            .get(model_handle)
            .expect("loaded Bistro model");
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

#[allow(clippy::too_many_arguments)]
fn virtual_frame(
    engine: &mut EngineState,
    compatibility_handle: u64,
    route: &bloom_shared::models::ModelVirtualGeometryRoute,
    instances: &[bloom_shared::virtual_geometry::GpuVirtualInstance],
    x: f32,
    z: f32,
    yaw: f32,
    screenshot: bool,
) -> Option<Vec<u8>> {
    engine.begin_frame();
    begin_camera(&mut engine.renderer, x, z, yaw);
    if std::env::var_os("BLOOM_BISTRO_VIRTUAL_DISABLE_HIZ").is_some() {
        engine.renderer.reset_temporal_history();
    }
    assert!(engine.renderer.draw_model_cached_compatibility(
        compatibility_handle,
        [0.0; 3],
        1.0,
        [1.0; 4],
        route,
    ));
    engine
        .renderer
        .submit_virtual_geometry_current_view(
            instances,
            std::env::var("BLOOM_BISTRO_VIRTUAL_TARGET_ERROR")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(1.0),
        )
        .expect("submit detailed Bistro virtual frame");
    engine.renderer.screenshot_requested = screenshot;
    engine.end_frame();
    screenshot.then(|| take_screenshot(&mut engine.renderer))
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

fn luminance(pixel: &[u8]) -> f64 {
    (0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2]))
        / 255.0
}

fn image_metrics(reference: &[u8], candidate: &[u8], width: u32, height: u32) -> ImageMetrics {
    const WINDOW: usize = 8;
    const C1: f64 = 0.0001;
    const C2: f64 = 0.0009;
    assert_eq!(reference.len(), candidate.len());
    let mean_rgb = reference
        .chunks_exact(4)
        .zip(candidate.chunks_exact(4))
        .map(|(a, b)| {
            (0..3)
                .map(|channel| f64::from(a[channel].abs_diff(b[channel])))
                .sum::<f64>()
        })
        .sum::<f64>()
        / (f64::from(width) * f64::from(height) * 3.0);
    let missing_geometry_fraction = reference
        .chunks_exact(4)
        .zip(candidate.chunks_exact(4))
        .filter(|(reference, candidate)| {
            let reference = luminance(reference);
            let candidate = luminance(candidate);
            reference > 0.15 && candidate < 0.06 && reference - candidate > 0.12
        })
        .count() as f64
        / (f64::from(width) * f64::from(height));
    // With the sky intentionally skipped, the fixed Bistro clear color is an
    // exact geometry-hole sentinel. Compare against the ordinary reference so
    // legitimate background remains excluded while virtual-only leaks fail
    // even when broad SSIM/luminance metrics dilute them.
    const CLEAR_RGB: [u8; 3] = [29, 39, 60];
    let background_leak_fraction = reference
        .chunks_exact(4)
        .zip(candidate.chunks_exact(4))
        .filter(|(reference, candidate)| {
            let candidate_clear_delta = (0..3)
                .map(|channel| candidate[channel].abs_diff(CLEAR_RGB[channel]) as u32)
                .sum::<u32>();
            let reference_clear_delta = (0..3)
                .map(|channel| reference[channel].abs_diff(CLEAR_RGB[channel]) as u32)
                .sum::<u32>();
            candidate_clear_delta <= 3 && reference_clear_delta >= 48
        })
        .count() as f64
        / (f64::from(width) * f64::from(height));
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
        mean_rgb,
        ssim: ssim / windows as f64,
        missing_geometry_fraction,
        background_leak_fraction,
    }
}

fn load_rgba(path: &Path) -> Vec<u8> {
    let image = image::open(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .to_rgba8();
    assert_eq!(image.dimensions(), (WIDTH, HEIGHT));
    image.into_raw()
}

fn save_rgba(path: &Path, pixels: &[u8]) {
    image::save_buffer(path, pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)
        .expect("write Bistro virtual diagnostic image");
}
