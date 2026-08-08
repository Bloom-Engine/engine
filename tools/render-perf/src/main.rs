use bloom_shared::engine::EngineState;
use bloom_shared::models::MaterialAlphaMode;
use bloom_shared::renderer::device_negotiation::{
    request_device_with_fallback_and_trace, DeviceRequestOptions, DeviceRequestProfile,
};
use bloom_shared::renderer::{Renderer, Vertex3D};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Config {
    width: u32,
    height: u32,
    warmup_frames: u32,
    measured_frames: u32,
    quality_preset: u32,
    render_scale: Option<f32>,
    reactive_transparency: bool,
    profile_passes: bool,
    trace_dir: Option<PathBuf>,
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, Default)]
struct Percentiles {
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct UploadStats {
    submit_count: usize,
    measured_submit_count: usize,
    buffer_total_bytes: u64,
    texture_total_bytes: u64,
    per_frame: Percentiles,
}

#[derive(Clone, Copy, Debug, Default)]
struct FrameTiming {
    render_submit_ms: f64,
    prepare_ms: f64,
    end_frame_ms: f64,
}

fn parse_u32(value: Option<String>, flag: &str) -> Result<u32, String> {
    value
        .ok_or_else(|| format!("missing value for {flag}"))?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn parse_f32(value: Option<String>, flag: &str) -> Result<f32, String> {
    value
        .ok_or_else(|| format!("missing value for {flag}"))?
        .parse()
        .map_err(|_| format!("{flag} must be a number"))
}

fn config() -> Result<Config, String> {
    let mut width = 1920;
    let mut height = 1080;
    let mut warmup_frames = 180;
    let mut measured_frames = 300;
    let mut quality_preset = 4;
    let mut render_scale = None;
    let mut reactive_transparency = false;
    let mut profile_passes = false;
    let mut trace_dir = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--width" => width = parse_u32(args.next(), "--width")?,
            "--height" => height = parse_u32(args.next(), "--height")?,
            "--warmup" => warmup_frames = parse_u32(args.next(), "--warmup")?,
            "--frames" => measured_frames = parse_u32(args.next(), "--frames")?,
            "--quality-preset" => {
                quality_preset = parse_u32(args.next(), "--quality-preset")?.min(4);
            }
            "--render-scale" => {
                render_scale = Some(parse_f32(args.next(), "--render-scale")?.clamp(0.15, 1.0));
            }
            "--reactive-transparency" => reactive_transparency = true,
            "--profile-passes" => profile_passes = true,
            "--trace-dir" => {
                trace_dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "missing value for --trace-dir".to_owned())?,
                ));
            }
            "--out" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "missing value for --out".to_owned())?,
                ));
            }
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if width == 0 || height == 0 || warmup_frames == 0 || measured_frames < 2 {
        return Err("width/height/warmup must be positive and frames must be >= 2".to_owned());
    }
    Ok(Config {
        width,
        height,
        warmup_frames,
        measured_frames,
        quality_preset,
        render_scale,
        reactive_transparency,
        profile_passes,
        trace_dir,
        output: output.ok_or_else(|| "--out is required".to_owned())?,
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}

fn percentiles(values: impl IntoIterator<Item = f64>) -> Percentiles {
    let mut sorted: Vec<f64> = values.into_iter().collect();
    if sorted.is_empty() {
        return Percentiles::default();
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    Percentiles {
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        p50: percentile(&sorted, 0.50),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
        max: *sorted.last().expect("non-empty percentile input"),
    }
}

fn draw_static_ultra_scene(engine: &mut EngineState) {
    let renderer = &mut engine.renderer;
    renderer.set_clear_color(2.0, 2.0, 4.0, 255.0);
    renderer.begin_mode_3d(
        0.0, 8.0, 7.0, // eye
        0.0, 0.0, 0.0, // target
        0.0, 1.0, 0.0, 55.0, 0.0,
    );
    renderer.set_ambient_light(20.0, 24.0, 32.0, 0.12);
    renderer.set_directional_light(-0.5, -1.0, -0.3, 255.0, 242.0, 230.0, 1.2);
    renderer.draw_plane(0.0, 0.0, 0.0, 14.0, 14.0, 110.0, 110.0, 110.0, 255.0);
    renderer.draw_cube(0.0, 0.8, 0.0, 1.6, 1.6, 1.6, 210.0, 105.0, 35.0, 255.0);
    for i in 0..40u32 {
        let t = i as f32 / 40.0 * std::f32::consts::TAU;
        renderer.add_point_light(
            t.cos() * 4.0,
            1.2,
            t.sin() * 4.0,
            3.5,
            0.5 + 0.5 * t.cos(),
            0.5 + 0.5 * (t + 2.094).cos(),
            0.5 + 0.5 * (t + 4.189).cos(),
            1.6,
        );
    }
}

fn setup_reactive_transparency(engine: &mut EngineState) {
    let h = 0.9;
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [0.0, 0.0, -1.0],
            [[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h]],
        ),
        (
            [0.0, 0.0, 1.0],
            [[h, -h, h], [-h, -h, h], [-h, h, h], [h, h, h]],
        ),
        (
            [-1.0, 0.0, 0.0],
            [[-h, -h, h], [-h, -h, -h], [-h, h, -h], [-h, h, h]],
        ),
        (
            [1.0, 0.0, 0.0],
            [[h, -h, -h], [h, -h, h], [h, h, h], [h, h, -h]],
        ),
        (
            [0.0, 1.0, 0.0],
            [[-h, h, -h], [h, h, -h], [h, h, h], [-h, h, h]],
        ),
        (
            [0.0, -1.0, 0.0],
            [[-h, -h, h], [h, -h, h], [h, -h, -h], [-h, -h, -h]],
        ),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, positions) in faces {
        let base = vertices.len() as u32;
        for position in positions {
            vertices.push(Vertex3D {
                position,
                normal,
                color: [0.1, 0.8, 1.0, 0.65],
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    let node = engine.scene.create_node();
    engine.scene.update_geometry(node, vertices, indices);
    engine.scene.set_transform(
        node,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.2, 0.0, 1.0],
        ],
    );
    engine
        .scene
        .set_material_gltf_alpha(node, MaterialAlphaMode::Blend, 0.0, false);
    engine.scene.set_material_color(node, 0.1, 0.8, 1.0, 0.65);
}

fn render_frame(engine: &mut EngineState) -> FrameTiming {
    let frame_start = Instant::now();
    engine.begin_frame();
    draw_static_ultra_scene(engine);
    let prepare_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
    let end_start = Instant::now();
    engine.end_frame();
    FrameTiming {
        render_submit_ms: frame_start.elapsed().as_secs_f64() * 1000.0,
        prepare_ms,
        end_frame_ms: end_start.elapsed().as_secs_f64() * 1000.0,
    }
}

fn create_engine(config: &Config) -> Result<(EngineState, String), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|error| format!("request_adapter failed: {error}"))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!(
            "refusing performance qualification on CPU adapter {}",
            info.name
        ));
    }
    if let Some(directory) = &config.trace_dir {
        if directory.exists() {
            std::fs::remove_dir_all(directory)
                .map_err(|error| format!("clear trace directory: {error}"))?;
        }
        std::fs::create_dir_all(directory)
            .map_err(|error| format!("create trace directory: {error}"))?;
    }
    let trace = config.trace_dir.as_ref().map_or(wgpu::Trace::Off, |path| {
        wgpu::Trace::Directory(path.clone())
    });
    let negotiated = pollster::block_on(request_device_with_fallback_and_trace(
        &adapter,
        DeviceRequestOptions {
            // This workload measures the raster steady-state path. Requesting
            // ray query can itself change backend scheduling even while PT is
            // off, so keep the unused feature out of this comparison.
            allow_ray_query: false,
            profile: DeviceRequestProfile::NativeFull,
        },
        trace,
    ))
    .map_err(|error| format!("request_device failed: {error}"))?;
    let negotiation_report = negotiated.report.report_json();
    let mut renderer = Renderer::new_headless(
        negotiated.device,
        negotiated.queue,
        config.width,
        config.height,
    );
    renderer.set_device_negotiation_report(negotiation_report);
    renderer.apply_quality_preset(config.quality_preset);
    if let Some(render_scale) = config.render_scale {
        renderer.set_render_scale(render_scale);
    }
    let adapter_snapshot = renderer.quality_adapter_json();
    let mut engine = EngineState::new(renderer);
    if config.reactive_transparency {
        setup_reactive_transparency(&mut engine);
    }
    engine.target_fps = 0.0;
    Ok((engine, adapter_snapshot))
}

fn data_file(line: &str) -> Option<&str> {
    let marker = "File(\"";
    let start = line.find(marker)? + marker.len();
    let end = line[start..].find("\")")? + start;
    Some(&line[start..end])
}

fn trace_uploads(directory: &Path, measured_frames: u32) -> Result<UploadStats, String> {
    let trace_path = directory.join("trace.ron");
    let trace = std::fs::read_to_string(&trace_path)
        .map_err(|error| format!("read {}: {error}", trace_path.display()))?;
    // wgpu-core keeps its global trace recorder alive until process teardown,
    // so a live-process snapshot need not have the final closing bracket yet.
    // DiskTrace writes each action directly to File; complete Submit actions
    // and their preceding upload payloads are safe to analyze here.
    #[derive(Clone, Copy)]
    enum WriteKind {
        Buffer,
        Texture,
    }
    let mut current_write = None;
    let mut frame_buffer_bytes = 0u64;
    let mut frame_texture_bytes = 0u64;
    let mut frames = Vec::new();
    for line in trace.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("WriteBuffer(") {
            current_write = Some(WriteKind::Buffer);
        } else if trimmed.starts_with("WriteTexture(") {
            current_write = Some(WriteKind::Texture);
        }
        if let (Some(kind), Some(file)) = (current_write, data_file(trimmed)) {
            let bytes = std::fs::metadata(directory.join(file))
                .map_err(|error| format!("stat traced upload {file}: {error}"))?
                .len();
            match kind {
                WriteKind::Buffer => frame_buffer_bytes += bytes,
                WriteKind::Texture => frame_texture_bytes += bytes,
            }
            current_write = None;
        }
        if trimmed.starts_with("Submit(") {
            frames.push((frame_buffer_bytes, frame_texture_bytes));
            frame_buffer_bytes = 0;
            frame_texture_bytes = 0;
            current_write = None;
        }
    }
    let measured = measured_frames as usize;
    if frames.len() < measured {
        return Err(format!(
            "trace has {} submissions, fewer than {measured} measured frames",
            frames.len()
        ));
    }
    let selected = &frames[frames.len() - measured..];
    let buffer_total_bytes = selected.iter().map(|(buffer, _)| *buffer).sum();
    let texture_total_bytes = selected.iter().map(|(_, texture)| *texture).sum();
    Ok(UploadStats {
        submit_count: frames.len(),
        measured_submit_count: selected.len(),
        buffer_total_bytes,
        texture_total_bytes,
        per_frame: percentiles(
            selected
                .iter()
                .map(|(buffer, texture)| (buffer + texture) as f64),
        ),
    })
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn write_report(
    config: &Config,
    adapter_snapshot: &str,
    renderer_paths: &str,
    actual_render_scale: f32,
    render_submit: Percentiles,
    prepare: Percentiles,
    end_frame: Percentiles,
    uploads: Option<UploadStats>,
    pass_profile: Option<&str>,
) -> Result<(), String> {
    let revision = std::env::var("BLOOM_RENDER_PERF_ENGINE_REVISION").unwrap_or_else(|_| {
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_owned())
            .unwrap_or_else(|| "unknown".to_owned())
    });
    let upload_json = uploads.map_or_else(
        || "null".to_owned(),
        |value| {
            format!(
                concat!(
                    "{{\"trace_submit_count\":{},\"measured_submit_count\":{},",
                    "\"buffer_total_bytes\":{},\"texture_total_bytes\":{},",
                    "\"total_bytes\":{},\"per_frame_mean_bytes\":{:.3},",
                    "\"per_frame_p50_bytes\":{:.3},\"per_frame_p95_bytes\":{:.3},",
                    "\"per_frame_p99_bytes\":{:.3},\"per_frame_max_bytes\":{:.3}}}"
                ),
                value.submit_count,
                value.measured_submit_count,
                value.buffer_total_bytes,
                value.texture_total_bytes,
                value.buffer_total_bytes + value.texture_total_bytes,
                value.per_frame.mean,
                value.per_frame.p50,
                value.per_frame.p95,
                value.per_frame.p99,
                value.per_frame.max,
            )
        },
    );
    let pass_profile_json = pass_profile.unwrap_or("null");
    let report = format!(
        concat!(
            "{{\n  \"schema\":\"bloom-render-perf-v1\",\n",
            "  \"revision\":\"{}\",\n",
            "  \"adapter\":{},\n",
            "  \"renderer_paths\":{},\n",
            "  \"resolution\":[{},{}],\n",
            "  \"quality_preset\":{},\n",
            "  \"render_scale\":{:.6},\n",
            "  \"reactive_transparency_workload\":{},\n",
            "  \"pass_profile\":{},\n",
            "  \"headless_uncapped\":true,\n",
            "  \"warmup_frames\":{},\n",
            "  \"measured_frames\":{},\n",
            "  \"timing_includes_trace_io\":{},\n",
            "  \"cpu_render_submit_ms\":{{\"mean\":{:.6},\"p50\":{:.6},",
            "\"p95\":{:.6},\"p99\":{:.6},\"max\":{:.6}}},\n",
            "  \"cpu_prepare_ms\":{{\"mean\":{:.6},\"p50\":{:.6},",
            "\"p95\":{:.6},\"p99\":{:.6},\"max\":{:.6}}},\n",
            "  \"cpu_end_frame_ms\":{{\"mean\":{:.6},\"p50\":{:.6},",
            "\"p95\":{:.6},\"p99\":{:.6},\"max\":{:.6}}},\n",
            "  \"uploads\":{}\n}}\n"
        ),
        json_escape(&revision),
        adapter_snapshot,
        renderer_paths,
        config.width,
        config.height,
        config.quality_preset,
        actual_render_scale,
        config.reactive_transparency,
        pass_profile_json,
        config.warmup_frames,
        config.measured_frames,
        config.trace_dir.is_some(),
        render_submit.mean,
        render_submit.p50,
        render_submit.p95,
        render_submit.p99,
        render_submit.max,
        prepare.mean,
        prepare.p50,
        prepare.p95,
        prepare.p99,
        prepare.max,
        end_frame.mean,
        end_frame.p50,
        end_frame.p95,
        end_frame.p99,
        end_frame.max,
        upload_json,
    );
    if let Some(parent) = config.output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create report directory: {error}"))?;
    }
    std::fs::write(&config.output, report)
        .map_err(|error| format!("write {}: {error}", config.output.display()))
}

fn run() -> Result<(), String> {
    let config = config()?;
    let (mut engine, adapter_snapshot) = create_engine(&config)?;
    let actual_render_scale = engine.renderer.render_scale();
    for _ in 0..config.warmup_frames {
        let _ = render_frame(&mut engine);
    }
    if config.profile_passes {
        engine.profiler.set_enabled(true);
    }
    let measurement_start = Instant::now();
    let mut timing_samples = Vec::with_capacity(config.measured_frames as usize);
    for _ in 0..config.measured_frames {
        timing_samples.push(render_frame(&mut engine));
    }
    let measurement_wall_ms = measurement_start.elapsed().as_secs_f64() * 1000.0;
    let _ = engine.renderer.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let render_submit = percentiles(timing_samples.iter().map(|sample| sample.render_submit_ms));
    let prepare = percentiles(timing_samples.iter().map(|sample| sample.prepare_ms));
    let end_frame = percentiles(timing_samples.iter().map(|sample| sample.end_frame_ms));
    let renderer_paths = engine.renderer.quality_runtime_paths_json();
    let pass_profile = config.profile_passes.then(|| {
        engine.profiler.quality_report_json(
            3,
            config.warmup_frames,
            config.measured_frames,
            1.0 / 60.0,
            config.quality_preset,
            actual_render_scale as f64,
            measurement_wall_ms,
            &adapter_snapshot,
            &renderer_paths,
        )
    });
    drop(engine);
    let uploads = config
        .trace_dir
        .as_deref()
        .map(|directory| trace_uploads(directory, config.measured_frames))
        .transpose()?;
    write_report(
        &config,
        &adapter_snapshot,
        &renderer_paths,
        actual_render_scale,
        render_submit,
        prepare,
        end_frame,
        uploads,
        pass_profile.as_deref(),
    )?;
    println!(
        "bloom-render-perf {}x{} CPU p50={:.3} p95={:.3} p99={:.3} ms{}",
        config.width,
        config.height,
        render_submit.p50,
        render_submit.p95,
        render_submit.p99,
        uploads.map_or(String::new(), |value| format!(
            " upload/frame={:.0} bytes",
            value.per_frame.mean
        ))
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("bloom-render-perf: {error}");
        std::process::exit(2);
    }
}
