use super::*;

#[derive(Debug, Eq, PartialEq)]
struct LiveGpuObjects {
    buffers: isize,
    textures: isize,
    texture_views: isize,
    bind_groups: isize,
    bind_group_layouts: isize,
    render_pipelines: isize,
    compute_pipelines: isize,
    pipeline_layouts: isize,
    samplers: isize,
    command_encoders: isize,
    shader_modules: isize,
    query_sets: isize,
    fences: isize,
    buffer_memory: isize,
    texture_memory: isize,
    acceleration_structure_memory: isize,
    memory_allocations: isize,
}

fn live_gpu_objects(device: &wgpu::Device) -> LiveGpuObjects {
    let counters = device.get_internal_counters();
    let hal = counters.hal;
    LiveGpuObjects {
        buffers: hal.buffers.read(),
        textures: hal.textures.read(),
        texture_views: hal.texture_views.read(),
        bind_groups: hal.bind_groups.read(),
        bind_group_layouts: hal.bind_group_layouts.read(),
        render_pipelines: hal.render_pipelines.read(),
        compute_pipelines: hal.compute_pipelines.read(),
        pipeline_layouts: hal.pipeline_layouts.read(),
        samplers: hal.samplers.read(),
        command_encoders: hal.command_encoders.read(),
        shader_modules: hal.shader_modules.read(),
        query_sets: hal.query_sets.read(),
        fences: hal.fences.read(),
        buffer_memory: hal.buffer_memory.read(),
        texture_memory: hal.texture_memory.read(),
        acceleration_structure_memory: hal.acceleration_structure_memory.read(),
        memory_allocations: hal.memory_allocations.read(),
    }
}

fn assert_gpu_objects_did_not_grow(before: &LiveGpuObjects, after: &LiveGpuObjects) {
    let before = [
        ("buffers", before.buffers),
        ("textures", before.textures),
        ("texture_views", before.texture_views),
        ("bind_groups", before.bind_groups),
        ("bind_group_layouts", before.bind_group_layouts),
        ("render_pipelines", before.render_pipelines),
        ("compute_pipelines", before.compute_pipelines),
        ("pipeline_layouts", before.pipeline_layouts),
        ("samplers", before.samplers),
        ("command_encoders", before.command_encoders),
        ("shader_modules", before.shader_modules),
        ("query_sets", before.query_sets),
        ("fences", before.fences),
        ("buffer_memory", before.buffer_memory),
        ("texture_memory", before.texture_memory),
        (
            "acceleration_structure_memory",
            before.acceleration_structure_memory,
        ),
        ("memory_allocations", before.memory_allocations),
    ];
    let after = [
        ("buffers", after.buffers),
        ("textures", after.textures),
        ("texture_views", after.texture_views),
        ("bind_groups", after.bind_groups),
        ("bind_group_layouts", after.bind_group_layouts),
        ("render_pipelines", after.render_pipelines),
        ("compute_pipelines", after.compute_pipelines),
        ("pipeline_layouts", after.pipeline_layouts),
        ("samplers", after.samplers),
        ("command_encoders", after.command_encoders),
        ("shader_modules", after.shader_modules),
        ("query_sets", after.query_sets),
        ("fences", after.fences),
        ("buffer_memory", after.buffer_memory),
        ("texture_memory", after.texture_memory),
        (
            "acceleration_structure_memory",
            after.acceleration_structure_memory,
        ),
        ("memory_allocations", after.memory_allocations),
    ];
    for ((before_name, before_count), (after_name, after_count)) in before.into_iter().zip(after) {
        assert_eq!(before_name, after_name);
        assert!(
            after_count <= before_count,
            "renderer-owned {after_name} grew over 1,000 static frames: \
             before={before_count}, after={after_count}"
        );
    }
}

fn wait_for_gpu(device: &wgpu::Device) {
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

#[test]
fn static_ultra_scene_has_stable_renderer_owned_memory_for_1000_frames() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.apply_quality_preset(4);

    let draw = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(2.0, 2.0, 4.0, 255.0);
        r.begin_mode_3d(
            0.0, 8.0, 7.0, // eye
            0.0, 0.0, 0.0, // target
            0.0, 1.0, 0.0, 55.0, 0.0,
        );
        r.draw_plane(0.0, 0.0, 0.0, 14.0, 14.0, 110.0, 110.0, 110.0, 255.0);
        r.draw_cube(0.0, 0.8, 0.0, 1.6, 1.6, 1.6, 210.0, 105.0, 35.0, 255.0);
        for i in 0..40u32 {
            let t = i as f32 / 40.0 * std::f32::consts::TAU;
            r.add_point_light(
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
    };
    let run_frames = |eng: &mut EngineState, count: u32| {
        for _ in 0..count {
            eng.begin_frame();
            draw(eng);
            eng.end_frame();
        }
    };

    // Settle temporal histories, rotating bind-group caches, queue-owned
    // staging allocations, and the three-frame headless submission window.
    run_frames(&mut eng, 16);
    wait_for_gpu(&eng.renderer.device);
    let before = live_gpu_objects(&eng.renderer.device);
    assert!(
        before.buffers > 0 && before.textures > 0,
        "wgpu test counters are disabled; this would be a vacuous memory gate: {before:?}"
    );
    let paths_before: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("pre-run runtime paths are valid JSON");
    let cpu_capacity_before = eng.renderer.quality_frame_cpu_capacity_bytes();

    run_frames(&mut eng, 1_000);
    wait_for_gpu(&eng.renderer.device);
    let after = live_gpu_objects(&eng.renderer.device);
    let paths_after: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("post-run runtime paths are valid JSON");
    let cpu_capacity_after = eng.renderer.quality_frame_cpu_capacity_bytes();

    assert_gpu_objects_did_not_grow(&before, &after);
    assert_eq!(
        cpu_capacity_after, cpu_capacity_before,
        "renderer-owned growable frame-container capacity changed over 1,000 static frames"
    );
    assert_eq!(
        paths_after["render_graph"]["cached_plan_count"],
        paths_before["render_graph"]["cached_plan_count"],
        "render-graph plan cache grew after warm-up"
    );
    assert_eq!(
        paths_after["render_graph"]["physical_transient_slots"],
        paths_before["render_graph"]["physical_transient_slots"],
        "compiled transient pool grew after warm-up"
    );
    let steady = &paths_after["steady_state_resources"];
    assert_eq!(steady["graph_compiles"].as_u64(), Some(0));
    assert_eq!(steady["pipeline_creations"]["first_use"].as_u64(), Some(0));
    assert_eq!(
        steady["transient_physical_creations"]["textures"].as_u64(),
        Some(0)
    );
    assert_eq!(
        steady["transient_physical_creations"]["buffers"].as_u64(),
        Some(0)
    );
    assert_eq!(steady["bind_group_creations"]["total"].as_u64(), Some(0));
    eprintln!(
        "1,000-frame renderer memory stable: {before:?}; frame_cpu_capacity_bytes={cpu_capacity_before}; graph_plans={}; transient_slots={}",
        paths_before["render_graph"]["cached_plan_count"],
        paths_before["render_graph"]["physical_transient_slots"],
    );
}

#[test]
fn golden_many_point_lights() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    // 40 colored point lights in a ring over a dark floor — far past the
    // old 16-light cap. If the cap regressed, lights 17..40 vanish and
    // the right side of the ring goes dark (well past tolerance).
    let (w, h, rgba) = render(&mut eng, 6, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(2.0, 2.0, 4.0, 255.0);
        r.begin_mode_3d(
            0.0, 9.0, 0.01, // eye: straight above
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 60.0, 0.0,
        );
        r.draw_plane(0.0, 0.0, 0.0, 14.0, 14.0, 110.0, 110.0, 110.0, 255.0);
        for i in 0..40u32 {
            let t = i as f32 / 40.0 * std::f32::consts::TAU;
            let (sx, sz) = (t.cos() * 4.0, t.sin() * 4.0);
            // Hue cycles so neighboring lights are distinguishable.
            let (lr, lg, lb) = (
                0.5 + 0.5 * t.cos(),
                0.5 + 0.5 * (t + 2.094).cos(),
                0.5 + 0.5 * (t + 4.189).cos(),
            );
            r.add_point_light(sx, 1.2, sz, 3.5, lr, lg, lb, 1.6);
        }
    });
    compare_or_update("many_point_lights", w, h, &rgba);

    // The sixth frame is unchanged apart from per-frame/view values. Forty
    // repeated light setters must not enqueue forty copies of the full block.
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("runtime path telemetry is valid JSON");
    let uploads = &paths["steady_state_uploads"]["lighting"];
    let writes = uploads["write_count"].as_u64().expect("write count");
    let bytes = uploads["byte_count"].as_u64().expect("byte count");
    let full_bytes = uploads["full_buffer_bytes"].as_u64().expect("buffer size");
    assert!(
        writes <= 3 && bytes <= 512 && bytes < full_bytes,
        "steady point-light frame uploaded too much lighting data: {uploads}"
    );

    let bind_groups = &paths["steady_state_resources"]["bind_group_creations"];
    let total = bind_groups["total"].as_u64().expect("bind-group total");
    let sites = bind_groups["sites"]
        .as_object()
        .expect("named bind-group creation sites");
    let named_total: u64 = sites
        .values()
        .map(|count| count.as_u64().expect("site count"))
        .sum();
    assert_eq!(
        total, named_total,
        "bind-group total must match named sites"
    );
    assert_eq!(
        sites["scene_compose"].as_u64(),
        Some(0),
        "steady scene compose must reuse its SSR-source-specific bind group"
    );
    assert_eq!(
        sites["final_composite"].as_u64(),
        Some(0),
        "steady final composite must reuse its source/exposure-specific bind group"
    );
    assert_eq!(
        sites["ssr_temporal"].as_u64(),
        Some(0),
        "steady SSR temporal must reuse its previous-history-specific bind group"
    );
    assert_eq!(
        sites["taa"].as_u64(),
        Some(0),
        "steady ordinary TAA must reuse its previous-history-specific bind group"
    );
    let resources = &paths["steady_state_resources"];
    assert_eq!(
        resources["graph_compiles"].as_u64(),
        Some(0),
        "stable topology must not compile after warm-up"
    );
    assert_eq!(
        resources["pipeline_creations"]["first_use"].as_u64(),
        Some(0),
        "warmed frame must not create a first-use pipeline"
    );
    assert_eq!(
        resources["command_encoder_creations"]["total"].as_u64(),
        Some(1),
        "steady rendering must use one submission encoder"
    );
    assert_eq!(
        resources["transient_physical_creations"]["textures"].as_u64(),
        Some(0),
        "stable graph must not allocate physical textures after warm-up"
    );
    assert_eq!(
        resources["transient_physical_creations"]["buffers"].as_u64(),
        Some(0),
        "stable graph must not allocate physical buffers after warm-up"
    );
}

#[test]
fn steady_half_resolution_upscale_reuses_its_bind_group() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_render_scale(0.5);
    eng.renderer.set_taa_enabled(false);
    let (_, _, rgba) = render(&mut eng, 4, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 20.0, 255.0);
        r.begin_mode_3d(3.0, 2.5, 5.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.draw_cube(0.0, 0.75, 0.0, 1.5, 1.5, 1.5, 220.0, 90.0, 35.0, 255.0);
    });
    assert!(
        rgba.chunks_exact(4)
            .any(|pixel| pixel[0] != 8 || pixel[1] != 12 || pixel[2] != 20),
        "half-resolution upscale frame did not render scene geometry"
    );
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("upscale telemetry is valid JSON");
    assert_eq!(
        paths["steady_state_resources"]["bind_group_creations"]["sites"]["upscale"].as_u64(),
        Some(0),
        "warmed upscale path must reuse its persistent bind group"
    );
}

#[test]
fn steady_depth_of_field_reuses_its_history_specific_bind_group() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_taa_enabled(true);
    eng.renderer.set_dof_enabled(true);
    eng.renderer.set_dof_focus_distance(4.0);
    eng.renderer.set_dof_aperture(0.04);
    let (_, _, rgba) = render(&mut eng, 5, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 20.0, 255.0);
        r.begin_mode_3d(3.0, 2.5, 5.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.draw_cube(0.0, 0.75, 0.0, 1.5, 1.5, 1.5, 220.0, 90.0, 35.0, 255.0);
    });
    assert!(
        rgba.chunks_exact(4)
            .any(|pixel| pixel[0] != 8 || pixel[1] != 12 || pixel[2] != 20),
        "depth-of-field frame did not render scene geometry"
    );
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("depth-of-field telemetry is valid JSON");
    assert_eq!(
        paths["steady_state_resources"]["bind_group_creations"]["sites"]["depth_of_field"].as_u64(),
        Some(0),
        "warmed depth of field must reuse its TAA-history-specific bind group"
    );
}

#[test]
fn steady_optional_postfx_chain_reuses_every_source_specific_bind_group() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    r.set_taa_enabled(true);
    r.set_dof_enabled(true);
    r.set_dof_focus_distance(4.0);
    r.set_dof_aperture(0.04);
    r.set_motion_blur_enabled(true);
    r.set_motion_blur_strength(0.75);
    r.set_sss_enabled(true);
    r.set_sss_strength(0.4);
    r.set_cas_strength(0.35);
    r.set_auto_exposure(true);

    let (_, _, rgba) = render(&mut eng, 5, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 20.0, 255.0);
        r.begin_mode_3d(3.0, 2.5, 5.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.draw_cube(0.0, 0.75, 0.0, 1.5, 1.5, 1.5, 220.0, 90.0, 35.0, 255.0);
    });
    assert!(
        rgba.chunks_exact(4)
            .any(|pixel| pixel[0] != 8 || pixel[1] != 12 || pixel[2] != 20),
        "optional post-FX frame did not render scene geometry"
    );
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("optional post-FX telemetry is valid JSON");
    let sites = &paths["steady_state_resources"]["bind_group_creations"]["sites"];
    for site in [
        "depth_of_field",
        "motion_blur",
        "subsurface_scattering",
        "contrast_adaptive_sharpen",
        "auto_exposure",
    ] {
        assert_eq!(
            sites[site].as_u64(),
            Some(0),
            "warmed optional post-FX site {site} must reuse its source-specific bind group"
        );
    }
}

#[test]
fn steady_custom_post_pass_stack_reuses_parity_specific_bind_groups() {
    const COPY_PASS: &str = r#"
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(scene_color_tex, scene_color_samp, uv);
}
"#;
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let draw = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 20.0, 255.0);
        r.begin_mode_3d(3.0, 2.5, 5.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.draw_cube(0.0, 0.75, 0.0, 1.5, 1.5, 1.5, 220.0, 90.0, 35.0, 255.0);
    };
    eng.begin_frame();
    eng.renderer
        .add_post_pass(COPY_PASS)
        .expect("first copy post pass compiles");
    eng.renderer
        .add_post_pass(COPY_PASS)
        .expect("second copy post pass compiles");
    draw(&mut eng);
    eng.end_frame();
    let first_use: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("first-use pipeline telemetry is valid JSON");
    assert_eq!(
        first_use["steady_state_resources"]["pipeline_creations"]["first_use"].as_u64(),
        Some(2),
        "two post-pass pipeline compilations must be measured in their creation frame"
    );
    let (_, _, rgba) = render(&mut eng, 4, draw);
    assert!(
        rgba.chunks_exact(4)
            .any(|pixel| pixel[0] != 8 || pixel[1] != 12 || pixel[2] != 20),
        "custom post-pass stack did not preserve scene geometry"
    );
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("custom post-pass telemetry is valid JSON");
    assert_eq!(
        paths["steady_state_resources"]["bind_group_creations"]["sites"]["custom_post_pass"]
            .as_u64(),
        Some(0),
        "warmed custom post-pass stack must reuse parity-specific bind groups"
    );
    assert_eq!(
        paths["steady_state_resources"]["pipeline_creations"]["first_use"].as_u64(),
        Some(0),
        "warmed custom post-pass stack must not recreate pipelines"
    );

    eng.renderer.resize(320, 192, 320, 192);
    let _ = render(&mut eng, 3, draw);
    let resized_paths: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
            .expect("resized custom post-pass telemetry is valid JSON");
    assert_eq!(
        resized_paths["steady_state_resources"]["bind_group_creations"]["sites"]
            ["custom_post_pass"]
            .as_u64(),
        Some(0),
        "post-pass bindings must rebuild after resize then return to zero churn"
    );
    assert_eq!(
        resized_paths["steady_state_resources"]["pipeline_creations"]["first_use"].as_u64(),
        Some(0),
        "resize must retain custom pipelines"
    );
}
