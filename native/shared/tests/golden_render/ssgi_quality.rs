use super::super::*;

#[test]
fn two_sided_mesh_cards_do_not_light_the_hidden_face_with_the_front_normal() {
    let _guard = lock_rt_goldens();
    let Some((mut eng, _adapter)) = try_engine_rt().unwrap_or_else(|error| {
        panic!("two-sided Mesh Card hardware setup failed: {error}");
    }) else {
        skip_rt_golden(
            "two_sided_mesh_cards_do_not_light_the_hidden_face_with_the_front_normal",
            "adapter does not expose experimental ray query",
        );
        return;
    };

    let renderer = &mut eng.renderer;
    renderer.set_taa_enabled(false);
    renderer.set_ssao_enabled(false);
    renderer.set_ssr_enabled(false);
    renderer.set_ssgi_enabled(true);
    renderer.set_bloom_enabled(false);
    renderer.set_auto_exposure(false);
    renderer.set_shadows_enabled(false);

    // A zero-thickness, two-sided red sheet with one authored +Y normal is
    // the focused form of Bistro's fabric awning. The +Y card must receive
    // overhead sun, while the -Y card represents its visible back face and
    // must orient that normal downward before coherent relighting.
    let vertex = |position| Vertex3D {
        position,
        normal: [0.0, 1.0, 0.0],
        color: [1.0; 4],
        uv: [0.0; 2],
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    let sheet = eng.scene.create_node();
    eng.scene.update_geometry(
        sheet,
        vec![
            vertex([-1.0, 0.0, -1.0]),
            vertex([1.0, 0.0, -1.0]),
            vertex([1.0, 0.0, 1.0]),
            vertex([-1.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2, 0, 2, 3],
    );
    eng.scene.set_material_color(sheet, 1.0, 0.04, 0.02, 1.0);
    eng.scene
        .set_material_gltf_alpha(sheet, MaterialAlphaMode::Opaque, 0.0, true);

    let draw = |eng: &mut EngineState| {
        let renderer = &mut eng.renderer;
        renderer.set_clear_color(5.0, 7.0, 10.0, 255.0);
        renderer.begin_mode_3d(0.0, 2.5, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 52.0, 0.0);
        renderer.set_ambient_light(255.0, 255.0, 255.0, 0.0);
        renderer.set_directional_light(0.0, 1.0, 0.0, 255.0, 255.0, 255.0, 2.0);
    };
    let (_, _, output) = render(&mut eng, 8, draw);
    assert!(
        output.chunks_exact(4).any(|pixel| pixel[0..3] != [0, 0, 0]),
        "two-sided Mesh Card qualification frame was blank"
    );

    let first_slot = eng
        .scene
        .nodes
        .get(sheet)
        .and_then(|node| node.card_first_slot)
        .expect("test sheet did not receive Mesh Card slots");
    let slots = eng.renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("two_sided_card_slot_uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    eng.renderer.queue.write_buffer(
        &slots,
        0,
        bytemuck::cast_slice(&[first_slot + 2, first_slot + 3, 0_u32, 0_u32]),
    );
    let samples = eng.renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("two_sided_card_radiance_samples"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = eng.renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("two_sided_card_radiance_staging"),
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let shader = eng
        .renderer
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("two_sided_card_radiance_readback_shader"),
            source: wgpu::ShaderSource::Wgsl(
                "
struct Slots { values: vec4<u32> };
@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var<uniform> slots: Slots;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>, 2>;

fn slot_center(slot: u32) -> vec2<i32> {
    return vec2<i32>(
        i32((slot % 64u) * 16u + 8u),
        i32((slot / 64u) * 16u + 8u),
    );
}

@compute @workgroup_size(2, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 2u) { return; }
    output[gid.x] = textureLoad(atlas, slot_center(slots.values[gid.x]), 0);
}
"
                .into(),
            ),
        });
    let pipeline = eng
        .renderer
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("two_sided_card_radiance_readback_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = eng
        .renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("two_sided_card_radiance_readback_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &eng.renderer.mesh_card_radiance_view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: samples.as_entire_binding(),
                },
            ],
        });
    let mut encoder = eng
        .renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("two_sided_card_radiance_readback_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("two_sided_card_radiance_readback_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&samples, 0, &staging, 0, 32);
    eng.renderer.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = eng.renderer.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .expect("card sample map callback")
        .expect("card sample map");
    let data = slice.get_mapped_range();
    let channel = |sample: usize, component: usize| {
        let offset = sample * 16 + component * 4;
        f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
    };
    let front = [channel(0, 0), channel(0, 1), channel(0, 2)];
    let back = [channel(1, 0), channel(1, 1), channel(1, 2)];
    drop(data);
    staging.unmap();

    assert!(
        front[0] > 0.1,
        "sun-facing +Y card did not retain direct red radiance: {front:?}"
    );
    assert!(
        back.iter().all(|value| value.abs() < 0.005),
        "hidden -Y card inherited the sun-facing normal: front={front:?}, back={back:?}"
    );
}

#[test]
fn hardware_ssgi_skips_single_sided_hidden_faces_but_keeps_two_sided_materials() {
    let _guard = lock_rt_goldens();

    let capture = |double_sided: bool| -> Vec<u8> {
        let Some((mut eng, _adapter)) = try_engine_rt().unwrap_or_else(|error| {
            panic!("single-sided SSGI hardware setup failed: {error}");
        }) else {
            return Vec::new();
        };
        let renderer = &mut eng.renderer;
        renderer.set_taa_enabled(false);
        renderer.set_ssao_enabled(false);
        renderer.set_ssr_enabled(false);
        renderer.set_ssgi_enabled(true);
        renderer.set_bloom_enabled(false);
        renderer.set_auto_exposure(false);
        renderer.set_shadows_enabled(false);

        let vertex = |position| Vertex3D {
            position,
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 4],
            uv: [0.0; 2],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [1.0, 0.0, 0.0, 1.0],
        };
        let plane = |y: f32, extent: f32| {
            (
                vec![
                    vertex([-extent, y, -extent]),
                    vertex([-extent, y, extent]),
                    vertex([extent, y, extent]),
                    vertex([extent, y, -extent]),
                ],
                vec![0, 1, 2, 0, 2, 3],
            )
        };

        let receiver = eng.scene.create_node();
        let (receiver_vertices, receiver_indices) = plane(0.0, 6.0);
        eng.scene
            .update_geometry(receiver, receiver_vertices, receiver_indices);
        eng.scene.set_material_color(receiver, 0.8, 0.8, 0.8, 1.0);
        eng.scene
            .set_material_gltf_alpha(receiver, MaterialAlphaMode::Opaque, 0.0, false);

        // The emitter's authored winding faces +Y. A floor probe below it sees
        // only the hidden -Y face. A single-sided material must therefore be
        // skipped exactly as it is by the forward pass, while the two-sided
        // control deliberately retains the emissive back face.
        let emitter = eng.scene.create_node();
        let (emitter_vertices, emitter_indices) = plane(1.0, 6.0);
        eng.scene
            .update_geometry(emitter, emitter_vertices, emitter_indices);
        eng.scene.set_material_color(emitter, 1.0, 0.0, 0.0, 1.0);
        eng.scene
            .set_material_emissive_factor(emitter, 1.0, 0.0, 0.0);
        eng.scene
            .set_material_gltf_alpha(emitter, MaterialAlphaMode::Opaque, 0.0, double_sided);

        render(&mut eng, 12, |eng| {
            let renderer = &mut eng.renderer;
            renderer.set_clear_color(0.0, 0.0, 0.0, 255.0);
            renderer.begin_mode_3d(0.0, 0.55, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 52.0, 0.0);
            renderer.set_ambient_light(255.0, 255.0, 255.0, 0.0);
            renderer.set_directional_light(0.0, 1.0, 0.0, 255.0, 255.0, 255.0, 0.0);
        })
        .2
    };

    let single_sided = capture(false);
    if single_sided.is_empty() {
        skip_rt_golden(
            "hardware_ssgi_skips_single_sided_hidden_faces_but_keeps_two_sided_materials",
            "adapter does not expose experimental ray query",
        );
        return;
    }
    let double_sided = capture(true);
    assert_eq!(single_sided.len(), double_sided.len());
    let mut affected = 0usize;
    let mut red_delta = 0i64;
    for (single, double) in single_sided
        .chunks_exact(4)
        .zip(double_sided.chunks_exact(4))
    {
        let delta = i64::from(double[0]) - i64::from(single[0]);
        if delta > 2 {
            affected += 1;
            red_delta += delta;
        }
    }
    assert!(
        affected > 100 && red_delta > 2_000,
        "single- and two-sided GI produced no measurable hidden-face separation: \
         affected={affected}, red_delta={red_delta}"
    );
}

#[test]
fn wsrc_bake_double_wraps_all_octahedral_corners() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(true);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_shadows_enabled(false);

    let capture = |eng: &mut EngineState| {
        eng.begin_frame();
        let r = &mut eng.renderer;
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, 1.8);
        r.draw_cube(0.0, -0.1, 0.0, 12.0, 0.2, 12.0, 90.0, 96.0, 107.0, 255.0);
        eng.end_frame();
    };

    // Profile the fixed-resolution bake independently of its normal
    // once-per-cascade amortization. This is test-only state manipulation.
    eng.profiler.set_enabled(true);
    for _ in 0..120 {
        eng.renderer.wsrc_built = [false; 3];
        capture(&mut eng);
    }
    let bake_gpu_us = eng
        .profiler
        .snapshot()
        .into_iter()
        .find_map(|(label, _, gpu)| (label == "wsrc_bake_pass").then_some(gpu?))
        .unwrap_or(0.0);
    eng.profiler.set_enabled(false);

    // The loop above repeatedly baked cascade zero. Let the established
    // amortizer finish cascades one and two before reading the atlas.
    capture(&mut eng);
    capture(&mut eng);
    assert_eq!(eng.renderer.wsrc_built, [true; 3]);

    let shader = eng
        .renderer
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wsrc_corner_readback_shader"),
            source: wgpu::ShaderSource::Wgsl(
                "
@group(0) @binding(0) var atlas: texture_3d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> samples: array<vec4<f32>, 16>;

@compute @workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 8u) { return; }
    let coords = array<vec2<i32>, 8>(
        vec2<i32>(0, 0), vec2<i32>(8, 8),
        vec2<i32>(0, 9), vec2<i32>(8, 1),
        vec2<i32>(9, 0), vec2<i32>(1, 8),
        vec2<i32>(9, 9), vec2<i32>(1, 1),
    );
    samples[gid.x] = textureLoad(atlas, vec3<i32>(coords[gid.x], 0), 0);
    if (gid.x < 4u) {
        let bases = array<vec2<i32>, 4>(
            vec2<i32>(0, 0), vec2<i32>(8, 0),
            vec2<i32>(0, 8), vec2<i32>(8, 8),
        );
        let sample_texels = array<vec2<f32>, 4>(
            vec2<f32>(1.0, 1.0), vec2<f32>(9.0, 1.0),
            vec2<f32>(1.0, 9.0), vec2<f32>(9.0, 9.0),
        );
        let base = bases[gid.x];
        let filtered = textureSampleLevel(
            atlas,
            atlas_sampler,
            vec3<f32>(sample_texels[gid.x] / 160.0, 0.5 / 48.0),
            0.0,
        );
        let expected = (
            textureLoad(atlas, vec3<i32>(base, 0), 0)
            + textureLoad(atlas, vec3<i32>(base + vec2<i32>(1, 0), 0), 0)
            + textureLoad(atlas, vec3<i32>(base + vec2<i32>(0, 1), 0), 0)
            + textureLoad(atlas, vec3<i32>(base + vec2<i32>(1, 1), 0), 0)
        ) * 0.25;
        samples[8u + gid.x * 2u] = filtered;
        samples[9u + gid.x * 2u] = expected;
    }
}
"
                .into(),
            ),
        });
    let pipeline = eng
        .renderer
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wsrc_corner_readback_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let output = eng.renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wsrc_corner_samples"),
        size: 16 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = eng.renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wsrc_corner_staging"),
        size: 16 * 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = eng
        .renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wsrc_corner_readback_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&eng.renderer.wsrc_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&eng.renderer.wsrc_atlas_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
    let mut encoder = eng
        .renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wsrc_corner_readback_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(8, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 16 * 16);
    eng.renderer.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    eng.renderer
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("WSRC corner readback poll failed");
    rx.recv()
        .expect("WSRC corner map sender dropped")
        .expect("WSRC corner map failed");
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped);
    for pair in 0..4 {
        let corner = &values[pair * 8..pair * 8 + 4];
        let wrapped = &values[pair * 8 + 4..pair * 8 + 8];
        assert!(corner.iter().all(|value| value.is_finite()));
        assert_eq!(
            corner, wrapped,
            "WSRC padded corner pair {pair} did not double-wrap"
        );
    }
    for corner in 0..4 {
        let filtered_start = (8 + corner * 2) * 4;
        let expected_start = filtered_start + 4;
        for channel in 0..4 {
            let filtered = values[filtered_start + channel];
            let expected = values[expected_start + channel];
            assert!(
                (filtered - expected).abs() <= 0.0001,
                "WSRC corner {corner} channel {channel} filter weight mismatch: \
                 filtered={filtered}, expected={expected}"
            );
        }
    }
    eprintln!("wsrc-corner-wrap bake_gpu_us={bake_gpu_us:.3}");
    drop(mapped);
    staging.unmap();
}

#[test]
fn ssgi_hiz_immediate_scene_produces_finite_indirect_radiance() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    if std::env::var_os("BLOOM_SSGI_PROFILE_HD").is_some() {
        r.resize(1280, 720, 1280, 720);
    }
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(true);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_shadows_enabled(false);

    let draw = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, 1.8);
        r.draw_cube(0.0, -0.1, 0.0, 12.0, 0.2, 12.0, 90.0, 96.0, 107.0, 255.0);
        r.draw_cube(0.0, 2.0, -3.0, 8.0, 4.0, 0.2, 230.0, 166.0, 31.0, 255.0);
        r.draw_cube(-1.1, 1.0, 0.0, 1.8, 2.0, 1.8, 230.0, 45.0, 25.0, 255.0);
        r.draw_sphere(1.1, 0.9, -0.8, 0.9, 30.0, 110.0, 240.0, 255.0);
    };
    let capture = |eng: &mut EngineState| {
        eng.begin_frame();
        draw(eng);
        eng.end_frame();
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..24 {
        capture(&mut eng);
    }
    eng.profiler.set_enabled(true);
    for _ in 0..120 {
        capture(&mut eng);
    }
    let probe_gpu_us = eng
        .profiler
        .snapshot()
        .into_iter()
        .filter_map(|(label, _, gpu)| label.starts_with("probe_").then_some((label, gpu?)))
        .collect::<Vec<_>>();
    let probe_total_gpu_us = probe_gpu_us.iter().map(|(_, gpu)| gpu).sum::<f64>();
    eng.profiler.set_enabled(false);
    let directory = std::env::temp_dir().join(format!("bloom-ssgi-hiz-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng);

    // The SSGI-owned low-discrepancy sequence must advance the raw eight-ray
    // realization while the integrated, reprojected result remains stable.
    // Four times as many receiver probes distribute the former 32-ray spatial
    // budget; a single probe therefore has slightly more phase variance even
    // though the resolved neighborhood retains the same total ray density.
    // Directional history is never retained, so this cannot reproduce the old
    // bug where a previous lane was reinterpreted as a different direction.
    let next_directory =
        std::env::temp_dir().join(format!("bloom-ssgi-hiz-next-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&next_directory);
    eng.renderer.pending_quality_capture_dir = Some(next_directory.to_string_lossy().into_owned());
    capture(&mut eng);
    let current_radiance = std::fs::read(directory.join("ssgi-current-radiance.png"))
        .expect("Hi-Z SSGI capture did not emit current radiance");
    let next_current_radiance = std::fs::read(next_directory.join("ssgi-current-radiance.png"))
        .expect("next Hi-Z SSGI capture did not emit current radiance");
    assert_ne!(
        current_radiance, next_current_radiance,
        "SSGI angular sequence did not advance the current ray realization"
    );
    let settled_ssgi = image::open(directory.join("ssgi.png"))
        .expect("Hi-Z SSGI capture did not emit settled resolved radiance")
        .to_rgba8();
    let next_settled_ssgi = image::open(next_directory.join("ssgi.png"))
        .expect("next Hi-Z SSGI capture did not emit settled resolved radiance")
        .to_rgba8();
    let settled_metrics = calculate_diff_metrics(
        settled_ssgi.as_raw(),
        next_settled_ssgi.as_raw(),
        settled_ssgi.width(),
        settled_ssgi.height(),
    );
    assert!(
        settled_metrics.ssim >= 0.9985,
        "temporally rotated rays destabilized settled SSGI: {settled_metrics:?}"
    );

    let confidence = image::open(directory.join("ssgi-temporal-confidence.png"))
        .expect("Hi-Z SSGI capture did not emit temporal confidence")
        .to_rgb8();
    let current = confidence.pixels().filter(|pixel| pixel[1] > 0).count();
    let retained = confidence.pixels().filter(|pixel| pixel[2] > 16).count();
    let metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("ssgi.metrics.json"))
            .expect("Hi-Z SSGI capture did not emit resolved HDR metrics"),
    )
    .unwrap();
    let non_finite = metrics["non_finite_pixels"].as_u64().unwrap();
    let max_luminance = metrics["max_luminance"].as_f64().unwrap();
    let paths = eng.renderer.quality_runtime_paths_json();
    eprintln!(
        "hiz-corpus ssgi current={current} retained={retained} non_finite={non_finite} \
         max_luma={max_luminance:.6} probe_total_gpu_us={probe_total_gpu_us:.3} \
         probe_gpu_us={probe_gpu_us:?}"
    );
    assert!(
        paths.contains("\"ssgi_trace_backend\":\"hiz-screen\""),
        "immediate-only SSGI scene escaped the Hi-Z backend: {paths}"
    );
    assert_eq!(non_finite, 0, "Hi-Z SSGI emitted non-finite radiance");
    assert!(
        current >= 100 && retained >= 100 && max_luminance > 0.0001,
        "Hi-Z SSGI produced no usable indirect radiance: current={current}, \
         retained={retained}, max_luma={max_luminance:.6}"
    );

    // TAA jitters the primary projection even at a stationary camera. SSGI's
    // world-owned angular phase must respond continuously to that subpixel
    // displacement; a quantized world-cell hash can replace all probe rays at
    // once and turn a nearby colored source into a flickering strip.
    eng.renderer.set_taa_enabled(true);
    eng.renderer.reset_temporal_history();
    for _ in 0..16 {
        capture(&mut eng);
    }
    let taa_directory =
        std::env::temp_dir().join(format!("bloom-ssgi-hiz-taa-{}", std::process::id()));
    let taa_next_directory =
        std::env::temp_dir().join(format!("bloom-ssgi-hiz-taa-next-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&taa_directory);
    let _ = std::fs::remove_dir_all(&taa_next_directory);
    eng.renderer.pending_quality_capture_dir = Some(taa_directory.to_string_lossy().into_owned());
    capture(&mut eng);
    eng.renderer.pending_quality_capture_dir =
        Some(taa_next_directory.to_string_lossy().into_owned());
    capture(&mut eng);
    let taa_ssgi = image::open(taa_directory.join("ssgi.png"))
        .expect("TAA SSGI capture did not emit resolved radiance")
        .to_rgba8();
    let taa_next_ssgi = image::open(taa_next_directory.join("ssgi.png"))
        .expect("next TAA SSGI capture did not emit resolved radiance")
        .to_rgba8();
    let taa_metrics = calculate_diff_metrics(
        taa_ssgi.as_raw(),
        taa_next_ssgi.as_raw(),
        taa_ssgi.width(),
        taa_ssgi.height(),
    );
    eprintln!("hiz-corpus stationary TAA SSGI={taa_metrics:?}");
    // This capture is the raw half-resolution SSGI buffer before TAA. Two
    // Halton phases correctly move geometric silhouettes by a subpixel, so a
    // near-identity whole-image SSIM would reject the intended sampling
    // aperture rather than GI instability. Bound the low-frequency color
    // change, edge change, and affected-pixel footprint; the non-jittered
    // settled-radiance comparison above remains the strict 0.999 angular-
    // stability gate.
    assert!(
        taa_metrics.mean_rgb <= 0.75
            && taa_metrics.mean_edge_delta <= 0.004
            && taa_metrics.outlier_pixel_fraction <= 0.005
            && taa_metrics.ssim >= 0.975,
        "stationary TAA jitter destabilized resolved SSGI beyond its subpixel footprint: \
         {taa_metrics:?}"
    );

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!(
            "kept Hi-Z SSGI diagnostics at {directory:?}, {next_directory:?}, \
             {taa_directory:?}, and {taa_next_directory:?}"
        );
    } else {
        let _ = std::fs::remove_dir_all(directory);
        let _ = std::fs::remove_dir_all(next_directory);
        let _ = std::fs::remove_dir_all(taa_directory);
        let _ = std::fs::remove_dir_all(taa_next_directory);
    }
}

#[test]
fn ssgi_capture_exposes_probe_history_without_normal_frame_resources() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(true);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_shadows_enabled(false);
    super::transformed_box(
        &mut eng,
        [0.0, -0.1, 0.0],
        [12.0, 0.2, 12.0],
        [0.35, 0.38, 0.42, 1.0],
        0.8,
        0.0,
        [0.0; 3],
    );
    super::transformed_box(
        &mut eng,
        [0.0, 2.0, -3.0],
        [8.0, 4.0, 0.2],
        [0.9, 0.65, 0.12, 1.0],
        0.7,
        0.0,
        [2.5, 1.2, 0.15],
    );

    let draw = |eng: &mut EngineState, camera_x: f32| {
        let r = &mut eng.renderer;
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(camera_x, 3.0, 6.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, 1.8);
        r.draw_cube(-1.1, 1.0, 0.0, 1.8, 2.0, 1.8, 230.0, 45.0, 25.0, 255.0);
        r.draw_sphere(1.1, 0.9, -0.8, 0.9, 30.0, 110.0, 240.0, 255.0);
    };
    let capture = |eng: &mut EngineState, camera_x: f32| {
        eng.begin_frame();
        draw(eng, camera_x);
        eng.end_frame();
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..24 {
        capture(&mut eng, 4.0);
    }
    let directory =
        std::env::temp_dir().join(format!("bloom-ssgi-diagnostics-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng, 4.0);

    let reasons = image::open(directory.join("ssgi-rejection-reason.png"))
        .expect("SSGI capture did not emit probe-domain temporal reasons")
        .to_rgb8();
    let accepted = reasons
        .pixels()
        .filter(|pixel| pixel[0] < 40 && pixel[1] > 140 && pixel[2] < 60)
        .count();
    let refreshed = reasons
        .pixels()
        .filter(|pixel| pixel[0] > 220 && pixel[1] < 40 && pixel[2] > 160)
        .count();
    let confidence = image::open(directory.join("ssgi-temporal-confidence.png"))
        .expect("SSGI capture did not emit probe-domain temporal confidence")
        .to_rgb8();
    let retained = confidence.pixels().filter(|pixel| pixel[2] > 16).count();
    let current = confidence.pixels().filter(|pixel| pixel[1] > 0).count();
    let resolved = image::open(directory.join("ssgi.png"))
        .expect("SSGI capture did not emit resolved indirect radiance")
        .to_rgb8();
    let resolve_support = image::open(directory.join("ssgi-resolve-support.png"))
        .expect("SSGI capture did not emit screen-space resolve support")
        .to_rgb8();
    let resolve_geometry = image::open(directory.join("ssgi-resolve-geometry.png"))
        .expect("SSGI capture did not emit screen-space resolve geometry")
        .to_rgb8();
    let resolve_plane_ratios = image::open(directory.join("ssgi-resolve-plane-ratios.png"))
        .expect("SSGI capture did not emit screen-space resolve plane ratios")
        .to_rgb8();
    let resolve_plane_ratio_w = image::open(directory.join("ssgi-resolve-plane-ratio-w.png"))
        .expect("SSGI capture did not emit the fourth screen-space resolve plane ratio")
        .to_rgb8();
    let metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("ssgi.metrics.json"))
            .expect("SSGI capture did not emit resolved HDR metrics"),
    )
    .unwrap();
    let non_finite = metrics["non_finite_pixels"].as_u64().unwrap();
    let max_luminance = metrics["max_luminance"].as_f64().unwrap();
    eprintln!(
        "temporal-corpus ssgi-probes accepted={accepted} refreshed={refreshed} current={current} \
         retained={retained} non_finite={non_finite} max_luma={max_luminance:.4} total={}",
        reasons.width() * reasons.height()
    );
    assert!(
        accepted >= 100 && current >= 100 && retained >= 100,
        "settled SSGI probes exposed no current radiance or retained history"
    );
    assert_eq!(non_finite, 0, "SSGI resolve emitted non-finite radiance");
    assert!(
        max_luminance > 0.0001,
        "SSGI probe resolve produced no indirect radiance"
    );
    assert_eq!(
        reasons.dimensions(),
        confidence.dimensions(),
        "SSGI reason/confidence atlases describe different probe domains"
    );
    assert_eq!(
        resolved.dimensions(),
        resolve_support.dimensions(),
        "SSGI resolve/support captures describe different screen domains"
    );
    assert_eq!(
        resolved.dimensions(),
        resolve_geometry.dimensions(),
        "SSGI resolve/geometry captures describe different screen domains"
    );
    assert_eq!(
        resolved.dimensions(),
        resolve_plane_ratios.dimensions(),
        "SSGI resolve/plane-ratio captures describe different screen domains"
    );
    assert_eq!(
        resolved.dimensions(),
        resolve_plane_ratio_w.dimensions(),
        "SSGI resolve/fourth-plane-ratio captures describe different screen domains"
    );
    // Exercise the exact grazing-receiver condition behind the Bistro floor
    // strips: every surrounding probe belongs to the receiver plane, while
    // the narrower strict kernel has no support at this screen-grid phase.
    // The production shader's structural test requires this class to use the
    // broad path, so the two tests together prevent a hard strict/fallback
    // switch from silently returning.
    let coherent_strict_gaps = resolve_support
        .pixels()
        .filter(|pixel| pixel[0] < 16 && pixel[1] > 240 && pixel[2] < 8)
        .count();
    eprintln!("temporal-corpus ssgi-resolve coherent_strict_gaps={coherent_strict_gaps}");
    assert!(
        coherent_strict_gaps >= 16,
        "SSGI resolve corpus did not exercise a complete coherent footprint with zero strict support"
    );

    // The coarse probe lattice is screen tiled, so camera motion must consume
    // geometry-validated per-pixel resolve history. Without this acceptance
    // path, grazing floor rows remain fixed to the display even though a
    // stationary image is smooth.
    let motion_directory =
        std::env::temp_dir().join(format!("bloom-ssgi-resolve-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&motion_directory);
    eng.renderer.pending_quality_capture_dir =
        Some(motion_directory.to_string_lossy().into_owned());
    capture(&mut eng, 4.02);
    let resolve_motion = image::open(motion_directory.join("ssgi-resolve-plane-ratio-w.png"))
        .expect("moving SSGI capture did not emit resolve-history validation")
        .to_rgb8();
    let resolve_history_accepted = resolve_motion
        .pixels()
        .filter(|pixel| pixel[1] > 240)
        .count();
    eprintln!("temporal-corpus ssgi-resolve moving_history_accepted={resolve_history_accepted}");
    assert!(
        resolve_history_accepted >= 100,
        "camera motion accepted no geometry-valid SSGI resolve history"
    );

    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"ray_scene_preparation\":\"ssgi\""));
    assert!(paths.contains("\"ssgi_diagnostic_persistent_bytes\":0"));
    assert!(paths.contains("\"ssgi_diagnostic_capture_passes\":3"));
    assert!(paths.contains("\"ssgi_diagnostic_resources_live\":false"));
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept SSGI diagnostics at {directory:?} and {motion_directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
        let _ = std::fs::remove_dir_all(motion_directory);
    }
}

#[test]
fn ssgi_rotation_refreshes_changed_and_reprojected_probe_surfaces() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let r = &mut eng.renderer;
    r.set_taa_enabled(false);
    r.set_ssao_enabled(false);
    r.set_ssr_enabled(false);
    r.set_ssgi_enabled(true);
    r.set_bloom_enabled(false);
    r.set_auto_exposure(false);
    r.set_shadows_enabled(false);

    let capture = |eng: &mut EngineState, camera_x: f32, target_x: f32| {
        eng.begin_frame();
        let r = &mut eng.renderer;
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(
            camera_x, 3.0, 6.0, target_x, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0,
        );
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, 1.8);
        r.draw_cube(0.0, -0.1, 0.0, 12.0, 0.2, 12.0, 90.0, 96.0, 107.0, 255.0);
        r.draw_cube(0.0, 2.0, -3.0, 8.0, 4.0, 0.2, 230.0, 166.0, 31.0, 255.0);
        r.draw_cube(-1.1, 1.0, 0.0, 1.8, 2.0, 1.8, 230.0, 45.0, 25.0, 255.0);
        r.draw_sphere(1.1, 0.9, -0.8, 0.9, 30.0, 110.0, 240.0, 255.0);
        eng.end_frame();
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..24 {
        capture(&mut eng, 4.0, 0.0);
    }

    let directory = std::env::temp_dir().join(format!("bloom-ssgi-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng, 4.0, 4.0);

    let reasons = image::open(directory.join("ssgi-rejection-reason.png"))
        .expect("moving SSGI capture did not emit probe-domain temporal reasons")
        .to_rgb8();
    let changed_surface = reasons
        .pixels()
        .filter(|pixel| pixel[0] < 30 && pixel[1] > 170 && pixel[2] > 220)
        .count();
    let motion_refreshed = reasons
        .pixels()
        .filter(|pixel| pixel[0] < 30 && pixel[1] > 40 && pixel[1] < 100 && pixel[2] > 220)
        .count();
    let radiance_refreshed = reasons
        .pixels()
        .filter(|pixel| pixel[0] > 220 && pixel[1] < 40 && pixel[2] > 160)
        .count();
    let offscreen_refreshed = reasons
        .pixels()
        .filter(|pixel| pixel[0] > 220 && pixel[1] < 60 && pixel[2] < 30)
        .count();
    let total = u64::from(reasons.width()) * u64::from(reasons.height());
    eprintln!(
        "temporal-corpus ssgi-motion changed_surface={changed_surface} \
         motion_refreshed={motion_refreshed} radiance_refreshed={radiance_refreshed} \
         offscreen_refreshed={offscreen_refreshed} total={total}"
    );
    assert!(
        u64::try_from(
            changed_surface + motion_refreshed + radiance_refreshed + offscreen_refreshed,
        )
        .unwrap()
            >= total * 45 / 100,
        "camera motion neither rejected changed probe surfaces nor refreshed \
         reprojected history"
    );

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept moving SSGI diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

fn attach_model_placements(eng: &mut EngineState, source_path: &Path) {
    let data = std::fs::read(source_path).expect("read detailed Bistro glTF");
    let model_handle =
        eng.models
            .load_model_with_textures_from_source_path(&data, source_path, &mut eng.renderer);
    assert!(model_handle > 0.0, "load detailed Bistro glTF");
    let placements = {
        let model = eng.models.get(model_handle).expect("loaded Bistro model");
        model
            .meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| {
                (
                    std::sync::Arc::clone(mesh),
                    model.mesh_transform(index),
                    model.mesh_cast_shadow(index),
                )
            })
            .collect::<Vec<_>>()
    };
    eprintln!("detailed-bistro placements={}", placements.len());
    assert_eq!(
        placements.len(),
        1_176,
        "unexpected Bistro placement corpus"
    );

    for (mesh, source_transform, cast_shadow) in placements {
        let node = eng.scene.create_node();
        let mut transmission = mesh.transmission;
        let axis_length = |column: usize| {
            let x = source_transform[column][0];
            let y = source_transform[column][1];
            let z = source_transform[column][2];
            (x * x + y * y + z * z).sqrt()
        };
        transmission.baked_thickness_scale *=
            (axis_length(0) + axis_length(1) + axis_length(2)) / 3.0;

        eng.scene.update_shared_model_geometry(
            node,
            std::sync::Arc::clone(&mesh),
            source_transform,
        );
        eng.scene.set_cast_shadow(node, cast_shadow);
        if let Some(texture) = mesh.texture_idx {
            eng.scene.set_material_texture(node, texture);
        }
        if let Some(texture) = mesh.normal_texture_idx {
            eng.scene.set_material_normal_texture(node, texture);
        }
        if let Some(texture) = mesh.metallic_roughness_texture_idx {
            eng.scene
                .set_material_metallic_roughness_texture(node, texture);
        }
        eng.scene
            .set_material_specular_glossiness_factor(node, mesh.specular_glossiness_factor);
        if let Some(texture) = mesh.emissive_texture_idx {
            eng.scene.set_material_emissive_texture(node, texture);
        }
        eng.scene.set_material_emissive_factor(
            node,
            mesh.emissive_factor[0],
            mesh.emissive_factor[1],
            mesh.emissive_factor[2],
        );
        eng.scene
            .set_material_pbr(node, mesh.roughness_factor, mesh.metallic_factor);
        eng.scene.set_material_gltf_alpha(
            node,
            mesh.alpha_mode,
            mesh.alpha_cutoff,
            mesh.double_sided,
        );
        eng.scene
            .set_material_alpha_coverage_mips(node, mesh.alpha_coverage_mips);
        eng.scene.set_material_transmission(node, transmission);
        eng.scene.set_material_layered_pbr(node, mesh.layered_pbr);
    }
}

fn configure(eng: &mut EngineState, shadows_enabled: bool, shadow_always_fresh: bool) {
    let renderer = &mut eng.renderer;
    renderer.apply_quality_preset(4);
    renderer.set_render_scale(1.0);
    renderer.set_taa_enabled(false);
    renderer.set_ssao_enabled(false);
    renderer.set_ssr_enabled(false);
    renderer.set_ssgi_enabled(true);
    renderer.set_ssgi_radius(2.3);
    renderer.set_ssgi_intensity(1.1);
    renderer.set_bloom_enabled(false);
    renderer.set_motion_blur_enabled(false);
    renderer.set_sss_enabled(false);
    renderer.set_sharpen_strength(0.0);
    renderer.set_auto_exposure(false);
    renderer.set_manual_exposure(1.2);
    renderer.set_shadows_enabled(shadows_enabled);
    renderer.shadow_map.always_fresh = shadow_always_fresh;
}

const BISTRO_YAW: f32 = -0.344;

fn draw(eng: &mut EngineState, camera_x: f32, camera_z: f32, yaw: f32) {
    let forward_x = -yaw.sin();
    let forward_z = -yaw.cos();
    let renderer = &mut eng.renderer;
    renderer.set_clear_color(5.0, 7.0, 12.0, 255.0);
    renderer.begin_mode_3d(
        camera_x,
        1.544,
        camera_z,
        camera_x + forward_x * 100.0,
        1.544,
        camera_z + forward_z * 100.0,
        0.0,
        1.0,
        0.0,
        60.0,
        0.0,
    );
    renderer.set_ambient_light(255.0, 245.0, 232.0, 0.06);
    renderer.set_directional_light(0.59732, 0.79653, -0.0935387, 255.0, 212.0, 177.0, 2.60);
}

fn frame(eng: &mut EngineState, camera_x: f32, camera_z: f32, yaw: f32) {
    eng.begin_frame();
    draw(eng, camera_x, camera_z, yaw);
    eng.end_frame();
}

#[test]
fn detailed_bistro_ssgi_is_camera_path_stable_at_same_endpoint() {
    let Some(scene_path) = std::env::var_os("BLOOM_BISTRO_SSGI_SCENE").map(PathBuf::from) else {
        eprintln!("skip: BLOOM_BISTRO_SSGI_SCENE is not set");
        return;
    };
    let _guard = lock_rt_goldens();
    let endpoint_settle_frames = std::env::var("BLOOM_BISTRO_SSGI_SETTLE_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let warmup_frames = std::env::var("BLOOM_BISTRO_SSGI_WARMUP_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        .max(1_176 + 108);
    let shadows_enabled =
        std::env::var_os("BLOOM_BISTRO_SSGI_SHADOWS").is_none_or(|value| value != "0");
    let shadow_always_fresh = std::env::var_os("BLOOM_BISTRO_SSGI_SHADOW_ALWAYS_FRESH").is_some();

    fn capture(
        scene_path: &Path,
        moved: bool,
        warmup_frames: usize,
        endpoint_settle_frames: usize,
        shadows_enabled: bool,
        shadow_always_fresh: bool,
        label: &str,
    ) -> (Vec<u8>, image::RgbaImage, String) {
        let Some((mut eng, _)) = try_engine_rt().expect("create ray-query Bistro engine") else {
            panic!("detailed Bistro SSGI qualification requires hardware ray query");
        };
        configure(&mut eng, shadows_enabled, shadow_always_fresh);
        attach_model_placements(&mut eng, scene_path);

        // Hardware admission is intentionally one BLAS per frame in
        // production. Run the camera path only after every detailed Bistro
        // placement is queryable; otherwise this oracle compares two mostly
        // empty ray scenes and cannot reproduce the delayed awning strip.
        let admission_frames = eng.scene.pending_blas_builds.len();
        let warmup_frames = warmup_frames.max(admission_frames + 108);

        const START_X: f32 = -3.2720;
        const START_Z: f32 = 7.2358;
        const END_X: f32 = START_X + 3.748170285;
        const END_Z: f32 = START_Z + 4.685212856;
        if moved {
            // Keep the two paths at the same global frame index. A long
            // warmup can cross the delayed mesh-card-light handoff while the
            // moved path still visits bright geometry before returning to the
            // exact same endpoint as the direct path.
            for _ in 0..(warmup_frames - 108) {
                frame(&mut eng, START_X, START_Z, BISTRO_YAW);
            }
            for step in 1..=30 {
                let t = step as f32 / 30.0;
                frame(
                    &mut eng,
                    START_X + (END_X - START_X) * t,
                    START_Z + (END_Z - START_Z) * t,
                    BISTRO_YAW,
                );
            }
            for step in 1..=30 {
                let t = step as f32 / 30.0;
                frame(
                    &mut eng,
                    END_X + (START_X - END_X) * t,
                    END_Z + (START_Z - END_Z) * t,
                    BISTRO_YAW,
                );
            }
            // Rotate away and return to the launch facade with no settle. This
            // is the exact interactive view where the delayed awning-colored
            // strip appears; the former oracle stopped at END and therefore
            // never qualified the reported surface.
            for step in 1..=24 {
                let t = step as f32 / 24.0;
                frame(&mut eng, START_X, START_Z, BISTRO_YAW + 0.45 * t);
            }
            for step in 1..=24 {
                let t = step as f32 / 24.0;
                frame(&mut eng, START_X, START_Z, BISTRO_YAW + 0.45 * (1.0 - t));
            }
            for _ in 0..endpoint_settle_frames {
                frame(&mut eng, START_X, START_Z, BISTRO_YAW);
            }
        } else {
            for _ in 0..(warmup_frames + endpoint_settle_frames) {
                frame(&mut eng, START_X, START_Z, BISTRO_YAW);
            }
        }

        let profile_frames = std::env::var("BLOOM_BISTRO_SSGI_PROFILE_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        if profile_frames > 0 {
            eng.profiler.set_enabled(true);
            for _ in 0..profile_frames {
                frame(&mut eng, START_X, START_Z, BISTRO_YAW);
            }
            let timings = eng
                .profiler
                .snapshot()
                .into_iter()
                .filter_map(|(label, _, gpu)| {
                    (label.starts_with("probe_") || label == "wsrc_bake_pass")
                        .then_some((label, gpu?))
                })
                .collect::<Vec<_>>();
            eprintln!("detailed-bistro-ssgi profile={timings:?}");
            eng.profiler.set_enabled(false);
        }

        let directory =
            std::env::temp_dir().join(format!("bloom-bistro-ssgi-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
        let (output_w, output_h, output) =
            render(&mut eng, 1, |eng| draw(eng, START_X, START_Z, BISTRO_YAW));
        if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
            image::save_buffer(
                directory.join("final-output.png"),
                &output,
                output_w,
                output_h,
                image::ColorType::Rgba8,
            )
            .expect("write detailed Bistro final output");

            // Preserve the seven vec4 probe records at the exact compared
            // endpoint. The diagnostic PNGs prove whether current rays differ;
            // this readback identifies which world-owned history record did.
            const PROBE_HEADER_BYTES: u64 = 112;
            let header_bytes = u64::from(eng.renderer.probe_grid_w * eng.renderer.probe_grid_h)
                * PROBE_HEADER_BYTES;
            let device = eng.renderer.device.clone();
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bistro_endpoint_probe_header_staging"),
                size: header_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bistro_endpoint_probe_header_copy"),
            });
            encoder.copy_buffer_to_buffer(
                &eng.renderer.probe_header_buffer,
                0,
                &staging,
                0,
                header_bytes,
            );
            eng.renderer.queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            rx.recv()
                .expect("endpoint probe map callback")
                .expect("endpoint probe map");
            let header_data = slice.get_mapped_range().to_vec();
            staging.unmap();
            std::fs::write(directory.join("probe-headers.bin"), header_data)
                .expect("write endpoint probe headers");

            // Preserve all current-sample scratch layers and every retained
            // phase estimate. Rows stay GPU-aligned; the sidecar records the
            // compact extent and stride so offline analysis can compare the
            // finite ring without adding any production diagnostic resource.
            let probe_grid_w = eng.renderer.probe_grid_w;
            let probe_grid_h = eng.renderer.probe_grid_h;
            let unpadded_bytes_per_row = probe_grid_w * 8;
            let bytes_per_row = unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let history_bytes = u64::from(bytes_per_row) * u64::from(probe_grid_h) * 64;
            let history_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bistro_endpoint_probe_history_staging"),
                size: history_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bistro_endpoint_probe_history_copy"),
            });
            let latest_history = 1 - eng.renderer.probe_history_idx;
            encoder.copy_texture_to_buffer(
                eng.renderer.probe_history_textures[latest_history].as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &history_staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(probe_grid_h),
                    },
                },
                wgpu::Extent3d {
                    width: probe_grid_w,
                    height: probe_grid_h,
                    depth_or_array_layers: 64,
                },
            );
            eng.renderer.queue.submit(std::iter::once(encoder.finish()));
            let history_slice = history_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            history_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            rx.recv()
                .expect("endpoint history map callback")
                .expect("endpoint history map");
            let history_data = history_slice.get_mapped_range().to_vec();
            history_staging.unmap();
            std::fs::write(
                directory.join("probe-history-rgba16f-padded.bin"),
                history_data,
            )
            .expect("write endpoint probe history");
            std::fs::write(
                directory.join("probe-history-layout.txt"),
                format!(
                    "width={probe_grid_w}\nheight={probe_grid_h}\nlayers=64\nbytes_per_row={bytes_per_row}\nbytes_per_texel=8\nlatest_index={latest_history}\n"
                ),
            )
            .expect("write endpoint probe history layout");
        }
        let ssgi = image::open(directory.join("ssgi.png"))
            .expect("Bistro capture did not emit raw SSGI")
            .to_rgba8();
        let paths = eng.renderer.quality_runtime_paths_json();
        if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
            eprintln!("kept detailed Bistro SSGI diagnostics at {directory:?}");
        } else {
            let _ = std::fs::remove_dir_all(directory);
        }
        (output, ssgi, paths)
    }

    let (moved_output, moved_ssgi, moved_paths) = capture(
        &scene_path,
        true,
        warmup_frames,
        endpoint_settle_frames,
        shadows_enabled,
        shadow_always_fresh,
        "moved",
    );
    let (direct_output, direct_ssgi, direct_paths) = capture(
        &scene_path,
        false,
        warmup_frames,
        endpoint_settle_frames,
        shadows_enabled,
        shadow_always_fresh,
        "direct",
    );
    assert!(
        moved_paths.contains("\"ssgi_trace_backend\":\"hw-ray-query\"")
            && direct_paths.contains("\"ssgi_trace_backend\":\"hw-ray-query\""),
        "Bistro endpoint oracle did not run hardware SSGI"
    );
    assert_eq!(moved_ssgi.dimensions(), direct_ssgi.dimensions());
    let (ssgi_w, ssgi_h) = moved_ssgi.dimensions();
    let ssgi_metrics =
        calculate_diff_metrics(moved_ssgi.as_raw(), direct_ssgi.as_raw(), ssgi_w, ssgi_h);
    let output_metrics = calculate_diff_metrics(&moved_output, &direct_output, W, H);
    eprintln!(
        "detailed-bistro-ssgi endpoint shadows={shadows_enabled} \
         shadow_always_fresh={shadow_always_fresh} \
         warmup_frames={warmup_frames} settle_frames={endpoint_settle_frames} \
         raw={ssgi_metrics:?} output={output_metrics:?}"
    );

    assert!(
        ssgi_metrics.ssim >= 0.9995 && output_metrics.ssim >= 0.9998,
        "detailed Bistro SSGI retained camera-path-dependent lighting: \
         raw={ssgi_metrics:?}, output={output_metrics:?}"
    );
}

/// Opt-in diagnostic reproduction of the delayed Bistro awning glow.
///
/// Runs the exact stationary façade view past full BLAS admission and the
/// coherent card-light handoff, then dumps full-precision probe headers and
/// the raw 32-ray trace texture (with per-ray source provenance in alpha)
/// for offline analysis. Gated on BLOOM_BISTRO_PROBE_DUMP_DIR +
/// BLOOM_BISTRO_SSGI_SCENE so ordinary suite runs skip it.
#[test]
fn dump_detailed_bistro_probe_state() {
    let Some(dump_dir) = std::env::var_os("BLOOM_BISTRO_PROBE_DUMP_DIR").map(PathBuf::from) else {
        eprintln!("skip: BLOOM_BISTRO_PROBE_DUMP_DIR is not set");
        return;
    };
    let Some(scene_path) = std::env::var_os("BLOOM_BISTRO_SSGI_SCENE").map(PathBuf::from) else {
        eprintln!("skip: BLOOM_BISTRO_SSGI_SCENE is not set");
        return;
    };
    let _guard = lock_rt_goldens();
    let context = match RT_DEVICE.get_or_init(create_rt_device_context) {
        Ok(Some(context)) => context,
        Ok(None) => {
            skip_rt_golden(
                "dump_detailed_bistro_probe_state",
                "adapter does not expose experimental ray query",
            );
            return;
        }
        Err(error) => panic!("hardware context failed: {error}"),
    };
    let _ = std::fs::create_dir_all(&dump_dir);

    // Higher-than-golden resolution so the probe grid resolves the façade.
    // Match an interactive reproduction exactly when requested: screen-probe
    // placement changes with render resolution, so a lower-resolution dump
    // can otherwise miss the sample pattern under investigation.
    let (dump_w, dump_h) = std::env::var("BLOOM_BISTRO_PROBE_DUMP_SIZE")
        .ok()
        .and_then(|value| {
            let parts: Vec<u32> = value
                .split('x')
                .filter_map(|part| part.trim().parse::<u32>().ok())
                .collect();
            (parts.len() == 2 && parts[0] > 0 && parts[1] > 0).then(|| (parts[0], parts[1]))
        })
        .unwrap_or((1024, 576));
    let renderer = Renderer::new_headless(
        context.device.clone(),
        context.queue.clone(),
        dump_w,
        dump_h,
    );
    let mut eng = EngineState::new(renderer);
    configure(&mut eng, true, false);
    let dump_render_scale = std::env::var("BLOOM_BISTRO_PROBE_DUMP_RENDER_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(1.0)
        .clamp(0.5, 1.0);
    eng.renderer.set_render_scale(dump_render_scale);
    let dump_ssgi_enabled =
        std::env::var_os("BLOOM_BISTRO_PROBE_DUMP_SSGI").is_none_or(|value| value != "0");
    if std::env::var_os("BLOOM_BISTRO_PROBE_DUMP_SSR").is_some() {
        eng.renderer.set_ssr_enabled(true);
    }
    if std::env::var_os("BLOOM_BISTRO_PROBE_DUMP_TAA").is_some() {
        eng.renderer.set_taa_enabled(true);
    }
    if std::env::var_os("BLOOM_BISTRO_PROBE_DUMP_OCCLUSION").is_some_and(|value| value == "0") {
        eng.renderer.occlusion.set_enabled(false);
    }
    attach_model_placements(&mut eng, &scene_path);

    // Default pose is the shared façade oracle view; an override aims the
    // dump at a specific surface (e.g. the wall above the red awnings).
    let (cam_x, cam_z, cam_yaw) = std::env::var("BLOOM_BISTRO_PROBE_DUMP_CAMERA")
        .ok()
        .and_then(|value| {
            let parts: Vec<f32> = value
                .split(',')
                .filter_map(|part| part.trim().parse::<f32>().ok())
                .collect();
            (parts.len() == 3).then(|| (parts[0], parts[1], parts[2]))
        })
        .unwrap_or((-3.2720, 7.2358, BISTRO_YAW));
    let start_x = cam_x;
    let start_z = cam_z;
    // The BLAS queue fills lazily, so measure it after the first frame and
    // keep rendering until hardware admission has fully drained plus a
    // settle window past the coherent card-light handoff.
    frame(&mut eng, start_x, start_z, cam_yaw);
    let admission_frames = eng.scene.pending_blas_builds.len();
    let mut warmup_frames = 1;
    while !eng.scene.pending_blas_builds.is_empty() || warmup_frames < admission_frames + 156 {
        frame(&mut eng, start_x, start_z, cam_yaw);
        warmup_frames += 1;
        assert!(warmup_frames < 20_000, "BLAS admission never drained");
    }

    // Hardware admission requires an active ray consumer. For the SSGI-off
    // control, disable only after the queue drains. When movement is enabled,
    // the matched 108-frame excursion flushes prior SSGI-composited display
    // history without giving this control extra stationary settle frames.
    let dump_move_enabled = std::env::var_os("BLOOM_BISTRO_PROBE_DUMP_MOVE").is_some();
    if !dump_ssgi_enabled {
        eng.renderer.set_ssgi_enabled(false);
        if !dump_move_enabled {
            for _ in 0..64 {
                frame(&mut eng, start_x, start_z, cam_yaw);
            }
        }
    }

    if dump_move_enabled {
        let end_x = start_x + 3.748170285;
        let end_z = start_z + 4.685212856;
        for step in 1..=30 {
            let t = step as f32 / 30.0;
            frame(
                &mut eng,
                start_x + (end_x - start_x) * t,
                start_z + (end_z - start_z) * t,
                cam_yaw,
            );
        }
        for step in 1..=30 {
            let t = step as f32 / 30.0;
            frame(
                &mut eng,
                end_x + (start_x - end_x) * t,
                end_z + (start_z - end_z) * t,
                cam_yaw,
            );
        }
        for step in 1..=24 {
            let t = step as f32 / 24.0;
            frame(&mut eng, start_x, start_z, cam_yaw + 0.45 * t);
        }
        for step in 1..=24 {
            let t = step as f32 / 24.0;
            frame(&mut eng, start_x, start_z, cam_yaw + 0.45 * (1.0 - t));
        }
    }

    let dump_diagnostics =
        std::env::var_os("BLOOM_BISTRO_PROBE_DUMP_DIAGNOSTICS").is_none_or(|value| value != "0");
    if dump_diagnostics {
        eng.renderer.pending_quality_capture_dir = Some(dump_dir.to_string_lossy().into_owned());
    }
    let (shot_w, shot_h, screenshot) =
        render(&mut eng, 1, |eng| draw(eng, start_x, start_z, cam_yaw));
    image::save_buffer(
        dump_dir.join("final-output.png"),
        &screenshot,
        shot_w,
        shot_h,
        image::ColorType::Rgba8,
    )
    .expect("write final Bistro output");

    // Optional stationary or linearly moving burst for offline temporal
    // analysis. This is gated so the ordinary corpus keeps exactly one
    // readback; unlike an interactive window it cannot miss a transient
    // because macOS withheld a drawable.
    let sequence_frames = std::env::var("BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        .min(64);
    let sequence_diagnostic_frames = std::env::var("BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_DIAGNOSTICS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|index| *index < sequence_frames)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let sequence_motion = std::env::var("BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_MOTION")
        .ok()
        .and_then(|value| {
            let parts = value
                .split(',')
                .filter_map(|part| part.trim().parse::<f32>().ok())
                .collect::<Vec<_>>();
            (parts.len() == 3).then(|| (parts[0], parts[1], parts[2]))
        });
    if sequence_frames > 0 {
        image::save_buffer(
            dump_dir.join("sequence-000.png"),
            &screenshot,
            shot_w,
            shot_h,
            image::ColorType::Rgba8,
        )
        .expect("write first Bistro sequence frame");
        for sequence_index in 1..sequence_frames {
            if sequence_diagnostic_frames.contains(&sequence_index) {
                eng.renderer.pending_quality_capture_dir = Some(
                    dump_dir
                        .join(format!("sequence-diagnostics-{sequence_index:03}"))
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            let sequence_t = sequence_index as f32 / (sequence_frames - 1) as f32;
            let (sequence_x, sequence_z, sequence_yaw) = sequence_motion.map_or(
                (start_x, start_z, cam_yaw),
                |(delta_x, delta_z, delta_yaw)| {
                    (
                        start_x + delta_x * sequence_t,
                        start_z + delta_z * sequence_t,
                        cam_yaw + delta_yaw * sequence_t,
                    )
                },
            );
            let (sequence_w, sequence_h, sequence) = render(&mut eng, 1, |eng| {
                draw(eng, sequence_x, sequence_z, sequence_yaw)
            });
            image::save_buffer(
                dump_dir.join(format!("sequence-{sequence_index:03}.png")),
                &sequence,
                sequence_w,
                sequence_h,
                image::ColorType::Rgba8,
            )
            .expect("write Bistro sequence frame");
        }
    }

    let gw = eng.renderer.probe_grid_w;
    let gh = eng.renderer.probe_grid_h;
    const PROBE_HEADER_BYTES: u64 = 112;
    let header_bytes = u64::from(gw * gh) * PROBE_HEADER_BYTES;
    let trace_row_bytes = u64::from(gw) * 8;
    let trace_padded_row = trace_row_bytes.div_ceil(256) * 256;
    let trace_bytes = trace_padded_row * u64::from(gh) * 64;

    let device = eng.renderer.device.clone();
    let header_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_header_dump_staging"),
        size: header_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let trace_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_trace_dump_staging"),
        size: trace_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe_dump_encoder"),
    });
    encoder.copy_buffer_to_buffer(
        &eng.renderer.probe_header_buffer,
        0,
        &header_staging,
        0,
        header_bytes,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &eng.renderer.probe_trace_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &trace_staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(trace_padded_row as u32),
                rows_per_image: Some(gh),
            },
        },
        wgpu::Extent3d {
            width: gw,
            height: gh,
            depth_or_array_layers: 64,
        },
    );
    eng.renderer.queue.submit(std::iter::once(encoder.finish()));

    let read_buffer = |buffer: &wgpu::Buffer| -> Vec<u8> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().expect("dump map callback").expect("dump map");
        let data = slice.get_mapped_range().to_vec();
        buffer.unmap();
        data
    };
    let header_data = read_buffer(&header_staging);
    let trace_data = read_buffer(&trace_staging);
    std::fs::write(dump_dir.join("probe-headers.bin"), &header_data).expect("write probe headers");
    std::fs::write(dump_dir.join("probe-trace.bin"), &trace_data).expect("write probe trace");
    let meta = format!(
        "{{\"grid_w\":{gw},\"grid_h\":{gh},\"render_w\":{dump_w},\"render_h\":{dump_h},\
         \"probe_frame_index\":{},\"probe_header_bytes\":{PROBE_HEADER_BYTES},\
         \"trace_padded_row\":{trace_padded_row},\
         \"camera\":[{start_x},1.544,{start_z}],\"yaw\":{cam_yaw},\"fov_deg\":60.0,\
         \"ssgi_radius\":{},\"ssgi_intensity\":{},\"admission_frames\":{admission_frames},\
         \"warmup_frames\":{warmup_frames}}}",
        eng.renderer.probe_frame_index, eng.renderer.ssgi_radius, eng.renderer.ssgi_intensity,
    );
    std::fs::write(dump_dir.join("meta.json"), meta).expect("write dump meta");
    eprintln!("probe dump written to {dump_dir:?} (grid {gw}x{gh})");
}
