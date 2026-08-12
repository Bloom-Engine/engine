use super::super::*;

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

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept Hi-Z SSGI diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
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

    let draw = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(6.0, 8.0, 15.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.set_ambient_light(15.0, 18.0, 28.0, 0.2);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.85, 0.7, 1.8);
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
    let directory =
        std::env::temp_dir().join(format!("bloom-ssgi-diagnostics-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng);

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

    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"ray_scene_preparation\":\"ssgi\""));
    assert!(paths.contains("\"ssgi_diagnostic_persistent_bytes\":0"));
    assert!(paths.contains("\"ssgi_diagnostic_capture_passes\":1"));
    assert!(paths.contains("\"ssgi_diagnostic_resources_live\":false"));
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept SSGI diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
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
            >= total / 2,
        "camera motion neither rejected changed probe surfaces nor refreshed \
         reprojected history"
    );

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept moving SSGI diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
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
    let shadows_enabled =
        std::env::var_os("BLOOM_BISTRO_SSGI_SHADOWS").is_none_or(|value| value != "0");
    let shadow_always_fresh = std::env::var_os("BLOOM_BISTRO_SSGI_SHADOW_ALWAYS_FRESH").is_some();

    fn attach_model_placements(eng: &mut EngineState, source_path: &Path) {
        let data = std::fs::read(source_path).expect("read detailed Bistro glTF");
        let model_handle = eng.models.load_model_with_textures_from_source_path(
            &data,
            source_path,
            &mut eng.renderer,
        );
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

    fn draw(eng: &mut EngineState, camera_x: f32, camera_z: f32) {
        const YAW: f32 = -0.344;
        let forward_x = -YAW.sin();
        let forward_z = -YAW.cos();
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

    fn frame(eng: &mut EngineState, camera_x: f32, camera_z: f32) {
        eng.begin_frame();
        draw(eng, camera_x, camera_z);
        eng.end_frame();
    }

    fn capture(
        scene_path: &Path,
        moved: bool,
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

        const START_X: f32 = -3.2720;
        const START_Z: f32 = 7.2358;
        const END_X: f32 = START_X + 3.748170285;
        const END_Z: f32 = START_Z + 4.685212856;
        if moved {
            for _ in 0..96 {
                frame(&mut eng, START_X, START_Z);
            }
            for step in 1..=60 {
                let t = step as f32 / 60.0;
                frame(
                    &mut eng,
                    START_X + (END_X - START_X) * t,
                    START_Z + (END_Z - START_Z) * t,
                );
            }
            for _ in 0..endpoint_settle_frames {
                frame(&mut eng, END_X, END_Z);
            }
        } else {
            for _ in 0..(156 + endpoint_settle_frames) {
                frame(&mut eng, END_X, END_Z);
            }
        }

        let profile_frames = std::env::var("BLOOM_BISTRO_SSGI_PROFILE_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        if profile_frames > 0 {
            eng.profiler.set_enabled(true);
            for _ in 0..profile_frames {
                frame(&mut eng, END_X, END_Z);
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
        let (_, _, output) = render(&mut eng, 1, |eng| draw(eng, END_X, END_Z));
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
        endpoint_settle_frames,
        shadows_enabled,
        shadow_always_fresh,
        "moved",
    );
    let (direct_output, direct_ssgi, direct_paths) = capture(
        &scene_path,
        false,
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
         settle_frames={endpoint_settle_frames} \
         raw={ssgi_metrics:?} output={output_metrics:?}"
    );

    assert!(
        ssgi_metrics.ssim >= 0.9995 && output_metrics.ssim >= 0.9998,
        "detailed Bistro SSGI retained camera-path-dependent lighting: \
         raw={ssgi_metrics:?}, output={output_metrics:?}"
    );
}
