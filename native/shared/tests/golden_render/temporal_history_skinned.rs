//! Temporal history: particles, skinning, alpha-tested cards, emissive switches
//! and render-scale steps. Split from temporal_history.rs.

#[allow(unused_imports)]
use super::*;

#[test]
fn instanced_particle_reactive_opt_in_bounds_trails_without_taxing_opt_out() {
    const HANDLE: u64 = 0x7AA5_C011;
    const PARTICLE_SHADER_PREFIX: &str = r#"
#include "material_abi.wgsl"

struct ParticleInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
  @location(3) uv: vec2<f32>,
  @location(4) joints: vec4<f32>,
  @location(5) weights: vec4<f32>,
  @location(6) tangent: vec4<f32>,
  @location(7) instance_pos: vec3<f32>,
  @location(8) instance_rot_y: f32,
  @location(9) instance_scale: f32,
  @location(10) instance_tint: vec4<f32>,
};

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) tint: vec4<f32>,
};

@vertex
fn vs_main(in: ParticleInput) -> VsOut {
  var out: VsOut;
  let world = in.position * in.instance_scale + in.instance_pos;
  out.clip_position = view.view_proj * vec4<f32>(world, 1.0);
  out.tint = in.color * in.instance_tint;
  return out;
}

fn particle_color(in: VsOut) -> vec4<f32> {
  return vec4<f32>(in.tint.rgb * 3.0, in.tint.a);
}

@fragment
fn fs_main(in: VsOut) -> TranslucentOut {
  var out: TranslucentOut;
  out.hdr = particle_color(in);
  return out;
}
"#;
    const PARTICLE_REACTIVE_SUFFIX: &str = r#"

@fragment
fn fs_reactive(in: VsOut) -> ReactiveTranslucentOut {
  var out: ReactiveTranslucentOut;
  out.hdr = particle_color(in);
  out.reactive = in.tint.a;
  return out;
}
"#;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);
    eng.renderer.set_transparency_composition_mode(0);

    let vertex = |position| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [1.0, 0.13, 0.025, 1.0],
        uv: [0.0; 2],
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices: vec![
                vertex([-0.55, -0.55, 0.0]),
                vertex([0.55, -0.55, 0.0]),
                vertex([0.55, 0.55, 0.0]),
                vertex([-0.55, 0.55, 0.0]),
            ],
            secondary_tex_coords: None,
            indices: vec![0, 1, 2, 0, 2, 3],
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Blend,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: true,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));

    let ordinary_material = eng
        .renderer
        .compile_material_instanced_bucket(PARTICLE_SHADER_PREFIX, 2, false)
        .expect("ordinary instanced particle material compiles");
    let reactive_source = format!("{PARTICLE_SHADER_PREFIX}{PARTICLE_REACTIVE_SUFFIX}");
    let reactive_material = eng
        .renderer
        .compile_material_instanced_bucket(&reactive_source, 2, false)
        .expect("reactive instanced particle material compiles");
    let old_buffer = eng
        .renderer
        .create_instance_buffer(&[-1.35, 1.15, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1);
    let new_buffer = eng
        .renderer
        .create_instance_buffer(&[1.35, 1.15, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1);

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.0, 6.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.2, -1.7, 5.0, 3.2, 0.3, 30.0, 150.0, 220.0, 255.0);
    };
    let capture_particle = |eng: &mut EngineState, material, instance_buffer| -> Vec<u8> {
        render(eng, 1, |eng| {
            draw_scene(eng);
            eng.renderer
                .submit_material_draw_instanced(material, HANDLE, 0, instance_buffer, 1);
        })
        .2
    };
    let mut run_sequence = |material, label: &str, capture_reactive: bool| {
        eng.renderer.reset_temporal_history();
        for _ in 0..8 {
            capture_particle(&mut eng, material, old_buffer);
        }
        let old_pose = capture_particle(&mut eng, material, old_buffer);
        let directory = std::env::temp_dir().join(format!("bloom-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        if capture_reactive {
            eng.renderer.pending_quality_capture_dir =
                Some(directory.to_string_lossy().into_owned());
        }
        let mut frames = Vec::new();
        for _ in 0..24 {
            frames.push(capture_particle(&mut eng, material, new_buffer));
        }
        evaluate_motion_recovery(label, &old_pose, &frames);
        let paths: serde_json::Value =
            serde_json::from_str(&eng.renderer.quality_runtime_paths_json()).unwrap();
        assert_eq!(
            paths["temporal_reactive"]["active"].as_bool(),
            Some(capture_reactive),
            "{label} selected the wrong temporal-reactive topology"
        );
        if capture_reactive {
            let reasons = image::open(directory.join("taa-rejection-reason.png"))
                .expect("reactive particle capture did not emit rejection reasons")
                .to_rgb8();
            let reactive_pixels = reasons
                .pixels()
                .filter(|pixel| {
                    i32::from(pixel[0]).pow(2)
                        + (i32::from(pixel[1]) - 230).pow(2)
                        + (i32::from(pixel[2]) - 255).pow(2)
                        < 80_i32.pow(2)
                })
                .count();
            eprintln!("temporal-corpus {label} reactive_pixels={reactive_pixels}");
            assert!(
                reactive_pixels >= 100,
                "authored particle coverage did not reach TAA rejection"
            );
        }
        if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
            eprintln!("kept {label} diagnostics at {directory:?}");
        } else {
            let _ = std::fs::remove_dir_all(directory);
        }
    };

    run_sequence(reactive_material, "reactive-particle", true);
    run_sequence(ordinary_material, "ordinary-particle-control", false);
    eng.renderer.destroy_instance_buffer(old_buffer);
    eng.renderer.destroy_instance_buffer(new_buffer);
}

#[test]
fn legacy_skinned_motion_uses_staged_previous_palette_and_bounds_trails() {
    const PALETTE_KEY: u64 = 0x7AA5_2001;
    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn palette(bend: f32) -> [[[f32; 4]; 4]; 2] {
        let (sin, cos) = bend.sin_cos();
        [
            IDENTITY,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, sin, 0.0],
                [0.0, -sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    }

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let (mut vertices, indices) = cube_verts(0.9, [0.95, 0.12, 0.035, 1.0]);
    for vertex in &mut vertices {
        let upper = vertex.position[1] > 0.0;
        vertex.joints = if upper {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0; 4]
        };
        vertex.weights = [1.0, 0.0, 0.0, 0.0];
    }

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.88, 2.2);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.1, -1.8, 5.0, 3.2, 0.35, 30.0, 170.0, 235.0, 255.0);
    };
    let capture_pose = |eng: &mut EngineState, x: f32, bend: f32, facing: f32| {
        render(eng, 1, |eng| {
            draw_scene(eng);
            let (rot_sin, rot_cos) = facing.sin_cos();
            eng.renderer.set_joint_matrices_scaled(
                PALETTE_KEY,
                &palette(bend),
                1.0,
                [x, 1.0, 0.0],
                rot_sin,
                rot_cos,
            );
            eng.renderer
                .draw_model_mesh(&vertices, &indices, [0.0; 3], 1.0);
        })
        .2
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture_pose(&mut eng, -1.5, -0.55, -0.45);
    }
    let old_pose = capture_pose(&mut eng, -1.5, -0.55, -0.45);
    let directory =
        std::env::temp_dir().join(format!("bloom-legacy-skin-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture_pose(&mut eng, 1.5, 0.75, 0.6));
    }

    let motion = image::open(directory.join("taa-motion.png"))
        .expect("legacy skinned capture did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus legacy-skinned moving_pixels={moving_pixels}");
    assert!(
        moving_pixels >= 250,
        "legacy skinned pose wrote no meaningful velocity"
    );
    evaluate_motion_recovery("legacy-skinned", &old_pose, &frames);

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept legacy-skinned diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn cached_skinned_motion_sequence_bounds_animation_trails() {
    const HANDLE: u64 = 0x7AA5_0001;
    const PALETTE_KEY: u64 = 0x7AA5_1001;
    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn palette(bend: f32) -> [[[f32; 4]; 4]; 2] {
        let (sin, cos) = bend.sin_cos();
        [
            IDENTITY,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, sin, 0.0],
                [0.0, -sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    }

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let (mut vertices, indices) = cube_verts(0.9, [0.95, 0.12, 0.035, 1.0]);
    for vertex in &mut vertices {
        let upper = vertex.position[1] > 0.0;
        vertex.joints = if upper {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0; 4]
        };
        vertex.weights = [1.0, 0.0, 0.0, 0.0];
    }
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices,
            secondary_tex_coords: None,
            indices,
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.15,
            roughness_factor: 0.32,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));
    assert!(
        eng.renderer.is_model_skinned(HANDLE),
        "weighted temporal test mesh did not select the cached skinned path"
    );

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.88, 2.2);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.1, -1.8, 5.0, 3.2, 0.35, 30.0, 170.0, 235.0, 255.0);
    };
    let draw_pose = |eng: &mut EngineState, x: f32, bend: f32, facing: f32| {
        draw_scene(eng);
        let (rot_sin, rot_cos) = facing.sin_cos();
        eng.renderer.set_joint_matrices_scaled(
            PALETTE_KEY,
            &palette(bend),
            1.0,
            [x, 1.0, 0.0],
            rot_sin,
            rot_cos,
        );
        eng.renderer
            .draw_model_cached_skinned(HANDLE, [0.0; 3], 1.0, [1.0; 4]);
    };
    let capture_pose = |eng: &mut EngineState, x: f32, bend: f32, facing: f32| {
        render(eng, 1, |eng| draw_pose(eng, x, bend, facing)).2
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture_pose(&mut eng, -1.5, -0.55, -0.45);
    }
    let old_pose = capture_pose(&mut eng, -1.5, -0.55, -0.45);
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture_pose(&mut eng, 1.5, 0.75, 0.6));
    }
    evaluate_motion_recovery("cached-skinned", &old_pose, &frames);
}

#[test]
fn cached_alpha_tested_card_motion_writes_velocity_and_bounds_trails() {
    const HANDLE: u64 = 0x7AA5_F011;
    const TEX_SIZE: u32 = 64;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let mut pixels = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let dx = (x as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let dy = (y as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let ellipse = dx * dx / 0.82f32.powi(2) + dy * dy / 0.96f32.powi(2) < 1.0;
            let serrated_edge = ((y / 4 + x / 7) & 1) == 0 || dx.abs() < 0.68;
            let vein_gap = (x as i32 - TEX_SIZE as i32 / 2).abs() == 5 && y % 9 < 6;
            let opaque = ellipse && serrated_edge && !vein_gap;
            pixels.extend_from_slice(if opaque {
                &[30, 205, 55, 255]
            } else {
                &[0, 0, 0, 0]
            });
        }
    }
    let texture = eng.renderer.register_texture_kind_with_alpha_coverage(
        TEX_SIZE,
        TEX_SIZE,
        &pixels,
        false,
        Some(0.5),
    );
    let vertex = |position, uv| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [1.0; 4],
        uv,
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    let vertices = vec![
        vertex([-1.0, 0.0, 0.0], [0.0, 1.0]),
        vertex([1.0, 0.0, 0.0], [1.0, 1.0]),
        vertex([1.0, 2.8, 0.0], [1.0, 0.0]),
        vertex([-1.0, 2.8, 0.0], [0.0, 0.0]),
    ];
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices,
            secondary_tex_coords: None,
            indices: vec![0, 1, 2, 0, 2, 3],
            texture_idx: Some(texture),
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 0.72,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Mask,
            alpha_cutoff: 0.5,
            alpha_coverage_mips: true,
            double_sided: true,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 25.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 1.3, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.35, -1.0, -0.2, 0.9, 1.0, 0.8, 2.0);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 38.0, 45.0, 55.0, 255.0);
        r.draw_cube(0.0, 1.5, -1.1, 6.0, 3.4, 0.25, 35.0, 80.0, 125.0, 255.0);
    };
    let capture_pose = |eng: &mut EngineState, x: f32, angle: f32| {
        render(eng, 1, |eng| {
            draw_scene(eng);
            eng.renderer
                .draw_model_cached_rotated(HANDLE, [x, 0.0, 0.0], 1.0, angle, [1.0; 4]);
        })
        .2
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture_pose(&mut eng, -1.35, -0.35);
    }
    let old_pose = capture_pose(&mut eng, -1.35, -0.35);
    let directory =
        std::env::temp_dir().join(format!("bloom-foliage-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture_pose(&mut eng, 1.35, 0.5));
    }

    let motion = image::open(directory.join("taa-motion.png"))
        .expect("alpha-tested motion capture did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus alpha-tested moving_pixels={moving_pixels}");
    assert!(
        moving_pixels >= 250,
        "cached alpha-tested object motion wrote no meaningful velocity"
    );
    evaluate_motion_recovery("alpha-tested-card", &old_pose, &frames);

    let paths = eng.renderer.quality_runtime_paths_json();
    assert!(paths.contains("\"cached_model_motion_entries\":1"));
    assert!(paths.contains("\"cached_model_motion_gpu_bytes\":0"));
    assert!(paths.contains("\"cached_model_motion_passes\":0"));
    let paths: serde_json::Value = serde_json::from_str(&paths).unwrap();
    let cpu_capacity = paths["temporal_history"]["cached_model_motion_cpu_capacity_bytes"]
        .as_u64()
        .unwrap();
    eprintln!("temporal-corpus cached-model history_cpu_capacity={cpu_capacity} bytes");
    assert!(
        cpu_capacity <= 1024,
        "one cached instance retained excessive transform history: {cpu_capacity} bytes"
    );
    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept alpha-tested motion diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn emissive_light_switches_converge_without_radiance_trails() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let (vertices, indices) = cube_verts(0.65, [0.18, 0.035, 0.01, 1.0]);
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_transform(
        node,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
    );
    eng.scene.set_material_pbr(node, 0.24, 0.0);

    let draw_state = |eng: &mut EngineState, enabled: bool| {
        eng.scene.set_material_emissive_factor(
            node,
            if enabled { 7.0 } else { 0.0 },
            if enabled { 0.8 } else { 0.0 },
            if enabled { 0.08 } else { 0.0 },
        );
        let r = &mut eng.renderer;
        r.set_clear_color(5.0, 7.0, 14.0, 255.0);
        r.begin_mode_3d(4.2, 2.8, 6.2, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 0.55, 0.65, 0.9, 0.55);
        if enabled {
            r.add_point_light(0.0, 1.25, 0.2, 4.5, 1.0, 0.11, 0.02, 8.0);
        }
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(-1.4, 0.65, 0.0, 0.8, 1.3, 0.8, 105.0, 115.0, 135.0, 255.0);
        r.draw_cube(1.4, 0.65, 0.0, 0.8, 1.3, 0.8, 105.0, 115.0, 135.0, 255.0);
    };
    let capture_state =
        |eng: &mut EngineState, enabled| render(eng, 1, |eng| draw_state(eng, enabled)).2;
    let run_switch = |eng: &mut EngineState, from: bool, to: bool| {
        eng.renderer.reset_temporal_history();
        for _ in 0..8 {
            capture_state(eng, from);
        }
        let old_state = capture_state(eng, from);
        let mut frames = Vec::new();
        for _ in 0..24 {
            frames.push(capture_state(eng, to));
        }
        (old_state, frames)
    };

    for (label, render_scale) in [("native", 1.0), ("fractional", 0.75)] {
        eng.renderer.set_render_scale(render_scale);
        let (off_state, on_frames) = run_switch(&mut eng, false, true);
        evaluate_motion_recovery(&format!("emissive-on-{label}"), &off_state, &on_frames);
        let (on_state, off_frames) = run_switch(&mut eng, true, false);
        evaluate_motion_recovery(&format!("emissive-off-{label}"), &on_state, &off_frames);
    }
}

#[test]
fn render_scale_and_resize_steps_seed_without_prior_frame_residue() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 24.0, 255.0);
        r.begin_mode_3d(4.0, 2.8, 6.0, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 49.0, 0.0);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.9, 0.75, 2.0);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 35.0, 42.0, 58.0, 255.0);
        for x in -2..=2 {
            r.draw_cube(
                x as f64 * 0.75,
                0.45,
                (x & 1) as f64 * 0.45,
                0.42,
                0.9,
                0.42,
                210.0,
                55.0 + (x + 2) as f64 * 35.0,
                35.0,
                255.0,
            );
        }
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_scene(eng);
            eng.end_frame();
        }
    };
    let capture = |eng: &mut EngineState| render(eng, 1, draw_scene);

    eng.renderer.set_render_scale(0.5);
    eng.renderer.reset_temporal_history();
    let (_, _, fresh_half_scale) = capture(&mut eng);
    eng.renderer.set_render_scale(1.0);
    advance(&mut eng, 8);
    eng.renderer.set_render_scale(0.5);
    let (_, _, stepped_half_scale) = capture(&mut eng);
    let scale_metrics = calculate_diff_metrics(&fresh_half_scale, &stepped_half_scale, W, H);
    assert_eq!(
        scale_metrics.max_diff, 0,
        "render-scale step blended pixels from the incompatible full-scale history"
    );

    eng.renderer.set_render_scale(1.0);
    eng.renderer.reset_temporal_history();
    let (_, _, fresh_native_size) = capture(&mut eng);
    eng.renderer.resize(320, 192, 320, 192);
    advance(&mut eng, 3);
    eng.renderer.resize(W, H, W, H);
    let (width, height, returned_native_size) = capture(&mut eng);
    assert_eq!((width, height), (W, H));
    let resize_metrics = calculate_diff_metrics(&fresh_native_size, &returned_native_size, W, H);
    eprintln!("temporal-corpus resize metrics={resize_metrics:?}");
    assert!(
        resize_metrics.mean_rgb <= 0.5
            && resize_metrics.outlier_pixel_fraction == 0.0
            && resize_metrics.max_diff <= 32,
        "window resize restored a coherent image from destroyed prior-size history: \
         {resize_metrics:?}"
    );
    eprintln!(
        "temporal-corpus scale_step_max={} resize_step_max={}",
        scale_metrics.max_diff, resize_metrics.max_diff,
    );
}
