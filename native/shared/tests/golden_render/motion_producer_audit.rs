use super::*;

fn quad_vertices(half_extent: f32) -> Vec<Vertex3D> {
    let vertex = |position| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [1.0; 4],
        uv: [0.0; 2],
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    vec![
        vertex([-half_extent, -half_extent, 0.0]),
        vertex([half_extent, -half_extent, 0.0]),
        vertex([half_extent, half_extent, 0.0]),
        vertex([-half_extent, half_extent, 0.0]),
    ]
}

fn untextured_mesh(vertices: Vec<Vertex3D>, alpha_mode: MaterialAlphaMode) -> MeshData {
    MeshData {
        vertices,
        secondary_tex_coords: None,
        indices: vec![0, 1, 2, 0, 2, 3],
        texture_idx: None,
        normal_texture_idx: None,
        metallic_roughness_texture_idx: None,
        emissive_texture_idx: None,
        occlusion_texture_idx: None,
        metallic_factor: 0.0,
        roughness_factor: 1.0,
        emissive_factor: [0.0; 3],
        alpha_mode,
        alpha_cutoff: 0.5,
        alpha_coverage_mips: false,
        double_sided: true,
        transmission: Default::default(),
        layered_pbr: Default::default(),
    }
}

#[test]
fn procedural_foliage_wind_reconstructs_previous_deformation_velocity() {
    const HANDLE: u64 = 0x7AA5_3011;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let vertex = |position| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [0.08, 0.72, 0.12, 1.0],
        uv: [0.0; 2],
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[untextured_mesh(
            vec![
                vertex([-0.45, 0.0, 0.0]),
                vertex([0.45, 0.0, 0.0]),
                vertex([0.45, 4.0, 0.0]),
                vertex([-0.45, 4.0, 0.0]),
            ],
            MaterialAlphaMode::Opaque,
        )]
    ));
    eng.renderer.set_wind(1.0, 0.25, 0.8, 1.0);
    eng.renderer.set_model_foliage_wind(HANDLE, 1.0);

    let capture = |eng: &mut EngineState| {
        std::thread::sleep(std::time::Duration::from_millis(20));
        render(eng, 1, |eng| {
            let r = &mut eng.renderer;
            r.set_clear_color(7.0, 10.0, 20.0, 255.0);
            r.begin_mode_3d(0.0, 2.1, 7.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 46.0, 0.0);
            r.draw_model_cached(HANDLE, [0.0; 3], 1.0, [1.0; 4]);
        })
        .2
    };
    for _ in 0..4 {
        capture(&mut eng);
    }
    let directory = std::env::temp_dir().join(format!("bloom-foliage-wind-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng);

    let motion = image::open(directory.join("taa-motion.png"))
        .expect("foliage capture did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus foliage-wind moving_pixels={moving_pixels}");
    assert!(
        moving_pixels >= 100,
        "procedural foliage wrote no meaningful prior-deformation velocity"
    );

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept foliage-wind diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}

#[test]
fn static_decal_zero_velocity_opt_out_rejects_spawn_and_expiry() {
    const HANDLE: u64 = 0x7AA5_D011;
    const DECAL_SHADER: &str = r#"
#include "material_abi.wgsl"

struct DecalInput {
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
fn vs_main(in: DecalInput) -> VsOut {
  var out: VsOut;
  let world = in.position * in.instance_scale + in.instance_pos;
  out.clip_position = view.view_proj * vec4<f32>(world, 1.0);
  out.tint = in.color * in.instance_tint;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> OpaqueOut {
  if (in.tint.a < 0.5) { discard; }
  var out: OpaqueOut;
  out.hdr = vec4<f32>(in.tint.rgb, 1.0);
  out.material = vec2<f32>(0.0, 1.0);
  out.velocity = vec2<f32>(0.0);
  out.albedo = vec4<f32>(in.tint.rgb, 1.0);
  return out;
}
"#;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[untextured_mesh(quad_vertices(0.7), MaterialAlphaMode::Mask)]
    ));
    let material = eng
        .renderer
        .compile_material_instanced_bucket(DECAL_SHADER, 1, false)
        .expect("static decal opt-out material compiles");
    let instance_buffer = eng
        .renderer
        .create_instance_buffer(&[0.0, 1.1, 0.0, 0.0, 1.0, 0.05, 0.02, 0.01, 1.0], 1);

    let capture = |eng: &mut EngineState, visible: bool| {
        render(eng, 1, |eng| {
            let r = &mut eng.renderer;
            r.set_clear_color(7.0, 10.0, 20.0, 255.0);
            r.begin_mode_3d(0.0, 2.0, 6.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
            r.draw_cube(0.0, 1.1, -0.8, 5.0, 3.2, 0.3, 35.0, 155.0, 220.0, 255.0);
            if visible {
                r.submit_material_draw_instanced(material, HANDLE, 0, instance_buffer, 1);
            }
        })
        .2
    };
    let mut run_transition = |from_visible, to_visible, label| {
        eng.renderer.reset_temporal_history();
        for _ in 0..8 {
            capture(&mut eng, from_visible);
        }
        let old_pose = capture(&mut eng, from_visible);
        let mut frames = Vec::new();
        for _ in 0..24 {
            frames.push(capture(&mut eng, to_visible));
        }
        temporal_history::evaluate_motion_recovery(label, &old_pose, &frames);
    };

    run_transition(false, true, "static-decal-spawn-optout");
    run_transition(true, false, "static-decal-expiry-optout");
    eng.renderer.destroy_instance_buffer(instance_buffer);
}

#[test]
fn procedural_cloud_zero_velocity_opt_out_rejects_field_changes() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);
    eng.renderer.set_procedural_sky(true, 1.0, 1.0, 0.1);
    eng.renderer.set_sun_direction(0.35, 0.82, 0.28, 1.0);
    eng.renderer.set_cloud_shadows(0.45, 180.0, 0.006, 0.0);

    let capture = |eng: &mut EngineState| {
        render(eng, 1, |eng| {
            eng.renderer
                .begin_mode_3d(0.0, 2.0, 0.0, 0.0, 2.1, -1.0, 0.0, 1.0, 0.0, 70.0, 0.0);
        })
        .2
    };
    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture(&mut eng);
    }
    let old_field = capture(&mut eng);
    eng.renderer.set_cloud_shadows(0.45, 310.0, 0.014, 0.0);
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture(&mut eng));
    }
    temporal_history::evaluate_motion_recovery(
        "procedural-cloud-field-optout",
        &old_field,
        &frames,
    );
}

#[test]
fn unkeyed_editor_skin_pose_writes_velocity_and_bounds_trails() {
    fn palette(x: f32, bend: f32, facing: f32) -> [[[f32; 4]; 4]; 2] {
        let (sin, cos) = bend.sin_cos();
        let mut matrices = [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, sin, 0.0],
                [0.0, -sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ];
        let (rot_sin, rot_cos) = facing.sin_cos();
        for matrix in &mut matrices {
            for column in matrix.iter_mut() {
                let old_x = column[0];
                let old_z = column[2];
                column[0] = rot_cos * old_x + rot_sin * old_z;
                column[2] = -rot_sin * old_x + rot_cos * old_z;
            }
            matrix[3][0] += x;
            matrix[3][1] += 1.0;
        }
        matrices
    }

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);
    let (mut vertices, indices) = cube_verts(0.9, [0.9, 0.08, 0.025, 1.0]);
    for vertex in &mut vertices {
        vertex.joints = if vertex.position[1] > 0.0 {
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
    let capture = |eng: &mut EngineState, x, bend, facing| {
        render(eng, 1, |eng| {
            draw_scene(eng);
            eng.renderer.set_joint_matrices(&palette(x, bend, facing));
            eng.renderer
                .draw_model_mesh(&vertices, &indices, [0.0; 3], 1.0);
        })
        .2
    };
    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        capture(&mut eng, -1.5, -0.55, -0.45);
    }
    let old_pose = capture(&mut eng, -1.5, -0.55, -0.45);
    let directory =
        std::env::temp_dir().join(format!("bloom-unkeyed-skin-motion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    let mut frames = Vec::new();
    for _ in 0..24 {
        frames.push(capture(&mut eng, 1.5, 0.75, 0.6));
    }
    let motion = image::open(directory.join("taa-motion.png"))
        .expect("unkeyed skin capture did not emit the TAA velocity map")
        .to_rgb8();
    let moving_pixels = motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus unkeyed-skin moving_pixels={moving_pixels}");
    assert!(
        moving_pixels >= 250,
        "unkeyed skin pose wrote no meaningful velocity"
    );
    temporal_history::evaluate_motion_recovery("unkeyed-skin", &old_pose, &frames);

    // A missing frame breaks slot identity. Reappearance must seed from the
    // current pose instead of inheriting the last visible pose's velocity.
    render(&mut eng, 1, |eng| draw_scene(eng));
    eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
    capture(&mut eng, -0.5, -0.2, 0.1);
    let gap_motion = image::open(directory.join("taa-motion.png"))
        .expect("unkeyed skin gap capture did not emit the TAA velocity map")
        .to_rgb8();
    let gap_moving_pixels = gap_motion.pixels().filter(|pixel| pixel[2] > 8).count();
    eprintln!("temporal-corpus unkeyed-skin gap_moving_pixels={gap_moving_pixels}");
    assert!(
        gap_moving_pixels <= 16,
        "unkeyed skin reappearance inherited stale pre-gap velocity"
    );

    let paths: serde_json::Value =
        serde_json::from_str(&eng.renderer.quality_runtime_paths_json()).unwrap();
    assert_eq!(
        paths["temporal_history"]["unkeyed_skin_motion_entries"].as_u64(),
        Some(1)
    );
    assert_eq!(
        paths["temporal_history"]["unkeyed_skin_motion_gpu_bytes"].as_u64(),
        Some(0)
    );
    assert_eq!(
        paths["temporal_history"]["unkeyed_skin_motion_passes"].as_u64(),
        Some(0)
    );
    let cpu_capacity = paths["temporal_history"]["unkeyed_skin_motion_cpu_capacity_bytes"]
        .as_u64()
        .unwrap();
    eprintln!("temporal-corpus unkeyed-skin cpu_capacity_bytes={cpu_capacity}");
    assert!(
        cpu_capacity <= 1024,
        "one two-joint unkeyed skin retained excessive CPU history"
    );

    if std::env::var_os("BLOOM_KEEP_TEMPORAL_DIAGNOSTICS").is_some() {
        eprintln!("kept unkeyed skin diagnostics at {directory:?}");
    } else {
        let _ = std::fs::remove_dir_all(directory);
    }
}
