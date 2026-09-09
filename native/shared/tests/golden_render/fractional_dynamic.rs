use super::*;

const TRANSITION_FRAMES: usize = 4;
const RECOVERY_FRAMES: usize = 8;
const SETTLE_FRAMES: u32 = 16;

#[derive(Clone, Copy)]
struct MotionPose {
    x: f32,
    turn: f32,
}

#[derive(Clone, Copy)]
struct NativeMatchBounds {
    mean_rgb: f64,
    maximum_mean_rgb: f64,
    minimum_ssim: f64,
    derivative_error: f64,
    recovery_mean_rgb: f64,
    recovery_maximum_outliers: f64,
    final_recovery_mean_rgb: f64,
}

fn transform(pose: MotionPose) -> [[f32; 4]; 4] {
    let (sin, cos) = pose.turn.sin_cos();
    [
        [cos, 0.0, -sin, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sin, 0.0, cos, 0.0],
        [pose.x, 1.0, 0.0, 1.0],
    ]
}

fn motion_poses(old: MotionPose, new: MotionPose) -> Vec<MotionPose> {
    (0..TRANSITION_FRAMES + RECOVERY_FRAMES)
        .map(|frame| {
            let t = if frame < TRANSITION_FRAMES {
                (frame + 1) as f32 / TRANSITION_FRAMES as f32
            } else {
                1.0
            };
            MotionPose {
                x: old.x + (new.x - old.x) * t,
                turn: old.turn + (new.turn - old.turn) * t,
            }
        })
        .collect()
}

fn capture_path(
    eng: &mut EngineState,
    render_scale: f32,
    old: MotionPose,
    poses: &[MotionPose],
    mut set_pose: impl FnMut(&mut EngineState, MotionPose),
    mut draw_scene: impl FnMut(&mut EngineState),
) -> Vec<Vec<u8>> {
    eng.renderer.set_render_scale(render_scale);
    eng.renderer.reset_temporal_history();
    for _ in 0..SETTLE_FRAMES {
        set_pose(eng, old);
        render(eng, 1, &mut draw_scene);
    }
    poses
        .iter()
        .map(|pose| {
            set_pose(eng, *pose);
            render(eng, 1, &mut draw_scene).2
        })
        .collect()
}

fn capture_fresh_endpoint(
    eng: &mut EngineState,
    pose: MotionPose,
    frames: usize,
    mut set_pose: impl FnMut(&mut EngineState, MotionPose),
    mut draw_scene: impl FnMut(&mut EngineState),
) -> Vec<Vec<u8>> {
    eng.renderer.set_render_scale(0.75);
    eng.renderer.reset_temporal_history();
    (0..frames)
        .map(|_| {
            set_pose(eng, pose);
            render(eng, 1, &mut draw_scene).2
        })
        .collect()
}

fn mean_motion_derivative_error(native: &[Vec<u8>], fractional: &[Vec<u8>]) -> f64 {
    native
        .windows(2)
        .zip(fractional.windows(2))
        .map(|(native, fractional)| {
            let mut error = 0u64;
            let mut samples = 0u64;
            for (((native_previous, native_current), fractional_previous), fractional_current) in
                native[0]
                    .chunks_exact(4)
                    .zip(native[1].chunks_exact(4))
                    .zip(fractional[0].chunks_exact(4))
                    .zip(fractional[1].chunks_exact(4))
            {
                for channel in 0..3 {
                    let native_delta =
                        i16::from(native_current[channel]) - i16::from(native_previous[channel]);
                    let fractional_delta = i16::from(fractional_current[channel])
                        - i16::from(fractional_previous[channel]);
                    error += u64::from(native_delta.abs_diff(fractional_delta));
                    samples += 1;
                }
            }
            error as f64 / samples as f64
        })
        .sum::<f64>()
        / (native.len() - 1) as f64
}

fn qualify_native_match(
    label: &str,
    native: &[Vec<u8>],
    fractional: &[Vec<u8>],
    fresh_endpoint: &[Vec<u8>],
    negative_reference: Option<&[u8]>,
    bounds: NativeMatchBounds,
) {
    let native_metrics = native
        .iter()
        .zip(fractional)
        .map(|(native, fractional)| calculate_diff_metrics(native, fractional, W, H))
        .collect::<Vec<_>>();
    let recovery_metrics = fresh_endpoint[TRANSITION_FRAMES..]
        .iter()
        .zip(&fractional[TRANSITION_FRAMES..])
        .map(|(fresh, recovered)| calculate_diff_metrics(fresh, recovered, W, H))
        .collect::<Vec<_>>();
    let movement = calculate_diff_metrics(
        negative_reference.unwrap_or(&native[0]),
        if negative_reference.is_some() {
            native.last().unwrap()
        } else {
            &native[TRANSITION_FRAMES - 1]
        },
        W,
        H,
    );
    let mean_rgb = native_metrics
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .sum::<f64>()
        / native_metrics.len() as f64;
    let maximum_mean_rgb = native_metrics
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .fold(0.0f64, f64::max);
    let minimum_ssim = native_metrics
        .iter()
        .map(|metrics| metrics.ssim)
        .fold(1.0f64, f64::min);
    let derivative_error = mean_motion_derivative_error(native, fractional);
    let recovery_mean_rgb = recovery_metrics
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .sum::<f64>()
        / recovery_metrics.len() as f64;
    let recovery_maximum_outliers = recovery_metrics
        .iter()
        .map(|metrics| metrics.outlier_pixel_fraction)
        .fold(0.0f64, f64::max);
    let final_recovery = recovery_metrics.last().unwrap();
    eprintln!(
        "temporal-corpus {label} movement_rgb={:.6} movement_outliers={:.4}% \
         native_mean_rgb={mean_rgb:.6} native_max_rgb={maximum_mean_rgb:.6} \
         native_min_ssim={minimum_ssim:.6} derivative_error={derivative_error:.6} \
         recovery_mean_rgb={recovery_mean_rgb:.6} recovery_max_outliers={:.4}% \
         recovery_final_rgb={:.6} recovery_final_ssim={:.6}",
        movement.mean_rgb,
        movement.outlier_pixel_fraction * 100.0,
        recovery_maximum_outliers * 100.0,
        final_recovery.mean_rgb,
        final_recovery.ssim,
    );

    assert!(
        movement.mean_rgb >= 5.0 && movement.outlier_pixel_fraction >= 0.05,
        "{label} negative control did not produce material object motion: {movement:?}"
    );
    assert!(
        mean_rgb <= bounds.mean_rgb
            && maximum_mean_rgb <= bounds.maximum_mean_rgb
            && minimum_ssim >= bounds.minimum_ssim,
        "{label} fractional reconstruction diverged from native TAA: \
         mean_rgb={mean_rgb:.6}, maximum_mean_rgb={maximum_mean_rgb:.6}, \
         minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        derivative_error <= bounds.derivative_error,
        "{label} fractional reconstruction added excessive derivative error: \
         {derivative_error:.6}"
    );
    assert!(
        recovery_mean_rgb <= bounds.recovery_mean_rgb
            && recovery_maximum_outliers <= bounds.recovery_maximum_outliers
            && final_recovery.mean_rgb <= bounds.final_recovery_mean_rgb
            && final_recovery.mean_rgb <= recovery_metrics[0].mean_rgb * 0.55,
        "{label} retained path-dependent history after object motion: \
         mean={recovery_mean_rgb:.6}, max_outliers={recovery_maximum_outliers:.6}, \
         final={final_recovery:?}"
    );
}

#[test]
fn fractional_textured_rigid_motion_tracks_native_and_fresh_recovery() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let mut texels = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let bright = ((x / 2) + (y / 2)) & 1 == 0;
            texels.extend_from_slice(if bright {
                &[245, 218, 32, 255]
            } else {
                &[18, 42, 230, 255]
            });
        }
    }
    let texture = eng.renderer.register_texture_no_mips(64, 64, &texels);
    let (vertices, indices) = cube_verts(1.0, [1.0; 4]);
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_material_pbr(node, 0.12, 0.38);
    eng.scene.set_material_color(node, 1.0, 1.0, 1.0, 1.0);
    eng.scene.set_material_texture(node, texture);

    let old = MotionPose {
        x: -1.65,
        turn: -0.55,
    };
    let new = MotionPose {
        x: 1.55,
        turn: 0.70,
    };
    let poses = motion_poses(old, new);
    let set_pose = |eng: &mut EngineState, pose| eng.scene.set_transform(node, transform(pose));
    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.4, 7.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.82, 2.2);
        r.draw_grid(64, 0.18);
        r.draw_cube(0.0, 1.2, -2.0, 6.0, 3.4, 0.3, 28.0, 155.0, 225.0, 255.0);
    };

    let native = capture_path(&mut eng, 1.0, old, &poses, set_pose, draw_scene);
    let fractional = capture_path(&mut eng, 0.75, old, &poses, set_pose, draw_scene);
    let fresh_endpoint = capture_fresh_endpoint(&mut eng, new, poses.len(), set_pose, draw_scene);
    qualify_native_match(
        "fractional-textured-rigid",
        &native,
        &fractional,
        &fresh_endpoint,
        None,
        NativeMatchBounds {
            mean_rgb: 0.36,
            maximum_mean_rgb: 0.52,
            minimum_ssim: 0.980,
            derivative_error: 0.30,
            recovery_mean_rgb: 0.17,
            recovery_maximum_outliers: 0.005,
            final_recovery_mean_rgb: 0.13,
        },
    );
}

#[test]
fn fractional_skinned_motion_tracks_native_and_fresh_recovery() {
    const HANDLE: u64 = 0x7AA5_DA11;
    const PALETTE_KEY: u64 = 0x7AA5_DA12;
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
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let mut texels = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let stripe = ((x / 2) + (y / 4)) & 1 == 0;
            texels.extend_from_slice(if stripe {
                &[242, 38, 28, 255]
            } else {
                &[26, 222, 238, 255]
            });
        }
    }
    let texture = eng.renderer.register_texture_no_mips(64, 64, &texels);
    let (mut vertices, indices) = cube_verts(1.0, [1.0; 4]);
    for vertex in &mut vertices {
        vertex.joints = if vertex.position[1] > 0.0 {
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
            texture_idx: Some(texture),
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
    assert!(eng.renderer.is_model_skinned(HANDLE));

    let old = MotionPose {
        x: -1.55,
        turn: -0.62,
    };
    let new = MotionPose {
        x: 1.50,
        turn: 0.82,
    };
    let poses = motion_poses(old, new);
    let current_pose = std::cell::Cell::new(old);
    let set_pose = |_: &mut EngineState, pose| current_pose.set(pose);
    let draw_scene = |eng: &mut EngineState| {
        let pose = current_pose.get();
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.4, 7.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.82, 2.2);
        r.draw_grid(64, 0.18);
        r.draw_cube(0.0, 1.2, -2.0, 6.0, 3.4, 0.3, 28.0, 155.0, 225.0, 255.0);
        let facing = pose.turn * 0.55;
        let (rot_sin, rot_cos) = facing.sin_cos();
        r.set_joint_matrices_scaled(
            PALETTE_KEY,
            &palette(pose.turn),
            1.0,
            [pose.x, 1.0, 0.0],
            rot_sin,
            rot_cos,
        );
        r.draw_model_cached_skinned(HANDLE, [0.0; 3], 1.0, [1.0; 4]);
    };

    let native = capture_path(&mut eng, 1.0, old, &poses, set_pose, draw_scene);
    let fractional = capture_path(&mut eng, 0.75, old, &poses, set_pose, draw_scene);
    let fresh_endpoint = capture_fresh_endpoint(&mut eng, new, poses.len(), set_pose, draw_scene);
    qualify_native_match(
        "fractional-skinned",
        &native,
        &fractional,
        &fresh_endpoint,
        None,
        NativeMatchBounds {
            mean_rgb: 0.36,
            maximum_mean_rgb: 0.50,
            minimum_ssim: 0.980,
            derivative_error: 0.28,
            recovery_mean_rgb: 0.20,
            recovery_maximum_outliers: 0.006,
            final_recovery_mean_rgb: 0.15,
        },
    );
}

#[test]
fn fractional_alpha_tested_motion_tracks_native_and_fresh_recovery() {
    const HANDLE: u64 = 0x7AA5_DA21;
    const TEX_SIZE: u32 = 64;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let mut texels = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let fx = (x as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let fy = (y as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let leaf = fx * fx / 0.82f32.powi(2) + fy * fy / 0.96f32.powi(2) < 1.0;
            let serrated = ((y / 3 + x / 5) & 1) == 0 || fx.abs() < 0.66;
            let veins = (x as i32 - TEX_SIZE as i32 / 2).abs() == 5 && y % 8 < 6;
            let opaque = leaf && serrated && !veins;
            texels.extend_from_slice(if opaque {
                if ((x / 3) + (y / 3)) & 1 == 0 {
                    &[28, 220, 52, 255]
                } else {
                    &[215, 238, 35, 255]
                }
            } else {
                &[0, 0, 0, 0]
            });
        }
    }
    let texture = eng.renderer.register_texture_kind_with_alpha_coverage(
        TEX_SIZE,
        TEX_SIZE,
        &texels,
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
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices: vec![
                vertex([-1.0, 0.0, 0.0], [0.0, 1.0]),
                vertex([1.0, 0.0, 0.0], [1.0, 1.0]),
                vertex([1.0, 2.8, 0.0], [1.0, 0.0]),
                vertex([-1.0, 2.8, 0.0], [0.0, 0.0]),
            ],
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

    let old = MotionPose {
        x: -1.35,
        turn: -0.42,
    };
    let new = MotionPose {
        x: 1.35,
        turn: 0.58,
    };
    let poses = motion_poses(old, new);
    let current_pose = std::cell::Cell::new(old);
    let set_pose = |_: &mut EngineState, pose| current_pose.set(pose);
    let draw_scene = |eng: &mut EngineState| {
        let pose = current_pose.get();
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 25.0, 255.0);
        r.begin_mode_3d(0.0, 2.35, 7.0, 0.0, 1.25, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.35, -1.0, -0.2, 0.9, 1.0, 0.8, 2.0);
        r.draw_grid(64, 0.18);
        r.draw_cube(0.0, 1.5, -1.15, 6.0, 3.4, 0.25, 35.0, 80.0, 125.0, 255.0);
        r.draw_model_cached_rotated(HANDLE, [pose.x, 0.0, 0.0], 1.0, pose.turn, [1.0; 4]);
    };

    let native = capture_path(&mut eng, 1.0, old, &poses, set_pose, draw_scene);
    let fractional = capture_path(&mut eng, 0.75, old, &poses, set_pose, draw_scene);
    let fresh_endpoint = capture_fresh_endpoint(&mut eng, new, poses.len(), set_pose, draw_scene);
    qualify_native_match(
        "fractional-alpha-tested",
        &native,
        &fractional,
        &fresh_endpoint,
        None,
        NativeMatchBounds {
            mean_rgb: 1.10,
            maximum_mean_rgb: 1.30,
            minimum_ssim: 0.980,
            derivative_error: 0.95,
            recovery_mean_rgb: 0.56,
            recovery_maximum_outliers: 0.020,
            final_recovery_mean_rgb: 0.38,
        },
    );
}

#[test]
fn fractional_refractive_motion_tracks_native_and_fresh_recovery() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);
    eng.renderer.set_transparency_composition_mode(0);

    let (vertices, indices) = cube_verts(0.9, [1.0; 4]);
    let glass = eng.scene.create_node();
    eng.scene.update_geometry(glass, vertices, indices);
    eng.scene.set_material_pbr(glass, 0.08, 0.0);
    eng.scene.set_material_transmission(
        glass,
        MaterialTransmission {
            authored: true,
            factor: 1.0,
            ior_authored: true,
            ior: 1.52,
            volume_authored: true,
            thickness_factor: 0.9,
            attenuation_distance: 0.7,
            attenuation_color: [0.08, 0.75, 1.0],
            thickness_source: MaterialThicknessSource::Authored,
            ..Default::default()
        },
    );

    let old = MotionPose {
        x: -1.45,
        turn: -0.48,
    };
    let new = MotionPose {
        x: 1.45,
        turn: 0.62,
    };
    let poses = motion_poses(old, new);
    let set_pose = |eng: &mut EngineState, pose| {
        eng.scene.set_transform(glass, transform(pose));
    };
    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 22.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 7.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.9, 0.75, 1.8);
        r.draw_grid(64, 0.18);
        for y in 0..4 {
            for x in -4..=4 {
                let alternate = (x + y) & 1 == 0;
                r.draw_cube(
                    x as f64 * 0.68,
                    0.38 + y as f64 * 0.62,
                    -1.9,
                    0.56,
                    0.54,
                    0.26,
                    if alternate { 238.0 } else { 24.0 },
                    if alternate { 45.0 } else { 182.0 },
                    if alternate { 24.0 } else { 240.0 },
                    255.0,
                );
            }
        }
    };

    let native = capture_path(&mut eng, 1.0, old, &poses, set_pose, draw_scene);
    let fractional = capture_path(&mut eng, 0.75, old, &poses, set_pose, draw_scene);
    let fresh_endpoint = capture_fresh_endpoint(&mut eng, new, poses.len(), set_pose, draw_scene);
    qualify_native_match(
        "fractional-refractive",
        &native,
        &fractional,
        &fresh_endpoint,
        None,
        NativeMatchBounds {
            mean_rgb: 1.55,
            maximum_mean_rgb: 1.85,
            minimum_ssim: 0.950,
            derivative_error: 1.35,
            recovery_mean_rgb: 0.50,
            recovery_maximum_outliers: 0.022,
            final_recovery_mean_rgb: 0.30,
        },
    );
}

fn capture_switch_path(
    eng: &mut EngineState,
    render_scale: f32,
    from: bool,
    to: bool,
    mut draw_state: impl FnMut(&mut EngineState, bool),
) -> (Vec<u8>, Vec<Vec<u8>>) {
    eng.renderer.set_render_scale(render_scale);
    eng.renderer.reset_temporal_history();
    let mut old_state = Vec::new();
    for frame in 0..SETTLE_FRAMES {
        let captured = render(eng, 1, |eng| draw_state(eng, from)).2;
        if frame + 1 == SETTLE_FRAMES {
            old_state = captured;
        }
    }
    let frames = (0..TRANSITION_FRAMES + RECOVERY_FRAMES)
        .map(|_| render(eng, 1, |eng| draw_state(eng, to)).2)
        .collect();
    (old_state, frames)
}

fn capture_fresh_state(
    eng: &mut EngineState,
    state: bool,
    mut draw_state: impl FnMut(&mut EngineState, bool),
) -> Vec<Vec<u8>> {
    eng.renderer.set_render_scale(0.75);
    eng.renderer.reset_temporal_history();
    (0..TRANSITION_FRAMES + RECOVERY_FRAMES)
        .map(|_| render(eng, 1, |eng| draw_state(eng, state)).2)
        .collect()
}

#[test]
fn fractional_emissive_switch_tracks_native_and_fresh_recovery() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let (vertices, indices) = cube_verts(0.7, [0.18, 0.035, 0.01, 1.0]);
    let emitter = eng.scene.create_node();
    eng.scene.update_geometry(emitter, vertices, indices);
    eng.scene.set_transform(
        emitter,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
    );
    eng.scene.set_material_pbr(emitter, 0.24, 0.0);

    let draw_state = |eng: &mut EngineState, enabled: bool| {
        eng.scene.set_material_emissive_factor(
            emitter,
            if enabled { 8.0 } else { 0.0 },
            if enabled { 0.9 } else { 0.0 },
            if enabled { 0.08 } else { 0.0 },
        );
        let r = &mut eng.renderer;
        r.set_clear_color(5.0, 7.0, 14.0, 255.0);
        r.begin_mode_3d(4.2, 2.8, 6.2, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 0.55, 0.65, 0.9, 0.55);
        if enabled {
            r.add_point_light(0.0, 1.25, 0.2, 4.5, 1.0, 0.11, 0.02, 8.0);
        }
        r.draw_grid(64, 0.18);
        r.draw_cube(-1.4, 0.65, 0.0, 0.8, 1.3, 0.8, 105.0, 115.0, 135.0, 255.0);
        r.draw_cube(1.4, 0.65, 0.0, 0.8, 1.3, 0.8, 105.0, 115.0, 135.0, 255.0);
    };

    for (label, from, to) in [("on", false, true), ("off", true, false)] {
        let (native_old, native) = capture_switch_path(&mut eng, 1.0, from, to, draw_state);
        let (_, fractional) = capture_switch_path(&mut eng, 0.75, from, to, draw_state);
        let fresh_endpoint = capture_fresh_state(&mut eng, to, draw_state);
        let bounds = if to {
            NativeMatchBounds {
                mean_rgb: 0.48,
                maximum_mean_rgb: 0.60,
                minimum_ssim: 0.985,
                derivative_error: 0.22,
                recovery_mean_rgb: 0.40,
                recovery_maximum_outliers: 0.012,
                final_recovery_mean_rgb: 0.28,
            }
        } else {
            NativeMatchBounds {
                mean_rgb: 0.38,
                maximum_mean_rgb: 0.48,
                minimum_ssim: 0.986,
                derivative_error: 0.15,
                recovery_mean_rgb: 0.30,
                recovery_maximum_outliers: 0.009,
                final_recovery_mean_rgb: 0.22,
            }
        };
        qualify_native_match(
            &format!("fractional-emissive-{label}"),
            &native,
            &fractional,
            &fresh_endpoint,
            Some(&native_old),
            bounds,
        );
    }
}

fn capture_continuous_path(
    eng: &mut EngineState,
    render_scale: f32,
    frames: usize,
    mut draw_scene: impl FnMut(&mut EngineState),
) -> Vec<Vec<u8>> {
    eng.set_quality_fixed_timestep(Some(1.0 / 60.0));
    eng.target_fps = 0.0;
    eng.renderer.set_render_scale(render_scale);
    eng.renderer.reset_temporal_history();
    for _ in 0..SETTLE_FRAMES {
        render(eng, 1, &mut draw_scene);
    }
    (0..frames)
        .map(|_| render(eng, 1, &mut draw_scene).2)
        .collect()
}

#[test]
fn fractional_procedural_foliage_tracks_native_at_fixed_time() {
    const HANDLE: u64 = 0x7AA5_DA31;
    const TEX_SIZE: u32 = 64;
    const FRAMES: usize = 16;

    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let mut texels = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let fx = (x as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let fy = (y as f32 + 0.5) / TEX_SIZE as f32 * 2.0 - 1.0;
            let leaf = fx * fx / 0.78f32.powi(2) + fy * fy / 0.98f32.powi(2) < 1.0;
            let serrated = ((y / 3 + x / 5) & 1) == 0 || fx.abs() < 0.63;
            let vein_gap = (x as i32 - TEX_SIZE as i32 / 2).abs() == 5 && y % 8 < 6;
            let opaque = leaf && serrated && !vein_gap;
            texels.extend_from_slice(if opaque {
                if ((x / 2) + (y / 3)) & 1 == 0 {
                    &[24, 218, 48, 255]
                } else {
                    &[205, 235, 34, 255]
                }
            } else {
                &[0, 0, 0, 0]
            });
        }
    }
    let texture = eng.renderer.register_texture_kind_with_alpha_coverage(
        TEX_SIZE,
        TEX_SIZE,
        &texels,
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
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices: vec![
                vertex([-1.15, 0.0, 0.0], [0.0, 1.0]),
                vertex([1.15, 0.0, 0.0], [1.0, 1.0]),
                vertex([1.15, 4.2, 0.0], [1.0, 0.0]),
                vertex([-1.15, 4.2, 0.0], [0.0, 0.0]),
            ],
            secondary_tex_coords: None,
            indices: vec![0, 1, 2, 0, 2, 3],
            texture_idx: Some(texture),
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 0.78,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Mask,
            alpha_cutoff: 0.5,
            alpha_coverage_mips: true,
            double_sided: true,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        }]
    ));
    eng.renderer.set_wind(1.0, 0.3, 0.55, 1.35);
    eng.renderer.set_model_foliage_wind(HANDLE, 1.0);

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 25.0, 255.0);
        r.begin_mode_3d(0.0, 2.4, 8.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 46.0, 0.0);
        r.add_directional_light(-0.35, -1.0, -0.2, 0.9, 1.0, 0.8, 2.0);
        r.draw_grid(64, 0.18);
        r.draw_cube(0.0, 2.1, -1.2, 6.0, 4.6, 0.25, 35.0, 80.0, 125.0, 255.0);
        r.draw_model_cached(HANDLE, [0.0; 3], 1.0, [1.0; 4]);
    };

    let native = capture_continuous_path(&mut eng, 1.0, FRAMES, draw_scene);
    let fractional = capture_continuous_path(&mut eng, 0.75, FRAMES, draw_scene);
    let fractional_repeat = capture_continuous_path(&mut eng, 0.75, FRAMES, draw_scene);
    assert_eq!(
        fractional, fractional_repeat,
        "fixed quality clock did not replay procedural foliage byte-for-byte"
    );

    let metrics = native
        .iter()
        .zip(&fractional)
        .map(|(native, fractional)| calculate_diff_metrics(native, fractional, W, H))
        .collect::<Vec<_>>();
    let movement = calculate_diff_metrics(native.first().unwrap(), native.last().unwrap(), W, H);
    let mean_rgb = metrics.iter().map(|metrics| metrics.mean_rgb).sum::<f64>() / FRAMES as f64;
    let maximum_mean_rgb = metrics
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .fold(0.0f64, f64::max);
    let minimum_ssim = metrics
        .iter()
        .map(|metrics| metrics.ssim)
        .fold(1.0f64, f64::min);
    let derivative_error = mean_motion_derivative_error(&native, &fractional);
    eprintln!(
        "temporal-corpus fractional-procedural-foliage movement_rgb={:.6} \
         movement_outliers={:.4}% native_mean_rgb={mean_rgb:.6} \
         native_max_rgb={maximum_mean_rgb:.6} native_min_ssim={minimum_ssim:.6} \
         derivative_error={derivative_error:.6}",
        movement.mean_rgb,
        movement.outlier_pixel_fraction * 100.0,
    );
    assert!(
        movement.mean_rgb >= 0.5 && movement.outlier_pixel_fraction >= 0.005,
        "procedural foliage negative control did not produce material deformation: {movement:?}"
    );
    assert!(
        mean_rgb <= 0.92 && maximum_mean_rgb <= 1.35 && minimum_ssim >= 0.985,
        "fractional procedural foliage diverged from native TAA: \
         mean_rgb={mean_rgb:.6}, maximum_mean_rgb={maximum_mean_rgb:.6}, \
         minimum_ssim={minimum_ssim:.6}"
    );
    assert!(
        derivative_error <= 1.15,
        "fractional procedural foliage added excessive derivative error: {derivative_error:.6}"
    );
}
