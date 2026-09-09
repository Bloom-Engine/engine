use super::*;

const NORMAL_MAP_SIZE: u32 = 64;
const MOTION_FRAMES: usize = 16;

fn normal_map_texels(filtered_variance: bool) -> Vec<u8> {
    let mut texels = Vec::with_capacity((NORMAL_MAP_SIZE * NORMAL_MAP_SIZE * 4) as usize);
    for y in 0..NORMAL_MAP_SIZE {
        for x in 0..NORMAL_MAP_SIZE {
            if filtered_variance {
                // Unit normals approximately (+/-0.8, 0, 0.6), arranged in
                // 2x2 islands so camera motion crosses several authored
                // directions per pixel before the vector/variance mips filter
                // them.
                let positive = ((x / 2) + (y / 2)) & 1 == 0;
                texels.extend_from_slice(if positive {
                    &[230, 128, 204, 255]
                } else {
                    &[26, 128, 204, 255]
                });
            } else {
                texels.extend_from_slice(&[128, 128, 255, 255]);
            }
        }
    }
    texels
}

fn base_normal_map_texels(filtered_variance: bool) -> Vec<u8> {
    let mut texels = Vec::with_capacity((NORMAL_MAP_SIZE * NORMAL_MAP_SIZE * 4) as usize);
    for y in 0..NORMAL_MAP_SIZE {
        for x in 0..NORMAL_MAP_SIZE {
            if filtered_variance {
                // Eight-texel islands deliberately straddle the material's
                // native screen footprint: the footprint-selected mip keeps
                // their mapped direction, while an extra +1 LOD collapses
                // adjacent islands to the flat mean. This makes the oracle
                // sensitive to accidental one-mip over-filtering without
                // relying on unstable single-texel detail.
                let positive = ((x / 8) + (y / 8)) & 1 == 0;
                texels.extend_from_slice(if positive {
                    &[230, 128, 204, 255]
                } else {
                    &[26, 128, 204, 255]
                });
            } else {
                texels.extend_from_slice(&[128, 128, 255, 255]);
            }
        }
    }
    texels
}

fn textured_quad() -> (Vec<Vertex3D>, Vec<u32>) {
    let positions = [
        [-2.0, -2.0, 0.0],
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [-2.0, 2.0, 0.0],
    ];
    let uvs = [[0.0, 0.0], [24.0, 0.0], [24.0, 24.0], [0.0, 24.0]];
    let vertices = positions
        .into_iter()
        .zip(uvs)
        .map(|(position, uv)| Vertex3D {
            position,
            normal: [0.0, 0.0, 1.0],
            color: [0.32, 0.34, 0.38, 1.0],
            uv,
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [1.0, 0.0, 0.0, 1.0],
        })
        .collect();
    (vertices, vec![0, 1, 2, 0, 2, 3])
}

fn render_motion_sequence(filtered_variance: bool) -> Option<Vec<Vec<u8>>> {
    let mut eng = try_engine()?;
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let texels = normal_map_texels(filtered_variance);
    let normal_texture =
        eng.renderer
            .register_texture_kind(NORMAL_MAP_SIZE, NORMAL_MAP_SIZE, &texels, true);
    let (vertices, indices) = textured_quad();
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_material_pbr(node, 0.62, 0.0);
    eng.scene.set_material_layered_pbr(
        node,
        MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 1.0,
            clearcoat_roughness_factor: 0.04,
            clearcoat_normal_scale: 1.0,
            clearcoat_normal_texture: Some(MaterialTextureBinding {
                source_texture_index: 0,
                source_image_index: 0,
                runtime_texture_idx: Some(normal_texture),
                transform: Default::default(),
            }),
            ..Default::default()
        },
    );

    let draw_pose = |eng: &mut EngineState, camera_x: f32| {
        let renderer = &mut eng.renderer;
        renderer.set_clear_color(0.012, 0.016, 0.025, 1.0);
        renderer.begin_mode_3d(camera_x, 0.18, 5.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        renderer.set_ambient_light(28.0, 32.0, 42.0, 0.18);
        renderer.set_directional_light(-0.55, 0.35, 0.76, 255.0, 246.0, 230.0, 4.0);
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        eng.begin_frame();
        draw_pose(&mut eng, -0.06);
        eng.end_frame();
    }

    let mut frames = Vec::with_capacity(MOTION_FRAMES);
    for frame in 0..MOTION_FRAMES {
        let camera_x = -0.06 + frame as f32 * (0.12 / (MOTION_FRAMES - 1) as f32);
        frames.push(render(&mut eng, 1, |eng| draw_pose(eng, camera_x)).2);
    }
    Some(frames)
}

fn render_base_normal_motion_sequence(filtered_variance: bool) -> Option<Vec<Vec<u8>>> {
    let mut eng = try_engine()?;
    temporal_history::configure_taa_motion_corpus(&mut eng.renderer);

    let texels = base_normal_map_texels(filtered_variance);
    let normal_texture =
        eng.renderer
            .register_texture_kind(NORMAL_MAP_SIZE, NORMAL_MAP_SIZE, &texels, true);
    let (vertices, indices) = textured_quad();
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_material_pbr(node, 0.24, 0.0);
    eng.scene.set_material_normal_texture(node, normal_texture);

    let draw_pose = |eng: &mut EngineState, camera_x: f32| {
        let renderer = &mut eng.renderer;
        renderer.set_clear_color(0.012, 0.016, 0.025, 1.0);
        renderer.begin_mode_3d(camera_x, 0.18, 5.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        renderer.set_ambient_light(28.0, 32.0, 42.0, 0.18);
        renderer.set_directional_light(-0.55, 0.35, 0.76, 255.0, 246.0, 230.0, 4.0);
    };

    eng.renderer.reset_temporal_history();
    for _ in 0..8 {
        eng.begin_frame();
        draw_pose(&mut eng, -0.06);
        eng.end_frame();
    }

    let mut frames = Vec::with_capacity(MOTION_FRAMES);
    for frame in 0..MOTION_FRAMES {
        let camera_x = -0.06 + frame as f32 * (0.12 / (MOTION_FRAMES - 1) as f32);
        frames.push(render(&mut eng, 1, |eng| draw_pose(eng, camera_x)).2);
    }
    Some(frames)
}

#[derive(Clone, Copy, Debug)]
struct ResidualMetrics {
    mean_rgb: f64,
    outlier_fraction: f64,
}

fn normal_response_residual(
    previous_mapped: &[u8],
    mapped: &[u8],
    previous_flat: &[u8],
    flat: &[u8],
) -> ResidualMetrics {
    let mut sum = 0u64;
    let mut samples = 0u64;
    let mut outliers = 0u64;
    // Ignore the outer frame where clear colour dominates. This region
    // remains large enough to include both the coat highlight and moving
    // geometry edges on every qualified backend.
    for y in 36..H - 36 {
        for x in 36..W - 36 {
            let pixel = ((y * W + x) * 4) as usize;
            for channel in 0..3 {
                let previous_response = i16::from(previous_mapped[pixel + channel])
                    - i16::from(previous_flat[pixel + channel]);
                let response =
                    i16::from(mapped[pixel + channel]) - i16::from(flat[pixel + channel]);
                let delta = response.abs_diff(previous_response);
                sum += u64::from(delta);
                samples += 1;
                outliers += u64::from(delta > 16);
            }
        }
    }
    ResidualMetrics {
        mean_rgb: sum as f64 / samples as f64,
        outlier_fraction: outliers as f64 / samples as f64,
    }
}

#[test]
fn clearcoat_normal_minification_bounds_motion_sparkle() {
    let Some(mapped) = render_motion_sequence(true) else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let flat = render_motion_sequence(false).expect("same GPU adapter remains available");

    let motion = calculate_diff_metrics(&flat[0], flat.last().unwrap(), W, H);
    let response = mapped
        .iter()
        .zip(&flat)
        .map(|(mapped, flat)| calculate_diff_metrics(flat, mapped, W, H).mean_rgb)
        .sum::<f64>()
        / MOTION_FRAMES as f64;
    let residuals = (1..MOTION_FRAMES)
        .map(|frame| {
            normal_response_residual(
                &mapped[frame - 1],
                &mapped[frame],
                &flat[frame - 1],
                &flat[frame],
            )
        })
        .collect::<Vec<_>>();
    let max_residual_mean = residuals
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .fold(0.0f64, f64::max);
    let max_residual_outliers = residuals
        .iter()
        .map(|metrics| metrics.outlier_fraction)
        .fold(0.0f64, f64::max);

    eprintln!(
        "layered-normal-motion motion_mean={:.6} response_mean={response:.6} \
         max_residual_mean={max_residual_mean:.6} max_residual_outliers={:.4}%",
        motion.mean_rgb,
        max_residual_outliers * 100.0,
    );
    assert!(
        motion.mean_rgb >= 0.05,
        "normal-minification corpus did not exercise visible camera motion: {motion:?}"
    );
    assert!(
        response >= 0.1,
        "variance-bearing clearcoat normals did not produce a visible filtered response"
    );
    assert!(
        max_residual_mean <= 1.5,
        "minified clearcoat-normal response flickered under motion: {residuals:?}"
    );
    assert!(
        max_residual_outliers <= 0.02,
        "coherent clearcoat-normal sparkle exceeded 2% of sampled channels: {residuals:?}"
    );
}

#[test]
fn base_normal_minification_preserves_response_without_motion_sparkle() {
    let Some(mapped) = render_base_normal_motion_sequence(true) else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let flat =
        render_base_normal_motion_sequence(false).expect("same GPU adapter remains available");

    let response = mapped
        .iter()
        .zip(&flat)
        .map(|(mapped, flat)| calculate_diff_metrics(flat, mapped, W, H).mean_rgb)
        .sum::<f64>()
        / MOTION_FRAMES as f64;
    let residuals = (1..MOTION_FRAMES)
        .map(|frame| {
            normal_response_residual(
                &mapped[frame - 1],
                &mapped[frame],
                &flat[frame - 1],
                &flat[frame],
            )
        })
        .collect::<Vec<_>>();
    let max_residual_mean = residuals
        .iter()
        .map(|metrics| metrics.mean_rgb)
        .fold(0.0f64, f64::max);
    let max_residual_outliers = residuals
        .iter()
        .map(|metrics| metrics.outlier_fraction)
        .fold(0.0f64, f64::max);

    eprintln!(
        "base-normal-motion response_mean={response:.6} \
         max_residual_mean={max_residual_mean:.6} max_residual_outliers={:.4}%",
        max_residual_outliers * 100.0,
    );
    assert!(
        response >= 1.5,
        "base-normal sampling collapsed stable authored response into an over-filtered mip"
    );
    assert!(
        max_residual_mean <= 1.5,
        "minified base-normal response flickered under motion: {residuals:?}"
    );
    assert!(
        max_residual_outliers <= 0.02,
        "coherent base-normal sparkle exceeded 2% of sampled channels: {residuals:?}"
    );
}
