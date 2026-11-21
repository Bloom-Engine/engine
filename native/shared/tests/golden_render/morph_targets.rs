//! Morph targets: a morphed skinned mesh at weights 0 and 1 (golden), the
//! CPU path against the GPU pass, culling bounds that hold every pose, the
//! cost of an instance at zero, and motion vectors of a morph alone.

use super::temporal_history::{configure_taa_motion_corpus, evaluate_motion_recovery};
#[allow(unused_imports)]
use super::*;
use bloom_shared::morph_targets::{MorphTargets, MorphTargetsBuilder};
use std::sync::Arc;

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const GRID: usize = 24;

fn mesh(vertices: Vec<Vertex3D>, indices: Vec<u32>, morph: Option<Arc<MorphTargets>>) -> MeshData {
    MeshData {
        vertices,
        secondary_tex_coords: None,
        indices,
        texture_idx: None,
        normal_texture_idx: None,
        metallic_roughness_texture_idx: None,
        specular_glossiness_factor: None,
        emissive_texture_idx: None,
        occlusion_texture_idx: None,
        metallic_factor: 0.0,
        roughness_factor: 0.55,
        emissive_factor: [0.0; 3],
        alpha_mode: MaterialAlphaMode::Opaque,
        alpha_cutoff: 0.0,
        alpha_coverage_mips: false,
        double_sided: true,
        transmission: Default::default(),
        layered_pbr: Default::default(),
        morph,
    }
}

/// A skinned 2 x 2 m sheet facing the camera (GRID x GRID quads), and two
/// targets: "bulge" pushes a dome out of its middle (with the dome's normal
/// deltas), "shift" slides its top band sideways.
fn sheet() -> (Vec<Vertex3D>, Vec<u32>, Arc<MorphTargets>) {
    let n = GRID + 1;
    let mut vertices = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let x = i as f32 / GRID as f32 * 2.0 - 1.0;
            let y = j as f32 / GRID as f32 * 2.0;
            vertices.push(Vertex3D {
                position: [x, y, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [0.92, 0.55, 0.32, 1.0],
                uv: [0.0; 2],
                joints: [0.0; 4],
                weights: [1.0, 0.0, 0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
        }
    }
    let mut indices = Vec::new();
    for j in 0..GRID {
        for i in 0..GRID {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    let mut bulge_p = vec![[0.0f32; 3]; n * n];
    let mut bulge_n = vec![[0.0f32; 3]; n * n];
    let mut shift = vec![[0.0f32; 3]; n * n];
    for (k, v) in vertices.iter().enumerate() {
        let (x, y) = (v.position[0], v.position[1] - 1.0);
        let r2 = (x * x + y * y) / 0.36;
        if r2 < 1.0 {
            let h = 0.55 * (1.0 - r2).powi(2);
            // dh/dx, dh/dy of h = 0.55 (1 - r2)^2
            let dhdx = 0.55 * 2.0 * (1.0 - r2) * (-2.0 * x / 0.36);
            let dhdy = 0.55 * 2.0 * (1.0 - r2) * (-2.0 * y / 0.36);
            let nrm = [-dhdx, -dhdy, 1.0];
            let l = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + 1.0).sqrt();
            bulge_p[k] = [0.0, 0.0, h];
            bulge_n[k] = [nrm[0] / l, nrm[1] / l, 1.0 / l - 1.0];
        }
        if v.position[1] > 1.5 {
            shift[k] = [0.35 * (v.position[1] - 1.5) * 2.0, 0.0, 0.0];
        }
    }
    let mut b = MorphTargetsBuilder::new(n * n);
    b.add_target(0, Some(&bulge_p), Some(&bulge_n), None, |_, d| d);
    b.add_target(1, Some(&shift), None, None, |_, d| d);
    let names = Arc::new(vec!["bulge".to_string(), "shift".to_string()]);
    let morph = Arc::new(b.build(names).expect("targets move vertices"));
    (vertices, indices, morph)
}

fn scene(eng: &mut EngineState) {
    let r = &mut eng.renderer;
    r.set_clear_color(18.0, 22.0, 30.0, 255.0);
    r.begin_mode_3d(1.6, 1.6, 4.4, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 42.0, 0.0);
    r.add_directional_light(-0.5, -0.6, -0.8, 1.0, 0.96, 0.9, 2.6);
    r.draw_plane(0.0, 0.0, 0.0, 8.0, 8.0, 40.0, 44.0, 52.0, 255.0);
}

fn draw_morphed(eng: &mut EngineState, handle: u64, key: u64, weights: &[f32], x: f32) {
    eng.renderer
        .set_joint_matrices_scaled(key, &[IDENTITY], 1.0, [x, 0.0, 0.0], 0.0, 1.0);
    eng.renderer.stage_morph_weights(key, weights);
    eng.renderer
        .draw_model_cached_skinned(handle, [0.0; 3], 1.0, [1.0; 4]);
}

fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| (0..3).any(|c| (p[c] as i32 - q[c] as i32).abs() > 24))
        .count()
}

#[test]
fn morphed_skinned_mesh_at_zero_and_one_and_the_cpu_path_agrees() {
    const HANDLE: u64 = 0x3044_0001;
    const PLAIN: u64 = 0x3044_0002;
    const KEY: u64 = 0x3044_1001;
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let (vertices, indices, morph) = sheet();
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[mesh(vertices.clone(), indices.clone(), Some(morph))]
    ));
    assert!(eng
        .renderer
        .cache_model_if_static(PLAIN, &[mesh(vertices, indices, None)]));

    let shot = |eng: &mut EngineState, handle: u64, w: &[f32]| {
        render(eng, 3, |eng| {
            scene(eng);
            draw_morphed(eng, handle, KEY, w, 0.0);
        })
    };
    let (w, h, rest) = shot(&mut eng, HANDLE, &[0.0, 0.0]);
    // At zero (after the frame that returns to zero) nothing is morphed.
    assert_eq!(eng.renderer.morph_frame_stats().jobs, 0);
    let (_, _, plain) = shot(&mut eng, PLAIN, &[0.0, 0.0]);
    assert_eq!(
        differing_pixels(&rest, &plain),
        0,
        "weights 0 must draw the bind pose exactly"
    );
    compare_or_update("morph_targets_w0", w, h, &rest);

    let (_, _, open) = shot(&mut eng, HANDLE, &[1.0, 1.0]);
    let stats = eng.renderer.morph_frame_stats();
    assert_eq!(stats.jobs, 1, "{stats:?}");
    assert!(stats.vertices as usize >= (GRID + 1) * (GRID + 1));
    let moved = differing_pixels(&rest, &open);
    eprintln!("morph: {moved} pixels differ between weights 0 and 1 ({stats:?})");
    assert!(moved > 800, "the morph did not change the picture: {moved}");
    compare_or_update("morph_targets_w1", w, h, &open);

    // The CPU path (WebGL2 tier) draws the same.
    if eng.renderer.morph_frame_stats().gpu {
        eng.renderer.set_morph_cpu_fallback(true);
        let (_, _, cpu) = shot(&mut eng, HANDLE, &[1.0, 1.0]);
        assert!(!eng.renderer.morph_frame_stats().gpu);
        let metrics = calculate_diff_metrics(&open, &cpu, w, h);
        assert!(
            metrics.mean_rgb < 0.5 && metrics.outlier_pixel_fraction < 0.001,
            "CPU morph differs from the GPU pass: {metrics:?}"
        );
        eng.renderer.set_morph_cpu_fallback(false);
    }

    // Back to zero: the instance falls back to the shared buffer.
    let (_, _, back) = shot(&mut eng, HANDLE, &[0.0, 0.0]);
    assert_eq!(differing_pixels(&rest, &back), 0);
    assert_eq!(eng.renderer.morph_frame_stats().jobs, 0);
}

#[test]
fn bounds_hold_every_pose_a_morph_can_reach() {
    const HANDLE: u64 = 0x3044_0003;
    const KEY: u64 = 0x3044_1003;
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    // A cube far left of the view; its one target slides it into the
    // middle. Culling by the rest bounds would drop it.
    let (mut vertices, indices) = cube_verts(0.45, [0.2, 0.85, 0.35, 1.0]);
    for v in &mut vertices {
        v.position[0] -= 9.0;
        v.position[1] += 0.8;
        v.weights = [1.0, 0.0, 0.0, 0.0];
    }
    let slide = vec![[9.0f32, 0.0, 0.0]; vertices.len()];
    let mut b = MorphTargetsBuilder::new(vertices.len());
    b.add_target(0, Some(&slide), None, None, |_, d| d);
    let morph = Arc::new(b.build(Arc::new(vec!["slide".into()])).unwrap());
    assert!(eng
        .renderer
        .cache_model_if_static(HANDLE, &[mesh(vertices, indices, Some(morph))]));
    let shot = |eng: &mut EngineState, w: f32| {
        render(eng, 3, |eng| {
            scene(eng);
            draw_morphed(eng, HANDLE, KEY, &[w], 0.0);
        })
        .2
    };
    let away = shot(&mut eng, 0.0);
    let here = shot(&mut eng, 1.0);
    let visible = differing_pixels(&away, &here);
    assert!(
        visible > 300,
        "the morphed cube was culled: {visible} pixels"
    );
}

#[test]
fn a_morph_alone_writes_velocity_and_leaves_no_trail() {
    const HANDLE: u64 = 0x3044_0004;
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    configure_taa_motion_corpus(&mut eng.renderer);
    // A block that a morph (no joint, no model motion) moves 1.6 m sideways.
    let (mut vertices, indices) = cube_verts(0.7, [0.95, 0.12, 0.035, 1.0]);
    for v in &mut vertices {
        v.position[0] -= 0.8;
        v.position[1] += 1.0;
        v.weights = [1.0, 0.0, 0.0, 0.0];
    }
    let slide = vec![[1.6f32, 0.0, 0.0]; vertices.len()];
    let mut b = MorphTargetsBuilder::new(vertices.len());
    b.add_target(0, Some(&slide), None, None, |_, d| d);
    let morph = Arc::new(b.build(Arc::new(vec!["slide".into()])).unwrap());
    assert!(eng
        .renderer
        .cache_model_if_static(HANDLE, &[mesh(vertices, indices, Some(morph))]));

    let draw_scene = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(7.0, 10.0, 20.0, 255.0);
        r.begin_mode_3d(0.0, 2.2, 6.5, 0.0, 0.8, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.25, 1.0, 0.95, 0.88, 2.2);
        r.draw_plane(0.0, 0.0, 0.0, 12.0, 12.0, 30.0, 38.0, 52.0, 255.0);
        r.draw_cube(0.0, 1.1, -1.8, 5.0, 3.2, 0.35, 30.0, 170.0, 235.0, 255.0);
    };
    // `key_of(frame)`: the same key every frame is one instance whose last
    // weights are known; a new key every frame is a control with no history.
    let mut run = |label: &str, keyed: bool| -> usize {
        let mut frame = 0u64;
        let mut capture = |eng: &mut EngineState, w: f32| {
            frame += 1;
            let key = if keyed {
                0x3044_2000
            } else {
                0x3044_3000 + frame
            };
            render(eng, 1, |eng| {
                draw_scene(eng);
                draw_morphed(eng, HANDLE, key, &[w], 0.0);
            })
            .2
        };
        eng.renderer.reset_temporal_history();
        for _ in 0..8 {
            capture(&mut eng, 0.05);
        }
        let old_pose = capture(&mut eng, 0.05);
        let directory =
            std::env::temp_dir().join(format!("bloom-morph-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        eng.renderer.pending_quality_capture_dir = Some(directory.to_string_lossy().into_owned());
        let mut frames = Vec::new();
        for _ in 0..24 {
            frames.push(capture(&mut eng, 1.0));
        }
        let motion = image::open(directory.join("taa-motion.png"))
            .expect("morph capture did not emit the TAA velocity map")
            .to_rgb8();
        let moving = motion.pixels().filter(|p| p[2] > 8).count();
        eprintln!("temporal-corpus morph-{label} moving_pixels={moving}");
        if keyed {
            evaluate_motion_recovery("morph-only", &old_pose, &frames);
        }
        let _ = std::fs::remove_dir_all(&directory);
        moving
    };
    let keyed = run("keyed", true);
    let control = run("unkeyed", false);
    assert!(
        keyed >= 250,
        "a morph's motion wrote no meaningful velocity: {keyed} pixels"
    );
    assert!(
        control * 4 < keyed,
        "the velocity did not come from the morph history (control {control}, keyed {keyed})"
    );
}

/// Cost of the morph pass: 8 characters of 50,176 vertices, 12,000 of them
/// touched by a face rig of 56 targets (20 targets on each touched vertex),
/// every weight non-zero and changing each frame. Prints the median GPU time
/// of the pass. `cargo test --release --test golden_render bench_morph -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_morph_pass_eight_faces() {
    const SIDE: usize = 224;
    let touched_env = std::env::var("MB_TOUCHED")
        .ok()
        .and_then(|v| v.parse().ok());
    let TOUCHED: usize = touched_env.unwrap_or(12_000);
    const TARGETS: usize = 56;
    let PER_VERTEX: usize = std::env::var("MB_PER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    if !eng.renderer.set_morph_gpu_timing(true) {
        eprintln!("skip: no timestamp queries");
        return;
    }
    let n = SIDE * SIDE;
    let vertices: Vec<Vertex3D> = (0..n)
        .map(|i| Vertex3D {
            position: [(i % SIDE) as f32 * 0.01, (i / SIDE) as f32 * 0.01, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [1.0; 4],
            uv: [0.0; 2],
            joints: [0.0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        })
        .collect();
    let mut seed = 0x2545_f491_u32;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut targets = vec![(vec![[0.0f32; 3]; n], vec![[0.0f32; 3]; n]); TARGETS];
    for v in 0..TOUCHED {
        for _ in 0..PER_VERTEX {
            let t = rand() as usize % TARGETS;
            let d = (rand() % 1000) as f32 * 1e-5 + 1e-4;
            targets[t].0[v] = [d, -d, d * 0.5];
            targets[t].1[v] = [0.0, d, -d];
        }
    }
    let mut b = MorphTargetsBuilder::new(n);
    for (slot, (p, nr)) in targets.iter().enumerate() {
        b.add_target(slot as u32, Some(p), Some(nr), Some(nr), |_, d| d);
    }
    let names = Arc::new((0..TARGETS).map(|i| format!("t{i}")).collect::<Vec<_>>());
    let morph = Arc::new(b.build(names).unwrap());
    eprintln!(
        "bench mesh: {n} vertices, {} touched, {} entries, {:.1} MB of deltas",
        morph.touched_vertices,
        morph.entries.len(),
        morph.byte_size() as f64 / 1e6
    );
    const HANDLE: u64 = 0x3044_0B00;
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[mesh(vertices, vec![0, 1, SIDE as u32], Some(morph))]
    ));
    let mut times = Vec::new();
    for frame in 0..90 {
        let weights: Vec<f32> = (0..TARGETS)
            .map(|t| 0.5 + 0.5 * ((frame + t) as f32 * 0.3).sin())
            .collect();
        render(&mut eng, 1, |eng| {
            scene(eng);
            for c in 0..8u64 {
                draw_morphed(eng, HANDLE, 0x3044_0C00 + c, &weights, c as f32 * 0.1);
            }
        });
        let stats = eng.renderer.morph_frame_stats();
        assert_eq!(stats.jobs, 8);
        if frame >= 10 {
            times.push(stats.gpu_ms);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "bench morph pass, 8 x {n} vertices: median {:.3} ms, min {:.3} ms, p90 {:.3} ms (GPU)",
        times[times.len() / 2],
        times[0],
        times[times.len() * 9 / 10]
    );
}
