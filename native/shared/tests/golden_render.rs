//! Golden-image regression tests — render small reference scenes through
//! the real engine pipeline (headless) and compare against checked-in
//! PNGs.
//!
//! These exist to make renderer architecture work safe: clustered
//! lighting, the render-graph migration, pass reordering — any change
//! that should be pixel-neutral gets caught here if it isn't, and any
//! intentional visual change shows up as an explicit golden update in
//! the diff.
//!
//! - Runs on any machine/CI runner with a GPU adapter (CI: the macos-14
//!   shared-tests job); skips gracefully without one.
//! - TAA is disabled in the scenes (sub-pixel jitter is intentionally
//!   non-deterministic across frame counts); a fixed number of warm-up
//!   frames settles the temporal passes that remain.
//! - Tolerances absorb GPU-family rasterization differences; goldens are
//!   regenerated with BLOOM_UPDATE_GOLDEN=1 `cargo test golden`.

use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;

const W: u32 = 256;
const H: u32 = 256;
/// Mean absolute per-channel difference (0..255 scale) allowed before a
/// test fails. Cross-GPU rasterization differences land well under 1.0;
/// real regressions (missing pass, broken lighting) land far above.
const MEAN_TOLERANCE: f64 = 2.0;
/// Fraction of pixels allowed to differ by more than 32/255 — absorbs
/// single-pixel edge flicker without letting a broken region through.
const OUTLIER_FRACTION: f64 = 0.01;

fn try_engine() -> Option<EngineState> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    // Software rasterizers (WARP on the Windows CI runners, llvmpipe on
    // Linux) are not regression targets — WARP crashes outright in the
    // surface-less path, and software fidelity differs from the real
    // GPUs the goldens were generated on. Real-GPU coverage comes from
    // the macos-14 runners.
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return None;
    }
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
    let renderer = Renderer::new_headless(device, queue, W, H);
    let mut eng = EngineState::new(renderer);
    // Deterministic output: no sub-pixel jitter accumulation.
    eng.renderer.set_taa_enabled(false);
    Some(eng)
}

/// Render `frames` frames of `draw`, capturing the last one as RGBA.
fn render(eng: &mut EngineState, frames: u32, mut draw: impl FnMut(&mut EngineState)) -> (u32, u32, Vec<u8>) {
    let mut shot = None;
    for i in 0..frames {
        eng.begin_frame();
        draw(eng);
        if i + 1 == frames {
            eng.renderer.screenshot_requested = true;
        }
        eng.end_frame();
        if i + 1 == frames {
            shot = eng.renderer.screenshot_data.take();
        }
    }
    let (w, h, mut data) =
        shot.expect("screenshot capture produced no data — headless target path broken");
    // screenshot_data is raw surface-format bytes; swizzle BGRA-family
    // formats to RGBA so goldens are stored in a fixed channel order.
    if matches!(
        eng.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for px in data.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    (w, h, data)
}

fn compare_or_update(name: &str, width: u32, height: u32, rgba: &[u8]) {
    let golden_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let path = golden_dir.join(format!("{name}.png"));
    let update = std::env::var("BLOOM_UPDATE_GOLDEN").map(|v| v == "1").unwrap_or(false);

    if update || !path.exists() {
        std::fs::create_dir_all(&golden_dir).unwrap();
        image::save_buffer(&path, rgba, width, height, image::ColorType::Rgba8).unwrap();
        if !update {
            panic!(
                "golden {name} did not exist — wrote it; verify the image looks right and commit it"
            );
        }
        eprintln!("golden {name} updated");
        return;
    }

    let golden = image::open(&path).unwrap().to_rgba8();
    assert_eq!(
        (golden.width(), golden.height()),
        (width, height),
        "golden {name} size mismatch"
    );
    let gold = golden.as_raw();
    let mut sum_abs: f64 = 0.0;
    let mut outliers: usize = 0;
    let mut max_diff: u8 = 0;
    for (a, b) in rgba.iter().zip(gold.iter()) {
        let d = a.abs_diff(*b);
        sum_abs += d as f64;
        if d > 32 {
            outliers += 1;
        }
        max_diff = max_diff.max(d);
    }
    let mean = sum_abs / rgba.len() as f64;
    let outlier_frac = outliers as f64 / rgba.len() as f64;
    // On failure, write the actual image next to the golden for diffing.
    if mean > MEAN_TOLERANCE || outlier_frac > OUTLIER_FRACTION {
        let actual = golden_dir.join(format!("{name}.actual.png"));
        image::save_buffer(&actual, rgba, width, height, image::ColorType::Rgba8).unwrap();
        panic!(
            "golden {name} mismatch: mean diff {mean:.3} (tol {MEAN_TOLERANCE}), \
             outliers {:.4}% (tol {:.4}%), max {max_diff}. Actual written to {actual:?}; \
             if the change is intentional, regenerate with BLOOM_UPDATE_GOLDEN=1.",
            outlier_frac * 100.0,
            OUTLIER_FRACTION * 100.0,
        );
    }
}

#[test]
fn golden_shapes_2d() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let (w, h, rgba) = render(&mut eng, 3, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(0.12, 0.12, 0.15, 1.0);
        r.draw_rect(20.0, 20.0, 100.0, 60.0, 230.0, 41.0, 55.0, 255.0);
        r.draw_rect_lines(140.0, 20.0, 90.0, 90.0, 4.0, 0.0, 228.0, 48.0, 255.0);
        r.draw_circle(70.0, 160.0, 40.0, 0.0, 121.0, 241.0, 255.0);
        r.draw_circle_lines(180.0, 170.0, 50.0, 253.0, 249.0, 0.0, 255.0);
        r.draw_line(10.0, 240.0, 246.0, 200.0, 3.0, 255.0, 255.0, 255.0, 255.0);
    });
    compare_or_update("shapes_2d", w, h, &rgba);
}

#[test]
fn golden_lit_primitives_3d() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    // Several warm-up frames: SSAO/SSGI history seeds on the first
    // frames; by frame 6 the EMA is settled enough to be deterministic
    // within tolerance.
    let (w, h, rgba) = render(&mut eng, 6, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(0.05, 0.07, 0.1, 1.0);
        r.begin_mode_3d(
            4.0, 3.0, 6.0, // eye
            0.0, 0.5, 0.0, // target
            0.0, 1.0, 0.0, // up
            45.0, 0.0, // fovy, perspective
        );
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.95, 0.9, 1.2);
        r.add_point_light(2.0, 2.0, 2.0, 10.0, 0.2, 0.4, 1.0, 2.0);
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
        r.draw_cube(-1.2, 0.5, 0.0, 1.0, 1.0, 1.0, 230.0, 41.0, 55.0, 255.0);
        r.draw_sphere(1.2, 0.75, 0.5, 0.75, 0.0, 228.0, 48.0, 255.0);
        r.draw_cube(0.0, 1.6, -1.0, 0.8, 0.8, 0.8, 253.0, 249.0, 0.0, 255.0);
        r.draw_cylinder(-2.6, 0.02, 1.0, 0.4, 0.4, 1.4, 200.0, 122.0, 255.0, 255.0);
    });
    compare_or_update("lit_primitives_3d", w, h, &rgba);
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
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            60.0, 0.0,
        );
        r.draw_plane(0.0, 0.0, 0.0, 14.0, 14.0, 110.0, 110.0, 110.0, 255.0);
        for i in 0..40u32 {
            let t = i as f32 / 40.0 * std::f32::consts::TAU;
            let (sx, sz) = (t.cos() * 4.0, t.sin() * 4.0);
            // hue cycles so neighboring lights are distinguishable
            let (lr, lg, lb) = (
                0.5 + 0.5 * (t).cos(),
                0.5 + 0.5 * (t + 2.094).cos(),
                0.5 + 0.5 * (t + 4.189).cos(),
            );
            r.add_point_light(sx, 1.2, sz, 3.5, lr, lg, lb, 1.6);
        }
    });
    compare_or_update("many_point_lights", w, h, &rgba);
}

#[test]
fn golden_lod_selection() {
    use bloom_shared::renderer::Vertex3D;
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };

    fn cube_verts(half: f32, color: [f32; 4]) -> (Vec<Vertex3D>, Vec<u32>) {
        // 6 faces, outward winding (matches scene-node conventions:
        // prepare() recomputes bounds from positions).
        let h = half;
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            ([0.0, 0.0, -1.0], [[-h,-h,-h],[ h,-h,-h],[ h, h,-h],[-h, h,-h]]),
            ([0.0, 0.0,  1.0], [[ h,-h, h],[-h,-h, h],[-h, h, h],[ h, h, h]]),
            ([-1.0, 0.0, 0.0], [[-h,-h, h],[-h,-h,-h],[-h, h,-h],[-h, h, h]]),
            ([1.0, 0.0, 0.0],  [[ h,-h,-h],[ h,-h, h],[ h, h, h],[ h, h,-h]]),
            ([0.0, 1.0, 0.0],  [[-h, h,-h],[ h, h,-h],[ h, h, h],[-h, h, h]]),
            ([0.0, -1.0, 0.0], [[-h,-h, h],[ h,-h, h],[ h,-h,-h],[-h,-h,-h]]),
        ];
        let mut verts = Vec::new();
        let mut idx = Vec::new();
        for (normal, vs) in faces {
            let base = verts.len() as u32;
            for p in vs {
                verts.push(Vertex3D {
                    position: p,
                    normal,
                    color,
                    uv: [0.0, 0.0],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    tangent: [0.0; 4],
                });
            }
            idx.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
        (verts, idx)
    }

    let (red_v, red_i) = cube_verts(0.5, [0.9, 0.1, 0.1, 1.0]);
    let (green_v, green_i) = cube_verts(0.5, [0.1, 0.9, 0.1, 1.0]);

    let mut translate = |x: f32, z: f32| -> [[f32; 4]; 4] {
        let mut m = [[0.0f32; 4]; 4];
        m[0][0] = 1.0; m[1][1] = 1.0; m[2][2] = 1.0; m[3][3] = 1.0;
        m[3][0] = x; m[3][2] = z;
        m
    };

    // Near node: large on screen → base (red) geometry.
    let near = eng.scene.create_node();
    eng.scene.update_geometry(near, red_v.clone(), red_i.clone());
    eng.scene.set_lod_geometry(near, 0, green_v.clone(), green_i.clone(), 0.12);
    eng.scene.set_transform(near, translate(-1.0, 2.0));

    // Far node: small on screen → LOD 0 (green) variant.
    let far = eng.scene.create_node();
    eng.scene.update_geometry(far, red_v, red_i);
    eng.scene.set_lod_geometry(far, 0, green_v, green_i, 0.12);
    eng.scene.set_transform(far, translate(6.0, -22.0));

    let (w, h, rgba) = render(&mut eng, 4, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 8.0, 12.0, 255.0);
        r.begin_mode_3d(0.0, 1.5, 6.0, 0.0, 0.0, -4.0, 0.0, 1.0, 0.0, 50.0, 0.0);
        r.add_directional_light(-0.4, -1.0, -0.4, 1.0, 1.0, 1.0, 1.5);
    });
    compare_or_update("lod_selection", w, h, &rgba);
}

#[test]
fn cooked_bc7_texture_matches_raw() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let png = std::fs::read(fixtures.join("quadrants.png")).unwrap();
    let dds = std::fs::read(fixtures.join("quadrants_bc7.dds")).unwrap();

    // Load the same image through both paths: raw PNG (decode +
    // runtime mips) and cooked BC7 DDS (compressed upload where the
    // adapter has BC, CPU decode otherwise — both exercised by CI
    // across runners).
    let renderer = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
    let raw = eng.textures.load_texture(unsafe { &mut *renderer }, &png);
    let cooked = eng.textures.load_texture(unsafe { &mut *renderer }, &dds);
    assert_ne!(raw, 0.0);
    assert_ne!(cooked, 0.0, "cooked DDS failed to load");
    assert_eq!(
        {
            let t = eng.textures.get(cooked).unwrap();
            (t.width, t.height)
        },
        (64, 64)
    );

    let raw_idx = eng.textures.get(raw).unwrap().bind_group_idx;
    let cooked_idx = eng.textures.get(cooked).unwrap().bind_group_idx;

    let (w, _h, frame_raw) = render(&mut eng, 2, |eng| {
        eng.renderer.set_clear_color(0.0, 0.0, 0.0, 255.0);
        eng.renderer.draw_texture(raw_idx, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0);
    });
    let (_, _, frame_cooked) = render(&mut eng, 2, |eng| {
        eng.renderer.set_clear_color(0.0, 0.0, 0.0, 255.0);
        eng.renderer.draw_texture(cooked_idx, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0);
    });

    // BC7 is lossy but high quality: the two frames must agree closely
    // wherever the texture landed. Compare the texture region.
    let mut max_diff = 0u8;
    for y in 0..64u32 {
        for x in 0..64u32 {
            let i = ((y * w + x) * 4) as usize;
            for c in 0..3 {
                max_diff = max_diff.max(frame_raw[i + c].abs_diff(frame_cooked[i + c]));
            }
        }
    }
    assert!(
        max_diff <= 16,
        "cooked render diverges from raw render: max channel diff {max_diff}"
    );
}
