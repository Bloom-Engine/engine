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
