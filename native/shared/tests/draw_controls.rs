//! Self-checking render tests (no reference images) for two per-frame
//! draw controls:
//!
//! - `set_immediate_cast_shadows` (`setDrawShadowCasting` in TypeScript):
//!   immediate 3D draws issued while it is off draw but cast no shadow.
//! - the post-pass frame clock `post_frame` (binding 4 of every game post
//!   pass): frame counter and output resolution reach the shader.
//!
//! Runs on a non-CPU GPU adapter and skips gracefully without one, like the
//! golden suite.

use bloom_shared::engine::EngineState;
use bloom_shared::renderer::{Renderer, Vertex3D};

const W: u32 = 192;
const H: u32 = 192;

fn try_engine() -> Option<EngineState> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return None;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    let renderer = Renderer::new_headless(device, queue, W, H);
    let mut eng = EngineState::new(renderer);
    eng.renderer.set_taa_enabled(false);
    eng.renderer.set_render_scale(1.0);
    Some(eng)
}

/// Render `frames` frames of `draw` and return the last one (raw surface
/// bytes; only brightness and per-channel thresholds are read here).
fn render(eng: &mut EngineState, frames: u32, mut draw: impl FnMut(&mut EngineState)) -> Vec<u8> {
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
    shot.expect("screenshot capture produced no data").2
}

fn luma(px: &[u8]) -> f64 {
    (f64::from(px[0]) + f64::from(px[1]) + f64::from(px[2])) / 3.0
}

/// The ground: a scene node (immediate geometry does not receive shadows).
fn add_ground(eng: &mut EngineState) {
    let s = 6.0f32;
    let corner = |x: f32, z: f32| Vertex3D {
        position: [x, 0.0, z],
        normal: [0.0, 1.0, 0.0],
        color: [0.8, 0.8, 0.8, 1.0],
        uv: [0.0, 0.0],
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    let vertices = vec![corner(-s, -s), corner(s, -s), corner(s, s), corner(-s, s)];
    let node = eng.scene.create_node();
    eng.scene
        .update_geometry(node, vertices, vec![0, 2, 1, 0, 3, 2]);
}

/// A ball held over the ground, the sun high to one side so its shadow
/// falls on open ground. `cast` is the shadow-casting state set before the
/// ball is drawn; a small cube is drawn afterwards with the state put back
/// on, so a split segment is exercised too.
fn ball_scene(eng: &mut EngineState, cast: Option<bool>) {
    let r = &mut eng.renderer;
    r.set_clear_color(0.05, 0.07, 0.1, 1.0);
    r.begin_mode_3d(0.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    r.set_ambient_light(180.0, 180.0, 190.0, 0.3);
    r.set_directional_light(0.6, 1.0, 0.2, 1.0, 0.97, 0.92, 2.0);
    if let Some(on) = cast {
        r.set_immediate_cast_shadows(on);
    }
    r.draw_sphere(-0.6, 1.4, 0.0, 0.6, 230.0, 80.0, 60.0, 255.0);
    r.set_immediate_cast_shadows(true);
    r.draw_cube(2.5, 0.2, 2.0, 0.4, 0.4, 0.4, 60.0, 200.0, 60.0, 255.0);
}

/// Pixels clearly darker in `a` than in `b`, and the other way round.
fn darker_counts(a: &[u8], b: &[u8]) -> (usize, usize) {
    let (mut a_darker, mut b_darker) = (0, 0);
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = luma(pb) - luma(pa);
        if d > 25.0 {
            a_darker += 1;
        } else if d < -25.0 {
            b_darker += 1;
        }
    }
    (a_darker, b_darker)
}

#[test]
fn immediate_draws_can_leave_out_their_shadow() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_shadows_enabled(true);
    add_ground(&mut eng);
    // One engine through every case: the shadow cache must notice a caster
    // leaving and coming back.
    let casting = render(&mut eng, 6, |eng| ball_scene(eng, None));
    let silent = render(&mut eng, 6, |eng| ball_scene(eng, Some(false)));
    let explicit_on = render(&mut eng, 6, |eng| ball_scene(eng, Some(true)));
    // The state is reset at the start of each frame: a frame that set it
    // off is followed by frames that never touch it.
    let _ = render(&mut eng, 1, |eng| ball_scene(eng, Some(false)));
    let after_reset = render(&mut eng, 6, |eng| ball_scene(eng, None));

    if let Ok(dir) = std::env::var("DRAW_CONTROLS_DUMP") {
        for (name, img) in [("casting", &casting), ("silent", &silent)] {
            let mut ppm = format!("P6 {W} {H} 255\n").into_bytes();
            for px in img.chunks_exact(4) {
                ppm.extend_from_slice(&px[..3]);
            }
            std::fs::write(format!("{dir}/{name}.ppm"), ppm).ok();
        }
    }
    let (shadowed, lit) = darker_counts(&casting, &silent);
    assert!(
        shadowed > 200,
        "the ball's shadow should be missing with casting off ({shadowed} pixels darker with it on)"
    );
    assert!(
        lit < shadowed / 10,
        "turning casting off should only remove a shadow ({lit} pixels brighter with it on)"
    );
    let (d1, d2) = darker_counts(&casting, &explicit_on);
    assert!(
        d1 + d2 < 20,
        "casting on explicitly should match the default ({d1}, {d2})"
    );
    let (d1, d2) = darker_counts(&casting, &after_reset);
    assert!(
        d1 + d2 < 20,
        "casting should be back on the next frame ({d1}, {d2})"
    );
}

#[test]
fn post_passes_see_the_frame_clock() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    // Red: odd frames; green: the output width arrived; blue: one pixel in
    // uv times the height is one. Each channel is 0 or 1, so the output
    // encoding cannot blur the answer.
    let wgsl = format!(
        r#"
        @fragment
        fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {{
            let odd = f32(post_frame.frame_index % 2u);
            let width_ok = select(0.0, 1.0, abs(post_frame.resolution.x - {w}.0) < 0.5);
            let inv_ok = select(0.0, 1.0,
                abs(post_frame.inv_resolution.y * post_frame.resolution.y - 1.0) < 0.001
                && post_frame.delta_time >= 0.0 && post_frame.time >= 0.0);
            return vec4<f32>(odd, width_ok, inv_ok, 1.0);
        }}
        "#,
        w = W
    );
    eng.renderer
        .add_post_pass(&wgsl)
        .unwrap_or_else(|error| panic!("post pass failed to compile: {error:?}"));
    let draw = |eng: &mut EngineState| eng.renderer.set_clear_color(0.0, 0.0, 0.0, 1.0);
    let bgra = matches!(
        eng.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let mut reds = Vec::new();
    for _ in 0..4 {
        let px = render(&mut eng, 1, draw);
        let (r, g, b) = if bgra {
            (px[2], px[1], px[0])
        } else {
            (px[0], px[1], px[2])
        };
        assert!(g > 200, "post_frame.resolution should be the output size");
        assert!(
            b > 200,
            "post_frame.inv_resolution should be one pixel in uv"
        );
        reds.push(r > 127);
    }
    assert_eq!(
        reds,
        vec![reds[0], !reds[0], reds[0], !reds[0]],
        "post_frame.frame_index should count one per frame"
    );
}
