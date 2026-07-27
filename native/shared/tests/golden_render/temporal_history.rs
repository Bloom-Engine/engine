use super::*;

#[test]
fn ssr_history_lifetime_is_independent_from_the_taa_frame_counter() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_taa_enabled(true);

    let draw_frame = |eng: &mut EngineState| {
        let r = &mut eng.renderer;
        r.set_clear_color(13.0, 18.0, 26.0, 255.0);
        r.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        r.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.95, 0.9, 1.2);
        r.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
        r.draw_sphere(0.0, 0.75, 0.0, 0.75, 220.0, 228.0, 240.0, 255.0);
    };
    let advance = |eng: &mut EngineState, frames: u32| {
        for _ in 0..frames {
            eng.begin_frame();
            draw_frame(eng);
            eng.end_frame();
        }
    };

    advance(&mut eng, 3);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":true"));

    eng.renderer.set_ssr_enabled(false);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
    advance(&mut eng, 2);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));

    eng.renderer.set_ssr_enabled(true);
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":true"));

    eng.renderer.set_ssr_strength(0.75);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
    advance(&mut eng, 1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":true"));

    eng.renderer.set_path_tracing(1);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
    eng.renderer.set_path_tracing(0);
    eng.renderer.set_render_scale(0.75);
    assert!(eng
        .renderer
        .quality_runtime_paths_json()
        .contains("\"temporal_history\":{\"ssr_valid\":false"));
}
