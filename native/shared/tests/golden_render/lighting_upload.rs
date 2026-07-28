use super::*;

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
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 60.0, 0.0,
        );
        r.draw_plane(0.0, 0.0, 0.0, 14.0, 14.0, 110.0, 110.0, 110.0, 255.0);
        for i in 0..40u32 {
            let t = i as f32 / 40.0 * std::f32::consts::TAU;
            let (sx, sz) = (t.cos() * 4.0, t.sin() * 4.0);
            // Hue cycles so neighboring lights are distinguishable.
            let (lr, lg, lb) = (
                0.5 + 0.5 * t.cos(),
                0.5 + 0.5 * (t + 2.094).cos(),
                0.5 + 0.5 * (t + 4.189).cos(),
            );
            r.add_point_light(sx, 1.2, sz, 3.5, lr, lg, lb, 1.6);
        }
    });
    compare_or_update("many_point_lights", w, h, &rgba);

    // The sixth frame is unchanged apart from per-frame/view values. Forty
    // repeated light setters must not enqueue forty copies of the full block.
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("runtime path telemetry is valid JSON");
    let uploads = &paths["steady_state_uploads"]["lighting"];
    let writes = uploads["write_count"].as_u64().expect("write count");
    let bytes = uploads["byte_count"].as_u64().expect("byte count");
    let full_bytes = uploads["full_buffer_bytes"].as_u64().expect("buffer size");
    assert!(
        writes <= 3 && bytes <= 512 && bytes < full_bytes,
        "steady point-light frame uploaded too much lighting data: {uploads}"
    );

    let bind_groups = &paths["steady_state_resources"]["bind_group_creations"];
    let total = bind_groups["total"].as_u64().expect("bind-group total");
    let sites = bind_groups["sites"]
        .as_object()
        .expect("named bind-group creation sites");
    let named_total: u64 = sites
        .values()
        .map(|count| count.as_u64().expect("site count"))
        .sum();
    assert_eq!(
        total, named_total,
        "bind-group total must match named sites"
    );
    assert_eq!(
        sites["scene_compose"].as_u64(),
        Some(0),
        "steady scene compose must reuse its SSR-source-specific bind group"
    );
    assert_eq!(
        sites["final_composite"].as_u64(),
        Some(0),
        "steady final composite must reuse its source/exposure-specific bind group"
    );
    assert_eq!(
        sites["ssr_temporal"].as_u64(),
        Some(0),
        "steady SSR temporal must reuse its previous-history-specific bind group"
    );
    assert_eq!(
        sites["taa"].as_u64(),
        Some(0),
        "steady ordinary TAA must reuse its previous-history-specific bind group"
    );
}

#[test]
fn steady_half_resolution_upscale_reuses_its_bind_group() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_render_scale(0.5);
    eng.renderer.set_taa_enabled(false);
    let (_, _, rgba) = render(&mut eng, 4, |eng| {
        let r = &mut eng.renderer;
        r.set_clear_color(8.0, 12.0, 20.0, 255.0);
        r.begin_mode_3d(3.0, 2.5, 5.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 48.0, 0.0);
        r.draw_cube(0.0, 0.75, 0.0, 1.5, 1.5, 1.5, 220.0, 90.0, 35.0, 255.0);
    });
    assert!(
        rgba.chunks_exact(4)
            .any(|pixel| pixel[0] != 8 || pixel[1] != 12 || pixel[2] != 20),
        "half-resolution upscale frame did not render scene geometry"
    );
    let paths: serde_json::Value = serde_json::from_str(&eng.renderer.quality_runtime_paths_json())
        .expect("upscale telemetry is valid JSON");
    assert_eq!(
        paths["steady_state_resources"]["bind_group_creations"]["sites"]["upscale"].as_u64(),
        Some(0),
        "warmed upscale path must reuse its persistent bind group"
    );
}
