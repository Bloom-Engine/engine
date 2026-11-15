//! Shadows from local lights on the engine's own scene shader
//! (`renderer/local_light_shadows.rs`, EN-069).
//!
//! Self-checking, no reference image. A lamp stands on one side of a wall. A
//! red cube stands behind the wall, a green cube beside it in the lamp's
//! light. With the lamp's shadow switched on the red cube must lose the
//! lamp's light and the green cube must keep it; and a lamp that did not ask
//! for a shadow must draw exactly as it does with the feature off.

#[allow(unused_imports)]
use super::*;

fn placed(sx: f32, sy: f32, sz: f32, x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = sx;
    m[1][1] = sy;
    m[2][2] = sz;
    m[3] = [x, y, z, 1.0];
    m
}

/// Mean brightness (0 to 255) of the pixels `mask` selects, and how many.
fn mean_where(rgba: &[u8], mask: &[bool]) -> (f64, usize) {
    let (mut sum, mut count) = (0.0f64, 0usize);
    for (pixel, selected) in rgba.chunks_exact(4).zip(mask) {
        if *selected {
            sum += (f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2])) / 3.0;
            count += 1;
        }
    }
    (sum / count.max(1) as f64, count)
}

#[test]
fn a_wall_shadows_a_scene_node_from_a_local_light() {
    // The feature needs the high capability tier, which the plain golden
    // device (no optional features requested) is not; the path-tracing
    // goldens' device is. The tier is decided once per process by the first
    // renderer, so in a full run of this suite this test only does its work
    // when a high-tier renderer came first; run alone, it always does.
    let Ok(Some((mut eng, _))) = try_engine_rt() else {
        eprintln!("skip: no high-tier GPU device");
        return;
    };
    // The local-light maps are drawn from the sun's caster lists.
    eng.renderer.set_shadows_enabled(true);
    if !eng.renderer.set_local_light_shadows(4, 512) {
        eprintln!("skip: this device does not carry local-light shadows");
        return;
    }
    if !eng.renderer.local_light_shadows_on_scene_models() {
        eprintln!("skip: the scene shader does not sample local-light shadows here");
        return;
    }

    let mut add = |color: [f32; 4], transform: [[f32; 4]; 4]| {
        let (vertices, indices) = cube_verts(0.5, color);
        let node = eng.scene.create_node();
        eng.scene.update_geometry(node, vertices, indices);
        eng.scene.set_transform(node, transform);
    };
    // Floor, a wall in the plane x = 0, the red cube behind it (seen from the
    // lamp at x = -3), the green cube past the wall's end, in the light.
    add(
        [0.5, 0.5, 0.5, 1.0],
        placed(14.0, 0.2, 14.0, 0.0, -0.1, 0.0),
    );
    add([0.5, 0.5, 0.5, 1.0], placed(0.2, 5.0, 3.0, 0.0, 2.5, 0.0));
    add([1.0, 0.1, 0.1, 1.0], placed(1.0, 1.0, 1.0, 2.5, 0.5, 0.0));
    add([0.1, 1.0, 0.1, 1.0], placed(1.0, 1.0, 1.0, -2.0, 0.5, 3.0));

    #[derive(Clone, Copy, PartialEq)]
    enum Lamp {
        None,
        Plain,
        Shadowed,
    }
    let shoot = |eng: &mut EngineState, lamp: Lamp| {
        // 3 frames to draw the six faces, 12 to fade the shadow in.
        render(eng, 24, |eng| {
            let r = &mut eng.renderer;
            r.set_clear_color(0.0, 0.0, 40.0, 255.0);
            r.begin_mode_3d(5.0, 6.0, 0.01, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 60.0, 0.0);
            r.set_ambient_light(255.0, 255.0, 255.0, 0.02);
            r.set_directional_light(0.0, 1.0, 0.0, 255.0, 255.0, 255.0, 0.0);
            match lamp {
                Lamp::None => {}
                Lamp::Plain => r.add_point_light(-3.0, 3.0, 0.0, 12.0, 1.0, 1.0, 1.0, 12.0),
                Lamp::Shadowed => {
                    r.add_point_light_with_shadow(-3.0, 3.0, 0.0, 12.0, 1.0, 1.0, 1.0, 12.0);
                }
            }
        })
        .2
    };

    let dark = shoot(&mut eng, Lamp::None);
    let plain = shoot(&mut eng, Lamp::Plain);
    let shadowed = shoot(&mut eng, Lamp::Shadowed);
    // Off again: a lamp that asks for a shadow is an ordinary lamp.
    assert!(!eng.renderer.set_local_light_shadows(0, 512));
    let switched_off = shoot(&mut eng, Lamp::Shadowed);

    // The cubes, found by their colour where the unshadowed lamp lights them.
    let red: Vec<bool> = plain
        .chunks_exact(4)
        .map(|p| p[0] > p[1].max(p[2]).saturating_add(40))
        .collect();
    let green: Vec<bool> = plain
        .chunks_exact(4)
        .map(|p| p[1] > p[0].max(p[2]).saturating_add(40))
        .collect();
    let (red_plain, red_pixels) = mean_where(&plain, &red);
    let (green_plain, green_pixels) = mean_where(&plain, &green);
    assert!(
        red_pixels > 200 && green_pixels > 200,
        "the cubes are not in the picture: {red_pixels} red, {green_pixels} green pixels"
    );
    let (red_dark, _) = mean_where(&dark, &red);
    let (red_shadowed, _) = mean_where(&shadowed, &red);
    let (green_shadowed, _) = mean_where(&shadowed, &green);
    eprintln!(
        "local light shadow on scene nodes: red cube {red_plain:.1} lit, {red_shadowed:.1} \
         shadowed, {red_dark:.1} without the lamp; green cube {green_plain:.1} lit, \
         {green_shadowed:.1} with the shadow on"
    );
    // Without the fix the red cube is lit straight through the wall.
    assert!(
        red_plain > red_dark + 10.0,
        "the lamp does not reach the red cube at all: {red_plain:.1} against {red_dark:.1}"
    );
    // Behind the wall it gets what it gets without the lamp.
    assert!(
        (red_shadowed - red_dark).abs() < 2.0,
        "the wall does not shadow the red cube: {red_shadowed:.1} shadowed, {red_dark:.1} \
         without the lamp, {red_plain:.1} lit"
    );
    // In the light nothing is lost.
    assert!(
        (green_shadowed - green_plain).abs() < 2.0,
        "the cube in the light changed: {green_plain:.1} to {green_shadowed:.1}"
    );
    // A shadow only ever removes light. (The margin is for the sharpening
    // pass, which lifts a pixel a little when its neighbour gets darker.)
    let brighter = plain
        .chunks_exact(4)
        .zip(shadowed.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|c| b[c] > a[c].saturating_add(12)))
        .count();
    assert_eq!(
        brighter, 0,
        "{brighter} pixels got brighter with the shadow on"
    );
    // Switched off, the shadow-asking lamp is the plain lamp (to within what
    // the temporal passes leave between two runs of the same frame).
    let largest = plain
        .iter()
        .zip(&switched_off)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        largest <= 2,
        "a shadow-asking lamp with the feature switched off differs from a plain lamp by up \
         to {largest} of 255"
    );
}
