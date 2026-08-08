//! Renderer-settings FFI surface for web — wasm_bindgen wrappers over
//! the same shared engine methods native's define_core_ffi! generates
//! wrappers for. Split from lib.rs (2000-line file policy).

use crate::engine;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Renderer settings — parity with the native define_core_ffi! surface.
// (Added when tools/validate-ffi.js gained web coverage; these were the
// silent gaps that made web games' graphics settings no-ops.)
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub fn bloom_set_fog(
    r: f64,
    g: f64,
    b: f64,
    density: f64,
    height_ref: f64,
    height_falloff: f64,
) -> f64 {
    let r_ = engine();
    r_.renderer.set_fog_color(r as f32, g as f32, b as f32);
    r_.renderer.set_fog_density(density as f32);
    r_.renderer
        .set_fog_height_falloff(height_ref as f32, height_falloff as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_chromatic_aberration(strength: f64) -> f64 {
    engine().renderer.set_chromatic_aberration(strength as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_vignette(strength: f64, softness: f64) -> f64 {
    engine()
        .renderer
        .set_vignette(strength as f32, softness as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_film_grain(strength: f64) -> f64 {
    engine().renderer.set_film_grain(strength as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_sharpen_strength(strength: f64) -> f64 {
    engine().renderer.set_sharpen_strength(strength as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) -> f64 {
    let eng = engine();
    eng.renderer.set_sun_shaft_strength(strength as f32);
    eng.renderer.set_sun_shaft_decay(decay as f32);
    eng.renderer
        .set_sun_shaft_color(r as f32, g as f32, b as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_auto_exposure(on: f64) -> f64 {
    engine().renderer.set_auto_exposure(on != 0.0);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_manual_exposure(value: f64) -> f64 {
    engine().renderer.set_manual_exposure(value as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_taa_enabled(on: f64) -> f64 {
    engine().renderer.set_taa_enabled(on != 0.0);
    1.0
}
#[wasm_bindgen]
pub fn bloom_reset_temporal_history() -> f64 {
    engine().renderer.reset_temporal_history();
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_occlusion_culling(on: f64) -> f64 {
    engine().renderer.occlusion.enabled = on != 0.0;
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_render_scale(scale: f64) -> f64 {
    engine().renderer.set_render_scale(scale as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_get_render_scale() -> f64 {
    engine().renderer.render_scale() as f64
}
#[wasm_bindgen]
pub fn bloom_set_upscale_mode(mode: f64) -> f64 {
    engine().renderer.set_upscale_mode(mode as u32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_cas_strength(strength: f64) -> f64 {
    engine().renderer.set_cas_strength(strength as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_get_physical_width() -> f64 {
    engine().renderer.physical_width() as f64
}
#[wasm_bindgen]
pub fn bloom_get_physical_height() -> f64 {
    engine().renderer.physical_height() as f64
}
#[wasm_bindgen]
pub fn bloom_set_auto_resolution(target_hz: f64, enabled: f64) -> f64 {
    let eng = engine();
    if enabled != 0.0 {
        let current = eng.renderer.render_scale();
        eng.drs.enable(target_hz as f32, current);
    } else {
        eng.drs.disable();
    }
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_env_intensity(intensity: f64) -> f64 {
    engine().renderer.set_env_intensity(intensity as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_ssgi_enabled(enabled: f64) -> f64 {
    engine().renderer.set_ssgi_enabled(enabled != 0.0);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_ssgi_intensity(intensity: f64) -> f64 {
    engine().renderer.set_ssgi_intensity(intensity as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_ssgi_radius(radius: f64) -> f64 {
    engine().renderer.set_ssgi_radius(radius as f32);
    1.0
}
#[wasm_bindgen]
pub fn bloom_set_dof(enabled: f64, focus_distance: f64, aperture: f64) -> f64 {
    let r = &mut engine().renderer;
    r.set_dof_enabled(enabled != 0.0);
    r.set_dof_focus_distance(focus_distance as f32);
    r.set_dof_aperture(aperture as f32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_bloom_intensity(value: f64) -> f64 {
    engine().renderer.set_bloom_intensity(value as f32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_tonemap(kind: f64) -> f64 {
    engine().renderer.set_tonemap_kind(kind as u32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_auto_exposure_key(key: f64) -> f64 {
    engine().renderer.set_auto_exposure_key(key as f32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_auto_exposure_rate(rate: f64) -> f64 {
    engine().renderer.set_auto_exposure_rate(rate as f32);
    1.0
}

// Render quality toggles (individual + preset) — ticket 011.
// These mirror native's settings surface so browser games never discover
// missing controls only after deployment.
#[wasm_bindgen]
pub fn bloom_set_quality_preset(preset: f64) -> f64 {
    engine().renderer.apply_quality_preset(preset as u32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_shadows_enabled(on: f64) -> f64 {
    engine().renderer.set_shadows_enabled(on != 0.0);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_shadows_always_fresh(on: f64) -> f64 {
    engine().renderer.set_shadows_always_fresh(on != 0.0);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_bloom_enabled(on: f64) -> f64 {
    engine().renderer.set_bloom_enabled(on != 0.0);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_ssao_enabled(on: f64) -> f64 {
    engine().renderer.set_ssao_enabled(on != 0.0);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_ssao_intensity(value: f64) -> f64 {
    engine().renderer.set_ssao_strength(value as f32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_ssao_radius(world_radius: f64) -> f64 {
    engine().renderer.set_ssao_radius(world_radius as f32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_wind(dir_x: f64, dir_z: f64, amplitude: f64, frequency: f64) -> f64 {
    engine().renderer.set_wind(
        dir_x as f32,
        dir_z as f32,
        amplitude as f32,
        frequency as f32,
    );
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_output_scale(scale: f64) -> f64 {
    engine().renderer.set_output_scale(scale as f32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_get_output_scale() -> f64 {
    engine().renderer.output_scale() as f64
}

#[wasm_bindgen]
pub fn bloom_set_model_foliage_wind(model: f64, amount: f64) -> f64 {
    engine()
        .renderer
        .set_model_foliage_wind(model.to_bits(), amount as f32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_foliage_shadow_motion(on: f64) -> f64 {
    engine().renderer.set_foliage_shadow_motion(on > 0.5);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_cloud_shadows(
    strength: f64,
    deck_height: f64,
    feature_scale: f64,
    drift_speed: f64,
) -> f64 {
    engine().renderer.set_cloud_shadows(
        strength as f32,
        deck_height as f32,
        feature_scale as f32,
        drift_speed as f32,
    );
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_ssr_enabled(on: f64) -> f64 {
    engine().renderer.set_ssr_enabled(on != 0.0);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_motion_blur_enabled(on: f64) -> f64 {
    engine().renderer.set_motion_blur_enabled(on != 0.0);
    1.0
}

#[wasm_bindgen]
pub fn bloom_set_sss_enabled(on: f64) -> f64 {
    engine().renderer.set_sss_enabled(on != 0.0);
    1.0
}

// LOD: pointer-taking variants share the cross-module WASM memory TODO
// with bloom_scene_update_geometry above; the model-based variant works.
#[wasm_bindgen]
pub fn bloom_scene_set_lod(
    _handle: f64,
    _lod_index: f64,
    _vert_ptr: f64,
    _vert_count: f64,
    _idx_ptr: f64,
    _idx_count: f64,
    _max_coverage: f64,
) -> f64 {
    // TODO: Phase 4 — pointer/buffer passing from Perry WASM linear memory.
    0.0
}
#[wasm_bindgen]
pub fn bloom_scene_attach_model_lod(
    node: f64,
    model: f64,
    mesh_index: f64,
    lod_index: f64,
    max_coverage: f64,
) -> f64 {
    let eng = engine();
    let mi = mesh_index as usize;
    let Some(md) = eng.models.models.get(model) else {
        return 0.0;
    };
    if mi >= md.meshes.len() {
        return 0.0;
    }
    let mesh = &md.meshes[mi];
    let (v, i) = (mesh.vertices.clone(), mesh.indices.clone());
    eng.scene
        .set_lod_geometry(node, lod_index as usize, v, i, max_coverage as f32);
    1.0
}
