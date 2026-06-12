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
pub fn bloom_set_fog(r: f64, g: f64, b: f64, density: f64, height_ref: f64, height_falloff: f64) {
    let r_ = engine();
    r_.renderer.set_fog_color(r as f32, g as f32, b as f32);
    r_.renderer.set_fog_density(density as f32);
    r_.renderer.set_fog_height_falloff(height_ref as f32, height_falloff as f32);
}
#[wasm_bindgen]
pub fn bloom_set_chromatic_aberration(strength: f64) { engine().renderer.set_chromatic_aberration(strength as f32); }
#[wasm_bindgen]
pub fn bloom_set_vignette(strength: f64, softness: f64) { engine().renderer.set_vignette(strength as f32, softness as f32); }
#[wasm_bindgen]
pub fn bloom_set_film_grain(strength: f64) { engine().renderer.set_film_grain(strength as f32); }
#[wasm_bindgen]
pub fn bloom_set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) {
    let eng = engine();
    eng.renderer.set_sun_shaft_strength(strength as f32);
    eng.renderer.set_sun_shaft_decay(decay as f32);
    eng.renderer.set_sun_shaft_color(r as f32, g as f32, b as f32);
}
#[wasm_bindgen]
pub fn bloom_set_auto_exposure(on: f64) { engine().renderer.set_auto_exposure(on != 0.0); }
#[wasm_bindgen]
pub fn bloom_set_manual_exposure(value: f64) { engine().renderer.set_manual_exposure(value as f32); }
#[wasm_bindgen]
pub fn bloom_set_taa_enabled(on: f64) { engine().renderer.set_taa_enabled(on != 0.0); }
#[wasm_bindgen]
pub fn bloom_set_occlusion_culling(on: f64) { engine().renderer.occlusion.enabled = on != 0.0; }
#[wasm_bindgen]
pub fn bloom_set_render_scale(scale: f64) { engine().renderer.set_render_scale(scale as f32); }
#[wasm_bindgen]
pub fn bloom_get_render_scale() -> f64 { engine().renderer.render_scale() as f64 }
#[wasm_bindgen]
pub fn bloom_set_upscale_mode(mode: f64) { engine().renderer.set_upscale_mode(mode as u32); }
#[wasm_bindgen]
pub fn bloom_set_cas_strength(strength: f64) { engine().renderer.set_cas_strength(strength as f32); }
#[wasm_bindgen]
pub fn bloom_get_physical_width() -> f64 { engine().renderer.physical_width() as f64 }
#[wasm_bindgen]
pub fn bloom_get_physical_height() -> f64 { engine().renderer.physical_height() as f64 }
#[wasm_bindgen]
pub fn bloom_set_auto_resolution(target_hz: f64, enabled: f64) {
    let eng = engine();
    if enabled != 0.0 {
        let current = eng.renderer.render_scale();
        eng.drs.enable(target_hz as f32, current);
    } else {
        eng.drs.disable();
    }
}
#[wasm_bindgen]
pub fn bloom_set_env_intensity(intensity: f64) { engine().renderer.set_env_intensity(intensity as f32); }
#[wasm_bindgen]
pub fn bloom_set_ssgi_enabled(enabled: f64) { engine().renderer.set_ssgi_enabled(enabled != 0.0); }
#[wasm_bindgen]
pub fn bloom_set_ssgi_intensity(intensity: f64) { engine().renderer.set_ssgi_intensity(intensity as f32); }
#[wasm_bindgen]
pub fn bloom_set_ssgi_radius(radius: f64) { engine().renderer.set_ssgi_radius(radius as f32); }
#[wasm_bindgen]
pub fn bloom_set_dof(enabled: f64, focus_distance: f64, aperture: f64) {
    let r = &mut engine().renderer;
    r.set_dof_enabled(enabled != 0.0);
    r.set_dof_focus_distance(focus_distance as f32);
    r.set_dof_aperture(aperture as f32);
}

// LOD: pointer-taking variants share the cross-module WASM memory TODO
// with bloom_scene_update_geometry above; the model-based variant works.
#[wasm_bindgen]
pub fn bloom_scene_set_lod(
    _handle: f64, _lod_index: f64, _vert_ptr: f64, _vert_count: f64,
    _idx_ptr: f64, _idx_count: f64, _max_coverage: f64,
) {
    // TODO: Phase 4 — pointer/buffer passing from Perry WASM linear memory.
}
#[wasm_bindgen]
pub fn bloom_scene_attach_model_lod(node: f64, model: f64, mesh_index: f64, lod_index: f64, max_coverage: f64) {
    let eng = engine();
    let mi = mesh_index as usize;
    let Some(md) = eng.models.models.get(model) else { return };
    if mi >= md.meshes.len() { return; }
    let mesh = &md.meshes[mi];
    let (v, i) = (mesh.vertices.clone(), mesh.indices.clone());
    eng.scene.set_lod_geometry(node, lod_index as usize, v, i, max_coverage as f32);
}
