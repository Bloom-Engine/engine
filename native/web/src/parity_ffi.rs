//! EN-063 — web FFI parity for full-3D games (driven by the shooter port).
//!
//! Every function here mirrors an existing native implementation in
//! `bloom-shared/src/ffi_core/`; the shared engine methods they call compile
//! unchanged on wasm32. The groups:
//!
//!   - Mesh / instance-buffer / texture-array **scratch** entry points. The
//!     TS wrappers never pass array pointers (Perry 0.5.x rejects `number[]`
//!     into i64 params); they push scalars through the mesh scratch and then
//!     call the `_scratch` builder. Without these exports every generated
//!     mesh (water ribbon, grass blades, splat map) silently came back 0 on
//!     web while the same game worked native.
//!   - `bloom_stage_model_bytes` — the synchronous web half of the staged
//!     model loader. JS glue fetches the GLB and calls this; `stageModels`'
//!     worker-thread decode does not exist on wasm, but the contract (stage
//!     returns a ticket, commit returns a model) is preserved.
//!   - Env-HDR from bytes, splat impulses, scene GI-only flags, profiler
//!     text — small native features 3D games actually call.
//!   - Host stubs (`present_mode`, `take_screenshot`) that must exist so the
//!     calls are cheap no-ops rather than auto-stubbed `undefined` returns.

use crate::engine;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

// ============================================================
// Mesh scratch (mirrors ffi_core/models.rs)
// ============================================================

#[wasm_bindgen]
pub fn bloom_mesh_scratch_reset() {
    engine().models.mesh_scratch_reset();
}

#[wasm_bindgen]
pub fn bloom_mesh_scratch_push_f32(v: f64) {
    engine().models.mesh_scratch_push_f32(v as f32);
}

#[wasm_bindgen]
pub fn bloom_mesh_scratch_push_u32(v: f64) {
    engine().models.mesh_scratch_push_u32(v as u32);
}

#[wasm_bindgen]
pub fn bloom_create_mesh_scratch(vertex_count: f64, index_count: f64) -> f64 {
    engine()
        .models
        .create_mesh_from_scratch(vertex_count as u32, index_count as u32)
}

#[wasm_bindgen]
pub fn bloom_gen_mesh_spline_ribbon_scratch(point_count: f64, width_count: f64) -> f64 {
    let n = point_count as usize;
    let wn = width_count as usize;
    let (points, widths) = {
        let scratch = engine().models.scratch_floats();
        if n < 2 || wn == 0 || scratch.len() < n * 3 + wn {
            return 0.0;
        }
        (
            scratch[..n * 3].to_vec(),
            scratch[n * 3..n * 3 + wn].to_vec(),
        )
    };
    engine().models.gen_mesh_spline_ribbon(&points, &widths)
}

#[wasm_bindgen]
pub fn bloom_create_instance_buffer_scratch(instance_count: f64) -> f64 {
    let eng = engine();
    let count = instance_count as u32;
    let need = (count as usize) * 9;
    if count == 0 || eng.models.scratch_f32.len() < need {
        return 0.0;
    }
    let data: Vec<f32> = eng.models.scratch_f32[..need].to_vec();
    eng.models.mesh_scratch_reset();
    eng.renderer.create_instance_buffer(&data, count) as f64
}

/// Params arrive via the mesh scratch (reset + push_f32 × N) — the idiom the
/// TS `setMaterialParams` wrapper uses everywhere (Perry 0.5.x rejects JS
/// arrays in pointer params). Forwards to the existing floats path.
#[wasm_bindgen]
pub fn bloom_set_material_params_scratch(handle: f64, param_count: f64) {
    let count = param_count as usize;
    let params: Vec<f32> = {
        let eng = engine();
        if eng.models.scratch_f32.len() < count {
            return;
        }
        let p = eng.models.scratch_f32[..count].to_vec();
        eng.models.mesh_scratch_reset();
        p
    };
    crate::material_ffi::bloom_set_material_params_floats(handle, &params);
}

// ============================================================
// Texture arrays (mirrors ffi_core/game_loop.rs)
// ============================================================

#[wasm_bindgen]
pub fn bloom_create_texture_array_scratch(
    width: f64,
    height: f64,
    layer_count: f64,
    format: f64,
    mip_levels: f64,
) -> f64 {
    let w = width as u32;
    let h = height as u32;
    if w == 0 || h == 0 {
        return 0.0;
    }
    let layers_count = (layer_count as u32)
        .min(bloom_shared::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
    if layers_count == 0 {
        return 0.0;
    }

    let texels = (w as usize) * (h as usize) * (layers_count as usize);
    // Scoped so the scratch borrow ends before the renderer borrow.
    let bytes: Vec<u8> = {
        let eng = engine();
        if eng.models.scratch_u32.len() < texels {
            // Short buffer: refuse rather than upload uninitialised memory.
            return 0.0;
        }
        eng.models.scratch_u32[..texels]
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect()
    };
    let layer_size = (w as usize) * (h as usize) * 4;
    let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(layers_count as usize);
    for i in 0..(layers_count as usize) {
        let start = i * layer_size;
        let end = start + layer_size;
        if end > bytes.len() {
            break;
        }
        layers.push((&bytes[start..end], w, h));
    }
    engine()
        .renderer
        .create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
}

// The from-files path can't read files on wasm, so it is split: JS glue
// fetches each path in the comma-separated list and pushes the raw file
// bytes here (decoded engine-side, same as native's `image::open`), then
// commits. Layers must share dimensions; first file's size wins.
thread_local! {
    static TEXARRAY_FILES: RefCell<Vec<(Vec<u8>, u32, u32)>> = const { RefCell::new(Vec::new()) };
}

#[wasm_bindgen]
pub fn bloom_texture_array_files_reset() {
    TEXARRAY_FILES.with(|f| f.borrow_mut().clear());
}

#[wasm_bindgen]
pub fn bloom_texture_array_files_push(data: &[u8]) -> f64 {
    match bloom_shared::textures::TextureManager::decode_rgba8(data) {
        Some((bytes, w, h)) => {
            TEXARRAY_FILES.with(|f| f.borrow_mut().push((bytes, w, h)));
            1.0
        }
        None => 0.0,
    }
}

#[wasm_bindgen]
pub fn bloom_texture_array_files_commit(format: f64, mip_levels: f64) -> f64 {
    let decoded = TEXARRAY_FILES.with(|f| f.borrow_mut().split_off(0));
    if decoded.is_empty() {
        return 0.0;
    }
    let (w, h) = (decoded[0].1, decoded[0].2);
    let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(decoded.len());
    for (bytes, lw, lh) in decoded.iter() {
        if *lw != w || *lh != h {
            crate::console_warn(&format!(
                "[texarray] layer size {}x{} != {}x{}; skipped",
                lw, lh, w, h
            ));
            continue;
        }
        layers.push((bytes.as_slice(), w, h));
    }
    if layers.is_empty() {
        return 0.0;
    }
    engine()
        .renderer
        .create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
}

// ============================================================
// Staged model loading (bytes half; JS glue fetches the file)
// ============================================================

#[wasm_bindgen]
pub fn bloom_stage_model_bytes(data: &[u8]) -> f64 {
    match bloom_shared::models::load_gltf_staged(data) {
        Some(staged) => bloom_shared::staging::stage_model(staged),
        None => 0.0,
    }
}

// ============================================================
// Environment / water / scene odds and ends
// ============================================================

#[cfg(feature = "image-extras")]
#[wasm_bindgen]
pub fn bloom_set_env_clear_from_hdr_bytes(data: &[u8]) {
    engine().renderer.set_env_clear_from_hdr_bytes(data);
}

#[wasm_bindgen]
pub fn bloom_splat_impulse(x: f64, z: f64, radius: f64, strength: f64) {
    engine().renderer.impulse_field.submit_splat(
        x as f32,
        z as f32,
        radius as f32,
        strength as f32,
    );
}

#[wasm_bindgen]
pub fn bloom_scene_set_gi_only(handle: f64, gi_only: f64) {
    engine().scene.set_gi_only(handle, gi_only != 0.0);
}

// ============================================================
// Profiler text (mirrors ffi_core/game_loop.rs; GPU column is 0 on
// web — no TIMESTAMP_QUERY in the WebGPU spec at wgpu 29)
// ============================================================

#[wasm_bindgen]
pub fn bloom_profiler_overlay_text() -> String {
    let snap = engine().profiler.snapshot();
    let mut s = String::with_capacity(snap.len() * 48);
    for (label, cpu, gpu) in &snap {
        s.push_str(label);
        s.push('|');
        s.push_str(&format!("{:.2}", cpu));
        s.push('|');
        match gpu {
            Some(g) => s.push_str(&format!("{:.2}", g)),
            None => s.push_str("-1"),
        }
        s.push('\n');
    }
    s
}

#[wasm_bindgen]
pub fn bloom_profiler_frame_history() -> String {
    let hist = engine().profiler.frame_history();
    let mut s = String::with_capacity(hist.len() * 24);
    for (cpu, gpu) in &hist {
        s.push_str(&format!("{:.2}|{:.2}\n", cpu, gpu));
    }
    s
}

#[wasm_bindgen]
pub fn bloom_profiler_row_count() -> f64 {
    engine().profiler.snapshot().len() as f64
}

#[wasm_bindgen]
pub fn bloom_profiler_row_label(i: f64) -> String {
    engine()
        .profiler
        .snapshot()
        .get(i as usize)
        .map(|(label, _, _)| label.to_string())
        .unwrap_or_default()
}

#[wasm_bindgen]
pub fn bloom_profiler_row_cpu_us(i: f64) -> f64 {
    engine()
        .profiler
        .snapshot()
        .get(i as usize)
        .map(|r| r.1)
        .unwrap_or(0.0)
}

#[wasm_bindgen]
pub fn bloom_profiler_row_gpu_us(i: f64) -> f64 {
    match engine().profiler.snapshot().get(i as usize) {
        Some((_, _, Some(g))) => *g,
        _ => -1.0,
    }
}

#[wasm_bindgen]
pub fn bloom_profiler_hist_count() -> f64 {
    engine().profiler.frame_history().len() as f64
}

#[wasm_bindgen]
pub fn bloom_profiler_hist_cpu_us(i: f64) -> f64 {
    engine()
        .profiler
        .frame_history()
        .get(i as usize)
        .map(|(cpu, _)| *cpu)
        .unwrap_or(0.0)
}

#[wasm_bindgen]
pub fn bloom_profiler_hist_gpu_us(i: f64) -> f64 {
    engine()
        .profiler
        .frame_history()
        .get(i as usize)
        .map(|(_, gpu)| *gpu)
        .unwrap_or(0.0)
}

// ============================================================
// Host stubs — features a browser page cannot provide, exported so
// the calls stay cheap typed no-ops
// ============================================================

/// The browser owns presentation (rAF + compositor); Fifo-equivalent
/// behaviour is all a canvas surface can do, so the mode is fixed.
#[wasm_bindgen]
pub fn bloom_set_present_mode(_mode: f64) {}

/// Screenshot readback needs a blocking `device.poll(Wait)`, which a
/// single-threaded wasm host cannot do. Right-click-save or the DOM
/// `canvas.toBlob` path (from JS) are the web equivalents.
#[wasm_bindgen]
pub fn bloom_take_screenshot(_path: f64) {}
