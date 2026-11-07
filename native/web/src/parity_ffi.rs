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

/// Upload one packed little-endian RGBA8 texel per scratch entry. This is the
/// Web mirror of the native compatibility-texture path; the scalar scratch
/// ABI avoids passing a typed-array pointer between two WASM modules.
#[wasm_bindgen]
pub fn bloom_load_texture_rgba8_scratch(width: f64, height: f64, kind: f64) -> f64 {
    let width = width as u32;
    let height = height as u32;
    let Some(texel_count) = width.checked_mul(height).map(|count| count as usize) else {
        return 0.0;
    };
    let bytes = {
        let eng = engine();
        if width == 0 || height == 0 || eng.models.scratch_u32s().len() < texel_count {
            return 0.0;
        }
        let bytes = eng.models.scratch_u32s()[..texel_count]
            .iter()
            .flat_map(|texel| texel.to_le_bytes())
            .collect::<Vec<_>>();
        eng.models.mesh_scratch_reset();
        bytes
    };
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
    eng.textures.load_texture_rgba8(
        unsafe { &mut *renderer_ptr },
        width,
        height,
        &bytes,
        kind as u32,
    )
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
pub fn bloom_set_material_params_scratch(handle: f64, param_count: f64) -> f64 {
    let count = param_count as usize;
    let params: Vec<f32> = {
        let eng = engine();
        if eng.models.scratch_f32.len() < count {
            return 0.0;
        }
        let p = eng.models.scratch_f32[..count].to_vec();
        eng.models.mesh_scratch_reset();
        p
    };
    crate::material_ffi::bloom_set_material_params_floats(handle, &params)
}

// ============================================================
// Three/glTF retained-material parity
// ============================================================

#[wasm_bindgen]
pub fn bloom_scene_set_render_layer(handle: f64, layer: f64) -> f64 {
    engine()
        .scene
        .set_render_layer(handle, layer.max(0.0).min(u32::MAX as f64) as u32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_scene_set_material_texture_handles(
    handle: f64,
    base_color: f64,
    normal: f64,
    metallic_roughness: f64,
    emissive: f64,
    occlusion: f64,
) -> f64 {
    let eng = engine();
    let resolve = |texture_handle: f64| -> u32 {
        if texture_handle <= 0.0 {
            0
        } else {
            eng.textures
                .get(texture_handle)
                .map(|texture| texture.bind_group_idx)
                .unwrap_or(0)
        }
    };
    let resolved = [
        resolve(base_color),
        resolve(normal),
        resolve(metallic_roughness),
        resolve(emissive),
        resolve(occlusion),
    ];
    eng.scene.set_material_texture(handle, resolved[0]);
    eng.scene.set_material_normal_texture(handle, resolved[1]);
    eng.scene
        .set_material_metallic_roughness_texture(handle, resolved[2]);
    eng.scene.set_material_emissive_texture(handle, resolved[3]);
    eng.scene
        .set_material_occlusion_texture(handle, resolved[4]);
    1.0
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn bloom_scene_set_material_texture_transform(
    handle: f64,
    slot: f64,
    m00: f64,
    m01: f64,
    m02: f64,
    m10: f64,
    m11: f64,
    m12: f64,
) -> f64 {
    let Some(slot) =
        bloom_shared::models::StandardMaterialTextureSlot::from_u32(slot.max(0.0) as u32)
    else {
        return 0.0;
    };
    let transform = bloom_shared::models::MaterialTextureAffineTransform::from_rows(
        [m00 as f32, m01 as f32, m02 as f32],
        [m10 as f32, m11 as f32, m12 as f32],
    );
    engine()
        .scene
        .set_material_texture_transform(handle, slot, transform);
    1.0
}

#[wasm_bindgen]
pub fn bloom_scene_set_material_texture_strengths(
    handle: f64,
    normal_scale_x: f64,
    normal_scale_y: f64,
    occlusion_strength: f64,
) -> f64 {
    engine().scene.set_material_texture_strengths(
        handle,
        [normal_scale_x as f32, normal_scale_y as f32],
        occlusion_strength as f32,
    );
    1.0
}

// ============================================================
// Compatibility render targets
// ============================================================

#[wasm_bindgen]
pub fn bloom_load_render_texture_kind(width: f64, height: f64, kind: f64) -> f64 {
    let width = (width as u32).max(1);
    let height = (height as u32).max(1);
    let kind = (kind as u32).min(2);
    let eng = engine();
    let render_texture = eng.textures.load_render_texture_kind(width, height);
    let (bind_group_idx, _) = eng.renderer.create_render_texture_kind(width, height, kind);
    let texture = eng
        .textures
        .textures
        .alloc(bloom_shared::textures::TextureData {
            bind_group_idx,
            width,
            height,
        });
    eng.textures
        .set_render_texture_handle(render_texture, texture);
    render_texture
}

#[wasm_bindgen]
pub fn bloom_render_texture_glsl(render_texture: f64, source: &str) -> f64 {
    let eng = engine();
    let Some(target_index) = eng
        .textures
        .render_textures
        .get(render_texture)
        .and_then(|target| eng.textures.textures.get(target.texture_handle))
        .map(|texture| texture.bind_group_idx as usize)
    else {
        return 0.0;
    };
    match eng
        .renderer
        .render_glsl_fragment_to_texture(target_index, source)
    {
        Ok(()) => 1.0,
        Err(error) => {
            crate::console_warn(&format!("[compat-glsl] {error}"));
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_render_texture_sobel(render_texture: f64, source_texture: f64, strength: f64) -> f64 {
    let eng = engine();
    let Some(target_index) = eng
        .textures
        .render_textures
        .get(render_texture)
        .and_then(|target| eng.textures.textures.get(target.texture_handle))
        .map(|texture| texture.bind_group_idx as usize)
    else {
        return 0.0;
    };
    let Some(source_index) = eng
        .textures
        .get(source_texture)
        .map(|texture| texture.bind_group_idx as usize)
    else {
        return 0.0;
    };
    match eng
        .renderer
        .render_sobel_to_texture(target_index, source_index, strength as f32)
    {
        Ok(()) => 1.0,
        Err(error) => {
            crate::console_warn(&format!("[compat-sobel] {error}"));
            0.0
        }
    }
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
    let layers_count =
        (layer_count as u32).min(bloom_shared::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
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
pub fn bloom_scene_set_gi_only(handle: f64, gi_only: f64) -> f64 {
    engine().scene.set_gi_only(handle, gi_only != 0.0);
    1.0
}

#[wasm_bindgen]
pub fn bloom_scene_set_material_emissive(handle: f64, r: f64, g: f64, b: f64) -> f64 {
    let finite_non_negative = |value: f64| {
        if value.is_finite() {
            (value as f32).max(0.0)
        } else {
            0.0
        }
    };
    engine().scene.set_material_emissive_factor(
        handle,
        finite_non_negative(r),
        finite_non_negative(g),
        finite_non_negative(b),
    );
    1.0
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn bloom_scene_set_material_layered_pbr(
    handle: f64,
    lobe_mask: f64,
    clearcoat_factor: f64,
    clearcoat_roughness: f64,
    clearcoat_normal_scale: f64,
    specular_factor: f64,
    specular_r: f64,
    specular_g: f64,
    specular_b: f64,
    ior: f64,
    sheen_r: f64,
    sheen_g: f64,
    sheen_b: f64,
    sheen_roughness: f64,
    anisotropy_strength: f64,
    anisotropy_rotation: f64,
    iridescence_factor: f64,
    iridescence_ior: f64,
    iridescence_thickness_minimum: f64,
    iridescence_thickness_maximum: f64,
) -> f64 {
    let layered = bloom_shared::models::MaterialLayeredPbr::from_authoring_factors(
        lobe_mask as u32,
        clearcoat_factor as f32,
        clearcoat_roughness as f32,
        clearcoat_normal_scale as f32,
        specular_factor as f32,
        [specular_r as f32, specular_g as f32, specular_b as f32],
        ior as f32,
        [sheen_r as f32, sheen_g as f32, sheen_b as f32],
        sheen_roughness as f32,
        anisotropy_strength as f32,
        anisotropy_rotation as f32,
        iridescence_factor as f32,
        iridescence_ior as f32,
        iridescence_thickness_minimum as f32,
        iridescence_thickness_maximum as f32,
    );
    engine().scene.set_material_layered_pbr(handle, layered);
    1.0
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

/// Browser builds cannot resolve a native filesystem path for HDR input.
/// Call the byte-oriented web loader instead.
#[wasm_bindgen]
pub fn bloom_set_env_clear_from_hdr(_path: f64) -> f64 {
    0.0
}

/// The browser owns presentation (rAF + compositor); Fifo-equivalent
/// behaviour is all a canvas surface can do, so the mode is fixed.
#[wasm_bindgen]
pub fn bloom_set_present_mode(mode: f64) -> f64 {
    if mode == 0.0 {
        1.0
    } else {
        0.0
    }
}

#[wasm_bindgen]
pub fn bloom_get_present_mode() -> f64 {
    0.0
}

#[wasm_bindgen]
pub fn bloom_set_path_tracing(mode: f64) -> f64 {
    if mode == 0.0 {
        1.0
    } else {
        0.0
    }
}

#[wasm_bindgen]
pub fn bloom_path_tracing_supported() -> f64 {
    0.0
}

#[wasm_bindgen]
pub fn bloom_get_material_binding_capabilities() -> String {
    engine().renderer.material_binding_report_json()
}

#[wasm_bindgen]
pub fn bloom_get_renderer_capabilities() -> String {
    engine().renderer.renderer_capability_report_json()
}

#[wasm_bindgen]
pub fn bloom_get_imported_refraction_mode() -> f64 {
    engine().renderer.imported_refraction_mode_code() as f64
}

#[wasm_bindgen]
pub fn bloom_set_transparency_composition_mode(mode: f64) -> f64 {
    engine()
        .renderer
        .set_transparency_composition_mode(mode as u32);
    1.0
}

#[wasm_bindgen]
pub fn bloom_get_transparency_composition_mode() -> f64 {
    engine().renderer.transparency_composition_mode_code() as f64
}

#[wasm_bindgen]
pub fn bloom_get_active_transparency_composition_mode() -> f64 {
    engine()
        .renderer
        .active_transparency_composition_mode_code() as f64
}

#[wasm_bindgen]
pub fn bloom_set_material_binding_tier_override(tier: f64) -> f64 {
    engine()
        .renderer
        .set_material_binding_tier_override(tier as u32) as u8 as f64
}

/// Screenshot readback needs a blocking `device.poll(Wait)`, which a
/// single-threaded wasm host cannot do. Right-click-save or the DOM
/// `canvas.toBlob` path (from JS) are the web equivalents.
#[wasm_bindgen]
pub fn bloom_take_screenshot(_path: f64) {}

#[wasm_bindgen]
pub fn bloom_capture_frame_to_png(_path: f64) -> f64 {
    0.0
}

#[wasm_bindgen]
pub fn bloom_capture_debug_intermediates(_path: f64) -> f64 {
    0.0
}

#[wasm_bindgen]
pub fn bloom_capture_frame_ready() -> f64 {
    0.0
}

/// Browser builds have no direct filesystem path for qualification output.
#[wasm_bindgen]
pub fn bloom_write_quality_telemetry(
    _path: f64,
    _warmup_frames: f64,
    _measured_frames: f64,
    _fixed_timestep: f64,
    _quality_preset: f64,
    _render_scale: f64,
    _measurement_wall_ms: f64,
) -> f64 {
    0.0
}
