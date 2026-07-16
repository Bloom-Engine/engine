//! Material-system FFI surface for web: material compile/param/
//! draw-submission exports plus instance buffers, texture arrays and
//! reflection-probe wiring. Split from lib.rs (2000-line file policy).

use crate::engine;
use wasm_bindgen::prelude::*;

// ============================================================
// Phase 1c — material system FFI
// ============================================================

#[wasm_bindgen]
pub fn bloom_set_material_params(_handle: f64, _params_ptr: f64, _param_count: f64) {
    // No-op: JS glue calls bloom_set_material_params_floats instead
}

#[wasm_bindgen]
pub fn bloom_set_material_params_floats(handle: f64, params: &[f32]) {
    let count = params.len();
    if count > 64 {
        web_sys::console::error_1(&format!(
            "[material] set_material_params: param_count {} > 64 (256-byte UBO cap)",
            count
        ).into());
        return;
    }
    let mut bytes = vec![0u8; count * 4];
    for (i, &v) in params.iter().enumerate() {
        bytes[i*4..i*4+4].copy_from_slice(&v.to_le_bytes());
    }
    let eng = engine();
    if let Err(e) = eng.renderer.material_system.set_user_params(
        &eng.renderer.device, &eng.renderer.queue,
        handle as u32, &bytes,
    ) {
        web_sys::console::error_1(&format!("[material] set_material_params failed: {}", e).into());
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_str(source: &str) -> f64 {
    match engine().renderer.compile_material(source) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_refractive(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_refractive_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_refractive_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Refractive, true, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[refractive] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_transparent(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_transparent_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_transparent_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Transparent, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_additive(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_additive_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_additive_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Additive, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_cutout(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_cutout_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_cutout_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Opaque, Bucket::Cutout, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_instanced(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_instanced_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_instanced_str(source: &str) -> f64 {
    match engine().renderer.compile_material_instanced(source) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] instanced compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_create_instance_buffer(_data_ptr: f64, _instance_count: f64) -> f64 {
    // No-op: JS glue calls bloom_create_instance_buffer_floats instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_create_instance_buffer_floats(data: &[f32], instance_count: f64) -> f64 {
    if instance_count <= 0.0 { return 0.0; }
    let count = instance_count as u32;
    engine().renderer.create_instance_buffer(data, count) as f64
}

#[wasm_bindgen]
pub fn bloom_submit_material_draw_instanced(
    material: f64, mesh_handle: f64, mesh_idx: f64,
    instance_buffer: f64, instance_count: f64,
) {
    let eng = engine();
    let handle_bits = mesh_handle.to_bits();
    if let Some(model) = eng.models.get(mesh_handle) {
        eng.renderer.cache_model_if_static(handle_bits, &model.meshes);
    }
    eng.renderer.submit_material_draw_instanced(
        material as u32,
        handle_bits,
        mesh_idx as usize,
        instance_buffer as u32,
        instance_count as u32,
    );
}

#[wasm_bindgen]
pub fn bloom_destroy_instance_buffer(handle: f64) {
    engine().renderer.destroy_instance_buffer(handle as u32);
}

/// EN-011 — create a planar reflection probe. See macOS lib.rs for the
/// full doc comment. Web ports the same FFI surface so a TypeScript
/// game targets one API across native + browser.
#[wasm_bindgen]
pub fn bloom_create_planar_reflection(
    plane_y: f64, nx: f64, ny: f64, nz: f64, resolution: f64,
) -> f64 {
    engine().renderer.create_planar_reflection(
        plane_y as f32,
        [nx as f32, ny as f32, nz as f32],
        resolution as u32,
    ) as f64
}

/// EN-011 — link a material to a planar reflection probe. `probe = 0`
/// reverts the binding to the engine's default 1×1 black texture.
#[wasm_bindgen]
pub fn bloom_set_material_reflection_probe(
    material: f64, probe: f64,
) {
    engine().renderer.set_material_reflection_probe(material as u32, probe as u32);
}

/// EN-014 — pointer-shaped variant exists only so the FFI manifest
/// validates against the Perry surface; JS glue calls
/// `bloom_create_texture_array_bytes` with a `Uint8Array` instead.
/// Same precedent as `bloom_create_instance_buffer` (see EN-001).
#[wasm_bindgen]
pub fn bloom_create_texture_array(
    _data_ptr: f64, _data_len: f64, _width: f64, _height: f64, _layer_count: f64,
) -> f64 {
    0.0
}

/// EN-014 — create a texture array from concatenated RGBA8 bytes.
/// JS glue passes a `Uint8Array` of `layer_count × width × height × 4`
/// bytes (each layer back-to-back). Layer count is capped at 16; the
/// rest are silently dropped. Returns a 1-based handle (0 on failure).
#[wasm_bindgen]
pub fn bloom_create_texture_array_bytes(
    data: &[u8],
    width: f64, height: f64, layer_count: f64,
) -> f64 {
    // EN-014 V2 — V1 forwards to _ex with default sRGB / no mips.
    bloom_create_texture_array_ex_bytes(data, width, height, layer_count, 0.0, 1.0)
}

/// EN-014 V2 — pointer-shaped Ex variant exists only so the FFI manifest
/// validates against the Perry surface; JS glue uses the `_bytes` form.
#[wasm_bindgen]
pub fn bloom_create_texture_array_ex(
    _data_ptr: f64, _data_len: f64, _width: f64, _height: f64,
    _layer_count: f64, _format: f64, _mip_levels: f64,
) -> f64 {
    0.0
}

/// EN-014 V2 — bytes form of `_ex`. See `MaterialSystem::create_texture_array_ex`
/// for `format` (0 = sRGB, 1 = linear) and `mip_levels` (1 = none, 0 =
/// auto-generate via point-sample copies) semantics.
#[wasm_bindgen]
pub fn bloom_create_texture_array_ex_bytes(
    data: &[u8],
    width: f64, height: f64, layer_count: f64,
    format: f64, mip_levels: f64,
) -> f64 {
    let w = width as u32;
    let h = height as u32;
    if w == 0 || h == 0 { return 0.0; }
    let layers_count = (layer_count as u32)
        .min(bloom_shared::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
    if layers_count == 0 { return 0.0; }
    let layer_size = (w as usize) * (h as usize) * 4;
    let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(layers_count as usize);
    for i in 0..(layers_count as usize) {
        let start = i * layer_size;
        let end   = start + layer_size;
        if end > data.len() { break; }
        layers.push((&data[start..end], w, h));
    }
    engine().renderer.create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
}

/// EN-014 — link a texture-array handle to a material slot
/// (0 = albedo / 1 = normal / 2 = MR). `array = 0` reverts to the
/// engine's 1×1×1 stub.
#[wasm_bindgen]
pub fn bloom_set_material_texture_array(
    material: f64, slot: f64, array: f64,
) {
    engine().renderer.set_material_texture_array(
        material as u32, slot as u32, array as u32,
    );
}

/// EN-012 — set the shading model for a material (0=default lit,
/// 1=foliage, 2=subsurface V2 stub).
#[wasm_bindgen]
pub fn bloom_set_material_shading_model(
    material: f64, model: f64,
) {
    engine().renderer.set_material_shading_model(material as u32, model as u32);
}

/// Whether a material's draws render into planar-reflection probes
/// (default true). Authoring control for content that is sub-pixel at
/// probe resolution (e.g. instanced grass).
#[wasm_bindgen]
pub fn bloom_set_material_probe_visible(
    material: f64, visible: f64,
) {
    engine().renderer.set_material_probe_visible(material as u32, visible != 0.0);
}

/// EN-012 — set the foliage shading parameters for a material.
/// Only takes effect when shading_model == 1 (foliage).
#[wasm_bindgen]
pub fn bloom_set_material_foliage(
    material: f64,
    trans_r: f64, trans_g: f64, trans_b: f64,
    trans_amount: f64, wrap_factor: f64,
) {
    engine().renderer.set_material_foliage(
        material as u32,
        [trans_r as f32, trans_g as f32, trans_b as f32],
        trans_amount as f32, wrap_factor as f32,
    );
}

#[wasm_bindgen]
pub fn bloom_compile_material_from_file(_path: f64, _bucket_kind: f64) -> f64 {
    // No-op: web has no filesystem — JS glue would have to fetch + call
    // bloom_compile_material_from_file_str instead.
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_from_file_str(path: &str, bucket_kind: f64) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let (profile, bucket, reads_scene) = match bucket_kind as u32 {
        0 => (FragmentProfile::Opaque,      Bucket::Opaque,      false),
        1 => (FragmentProfile::Translucent, Bucket::Transparent, false),
        2 => (FragmentProfile::Translucent, Bucket::Refractive,  true),
        3 => (FragmentProfile::Translucent, Bucket::Additive,    false),
        4 => (FragmentProfile::Opaque,      Bucket::Cutout,      false),
        _ => {
            web_sys::console::error_1(&format!(
                "[material] from_file: unknown bucket_kind {}", bucket_kind
            ).into());
            return 0.0;
        }
    };
    match engine().renderer.compile_material_from_file(
        std::path::Path::new(path), profile, bucket, reads_scene,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] from_file failed: {}", e).into());
            0.0
        }
    }
}

/// EN-017 — stub: JS glue calls `bloom_set_post_pass_str` instead.
#[wasm_bindgen]
pub fn bloom_set_post_pass(_source: f64) -> f64 { 0.0 }

/// EN-017 — compile + install a fullscreen post-pass material on web.
/// See `bloom-macos::bloom_set_post_pass` for the full ABI. Returns
/// 1.0 on success, 0.0 on compile failure.
#[wasm_bindgen]
pub fn bloom_set_post_pass_str(source: &str) -> f64 {
    match engine().renderer.set_post_pass(source) {
        Ok(()) => 1.0,
        Err(e) => {
            web_sys::console::error_1(
                &format!("[post_pass] compile failed: {:?}", e).into(),
            );
            0.0
        }
    }
}

/// EN-017 — uninstall the active post-pass.
#[wasm_bindgen]
pub fn bloom_clear_post_pass() {
    engine().renderer.clear_post_pass();
}

/// EN-017 V2 — stub: JS glue calls `bloom_add_post_pass_str` instead.
#[wasm_bindgen]
pub fn bloom_add_post_pass(_source: f64) -> f64 { 0.0 }

/// EN-017 V2 — append a fullscreen post-pass to the stack on web.
/// See `bloom-macos::bloom_add_post_pass` for the full ABI. Returns
/// 1-based handle on success, 0.0 on compile failure.
#[wasm_bindgen]
pub fn bloom_add_post_pass_str(source: &str) -> f64 {
    match engine().renderer.add_post_pass(source) {
        Ok(h) => h as f64,
        Err(e) => {
            web_sys::console::error_1(
                &format!("[post_pass] compile failed: {:?}", e).into(),
            );
            0.0
        }
    }
}

/// EN-017 V2 — wipe the entire post-pass stack.
#[wasm_bindgen]
pub fn bloom_clear_all_post_passes() {
    engine().renderer.clear_all_post_passes();
}

#[wasm_bindgen]
pub fn bloom_draw_material(
    material: f64,
    mesh_handle: f64,
    mesh_idx: f64,
    x: f64, y: f64, z: f64, scale: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let eng = engine();
    let handle_bits = mesh_handle.to_bits();
    if let Some(model) = eng.models.get(mesh_handle) {
        eng.renderer.cache_model_if_static(handle_bits, &model.meshes);
    }
    eng.renderer.submit_material_draw(
        material as u32,
        handle_bits,
        mesh_idx as usize,
        [x as f32, y as f32, z as f32],
        scale as f32,
        [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32],
    );
}

#[wasm_bindgen]
pub fn bloom_load_model_animation(_path: f64) -> f64 { 0.0 }

#[wasm_bindgen]
pub fn bloom_load_model_animation_bytes(data: &[u8]) -> f64 {
    engine().models.load_model_animation(data)
}

#[wasm_bindgen]
pub fn bloom_update_model_animation(_handle: f64, _anim_index: f64, _time: f64, _scale: f64, _px: f64, _py: f64, _pz: f64, _rot_sin: f64, _rot_cos: f64) {
    // TODO: Phase 4 — depends on bloom_load_model_animation
}

#[wasm_bindgen]
pub fn bloom_get_model_mesh_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

#[wasm_bindgen]
pub fn bloom_get_model_material_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

// ============================================================
// EN-028 animation mixer / EN-033 sockets / EN-026 particles /
// EN-027 decals — web ports.
//
// The mixer and both VFX systems are pure CPU state in bloom-shared, so the
// web crate gets the real behaviour, not a stub: same ModelManager, same
// ParticleManager, same DecalManager. The only web-specific care is that the
// particle/decal *upload* path goes through the same material-system dynamic
// instance buffer, which WebGPU supports.
// ============================================================

#[wasm_bindgen]
pub fn bloom_anim_play(handle: f64, clip: f64, fade: f64, speed: f64, looping: f64) {
    engine().models.anim_play(handle, clip as usize, fade as f32, speed as f32, looping != 0.0);
}

#[wasm_bindgen]
pub fn bloom_anim_set_layer(handle: f64, clip: f64, weight: f64, mask_root: f64, speed: f64, looping: f64) {
    engine().models.anim_set_layer(handle, clip as i32, weight as f32, mask_root as i32, speed as f32, looping != 0.0);
}

#[wasm_bindgen]
pub fn bloom_anim_set_root_motion(handle: f64, on: f64) {
    engine().models.anim_set_root_motion(handle, on != 0.0);
}

#[wasm_bindgen]
pub fn bloom_anim_update(handle: f64, dt: f64, scale: f64, px: f64, py: f64, pz: f64, rot_y: f64) {
    let rot_y_f = rot_y as f32;
    let eng = engine();
    eng.models.advance_and_update(handle, dt as f32);
    if let Some(anim) = eng.models.get_animation(handle) {
        if !anim.joint_matrices.is_empty() {
            // PT-7: the anim handle keys the prev-palette pairing for skinned
            // motion vectors — the same key the shared FFI passes
            // (ffi_core/models.rs). Without it this call does not compile: the
            // web backend was left behind when `key` was added, which is what
            // broke build-web on main.
            eng.renderer.set_joint_matrices_scaled(
                handle.to_bits(), &anim.joint_matrices, scale as f32,
                [px as f32, py as f32, pz as f32], rot_y_f.sin(), rot_y_f.cos());
        }
    }
}

#[wasm_bindgen]
pub fn bloom_anim_finished(handle: f64) -> f64 {
    if engine().models.anim_finished(handle) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_anim_clip_duration(handle: f64, clip: f64) -> f64 {
    engine().models.anim_clip_duration(handle, clip as usize) as f64
}

#[wasm_bindgen]
pub fn bloom_anim_root_delta(handle: f64, axis: f64) -> f64 {
    let d = engine().models.anim_root_delta(handle);
    let i = axis as usize;
    if i < 3 { d[i] as f64 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_model_find_joint(handle: f64, name: String) -> f64 {
    engine().models.find_joint(handle, &name) as f64
}

#[wasm_bindgen]
pub fn bloom_model_joint_world(handle: f64, joint: f64, comp: f64) -> f64 {
    let j = joint as i64;
    let c = comp as usize;
    if j < 0 || c > 15 { return 0.0; }
    match engine().models.joint_world(handle, j as usize) {
        Some(m) => m[c / 4][c % 4] as f64,
        None => 0.0,
    }
}

#[wasm_bindgen]
pub fn bloom_particles_create(capacity: f64) -> f64 {
    let cap = (capacity as usize).clamp(1, 100_000);
    let eng = engine();
    let ib = eng.renderer.material_system.create_dynamic_instance_buffer(&eng.renderer.device, cap as u32);
    eng.particles.create(cap, ib) as f64
}

#[wasm_bindgen]
pub fn bloom_particles_configure(sys: f64) {
    let eng = engine();
    let params: Vec<f32> = eng.models.scratch_f32.clone();
    eng.models.mesh_scratch_reset();
    if let Some(s) = eng.particles.get_mut(sys as u32) {
        s.configure_from_slice(&params);
    }
}

#[wasm_bindgen]
pub fn bloom_particles_emit(sys: f64, x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64, count: f64) {
    if let Some(s) = engine().particles.get_mut(sys as u32) {
        s.emit([x as f32, y as f32, z as f32], [dx as f32, dy as f32, dz as f32], (count as usize).min(4096));
    }
}

#[wasm_bindgen]
pub fn bloom_particles_update(sys: f64, dt: f64) -> f64 {
    let eng = engine();
    let (live, ib) = match eng.particles.get_mut(sys as u32) {
        Some(s) => (s.update(dt as f32), s.instance_buffer),
        None => return 0.0,
    };
    if live > 0 {
        let packed: Vec<f32> = match eng.particles.get_mut(sys as u32) {
            Some(s) => s.packed()[..(live as usize) * 12].to_vec(),
            None => return 0.0,
        };
        eng.renderer.material_system.update_instance_buffer(&eng.renderer.queue, ib, &packed, live);
    }
    live as f64
}

#[wasm_bindgen]
pub fn bloom_particles_instance_buffer(sys: f64) -> f64 {
    engine().particles.get_mut(sys as u32).map(|s| s.instance_buffer as f64).unwrap_or(0.0)
}

#[wasm_bindgen]
pub fn bloom_particles_clear(sys: f64) {
    if let Some(s) = engine().particles.get_mut(sys as u32) { s.clear(); }
}

#[wasm_bindgen]
pub fn bloom_particles_live(sys: f64) -> f64 {
    engine().particles.get_mut(sys as u32).map(|s| s.live as f64).unwrap_or(0.0)
}

#[wasm_bindgen]
pub fn bloom_decals_init(capacity: f64) -> f64 {
    let cap = (capacity as usize).clamp(1, 8192);
    let eng = engine();
    let ib = eng.renderer.material_system.create_dynamic_instance_buffer(&eng.renderer.device, cap as u32);
    eng.decals.init(cap, ib);
    ib as f64
}

#[wasm_bindgen]
pub fn bloom_decals_spawn(x: f64, y: f64, z: f64, nx: f64, ny: f64, nz: f64, size: f64, roll: f64) {
    engine().decals.spawn_styled(
        [x as f32, y as f32, z as f32],
        [nx as f32, ny as f32, nz as f32],
        size as f32, roll as f32);
}

#[wasm_bindgen]
pub fn bloom_decals_set_style(frame: f64, r: f64, g: f64, b: f64, a: f64, life: f64, fade: f64) {
    engine().decals.style = bloom_shared::decals::DecalStyle {
        frame: frame as f32,
        color: [r as f32, g as f32, b as f32, a as f32],
        life: life as f32,
        fade: fade as f32,
    };
}

#[wasm_bindgen]
pub fn bloom_decals_update(dt: f64) -> f64 {
    let eng = engine();
    let live = eng.decals.update(dt as f32);
    let ib = eng.decals.instance_buffer;
    if live > 0 {
        let packed: Vec<f32> = eng.decals.packed()[..(live as usize) * 12].to_vec();
        eng.renderer.material_system.update_instance_buffer(&eng.renderer.queue, ib, &packed, live);
    }
    live as f64
}

#[wasm_bindgen]
pub fn bloom_decals_instance_buffer() -> f64 {
    engine().decals.instance_buffer as f64
}

#[wasm_bindgen]
pub fn bloom_decals_clear() {
    engine().decals.clear();
}

// EN-029 — audio buses / reverb / occlusion low-pass. All CPU DSP in
// bloom-shared's AudioMixer, so web gets the real thing.

#[wasm_bindgen]
pub fn bloom_set_sound_bus(handle: f64, bus: f64) {
    engine().audio.set_sound_bus(handle, bus as u8);
}

#[wasm_bindgen]
pub fn bloom_set_sound_reverb_send(handle: f64, send: f64) {
    engine().audio.set_sound_reverb_send(handle, send as f32);
}

#[wasm_bindgen]
pub fn bloom_set_sound_lowpass(handle: f64, cutoff: f64) {
    engine().audio.set_sound_lowpass(handle, cutoff as f32);
}

#[wasm_bindgen]
pub fn bloom_set_bus_gain(bus: f64, gain: f64) {
    engine().audio.set_bus_gain(bus as u8, gain as f32);
}

#[wasm_bindgen]
pub fn bloom_duck_bus(bus: f64, amount: f64, attack: f64, release: f64, hold: f64) {
    engine().audio.duck_bus(bus as u8, amount as f32, attack as f32, release as f32, hold as f32);
}

#[wasm_bindgen]
pub fn bloom_set_reverb(size: f64, damp: f64, wet: f64) {
    engine().audio.set_reverb(size as f32, damp as f32, wet as f32);
}

#[wasm_bindgen]
pub fn bloom_compile_material_instanced_bucket(source: String, bucket: f64, reads_scene: f64) -> f64 {
    match engine().renderer.compile_material_instanced_bucket(&source, bucket as u32, reads_scene != 0.0) {
        Ok(handle) => handle as f64,
        Err(e) => { web_sys::console::error_1(&format!("[material] instanced compile failed: {:?}", e).into()); 0.0 }
    }
}
