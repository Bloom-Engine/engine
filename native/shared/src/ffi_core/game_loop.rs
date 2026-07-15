//! Game loop, timing, frame callbacks, profiler, window-state getters.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_game_loop {
    () => {

        // --- game loop hooks --------------------------------------------

        // No-op on native: the TypeScript runGame() helper drives the
        // loop. (On web, the JS glue hooks requestAnimationFrame.)
        #[no_mangle]
        pub extern "C" fn bloom_run_game(_callback: extern "C" fn(f64)) {}

        #[no_mangle]
        pub extern "C" fn bloom_register_frame_callback(priority: f64, callback: extern "C" fn(f64)) -> f64 {
            $crate::ffi::guard("bloom_register_frame_callback", move || {
                engine().frame_callbacks.register(priority as i32, callback) as f64
            })
        }

        // bloom_get_delta_time  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_delta_time() -> f64 {
            $crate::ffi::guard("bloom_get_delta_time", move || {
                engine().delta_time
        })
        }

        // bloom_get_fps  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_fps() -> f64 {
            $crate::ffi::guard("bloom_get_fps", move || {
                engine().get_fps()
        })
        }

        // bloom_get_screen_width  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_screen_width() -> f64 {
            $crate::ffi::guard("bloom_get_screen_width", move || {
                engine().screen_width()
        })
        }

        // bloom_get_screen_height  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_screen_height() -> f64 {
            $crate::ffi::guard("bloom_get_screen_height", move || {
                engine().screen_height()
        })
        }

        // bloom_create_instance_buffer  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_create_instance_buffer(
            data_ptr: *const f64, instance_count: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_create_instance_buffer", move || {
                if data_ptr.is_null() || instance_count <= 0.0 { return 0.0; }
                let count = instance_count as u32;
                let slot_count = (count as usize) * 9;
                let raw_f64 = unsafe { std::slice::from_raw_parts(data_ptr, slot_count) };
                let raw_f32: Vec<f32> = raw_f64.iter().map(|&v| v as f32).collect();
                engine().renderer.create_instance_buffer(&raw_f32, count) as f64
        })
        }

        // bloom_create_instance_buffer_scratch — instance data arrives via
        // the mesh scratch (bloom_mesh_scratch_reset + push_f32 × 9·N)
        // instead of the i64-array pointer above, which Perry 0.5.x rejects
        // ("Expected safe integer for native i64 parameter"). Same idiom as
        // bloom_create_mesh_scratch. 9 f32 slots per instance.
        #[no_mangle]
        pub extern "C" fn bloom_create_instance_buffer_scratch(instance_count: f64) -> f64 {
            $crate::ffi::guard("bloom_create_instance_buffer_scratch", move || {
                let eng = engine();
                let count = instance_count as u32;
                let need = (count as usize) * 9;
                if count == 0 || eng.models.scratch_f32.len() < need { return 0.0; }
                let data: Vec<f32> = eng.models.scratch_f32[..need].to_vec();
                eng.models.mesh_scratch_reset();
                eng.renderer.create_instance_buffer(&data, count) as f64
        })
        }

        // bloom_destroy_instance_buffer  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_destroy_instance_buffer(handle: f64) {
            $crate::ffi::guard("bloom_destroy_instance_buffer", move || {
                engine().renderer.destroy_instance_buffer(handle as u32);
        })
        }

        // bloom_create_planar_reflection  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_create_planar_reflection(
            plane_y: f64, nx: f64, ny: f64, nz: f64, resolution: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_create_planar_reflection", move || {
                engine().renderer.create_planar_reflection(
                    plane_y as f32,
                    [nx as f32, ny as f32, nz as f32],
                    resolution as u32,
                ) as f64
        })
        }

        // bloom_create_texture_array  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_create_texture_array(
            data_ptr:    *const u8,
            data_len:    f64,
            width:       f64,
            height:      f64,
            layer_count: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_create_texture_array", move || {
                // EN-014 V2 — V1 stays callable; forwards to _ex with default
                // format = sRGB (0) and mip_levels = 1 (no mips).
                bloom_create_texture_array_ex(data_ptr, data_len, width, height, layer_count, 0.0, 1.0)
        })
        }

        // bloom_create_texture_array_ex  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_create_texture_array_ex(
            data_ptr:    *const u8,
            data_len:    f64,
            width:       f64,
            height:      f64,
            layer_count: f64,
            format:      f64,
            mip_levels:  f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_create_texture_array_ex", move || {
                if data_ptr.is_null() || data_len <= 0.0 { return 0.0; }
                let w = width as u32;
                let h = height as u32;
                if w == 0 || h == 0 { return 0.0; }
                let layers_count = (layer_count as u32)
                    .min($crate::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
                if layers_count == 0 { return 0.0; }
                let layer_size = (w as usize) * (h as usize) * 4;
                let total_bytes = (data_len as usize)
                    .min(layers_count as usize * layer_size);
                let bytes = unsafe { std::slice::from_raw_parts(data_ptr, total_bytes) };
                let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(layers_count as usize);
                for i in 0..(layers_count as usize) {
                    let start = i * layer_size;
                    let end   = start + layer_size;
                    if end > bytes.len() { break; }
                    layers.push((&bytes[start..end], w, h));
                }
                engine().renderer.create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
        })
        }

        // bloom_create_texture_array_scratch  [EN-049]
        //
        // The byte-array path above takes a `*const u8`, which the manifest has
        // to declare as `i64` — and Perry cannot put a `number[]` in an i64
        // param (it raises "Expected safe integer for native i64 parameter").
        // So from Perry, `bloom_create_texture_array_ex` is only callable with a
        // pointer nobody has. Every real caller ended up on the from_files path,
        // which is fine for ART and useless for DATA: a splat map is computed at
        // load from the world file, and there is no file to name.
        //
        // Same fix the mesh path already uses (`bloom_create_mesh_scratch`):
        // push the payload through the scratch buffer, then call this with the
        // dimensions. Texels are pushed as PACKED u32 (one per texel, RGBA in
        // little-endian byte order = R | G<<8 | B<<16 | A<<24), so a 128² splat
        // is 16,384 pushes rather than 65,536.
        #[no_mangle]
        pub extern "C" fn bloom_create_texture_array_scratch(
            width:       f64,
            height:      f64,
            layer_count: f64,
            format:      f64,
            mip_levels:  f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_create_texture_array_scratch", move || {
                let w = width as u32;
                let h = height as u32;
                if w == 0 || h == 0 { return 0.0; }
                let layers_count = (layer_count as u32)
                    .min($crate::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
                if layers_count == 0 { return 0.0; }

                let texels = (w as usize) * (h as usize) * (layers_count as usize);
                // Scoped so the scratch borrow ends before the renderer borrow.
                let bytes: Vec<u8> = {
                    let eng = engine();
                    if eng.models.scratch_u32.len() < texels {
                        // Short buffer: refuse rather than upload uninitialised
                        // memory as a texture. Silent garbage here surfaces as a
                        // terrain painted in noise, three layers from the cause.
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
                    let end   = start + layer_size;
                    if end > bytes.len() { break; }
                    layers.push((&bytes[start..end], w, h));
                }
                engine().renderer.create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
        })
        }

        // bloom_create_texture_array_from_files  [EN-014 V3]
        //
        // The byte-array path above asks the game to marshal every texel across
        // the FFI — a 6-layer 256² array is 1.5 M numbers, which is exactly the
        // shape of call Perry's array bridge is worst at. Decoding on this side
        // from a path list is both faster and the API every actual consumer
        // wants (VFX atlas layers, splat-terrain layers).
        //
        // `paths` is comma-separated, and is parsed HERE, at load time — never
        // on a per-frame path (perry-quirks #5). Layers must share dimensions;
        // the first file's size wins and any mismatch is skipped with a warning.
        #[no_mangle]
        pub extern "C" fn bloom_create_texture_array_from_files(
            paths_ptr: *const u8, format: f64, mip_levels: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_create_texture_array_from_files", move || {
                let list = $crate::string_header::str_from_header(paths_ptr);
                let mut decoded: Vec<(Vec<u8>, u32, u32)> = Vec::new();
                for p in list.split(',') {
                    let p = p.trim();
                    if p.is_empty() { continue; }
                    let resolved = bloom_resolve_asset_path(p);
                    match image::open(resolved.as_ref()) {
                        Ok(img) => {
                            let rgba = img.to_rgba8();
                            let (w, h) = rgba.dimensions();
                            decoded.push((rgba.into_raw(), w, h));
                        }
                        Err(e) => {
                            eprintln!("[texarray] failed to load {}: {}", p, e);
                        }
                    }
                }
                if decoded.is_empty() { return 0.0; }
                let (w, h) = (decoded[0].1, decoded[0].2);
                let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(decoded.len());
                for (bytes, lw, lh) in decoded.iter() {
                    if *lw != w || *lh != h {
                        eprintln!("[texarray] layer size {}x{} != {}x{}; skipped", lw, lh, w, h);
                        continue;
                    }
                    layers.push((bytes.as_slice(), w, h));
                }
                if layers.is_empty() { return 0.0; }
                engine().renderer.create_texture_array_ex(
                    &layers, format as u32, mip_levels as u32) as f64
        })
        }

        // bloom_clear_post_pass  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_clear_post_pass() {
            $crate::ffi::guard("bloom_clear_post_pass", move || {
                engine().renderer.clear_post_pass();
        })
        }

        // bloom_add_post_pass  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_add_post_pass(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_add_post_pass", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.add_post_pass(source) {
                    Ok(h) => h as f64,
                    Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_clear_all_post_passes  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_clear_all_post_passes() {
            $crate::ffi::guard("bloom_clear_all_post_passes", move || {
                engine().renderer.clear_all_post_passes();
        })
        }

        // bloom_get_render_scale  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_render_scale() -> f64 {
            $crate::ffi::guard("bloom_get_render_scale", move || {
                engine().renderer.render_scale() as f64
        })
        }

        // bloom_get_physical_width  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_physical_width() -> f64 {
            $crate::ffi::guard("bloom_get_physical_width", move || {
                engine().renderer.physical_width() as f64
        })
        }

        // bloom_get_physical_height  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_physical_height() -> f64 {
            $crate::ffi::guard("bloom_get_physical_height", move || {
                engine().renderer.physical_height() as f64
        })
        }

        // bloom_get_profiler_frame_cpu_us  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_profiler_frame_cpu_us() -> f64 {
            $crate::ffi::guard("bloom_get_profiler_frame_cpu_us", move || {
                engine().profiler.avg_frame_cpu_us()
        })
        }

        // bloom_get_profiler_frame_gpu_us  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_profiler_frame_gpu_us() -> f64 {
            $crate::ffi::guard("bloom_get_profiler_frame_gpu_us", move || {
                engine().profiler.avg_frame_gpu_us()
        })
        }

        // bloom_print_profiler_summary  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_print_profiler_summary() {
            $crate::ffi::guard("bloom_print_profiler_summary", move || {
                print!("{}", engine().profiler.summary());
        })
        }

        // bloom_inject_key_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_inject_key_down(key: f64) {
            $crate::ffi::guard("bloom_inject_key_down", move || {
                engine().input.inject_key_down(key as usize);
        })
        }

        // bloom_inject_key_up  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_inject_key_up(key: f64) {
            $crate::ffi::guard("bloom_inject_key_up", move || {
                engine().input.inject_key_up(key as usize);
        })
        }

        // bloom_inject_gamepad_axis  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_inject_gamepad_axis(axis: f64, value: f64) {
            $crate::ffi::guard("bloom_inject_gamepad_axis", move || {
                engine().input.set_gamepad_axis(axis as usize, value as f32);
        })
        }

        // bloom_inject_gamepad_button_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_inject_gamepad_button_down(button: f64) {
            $crate::ffi::guard("bloom_inject_gamepad_button_down", move || {
                engine().input.set_gamepad_button_down(button as usize);
        })
        }

        // bloom_inject_gamepad_button_up  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_inject_gamepad_button_up(button: f64) {
            $crate::ffi::guard("bloom_inject_gamepad_button_up", move || {
                engine().input.set_gamepad_button_up(button as usize);
        })
        }

        // bloom_is_any_input_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
            $crate::ffi::guard("bloom_is_any_input_pressed", move || {
                if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
        })
        }

        // bloom_update_music_stream  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_update_music_stream(handle: f64) {
            $crate::ffi::guard("bloom_update_music_stream", move || {
                engine().audio.update_music_stream(handle);
        })
        }

        // bloom_get_time  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_time() -> f64 {
            $crate::ffi::guard("bloom_get_time", move || {
                engine().get_time()
        })
        }

        // bloom_unregister_frame_callback  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_unregister_frame_callback(id: f64) {
            $crate::ffi::guard("bloom_unregister_frame_callback", move || {
                engine().frame_callbacks.unregister(id as u64);
        })
        }

        // bloom_get_render_texture_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_render_texture_texture(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_render_texture_texture", move || {
                engine().textures.get_render_texture_texture(handle)
        })
        }

        // bloom_postfx_set_selected  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_postfx_set_selected(handle: f64) {
            $crate::ffi::guard("bloom_postfx_set_selected", move || {
                if let Some(pfx) = &mut engine().postfx {
                    if handle == 0.0 {
                        pfx.set_selected(Vec::new());
                    } else {
                        pfx.set_selected(vec![handle]);
                    }
                }
        })
        }

        // bloom_postfx_set_hovered  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_postfx_set_hovered(handle: f64) {
            $crate::ffi::guard("bloom_postfx_set_hovered", move || {
                if let Some(pfx) = &mut engine().postfx {
                    pfx.set_hovered(handle);
                }
        })
        }

        // bloom_postfx_set_outline_color  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_postfx_set_outline_color(r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_postfx_set_outline_color", move || {
                if let Some(pfx) = &mut engine().postfx {
                    pfx.outline_params.color_selected = [r as f32, g as f32, b as f32, a as f32];
                }
        })
        }

        // bloom_postfx_set_outline_thickness  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_postfx_set_outline_thickness(thickness: f64) {
            $crate::ffi::guard("bloom_postfx_set_outline_thickness", move || {
                if let Some(pfx) = &mut engine().postfx {
                    pfx.outline_params.thickness[0] = thickness as f32;
                }
        })
        }

        // bloom_profiler_frame_history  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_frame_history() -> *const u8 {
            $crate::ffi::guard("bloom_profiler_frame_history", move || {
                let hist = engine().profiler.frame_history();
                let mut s = String::with_capacity(hist.len() * 24);
                for (cpu, gpu) in &hist {
                    s.push_str(&format!("{:.2}|{:.2}\n", cpu, gpu));
                }
                $crate::string_header::alloc_perry_string(&s)
        })
        }

        // bloom_profiler_overlay_text  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_overlay_text() -> *const u8 {
            $crate::ffi::guard("bloom_profiler_overlay_text", move || {
                let snap = engine().profiler.snapshot();
                let mut s = String::with_capacity(snap.len() * 48);
                for (label, cpu, gpu) in &snap {
                    s.push_str(label);
                    s.push('|');
                    s.push_str(&format!("{:.2}", cpu));
                    s.push('|');
                    match gpu {
                        Some(g) => s.push_str(&format!("{:.2}", g)),
                        None    => s.push_str("-1"),
                    }
                    s.push('\n');
                }
                $crate::string_header::alloc_perry_string(&s)
        })
        }

        // EN-020 — numeric profiler ABI. The packed-text FFIs above stay
        // for back-compat, but Perry 0.5.x's split()/parseFloat() overread
        // their own slice allocations, so per-frame consumers must use
        // these instead: f64s cross the FFI clean and the label is only
        // ever drawn, never parsed. Row/hist indices address the same
        // snapshot the text FFIs serialize; rows are few (≤ MAX_GPU_PAIRS)
        // so the per-call snapshot() is negligible overlay-only cost.

        // bloom_profiler_row_count  [source: windows]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_row_count() -> f64 {
            $crate::ffi::guard("bloom_profiler_row_count", move || {
                engine().profiler.snapshot().len() as f64
        })
        }

        // bloom_profiler_row_label  [source: windows]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_row_label(i: f64) -> *const u8 {
            $crate::ffi::guard("bloom_profiler_row_label", move || {
                let snap = engine().profiler.snapshot();
                match snap.get(i as usize) {
                    Some((label, _, _)) => $crate::string_header::alloc_perry_string(label),
                    None => $crate::string_header::alloc_perry_string(""),
                }
        })
        }

        // bloom_profiler_row_cpu_us  [source: windows]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_row_cpu_us(i: f64) -> f64 {
            $crate::ffi::guard("bloom_profiler_row_cpu_us", move || {
                engine().profiler.snapshot().get(i as usize).map(|r| r.1).unwrap_or(0.0)
        })
        }

        // bloom_profiler_row_gpu_us  [source: windows]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_row_gpu_us(i: f64) -> f64 {
            $crate::ffi::guard("bloom_profiler_row_gpu_us", move || {
                match engine().profiler.snapshot().get(i as usize) {
                    Some((_, _, Some(g))) => *g,
                    _ => -1.0,
                }
        })
        }

        // bloom_profiler_hist_count  [source: windows]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_hist_count() -> f64 {
            $crate::ffi::guard("bloom_profiler_hist_count", move || {
                engine().profiler.frame_history().len() as f64
        })
        }

        // bloom_profiler_hist_cpu_us  [source: windows]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_hist_cpu_us(i: f64) -> f64 {
            $crate::ffi::guard("bloom_profiler_hist_cpu_us", move || {
                engine().profiler.frame_history().get(i as usize).map(|h| h.0).unwrap_or(0.0)
        })
        }

        // bloom_profiler_hist_gpu_us  [source: windows]
        #[no_mangle]
        pub extern "C" fn bloom_profiler_hist_gpu_us(i: f64) -> f64 {
            $crate::ffi::guard("bloom_profiler_hist_gpu_us", move || {
                engine().profiler.frame_history().get(i as usize).map(|h| h.1).unwrap_or(0.0)
        })
        }

    };
}
