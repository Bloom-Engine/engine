//! `define_core_ffi!` — the shared, non-physics `bloom_*` FFI surface.
//!
//! # Why this exists
//!
//! Every platform crate must export the full FFI surface declared in
//! `package.json` (`perry.nativeLibrary.functions`); a missing symbol is a
//! link error or dlopen crash in shipped games. Before this macro the ~250
//! non-physics functions were hand-copied into six platform crates
//! (~9,000 duplicated lines) and drifted constantly:
//!
//!   - Android shipped 60 functions behind (dlopen crash, PR #59), then
//!     "fixed" part of the gap with silent no-op stubs.
//!   - Windows stubbed the entire scene-graph / lighting / picking /
//!     post-FX surface with silent no-ops.
//!   - iOS/tvOS declared gamepad functions with an extra leading
//!     `gamepad` parameter the manifest doesn't have, so every axis and
//!     button read was off by one argument register.
//!   - `bloom_create_mesh` / `bloom_gen_mesh_spline_ribbon` read `*const
//!     f32` on some platforms; Perry passes arrays as pointers to inline
//!     f64 data (see perry-codegen `lower_call/extern_func.rs`), so those
//!     reads were garbage.
//!
//! One macro, expanded once per platform crate, makes per-platform drift
//! in this surface structurally impossible — the same fix the physics
//! surface got with `define_physics_ffi!`. `tools/validate-ffi.js`
//! cross-checks the macro, the platform crates, and package.json in CI.
//!
//! # Contract for the invoking crate
//!
//! The platform crate must define, before invoking the macro:
//!
//! ```ignore
//! /// Engine-state accessor. FFI is single-threaded (Perry calls in on
//! /// the run-loop thread); see the audio module for the one exception.
//! fn engine() -> &'static mut bloom_shared::engine::EngineState { ... }
//!
//! /// Asset-path resolver: identity on desktop; prepends the app asset
//! /// dir on Android/iOS/tvOS where relative paths don't resolve.
//! fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> { ... }
//!
//! bloom_shared::define_core_ffi!();
//! ```
//!
//! Functions NOT generated here (platform crates implement them by hand,
//! validated by tools/validate-ffi.js): window + event loop
//! (`bloom_init_window`, `bloom_begin/end_drawing`, ...), audio backend
//! init/teardown, fullscreen, cursor capture, clipboard, file dialogs,
//! window title/icon, `bloom_get_platform`, `bloom_get_language`.
//!
//! # Conventions inside the macro
//!
//!   - Every body is wrapped in [`crate::ffi::guard`] — panics log once
//!     and return a default instead of crossing the C boundary.
//!   - `models3d` / `image-extras` gated functions compile to a
//!     once-warning stub when the feature is off. Symbols never silently
//!     vanish and never silently no-op.
//!   - String params are Perry StringHeader pointers
//!     (`crate::string_header`); array params are pointers to inline f64
//!     data (the compiler skips the 8-byte ArrayHeader at the callsite).

#[macro_export]
macro_rules! define_core_ffi {
    () => {
        // --- shared FFI-local state ------------------------------------
        // Single-threaded by the same argument as the engine cell: Perry
        // calls all bloom_* functions from the run-loop thread.

        static mut LAST_PICK: Option<$crate::picking::PickResult> = None;
        static mut LAST_PICK_ALL: Vec<$crate::picking::PickResult> = Vec::new();
        static mut LAST_PROJECT: (f64, f64) = (0.0, 0.0);

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

        // bloom_take_screenshot  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_take_screenshot(path_ptr: *const u8) {
            $crate::ffi::guard("bloom_take_screenshot", move || {
                let path = $crate::string_header::str_from_header(path_ptr).to_string();
                let eng = engine();
                eng.renderer.screenshot_requested = true;
                eng.renderer.pending_screenshot_path = Some(path);
        })
        }

        // bloom_clear_background  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_clear_background", move || {
                engine().renderer.set_clear_color(r, g, b, a);
        })
        }

        // bloom_set_env_clear_from_hdr  [source: curated; gated: image-extras]
        #[cfg(feature = "image-extras")]
        #[no_mangle]
        pub extern "C" fn bloom_set_env_clear_from_hdr(path_ptr: *const u8) {
            $crate::ffi::guard("bloom_set_env_clear_from_hdr", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                engine().renderer.set_env_clear_from_hdr_file(path);
        })
        }
        #[cfg(not(feature = "image-extras"))]
        #[no_mangle]
        pub extern "C" fn bloom_set_env_clear_from_hdr(_path_ptr: *const u8) {
            $crate::ffi::feature_off_warn_once("bloom_set_env_clear_from_hdr", "image-extras");
        }

        // bloom_set_target_fps  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_target_fps(fps: f64) {
            $crate::ffi::guard("bloom_set_target_fps", move || {
                engine().target_fps = fps;
        })
        }

        // bloom_set_direct_2d_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_direct_2d_mode(on: f64) {
            $crate::ffi::guard("bloom_set_direct_2d_mode", move || {
                engine().direct_2d_mode = on > 0.5;
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

        // bloom_is_key_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_key_pressed(key: f64) -> f64 {
            $crate::ffi::guard("bloom_is_key_pressed", move || {
                if engine().input.is_key_pressed(key as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_key_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_key_down(key: f64) -> f64 {
            $crate::ffi::guard("bloom_is_key_down", move || {
                if engine().input.is_key_down(key as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_key_released  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_key_released(key: f64) -> f64 {
            $crate::ffi::guard("bloom_is_key_released", move || {
                if engine().input.is_key_released(key as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_mouse_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_x() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_x", move || {
                engine().input.mouse_x
        })
        }

        // bloom_get_mouse_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_y() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_y", move || {
                engine().input.mouse_y
        })
        }

        // bloom_is_mouse_button_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_mouse_button_pressed(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_mouse_button_pressed", move || {
                if engine().input.is_mouse_button_pressed(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_mouse_button_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_mouse_button_down(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_mouse_button_down", move || {
                if engine().input.is_mouse_button_down(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_mouse_button_released  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_mouse_button_released(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_mouse_button_released", move || {
                if engine().input.is_mouse_button_released(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_draw_line  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_line(x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_line", move || {
                engine().renderer.draw_line(x1, y1, x2, y2, thickness, r, g, b, a);
        })
        }

        // bloom_draw_rect  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_rect(x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_rect", move || {
                engine().renderer.draw_rect(x, y, w, h, r, g, b, a);
        })
        }

        // bloom_draw_rect_lines  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_rect_lines(x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_rect_lines", move || {
                engine().renderer.draw_rect_lines(x, y, w, h, thickness, r, g, b, a);
        })
        }

        // bloom_draw_circle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_circle(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_circle", move || {
                engine().renderer.draw_circle(cx, cy, radius, r, g, b, a);
        })
        }

        // bloom_draw_circle_lines  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_circle_lines(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_circle_lines", move || {
                engine().renderer.draw_circle_lines(cx, cy, radius, r, g, b, a);
        })
        }

        // bloom_draw_triangle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_triangle", move || {
                engine().renderer.draw_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a);
        })
        }

        // bloom_draw_poly  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_poly(cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_poly", move || {
                engine().renderer.draw_poly(cx, cy, sides, radius, rotation, r, g, b, a);
        })
        }

        // bloom_draw_text  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_text(text_ptr: *const u8, x: f64, y: f64, size: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_text", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                let eng = engine();
                // Need to split borrow: take text out temporarily
                let mut text_renderer = std::mem::replace(&mut eng.text, $crate::text_renderer::TextRenderer::empty());
                text_renderer.draw_text(&mut eng.renderer, text, x, y, size as u32, r, g, b, a);
                eng.text = text_renderer;
        })
        }

        // bloom_measure_text  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_measure_text(text_ptr: *const u8, size: f64) -> f64 {
            $crate::ffi::guard("bloom_measure_text", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                engine().text.measure_text(text, size as u32)
        })
        }

        // bloom_load_font  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_load_font(path_ptr: *const u8, _size: f64) -> f64 {
            $crate::ffi::guard("bloom_load_font", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => engine().text.load_font(&data) as f64,
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_unload_font  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_unload_font(font_handle: f64) {
            $crate::ffi::guard("bloom_unload_font", move || {
                engine().text.unload_font(font_handle as usize);
        })
        }

        // bloom_draw_text_ex  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_text_ex(font_handle: f64, text_ptr: *const u8, x: f64, y: f64, size: f64, spacing: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_text_ex", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                let eng = engine();
                let mut text_renderer = std::mem::replace(&mut eng.text, $crate::text_renderer::TextRenderer::empty());
                text_renderer.draw_text_ex(&mut eng.renderer, font_handle as usize, text, x, y, size as u32, spacing as f32, r, g, b, a);
                eng.text = text_renderer;
        })
        }

        // bloom_measure_text_ex  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_measure_text_ex(font_handle: f64, text_ptr: *const u8, size: f64, spacing: f64) -> f64 {
            $crate::ffi::guard("bloom_measure_text_ex", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                engine().text.measure_text_ex(font_handle as usize, text, size as u32, spacing as f32)
        })
        }

        // bloom_load_sound  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_load_sound(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_sound", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => match $crate::audio::decode_audio(path, &data) {
                        Some(s) => engine().audio.load_sound(s),
                        None => 0.0,
                    },
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_play_sound  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_play_sound(handle: f64) {
            $crate::ffi::guard("bloom_play_sound", move || {
                engine().audio.play_sound(handle);
        })
        }

        // bloom_stop_sound  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_stop_sound(handle: f64) {
            $crate::ffi::guard("bloom_stop_sound", move || {
                engine().audio.stop_sound(handle);
        })
        }

        // bloom_set_sound_volume  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sound_volume(handle: f64, volume: f64) {
            $crate::ffi::guard("bloom_set_sound_volume", move || {
                engine().audio.set_sound_volume(handle, volume as f32);
        })
        }

        // bloom_set_master_volume  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_master_volume(volume: f64) {
            $crate::ffi::guard("bloom_set_master_volume", move || {
                engine().audio.master_volume = volume as f32;
        })
        }

        // bloom_play_sound_3d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
            $crate::ffi::guard("bloom_play_sound_3d", move || {
                engine().audio.play_sound_3d(handle, x as f32, y as f32, z as f32);
        })
        }

        // bloom_set_listener_position  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
            $crate::ffi::guard("bloom_set_listener_position", move || {
                engine().audio.set_listener_position(x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32);
        })
        }

        // bloom_load_texture  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_load_texture(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_texture", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => {
                        let eng = engine();
                        let $crate::engine::EngineState { ref mut textures, ref mut renderer, .. } = *eng;
                        textures.load_texture(renderer, &data)
                    }
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_unload_texture  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_unload_texture(handle: f64) {
            $crate::ffi::guard("bloom_unload_texture", move || {
                let eng = engine();
                let $crate::engine::EngineState { ref mut textures, ref mut renderer, .. } = *eng;
                textures.unload_texture(handle, renderer);
        })
        }

        // bloom_draw_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_texture(handle: f64, x: f64, y: f64, tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64) {
            $crate::ffi::guard("bloom_draw_texture", move || {
                let eng = engine();
                if let Some(tex) = eng.textures.get(handle) {
                    let bind_group_idx = tex.bind_group_idx;
                    eng.renderer.draw_texture(bind_group_idx, x, y, tint_r, tint_g, tint_b, tint_a);
                }
        })
        }

        // bloom_draw_texture_rec  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_texture_rec(
            handle: f64,
            src_x: f64, src_y: f64, src_w: f64, src_h: f64,
            dst_x: f64, dst_y: f64,
            tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
        ) {
            $crate::ffi::guard("bloom_draw_texture_rec", move || {
                let eng = engine();
                if let Some(tex) = eng.textures.get(handle) {
                    let bind_group_idx = tex.bind_group_idx;
                    eng.renderer.draw_texture_rec(
                        bind_group_idx,
                        src_x, src_y, src_w, src_h,
                        dst_x, dst_y,
                        tint_r, tint_g, tint_b, tint_a,
                    );
                }
        })
        }

        // bloom_draw_texture_pro  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_texture_pro(
            handle: f64,
            src_x: f64, src_y: f64, src_w: f64, src_h: f64,
            dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64,
            origin_x: f64, origin_y: f64, rotation: f64,
            tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
        ) {
            $crate::ffi::guard("bloom_draw_texture_pro", move || {
                let eng = engine();
                if let Some(tex) = eng.textures.get(handle) {
                    let bind_group_idx = tex.bind_group_idx;
                    eng.renderer.draw_texture_pro(
                        bind_group_idx,
                        src_x, src_y, src_w, src_h,
                        dst_x, dst_y, dst_w, dst_h,
                        origin_x, origin_y, rotation,
                        tint_r, tint_g, tint_b, tint_a,
                    );
                }
        })
        }

        // bloom_get_texture_width  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_texture_width(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_texture_width", move || {
                let eng = engine();
                eng.textures.get(handle).map(|t| t.width as f64).unwrap_or(0.0)
        })
        }

        // bloom_get_texture_height  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_texture_height(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_texture_height", move || {
                let eng = engine();
                eng.textures.get(handle).map(|t| t.height as f64).unwrap_or(0.0)
        })
        }

        // bloom_load_image  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_load_image(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_image", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => engine().textures.load_image(&data),
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_image_resize  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_image_resize(handle: f64, w: f64, h: f64) {
            $crate::ffi::guard("bloom_image_resize", move || {
                engine().textures.image_resize(handle, w as u32, h as u32);
        })
        }

        // bloom_image_crop  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_image_crop(handle: f64, x: f64, y: f64, w: f64, h: f64) {
            $crate::ffi::guard("bloom_image_crop", move || {
                engine().textures.image_crop(handle, x as u32, y as u32, w as u32, h as u32);
        })
        }

        // bloom_image_flip_h  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_image_flip_h(handle: f64) {
            $crate::ffi::guard("bloom_image_flip_h", move || {
                engine().textures.image_flip_h(handle);
        })
        }

        // bloom_image_flip_v  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_image_flip_v(handle: f64) {
            $crate::ffi::guard("bloom_image_flip_v", move || {
                engine().textures.image_flip_v(handle);
        })
        }

        // bloom_load_texture_from_image  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_load_texture_from_image(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_load_texture_from_image", move || {
                let eng = engine();
                let $crate::engine::EngineState { ref mut textures, ref mut renderer, .. } = *eng;
                textures.load_texture_from_image(handle, renderer)
        })
        }

        // bloom_gen_texture_mipmaps  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_gen_texture_mipmaps(_handle: f64) {
            $crate::ffi::guard("bloom_gen_texture_mipmaps", move || {
                // Mipmap generation is handled by the GPU texture creation pipeline
                // This is a no-op for now as wgpu handles mipmaps internally
        })
        }

        // bloom_set_texture_filter  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_texture_filter(handle: f64, mode: f64) {
            $crate::ffi::guard("bloom_set_texture_filter", move || {
                let eng = engine();
                if let Some(tex) = eng.textures.get(handle) {
                    let bind_group_idx = tex.bind_group_idx;
                    eng.renderer.set_texture_filter(bind_group_idx, mode > 0.5);
                }
        })
        }

        // bloom_begin_mode_2d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_begin_mode_2d(offset_x: f64, offset_y: f64, target_x: f64, target_y: f64, rotation: f64, zoom: f64) {
            $crate::ffi::guard("bloom_begin_mode_2d", move || {
                engine().renderer.begin_mode_2d(
                    offset_x as f32, offset_y as f32,
                    target_x as f32, target_y as f32,
                    rotation as f32, zoom as f32,
                );
        })
        }

        // bloom_end_mode_2d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_end_mode_2d() {
            $crate::ffi::guard("bloom_end_mode_2d", move || {
                engine().renderer.end_mode_2d();
        })
        }

        // bloom_begin_mode_3d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_begin_mode_3d(
            pos_x: f64, pos_y: f64, pos_z: f64,
            target_x: f64, target_y: f64, target_z: f64,
            up_x: f64, up_y: f64, up_z: f64,
            fovy: f64, projection: f64,
        ) {
            $crate::ffi::guard("bloom_begin_mode_3d", move || {
                engine().renderer.begin_mode_3d(
                    pos_x as f32, pos_y as f32, pos_z as f32,
                    target_x as f32, target_y as f32, target_z as f32,
                    up_x as f32, up_y as f32, up_z as f32,
                    fovy as f32, projection as f32,
                );
        })
        }

        // bloom_end_mode_3d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_end_mode_3d() {
            $crate::ffi::guard("bloom_end_mode_3d", move || {
                engine().renderer.end_mode_3d();
        })
        }

        // bloom_draw_cube  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_cube", move || {
                engine().renderer.draw_cube(x, y, z, w, h, d, r, g, b, a);
        })
        }

        // bloom_draw_cube_wires  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_cube_wires", move || {
                engine().renderer.draw_cube_wires(x, y, z, w, h, d, r, g, b, a);
        })
        }

        // bloom_draw_sphere  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_sphere(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_sphere", move || {
                engine().renderer.draw_sphere(cx, cy, cz, radius, r, g, b, a);
        })
        }

        // bloom_draw_sphere_wires  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_sphere_wires(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_sphere_wires", move || {
                engine().renderer.draw_sphere_wires(cx, cy, cz, radius, r, g, b, a);
        })
        }

        // bloom_draw_cylinder  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_cylinder(x: f64, y: f64, z: f64, radius_top: f64, radius_bottom: f64, height: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_cylinder", move || {
                engine().renderer.draw_cylinder(x, y, z, radius_top, radius_bottom, height, r, g, b, a);
        })
        }

        // bloom_draw_plane  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_plane(cx: f64, cy: f64, cz: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_plane", move || {
                engine().renderer.draw_plane(cx, cy, cz, w, d, r, g, b, a);
        })
        }

        // bloom_draw_grid  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_grid(slices: f64, spacing: f64) {
            $crate::ffi::guard("bloom_draw_grid", move || {
                engine().renderer.draw_grid(slices as i32, spacing);
        })
        }

        // bloom_draw_ray  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_ray(origin_x: f64, origin_y: f64, origin_z: f64, dir_x: f64, dir_y: f64, dir_z: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_ray", move || {
                engine().renderer.draw_ray(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b, a);
        })
        }

        // bloom_load_model  [source: curated; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_load_model(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_model", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => {
                        let eng = engine();
                        let $crate::engine::EngineState { ref mut models, ref mut renderer, .. } = *eng;
                        models.load_model_with_textures(&data, renderer)
                    }
                    Err(_) => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_load_model(_path_ptr: *const u8) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_load_model", "models3d");
            0.0
        }

        // bloom_draw_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_model", move || {
                let eng = engine();
                if let Some(model) = eng.models.get(handle) {
                    let tint = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32];
                    let position = [x as f32, y as f32, z as f32];
                    let handle_bits = handle.to_bits();
                    if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
                        eng.renderer.draw_model_cached(handle_bits, position, scale as f32, tint);
                    } else {
                        for mesh in &model.meshes {
                            let tex_idx = mesh.texture_idx.unwrap_or(0);
                            eng.renderer.draw_model_mesh_tinted(&mesh.vertices, &mesh.indices, position, scale as f32, tint, tex_idx);
                        }
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model(_handle: f64, _x: f64, _y: f64, _z: f64, _scale: f64, _r: f64, _g: f64, _b: f64, _a: f64) {
            $crate::ffi::feature_off_warn_once("bloom_draw_model", "models3d");
        }

        // bloom_draw_model_rotated  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model_rotated(
            handle: f64, x: f64, y: f64, z: f64,
            scale: f64, rot_y: f64,
            color_packed_argb: f64,
        ) {
            $crate::ffi::guard("bloom_draw_model_rotated", move || {
                let bits = color_packed_argb as u32;
                let a = ((bits >> 24) & 0xff) as f32 / 255.0;
                let r = ((bits >> 16) & 0xff) as f32 / 255.0;
                let g = ((bits >>  8) & 0xff) as f32 / 255.0;
                let b = ( bits        & 0xff) as f32 / 255.0;
                let eng = engine();
                if let Some(model) = eng.models.get(handle) {
                    let position = [x as f32, y as f32, z as f32];
                    let scale = scale as f32;
                    let tint = [r, g, b, a];
                    for mesh in &model.meshes {
                        let tex_idx = mesh.texture_idx.unwrap_or(0);
                        eng.renderer.draw_model_mesh_tinted_rotated(
                            &mesh.vertices, &mesh.indices, position, scale, tint, tex_idx, rot_y as f32,
                        );
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model_rotated(_handle: f64, _x: f64, _y: f64, _z: f64, _scale: f64, _rot_y: f64, _color_packed_argb: f64) {
            $crate::ffi::feature_off_warn_once("bloom_draw_model_rotated", "models3d");
        }

        // bloom_unload_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_unload_model(handle: f64) {
            $crate::ffi::guard("bloom_unload_model", move || {     engine().models.unload_model(handle); })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_unload_model(_handle: f64) {
            $crate::ffi::feature_off_warn_once("bloom_unload_model", "models3d");
        }

        // bloom_gen_mesh_cube  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_cube(w: f64, h: f64, d: f64) -> f64 {
            $crate::ffi::guard("bloom_gen_mesh_cube", move || {
                engine().models.gen_mesh_cube(w as f32, h as f32, d as f32)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_cube(_w: f64, _h: f64, _d: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_gen_mesh_cube", "models3d");
            0.0
        }

        // bloom_gen_mesh_heightmap  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_heightmap(image_handle: f64, size_x: f64, size_y: f64, size_z: f64) -> f64 {
            $crate::ffi::guard("bloom_gen_mesh_heightmap", move || {
                let eng = engine();
                if let Some(img) = eng.textures.images.get(image_handle) {
                    let data = img.data.clone();
                    let w = img.width;
                    let h = img.height;
                    eng.models.gen_mesh_heightmap(&data, w, h, size_x as f32, size_y as f32, size_z as f32)
                } else {
                    0.0
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_heightmap(_image_handle: f64, _size_x: f64, _size_y: f64, _size_z: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_gen_mesh_heightmap", "models3d");
            0.0
        }

        // bloom_load_shader  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_load_shader(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_shader", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                engine().renderer.load_custom_shader(source) as f64
        })
        }

        // bloom_compile_material  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material(source) {
                    Ok(handle) => handle as f64,
                    Err(e) => {
                        eprintln!("[material] compile failed: {:?}", e);
                        0.0
                    }
                }
        })
        }

        // bloom_compile_material_refractive  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_refractive(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_refractive", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Translucent, Bucket::Refractive, true, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[refractive] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_transparent  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_transparent(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_transparent", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Translucent, Bucket::Transparent, false, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_additive  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_additive(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_additive", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Translucent, Bucket::Additive, false, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_cutout  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_cutout(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_cutout", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Opaque, Bucket::Cutout, false, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_instanced  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_instanced(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_instanced", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_instanced(source) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] instanced compile failed: {:?}", e); 0.0 }
                }
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

        // bloom_submit_material_draw_instanced  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_submit_material_draw_instanced(
            material: f64, mesh_handle: f64, mesh_idx: f64,
            instance_buffer: f64, instance_count: f64,
        ) {
            $crate::ffi::guard("bloom_submit_material_draw_instanced", move || {
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
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_submit_material_draw_instanced(_material: f64, _mesh_handle: f64, _mesh_idx: f64, _instance_buffer: f64, _instance_count: f64) {
            $crate::ffi::feature_off_warn_once("bloom_submit_material_draw_instanced", "models3d");
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

        // bloom_set_material_reflection_probe  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_reflection_probe(
            material: f64, probe: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_reflection_probe", move || {
                engine().renderer.set_material_reflection_probe(material as u32, probe as u32);
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

        // bloom_set_material_texture_array  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_texture_array(
            material: f64, slot: f64, array: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_texture_array", move || {
                engine().renderer.set_material_texture_array(
                    material as u32, slot as u32, array as u32,
                );
        })
        }

        // bloom_set_material_shading_model  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_shading_model(
            material: f64, model: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_shading_model", move || {
                engine().renderer.set_material_shading_model(material as u32, model as u32);
        })
        }

        // bloom_set_material_foliage  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_foliage(
            material: f64,
            trans_r: f64, trans_g: f64, trans_b: f64,
            trans_amount: f64, wrap_factor: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_foliage", move || {
                engine().renderer.set_material_foliage(
                    material as u32,
                    [trans_r as f32, trans_g as f32, trans_b as f32],
                    trans_amount as f32, wrap_factor as f32,
                );
        })
        }

        // bloom_set_post_pass  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_post_pass(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_set_post_pass", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.set_post_pass(source) {
                    Ok(()) => 1.0,
                    Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
                }
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

        // bloom_draw_material  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_draw_material(
            material: f64,
            mesh_handle: f64,
            mesh_idx: f64,
            x: f64, y: f64, z: f64, scale: f64,
            r: f64, g: f64, b: f64, a: f64,
        ) {
            $crate::ffi::guard("bloom_draw_material", move || {
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
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_draw_material(_material: f64, _mesh_handle: f64, _mesh_idx: f64, _x: f64, _y: f64, _z: f64, _scale: f64, _r: f64, _g: f64, _b: f64, _a: f64) {
            $crate::ffi::feature_off_warn_once("bloom_draw_material", "models3d");
        }

        // bloom_load_model_animation  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_model_animation", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => engine().models.load_model_animation(&data),
                    Err(_) => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_load_model_animation(_path_ptr: *const u8) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_load_model_animation", "models3d");
            0.0
        }

        // bloom_update_model_animation  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64, scale: f64, px: f64, py: f64, pz: f64, rot_y: f64) {
            $crate::ffi::guard("bloom_update_model_animation", move || {
                // Take a single Y-axis angle (radians) instead of sin/cos, so the
                // engine reconstructs both with full precision + correct signs.
                // Older callers that passed (rot_sin, rot_cos) hit a Perry-ARM64
                // 9th-arg garbling bug AND a sqrt(1-sin²) workaround that lost
                // the sign of cos — model rotation was correct only on half the
                // circle. 8-arg signature dodges both issues. Matches macOS.
                let rot_y_f = rot_y as f32;
                let rot_sin = rot_y_f.sin();
                let rot_cos = rot_y_f.cos();
                let eng = engine();
                eng.models.update_model_animation(handle, anim_index as usize, time as f32);
                if let Some(anim) = eng.models.get_animation(handle) {
                    if !anim.joint_matrices.is_empty() {
                        eng.renderer.set_joint_matrices_scaled(&anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32], rot_sin, rot_cos);
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_update_model_animation(_handle: f64, _anim_index: f64, _time: f64, _scale: f64, _px: f64, _py: f64, _pz: f64, _rot_y: f64) {
            $crate::ffi::feature_off_warn_once("bloom_update_model_animation", "models3d");
        }

        // bloom_create_mesh  [source: curated; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_create_mesh(vertex_ptr: *const f64, vertex_count: f64, index_ptr: *const f64, index_count: f64) -> f64 {
            $crate::ffi::guard("bloom_create_mesh", move || {
                // Perry arrays are pointers to inline f64 data (8-byte ArrayHeader
                // skipped by the compiler before the call) — see perry-codegen
                // lower_call/extern_func.rs. Reading f32/u32 here was a latent
                // misread on windows/android/ios/tvos before the FFI unification.
                if vertex_ptr.is_null() || index_ptr.is_null() { return 0.0; }
                let vcount = vertex_count as usize;
                let icount = index_count as usize;
                let vertex_f64 = unsafe { std::slice::from_raw_parts(vertex_ptr, vcount * 12) };
                let index_f64 = unsafe { std::slice::from_raw_parts(index_ptr, icount) };
                let vertices: Vec<f32> = vertex_f64.iter().map(|&v| v as f32).collect();
                let indices: Vec<u32> = index_f64.iter().map(|&v| v as u32).collect();
                engine().models.create_mesh(&vertices, &indices)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_create_mesh(_vertex_ptr: *const f64, _vertex_count: f64, _index_ptr: *const f64, _index_count: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_create_mesh", "models3d");
            0.0
        }

        // bloom_set_joint_test  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_joint_test(joint_index: f64, angle: f64) {
            $crate::ffi::guard("bloom_set_joint_test", move || {
                engine().renderer.set_joint_test(joint_index as usize, angle as f32);
        })
        }

        // bloom_set_ambient_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ambient_light(r: f64, g: f64, b: f64, intensity: f64) {
            $crate::ffi::guard("bloom_set_ambient_light", move || {
                engine().renderer.set_ambient_light(r, g, b, intensity);
        })
        }

        // bloom_set_directional_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_directional_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
            $crate::ffi::guard("bloom_set_directional_light", move || {
                engine().renderer.set_directional_light(dx, dy, dz, r, g, b, intensity);
        })
        }

        // bloom_set_procedural_sky  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_procedural_sky(enabled: f64, rayleigh_density: f64, mie_density: f64, ground_albedo: f64) {
            $crate::ffi::guard("bloom_set_procedural_sky", move || {
                engine().renderer.set_procedural_sky(
                    enabled != 0.0,
                    rayleigh_density as f32,
                    mie_density as f32,
                    ground_albedo as f32,
                );
        })
        }

        // bloom_set_sun_direction  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sun_direction(dx: f64, dy: f64, dz: f64, intensity: f64) {
            $crate::ffi::guard("bloom_set_sun_direction", move || {
                engine().renderer.set_sun_direction(dx as f32, dy as f32, dz as f32, intensity as f32);
        })
        }

        // bloom_set_fog  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_fog(r: f64, g: f64, b: f64, density: f64, height_ref: f64, height_falloff: f64) {
            $crate::ffi::guard("bloom_set_fog", move || {
                let r_ = engine();
                r_.renderer.set_fog_color(r as f32, g as f32, b as f32);
                r_.renderer.set_fog_density(density as f32);
                r_.renderer.set_fog_height_falloff(height_ref as f32, height_falloff as f32);
        })
        }

        // bloom_set_chromatic_aberration  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_chromatic_aberration(strength: f64) {
            $crate::ffi::guard("bloom_set_chromatic_aberration", move || {
                engine().renderer.set_chromatic_aberration(strength as f32);
        })
        }

        // bloom_set_vignette  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_vignette(strength: f64, softness: f64) {
            $crate::ffi::guard("bloom_set_vignette", move || {
                engine().renderer.set_vignette(strength as f32, softness as f32);
        })
        }

        // bloom_set_film_grain  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_film_grain(strength: f64) {
            $crate::ffi::guard("bloom_set_film_grain", move || {
                engine().renderer.set_film_grain(strength as f32);
        })
        }

        // bloom_set_sun_shafts  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) {
            $crate::ffi::guard("bloom_set_sun_shafts", move || {
                let eng = engine();
                eng.renderer.set_sun_shaft_strength(strength as f32);
                eng.renderer.set_sun_shaft_decay(decay as f32);
                eng.renderer.set_sun_shaft_color(r as f32, g as f32, b as f32);
        })
        }

        // bloom_set_auto_exposure  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_auto_exposure(on: f64) {
            $crate::ffi::guard("bloom_set_auto_exposure", move || {
                engine().renderer.set_auto_exposure(on != 0.0);
        })
        }

        // bloom_set_taa_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_taa_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_taa_enabled", move || {
                engine().renderer.set_taa_enabled(on != 0.0);
        })
        }

        // bloom_set_render_scale  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_render_scale(scale: f64) {
            $crate::ffi::guard("bloom_set_render_scale", move || {
                engine().renderer.set_render_scale(scale as f32);
        })
        }

        // bloom_get_render_scale  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_render_scale() -> f64 {
            $crate::ffi::guard("bloom_get_render_scale", move || {
                engine().renderer.render_scale() as f64
        })
        }

        // bloom_set_upscale_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_upscale_mode(mode: f64) {
            $crate::ffi::guard("bloom_set_upscale_mode", move || {
                engine().renderer.set_upscale_mode(mode as u32);
        })
        }

        // bloom_set_cas_strength  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_cas_strength(strength: f64) {
            $crate::ffi::guard("bloom_set_cas_strength", move || {
                engine().renderer.set_cas_strength(strength as f32);
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

        // bloom_set_auto_resolution  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_auto_resolution(target_hz: f64, enabled: f64) {
            $crate::ffi::guard("bloom_set_auto_resolution", move || {
                let eng = engine();
                if enabled != 0.0 {
                    let current = eng.renderer.render_scale();
                    eng.drs.enable(target_hz as f32, current);
                } else {
                    eng.drs.disable();
                }
        })
        }

        // bloom_set_manual_exposure  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_manual_exposure(value: f64) {
            $crate::ffi::guard("bloom_set_manual_exposure", move || {
                engine().renderer.set_manual_exposure(value as f32);
        })
        }

        // bloom_set_env_intensity  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_env_intensity(intensity: f64) {
            $crate::ffi::guard("bloom_set_env_intensity", move || {
                engine().renderer.set_env_intensity(intensity as f32);
        })
        }

        // bloom_set_ssgi_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssgi_enabled(enabled: f64) {
            $crate::ffi::guard("bloom_set_ssgi_enabled", move || {
                engine().renderer.set_ssgi_enabled(enabled != 0.0);
        })
        }

        // bloom_set_ssgi_intensity  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssgi_intensity(intensity: f64) {
            $crate::ffi::guard("bloom_set_ssgi_intensity", move || {
                engine().renderer.set_ssgi_intensity(intensity as f32);
        })
        }

        // bloom_set_ssgi_radius  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssgi_radius(radius: f64) {
            $crate::ffi::guard("bloom_set_ssgi_radius", move || {
                engine().renderer.set_ssgi_radius(radius as f32);
        })
        }

        // bloom_set_dof  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_dof(enabled: f64, focus_distance: f64, aperture: f64) {
            $crate::ffi::guard("bloom_set_dof", move || {
                let r = &mut engine().renderer;
                r.set_dof_enabled(enabled != 0.0);
                r.set_dof_focus_distance(focus_distance as f32);
                r.set_dof_aperture(aperture as f32);
        })
        }

        // bloom_set_quality_preset  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_quality_preset(preset: f64) {
            $crate::ffi::guard("bloom_set_quality_preset", move || {
                engine().renderer.apply_quality_preset(preset as u32);
        })
        }

        // bloom_set_shadows_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_shadows_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_shadows_enabled", move || {
                engine().renderer.set_shadows_enabled(on != 0.0);
        })
        }

        // bloom_set_shadows_always_fresh  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_shadows_always_fresh(on: f64) {
            $crate::ffi::guard("bloom_set_shadows_always_fresh", move || {
                engine().renderer.set_shadows_always_fresh(on != 0.0);
        })
        }

        // bloom_set_bloom_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_bloom_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_bloom_enabled", move || {
                engine().renderer.set_bloom_enabled(on != 0.0);
        })
        }

        // bloom_set_ssao_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssao_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_ssao_enabled", move || {
                engine().renderer.set_ssao_enabled(on != 0.0);
        })
        }

        // bloom_set_ssao_intensity  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssao_intensity(value: f64) {
            $crate::ffi::guard("bloom_set_ssao_intensity", move || {
                engine().renderer.set_ssao_strength(value as f32);
        })
        }

        // bloom_set_ssao_radius  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssao_radius(world_radius: f64) {
            $crate::ffi::guard("bloom_set_ssao_radius", move || {
                engine().renderer.set_ssao_radius(world_radius as f32);
        })
        }

        // bloom_set_wind  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_wind(dir_x: f64, dir_z: f64, amplitude: f64, frequency: f64) {
            $crate::ffi::guard("bloom_set_wind", move || {
                engine().renderer.set_wind(dir_x as f32, dir_z as f32, amplitude as f32, frequency as f32);
        })
        }

        // bloom_set_ssr_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssr_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_ssr_enabled", move || {
                engine().renderer.set_ssr_enabled(on != 0.0);
        })
        }

        // bloom_set_motion_blur_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_motion_blur_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_motion_blur_enabled", move || {
                engine().renderer.set_motion_blur_enabled(on != 0.0);
        })
        }

        // bloom_set_sss_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sss_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_sss_enabled", move || {
                engine().renderer.set_sss_enabled(on != 0.0);
        })
        }

        // bloom_set_profiler_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_profiler_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_profiler_enabled", move || {
                engine().profiler.set_enabled(on != 0.0);
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

        // bloom_get_model_mesh_count  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_mesh_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_mesh_count", move || {
                match engine().models.get(handle) {
                    Some(model) => model.meshes.len() as f64,
                    None => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_mesh_count(_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_mesh_count", "models3d");
            0.0
        }

        // bloom_get_model_material_count  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_material_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_material_count", move || {
                match engine().models.get(handle) {
                    Some(model) => model.meshes.len() as f64,
                    None => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_material_count(_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_material_count", "models3d");
            0.0
        }

        // bloom_inject_key_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_inject_key_down(key: f64) {
            $crate::ffi::guard("bloom_inject_key_down", move || {
                engine().input.set_key_down(key as usize);
        })
        }

        // bloom_inject_key_up  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_inject_key_up(key: f64) {
            $crate::ffi::guard("bloom_inject_key_up", move || {
                engine().input.set_key_up(key as usize);
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

        // bloom_get_crown_rotation  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_crown_rotation() -> f64 {
            $crate::ffi::guard("bloom_get_crown_rotation", move || {
                engine().input.consume_crown_rotation()
        })
        }

        // bloom_load_music  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_load_music(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_music", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => match $crate::audio::decode_audio(path, &data) {
                        Some(s) => engine().audio.load_music(s),
                        None => 0.0,
                    },
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_play_music  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_play_music(handle: f64) {
            $crate::ffi::guard("bloom_play_music", move || {
                engine().audio.play_music(handle);
        })
        }

        // bloom_stop_music  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_stop_music(handle: f64) {
            $crate::ffi::guard("bloom_stop_music", move || {
                engine().audio.stop_music(handle);
        })
        }

        // bloom_update_music_stream  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_update_music_stream(handle: f64) {
            $crate::ffi::guard("bloom_update_music_stream", move || {
                engine().audio.update_music_stream(handle);
        })
        }

        // bloom_set_music_volume  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_music_volume(handle: f64, volume: f64) {
            $crate::ffi::guard("bloom_set_music_volume", move || {
                engine().audio.set_music_volume(handle, volume as f32);
        })
        }

        // bloom_is_music_playing  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_music_playing(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_is_music_playing", move || {
                if engine().audio.is_music_playing(handle) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_gamepad_available  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_available() -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_available", move || {
                if engine().input.is_gamepad_available() { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_gamepad_axis  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_gamepad_axis(axis: f64) -> f64 {
            $crate::ffi::guard("bloom_get_gamepad_axis", move || {
                engine().input.get_gamepad_axis(axis as usize) as f64
        })
        }

        // bloom_is_gamepad_button_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_button_pressed(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_button_pressed", move || {
                if engine().input.is_gamepad_button_pressed(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_gamepad_button_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_button_down(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_button_down", move || {
                if engine().input.is_gamepad_button_down(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_gamepad_button_released  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_button_released(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_button_released", move || {
                if engine().input.is_gamepad_button_released(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_gamepad_axis_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_gamepad_axis_count() -> f64 {
            $crate::ffi::guard("bloom_get_gamepad_axis_count", move || {
                engine().input.get_gamepad_axis_count() as f64
        })
        }

        // bloom_get_mouse_delta_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_delta_x() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_delta_x", move || {
                engine().input.mouse_delta_x
        })
        }

        // bloom_get_mouse_delta_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_delta_y() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_delta_y", move || {
                engine().input.mouse_delta_y
        })
        }

        // bloom_get_mouse_wheel  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_wheel() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_wheel", move || {
                engine().input.consume_mouse_wheel()
        })
        }

        // bloom_get_char_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_char_pressed() -> f64 {
            $crate::ffi::guard("bloom_get_char_pressed", move || {
                engine().input.pop_char() as f64
        })
        }

        // bloom_get_model_bounds_min_x  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_x(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_min_x", move || {
                engine().models.get_bounds(model_handle).0[0] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_x(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_min_x", "models3d");
            0.0
        }

        // bloom_get_model_bounds_min_y  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_y(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_min_y", move || {
                engine().models.get_bounds(model_handle).0[1] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_y(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_min_y", "models3d");
            0.0
        }

        // bloom_get_model_bounds_min_z  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_z(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_min_z", move || {
                engine().models.get_bounds(model_handle).0[2] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_z(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_min_z", "models3d");
            0.0
        }

        // bloom_get_model_bounds_max_x  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_x(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_max_x", move || {
                engine().models.get_bounds(model_handle).1[0] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_x(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_max_x", "models3d");
            0.0
        }

        // bloom_get_model_bounds_max_y  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_y(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_max_y", move || {
                engine().models.get_bounds(model_handle).1[1] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_y(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_max_y", "models3d");
            0.0
        }

        // bloom_get_model_bounds_max_z  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_z(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_max_z", move || {
                engine().models.get_bounds(model_handle).1[2] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_z(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_max_z", "models3d");
            0.0
        }

        // bloom_write_file  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_write_file(path_ptr: *const u8, data_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_write_file", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                let data = $crate::string_header::str_from_header(data_ptr);
                match std::fs::write(path, data.as_bytes()) {
                    Ok(_) => 1.0,
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_file_exists  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_file_exists(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_file_exists", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                if std::path::Path::new(path).exists() { 1.0 } else { 0.0 }
        })
        }

        // bloom_read_file  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_read_file(path_ptr: *const u8) -> *const u8 {
            $crate::ffi::guard("bloom_read_file", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read_to_string(path) {
                    Ok(contents) => $crate::string_header::alloc_perry_string(&contents),
                    Err(_) => $crate::string_header::alloc_perry_string(""),
                }
        })
        }

        // bloom_get_touch_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_touch_x(index: f64) -> f64 {
            $crate::ffi::guard("bloom_get_touch_x", move || {
                engine().input.get_touch_x(index as usize)
        })
        }

        // bloom_get_touch_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_touch_y(index: f64) -> f64 {
            $crate::ffi::guard("bloom_get_touch_y", move || {
                engine().input.get_touch_y(index as usize)
        })
        }

        // bloom_get_touch_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_touch_count() -> f64 {
            $crate::ffi::guard("bloom_get_touch_count", move || {
                engine().input.get_touch_count() as f64
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

        // bloom_add_directional_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_add_directional_light(
            dx: f64, dy: f64, dz: f64,
            r: f64, g: f64, b: f64,
            intensity: f64,
        ) {
            $crate::ffi::guard("bloom_add_directional_light", move || {
                engine().renderer.add_directional_light(
                    dx as f32, dy as f32, dz as f32,
                    r as f32, g as f32, b as f32,
                    intensity as f32,
                );
        })
        }

        // bloom_add_point_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_add_point_light(
            x: f64, y: f64, z: f64, range: f64,
            r: f64, g: f64, b: f64,
            intensity: f64,
        ) {
            $crate::ffi::guard("bloom_add_point_light", move || {
                engine().renderer.add_point_light(
                    x as f32, y as f32, z as f32, range as f32,
                    r as f32, g as f32, b as f32,
                    intensity as f32,
                );
        })
        }

        // bloom_scene_create_node  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_create_node() -> f64 {
            $crate::ffi::guard("bloom_scene_create_node", move || {
                engine().scene.create_node()
        })
        }

        // bloom_scene_destroy_node  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_destroy_node(handle: f64) {
            $crate::ffi::guard("bloom_scene_destroy_node", move || {
                engine().scene.destroy_node(handle);
        })
        }

        // bloom_scene_set_visible  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_visible(handle: f64, visible: f64) {
            $crate::ffi::guard("bloom_scene_set_visible", move || {
                engine().scene.set_visible(handle, visible != 0.0);
        })
        }

        // bloom_scene_set_cast_shadow  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_cast_shadow(handle: f64, cast: f64) {
            $crate::ffi::guard("bloom_scene_set_cast_shadow", move || {
                engine().scene.set_cast_shadow(handle, cast != 0.0);
        })
        }

        // bloom_scene_set_receive_shadow  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_receive_shadow(handle: f64, receive: f64) {
            $crate::ffi::guard("bloom_scene_set_receive_shadow", move || {
                engine().scene.set_receive_shadow(handle, receive != 0.0);
        })
        }

        // bloom_scene_set_parent  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_parent(handle: f64, parent: f64) {
            $crate::ffi::guard("bloom_scene_set_parent", move || {
                engine().scene.set_parent(handle, parent);
        })
        }

        // bloom_scene_set_transform  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_transform(handle: f64, mat_ptr: *const f64) {
            $crate::ffi::guard("bloom_scene_set_transform", move || {
                if mat_ptr.is_null() { return; }
                let slice = unsafe { std::slice::from_raw_parts(mat_ptr, 16) };
                let mut mat = [[0.0f32; 4]; 4];
                for col in 0..4 {
                    for row in 0..4 {
                        mat[col][row] = slice[col * 4 + row] as f32;
                    }
                }
                engine().scene.set_transform(handle, mat);
        })
        }

        // bloom_scene_update_geometry  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_update_geometry(
            handle: f64,
            vert_ptr: *const f64,
            vert_count: f64,
            idx_ptr: *const f64,
            idx_count: f64,
        ) {
            $crate::ffi::guard("bloom_scene_update_geometry", move || {
                if vert_ptr.is_null() || idx_ptr.is_null() { return; }
                let nv = vert_count as usize;
                let ni = idx_count as usize;

                let vert_floats = unsafe { std::slice::from_raw_parts(vert_ptr, nv * 12) };
                let idx_floats = unsafe { std::slice::from_raw_parts(idx_ptr, ni) };

                let mut vertices = Vec::with_capacity(nv);
                for i in 0..nv {
                    let base = i * 12;
                    vertices.push($crate::renderer::Vertex3D {
                        position: [vert_floats[base] as f32, vert_floats[base+1] as f32, vert_floats[base+2] as f32],
                        normal: [vert_floats[base+3] as f32, vert_floats[base+4] as f32, vert_floats[base+5] as f32],
                        color: [vert_floats[base+6] as f32, vert_floats[base+7] as f32, vert_floats[base+8] as f32, vert_floats[base+9] as f32],
                        uv: [vert_floats[base+10] as f32, vert_floats[base+11] as f32],
                        joints: [0.0; 4],
                        weights: [0.0; 4],
                        tangent: [0.0; 4],
                    });
                }

                let indices: Vec<u32> = idx_floats.iter().map(|&v| v as u32).collect();

                engine().scene.update_geometry(handle, vertices, indices);
        })
        }

        // bloom_scene_set_material_color  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_color(handle: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_scene_set_material_color", move || {
                engine().scene.set_material_color(handle, r as f32, g as f32, b as f32, a as f32);
        })
        }

        // bloom_scene_set_material_pbr  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_pbr(handle: f64, roughness: f64, metalness: f64) {
            $crate::ffi::guard("bloom_scene_set_material_pbr", move || {
                engine().scene.set_material_pbr(handle, roughness as f32, metalness as f32);
        })
        }

        // bloom_scene_set_material_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_texture(handle: f64, texture_idx: f64) {
            $crate::ffi::guard("bloom_scene_set_material_texture", move || {
                engine().scene.set_material_texture(handle, texture_idx as u32);
        })
        }

        // bloom_scene_node_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_node_count() -> f64 {
            $crate::ffi::guard("bloom_scene_node_count", move || {
                engine().scene.node_count() as f64
        })
        }

        // bloom_scene_node_vertex_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_node_vertex_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_node_vertex_count", move || {
                match engine().scene.nodes.get(handle) {
                    Some(node) => node.vertices.len() as f64,
                    None => -1.0,
                }
        })
        }

        // bloom_scene_node_index_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_node_index_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_node_index_count", move || {
                match engine().scene.nodes.get(handle) {
                    Some(node) => node.indices.len() as f64,
                    None => -1.0,
                }
        })
        }

        // bloom_set_cursor_shape  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_cursor_shape(shape: f64) {
            $crate::ffi::guard("bloom_set_cursor_shape", move || {
                engine().input.cursor_shape = shape as u32;
        })
        }

        // bloom_scene_pick_all  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_pick_all(screen_x: f64, screen_y: f64, max_results: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_pick_all", move || {
                let eng = engine();
                let inv_vp = eng.renderer.inverse_vp_matrix();
                let cam_pos = eng.renderer.camera_pos();
                let w = eng.renderer.width() as f32;
                let h = eng.renderer.height() as f32;
                let (origin, direction) = $crate::picking::screen_to_ray(
                    screen_x as f32, screen_y as f32, w, h, &inv_vp, &cam_pos,
                );
                let results = $crate::picking::raycast_scene_all(&eng.scene, &origin, &direction, max_results as usize);
                let count = results.len();
                unsafe { LAST_PICK_ALL = results; }
                count as f64
        })
        }

        // bloom_pick_all_handle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_all_handle(index: f64) -> f64 {
            $crate::ffi::guard("bloom_pick_all_handle", move || {
                let i = index as usize;
                unsafe { LAST_PICK_ALL.get(i).map(|r| r.handle).unwrap_or(0.0) }
        })
        }

        // bloom_pick_all_distance  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_all_distance(index: f64) -> f64 {
            $crate::ffi::guard("bloom_pick_all_distance", move || {
                let i = index as usize;
                unsafe { LAST_PICK_ALL.get(i).map(|r| r.distance as f64).unwrap_or(0.0) }
        })
        }

        // bloom_scene_set_material_water  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_water(handle: f64, wave_amp: f64, wave_speed: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_scene_set_material_water", move || {
                engine().scene.set_material_water(handle, wave_amp as f32, wave_speed as f32, r as f32, g as f32, b as f32, a as f32);
        })
        }

        // bloom_gen_mesh_spline_ribbon  [source: curated; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_spline_ribbon(points_ptr: *const f64, point_count: f64, widths_ptr: *const f64, width_count: f64) -> f64 {
            $crate::ffi::guard("bloom_gen_mesh_spline_ribbon", move || {
                // Perry arrays are f64 buffers (see bloom_create_mesh note).
                if points_ptr.is_null() || widths_ptr.is_null() { return 0.0; }
                let n = point_count as usize;
                let wn = width_count as usize;
                let points_f64 = unsafe { std::slice::from_raw_parts(points_ptr, n * 3) };
                let widths_f64 = unsafe { std::slice::from_raw_parts(widths_ptr, wn) };
                let points: Vec<f32> = points_f64.iter().map(|&v| v as f32).collect();
                let widths: Vec<f32> = widths_f64.iter().map(|&v| v as f32).collect();
                engine().models.gen_mesh_spline_ribbon(&points, &widths)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_spline_ribbon(_points_ptr: *const f64, _point_count: f64, _widths_ptr: *const f64, _width_count: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_gen_mesh_spline_ribbon", "models3d");
            0.0
        }

        // bloom_load_render_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_load_render_texture(width: f64, height: f64) -> f64 {
            $crate::ffi::guard("bloom_load_render_texture", move || {
                let w = width as u32;
                let h = height as u32;
                let eng = engine();
                let rt_handle = eng.textures.load_render_texture(w, h);

                // Create the GPU texture via the renderer's public method.
                let (bind_group_idx, _tex_vec_idx) = eng.renderer.create_render_texture(w, h);

                // Register as a texture handle so drawTexture can sample it.
                let tex_handle = eng.textures.textures.alloc($crate::textures::TextureData {
                    bind_group_idx, width: w, height: h,
                });
                eng.textures.set_render_texture_handle(rt_handle, tex_handle);

                rt_handle
        })
        }

        // bloom_unload_render_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_unload_render_texture(handle: f64) {
            $crate::ffi::guard("bloom_unload_render_texture", move || {
                engine().textures.unload_render_texture(handle);
        })
        }

        // bloom_begin_texture_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_begin_texture_mode(handle: f64) {
            $crate::ffi::guard("bloom_begin_texture_mode", move || {
                let eng = engine();
                let (w, h, bg_idx) = match eng.textures.render_textures.get(handle) {
                    Some(rt) => {
                        let tex_handle = rt.texture_handle;
                        match eng.textures.textures.get(tex_handle) {
                            Some(td) => (rt.width, rt.height, td.bind_group_idx as usize),
                            None => return,
                        }
                    }
                    None => return,
                };
                if let Some(texture) = eng.renderer.get_texture_ref(bg_idx) {
                    // We need to call begin_texture_mode with a reference to the texture,
                    // but get_texture_ref borrows renderer immutably. Clone the texture view
                    // data we need first, then call the mutable method.
                    let color_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    // Create depth texture for this RT.
                    let depth_tex = eng.renderer.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("rt_depth"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    });
                    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    eng.renderer.rt_color_view = Some(color_view);
                    eng.renderer.rt_depth_view = Some(depth_view);
                    eng.renderer.rt_depth_texture = Some(depth_tex);
                    eng.renderer.rt_width = w;
                    eng.renderer.rt_height = h;
                }
        })
        }

        // bloom_end_texture_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_end_texture_mode() {
            $crate::ffi::guard("bloom_end_texture_mode", move || {
                engine().renderer.end_texture_mode();
        })
        }

        // bloom_get_render_texture_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_render_texture_texture(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_render_texture_texture", move || {
                engine().textures.get_render_texture_texture(handle)
        })
        }

        // bloom_scene_get_transform  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_transform(handle: f64, index: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_transform", move || {
                let mat = engine().scene.get_transform(handle);
                let i = index as usize;
                let col = i / 4;
                let row = i % 4;
                if col < 4 && row < 4 { mat[col][row] as f64 } else { 0.0 }
        })
        }

        // bloom_scene_get_bounds_min_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_min_x(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_min_x", move || {     engine().scene.get_bounds(handle).0[0] as f64 })
        }

        // bloom_scene_get_bounds_min_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_min_y(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_min_y", move || {     engine().scene.get_bounds(handle).0[1] as f64 })
        }

        // bloom_scene_get_bounds_min_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_min_z(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_min_z", move || {     engine().scene.get_bounds(handle).0[2] as f64 })
        }

        // bloom_scene_get_bounds_max_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_max_x(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_max_x", move || {     engine().scene.get_bounds(handle).1[0] as f64 })
        }

        // bloom_scene_get_bounds_max_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_max_y(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_max_y", move || {     engine().scene.get_bounds(handle).1[1] as f64 })
        }

        // bloom_scene_get_bounds_max_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_max_z(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_max_z", move || {     engine().scene.get_bounds(handle).1[2] as f64 })
        }

        // bloom_scene_set_user_data  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_user_data(handle: f64, data: f64) {
            $crate::ffi::guard("bloom_scene_set_user_data", move || {
                engine().scene.set_user_data(handle, data as i64);
        })
        }

        // bloom_scene_get_user_data  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_user_data(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_user_data", move || {
                engine().scene.get_user_data(handle) as f64
        })
        }

        // bloom_scene_extrude_polygon  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_extrude_polygon(
            handle: f64,
            polygon_ptr: *const f64,
            polygon_count: f64,
            depth: f64,
        ) {
            $crate::ffi::guard("bloom_scene_extrude_polygon", move || {
                if polygon_ptr.is_null() { return; }
                let n = polygon_count as usize;
                let polygon = unsafe { std::slice::from_raw_parts(polygon_ptr, n * 2) };

                let geo = $crate::geometry::extrude_polygon(polygon, &[], depth);
                engine().scene.update_geometry(handle, geo.vertices, geo.indices);
        })
        }

        // bloom_scene_subtract_box  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_subtract_box(
            handle: f64,
            min_x: f64, min_y: f64, min_z: f64,
            max_x: f64, max_y: f64, max_z: f64,
        ) {
            $crate::ffi::guard("bloom_scene_subtract_box", move || {
                let eng = engine();
                if let Some(node) = eng.scene.nodes.get(handle) {
                    let current = $crate::geometry::GeometryData {
                        vertices: node.vertices.clone(),
                        indices: node.indices.clone(),
                    };
                    let result = $crate::geometry::subtract_box(
                        &current,
                        [min_x as f32, min_y as f32, min_z as f32],
                        [max_x as f32, max_y as f32, max_z as f32],
                    );
                    eng.scene.update_geometry(handle, result.vertices, result.indices);
                }
        })
        }

        // bloom_enable_shadows  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_enable_shadows() {
            $crate::ffi::guard("bloom_enable_shadows", move || {
                engine().renderer.shadow_map.enable();
        })
        }

        // bloom_disable_shadows  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_disable_shadows() {
            $crate::ffi::guard("bloom_disable_shadows", move || {
                engine().renderer.shadow_map.disable();
        })
        }

        // bloom_dump_shadow_map  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_dump_shadow_map(path_ptr: *const u8) {
            $crate::ffi::guard("bloom_dump_shadow_map", move || {
                let path = $crate::string_header::str_from_header(path_ptr).to_string();
                engine().renderer.dump_shadow_map(&path);
        })
        }

        // bloom_scene_attach_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_scene_attach_model(node_handle: f64, model_handle: f64, mesh_index: f64) {
            $crate::ffi::guard("bloom_scene_attach_model", move || {
                let eng = engine();
                let mi = mesh_index as usize;

                let model_data = match eng.models.models.get(model_handle) {
                    Some(md) => md,
                    None => return,
                };
                if mi >= model_data.meshes.len() { return; }
                let mesh = &model_data.meshes[mi];

                let vertices = mesh.vertices.clone();
                let indices = mesh.indices.clone();
                let base_color_tex = mesh.texture_idx;
                let normal_tex = mesh.normal_texture_idx;
                let mr_tex = mesh.metallic_roughness_texture_idx;
                let emissive_tex = mesh.emissive_texture_idx;
                let emissive_factor = mesh.emissive_factor;
                eng.scene.update_geometry(node_handle, vertices, indices);

                if let Some(tex_idx) = base_color_tex {
                    eng.scene.set_material_texture(node_handle, tex_idx);
                }
                if let Some(tex_idx) = normal_tex {
                    eng.scene.set_material_normal_texture(node_handle, tex_idx);
                }
                if let Some(tex_idx) = mr_tex {
                    eng.scene.set_material_metallic_roughness_texture(node_handle, tex_idx);
                }
                if let Some(tex_idx) = emissive_tex {
                    eng.scene.set_material_emissive_texture(node_handle, tex_idx);
                }
                eng.scene.set_material_emissive_factor(
                    node_handle,
                    emissive_factor[0],
                    emissive_factor[1],
                    emissive_factor[2],
                );
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_scene_attach_model(_node_handle: f64, _model_handle: f64, _mesh_index: f64) {
            $crate::ffi::feature_off_warn_once("bloom_scene_attach_model", "models3d");
        }

        // bloom_enable_postfx  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_enable_postfx() {
            $crate::ffi::guard("bloom_enable_postfx", move || {
                let eng = engine();
                let w = eng.renderer.width();
                let h = eng.renderer.height();
                let fmt = eng.renderer.surface_format();
                eng.postfx = Some($crate::postfx::PostFxPipeline::new(
                    &eng.renderer.device, w, h, fmt,
                ));
        })
        }

        // bloom_disable_postfx  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_disable_postfx() {
            $crate::ffi::guard("bloom_disable_postfx", move || {
                engine().postfx = None;
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

        // bloom_project_to_screen  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_project_to_screen(wx: f64, wy: f64, wz: f64) -> f64 {
            $crate::ffi::guard("bloom_project_to_screen", move || {
                let eng = engine();
                let vp = eng.renderer.vp_matrix();
                let w = eng.renderer.width() as f32;
                let h = eng.renderer.height() as f32;

                // Multiply by VP matrix
                let x = wx as f32;
                let y = wy as f32;
                let z = wz as f32;
                let clip_x = vp[0][0]*x + vp[1][0]*y + vp[2][0]*z + vp[3][0];
                let clip_y = vp[0][1]*x + vp[1][1]*y + vp[2][1]*z + vp[3][1];
                let clip_w = vp[0][3]*x + vp[1][3]*y + vp[2][3]*z + vp[3][3];

                if clip_w <= 0.0 {
                    unsafe { LAST_PROJECT = (-9999.0, -9999.0); }
                    return -9999.0;
                }

                // NDC to screen
                let ndc_x = clip_x / clip_w;
                let ndc_y = clip_y / clip_w;
                let screen_x = ((ndc_x + 1.0) * 0.5 * w) as f64;
                let screen_y = ((1.0 - ndc_y) * 0.5 * h) as f64;

                unsafe { LAST_PROJECT = (screen_x, screen_y); }
                screen_x
        })
        }

        // bloom_project_screen_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_project_screen_y() -> f64 {
            $crate::ffi::guard("bloom_project_screen_y", move || {
                unsafe { LAST_PROJECT.1 }
        })
        }

        // bloom_scene_pick  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_pick(screen_x: f64, screen_y: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_pick", move || {
                let eng = engine();
                let inv_vp = eng.renderer.inverse_vp_matrix();
                let cam_pos = eng.renderer.camera_pos();
                let w = eng.renderer.width() as f32;
                let h = eng.renderer.height() as f32;

                let (origin, direction) = $crate::picking::screen_to_ray(
                    screen_x as f32, screen_y as f32,
                    w, h, &inv_vp, &cam_pos,
                );

                let result = $crate::picking::raycast_scene(&eng.scene, &origin, &direction);
                let hit = result.hit;
                unsafe { LAST_PICK = Some(result); }
                if hit { 1.0 } else { 0.0 }
        })
        }

        // bloom_pick_hit_handle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_handle() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_handle", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.handle).unwrap_or(0.0) }
        })
        }

        // bloom_pick_hit_distance  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_distance() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_distance", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.distance as f64).unwrap_or(0.0) }
        })
        }

        // bloom_pick_hit_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_x() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_x", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.point[0] as f64).unwrap_or(0.0) }
        })
        }

        // bloom_pick_hit_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_y() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_y", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.point[1] as f64).unwrap_or(0.0) }
        })
        }

        // bloom_pick_hit_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_z() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_z", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.point[2] as f64).unwrap_or(0.0) }
        })
        }

        // bloom_pick_hit_normal_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_normal_x() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_normal_x", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.normal[0] as f64).unwrap_or(0.0) }
        })
        }

        // bloom_pick_hit_normal_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_normal_y() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_normal_y", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.normal[1] as f64).unwrap_or(0.0) }
        })
        }

        // bloom_pick_hit_normal_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_normal_z() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_normal_z", move || {
                unsafe { LAST_PICK.as_ref().map(|r| r.normal[2] as f64).unwrap_or(0.0) }
        })
        }

        // bloom_stage_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_stage_texture(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_stage_texture", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => $crate::staging::decode_and_stage_texture(&data),
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_stage_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_stage_model(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_stage_model", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                let data = match std::fs::read(path) {
                    Ok(d) => d,
                    Err(_) => return 0.0,
                };
                match $crate::models::load_gltf_staged(&data) {
                    Some(staged) => $crate::staging::stage_model(staged),
                    None => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_stage_model(_path_ptr: *const u8) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_stage_model", "models3d");
            0.0
        }

        // bloom_stage_sound  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_stage_sound(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_stage_sound", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                let data = match std::fs::read(path) { Ok(d) => d, Err(_) => return 0.0 };
                match $crate::audio::decode_audio(path, &data) {
                    Some(s) => $crate::staging::stage_sound(s),
                    None => 0.0,
                }
        })
        }

        // bloom_commit_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_commit_texture(staging_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_commit_texture", move || {
                let staged = match $crate::staging::take_texture(staging_handle) {
                    Some(s) => s,
                    None => return 0.0,
                };
                let eng = engine();
                let bind_group_idx = eng.renderer.register_texture(staged.width, staged.height, &staged.data);
                eng.textures.textures.alloc($crate::textures::TextureData {
                    bind_group_idx, width: staged.width, height: staged.height,
                })
        })
        }

        // bloom_commit_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_commit_model(staging_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_commit_model", move || {
                let staged = match $crate::staging::take_model(staging_handle) {
                    Some(s) => s,
                    None => return 0.0,
                };
                let eng = engine();
                let mut tex_map: Vec<u32> = Vec::with_capacity(staged.textures.len());
                for tex in &staged.textures {
                    tex_map.push(eng.renderer.register_texture(tex.width, tex.height, &tex.data));
                }
                let mut model = staged.model;
                for mesh in &mut model.meshes {
                    if let Some(ref mut idx) = mesh.texture_idx {
                        let staged_idx = *idx as usize;
                        if staged_idx > 0 && staged_idx <= tex_map.len() {
                            *idx = tex_map[staged_idx - 1];
                        } else {
                            mesh.texture_idx = None;
                        }
                    }
                }
                eng.models.models.alloc(model)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_commit_model(_staging_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_commit_model", "models3d");
            0.0
        }

        // bloom_commit_sound  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_commit_sound(staging_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_commit_sound", move || {
                match $crate::staging::take_sound(staging_handle) {
                    Some(sd) => engine().audio.load_sound(sd),
                    None => 0.0,
                }
        })
        }

        // bloom_commit_music  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_commit_music(staging_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_commit_music", move || {
                match $crate::staging::take_sound(staging_handle) {
                    Some(sd) => engine().audio.load_music(sd),
                    None => 0.0,
                }
        })
        }

        // bloom_compile_material_from_file  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_from_file(
            path_ptr: *const u8,
            bucket_kind: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_compile_material_from_file", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let path = $crate::string_header::str_from_header(path_ptr);
                let (profile, bucket, reads_scene) = match bucket_kind as u32 {
                    0 => (FragmentProfile::Opaque,      Bucket::Opaque,      false),
                    1 => (FragmentProfile::Translucent, Bucket::Transparent, false),
                    2 => (FragmentProfile::Translucent, Bucket::Refractive,  true),
                    3 => (FragmentProfile::Translucent, Bucket::Additive,    false),
                    4 => (FragmentProfile::Opaque,      Bucket::Cutout,      false),
                    _ => {
                        eprintln!("[material] from_file: unknown bucket_kind {bucket_kind}");
                        return 0.0;
                    }
                };
                match engine().renderer.compile_material_from_file(
                    std::path::Path::new(path), profile, bucket, reads_scene,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] from_file failed: {e}"); 0.0 }
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

        // bloom_set_material_params  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_params(
            handle: f64,
            params_ptr: *const f64,
            param_count: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_params", move || {
                let count = param_count as usize;
                if count > 64 {
                    eprintln!("[material] set_material_params: param_count {} > 64 (256-byte UBO cap)", count);
                    return;
                }
                let mut bytes = vec![0u8; count * 4];
                if !params_ptr.is_null() && count > 0 {
                    let slots = unsafe { std::slice::from_raw_parts(params_ptr, count) };
                    for (i, &v) in slots.iter().enumerate() {
                        let f = v as f32;
                        bytes[i*4..i*4+4].copy_from_slice(&f.to_le_bytes());
                    }
                }
                let eng = engine();
                if let Err(e) = eng.renderer.material_system.set_user_params(
                    &eng.renderer.device, &eng.renderer.queue,
                    handle as u32, &bytes,
                ) {
                    eprintln!("[material] set_material_params failed: {}", e);
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_params(_handle: f64, _params_ptr: *const f64, _param_count: f64) {
            $crate::ffi::feature_off_warn_once("bloom_set_material_params", "models3d");
        }

        // bloom_splat_impulse  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_splat_impulse(x: f64, z: f64, radius: f64, strength: f64) {
            $crate::ffi::guard("bloom_splat_impulse", move || {
                engine().renderer.impulse_field.submit_splat(
                    x as f32, z as f32, radius as f32, strength as f32,
                );
        })
        }
    };
}

// Compile-coverage for the macro body: expand it against mock hooks so
// `cargo test -p bloom-shared` catches breakage without building any
// platform crate. Nothing here runs — the hooks panic if called.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod macro_expansion_compile_check {
    #![allow(dead_code, unused_variables)]

    fn engine() -> &'static mut crate::engine::EngineState {
        unreachable!("compile-coverage mock — never called")
    }

    fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(path)
    }

    crate::define_core_ffi!();
}
