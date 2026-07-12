//! Texture/image/font/file loading, staging, screenshots.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_assets {
    () => {

        // bloom_take_screenshot  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_take_screenshot(path_ptr: *const u8) {
            $crate::ffi::guard("bloom_take_screenshot", move || {
                let path = $crate::string_header::str_from_header(path_ptr).to_string();
                eprintln!("bloom: screenshot requested -> '{}'", path);
                let eng = engine();
                eng.renderer.screenshot_requested = true;
                eng.renderer.pending_screenshot_path = Some(path);
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

        // bloom_load_shader  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_load_shader(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_shader", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                engine().renderer.load_custom_shader(source) as f64
        })
        }

        // bloom_load_music  [source: curated]
        #[no_mangle]
        pub extern "C" fn bloom_load_music(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_music", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    // Streams OGG/MP3 from the compressed bytes (background
                    // decode worker); WAV and wasm32 fully decode.
                    Ok(data) => engine().audio.load_music_bytes(path, data),
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_write_file  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_write_file(path_ptr: *const u8, data_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_write_file", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                // A string that failed ABI validation must NOT be written as "" and
                // reported as a success. That is not a save, it is a deletion with a
                // thumbs-up, and it is exactly what happened to every world the
                // editor ever saved.
                let data = match $crate::string_header::try_str_from_header(data_ptr) {
                    Some(d) => d,
                    None => return 0.0,
                };
                match std::fs::write(path, data.as_bytes()) {
                    Ok(_) => 1.0,
                    Err(_) => 0.0,
                }
        })
        }

        // bloom_launch_process  [EN-048]
        //
        // Perry's `child_process.spawn` COMPILES and then does nothing — it returns a
        // child with an undefined pid and no process is started. So a tool that wants
        // to launch another program (the editor's play-in-editor: save the level, run
        // the game on it) has nowhere to go.
        //
        // Fire-and-forget by design. The caller is a GUI that must not block on, or
        // die with, the thing it launched: close the game, and you are back in the
        // editor with your undo history intact.
        //
        // `args` is newline-separated. Not shell-escaped and not shell-interpreted —
        // there is no shell here, which is also why there is nothing to inject into.
        #[no_mangle]
        pub extern "C" fn bloom_launch_process(
            cmd_ptr: *const u8, args_ptr: *const u8, cwd_ptr: *const u8,
        ) -> f64 {
            $crate::ffi::guard("bloom_launch_process", move || {
                let cmd = $crate::string_header::str_from_header(cmd_ptr);
                if cmd.is_empty() { return 0.0; }
                let args = $crate::string_header::str_from_header(args_ptr);
                let cwd = $crate::string_header::str_from_header(cwd_ptr);

                // Resolve the program against `cwd` when it is a bare name.
                //
                // Rust's `Command::current_dir` sets the CHILD's working directory --
                // it does NOT affect how the program is FOUND, which happens in the
                // parent's context. So launching "main.exe" with cwd "<project>"
                // fails with "program not found" even though main.exe is sitting
                // right there in <project>. Which is exactly what it did.
                let bare = !cmd.chars().any(|ch| ch == '/' || ch == std::path::MAIN_SEPARATOR);
                let resolved: std::path::PathBuf = if !cwd.is_empty() && bare {
                    std::path::Path::new(cwd).join(cmd)
                } else {
                    std::path::PathBuf::from(cmd)
                };
                let mut c = std::process::Command::new(&resolved);
                for a in args.split('\n') {
                    if !a.is_empty() { c.arg(a); }
                }
                if !cwd.is_empty() { c.current_dir(cwd); }
                // Detach: we never wait on it, and we do not want its output in ours.
                c.stdin(std::process::Stdio::null())
                 .stdout(std::process::Stdio::null())
                 .stderr(std::process::Stdio::null());
                let r = c.spawn();
                match r {
                    Ok(child) => child.id() as f64,
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

    };
}
