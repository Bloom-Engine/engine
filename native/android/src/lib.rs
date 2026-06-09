use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::{str_from_header, alloc_perry_string};
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3};

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut NATIVE_WINDOW: *mut libc::c_void = std::ptr::null_mut();
static AUDIO_RUNNING: AtomicBool = AtomicBool::new(false);
static mut ASSET_BASE_PATH: Option<String> = None;

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}

/// Resolve relative asset paths to the app's base asset directory.
/// On Android, relative paths like "assets/models/tree.glb" won't resolve
/// because the working directory isn't the app's data directory.
fn resolve_path(path: &str) -> String {
    if path.starts_with('/') {
        return path.to_string();
    }
    unsafe {
        if let Some(ref base) = ASSET_BASE_PATH {
            format!("{}/{}", base, path)
        } else {
            path.to_string()
        }
    }
}

/// Called by the Android Activity to set the base path for asset resolution.
/// Should be set to the app's files directory where assets are extracted.
#[no_mangle]
pub extern "C" fn bloom_android_set_asset_path(path_ptr: *const u8) {
    let path = str_from_header(path_ptr);
    unsafe {
        ASSET_BASE_PATH = Some(path.to_string());
    }
}

// ============================================================
// ANativeWindow FFI
// ============================================================

extern "C" {
    fn ANativeWindow_getWidth(window: *mut libc::c_void) -> i32;
    fn ANativeWindow_getHeight(window: *mut libc::c_void) -> i32;
    fn ANativeWindow_acquire(window: *mut libc::c_void);
    fn ANativeWindow_release(window: *mut libc::c_void);
}

/// Called by Perry's Android runtime to set the ANativeWindow pointer
/// before bloom_init_window is invoked.
#[no_mangle]
pub extern "C" fn bloom_android_set_native_window(window: *mut libc::c_void) {
    unsafe {
        if !window.is_null() {
            ANativeWindow_acquire(window);
        }
        NATIVE_WINDOW = window;
    }
}

fn pollster_block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    use std::pin::Pin;
    use std::sync::Arc;
    struct NoopWaker;
    impl Wake for NoopWaker { fn wake(self: Arc<Self>) {} }
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(Box::new(future)) };
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

// ============================================================
// Window + Renderer init
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8, _fullscreen: f64) {
    let _title = str_from_header(title_ptr);

    unsafe {
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: starting\0".as_ptr());
        let window = NATIVE_WINDOW;
        // If no native window was set, use requested dimensions with a headless surface
        let (pixel_w, pixel_h) = if !window.is_null() {
            (ANativeWindow_getWidth(window) as u32, ANativeWindow_getHeight(window) as u32)
        } else {
            (width as u32, height as u32)
        };
        // Logical size is half of physical: the game UI (fonts, layout constants)
        // was sized for ~1170×540 landscape (iPhone-sized); on Android's 2340×1080
        // panel we'd otherwise render at native pixel size and everything looks
        // half-scale. wgpu still renders to the full physical surface; only the
        // `screen_width`/`screen_height` the game sees are halved.
        let logical_w = (pixel_w / 2).max(1);
        let logical_h = (pixel_h / 2).max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        // Create surface from ANativeWindow
        if window.is_null() {
            // Fallback: can't render without a window, but don't panic
            // Create a minimal engine state (game logic will work, rendering will no-op)
            return;
        }

        let handle = raw_window_handle::AndroidNdkWindowHandle::new(
            std::ptr::NonNull::new(window).unwrap()
        );
        let raw = raw_window_handle::RawWindowHandle::AndroidNdk(handle);
        let surface = instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Android(
                raw_window_handle::AndroidDisplayHandle::new()
            )),
            raw_window_handle: raw,
        }).expect("Failed to create surface");
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: surface created\0".as_ptr());

        let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }));
        let adapter = match adapter {
            Ok(a) => a,
            Err(_) => {
                // Try again without surface compatibility requirement
                match pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })) {
                    Ok(a) => a,
                    Err(_) => panic!("No GPU adapter found"),
                }
            }
        };
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: adapter found\0".as_ptr());
        {
            let info = adapter.get_info();
            let msg = std::ffi::CString::new(format!(
                "adapter: name='{}' backend={:?} device_type={:?} driver='{}' driver_info='{}'",
                info.name, info.backend, info.device_type, info.driver, info.driver_info
            )).unwrap();
            __android_log_print(3, b"BloomEngine\0".as_ptr(), b"%s\0".as_ptr(), msg.as_ptr());
        }

        // Ticket 007b: most Android GPUs lack RT, but recent Adreno /
        // Mali-Immortalis devices do — request the feature if advertised.
        // Limits merge: start from downlevel (required for older Android
        // adapters), then layer acceleration-structure minimums on top
        // when RT was granted.
        let supported = adapter.features();
        let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let rt_mask = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        let mut required_features = wgpu::Features::empty();
        // Ticket 011: request TIMESTAMP_QUERY when supported so the profiler
        // can record GPU timings. Optional — profiler falls back to CPU-only
        // when the adapter doesn't grant it (many Android GPUs won't).
        if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if !force_sw_gi && supported.contains(rt_mask) {
            required_features |= rt_mask;
        }
        let experimental_features = if required_features.intersects(rt_mask) {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::downlevel_defaults()
            .using_resolution(adapter_limits.clone());
        // The renderer's `joint_bg` binds a 64KB uniform buffer, but
        // downlevel_defaults caps uniform-buffer bindings at 16KB, so
        // create_bind_group panics on mobile GPUs (e.g. Adreno) with
        // "range 65536 exceeds max_*_buffer_binding_size limit 16384". Raise the
        // buffer-binding sizes (and bind-group count) to the adapter's maximum;
        // these are guaranteed-supported and match the desktop limits.
        required_limits.max_uniform_buffer_binding_size =
            adapter_limits.max_uniform_buffer_binding_size;
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_bind_groups =
            required_limits.max_bind_groups.max(5).min(adapter_limits.max_bind_groups);
        if required_features.intersects(rt_mask) {
            required_limits = required_limits
                .using_minimum_supported_acceleration_structure_values();
        }
        let (device, queue) = pollster_block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("bloom_device"),
                required_features,
                required_limits,
                experimental_features,
                ..Default::default()
            },
        )).expect("Failed to create device");
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: device created\0".as_ptr());

        let surface_caps = surface.get_capabilities(&adapter);
        if surface_caps.formats.is_empty() {
            panic!("Surface reports no supported formats (emulator Vulkan limitation)");
        }
        let format = surface_caps.formats.iter()
            .find(|f| f.is_srgb()).copied()
            .unwrap_or(surface_caps.formats[0]);

        let alpha_mode = if surface_caps.alpha_modes.is_empty() {
            wgpu::CompositeAlphaMode::Auto
        } else {
            surface_caps.alpha_modes[0]
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: pixel_w,
            height: pixel_h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: surface configured\0".as_ptr());
        let renderer = Renderer::new(device, queue, surface, surface_config, logical_w, logical_h);
        let _ = ENGINE.set(EngineState::new(renderer));
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: engine initialized\0".as_ptr());
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {
    unsafe {
        if !NATIVE_WINDOW.is_null() {
            ANativeWindow_release(NATIVE_WINDOW);
            NATIVE_WINDOW = std::ptr::null_mut();
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 {
    if engine().should_close { 1.0 } else { 0.0 }
}

// ============================================================
// Touch input polling (called from Perry's Android event pump)
// ============================================================

/// Called by Perry's Android layer when a touch event occurs.
#[no_mangle]
pub extern "C" fn bloom_android_on_touch(action: i32, x: f64, y: f64, pointer_index: i32) {
    unsafe {
        if let Some(eng) = ENGINE.get_mut() {
            let sw = eng.screen_width();
            let sh = eng.screen_height();
            let lx = x * 0.5;
            let ly = y * 0.5;
            let msg = std::ffi::CString::new(format!("touch a={} raw=({},{}) scaled=({},{}) sw={} sh={}", action, x, y, lx, ly, sw, sh)).unwrap();
            __android_log_print(3, b"BloomTouch\0".as_ptr(), b"%s\0".as_ptr(), msg.as_ptr());
            eng.input.set_mouse_position(lx, ly);
            if action == 1 || action == 3 {
                eng.input.release_touch(pointer_index as usize, lx, ly); // UP / CANCEL
            } else {
                eng.input.set_touch(pointer_index as usize, lx, ly, true); // DOWN / MOVE
            }
            match action {
                0 => eng.input.set_mouse_button_down(0),  // ACTION_DOWN
                1 => eng.input.set_mouse_button_up(0),    // ACTION_UP
                2 => {}                                      // ACTION_MOVE
                _ => {}
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine().end_frame();
    }));
    if let Err(e) = result {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            format!("unknown panic: {:?}", e.type_id())
        };
        // Write to file since eprintln doesn't reach logcat on Android
        static mut LOGGED: bool = false;
        unsafe {
            if !LOGGED {
                LOGGED = true;
                let path = resolve_path("bloom_panic.txt");
                let _ = std::fs::write(&path, format!("end_drawing panic: {}\n", msg));
            }
        }
    }
}

// ============================================================
// Audio (Oboe / AAudio)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.set_clear_color(r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_set_target_fps(fps: f64) { engine().target_fps = fps; }

#[no_mangle]
pub extern "C" fn bloom_set_direct_2d_mode(on: f64) { engine().direct_2d_mode = on > 0.5; }

#[no_mangle]
pub extern "C" fn bloom_get_delta_time() -> f64 { engine().delta_time }

#[no_mangle]
pub extern "C" fn bloom_get_fps() -> f64 { engine().get_fps() }

#[no_mangle]
pub extern "C" fn bloom_get_screen_width() -> f64 { engine().screen_width() }

#[no_mangle]
pub extern "C" fn bloom_get_screen_height() -> f64 { engine().screen_height() }

#[no_mangle]
pub extern "C" fn bloom_is_key_pressed(key: f64) -> f64 {
    if engine().input.is_key_pressed(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_key_down(key: f64) -> f64 {
    if engine().input.is_key_down(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_key_released(key: f64) -> f64 {
    if engine().input.is_key_released(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_x() -> f64 { engine().input.mouse_x }

#[no_mangle]
pub extern "C" fn bloom_get_mouse_y() -> f64 { engine().input.mouse_y }

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_pressed(btn: f64) -> f64 {
    if engine().input.is_mouse_button_pressed(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_down(btn: f64) -> f64 {
    if engine().input.is_mouse_button_down(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_released(btn: f64) -> f64 {
    if engine().input.is_mouse_button_released(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_draw_line(x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_line(x1, y1, x2, y2, thickness, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_rect(x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect(x, y, w, h, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_rect_lines(x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect_lines(x, y, w, h, thickness, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle(cx, cy, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle_lines(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle_lines(cx, cy, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_poly(cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_poly(cx, cy, sides, radius, rotation, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_text(text_ptr: *const u8, x: f64, y: f64, size: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text(&mut eng.renderer, text, x, y, size as u32, r, g, b, a);
    eng.text = text_renderer;
}

#[no_mangle]
pub extern "C" fn bloom_measure_text(text_ptr: *const u8, size: f64) -> f64 {
    let text = str_from_header(text_ptr);
    engine().text.measure_text(text, size as u32)
}

#[no_mangle]
pub extern "C" fn bloom_load_font(path_ptr: *const u8, size: f64) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) { Ok(data) => engine().text.load_font(&data) as f64, Err(_) => 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_unload_font(font_handle: f64) {
    engine().text.unload_font(font_handle as usize);
}

#[no_mangle]
pub extern "C" fn bloom_draw_text_ex(font_handle: f64, text_ptr: *const u8, x: f64, y: f64, size: f64, spacing: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text_ex(&mut eng.renderer, font_handle as usize, text, x, y, size as u32, spacing as f32, r, g, b, a);
    eng.text = text_renderer;
}

#[no_mangle]
pub extern "C" fn bloom_measure_text_ex(font_handle: f64, text_ptr: *const u8, size: f64, spacing: f64) -> f64 {
    let text = str_from_header(text_ptr);
    engine().text.measure_text_ex(font_handle as usize, text, size as u32, spacing as f32)
}

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    AUDIO_RUNNING.store(true, Ordering::SeqCst);
    std::thread::spawn(|| {
        android_audio_thread();
    });
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    AUDIO_RUNNING.store(false, Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
}

fn android_audio_thread() {
    // Use oboe (AAudio/OpenSL ES wrapper) for audio output
    use oboe::*;

    struct BloomAudioCallback;

    impl AudioOutputCallback for BloomAudioCallback {
        type FrameType = (f32, Stereo);

        fn on_audio_ready(&mut self, _stream: &mut dyn AudioOutputStreamSafe, frames: &mut [(f32, f32)]) -> DataCallbackResult {
            // Convert frame tuples to interleaved slice
            let len = frames.len() * 2;
            let ptr = frames.as_mut_ptr() as *mut f32;
            let interleaved = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            for s in interleaved.iter_mut() { *s = 0.0; }
            unsafe {
                ENGINE.get_mut().map(|eng| { eng.audio.mix_output(interleaved); });
            }
            if AUDIO_RUNNING.load(Ordering::SeqCst) {
                DataCallbackResult::Continue
            } else {
                DataCallbackResult::Stop
            }
        }
    }

    let mut stream = AudioStreamBuilder::default()
        .set_performance_mode(PerformanceMode::LowLatency)
        .set_sharing_mode(SharingMode::Shared)
        .set_format::<f32>()
        .set_channel_count::<Stereo>()
        .set_sample_rate(44100)
        .set_callback(BloomAudioCallback)
        .open_stream();

    match stream {
        Ok(ref mut s) => {
            let _ = s.start();
            // Keep thread alive while audio is running
            while AUDIO_RUNNING.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let _ = s.stop();
        }
        Err(_) => {}
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_sound(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            if let Some(s) = parse_wav(&data) { engine().audio.load_sound(s) }
            else if let Some(s) = parse_ogg(&data) { engine().audio.load_sound(s) }
            else if let Some(s) = parse_mp3(&data) { engine().audio.load_sound(s) }
            else { 0.0 }
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_play_sound(handle: f64) { engine().audio.play_sound(handle); }
#[no_mangle]
pub extern "C" fn bloom_stop_sound(handle: f64) { engine().audio.stop_sound(handle); }
#[no_mangle]
pub extern "C" fn bloom_set_sound_volume(handle: f64, volume: f64) { engine().audio.set_sound_volume(handle, volume as f32); }
#[no_mangle]
pub extern "C" fn bloom_set_master_volume(volume: f64) { engine().audio.master_volume = volume as f32; }

#[no_mangle]
pub extern "C" fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
    engine().audio.play_sound_3d(handle, x as f32, y as f32, z as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
    engine().audio.set_listener_position(x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32);
}

// --- Texture FFI ---

#[no_mangle]
pub extern "C" fn bloom_load_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            let eng = engine();
            let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
            eng.textures.load_texture(unsafe { &mut *renderer_ptr }, &data)
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_texture(handle: f64) {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
    eng.textures.unload_texture(handle, unsafe { &mut *renderer_ptr });
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture(handle: f64, x: f64, y: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture(idx, x, y, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_rec(handle: f64, src_x: f64, src_y: f64, src_w: f64, src_h: f64, dst_x: f64, dst_y: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture_rec(idx, src_x, src_y, src_w, src_h, dst_x, dst_y, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_pro(handle: f64, src_x: f64, src_y: f64, src_w: f64, src_h: f64, dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64, origin_x: f64, origin_y: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture_pro(idx, src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h, origin_x, origin_y, rotation, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_width(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.width as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_height(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.height as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_gen_texture_mipmaps(_handle: f64) {
    // No-op: wgpu handles mipmaps internally
}

#[no_mangle]
pub extern "C" fn bloom_set_texture_filter(handle: f64, mode: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.set_texture_filter(bind_group_idx, mode > 0.5);
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_image(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) { Ok(data) => engine().textures.load_image(&data), Err(_) => 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_image_resize(handle: f64, w: f64, h: f64) {
    engine().textures.image_resize(handle, w as u32, h as u32);
}

#[no_mangle]
pub extern "C" fn bloom_image_crop(handle: f64, x: f64, y: f64, w: f64, h: f64) {
    engine().textures.image_crop(handle, x as u32, y as u32, w as u32, h as u32);
}

#[no_mangle]
pub extern "C" fn bloom_image_flip_h(handle: f64) {
    engine().textures.image_flip_h(handle);
}

#[no_mangle]
pub extern "C" fn bloom_image_flip_v(handle: f64) {
    engine().textures.image_flip_v(handle);
}

#[no_mangle]
pub extern "C" fn bloom_load_texture_from_image(handle: f64) -> f64 {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
    eng.textures.load_texture_from_image(handle, unsafe { &mut *renderer_ptr })
}

// --- Camera FFI ---

#[no_mangle]
pub extern "C" fn bloom_begin_mode_2d(offset_x: f64, offset_y: f64, target_x: f64, target_y: f64, rotation: f64, zoom: f64) {
    engine().renderer.begin_mode_2d(offset_x as f32, offset_y as f32, target_x as f32, target_y as f32, rotation as f32, zoom as f32);
}
#[no_mangle]
pub extern "C" fn bloom_end_mode_2d() { engine().renderer.end_mode_2d(); }

#[no_mangle]
pub extern "C" fn bloom_begin_mode_3d(pos_x: f64, pos_y: f64, pos_z: f64, target_x: f64, target_y: f64, target_z: f64, up_x: f64, up_y: f64, up_z: f64, fovy: f64, projection: f64) {
    engine().renderer.begin_mode_3d(pos_x as f32, pos_y as f32, pos_z as f32, target_x as f32, target_y as f32, target_z as f32, up_x as f32, up_y as f32, up_z as f32, fovy as f32, projection as f32);
}
#[no_mangle]
pub extern "C" fn bloom_end_mode_3d() { engine().renderer.end_mode_3d(); }

// --- 3D Drawing FFI ---

#[no_mangle]
pub extern "C" fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube(x, y, z, w, h, d, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube_wires(x, y, z, w, h, d, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_sphere(x: f64, y: f64, z: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere(x, y, z, radius, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_sphere_wires(x: f64, y: f64, z: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere_wires(x, y, z, radius, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_cylinder(x: f64, y: f64, z: f64, rt: f64, rb: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cylinder(x, y, z, rt, rb, h, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_plane(x: f64, y: f64, z: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_plane(x, y, z, w, d, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_grid(slices: f64, spacing: f64) {
    engine().renderer.draw_grid(slices as i32, spacing);
}
#[no_mangle]
pub extern "C" fn bloom_draw_ray(ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_ray(ox, oy, oz, dx, dy, dz, r, g, b, a);
}

// --- Model FFI ---

#[no_mangle]
pub extern "C" fn bloom_load_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            let eng = engine();
            let renderer_ptr = &mut eng.renderer as *mut crate::Renderer;
            eng.models.load_model_with_textures(&data, unsafe { &mut *renderer_ptr })
        }
        Err(_) => 0.0,
    }
}
#[no_mangle]
pub extern "C" fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
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
}
#[no_mangle]
pub extern "C" fn bloom_draw_model_rotated(
    handle: f64, x: f64, y: f64, z: f64,
    scale: f64, rot_y: f64,
    color_packed_argb: f64,
) {
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
}
#[no_mangle]
pub extern "C" fn bloom_unload_model(handle: f64) { engine().models.unload_model(handle); }

#[no_mangle]
pub extern "C" fn bloom_get_model_mesh_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_model_material_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_cube(w: f64, h: f64, d: f64) -> f64 {
    engine().models.gen_mesh_cube(w as f32, h as f32, d as f32)
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_heightmap(image_handle: f64, size_x: f64, size_y: f64, size_z: f64) -> f64 {
    let eng = engine();
    if let Some(img) = eng.textures.images.get(image_handle) {
        let data = img.data.clone();
        let w = img.width;
        let h = img.height;
        eng.models.gen_mesh_heightmap(&data, w, h, size_x as f32, size_y as f32, size_z as f32)
    } else {
        0.0
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_shader(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    engine().renderer.load_custom_shader(source) as f64
}

#[no_mangle]
pub extern "C" fn bloom_create_mesh(vertex_ptr: *const f32, vertex_count: f64, index_ptr: *const u32, index_count: f64) -> f64 {
    if vertex_ptr.is_null() || index_ptr.is_null() { return 0.0; }
    let vcount = vertex_count as usize;
    let icount = index_count as usize;
    let vertex_data = unsafe { std::slice::from_raw_parts(vertex_ptr, vcount * 12) }; // 12 floats per vertex
    let index_data = unsafe { std::slice::from_raw_parts(index_ptr, icount) };
    engine().models.create_mesh(vertex_data, index_data)
}

// ============================================================
// Phase 1c — material system FFI
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_material_params(
    handle: f64,
    params_ptr: *const f64,
    param_count: f64,
) {
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
}

#[no_mangle]
pub extern "C" fn bloom_compile_material(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material(source) {
        Ok(handle) => handle as f64,
        Err(e) => {
            eprintln!("[material] compile failed: {:?}", e);
            0.0
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_refractive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Refractive, true, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[refractive] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_transparent(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Transparent, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_additive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Additive, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_cutout(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Opaque, Bucket::Cutout, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_instanced(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_instanced(source) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] instanced compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_create_instance_buffer(
    data_ptr: *const f64, instance_count: f64,
) -> f64 {
    if data_ptr.is_null() || instance_count <= 0.0 { return 0.0; }
    let count = instance_count as u32;
    let slot_count = (count as usize) * 9;
    let raw_f64 = unsafe { std::slice::from_raw_parts(data_ptr, slot_count) };
    let raw_f32: Vec<f32> = raw_f64.iter().map(|&v| v as f32).collect();
    engine().renderer.create_instance_buffer(&raw_f32, count) as f64
}

#[no_mangle]
pub extern "C" fn bloom_submit_material_draw_instanced(
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

#[no_mangle]
pub extern "C" fn bloom_destroy_instance_buffer(handle: f64) {
    engine().renderer.destroy_instance_buffer(handle as u32);
}

/// EN-011 — create a planar reflection probe. See macOS lib.rs for the
/// full doc comment; this entry-point exists on every native platform
/// so games can target the same FFI surface across iOS/tvOS/Windows/
/// Linux/Android.
#[no_mangle]
pub extern "C" fn bloom_create_planar_reflection(
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
#[no_mangle]
pub extern "C" fn bloom_set_material_reflection_probe(
    material: f64, probe: f64,
) {
    engine().renderer.set_material_reflection_probe(material as u32, probe as u32);
}

/// EN-014 — create a texture array from concatenated RGBA8 byte data.
/// See macOS lib.rs for the full doc comment; this entry-point exists
/// on every native platform so a TS game targets the same FFI across
/// iOS / tvOS / Windows / Linux / Android.
#[no_mangle]
pub extern "C" fn bloom_create_texture_array(
    data_ptr:    *const u8,
    data_len:    f64,
    width:       f64,
    height:      f64,
    layer_count: f64,
) -> f64 {
    // EN-014 V2 — V1 forwards to _ex with default sRGB / no mips.
    bloom_create_texture_array_ex(data_ptr, data_len, width, height, layer_count, 0.0, 1.0)
}

/// EN-014 V2 — explicit format + mip control. See macOS lib.rs for docs.
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
    if data_ptr.is_null() || data_len <= 0.0 { return 0.0; }
    let w = width as u32;
    let h = height as u32;
    if w == 0 || h == 0 { return 0.0; }
    let layers_count = (layer_count as u32)
        .min(bloom_shared::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
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
}

/// EN-014 — link a texture-array handle to a material at one of three
/// slots: 0 = albedo (binding 14), 1 = normal (binding 15),
/// 2 = MR (binding 16). Pass `array = 0` to revert to the stub.
#[no_mangle]
pub extern "C" fn bloom_set_material_texture_array(
    material: f64, slot: f64, array: f64,
) {
    engine().renderer.set_material_texture_array(
        material as u32, slot as u32, array as u32,
    );
}

/// EN-012 — set the shading model for a material (0=default lit,
/// 1=foliage, 2=subsurface V2 stub).
#[no_mangle]
pub extern "C" fn bloom_set_material_shading_model(
    material: f64, model: f64,
) {
    engine().renderer.set_material_shading_model(material as u32, model as u32);
}

/// EN-012 — set the foliage shading parameters for a material.
/// Only takes effect when shading_model == 1 (foliage).
#[no_mangle]
pub extern "C" fn bloom_set_material_foliage(
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

#[no_mangle]
pub extern "C" fn bloom_compile_material_from_file(
    path_ptr: *const u8,
    bucket_kind: f64,
) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let path = str_from_header(path_ptr);
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
}

/// EN-017 — compile + install a fullscreen post-pass material.
/// See `bloom-macos::bloom_set_post_pass` for the full ABI.
#[no_mangle]
pub extern "C" fn bloom_set_post_pass(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.set_post_pass(source) {
        Ok(()) => 1.0,
        Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
    }
}

/// EN-017 — uninstall the active post-pass.
#[no_mangle]
pub extern "C" fn bloom_clear_post_pass() {
    engine().renderer.clear_post_pass();
}

/// EN-017 V2 — append a post-pass to the stack.
/// See `bloom-macos::bloom_add_post_pass` for the full ABI.
#[no_mangle]
pub extern "C" fn bloom_add_post_pass(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.add_post_pass(source) {
        Ok(h) => h as f64,
        Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
    }
}

/// EN-017 V2 — wipe the entire post-pass stack.
#[no_mangle]
pub extern "C" fn bloom_clear_all_post_passes() {
    engine().renderer.clear_all_post_passes();
}

#[no_mangle]
pub extern "C" fn bloom_draw_material(
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

#[no_mangle]
pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => engine().models.load_model_animation(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64, scale: f64, px: f64, py: f64, pz: f64, rot_sin: f64, rot_cos: f64) {
    let eng = engine();
    eng.models.update_model_animation(handle, anim_index as usize, time as f32);
    if let Some(anim) = eng.models.get_animation(handle) {
        if !anim.joint_matrices.is_empty() {
            eng.renderer.set_joint_matrices_scaled(&anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32], rot_sin as f32, rot_cos as f32);
        }
    }
}

// --- Music FFI ---

#[no_mangle]
pub extern "C" fn bloom_load_music(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            if let Some(s) = parse_ogg(&data) { engine().audio.load_music(s) }
            else if let Some(s) = parse_wav(&data) { engine().audio.load_music(s) }
            else if let Some(s) = parse_mp3(&data) { engine().audio.load_music(s) }
            else { 0.0 }
        }
        Err(_) => 0.0,
    }
}
#[no_mangle]
pub extern "C" fn bloom_play_music(handle: f64) { engine().audio.play_music(handle); }
#[no_mangle]
pub extern "C" fn bloom_stop_music(handle: f64) { engine().audio.stop_music(handle); }
#[no_mangle]
pub extern "C" fn bloom_update_music_stream(handle: f64) { engine().audio.update_music_stream(handle); }
#[no_mangle]
pub extern "C" fn bloom_set_music_volume(handle: f64, volume: f64) { engine().audio.set_music_volume(handle, volume as f32); }
#[no_mangle]
pub extern "C" fn bloom_is_music_playing(handle: f64) -> f64 { if engine().audio.is_music_playing(handle) { 1.0 } else { 0.0 } }

// --- Gamepad FFI ---

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_available() -> f64 { if engine().input.is_gamepad_available() { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis(axis: f64) -> f64 { engine().input.get_gamepad_axis(axis as usize) as f64 }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_pressed(btn: f64) -> f64 { if engine().input.is_gamepad_button_pressed(btn as usize) { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_down(btn: f64) -> f64 { if engine().input.is_gamepad_button_down(btn as usize) { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_released(btn: f64) -> f64 { if engine().input.is_gamepad_button_released(btn as usize) { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis_count() -> f64 { engine().input.get_gamepad_axis_count() as f64 }

// --- Skeletal Animation Debug ---

#[no_mangle]
pub extern "C" fn bloom_set_joint_test(_joint: f64, _angle: f64) {
    // No-op for now — skeletal animation testing
}

// --- Lighting ---

#[no_mangle]
pub extern "C" fn bloom_set_ambient_light(r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_ambient_light(r, g, b, intensity);
}

#[no_mangle]
pub extern "C" fn bloom_set_directional_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_directional_light(dx, dy, dz, r, g, b, intensity);
}

#[no_mangle]
pub extern "C" fn bloom_set_procedural_sky(enabled: f64, rayleigh_density: f64, mie_density: f64, ground_albedo: f64) {
    engine().renderer.set_procedural_sky(
        enabled != 0.0,
        rayleigh_density as f32,
        mie_density as f32,
        ground_albedo as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_set_sun_direction(dx: f64, dy: f64, dz: f64, intensity: f64) {
    engine().renderer.set_sun_direction(dx as f32, dy as f32, dz as f32, intensity as f32);
}

// --- Utility FFI ---

#[no_mangle]
pub extern "C" fn bloom_toggle_fullscreen() {}
#[no_mangle]
pub extern "C" fn bloom_set_window_title(title_ptr: *const u8) { let _ = str_from_header(title_ptr); }
#[no_mangle]
pub extern "C" fn bloom_set_window_icon(path_ptr: *const u8) { let _ = str_from_header(path_ptr); }

#[no_mangle]
pub extern "C" fn bloom_disable_cursor() {
    engine().input.cursor_disabled = true;
}

#[no_mangle]
pub extern "C" fn bloom_enable_cursor() {
    engine().input.cursor_disabled = false;
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_delta_x() -> f64 {
    engine().input.mouse_delta_x
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_delta_y() -> f64 {
    engine().input.mouse_delta_y
}

// Accumulated scroll wheel delta since the last call. Reading consumes the
// value (returns 0 on the next call until the user scrolls again). Used by
// the editor's orbit camera and any scrollable UI panel.
#[no_mangle]
pub extern "C" fn bloom_get_mouse_wheel() -> f64 {
    engine().input.consume_mouse_wheel()
}

#[no_mangle]
pub extern "C" fn bloom_get_char_pressed() -> f64 {
    engine().input.pop_char() as f64
}

// Q2: Cursor shape
#[no_mangle]
pub extern "C" fn bloom_set_cursor_shape(shape: f64) {
    engine().input.cursor_shape = shape as u32;
}

// E4: Clipboard (stub on this platform)
#[no_mangle]
pub extern "C" fn bloom_set_clipboard_text(_text_ptr: *const u8) {}
#[no_mangle]
pub extern "C" fn bloom_get_clipboard_text() -> *const u8 { std::ptr::null() }

// E5b: File dialogs (stub on this platform)
#[no_mangle]
pub extern "C" fn bloom_open_file_dialog(_filter_ptr: *const u8, _title_ptr: *const u8) -> *const u8 { std::ptr::null() }
#[no_mangle]
pub extern "C" fn bloom_save_file_dialog(_default_name_ptr: *const u8, _title_ptr: *const u8) -> *const u8 { std::ptr::null() }

// Model bounds accessors. Return the axis-aligned bounding box of a loaded
// model in model-local coordinates. Editors use these to size gizmos, auto-
// frame the camera on selection, and snap placed entities onto terrain.
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[0] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[1] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[2] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[0] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[1] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[2] as f64
}

#[no_mangle]
pub extern "C" fn bloom_write_file(path_ptr: *const u8, data_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = str_from_header(data_ptr);
    match std::fs::write(path, data.as_bytes()) {
        Ok(_) => 1.0,
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_file_exists(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let resolved = resolve_path(path);
    if std::path::Path::new(&resolved).exists() { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_read_file(path_ptr: *const u8) -> *const u8 {
    let path = str_from_header(path_ptr);
    // Always return a valid Perry string. A null pointer would NaN-box into a
    // string-typed value pointing at address 0; subsequent `.length` /
    // `.charCodeAt` reads dereference the bogus StringHeader and segfault.
    // Callers detect "missing file" via `data.length === 0` (e.g. the jump
    // game's discoverLevels probe across level1..level10 / custom_*).
    // Parity with native/linux — Android previously returned null on Err,
    // crashing discoverLevels at the first non-existent level file.
    match std::fs::read_to_string(resolve_path(path)) {
        Ok(contents) => alloc_perry_string(&contents),
        Err(_)       => alloc_perry_string(""),
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_x(index: f64) -> f64 { engine().input.get_touch_x(index as usize) }
#[no_mangle]
pub extern "C" fn bloom_get_touch_y(index: f64) -> f64 { engine().input.get_touch_y(index as usize) }
#[no_mangle]
pub extern "C" fn bloom_get_touch_count() -> f64 { engine().input.get_touch_count() as f64 }
#[no_mangle]
pub extern "C" fn bloom_get_time() -> f64 { engine().get_time() }

// Input injection + platform detection
#[no_mangle]
pub extern "C" fn bloom_inject_key_down(key: f64) {
    engine().input.set_key_down(key as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_key_up(key: f64) {
    engine().input.set_key_up(key as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_axis(axis: f64, value: f64) {
    engine().input.set_gamepad_axis(axis as usize, value as f32);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_down(button: f64) {
    engine().input.set_gamepad_button_down(button as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_up(button: f64) {
    engine().input.set_gamepad_button_up(button as usize);
}
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 5.0 }

/// Preferred OS language packed as `c0*256+c1` (ISO-639 primary subtag), read
/// from the device locale system property via the NDK (no JNI). Tries the
/// user-set locale first, then the factory defaults. Falls back to "en".
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    fn parse(buf: &[u8], n: i32) -> Option<f64> {
        if n < 2 { return None; }
        let lc = |b: u8| if b.is_ascii_uppercase() { b + 32 } else { b };
        let (c0, c1) = (lc(buf[0]), lc(buf[1]));
        if c0.is_ascii_alphabetic() && c1.is_ascii_alphabetic() {
            Some((c0 as f64) * 256.0 + (c1 as f64))
        } else {
            None
        }
    }
    let props: [&[u8]; 3] = [
        b"persist.sys.locale\0",
        b"ro.product.locale\0",
        b"ro.product.locale.language\0",
    ];
    for prop in props {
        let mut buf = [0u8; 92]; // PROP_VALUE_MAX
        let n = unsafe {
            libc::__system_property_get(
                prop.as_ptr() as *const libc::c_char,
                buf.as_mut_ptr() as *mut libc::c_char,
            )
        };
        if let Some(v) = parse(&buf, n) { return v; }
    }
    25966.0
}
#[no_mangle]
pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
    if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
}
#[no_mangle]
pub extern "C" fn bloom_get_crown_rotation() -> f64 {
    engine().input.consume_crown_rotation()
}

// ============================================================
// JNI Bridge for Bloom game applications
// ============================================================
//
// These functions bridge the Android Java/Kotlin layer to the
// Bloom engine. Any Bloom game on Android should use the
// com.bloomengine.game.BloomGameBridge Kotlin class.

extern "C" {
    fn ANativeWindow_fromSurface(env: *mut libc::c_void, surface: *mut libc::c_void) -> *mut libc::c_void;
    fn mallopt(param: i32, value: i32) -> i32;
    fn __android_log_print(prio: i32, tag: *const u8, fmt: *const u8, ...) -> i32;
    fn main() -> i32;
}

/// JNI_OnLoad: called when System.loadLibrary() loads this .so.
/// Disables MTE heap tagging (required for Perry NaN-boxing) and
/// reads the asset base path from BLOOM_ASSET_PATH env var.
#[no_mangle]
pub extern "C" fn JNI_OnLoad(_vm: *mut libc::c_void, _reserved: *mut libc::c_void) -> i32 {
    unsafe {
        // Disable MTE heap tagging for Perry NaN-boxing compatibility.
        // Perry uses 48-bit pointers; Android's scudo allocator may tag
        // the top byte, corrupting NaN-boxed pointer values.
        mallopt(-204, 0);

        __android_log_print(
            3, b"BloomEngine\0".as_ptr(),
            b"JNI_OnLoad: MTE disabled\0".as_ptr(),
        );
    }

    // Read asset base path from environment (set by Activity before loadLibrary)
    if let Ok(path) = std::env::var("BLOOM_ASSET_PATH") {
        unsafe {
            __android_log_print(
                3, b"BloomEngine\0".as_ptr(),
                b"JNI_OnLoad: asset path set\0".as_ptr(),
            );
            ASSET_BASE_PATH = Some(path);
        }
    }

    0x00010006 // JNI_VERSION_1_6
}

/// Pass the Android Surface to the engine so it can create a wgpu rendering surface.
/// Called from BloomGameBridge.nativeSetSurface(surface).
#[no_mangle]
pub unsafe extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeSetSurface(
    env: *mut libc::c_void,
    _class: *mut libc::c_void,
    surface: *mut libc::c_void,
) {
    let window = ANativeWindow_fromSurface(env, surface);
    __android_log_print(
        3, b"BloomEngine\0".as_ptr(),
        b"nativeSetSurface: ANativeWindow acquired\0".as_ptr(),
    );
    bloom_android_set_native_window(window);
}

/// Run the compiled game's main() function on the game thread.
/// Called from BloomGameBridge.nativeMain().
#[no_mangle]
pub unsafe extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeMain(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
) {
    __android_log_print(
        3, b"BloomEngine\0".as_ptr(),
        b"nativeMain: calling main()\0".as_ptr(),
    );
    main();
    __android_log_print(
        3, b"BloomEngine\0".as_ptr(),
        b"nativeMain: main() returned\0".as_ptr(),
    );
}

/// Forward touch events from the Android UI thread to the engine's input system.
/// Called from BloomGameBridge.nativeOnTouch(action, x, y, pointerIndex).
#[no_mangle]
pub extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeOnTouch(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
    action: i32,
    x: f64,
    y: f64,
    pointer_index: i32,
) {
    bloom_android_on_touch(action, x, y, pointer_index);
}

/// Signal the engine to close when the Activity is destroyed.
/// Called from BloomGameBridge.nativeOnDestroy().
#[no_mangle]
pub extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeOnDestroy(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
) {
    unsafe {
        if let Some(eng) = ENGINE.get_mut() {
            eng.should_close = true;
        }
    }
}

// ============================================================
// Thread-safe staging (for async asset loading via Perry threads)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_stage_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => bloom_shared::staging::decode_and_stage_texture(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_stage_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = match std::fs::read(resolve_path(path)) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    match bloom_shared::models::load_gltf_staged(&data) {
        Some(staged) => bloom_shared::staging::stage_model(staged),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_stage_sound(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = match std::fs::read(resolve_path(path)) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    let sound_data = if path.ends_with(".ogg") || path.ends_with(".OGG") {
        parse_ogg(&data)
    } else if path.ends_with(".mp3") || path.ends_with(".MP3") {
        parse_mp3(&data)
    } else {
        parse_wav(&data)
    };
    match sound_data {
        Some(sd) => bloom_shared::staging::stage_sound(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_commit_texture(staging_handle: f64) -> f64 {
    let staged = match bloom_shared::staging::take_texture(staging_handle) {
        Some(s) => s,
        None => return 0.0,
    };
    let eng = engine();
    let bind_group_idx = eng.renderer.register_texture(staged.width, staged.height, &staged.data);
    eng.textures.textures.alloc(bloom_shared::textures::TextureData {
        bind_group_idx, width: staged.width, height: staged.height,
    })
}

#[no_mangle]
pub extern "C" fn bloom_commit_model(staging_handle: f64) -> f64 {
    let staged = match bloom_shared::staging::take_model(staging_handle) {
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
}

#[no_mangle]
pub extern "C" fn bloom_commit_sound(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_sound(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_commit_music(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_music(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_run_game(_callback: extern "C" fn(f64)) {
    // No-op on native. The TypeScript runGame() helper provides the while loop.
}



// Q6: Multi-hit picking
static mut LAST_PICK_ALL: Vec<bloom_shared::picking::PickResult> = Vec::new();

#[no_mangle]
pub extern "C" fn bloom_scene_pick_all(screen_x: f64, screen_y: f64, max_results: f64) -> f64 {
    let eng = engine();
    let inv_vp = eng.renderer.inverse_vp_matrix();
    let cam_pos = eng.renderer.camera_pos();
    let w = eng.renderer.width() as f32;
    let h = eng.renderer.height() as f32;
    let (origin, direction) = bloom_shared::picking::screen_to_ray(
        screen_x as f32, screen_y as f32, w, h, &inv_vp, &cam_pos,
    );
    let results = bloom_shared::picking::raycast_scene_all(&eng.scene, &origin, &direction, max_results as usize);
    let count = results.len();
    unsafe { LAST_PICK_ALL = results; }
    count as f64
}
#[no_mangle]
pub extern "C" fn bloom_pick_all_handle(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.handle).unwrap_or(0.0) }
}
#[no_mangle]
pub extern "C" fn bloom_pick_all_distance(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.distance as f64).unwrap_or(0.0) }
}
// ============================================================

#[no_mangle] pub extern "C" fn bloom_take_screenshot(_path_ptr: *const u8) {}
#[no_mangle] pub extern "C" fn bloom_set_env_clear_from_hdr(_path_ptr: *const u8) {}
#[no_mangle] pub extern "C" fn bloom_set_fog(_r: f64, _g: f64, _b: f64, _density: f64, _height_ref: f64, _height_falloff: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_chromatic_aberration(_strength: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_vignette(_strength: f64, _softness: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_film_grain(_strength: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_sun_shafts(_strength: f64, _decay: f64, _r: f64, _g: f64, _b: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_auto_exposure(_on: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_taa_enabled(_on: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_render_scale(_scale: f64) {}
#[no_mangle] pub extern "C" fn bloom_get_render_scale() -> f64 { 1.0 }
#[no_mangle] pub extern "C" fn bloom_set_upscale_mode(_mode: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_cas_strength(_strength: f64) {}
#[no_mangle] pub extern "C" fn bloom_get_physical_width() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_get_physical_height() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_set_auto_resolution(_target_hz: f64, _enabled: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_manual_exposure(_value: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_env_intensity(_intensity: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_ssgi_enabled(_enabled: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_ssgi_intensity(_intensity: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_ssgi_radius(_radius: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_dof(_enabled: f64, _focus_distance: f64, _aperture: f64) {}
// Ticket 011: real quality / profiler implementations. Prior build had
// no-op stubs — TS games calling setQualityPreset / setProfilerEnabled etc.
// linked fine but did nothing at runtime on Android.
#[no_mangle] pub extern "C" fn bloom_set_quality_preset(preset: f64) {
    engine().renderer.apply_quality_preset(preset as u32);
}
#[no_mangle] pub extern "C" fn bloom_set_shadows_enabled(on: f64) {
    engine().renderer.set_shadows_enabled(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_set_shadows_always_fresh(on: f64) {
    engine().renderer.set_shadows_always_fresh(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_set_bloom_enabled(on: f64) {
    engine().renderer.set_bloom_enabled(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_set_early_z_enabled(_on: f64) {}
#[no_mangle] pub extern "C" fn bloom_set_ssao_enabled(on: f64) {
    engine().renderer.set_ssao_enabled(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_set_ssao_intensity(value: f64) {
    engine().renderer.set_ssao_strength(value as f32);
}
#[no_mangle] pub extern "C" fn bloom_set_ssao_radius(world_radius: f64) {
    engine().renderer.set_ssao_radius(world_radius as f32);
}
#[no_mangle] pub extern "C" fn bloom_set_wind(dir_x: f64, dir_z: f64, amplitude: f64, frequency: f64) {
    engine().renderer.set_wind(dir_x as f32, dir_z as f32, amplitude as f32, frequency as f32);
}
#[no_mangle] pub extern "C" fn bloom_set_ssr_enabled(on: f64) {
    engine().renderer.set_ssr_enabled(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_set_motion_blur_enabled(on: f64) {
    engine().renderer.set_motion_blur_enabled(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_set_sss_enabled(on: f64) {
    engine().renderer.set_sss_enabled(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_set_profiler_enabled(on: f64) {
    engine().profiler.set_enabled(on != 0.0);
}
#[no_mangle] pub extern "C" fn bloom_get_profiler_frame_cpu_us() -> f64 {
    engine().profiler.avg_frame_cpu_us()
}
#[no_mangle] pub extern "C" fn bloom_get_profiler_frame_gpu_us() -> f64 {
    engine().profiler.avg_frame_gpu_us()
}
#[no_mangle] pub extern "C" fn bloom_print_profiler_summary() {
    // Android has no stdout — log the summary via android_log so
    // `adb logcat` picks it up alongside the rest of the engine log.
    // %s + ptr variant so `%` characters in the summary (none today,
    // but cheap safety) aren't interpreted as format specifiers.
    let summary = engine().profiler.summary();
    if let Ok(c) = std::ffi::CString::new(summary) {
        unsafe {
            __android_log_print(
                4,
                b"BloomEngine\0".as_ptr(),
                b"%s\0".as_ptr(),
                c.as_ptr(),
            );
        }
    }
}

// ============================================================
// Physics (Jolt 5.x) — FFI surface generated from shared macro
// ============================================================

#[cfg(feature = "jolt")]
#[inline]
fn bloom_jolt_ffi_physics() -> &'static mut bloom_shared::physics_jolt::JoltPhysics {
    &mut engine().jolt
}

#[cfg(feature = "jolt")]
bloom_shared::define_physics_ffi!();

// === Android FFI parity: ported from native/linux/src/lib.rs (shared renderer/scene) ===
// Backing statics for the ported pick/project FFI (mirror native/linux).
static mut LAST_PROJECT: (f64, f64) = (0.0, 0.0);
static mut LAST_PICK: Option<bloom_shared::picking::PickResult> = None;

#[no_mangle]
pub extern "C" fn bloom_add_directional_light(
    dx: f64, dy: f64, dz: f64,
    r: f64, g: f64, b: f64,
    intensity: f64,
) {
    engine().renderer.add_directional_light(
        dx as f32, dy as f32, dz as f32,
        r as f32, g as f32, b as f32,
        intensity as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_add_point_light(
    x: f64, y: f64, z: f64, range: f64,
    r: f64, g: f64, b: f64,
    intensity: f64,
) {
    engine().renderer.add_point_light(
        x as f32, y as f32, z as f32, range as f32,
        r as f32, g as f32, b as f32,
        intensity as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_begin_texture_mode(_handle: f64) {
    // Stub: no-op until GPU render-to-texture is wired.
}

#[no_mangle]
pub extern "C" fn bloom_disable_postfx() {
    engine().postfx = None;
}

#[no_mangle]
pub extern "C" fn bloom_disable_shadows() {
    engine().renderer.shadow_map.disable();
}

#[no_mangle]
pub extern "C" fn bloom_dump_shadow_map(path_ptr: *const u8) {
    let path = str_from_header(path_ptr).to_string();
    engine().renderer.dump_shadow_map(&path);
}

#[no_mangle]
pub extern "C" fn bloom_enable_postfx() {
    let eng = engine();
    let w = eng.renderer.width();
    let h = eng.renderer.height();
    let fmt = eng.renderer.surface_format();
    eng.postfx = Some(bloom_shared::postfx::PostFxPipeline::new(
        &eng.renderer.device, w, h, fmt,
    ));
}

#[no_mangle]
pub extern "C" fn bloom_enable_shadows() {
    engine().renderer.shadow_map.enable();
}

#[no_mangle]
pub extern "C" fn bloom_end_texture_mode() {
    // Stub: no-op.
}

// Q9: Generate a ribbon mesh along a Catmull-Rom spline.
#[no_mangle]
pub extern "C" fn bloom_gen_mesh_spline_ribbon(points_ptr: *const u8, point_count: f64, widths_ptr: *const u8, width_count: f64) -> f64 {
    let n = point_count as usize;
    let wn = width_count as usize;
    let points = unsafe { std::slice::from_raw_parts(points_ptr as *const f32, n * 3) };
    let widths = unsafe { std::slice::from_raw_parts(widths_ptr as *const f32, wn) };
    engine().models.gen_mesh_spline_ribbon(points, widths)
}

#[no_mangle]
pub extern "C" fn bloom_get_render_texture_texture(handle: f64) -> f64 {
    engine().textures.get_render_texture_texture(handle)
}

// Q1: Render texture FFI (stub — GPU implementation deferred).
#[no_mangle]
pub extern "C" fn bloom_load_render_texture(width: f64, height: f64) -> f64 {
    engine().textures.load_render_texture(width as u32, height as u32)
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_distance() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.distance as f64).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_handle() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.handle).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_normal_x() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.normal[0] as f64).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_normal_y() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.normal[1] as f64).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_normal_z() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.normal[2] as f64).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_x() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.point[0] as f64).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_y() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.point[1] as f64).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_z() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.point[2] as f64).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_hovered(handle: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.set_hovered(handle);
    }
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_outline_color(r: f64, g: f64, b: f64, a: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.outline_params.color_selected = [r as f32, g as f32, b as f32, a as f32];
    }
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_outline_thickness(thickness: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.outline_params.thickness[0] = thickness as f32;
    }
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_selected(handle: f64) {
    if let Some(pfx) = &mut engine().postfx {
        if handle == 0.0 {
            pfx.set_selected(Vec::new());
        } else {
            pfx.set_selected(vec![handle]);
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_profiler_frame_history() -> *const u8 {
    let hist = engine().profiler.frame_history();
    let mut s = String::with_capacity(hist.len() * 24);
    for (cpu, gpu) in &hist {
        s.push_str(&format!("{:.2}|{:.2}\n", cpu, gpu));
    }
    alloc_perry_string(&s)
}

#[no_mangle]
pub extern "C" fn bloom_profiler_overlay_text() -> *const u8 {
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
    alloc_perry_string(&s)
}

#[no_mangle]
pub extern "C" fn bloom_project_screen_y() -> f64 {
    unsafe { LAST_PROJECT.1 }
}

#[no_mangle]
pub extern "C" fn bloom_project_to_screen(wx: f64, wy: f64, wz: f64) -> f64 {
    let eng = engine();
    let vp = eng.renderer.vp_matrix();
    let w = eng.renderer.width() as f32;
    let h = eng.renderer.height() as f32;

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

    let ndc_x = clip_x / clip_w;
    let ndc_y = clip_y / clip_w;
    let screen_x = ((ndc_x + 1.0) * 0.5 * w) as f64;
    let screen_y = ((1.0 - ndc_y) * 0.5 * h) as f64;

    unsafe { LAST_PROJECT = (screen_x, screen_y); }
    screen_x
}

#[no_mangle]
pub extern "C" fn bloom_register_frame_callback(priority: f64, callback: extern "C" fn(f64)) -> f64 {
    engine().frame_callbacks.register(priority as i32, callback) as f64
}

#[no_mangle]
pub extern "C" fn bloom_scene_attach_model(node_handle: f64, model_handle: f64, mesh_index: f64) {
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
}

#[no_mangle]
pub extern "C" fn bloom_scene_create_node() -> f64 {
    engine().scene.create_node()
}

#[no_mangle]
pub extern "C" fn bloom_scene_destroy_node(handle: f64) {
    engine().scene.destroy_node(handle);
}

#[no_mangle]
pub extern "C" fn bloom_scene_extrude_polygon(
    handle: f64,
    polygon_ptr: *const f64,
    polygon_count: f64,
    depth: f64,
) {
    if polygon_ptr.is_null() { return; }
    let n = polygon_count as usize;
    let polygon = unsafe { std::slice::from_raw_parts(polygon_ptr, n * 2) };

    let geo = bloom_shared::geometry::extrude_polygon(polygon, &[], depth);
    engine().scene.update_geometry(handle, geo.vertices, geo.indices);
}

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[0] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[1] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[2] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[0] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[1] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[2] as f64 }

// Scene graph QoL — Q4/Q5/Q6/Q7
#[no_mangle]
pub extern "C" fn bloom_scene_get_transform(handle: f64, index: f64) -> f64 {
    let mat = engine().scene.get_transform(handle);
    let i = index as usize;
    let col = i / 4;
    let row = i % 4;
    if col < 4 && row < 4 { mat[col][row] as f64 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_scene_get_user_data(handle: f64) -> f64 { engine().scene.get_user_data(handle) as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_node_count() -> f64 {
    engine().scene.node_count() as f64
}

#[no_mangle]
pub extern "C" fn bloom_scene_node_index_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.indices.len() as f64,
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_scene_node_vertex_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.vertices.len() as f64,
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_scene_pick(screen_x: f64, screen_y: f64) -> f64 {
    let eng = engine();
    let inv_vp = eng.renderer.inverse_vp_matrix();
    let cam_pos = eng.renderer.camera_pos();
    let w = eng.renderer.width() as f32;
    let h = eng.renderer.height() as f32;

    let (origin, direction) = bloom_shared::picking::screen_to_ray(
        screen_x as f32, screen_y as f32,
        w, h, &inv_vp, &cam_pos,
    );

    let result = bloom_shared::picking::raycast_scene(&eng.scene, &origin, &direction);
    let hit = result.hit;
    unsafe { LAST_PICK = Some(result); }
    if hit { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_cast_shadow(handle: f64, cast: f64) {
    engine().scene.set_cast_shadow(handle, cast != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_color(handle: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().scene.set_material_color(handle, r as f32, g as f32, b as f32, a as f32);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_pbr(handle: f64, roughness: f64, metalness: f64) {
    engine().scene.set_material_pbr(handle, roughness as f32, metalness as f32);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_texture(handle: f64, texture_idx: f64) {
    engine().scene.set_material_texture(handle, texture_idx as u32);
}

// Q8: Set a water material on a scene node (translucent tint, low roughness).
#[no_mangle]
pub extern "C" fn bloom_scene_set_material_water(handle: f64, wave_amp: f64, wave_speed: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().scene.set_material_water(handle, wave_amp as f32, wave_speed as f32, r as f32, g as f32, b as f32, a as f32);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_parent(handle: f64, parent: f64) {
    engine().scene.set_parent(handle, parent);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_receive_shadow(handle: f64, receive: f64) {
    engine().scene.set_receive_shadow(handle, receive != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_transform(handle: f64, mat_ptr: *const f64) {
    if mat_ptr.is_null() { return; }
    let slice = unsafe { std::slice::from_raw_parts(mat_ptr, 16) };
    let mut mat = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            mat[col][row] = slice[col * 4 + row] as f32;
        }
    }
    engine().scene.set_transform(handle, mat);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_user_data(handle: f64, data: f64) { engine().scene.set_user_data(handle, data as i64); }

#[no_mangle]
pub extern "C" fn bloom_scene_set_visible(handle: f64, visible: f64) {
    engine().scene.set_visible(handle, visible != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_subtract_box(
    handle: f64,
    min_x: f64, min_y: f64, min_z: f64,
    max_x: f64, max_y: f64, max_z: f64,
) {
    let eng = engine();
    if let Some(node) = eng.scene.nodes.get(handle) {
        let current = bloom_shared::geometry::GeometryData {
            vertices: node.vertices.clone(),
            indices: node.indices.clone(),
        };
        let result = bloom_shared::geometry::subtract_box(
            &current,
            [min_x as f32, min_y as f32, min_z as f32],
            [max_x as f32, max_y as f32, max_z as f32],
        );
        eng.scene.update_geometry(handle, result.vertices, result.indices);
    }
}

#[no_mangle]
pub extern "C" fn bloom_scene_update_geometry(
    handle: f64,
    vert_ptr: *const f64,
    vert_count: f64,
    idx_ptr: *const f64,
    idx_count: f64,
) {
    if vert_ptr.is_null() || idx_ptr.is_null() { return; }
    let nv = vert_count as usize;
    let ni = idx_count as usize;

    let vert_floats = unsafe { std::slice::from_raw_parts(vert_ptr, nv * 12) };
    let idx_floats = unsafe { std::slice::from_raw_parts(idx_ptr, ni) };

    let mut vertices = Vec::with_capacity(nv);
    for i in 0..nv {
        let base = i * 12;
        vertices.push(bloom_shared::renderer::Vertex3D {
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
}

#[no_mangle]
pub extern "C" fn bloom_splat_impulse(x: f64, z: f64, radius: f64, strength: f64) {
    engine().renderer.impulse_field.submit_splat(
        x as f32, z as f32, radius as f32, strength as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_unload_render_texture(handle: f64) {
    engine().textures.unload_render_texture(handle);
}

#[no_mangle]
pub extern "C" fn bloom_unregister_frame_callback(id: f64) {
    engine().frame_callbacks.unregister(id as u64);
}
