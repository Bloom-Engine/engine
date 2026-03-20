use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::str_from_header;
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
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8) {
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

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            ..Default::default()
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
            raw_display_handle: raw_window_handle::RawDisplayHandle::Android(
                raw_window_handle::AndroidDisplayHandle::new()
            ),
            raw_window_handle: raw,
        }).expect("Failed to create surface");
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: surface created\0".as_ptr());

        let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }));
        let adapter = match adapter {
            Some(a) => a,
            None => {
                // Try again without surface compatibility requirement
                match pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })) {
                    Some(a) => a,
                    None => panic!("No GPU adapter found"),
                }
            }
        };
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"bloom_init_window: adapter found\0".as_ptr());

        let (device, queue) = pollster_block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("bloom_device"),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            },
            None,
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
        let renderer = Renderer::new(device, queue, surface, surface_config);
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
            eng.input.set_mouse_position(x, y);
            eng.input.set_touch(pointer_index as usize, x, y, action != 1); // 1 = ACTION_UP
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
    engine().end_frame();
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
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::new());
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
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::new());
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

#[no_mangle]
pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => engine().models.load_model_animation(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64, scale: f64, px: f64, py: f64, pz: f64) {
    let eng = engine();
    eng.models.update_model_animation(handle, anim_index as usize, time as f32);
    if let Some(anim) = eng.models.get_animation(handle) {
        if !anim.joint_matrices.is_empty() {
            eng.renderer.set_joint_matrices_scaled(&anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32]);
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
    match std::fs::read_to_string(resolve_path(path)) {
        Ok(contents) => {
            let c_str = std::ffi::CString::new(contents).unwrap_or_default();
            c_str.into_raw() as *const u8
        }
        Err(_) => std::ptr::null(),
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
#[no_mangle]
pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
    if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
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
