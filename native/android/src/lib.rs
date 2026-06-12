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
/// Asset-path hook for define_core_ffi! — routes through this platform's
/// resolve_path (relative paths don't resolve from the app working dir here).
fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Owned(resolve_path(path))
}

// The full shared (non-physics) FFI surface. See bloom_shared::ffi_core
// docs for the contract; tools/validate-ffi.js checks parity in CI.
bloom_shared::define_core_ffi!();


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

// --- Texture FFI ---

// --- Camera FFI ---

// --- 3D Drawing FFI ---

// --- Model FFI ---

// ============================================================
// Phase 1c — material system FFI
// ============================================================

// --- Music FFI ---

// --- Gamepad FFI ---

// --- Skeletal Animation Debug ---

// --- Lighting ---

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



// Q6: Multi-hit picking
// ============================================================

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
