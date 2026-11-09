use bloom_shared::audio::{parse_mp3, parse_ogg, parse_wav};
use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::{alloc_perry_string, str_from_header};

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut NATIVE_WINDOW: *mut libc::c_void = std::ptr::null_mut();
static AUDIO_RUNNING: AtomicBool = AtomicBool::new(false);
/// The app is in the background: the stream is paused and nothing is mixed, so
/// music picks up where it was when the app comes back.
static AUDIO_PAUSED: AtomicBool = AtomicBool::new(false);
static mut ASSET_BASE_PATH: Option<String> = None;

/// Device pixels, as on iOS: set once the Activity reports its display metrics
/// (`nativeSetDisplayMetrics`). The screen size the game sees is then the
/// surface's pixel size and `getDisplayScale()` is the display density.
/// Without it (an Activity that doesn't report them) the screen is half the
/// surface's size, the layout Bloom Jump was made for.
static NATIVE_PIXELS: AtomicBool = AtomicBool::new(false);
/// The surface's new pixel size (`w << 32 | h`) from `surfaceChanged`, applied
/// by the game thread at the start of its next frame; 0 = none.
static PENDING_SIZE: AtomicU64 = AtomicU64::new(0);
/// The surface's current pixel size (`w << 32 | h`).
static SURFACE_SIZE: AtomicU64 = AtomicU64::new(0);

fn pack_size(w: u32, h: u32) -> u64 {
    (w as u64) << 32 | h as u64
}

/// Resize the renderer to a surface of `w` x `h` pixels (game thread).
fn resize_to(eng: &mut EngineState, w: u32, h: u32) {
    let (lw, lh) = logical_size(w, h);
    eng.renderer.resize(w, h, lw, lh);
    SURFACE_SIZE.store(pack_size(w, h), Ordering::SeqCst);
}
/// Back button: 1 = press Escape next frame, 2 = release it the frame after.
static BACK_STATE: AtomicU8 = AtomicU8::new(0);
/// The wgpu instance the device came from: a new surface (after the app comes
/// back from the background) must be created from the same one.
static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
/// Set once bloom_init_window has created the engine.
static ENGINE_READY: AtomicBool = AtomicBool::new(false);

/// The surface's lifecycle between the UI thread (SurfaceHolder callbacks) and
/// the game thread. When the app goes to the background Android destroys the
/// surface: the game thread drops its wgpu surface (`dropped`) before the UI
/// thread lets the window go, then waits until a new window arrives.
struct SurfaceLife {
    lost: bool,
    dropped: bool,
    window: usize,
}
static LIFE: Mutex<SurfaceLife> = Mutex::new(SurfaceLife {
    lost: false,
    dropped: false,
    window: 0,
});
static LIFE_CHANGED: Condvar = Condvar::new();

/// The logical (game-visible) size for a surface of `w` x `h` pixels.
fn logical_size(w: u32, h: u32) -> (u32, u32) {
    if NATIVE_PIXELS.load(Ordering::Relaxed) {
        (w.max(1), h.max(1))
    } else {
        ((w / 2).max(1), (h / 2).max(1))
    }
}

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
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }
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

/// SH-055 diag — read an Android system property (empty string when unset).
/// Debug levers settable from adb with no rebuild:
///   adb shell setprop debug.bloom.backend gl|vulkan   (wgpu backend A/B)
///   adb shell setprop debug.bloom.skipsky 1           (skip the sky draw)
/// then force-stop + relaunch the app.
fn sysprop(name: &str) -> String {
    let cname = std::ffi::CString::new(name).unwrap();
    let mut buf = [0u8; 92]; // PROP_VALUE_MAX
    let n = unsafe {
        libc::__system_property_get(cname.as_ptr(), buf.as_mut_ptr() as *mut libc::c_char)
    };
    if n <= 0 {
        return String::new();
    }
    String::from_utf8_lossy(&buf[..n as usize])
        .trim()
        .to_string()
}

/// SH-055 diag — wgpu backend selection for the A/B experiment. Default keeps
/// the shipped behavior (both allowed, wgpu prefers Vulkan).
fn backend_choice() -> wgpu::Backends {
    match sysprop("debug.bloom.backend").as_str() {
        "gl" => wgpu::Backends::GL,
        "vulkan" => wgpu::Backends::VULKAN,
        _ => wgpu::Backends::VULKAN | wgpu::Backends::GL,
    }
}

#[no_mangle]
pub extern "C" fn bloom_init_window(
    width: f64,
    height: f64,
    title_ptr: *const u8,
    _fullscreen: f64,
) {
    let _title = str_from_header(title_ptr);

    // Re-install after perry-runtime's own hook (set during main() init) so the
    // upcoming wgpu/naga material-compile panics log their location (SH-055).
    install_panic_logger();

    unsafe {
        __android_log_print(
            3,
            b"BloomEngine\0".as_ptr(),
            b"bloom_init_window: starting\0".as_ptr(),
        );
        let window = NATIVE_WINDOW;
        // If no native window was set, use requested dimensions with a headless surface
        let (pixel_w, pixel_h) = if !window.is_null() {
            (
                ANativeWindow_getWidth(window) as u32,
                ANativeWindow_getHeight(window) as u32,
            )
        } else {
            (width as u32, height as u32)
        };
        // Without reported display metrics the logical size is half of physical:
        // Bloom Jump's UI (fonts, layout constants) was sized for ~1170×540
        // landscape (iPhone-sized); on Android's 2340×1080 panel it would
        // otherwise look half-scale. wgpu still renders to the full physical
        // surface; only the `screen_width`/`screen_height` the game sees are
        // halved. With them (NATIVE_PIXELS) the game sees pixels, as on iOS.
        let (logical_w, logical_h) = logical_size(pixel_w, pixel_h);

        let backends = backend_choice();
        {
            let msg = std::ffi::CString::new(format!(
                "bloom_init_window: requested backends={backends:?}"
            ))
            .unwrap();
            __android_log_print(3, b"BloomEngine\0".as_ptr(), b"%s\0".as_ptr(), msg.as_ptr());
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        // Create surface from ANativeWindow
        if window.is_null() {
            // Fallback: can't render without a window, but don't panic
            // Create a minimal engine state (game logic will work, rendering will no-op)
            return;
        }

        let handle =
            raw_window_handle::AndroidNdkWindowHandle::new(std::ptr::NonNull::new(window).unwrap());
        let raw = raw_window_handle::RawWindowHandle::AndroidNdk(handle);
        let surface = instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Android(
                    raw_window_handle::AndroidDisplayHandle::new(),
                )),
                raw_window_handle: raw,
            })
            .expect("Failed to create surface");
        __android_log_print(
            3,
            b"BloomEngine\0".as_ptr(),
            b"bloom_init_window: surface created\0".as_ptr(),
        );

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
        __android_log_print(
            3,
            b"BloomEngine\0".as_ptr(),
            b"bloom_init_window: adapter found\0".as_ptr(),
        );
        {
            let info = adapter.get_info();
            let msg = std::ffi::CString::new(format!(
                "adapter: name='{}' backend={:?} device_type={:?} driver='{}' driver_info='{}'",
                info.name, info.backend, info.device_type, info.driver, info.driver_info
            ))
            .unwrap();
            __android_log_print(3, b"BloomEngine\0".as_ptr(), b"%s\0".as_ptr(), msg.as_ptr());
        }

        let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let negotiated = pollster_block_on(
            bloom_shared::renderer::device_negotiation::request_device_with_fallback(
                &adapter,
                bloom_shared::renderer::device_negotiation::DeviceRequestOptions {
                    allow_ray_query: !force_sw_gi,
                    profile: bloom_shared::renderer::device_negotiation::DeviceRequestProfile::FoldedMobile,
                },
            ),
        )
        .unwrap_or_else(|error| panic!("Failed to create renderer device: {error}"));
        let negotiation_report = negotiated.report.report_json();
        let msg = std::ffi::CString::new(format!(
            "renderer device negotiation = {negotiation_report}"
        ))
        .unwrap();
        __android_log_print(3, b"BloomEngine\0".as_ptr(), b"%s\0".as_ptr(), msg.as_ptr());
        let device = negotiated.device;
        let queue = negotiated.queue;
        __android_log_print(
            3,
            b"BloomEngine\0".as_ptr(),
            b"bloom_init_window: device created\0".as_ptr(),
        );

        // SH-055 diag — capture wgpu validation/device errors with their full
        // detail instead of wgpu's default "handle as fatal" panic (which the
        // FFI guard swallows as "non-string panic payload"). This is what tells
        // us WHY the instanced/from-file material pipelines fail on the Adreno
        // Vulkan path. Logs to logcat tag "BloomWGPU".
        device.on_uncaptured_error(std::sync::Arc::new(|err| {
            log::error!("wgpu uncaptured error: {err}");
        }));

        let surface_caps = surface.get_capabilities(&adapter);
        if surface_caps.formats.is_empty() {
            panic!("Surface reports no supported formats (emulator Vulkan limitation)");
        }
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
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

        __android_log_print(
            3,
            b"BloomEngine\0".as_ptr(),
            b"bloom_init_window: surface configured\0".as_ptr(),
        );
        let mut renderer =
            Renderer::new(device, queue, surface, surface_config, logical_w, logical_h);
        renderer.set_device_negotiation_report(negotiation_report);
        let _ = ENGINE.set(EngineState::new(renderer));
        let _ = INSTANCE.set(instance);
        SURFACE_SIZE.store(pack_size(pixel_w, pixel_h), Ordering::SeqCst);
        ENGINE_READY.store(true, Ordering::SeqCst);
        __android_log_print(
            3,
            b"BloomEngine\0".as_ptr(),
            b"bloom_init_window: engine initialized\0".as_ptr(),
        );
    }
}

/// Attach the engine to a host-owned `ANativeWindow*` instead of pulling
/// it from the global set by `bloom_android_set_native_window`
/// (PerryTS/perry#5519). `handle` is the `ANativeWindow*` the host
/// (Perry UI's `BloomView`, backed by a `SurfaceView`/`TextureView`)
/// owns; `width`/`height` are the surface size in physical pixels.
/// Returns 1.0 on success, 0.0 on a null/invalid handle or surface
/// bring-up failure. Idempotent once attached.
#[no_mangle]
pub extern "C" fn bloom_attach_native(handle: i64, width: f64, height: f64) -> f64 {
    if handle == 0 {
        return 0.0;
    }
    if unsafe { ENGINE.get() }.is_some() {
        return 1.0;
    }
    let window = handle as *mut libc::c_void;
    let Some(win_nn) = std::ptr::NonNull::new(window) else {
        return 0.0;
    };
    // Hold a reference for as long as the engine renders into it.
    unsafe {
        ANativeWindow_acquire(window);
        NATIVE_WINDOW = window;
    }
    let target = {
        let h = raw_window_handle::AndroidNdkWindowHandle::new(win_nn);
        wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Android(
                raw_window_handle::AndroidDisplayHandle::new(),
            )),
            raw_window_handle: raw_window_handle::RawWindowHandle::AndroidNdk(h),
        }
    };
    match unsafe {
        bloom_shared::attach::attach_engine(
            target,
            bloom_shared::attach::AttachParams {
                backends: backend_choice(),
                logical_w: (width as u32).max(1),
                logical_h: (height as u32).max(1),
                physical_w: (width as u32).max(1),
                physical_h: (height as u32).max(1),
                format: bloom_shared::attach::FormatPreference::Srgb,
            },
        )
    } {
        Ok(es) => {
            unsafe {
                let _ = ENGINE.set(es);
            }
            1.0
        }
        Err(_) => 0.0,
    }
}

/// Resize the engine's surface (#70 parity; used by host-driven
/// BloomViews on layout changes). `phys_*` physical px, `log_*` logical.
#[no_mangle]
pub extern "C" fn bloom_resize(phys_w: f64, phys_h: f64, log_w: f64, log_h: f64) {
    if let Some(eng) = unsafe { ENGINE.get_mut() } {
        eng.renderer
            .resize(phys_w as u32, phys_h as u32, log_w as u32, log_h as u32);
    }
}

/// HWND host-embed (#70) — Windows only; a no-op here for FFI-manifest
/// parity. Non-Windows hosts attach via `bloom_attach_native`.
#[no_mangle]
pub extern "C" fn bloom_attach_hwnd(_hwnd_bits: f64, _width: f64, _height: f64) {}

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
    if engine().should_close {
        1.0
    } else {
        0.0
    }
}

// ============================================================
// Touch input polling (called from Perry's Android event pump)
// ============================================================

/// Touches from the UI thread, applied by the game thread between frames
/// (`apply_touches`): (action, x, y, pointer id).
static TOUCHES: Mutex<Vec<(i32, f64, f64, i32)>> = Mutex::new(Vec::new());

/// Called by Perry's Android layer when a touch event occurs (UI thread). The
/// game thread reads its input state while the UI thread delivers touches, so
/// a touch is queued and applied between frames, as on iOS where touches
/// arrive on the game's own thread between frames. A down and up within one
/// frame then show as a touch for exactly one frame (`release_touch`) instead
/// of being released before the game read them.
#[no_mangle]
pub extern "C" fn bloom_android_on_touch(action: i32, x: f64, y: f64, pointer_index: i32) {
    if let Ok(mut q) = TOUCHES.lock() {
        q.push((action, x, y, pointer_index));
    }
}

/// Apply the queued touches (game thread), after `end_frame`: where iOS's
/// touches land, between one frame and the game's input read for the next.
/// A tap's down and up then show as one frame with the finger down, and the
/// release in `end_frame` after it.
fn apply_touches() {
    if !ENGINE_READY.load(Ordering::SeqCst) {
        return;
    }
    let touches = TOUCHES.lock().map(|mut q| std::mem::take(&mut *q)).unwrap_or_default();
    let eng = engine();
    for (action, x, y, id) in touches {
        apply_touch(eng, action, x, y, id);
    }
}

/// Apply one queued touch (game thread).
fn apply_touch(eng: &mut EngineState, action: i32, x: f64, y: f64, pointer_index: i32) {
    let k = if NATIVE_PIXELS.load(Ordering::Relaxed) { 1.0 } else { 0.5 };
    let lx = x * k;
    let ly = y * k;
    eng.input.set_mouse_position(lx, ly);
    if action == 1 || action == 3 {
        eng.input.release_touch(pointer_index as usize, lx, ly); // UP / CANCEL
    } else {
        eng.input.set_touch(pointer_index as usize, lx, ly, true); // DOWN / MOVE
    }
    match action {
        0 => eng.input.set_mouse_button_down(0), // ACTION_DOWN
        1 => eng.input.set_mouse_button_up(0),   // ACTION_UP
        _ => {}                                  // MOVE, CANCEL
    }
}

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    android_frame_sync();
    engine().begin_frame();
}

/// What the UI thread asked for since the last frame, applied on the game
/// thread before the frame starts: the Back button, a new surface size, and the
/// surface going away (the app in the background) and coming back.
fn android_frame_sync() {
    if !ENGINE_READY.load(Ordering::SeqCst) {
        return;
    }
    let eng = engine();
    match BACK_STATE.load(Ordering::SeqCst) {
        1 => {
            eng.input.inject_key_down(27); // Key.ESCAPE
            BACK_STATE.store(2, Ordering::SeqCst);
        }
        2 => {
            eng.input.inject_key_up(27);
            BACK_STATE.store(0, Ordering::SeqCst);
        }
        _ => {}
    }

    let mut life = LIFE.lock().unwrap();
    if life.lost {
        // Drop the wgpu surface while its window still exists, tell the UI
        // thread, then wait (frames, timers and audio decoding stop with us)
        // until a new window arrives or the Activity is destroyed.
        eng.renderer.surface = None;
        let size = SURFACE_SIZE.load(Ordering::SeqCst);
        resize_to(eng, (size >> 32) as u32, size as u32); // an offscreen target meanwhile
        life.dropped = true;
        AUDIO_PAUSED.store(true, Ordering::SeqCst);
        LIFE_CHANGED.notify_all();
        while life.lost && !eng.should_close {
            life = LIFE_CHANGED
                .wait_timeout(life, std::time::Duration::from_millis(250))
                .unwrap()
                .0;
        }
        life.dropped = false;
        AUDIO_PAUSED.store(false, Ordering::SeqCst);
        if !life.lost && life.window != 0 {
            let window = life.window as *mut libc::c_void;
            if let Some(instance) = INSTANCE.get() {
                let handle = raw_window_handle::AndroidNdkWindowHandle::new(
                    std::ptr::NonNull::new(window).unwrap(),
                );
                let surface = unsafe {
                    instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                        raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Android(
                            raw_window_handle::AndroidDisplayHandle::new(),
                        )),
                        raw_window_handle: raw_window_handle::RawWindowHandle::AndroidNdk(handle),
                    })
                };
                match surface {
                    Ok(s) => {
                        eng.renderer.surface = Some(s);
                        let (w, h) = unsafe {
                            (
                                ANativeWindow_getWidth(window) as u32,
                                ANativeWindow_getHeight(window) as u32,
                            )
                        };
                        resize_to(eng, w, h); // configures the new surface
                        PENDING_SIZE.store(0, Ordering::SeqCst);
                    }
                    Err(e) => log::error!("resume: create_surface failed: {e}"),
                }
            }
        }
    }
    drop(life);

    let size = PENDING_SIZE.swap(0, Ordering::SeqCst);
    if size != 0 {
        resize_to(eng, (size >> 32) as u32, size as u32);
    }
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine().end_frame();
        apply_touches();
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
    // started once: a second call would open a second output stream
    if AUDIO_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    // Move the render half into the audio thread; the engine keeps only
    // the command-producing control half.
    let renderer = engine().audio.take_renderer();
    std::thread::spawn(move || {
        android_audio_thread(renderer);
    });
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    AUDIO_RUNNING.store(false, Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
}

fn android_audio_thread(renderer: Option<bloom_shared::audio::AudioRenderer>) {
    // Use oboe (AAudio/OpenSL ES wrapper) for audio output
    use oboe::*;

    // The render half is owned by the oboe callback — AAudio invokes
    // on_audio_ready from its own realtime thread, and the callback never
    // touches shared engine state (see audio/mod.rs contract).
    struct BloomAudioCallback {
        renderer: Option<bloom_shared::audio::AudioRenderer>,
    }

    impl AudioOutputCallback for BloomAudioCallback {
        type FrameType = (f32, Stereo);

        fn on_audio_ready(
            &mut self,
            _stream: &mut dyn AudioOutputStreamSafe,
            frames: &mut [(f32, f32)],
        ) -> DataCallbackResult {
            // Convert frame tuples to interleaved slice
            let len = frames.len() * 2;
            let ptr = frames.as_mut_ptr() as *mut f32;
            let interleaved = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            for s in interleaved.iter_mut() {
                *s = 0.0;
            }
            if let Some(r) = self.renderer.as_mut() {
                if !AUDIO_PAUSED.load(Ordering::SeqCst) {
                    r.mix(interleaved);
                }
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
        .set_callback(BloomAudioCallback { renderer })
        .open_stream();

    match stream {
        Ok(ref mut s) => {
            let _ = s.start();
            // Keep thread alive while audio is running; pause the stream while
            // the app is in the background
            let mut paused = false;
            while AUDIO_RUNNING.load(Ordering::SeqCst) {
                let want = AUDIO_PAUSED.load(Ordering::SeqCst);
                if want != paused {
                    paused = want;
                    let _ = if paused { s.pause() } else { s.start() };
                }
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
pub extern "C" fn bloom_set_window_title(title_ptr: *const u8) {
    let _ = str_from_header(title_ptr);
}
#[no_mangle]
pub extern "C" fn bloom_set_window_icon(path_ptr: *const u8) {
    let _ = str_from_header(path_ptr);
}

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
pub extern "C" fn bloom_get_clipboard_text() -> *const u8 {
    std::ptr::null()
}

// E5b: File dialogs (stub on this platform)
#[no_mangle]
pub extern "C" fn bloom_open_file_dialog(
    _filter_ptr: *const u8,
    _title_ptr: *const u8,
) -> *const u8 {
    std::ptr::null()
}
#[no_mangle]
pub extern "C" fn bloom_save_file_dialog(
    _default_name_ptr: *const u8,
    _title_ptr: *const u8,
) -> *const u8 {
    std::ptr::null()
}
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 {
    5.0
}

/// Preferred OS language packed as `c0*256+c1` (ISO-639 primary subtag), read
/// from the device locale system property via the NDK (no JNI). Tries the
/// user-set locale first, then the factory defaults. Falls back to "en".
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    fn parse(buf: &[u8], n: i32) -> Option<f64> {
        if n < 2 {
            return None;
        }
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
        if let Some(v) = parse(&buf, n) {
            return v;
        }
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
    fn ANativeWindow_fromSurface(
        env: *mut libc::c_void,
        surface: *mut libc::c_void,
    ) -> *mut libc::c_void;
    fn mallopt(param: i32, value: i32) -> i32;
    fn __android_log_print(prio: i32, tag: *const u8, fmt: *const u8, ...) -> i32;
    fn main() -> i32;
}

/// Install a panic hook that logs file:line + payload to logcat (tag
/// "BloomPanic"). The FFI `guard` (native/shared/src/ffi.rs) catches panics at
/// the boundary and only reports a generic "non-string panic payload",
/// discarding the location; this hook fires *before* catch_unwind and records
/// where it happened — the key to diagnosing the wgpu/naga material-compile
/// panics on the Adreno Vulkan path. Called from both JNI_OnLoad and
/// bloom_init_window because perry-runtime overrides the hook when main() runs.
fn install_panic_logger() {
    std::panic::set_hook(Box::new(|info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string payload>".to_string()
        };
        let msg = format!("PANIC @ {loc}: {payload}\0");
        unsafe {
            __android_log_print(6, b"BloomPanic\0".as_ptr(), b"%s\0".as_ptr(), msg.as_ptr());
        }
    }));
}

/// An app process's stdout and stderr go to /dev/null on Android, so the
/// game's console.log / console.error (and a Perry runtime error's report)
/// would be lost. Both go into a pipe instead, and a thread writes each line
/// to logcat: tag "BloomStdout", info for stdout's and stderr's alike.
fn route_stdio_to_logcat() {
    unsafe {
        let mut fds = [0i32; 2];
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return;
        }
        // line-buffered, so a line shows up when it is written
        libc::setvbuf(stdout_file(), std::ptr::null_mut(), libc::_IOLBF, 0);
        libc::dup2(fds[1], 1);
        libc::dup2(fds[1], 2);
        libc::close(fds[1]);
        let read_fd = fds[0];
        let _ = std::thread::Builder::new().name("bloom-stdout".into()).spawn(move || {
            let mut buf = [0u8; 4096];
            let mut line: Vec<u8> = Vec::new();
            loop {
                let n = libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len());
                if n <= 0 {
                    break;
                }
                for &b in &buf[..n as usize] {
                    if b == b'\n' || line.len() >= 1000 {
                        line.push(0);
                        __android_log_print(4, b"BloomStdout\0".as_ptr(), b"%s\0".as_ptr(), line.as_ptr());
                        line.clear();
                        if b == b'\n' {
                            continue;
                        }
                    }
                    line.push(b);
                }
            }
        });
    }
}

/// C's `stdout` FILE*, for setvbuf.
unsafe fn stdout_file() -> *mut libc::FILE {
    extern "C" {
        static mut stdout: *mut libc::FILE;
    }
    stdout
}

/// JNI_OnLoad: called when System.loadLibrary() loads this .so.
/// Disables MTE heap tagging (required for Perry NaN-boxing) and
/// reads the asset base path from BLOOM_ASSET_PATH env var.
#[no_mangle]
pub extern "C" fn JNI_OnLoad(_vm: *mut libc::c_void, _reserved: *mut libc::c_void) -> i32 {
    // SH-055 diag — route the `log` facade (wgpu/naga emit shader-validation
    // detail here) to logcat under tag "BloomWGPU". Warn-level keeps it quiet
    // in the steady state but surfaces the Adreno Vulkan compile failures.
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Warn)
            .with_tag("BloomWGPU"),
    );

    route_stdio_to_logcat();

    // Perry-runtime installs its own panic hook once `main()` runs, so this one
    // gets overridden; `bloom_init_window` re-installs it (see
    // `install_panic_logger`) after perry is up and before material compile.
    install_panic_logger();
    unsafe {
        // Disable MTE heap tagging for Perry NaN-boxing compatibility.
        // Perry uses 48-bit pointers; Android's scudo allocator may tag
        // the top byte, corrupting NaN-boxed pointer values.
        mallopt(-204, 0);

        __android_log_print(
            3,
            b"BloomEngine\0".as_ptr(),
            b"JNI_OnLoad: MTE disabled\0".as_ptr(),
        );
    }

    // SH-055 diag — propagate the skip-sky debug prop into an env var the
    // platform-neutral renderer can read (see scene_pass.rs). JNI_OnLoad runs
    // before main(), so the renderer's OnceLock sees the final value.
    if sysprop("debug.bloom.skipsky") == "1" {
        std::env::set_var("BLOOM_SKIP_SKY", "1");
    }
    // SH-055 diag — frame-graph bisection: comma list of pass-node names to
    // skip (see Renderer::dbg_skip). e.g.
    //   adb shell setprop debug.bloom.skip hdr_scene,translucent
    let skip_list = sysprop("debug.bloom.skip");
    if !skip_list.is_empty() {
        std::env::set_var("BLOOM_SKIP", skip_list);
    }
    // SH-055 diag — cap the cached-model draw count (binary search for the
    // pathological draw): adb shell setprop debug.bloom.maxmodels N
    let max_models = sysprop("debug.bloom.maxmodels");
    if !max_models.is_empty() {
        std::env::set_var("BLOOM_MAX_MODELS", max_models);
    }

    // Read asset base path from environment (set by Activity before loadLibrary)
    if let Ok(path) = std::env::var("BLOOM_ASSET_PATH") {
        unsafe {
            __android_log_print(
                3,
                b"BloomEngine\0".as_ptr(),
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
        3,
        b"BloomEngine\0".as_ptr(),
        b"nativeSetSurface: ANativeWindow acquired\0".as_ptr(),
    );
    // ANativeWindow_fromSurface already holds a reference for us
    NATIVE_WINDOW = window;
    // After nativeSurfaceLost: the waiting game thread picks the new window up.
    let mut life = LIFE.lock().unwrap();
    life.window = window as usize;
    life.lost = false;
    LIFE_CHANGED.notify_all();
}

/// The surface is going away (the app went to the background, or the Activity
/// is finishing). Waits until the game thread has dropped its wgpu surface,
/// then lets the window go. Call from SurfaceHolder.Callback.surfaceDestroyed;
/// a later nativeSetSurface resumes the game on the new surface.
#[no_mangle]
pub unsafe extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeSurfaceLost(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
) {
    let mut life = LIFE.lock().unwrap();
    life.lost = true;
    life.window = 0;
    LIFE_CHANGED.notify_all();
    if ENGINE_READY.load(Ordering::SeqCst) {
        // a frame takes milliseconds; the bound only keeps a stuck game thread
        // from blocking the UI thread into an ANR
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !life.dropped {
            let now = std::time::Instant::now();
            if now >= deadline {
                log::warn!("nativeSurfaceLost: the game thread didn't release the surface");
                break;
            }
            life = LIFE_CHANGED.wait_timeout(life, deadline - now).unwrap().0;
        }
    }
    drop(life);
    if !NATIVE_WINDOW.is_null() {
        ANativeWindow_release(NATIVE_WINDOW);
        NATIVE_WINDOW = std::ptr::null_mut();
    }
}

/// The surface's new size in pixels (SurfaceHolder.Callback.surfaceChanged:
/// rotation, split screen, a foldable opening).
#[no_mangle]
pub extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeOnResize(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
    width: i32,
    height: i32,
) {
    if width > 0 && height > 0 {
        PENDING_SIZE.store(pack_size(width as u32, height as u32), Ordering::SeqCst);
    }
}

/// The display's density (pixels per dp) and the safe-area insets in pixels
/// (system bars, display cutout). Once called, the game sees device pixels as
/// on iOS: getScreenWidth() is the surface's width in pixels and
/// getDisplayScale() is `density`. Call before nativeMain and whenever the
/// insets change.
#[no_mangle]
pub extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeSetDisplayMetrics(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
    density: f64,
    top: f64,
    right: f64,
    bottom: f64,
    left: f64,
) {
    NATIVE_PIXELS.store(true, Ordering::SeqCst);
    bloom_shared::display::set_display_scale(density);
    bloom_shared::display::set_safe_area_insets(top, right, bottom, left);
}

/// The Back button. Returns false when the game is at its top level
/// (setMenuButtonExits(true), as for the Apple TV's Menu button): the Activity
/// then does the system's Back (leaves the app). Otherwise the game gets an
/// Escape press and this returns true.
#[no_mangle]
pub extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeOnBack(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
) -> u8 {
    if bloom_shared::display::menu_exits() || !ENGINE_READY.load(Ordering::SeqCst) {
        return 0;
    }
    BACK_STATE.store(1, Ordering::SeqCst);
    1
}

/// Run the compiled game's main() function on the game thread.
/// Called from BloomGameBridge.nativeMain().
#[no_mangle]
pub unsafe extern "C" fn Java_com_bloomengine_game_BloomGameBridge_nativeMain(
    _env: *mut libc::c_void,
    _class: *mut libc::c_void,
) {
    __android_log_print(
        3,
        b"BloomEngine\0".as_ptr(),
        b"nativeMain: calling main()\0".as_ptr(),
    );
    main();
    __android_log_print(
        3,
        b"BloomEngine\0".as_ptr(),
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
