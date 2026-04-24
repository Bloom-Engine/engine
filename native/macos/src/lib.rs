use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::str_from_header;
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3};

use objc2::rc::Retained;
use objc2::{msg_send, MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy, NSEventMask, NSEventType, NSWindow, NSWindowStyleMask};
use objc2_foundation::{NSDate, NSDefaultRunLoopMode, NSPoint, NSRect, NSSize, NSString};

use raw_window_handle::{RawWindowHandle, AppKitWindowHandle};
use std::sync::OnceLock;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut WINDOW: Option<Retained<NSWindow>> = None;
// Set by bloom_init_window when BLOOM_HEADLESS=1. Skips the
// `!isVisible → should_close` shortcut (hidden windows aren't
// 'closed', just invisible) so headless --capture can run to
// completion.
static mut HEADLESS: bool = false;
static mut AUDIO_UNIT: Option<AudioUnitInstance> = None;

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}

/// Map macOS virtual key code to Bloom key code.
fn map_keycode(keycode: u16) -> usize {
    match keycode {
        0 => 65,   // A
        1 => 83,   // S
        2 => 68,   // D
        3 => 70,   // F
        4 => 72,   // H
        5 => 71,   // G
        6 => 90,   // Z
        7 => 88,   // X
        8 => 67,   // C
        9 => 86,   // V
        11 => 66,  // B
        12 => 81,  // Q
        13 => 87,  // W
        14 => 69,  // E
        15 => 82,  // R
        16 => 89,  // Y
        17 => 84,  // T
        18 => 49,  // 1
        19 => 50,  // 2
        20 => 51,  // 3
        21 => 52,  // 4
        22 => 54,  // 6
        23 => 53,  // 5
        24 => 61,  // =
        25 => 57,  // 9
        26 => 55,  // 7
        27 => 45,  // -
        28 => 56,  // 8
        29 => 48,  // 0
        30 => 93,  // ]
        31 => 79,  // O
        32 => 85,  // U
        33 => 91,  // [
        34 => 73,  // I
        35 => 80,  // P
        36 => 265, // Enter (mapped to Bloom ENTER = 265)
        37 => 76,  // L
        38 => 74,  // J
        39 => 39,  // '
        40 => 75,  // K
        41 => 59,  // ;
        42 => 92,  // backslash
        43 => 44,  // ,
        44 => 47,  // /
        45 => 78,  // N
        46 => 77,  // M
        47 => 46,  // .
        48 => 9,   // Tab
        49 => 32,  // Space
        50 => 96,  // `
        51 => 8,   // Backspace
        53 => 27,  // Escape
        // Arrow keys
        123 => 258, // Left
        124 => 259, // Right
        125 => 257, // Down
        126 => 256, // Up
        // Function keys
        122 => 112, // F1
        120 => 113, // F2
        99 => 114,  // F3
        118 => 115, // F4
        96 => 116,  // F5
        97 => 117,  // F6
        98 => 118,  // F7
        100 => 119, // F8
        101 => 120, // F9
        109 => 121, // F10
        103 => 122, // F11
        111 => 123, // F12
        // Modifiers
        56 => 280,  // Left Shift
        60 => 281,  // Right Shift
        59 => 282,  // Left Control
        62 => 283,  // Right Control
        58 => 284,  // Left Alt/Option
        61 => 285,  // Right Alt/Option
        55 => 286,  // Left Command
        54 => 287,  // Right Command
        _ => 0,
    }
}

// ============================================================
// CoreAudio FFI types and setup
// ============================================================

type AudioUnit = *mut std::ffi::c_void;
type OSStatus = i32;
type AudioUnitPropertyID = u32;
type AudioUnitScope = u32;
type AudioUnitElement = u32;

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioComponentDescription {
    component_type: u32,
    component_sub_type: u32,
    component_manufacturer: u32,
    component_flags: u32,
    component_flags_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBuffer; 1],
}

#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut std::ffi::c_void,
}

type AURenderCallback = unsafe extern "C" fn(
    in_ref_con: *mut std::ffi::c_void,
    io_action_flags: *mut u32,
    in_time_stamp: *const std::ffi::c_void,
    in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> OSStatus;

#[repr(C)]
struct AURenderCallbackStruct {
    input_proc: AURenderCallback,
    input_proc_ref_con: *mut std::ffi::c_void,
}

type AudioComponent = *mut std::ffi::c_void;

#[link(name = "AudioToolbox", kind = "framework")]
extern "C" {
    fn AudioComponentFindNext(component: AudioComponent, desc: *const AudioComponentDescription) -> AudioComponent;
    fn AudioComponentInstanceNew(component: AudioComponent, out: *mut AudioUnit) -> OSStatus;
    fn AudioUnitSetProperty(
        unit: AudioUnit,
        property_id: AudioUnitPropertyID,
        scope: AudioUnitScope,
        element: AudioUnitElement,
        data: *const std::ffi::c_void,
        data_size: u32,
    ) -> OSStatus;
    fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
    fn AudioComponentInstanceDispose(unit: AudioUnit) -> OSStatus;
}

const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = u32::from_be_bytes(*b"auou");
const K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT: u32 = u32::from_be_bytes(*b"def ");
const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = u32::from_be_bytes(*b"appl");

const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: AudioUnitPropertyID = 8;
const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: AudioUnitPropertyID = 23;
const K_AUDIO_UNIT_SCOPE_INPUT: AudioUnitScope = 1;

const K_AUDIO_FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 8;

struct AudioUnitInstance {
    unit: AudioUnit,
}

// Safety: AudioUnit is accessed only from audio thread callback + main thread init/deinit
unsafe impl Send for AudioUnitInstance {}
unsafe impl Sync for AudioUnitInstance {}

unsafe extern "C" fn audio_render_callback(
    _in_ref_con: *mut std::ffi::c_void,
    _io_action_flags: *mut u32,
    _in_time_stamp: *const std::ffi::c_void,
    _in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> OSStatus {
    let buffer_list = &mut *io_data;
    let buffer = &mut buffer_list.buffers[0];
    let num_samples = in_number_frames as usize * 2; // stereo
    let output = std::slice::from_raw_parts_mut(
        buffer.data as *mut f32,
        num_samples,
    );

    ENGINE.get_mut().map(|eng| {
        eng.audio.mix_output(output);
    });

    0 // noErr
}

// ============================================================
// FFI entry points
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8, fullscreen: f64) {
    let title = str_from_header(title_ptr);
    let mtm = MainThreadMarker::from(unsafe { MainThreadMarker::new_unchecked() });

    // Headless mode: BLOOM_HEADLESS=1 keeps the NSWindow + CAMetalLayer
    // alive (wgpu's Metal backend requires a CAMetalLayer-backed view)
    // but hides the window and suppresses dock icon / focus. Needed
    // so an agent can spin up the renderer in a batch loop without
    // stealing the user's focus on every sample.
    let headless = std::env::var("BLOOM_HEADLESS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    unsafe { HEADLESS = headless; }

    let app = NSApplication::sharedApplication(mtm);
    if headless {
        // Prohibited = no dock icon, no menu bar, no activation.
        app.setActivationPolicy(NSApplicationActivationPolicy::Prohibited);
    } else {
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    }

    // Far-off-screen origin keeps the window out of every display
    // even if the OS insists on showing something.
    let origin_x = if headless { -20000.0 } else { 100.0 };
    let content_rect = NSRect::new(NSPoint::new(origin_x, 100.0), NSSize::new(width, height));
    let style = NSWindowStyleMask::Titled
        | NSWindowStyleMask::Closable
        | NSWindowStyleMask::Miniaturizable
        | NSWindowStyleMask::Resizable;

    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            content_rect,
            style,
            objc2_app_kit::NSBackingStoreType(2), // NSBackingStoreBuffered
            false,
        )
    };

    let ns_title = NSString::from_str(title);
    window.setTitle(&ns_title);

    // Don't persist window size/fullscreen state across launches.
    // NSWindow restoration was resurrecting a prior fullscreen toggle on
    // the 4K display, which silently rendered benchmarks at 4× the
    // requested pixel count.
    unsafe { let _: () = msg_send![&window, setRestorable: false]; }

    // BLOOM_NO_FULLSCREEN=1 hard-disables fullscreen capability: the
    // window cannot be entered into fullscreen via the green button,
    // cmd-ctrl-F, or inheriting a fullscreen Space from the launching
    // terminal. Intended for benchmark harnesses where the 4K-display
    // fullscreen path would otherwise silently quadruple render cost.
    let no_fullscreen = std::env::var("BLOOM_NO_FULLSCREEN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if no_fullscreen {
        window.setCollectionBehavior(objc2_app_kit::NSWindowCollectionBehavior::FullScreenNone);
    }

    if headless {
        // Alpha 0 + off-screen origin + prohibited-activation app
        // policy = fully invisible window that still backs the
        // CAMetalLayer wgpu renders into. Don't call
        // `makeKeyAndOrderFront` — that brings it to front.
        unsafe {
            let _: () = msg_send![&window, setAlphaValue: 0.0_f64];
            let _: () = msg_send![&window, setIgnoresMouseEvents: true];
            let _: () = msg_send![&window, orderOut: std::ptr::null::<objc2::runtime::AnyObject>()];
        }
    } else {
        window.center();
        window.makeKeyAndOrderFront(None);
        unsafe { app.activateIgnoringOtherApps(true) };
    }

    // Set up CAMetalLayer on the content view
    let content_view = window.contentView().expect("No content view");
    unsafe {
        let _: () = msg_send![&content_view, setWantsLayer: true];
    }

    // Create wgpu surface and renderer
    // wgpu expects the NSView pointer (not NSWindow) for AppKit
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let surface = unsafe {
        let view_ptr = Retained::as_ptr(&content_view) as *mut std::ffi::c_void;
        let handle = AppKitWindowHandle::new(
            std::ptr::NonNull::new(view_ptr).unwrap()
        );
        let raw = RawWindowHandle::AppKit(handle);
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_window_handle::RawDisplayHandle::AppKit(raw_window_handle::AppKitDisplayHandle::new())),
            raw_window_handle: raw,
        }).expect("Failed to create surface")
    };

    let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("No adapter found");

    // Request TIMESTAMP_QUERY when the adapter supports it so the profiler
    // can collect GPU timings. It's optional — profiler falls back to CPU
    // only when the feature isn't available.
    let supported = adapter.features();
    let mut required_features = wgpu::Features::empty();
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    // Ticket 007b: request ray-query + BLAS/TLAS where the adapter
    // supports both (Apple Silicon Metal, DXR 1.1, VK_KHR_ray_query).
    // `BLOOM_FORCE_SW_GI=1` forces the SW fallback for testing parity
    // with non-RT adapters.
    let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // wgpu 29 gates BLAS/TLAS creation + ray-query WGSL on a single
    // feature bit; there's no separate "acceleration structure" flag.
    let rt_mask = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    if !force_sw_gi && supported.contains(rt_mask) {
        required_features |= rt_mask;
    }
    // wgpu 29 requires an explicit `ExperimentalFeatures::enabled()` token
    // when requesting any `EXPERIMENTAL_*` feature (ray query in our case).
    // The token is constructed through an `unsafe` API acknowledging that
    // experimental paths may hit undefined behavior — Apple Silicon's Metal
    // ray-query path has been stable in wgpu releases since v25 so we're
    // willing to take that risk here.
    let experimental_features = if required_features.intersects(rt_mask) {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };
    // Acceleration-structure limits default to 0 when RT is disabled.
    // `using_minimum_supported_acceleration_structure_values` bumps
    // them to the spec minimums (2^24 BLAS geometries / TLAS instances,
    // etc.) whenever ray query was granted.
    let mut required_limits = wgpu::Limits::default();
    // Phase 1c: the material ABI declares 5 bind groups (PerFrame,
    // PerView, PerMaterial, PerDraw, SceneInputs). wgpu's default
    // limit is 4. Metal / Vulkan / D3D12 support at least 7, so 5 is
    // safely within every real backend's capabilities.
    required_limits.max_bind_groups = 5;
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

    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps.formats.iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    // Retina/HiDPI: AppKit reports window dimensions in points, but
    // CAMetalLayer's drawable needs to be sized in physical pixels or
    // AppKit will bilinearly upscale a low-res image to the display.
    // `backingScaleFactor` is typically 2.0 on Retina Macs, 1.0
    // otherwise; on mixed-DPI setups it tracks whichever screen the
    // window is on.
    let scale: f64 = unsafe { msg_send![&*window, backingScaleFactor] };
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let logical_w = width as u32;
    let logical_h = height as u32;
    let physical_w = ((width * scale) as u32).max(1);
    let physical_h = ((height * scale) as u32).max(1);

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format,
        width: physical_w,
        height: physical_h,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let renderer = Renderer::new(device, queue, surface, surface_config, logical_w, logical_h);
    let engine_state = EngineState::new(renderer);

    unsafe {
        let _ = ENGINE.set(engine_state);
        WINDOW = Some(window);
    }

    // Register Bloom's GPU screenshot capture with perry-geisterhand (if linked)
    bloom_register_geisterhand_screenshot();

    if fullscreen != 0.0 {
        bloom_toggle_fullscreen();
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {
    unsafe {
        WINDOW = None;
    }
}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 {
    if engine().should_close { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    // Poll events
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    let app = NSApplication::sharedApplication(mtm);

    loop {
        let event = unsafe {
            app.nextEventMatchingMask_untilDate_inMode_dequeue(
                NSEventMask::Any,
                Some(&NSDate::distantPast()),
                NSDefaultRunLoopMode,
                true,
            )
        };
        match event {
            Some(event) => {
                let event_type = unsafe { event.r#type() };
                match event_type {
                    NSEventType::KeyDown => {
                        let keycode = unsafe { event.keyCode() };
                        let bloom_key = map_keycode(keycode);
                        if bloom_key > 0 {
                            engine().input.set_key_down(bloom_key);
                        }
                        // Extract typed characters for text input (E3b).
                        let chars_obj = unsafe { event.characters() };
                        if let Some(chars) = chars_obj {
                            let s = chars.to_string();
                            for c in s.chars() {
                                let cp = c as u32;
                                // Filter out control characters (keep printable + backspace).
                                if cp >= 32 || cp == 8 || cp == 13 || cp == 9 {
                                    engine().input.push_char(cp);
                                }
                            }
                        }
                    }
                    NSEventType::KeyUp => {
                        let keycode = unsafe { event.keyCode() };
                        let bloom_key = map_keycode(keycode);
                        if bloom_key > 0 {
                            engine().input.set_key_up(bloom_key);
                        }
                    }
                    NSEventType::MouseMoved | NSEventType::LeftMouseDragged | NSEventType::RightMouseDragged => {
                        if engine().input.cursor_disabled {
                            // In disabled-cursor mode, use raw deltas from NSEvent
                            let dx: f64 = unsafe { msg_send![&*event, deltaX] };
                            let dy: f64 = unsafe { msg_send![&*event, deltaY] };
                            engine().input.accumulate_mouse_delta(dx, dy);
                        } else if let Some(window) = unsafe { &WINDOW } {
                            let loc = unsafe { event.locationInWindow() };
                            let frame = window.contentView().map(|v| v.frame()).unwrap_or(NSRect::ZERO);
                            engine().input.set_mouse_position(loc.x, frame.size.height - loc.y);
                        }
                    }
                    NSEventType::LeftMouseDown => {
                        engine().input.set_mouse_button_down(0);
                    }
                    NSEventType::LeftMouseUp => {
                        engine().input.set_mouse_button_up(0);
                    }
                    NSEventType::RightMouseDown => {
                        engine().input.set_mouse_button_down(1);
                    }
                    NSEventType::RightMouseUp => {
                        engine().input.set_mouse_button_up(1);
                    }
                    NSEventType::ScrollWheel => {
                        // NSEvent's scrollingDeltaY is positive when scrolling up
                        // (away from the user). Normalize to "positive = zoom in"
                        // by flipping the sign — matches editor orbit convention.
                        let dy: f64 = unsafe { msg_send![&*event, scrollingDeltaY] };
                        engine().input.accumulate_mouse_wheel(dy);
                    }
                    _ => {}
                }
                unsafe { app.sendEvent(&event) };
            }
            None => break,
        }
    }

    // Check if window was closed. In headless mode the window is
    // intentionally hidden — skip the isVisible check so --capture
    // can actually run to completion without instant-exit.
    let is_headless = unsafe { HEADLESS };
    if !is_headless && unsafe { WINDOW.as_ref().map(|w| !w.isVisible()).unwrap_or(true) } {
        engine().should_close = true;
    }

    // Handle window resize — track physical (backing) size for the
    // swapchain while keeping the logical (points) size for user code.
    if let Some(window) = unsafe { &WINDOW } {
        if let Some(content_view) = window.contentView() {
            let frame = content_view.frame();
            let logical_w = frame.size.width as u32;
            let logical_h = frame.size.height as u32;
            let scale: f64 = unsafe { msg_send![&*window, backingScaleFactor] };
            let scale = if scale > 0.0 { scale } else { 1.0 };
            let physical_w = ((frame.size.width * scale) as u32).max(1);
            let physical_h = ((frame.size.height * scale) as u32).max(1);
            let eng = engine();
            if logical_w > 0 && logical_h > 0
                && (physical_w != eng.renderer.physical_width()
                    || physical_h != eng.renderer.physical_height()
                    || logical_w != eng.renderer.width()
                    || logical_h != eng.renderer.height())
            {
                eng.renderer.resize(physical_w, physical_h, logical_w, logical_h);
            }
        }
    }

    // Apply cursor shape (Q2).
    match engine().input.cursor_shape {
        1 => unsafe { objc2_app_kit::NSCursor::pointingHandCursor().set() },
        2 => unsafe { objc2_app_kit::NSCursor::openHandCursor().set() },
        3 => unsafe { objc2_app_kit::NSCursor::IBeamCursor().set() },
        4 => unsafe { objc2_app_kit::NSCursor::resizeLeftRightCursor().set() },
        5 => unsafe { objc2_app_kit::NSCursor::resizeUpDownCursor().set() },
        6 => unsafe { objc2_app_kit::NSCursor::crosshairCursor().set() },
        _ => {},
    }

    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    // Pump geisterhand BEFORE end_frame.
    // Screenshot function re-renders inline with captured VP + vertices.
    extern "C" { fn perry_geisterhand_pump(); }
    unsafe { perry_geisterhand_pump(); }

    engine().end_frame();
}

/// Request a PNG screenshot of the next rendered frame.
/// The capture happens during the next end_drawing(), so the caller
/// should call beginDrawing/endDrawing once after this for the file
/// to actually appear on disk. Used by bloom-diff and CI image
/// regression workflows.
#[no_mangle]
pub extern "C" fn bloom_take_screenshot(path_ptr: *const u8) {
    let path = str_from_header(path_ptr).to_string();
    let eng = engine();
    eng.renderer.screenshot_requested = true;
    eng.renderer.pending_screenshot_path = Some(path);
}

#[no_mangle]
pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.set_clear_color(r, g, b, a);
}

/// Load an HDR equirectangular environment map and upload it to the
/// GPU. Subsequent frames sample it per-background-pixel via a sky
/// pass, so the background matches a path-traced reference instead of
/// being a flat clear color. The file must be Radiance HDR (.hdr).
#[no_mangle]
pub extern "C" fn bloom_set_env_clear_from_hdr(path_ptr: *const u8) {
    use image::ImageDecoder;
    let path = str_from_header(path_ptr).to_string();
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let decoder = match image::codecs::hdr::HdrDecoder::new(std::io::BufReader::new(file)) {
        Ok(d) => d,
        Err(_) => return,
    };
    let (w, h) = decoder.dimensions();
    let byte_len = (w as usize) * (h as usize) * 3 * 4;
    let mut buf = vec![0u8; byte_len];
    if decoder.read_image(&mut buf).is_err() {
        return;
    }
    // Reinterpret the byte buffer as f32 RGB triples for the renderer.
    let rgb_f32: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    engine().renderer.load_env_from_hdr(w, h, &rgb_f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_target_fps(fps: f64) {
    engine().target_fps = fps;
}

#[no_mangle]
pub extern "C" fn bloom_set_direct_2d_mode(on: f64) {
    engine().direct_2d_mode = on > 0.5;
}

#[no_mangle]
pub extern "C" fn bloom_get_delta_time() -> f64 {
    engine().delta_time
}

#[no_mangle]
pub extern "C" fn bloom_get_fps() -> f64 {
    engine().get_fps()
}

#[no_mangle]
pub extern "C" fn bloom_get_screen_width() -> f64 {
    engine().screen_width()
}

#[no_mangle]
pub extern "C" fn bloom_get_screen_height() -> f64 {
    engine().screen_height()
}

#[no_mangle]
pub extern "C" fn bloom_get_time() -> f64 {
    engine().get_time()
}

// ============================================================
// Input - Keyboard
// ============================================================

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

// ============================================================
// Input - Mouse
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_get_mouse_x() -> f64 {
    engine().input.mouse_x
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_y() -> f64 {
    engine().input.mouse_y
}

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

// ============================================================
// Input - Gamepad
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_available(gamepad: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_available() { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis(gamepad: f64, axis: f64) -> f64 {
    let _ = gamepad;
    engine().input.get_gamepad_axis(axis as usize) as f64
}

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_pressed(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_pressed(button as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_down(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_down(button as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_released(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_released(button as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis_count(gamepad: f64) -> f64 {
    let _ = gamepad;
    engine().input.get_gamepad_axis_count() as f64
}

// ============================================================
// Input - Touch
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_get_touch_x() -> f64 {
    engine().input.get_touch_x(0)
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_y() -> f64 {
    engine().input.get_touch_y(0)
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_count() -> f64 {
    engine().input.get_touch_count() as f64
}

// ============================================================
// Shapes
// ============================================================

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

// ============================================================
// Text
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_draw_text(text_ptr: *const u8, x: f64, y: f64, size: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    // Need to split borrow: take text out temporarily
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
    match std::fs::read(path) {
        Ok(data) => engine().text.load_font(&data) as f64,
        Err(_) => 0.0,
    }
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

// ============================================================
// Textures
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            let eng = engine();
            let EngineState { ref mut textures, ref mut renderer, .. } = *eng;
            textures.load_texture(renderer, &data)
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_texture(handle: f64) {
    let eng = engine();
    let EngineState { ref mut textures, ref mut renderer, .. } = *eng;
    textures.unload_texture(handle, renderer);
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture(handle: f64, x: f64, y: f64, tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.draw_texture(bind_group_idx, x, y, tint_r, tint_g, tint_b, tint_a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_pro(
    handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64,
    origin_x: f64, origin_y: f64, rotation: f64,
    tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
) {
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
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_rec(
    handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64,
    tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
) {
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
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_width(handle: f64) -> f64 {
    let eng = engine();
    eng.textures.get(handle).map(|t| t.width as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_height(handle: f64) -> f64 {
    let eng = engine();
    eng.textures.get(handle).map(|t| t.height as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_load_image(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => engine().textures.load_image(&data),
        Err(_) => 0.0,
    }
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
    let EngineState { ref mut textures, ref mut renderer, .. } = *eng;
    textures.load_texture_from_image(handle, renderer)
}

#[no_mangle]
pub extern "C" fn bloom_gen_texture_mipmaps(_handle: f64) {
    // Mipmap generation is handled by the GPU texture creation pipeline
    // This is a no-op for now as wgpu handles mipmaps internally
}

#[no_mangle]
pub extern "C" fn bloom_set_texture_filter(handle: f64, mode: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.set_texture_filter(bind_group_idx, mode > 0.5);
    }
}

// ============================================================
// Camera 2D
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_begin_mode_2d(offset_x: f64, offset_y: f64, target_x: f64, target_y: f64, rotation: f64, zoom: f64) {
    engine().renderer.begin_mode_2d(
        offset_x as f32, offset_y as f32,
        target_x as f32, target_y as f32,
        rotation as f32, zoom as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_end_mode_2d() {
    engine().renderer.end_mode_2d();
}

// ============================================================
// Camera 3D and 3D drawing
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_begin_mode_3d(
    pos_x: f64, pos_y: f64, pos_z: f64,
    target_x: f64, target_y: f64, target_z: f64,
    up_x: f64, up_y: f64, up_z: f64,
    fovy: f64, projection: f64,
) {
    engine().renderer.begin_mode_3d(
        pos_x as f32, pos_y as f32, pos_z as f32,
        target_x as f32, target_y as f32, target_z as f32,
        up_x as f32, up_y as f32, up_z as f32,
        fovy as f32, projection as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_end_mode_3d() {
    engine().renderer.end_mode_3d();
}

#[no_mangle]
pub extern "C" fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube(x, y, z, w, h, d, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube_wires(x, y, z, w, h, d, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_sphere(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere(cx, cy, cz, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_sphere_wires(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere_wires(cx, cy, cz, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_cylinder(x: f64, y: f64, z: f64, radius_top: f64, radius_bottom: f64, height: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cylinder(x, y, z, radius_top, radius_bottom, height, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_plane(cx: f64, cy: f64, cz: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_plane(cx, cy, cz, w, d, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_grid(slices: f64, spacing: f64) {
    engine().renderer.draw_grid(slices as i32, spacing);
}

#[no_mangle]
pub extern "C" fn bloom_draw_ray(origin_x: f64, origin_y: f64, origin_z: f64, dir_x: f64, dir_y: f64, dir_z: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_ray(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b, a);
}

// ============================================================
// Joint test
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_joint_test(joint_index: f64, angle: f64) {
    engine().renderer.set_joint_test(joint_index as usize, angle as f32);
}

// ============================================================
// Lighting
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_ambient_light(r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_ambient_light(r, g, b, intensity);
}

#[no_mangle]
pub extern "C" fn bloom_set_directional_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_directional_light(dx, dy, dz, r, g, b, intensity);
}

// --- Post-FX knobs (heuristic visual layer; default-off) ---

#[no_mangle]
pub extern "C" fn bloom_set_fog(r: f64, g: f64, b: f64, density: f64, height_ref: f64, height_falloff: f64) {
    let r_ = engine();
    r_.renderer.set_fog_color(r as f32, g as f32, b as f32);
    r_.renderer.set_fog_density(density as f32);
    r_.renderer.set_fog_height_falloff(height_ref as f32, height_falloff as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_chromatic_aberration(strength: f64) {
    engine().renderer.set_chromatic_aberration(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_vignette(strength: f64, softness: f64) {
    engine().renderer.set_vignette(strength as f32, softness as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_film_grain(strength: f64) {
    engine().renderer.set_film_grain(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) {
    let eng = engine();
    eng.renderer.set_sun_shaft_strength(strength as f32);
    eng.renderer.set_sun_shaft_decay(decay as f32);
    eng.renderer.set_sun_shaft_color(r as f32, g as f32, b as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_auto_exposure(on: f64) {
    engine().renderer.set_auto_exposure(on != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_taa_enabled(on: f64) {
    engine().renderer.set_taa_enabled(on != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_manual_exposure(value: f64) {
    engine().renderer.set_manual_exposure(value as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_env_intensity(intensity: f64) {
    engine().renderer.set_env_intensity(intensity as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_enabled(enabled: f64) {
    engine().renderer.set_ssgi_enabled(enabled != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_intensity(intensity: f64) {
    engine().renderer.set_ssgi_intensity(intensity as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_radius(radius: f64) {
    engine().renderer.set_ssgi_radius(radius as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_dof(enabled: f64, focus_distance: f64, aperture: f64) {
    let r = &mut engine().renderer;
    r.set_dof_enabled(enabled != 0.0);
    r.set_dof_focus_distance(focus_distance as f32);
    r.set_dof_aperture(aperture as f32);
}

// ============================================================
// Render quality toggles (individual + preset)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_quality_preset(preset: f64) {
    engine().renderer.apply_quality_preset(preset as u32);
}
#[no_mangle]
pub extern "C" fn bloom_set_shadows_enabled(on: f64) {
    engine().renderer.set_shadows_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_shadows_always_fresh(on: f64) {
    engine().renderer.set_shadows_always_fresh(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_bloom_enabled(on: f64) {
    engine().renderer.set_bloom_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssao_enabled(on: f64) {
    engine().renderer.set_ssao_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssr_enabled(on: f64) {
    engine().renderer.set_ssr_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_motion_blur_enabled(on: f64) {
    engine().renderer.set_motion_blur_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_sss_enabled(on: f64) {
    engine().renderer.set_sss_enabled(on != 0.0);
}

// ============================================================
// Profiler — CPU phase timings (always available) + GPU timestamps
// (when the adapter supports TIMESTAMP_QUERY). Disabled by default.
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_profiler_enabled(on: f64) {
    engine().profiler.set_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_get_profiler_frame_cpu_us() -> f64 {
    engine().profiler.avg_frame_cpu_us()
}
#[no_mangle]
pub extern "C" fn bloom_get_profiler_frame_gpu_us() -> f64 {
    engine().profiler.avg_frame_gpu_us()
}
#[no_mangle]
pub extern "C" fn bloom_print_profiler_summary() {
    print!("{}", engine().profiler.summary());
}

// ============================================================
// Models
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            let eng = engine();
            let EngineState { ref mut models, ref mut renderer, .. } = *eng;
            // .gltf = loose file with external .bin + image URIs; pass
            // the directory so relative paths resolve. .glb (binary)
            // embeds everything and needs no base dir.
            let p = std::path::Path::new(&path);
            let is_loose = p.extension().and_then(|e| e.to_str()) == Some("gltf");
            if is_loose {
                if let Some(dir) = p.parent() {
                    return models.load_model_with_textures_from_path(&data, dir, renderer);
                }
            }
            models.load_model_with_textures(&data, renderer)
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(model) = eng.models.get(handle) {
        let position = [x as f32, y as f32, z as f32];
        let scale = scale as f32;
        let tint = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32];
        let handle_bits = handle.to_bits();
        if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
            eng.renderer.draw_model_cached(handle_bits, position, scale, tint);
        } else {
            for mesh in &model.meshes {
                let tex_idx = mesh.texture_idx.unwrap_or(0);
                eng.renderer.draw_model_mesh_tinted(&mesh.vertices, &mesh.indices, position, scale, tint, tex_idx);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_model(handle: f64) {
    engine().models.unload_model(handle);
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

// ============================================================
// Phase 1c — material system FFI
// ============================================================

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

/// Phase 4b — compile a refractive material. Shorthand for
/// `compile_material_with_options(src, Translucent, Refractive,
/// true)`. Perry's ARM64 FFI garbles mixed-signature calls with
/// more than a pointer + 0-1 floats, so we ship dedicated entry
/// points per (profile, bucket, reads_scene) combo that games
/// actually need.
#[no_mangle]
pub extern "C" fn bloom_compile_material_refractive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Refractive, true,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

/// Phase 4b — compile a transparent (non-refractive) material.
/// Profile=Translucent, Bucket=Transparent, reads_scene=false.
#[no_mangle]
pub extern "C" fn bloom_compile_material_transparent(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Transparent, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

/// Phase 4b — compile an additive material. Profile=Translucent,
/// Bucket=Additive, reads_scene=false.
#[no_mangle]
pub extern "C" fn bloom_compile_material_additive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Additive, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
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
        let _ = eng.renderer.cache_model_if_static(handle_bits, &model.meshes);
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
pub extern "C" fn bloom_create_mesh(vertex_ptr: *const f64, vertex_count: f64, index_ptr: *const f64, index_count: f64) -> f64 {
    // Perry's TS `number[]` is f64-laid-out in memory; Perry passes a
    // pointer to that data. A previous version of this FFI declared
    // *const f32 / *const u32, which silently read the low 4 bytes
    // of each f64 slot as garbage f32/u32 values — meshes registered
    // (non-zero handle) but were unrenderable.
    //
    // Caller must pass `vertex_count` and `index_count` derived from
    // a literal-initialized array OR from values it computed itself.
    // Don't compute these via `arr.length` after `.push()` — Perry's
    // `.length` property currently reflects the literal-init size,
    // not the post-push count (a Perry codegen bug). Workaround on
    // the TS side: track the count manually or use literal arrays.
    if vertex_ptr.is_null() || index_ptr.is_null() { return 0.0; }
    let vcount = vertex_count as usize;
    let icount = index_count as usize;
    let vertex_f64 = unsafe { std::slice::from_raw_parts(vertex_ptr, vcount * 12) };
    let index_f64 = unsafe { std::slice::from_raw_parts(index_ptr, icount) };
    let vertex_data: Vec<f32> = vertex_f64.iter().map(|&v| v as f32).collect();
    let index_data: Vec<u32> = index_f64.iter().map(|&v| v as u32).collect();
    engine().models.create_mesh(&vertex_data, &index_data)
}

#[no_mangle]
pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => engine().models.load_model_animation(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64, scale: f64, px: f64, py: f64, pz: f64, rot_sin: f64, _rot_cos: f64) {
    // Derive rot_cos from rot_sin (Perry ARM64 may garble 9th float param on stack)
    let rot_cos = (1.0 - rot_sin * rot_sin).sqrt();
    let eng = engine();
    eng.models.update_model_animation(handle, anim_index as usize, time as f32);
    if let Some(anim) = eng.models.get_animation(handle) {
        if !anim.joint_matrices.is_empty() {
            eng.renderer.set_joint_matrices_scaled(&anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32], rot_sin as f32, rot_cos as f32);
        }
    }
}

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
        Some(model) => model.meshes.len() as f64, // materials roughly equal meshes
        None => 0.0,
    }
}

// ============================================================
// Audio
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    unsafe {
        let desc = AudioComponentDescription {
            component_type: K_AUDIO_UNIT_TYPE_OUTPUT,
            component_sub_type: K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT,
            component_manufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
            component_flags: 0,
            component_flags_mask: 0,
        };

        let component = AudioComponentFindNext(std::ptr::null_mut(), &desc);
        if component.is_null() {
            return;
        }

        let mut unit: AudioUnit = std::ptr::null_mut();
        if AudioComponentInstanceNew(component, &mut unit) != 0 {
            return;
        }

        // Set stream format: 44100 Hz, stereo, float32
        let stream_desc = AudioStreamBasicDescription {
            sample_rate: 44100.0,
            format_id: K_AUDIO_FORMAT_LINEAR_PCM,
            format_flags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
            bytes_per_packet: 8,
            frames_per_packet: 1,
            bytes_per_frame: 8,
            channels_per_frame: 2,
            bits_per_channel: 32,
            reserved: 0,
        };

        AudioUnitSetProperty(
            unit,
            K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
            K_AUDIO_UNIT_SCOPE_INPUT,
            0,
            &stream_desc as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<AudioStreamBasicDescription>() as u32,
        );

        // Set render callback
        let callback_struct = AURenderCallbackStruct {
            input_proc: audio_render_callback,
            input_proc_ref_con: std::ptr::null_mut(),
        };

        AudioUnitSetProperty(
            unit,
            K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
            K_AUDIO_UNIT_SCOPE_INPUT,
            0,
            &callback_struct as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<AURenderCallbackStruct>() as u32,
        );

        AudioUnitInitialize(unit);
        AudioOutputUnitStart(unit);

        AUDIO_UNIT = Some(AudioUnitInstance { unit });
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    unsafe {
        if let Some(au) = AUDIO_UNIT.take() {
            AudioOutputUnitStop(au.unit);
            AudioComponentInstanceDispose(au.unit);
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_sound(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            let sound_data = if path.ends_with(".ogg") || path.ends_with(".OGG") {
                parse_ogg(&data)
            } else if path.ends_with(".mp3") || path.ends_with(".MP3") {
                parse_mp3(&data)
            } else {
                parse_wav(&data)
            };
            if let Some(sound_data) = sound_data {
                engine().audio.load_sound(sound_data)
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_play_sound(handle: f64) {
    engine().audio.play_sound(handle);
}

#[no_mangle]
pub extern "C" fn bloom_stop_sound(handle: f64) {
    engine().audio.stop_sound(handle);
}

#[no_mangle]
pub extern "C" fn bloom_set_sound_volume(handle: f64, volume: f64) {
    engine().audio.set_sound_volume(handle, volume as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_master_volume(volume: f64) {
    engine().audio.master_volume = volume as f32;
}

#[no_mangle]
pub extern "C" fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
    engine().audio.play_sound_3d(handle, x as f32, y as f32, z as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
    engine().audio.set_listener_position(x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32);
}

// ============================================================
// Music
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_music(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            let sound_data = if path.ends_with(".ogg") || path.ends_with(".OGG") {
                parse_ogg(&data)
            } else if path.ends_with(".mp3") || path.ends_with(".MP3") {
                parse_mp3(&data)
            } else {
                parse_wav(&data)
            };
            if let Some(sound_data) = sound_data {
                engine().audio.load_music(sound_data)
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_play_music(handle: f64) {
    engine().audio.play_music(handle);
}

#[no_mangle]
pub extern "C" fn bloom_stop_music(handle: f64) {
    engine().audio.stop_music(handle);
}

#[no_mangle]
pub extern "C" fn bloom_update_music_stream(handle: f64) {
    engine().audio.update_music_stream(handle);
}

#[no_mangle]
pub extern "C" fn bloom_set_music_volume(handle: f64, volume: f64) {
    engine().audio.set_music_volume(handle, volume as f32);
}

#[no_mangle]
pub extern "C" fn bloom_is_music_playing(handle: f64) -> f64 {
    if engine().audio.is_music_playing(handle) { 1.0 } else { 0.0 }
}

// ============================================================
// Utility
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_toggle_fullscreen() {
    unsafe {
        if let Some(window) = &WINDOW {
            let _: () = msg_send![window, toggleFullScreen: std::ptr::null::<std::ffi::c_void>()];
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_set_window_title(title_ptr: *const u8) {
    let title = str_from_header(title_ptr);
    unsafe {
        if let Some(window) = &WINDOW {
            let ns_title = NSString::from_str(title);
            window.setTitle(&ns_title);
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_set_window_icon(path_ptr: *const u8) {
    let path = str_from_header(path_ptr);
    unsafe {
        let ns_path = NSString::from_str(path);
        let image_cls = objc2::runtime::AnyClass::get(c"NSImage").unwrap();
        let image: *mut objc2::runtime::AnyObject =
            msg_send![image_cls, alloc];
        if image.is_null() { return; }
        let image: *mut objc2::runtime::AnyObject =
            msg_send![image, initWithContentsOfFile: &*ns_path];
        if image.is_null() { return; }
        let app = NSApplication::sharedApplication(MainThreadMarker::new_unchecked());
        let _: () = msg_send![&*app, setApplicationIconImage: image];
    }
}

extern "C" {
    fn CGDisplayHideCursor(display: u32) -> i32;
    fn CGDisplayShowCursor(display: u32) -> i32;
    fn CGAssociateMouseAndMouseCursorPosition(connected: u8) -> i32;
}

#[no_mangle]
pub extern "C" fn bloom_disable_cursor() {
    let input = &mut engine().input;
    input.cursor_disabled = true;
    input.clear_mouse_delta();
    unsafe {
        CGDisplayHideCursor(0);
        CGAssociateMouseAndMouseCursorPosition(0); // dissociate = relative mode
    }
}

#[no_mangle]
pub extern "C" fn bloom_enable_cursor() {
    engine().input.cursor_disabled = false;
    unsafe {
        CGAssociateMouseAndMouseCursorPosition(1);
        CGDisplayShowCursor(0);
    }
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

// Dequeue the next typed character (Unicode codepoint). Returns 0 when the
// queue is empty. Call in a loop each frame to consume all pending characters.
// Used by the editor's in-window text-input widget.
#[no_mangle]
pub extern "C" fn bloom_get_char_pressed() -> f64 {
    engine().input.pop_char() as f64
}

// Q2: Cursor shape
#[no_mangle]
pub extern "C" fn bloom_set_cursor_shape(shape: f64) {
    engine().input.cursor_shape = shape as u32;
}

// E4: Clipboard
#[no_mangle]
pub extern "C" fn bloom_set_clipboard_text(text_ptr: *const u8) {
    let text = str_from_header(text_ptr);
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        let _ = clipboard.set_text(text.to_string());
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_clipboard_text() -> *const u8 {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => {
            match clipboard.get_text() {
                Ok(text) => {
                    let bytes = text.as_bytes();
                    let len = bytes.len();
                    let total = 12 + len;
                    let layout = std::alloc::Layout::from_size_align(total, 4).unwrap();
                    unsafe {
                        let ptr = std::alloc::alloc(layout);
                        if ptr.is_null() { return std::ptr::null(); }
                        *(ptr as *mut u32) = len as u32;
                        *(ptr.add(4) as *mut u32) = len as u32;
                        *(ptr.add(8) as *mut u32) = 1;
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(12), len);
                        ptr
                    }
                }
                Err(_) => std::ptr::null(),
            }
        }
        Err(_) => std::ptr::null(),
    }
}

// E5b: Native file dialogs (via rfd crate)
#[no_mangle]
pub extern "C" fn bloom_open_file_dialog(filter_ptr: *const u8, title_ptr: *const u8) -> *const u8 {
    let filter = str_from_header(filter_ptr);
    let title = str_from_header(title_ptr);
    let mut dialog = rfd::FileDialog::new().set_title(title);
    if !filter.is_empty() {
        dialog = dialog.add_filter("Files", &[filter]);
    }
    match dialog.pick_file() {
        Some(path) => {
            let s = path.to_string_lossy();
            let bytes = s.as_bytes();
            let len = bytes.len();
            let total = 12 + len;
            let layout = std::alloc::Layout::from_size_align(total, 4).unwrap();
            unsafe {
                let ptr = std::alloc::alloc(layout);
                if ptr.is_null() { return std::ptr::null(); }
                *(ptr as *mut u32) = len as u32;
                *(ptr.add(4) as *mut u32) = len as u32;
                *(ptr.add(8) as *mut u32) = 1;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(12), len);
                ptr
            }
        }
        None => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn bloom_save_file_dialog(default_name_ptr: *const u8, title_ptr: *const u8) -> *const u8 {
    let default_name = str_from_header(default_name_ptr);
    let title = str_from_header(title_ptr);
    let dialog = rfd::FileDialog::new()
        .set_title(title)
        .set_file_name(default_name);
    match dialog.save_file() {
        Some(path) => {
            let s = path.to_string_lossy();
            let bytes = s.as_bytes();
            let len = bytes.len();
            let total = 12 + len;
            let layout = std::alloc::Layout::from_size_align(total, 4).unwrap();
            unsafe {
                let ptr = std::alloc::alloc(layout);
                if ptr.is_null() { return std::ptr::null(); }
                *(ptr as *mut u32) = len as u32;
                *(ptr.add(4) as *mut u32) = len as u32;
                *(ptr.add(8) as *mut u32) = 1;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(12), len);
                ptr
            }
        }
        None => std::ptr::null(),
    }
}

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
    if std::path::Path::new(path).exists() { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_read_file(path_ptr: *const u8) -> *const u8 {
    let path = str_from_header(path_ptr);
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            // Return Perry-format string: StringHeader (length u32 + capacity u32 + refcount u32) followed by UTF-8 data
            let bytes = contents.as_bytes();
            let len = bytes.len();
            let total = 12 + len; // 12 bytes header (3 × u32) + data
            let layout = std::alloc::Layout::from_size_align(total, 4).unwrap();
            unsafe {
                let ptr = std::alloc::alloc(layout);
                if ptr.is_null() { return std::ptr::null(); }
                *(ptr as *mut u32) = len as u32;           // length
                *(ptr.add(4) as *mut u32) = len as u32;    // capacity
                *(ptr.add(8) as *mut u32) = 1;             // refcount (unique)
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(12), len);
                ptr
            }
        }
        Err(_) => std::ptr::null(),
    }
}

// ============================================================
// Input injection + platform detection
// ============================================================

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
pub extern "C" fn bloom_get_platform() -> f64 { 1.0 }
#[no_mangle]
pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
    if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
}
#[no_mangle]
pub extern "C" fn bloom_get_crown_rotation() -> f64 {
    engine().input.consume_crown_rotation()
}

// ============================================================
// Frame callbacks
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_register_frame_callback(priority: f64, callback: extern "C" fn(f64)) -> f64 {
    engine().frame_callbacks.register(priority as i32, callback) as f64
}

#[no_mangle]
pub extern "C" fn bloom_unregister_frame_callback(id: f64) {
    engine().frame_callbacks.unregister(id as u64);
}

/// Emscripten-style game loop. On native, this is a no-op — the game uses
/// a while(!windowShouldClose()) loop in TypeScript instead. The TS-level
/// runGame() helper handles the native loop. Only used on web where the
/// JS glue intercepts this and drives requestAnimationFrame.
#[no_mangle]
pub extern "C" fn bloom_run_game(_callback: extern "C" fn(f64)) {
    // No-op on native. The TypeScript runGame() helper provides the while loop.
}

// ============================================================
// Multiple lights
// ============================================================

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

// ============================================================
// Scene graph (retained mode)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_scene_create_node() -> f64 {
    engine().scene.create_node()
}

#[no_mangle]
pub extern "C" fn bloom_scene_destroy_node(handle: f64) {
    engine().scene.destroy_node(handle);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_visible(handle: f64, visible: f64) {
    engine().scene.set_visible(handle, visible != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_cast_shadow(handle: f64, cast: f64) {
    engine().scene.set_cast_shadow(handle, cast != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_receive_shadow(handle: f64, receive: f64) {
    engine().scene.set_receive_shadow(handle, receive != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_parent(handle: f64, parent: f64) {
    engine().scene.set_parent(handle, parent);
}

/// Set 4x4 transform matrix (16 floats passed as pointer to f64 array).
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

/// Update geometry from flat vertex array (12 floats/vertex: xyz, nxnynz, rgba, uv)
/// and index array.
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

#[no_mangle]
pub extern "C" fn bloom_scene_node_count() -> f64 {
    engine().scene.node_count() as f64
}

/// Debug: get vertex count for a scene node (0 if not found or empty)
#[no_mangle]
pub extern "C" fn bloom_scene_node_vertex_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.vertices.len() as f64,
        None => -1.0,
    }
}

/// Debug: get index count for a scene node
#[no_mangle]
pub extern "C" fn bloom_scene_node_index_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.indices.len() as f64,
        None => -1.0,
    }
}


// Q8: Set a water material on a scene node (translucent tint, low roughness).
#[no_mangle]
pub extern "C" fn bloom_scene_set_material_water(handle: f64, wave_amp: f64, wave_speed: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().scene.set_material_water(handle, wave_amp as f32, wave_speed as f32, r as f32, g as f32, b as f32, a as f32);
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

// Q1: Render texture FFI — create GPU textures for render-to-texture.
#[no_mangle]
pub extern "C" fn bloom_load_render_texture(width: f64, height: f64) -> f64 {
    let w = width as u32;
    let h = height as u32;
    let eng = engine();
    let rt_handle = eng.textures.load_render_texture(w, h);

    // Create the GPU texture via the renderer's public method.
    let (bind_group_idx, _tex_vec_idx) = eng.renderer.create_render_texture(w, h);

    // Register as a texture handle so drawTexture can sample it.
    let tex_handle = eng.textures.textures.alloc(bloom_shared::textures::TextureData {
        bind_group_idx, width: w, height: h,
    });
    eng.textures.set_render_texture_handle(rt_handle, tex_handle);

    rt_handle
}
#[no_mangle]
pub extern "C" fn bloom_unload_render_texture(handle: f64) {
    engine().textures.unload_render_texture(handle);
}
#[no_mangle]
pub extern "C" fn bloom_begin_texture_mode(handle: f64) {
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
}
#[no_mangle]
pub extern "C" fn bloom_end_texture_mode() {
    engine().renderer.end_texture_mode();
}
#[no_mangle]
pub extern "C" fn bloom_get_render_texture_texture(handle: f64) -> f64 {
    engine().textures.get_render_texture_texture(handle)
}
// ============================================================
// Scene graph QoL — Q4/Q5/Q6/Q7
// ============================================================

/// Q4: Read back the 4x4 transform matrix of a scene node.
/// Returns the column-major float at the specified index (0-15).
#[no_mangle]
pub extern "C" fn bloom_scene_get_transform(handle: f64, index: f64) -> f64 {
    let mat = engine().scene.get_transform(handle);
    let i = index as usize;
    let col = i / 4;
    let row = i % 4;
    if col < 4 && row < 4 { mat[col][row] as f64 } else { 0.0 }
}

/// Q5: Get the cached AABB min of a scene node.
#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[0] as f64 }
#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[1] as f64 }
#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[2] as f64 }
#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[0] as f64 }
#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[1] as f64 }
#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[2] as f64 }

/// Q7: Set arbitrary user data on a scene node.
#[no_mangle]
pub extern "C" fn bloom_scene_set_user_data(handle: f64, data: f64) {
    engine().scene.set_user_data(handle, data as i64);
}
#[no_mangle]
pub extern "C" fn bloom_scene_get_user_data(handle: f64) -> f64 {
    engine().scene.get_user_data(handle) as f64
}

// ============================================================
// Geometry generation
// ============================================================

/// Extrude a 2D polygon along Y axis and store as a scene node's geometry.
/// polygon_ptr: pointer to flat f64 array [x0, z0, x1, z1, ...]
/// polygon_count: number of points (NOT number of floats)
/// depth: extrusion height
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

/// Subtract an axis-aligned box from a scene node's geometry.
/// Removes triangles fully inside the box (simplified CSG).
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

// ============================================================
// Shadow mapping
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_enable_shadows() {
    engine().renderer.shadow_map.enable();
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

// ============================================================
// Post-processing
// ============================================================

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
pub extern "C" fn bloom_disable_postfx() {
    engine().postfx = None;
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

// ============================================================
// 3D→2D Projection (for UI overlays positioned in 3D space)
// ============================================================

/// Project a world-space 3D point to screen coordinates.
/// Returns screen X. Call bloom_project_y for Y. Returns -9999 if behind camera.
static mut LAST_PROJECT: (f64, f64) = (0.0, 0.0);

#[no_mangle]
pub extern "C" fn bloom_project_to_screen(wx: f64, wy: f64, wz: f64) -> f64 {
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
}

#[no_mangle]
pub extern "C" fn bloom_project_screen_y() -> f64 {
    unsafe { LAST_PROJECT.1 }
}

/// Attach a loaded GLTF model's mesh geometry to a scene node.
/// Copies the vertex/index data from the model into the scene node.
#[no_mangle]
pub extern "C" fn bloom_scene_attach_model(node_handle: f64, model_handle: f64, mesh_index: f64) {
    let eng = engine();
    let mi = mesh_index as usize;

    // Get model mesh data
    let model_data = match eng.models.models.get(model_handle) {
        Some(md) => md,
        None => return,
    };
    if mi >= model_data.meshes.len() { return; }
    let mesh = &model_data.meshes[mi];

    // Copy vertices and indices to scene node
    let vertices = mesh.vertices.clone();
    let indices = mesh.indices.clone();
    let base_color_tex = mesh.texture_idx;
    let normal_tex = mesh.normal_texture_idx;
    let mr_tex = mesh.metallic_roughness_texture_idx;
    let emissive_tex = mesh.emissive_texture_idx;
    let emissive_factor = mesh.emissive_factor;
    eng.scene.update_geometry(node_handle, vertices, indices);

    // Pipe PBR textures through to the scene node material so the
    // renderer's scene pipeline can sample them.
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

// ============================================================
// Thread-safe staging (for async asset loading via Perry threads)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_stage_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => bloom_shared::staging::decode_and_stage_texture(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_stage_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = match std::fs::read(path) {
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
    let data = match std::fs::read(path) {
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
    // Register staged textures with GPU and build index map.
    // Staged texture_idx values are 1-based into staged.textures.
    // We map them to actual renderer bind_group_idx values.
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

// ============================================================
// Simple blocking executor for wgpu async calls
// ============================================================

fn pollster_block_on<F: std::future::Future>(future: F) -> F::Output {
    // Minimal block_on implementation using std::task
    use std::task::{Context, Poll, Wake, Waker};
    use std::pin::Pin;
    use std::sync::Arc;

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
// Geisterhand screenshot integration
// ============================================================

/// Register Bloom's GPU-based screenshot capture with perry-geisterhand.
/// This replaces perry-ui-macos's CGWindowListCreateImage approach with
/// direct wgpu texture readback — works for Metal/Vulkan rendered content.
fn bloom_register_geisterhand_screenshot() {
    // Try to register with geisterhand if it's linked (weak symbol)
    extern "C" {
        fn perry_geisterhand_register_screenshot_capture(
            f: extern "C" fn(*mut usize) -> *mut u8,
        );
    }
    unsafe {
        perry_geisterhand_register_screenshot_capture(bloom_screenshot_capture);
    }
}

/// Capture the Bloom framebuffer as PNG.
/// Called from geisterhand pump BEFORE end_frame in bloom_end_drawing.
/// The vertices_3d/2d and VP matrix from the game loop are still populated.
/// We render to a fresh surface texture with screenshot capture, producing
/// the same visual output as the real frame.
extern "C" fn bloom_screenshot_capture(out_len: *mut usize) -> *mut u8 {
    let eng = engine();

    // Set capture flag and render inline
    eng.renderer.screenshot_requested = true;
    eng.scene.prepare(
        &eng.renderer.device,
        &eng.renderer.queue,
        &eng.renderer.vp_matrix(),
        &eng.renderer.prev_vp_matrix,
        eng.renderer.uniform_3d_layout(),
    );
    eng.scene.prepare_materials(&eng.renderer);
    // Phase 1c: sync material PerFrame + PerView UBOs with the
    // current engine clock before the main HDR pass dispatches any
    // material draws that were submitted during this frame.
    {
        let t = eng.get_time() as f32;
        let dt = eng.delta_time as f32;
        eng.renderer.material_system_begin_frame(t, dt);
    }
    eng.renderer.end_frame_with_scene(&mut eng.scene, &mut eng.profiler);

    match eng.renderer.screenshot_data.take() {
        Some((width, height, rgba)) => {
            // Encode RGBA pixels to PNG
            match encode_png(width, height, &rgba) {
                Some(png_data) => {
                    let len = png_data.len();
                    // Allocate with libc::malloc (caller will free with libc::free)
                    let ptr = unsafe { libc::malloc(len) as *mut u8 };
                    if ptr.is_null() {
                        unsafe { *out_len = 0; }
                        return std::ptr::null_mut();
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(png_data.as_ptr(), ptr, len);
                        *out_len = len;
                    }
                    ptr
                }
                None => {
                    unsafe { *out_len = 0; }
                    std::ptr::null_mut()
                }
            }
        }
        None => {
            unsafe { *out_len = 0; }
            std::ptr::null_mut()
        }
    }
}

/// Minimal PNG encoder (no external dependency).
fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write;

    let mut png = Vec::new();
    // PNG signature
    png.write_all(&[137, 80, 78, 71, 13, 10, 26, 10]).ok()?;

    // IHDR chunk
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type: RGBA
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_png_chunk(&mut png, b"IHDR", &ihdr);

    // IDAT chunk — raw pixel data with zlib
    // Build raw scanlines: each row starts with filter byte 0 (None)
    let row_bytes = (width * 4) as usize;
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for y in 0..height as usize {
        raw.push(0); // filter: None
        let start = y * row_bytes;
        // Copy BGRA pixels, swapping B and R for PNG (which expects RGBA)
        for x in 0..width as usize {
            let idx = start + x * 4;
            // Metal Bgra8UnormSrgb: byte order is B, G, R, A
            raw.push(rgba[idx + 2]); // R (was at offset 2 in BGRA)
            raw.push(rgba[idx + 1]); // G (same position)
            raw.push(rgba[idx + 0]); // B (was at offset 0 in BGRA)
            raw.push(255);           // A (force opaque — alpha from sRGB surface is unreliable)
        }
    }

    // Compress with deflate (store blocks, no actual compression for simplicity)
    let deflated = deflate_store(&raw);
    write_png_chunk(&mut png, b"IDAT", &deflated);

    // IEND chunk
    write_png_chunk(&mut png, b"IEND", &[]);

    Some(png)
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    // CRC32 over type + data
    let crc = crc32(&[chunk_type.as_slice(), data].concat());
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Minimal deflate: store blocks (no compression). Wraps in zlib format.
fn deflate_store(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // Zlib header: CMF=0x78 (deflate, window=32K), FLG=0x01 (no dict, check bits)
    out.push(0x78);
    out.push(0x01);

    // Split into 65535-byte store blocks
    let mut remaining = data.len();
    let mut offset = 0;
    while remaining > 0 {
        let block_size = remaining.min(65535);
        let is_last = remaining <= 65535;
        out.push(if is_last { 1 } else { 0 }); // BFINAL + BTYPE=00 (store)
        let len = block_size as u16;
        let nlen = !len;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(&data[offset..offset + block_size]);
        offset += block_size;
        remaining -= block_size;
    }

    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// ============================================================
// Scene picking (raycasting)
// ============================================================

static mut LAST_PICK: Option<bloom_shared::picking::PickResult> = None;

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
pub extern "C" fn bloom_pick_hit_handle() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.handle).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn bloom_pick_hit_distance() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.distance as f64).unwrap_or(0.0) }
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

// Q6: Multi-hit picking — returns all hits sorted by distance.
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
// Physics (Jolt 5.x) — FFI surface generated from shared macro
// ============================================================

#[cfg(feature = "jolt")]
#[inline]
fn bloom_jolt_ffi_physics() -> &'static mut bloom_shared::physics_jolt::JoltPhysics {
    &mut engine().jolt
}

#[cfg(feature = "jolt")]
bloom_shared::define_physics_ffi!();
