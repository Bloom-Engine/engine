// `static mut` is intentional throughout this FFI surface — Perry calls
// us from a single OS thread (the macOS run-loop), so the engine
// singleton + scratch state never race. The 2024 lint flagging
// `&LAST_PICK`-style accesses is a real concern in multi-threaded
// code, but inapplicable here. Suppress at the crate root to avoid
// 16+ noise lines in every build.
#![allow(static_mut_refs)]

use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::{str_from_header, alloc_perry_string};

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
// Render half of the audio system. Moved here from EngineState by
// bloom_init_audio (AudioMixer::take_renderer) BEFORE the CoreAudio
// callback starts; after that it is owned exclusively by the audio
// render thread. The old design mixed through ENGINE from the callback —
// a cross-thread data race with every play/stop call on the main thread.
static mut AUDIO_RENDERER: Option<bloom_shared::audio::AudioRenderer> = None;

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}
/// Asset-path hook for define_core_ffi! — identity on desktop, where game
/// asset paths are valid relative to the working directory.
fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Borrowed(path)
}

// The full shared (non-physics) FFI surface. See bloom_shared::ffi_core
// docs for the contract; tools/validate-ffi.js checks parity in CI.
bloom_shared::define_core_ffi!();


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

    match AUDIO_RENDERER.as_mut() {
        Some(r) => r.mix(output),
        None => output.iter_mut().for_each(|s| *s = 0.0),
    }

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
        #[allow(deprecated)]
        app.activateIgnoringOtherApps(true);
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
                let event_type = event.r#type();
                match event_type {
                    NSEventType::KeyDown => {
                        let keycode = event.keyCode();
                        let bloom_key = map_keycode(keycode);
                        if bloom_key > 0 {
                            engine().input.set_key_down(bloom_key);
                        }
                        // Extract typed characters for text input (E3b).
                        let chars_obj = event.characters();
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
                        let keycode = event.keyCode();
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
                            let loc = event.locationInWindow();
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
                app.sendEvent(&event);
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

    // Apply cursor shape (Q2). resizeLeftRightCursor /
    // resizeUpDownCursor are deprecated in newer AppKit but still
    // function; the suggested replacements need a richer per-frame-
    // axis API we don't have. Keep + suppress the deprecation
    // warning until we land that.
    #[allow(deprecated)]
    match engine().input.cursor_shape {
        1 => objc2_app_kit::NSCursor::pointingHandCursor().set(),
        2 => objc2_app_kit::NSCursor::openHandCursor().set(),
        3 => objc2_app_kit::NSCursor::IBeamCursor().set(),
        4 => objc2_app_kit::NSCursor::resizeLeftRightCursor().set(),
        5 => objc2_app_kit::NSCursor::resizeUpDownCursor().set(),
        6 => objc2_app_kit::NSCursor::crosshairCursor().set(),
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

// ============================================================
// Input - Keyboard
// ============================================================

// ============================================================
// Input - Mouse
// ============================================================

// ============================================================
// Input - Gamepad
// ============================================================

// ============================================================
// Input - Touch
// ============================================================

// ============================================================
// Shapes
// ============================================================

// ============================================================
// Text
// ============================================================

// ============================================================
// Textures
// ============================================================

// ============================================================
// Camera 2D
// ============================================================

// ============================================================
// Camera 3D and 3D drawing
// ============================================================

// ============================================================
// Joint test
// ============================================================

// ============================================================
// Lighting
// ============================================================

// --- EN-005: procedural sky ---
// Toggle procedural-atmosphere rendering and steer the sun. The
// renderer owns the on/off flag + LUT state; setting the sun marks
// the sky-view LUT dirty so it re-bakes before the next frame.

// --- Post-FX knobs (heuristic visual layer; default-off) ---

// ============================================================
// Render quality toggles (individual + preset)
// ============================================================

// ============================================================
// Profiler — CPU phase timings (always available) + GPU timestamps
// (when the adapter supports TIMESTAMP_QUERY). Disabled by default.
// ============================================================

// ============================================================
// Models
// ============================================================

// ============================================================
// Phase 1c — material system FFI
// ============================================================

// ============================================================
// Audio
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    unsafe {
        // Hand the render half to the audio thread before the callback
        // can fire. Idempotent: a second init keeps the existing renderer.
        if AUDIO_RENDERER.is_none() {
            AUDIO_RENDERER = engine().audio.take_renderer();
        }
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

// ============================================================
// Music
// ============================================================

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
        Ok(mut clipboard) => match clipboard.get_text() {
            Ok(text) => alloc_perry_string(&text),
            Err(_) => alloc_perry_string(""),
        },
        Err(_) => alloc_perry_string(""),
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
        Some(path) => alloc_perry_string(&path.to_string_lossy()),
        None => alloc_perry_string(""),
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
        Some(path) => alloc_perry_string(&path.to_string_lossy()),
        None => alloc_perry_string(""),
    }
}

// ============================================================
// Input injection + platform detection
// ============================================================
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 1.0 }

/// Return the user's preferred OS language as a packed 2-letter code:
/// `c0 * 256 + c1`, where c0/c1 are the ASCII bytes of the lowercased
/// ISO-639 primary subtag (e.g. "en-US" -> "en" -> 101*256+110). The script
/// subtag is dropped (zh-Hans/zh-Hant both pack as "zh"); callers map that to
/// their supported variant. Falls back to "en" when no preference is set.
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    fn pack(code: &str) -> f64 {
        let lower = code.to_ascii_lowercase();
        let b = lower.as_bytes();
        if b.len() >= 2 { (b[0] as f64) * 256.0 + (b[1] as f64) } else { 101.0 * 256.0 + 110.0 }
    }
    let langs = objc2_foundation::NSLocale::preferredLanguages();
    match langs.firstObject() {
        Some(s) => pack(&s.to_string()),
        None => pack("en"),
    }
}

// ============================================================
// Frame callbacks
// ============================================================

// ============================================================
// Multiple lights
// ============================================================

// ============================================================
// Scene graph (retained mode)
// ============================================================

// ============================================================
// Scene graph QoL — Q4/Q5/Q6/Q7
// ============================================================

// ============================================================
// Geometry generation
// ============================================================

// ============================================================
// Shadow mapping
// ============================================================

// ============================================================
// Post-processing
// ============================================================

// ============================================================
// 3D→2D Projection (for UI overlays positioned in 3D space)
// ============================================================

/// Project a world-space 3D point to screen coordinates.
/// Returns screen X. Call bloom_project_y for Y. Returns -9999 if behind camera.

// ============================================================
// Thread-safe staging (for async asset loading via Perry threads)
// ============================================================

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


// Q6: Multi-hit picking — returns all hits sorted by distance.

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

