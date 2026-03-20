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
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8) {
    let title = str_from_header(title_ptr);
    let mtm = MainThreadMarker::from(unsafe { MainThreadMarker::new_unchecked() });

    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    let content_rect = NSRect::new(NSPoint::new(100.0, 100.0), NSSize::new(width, height));
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
    window.center();
    window.makeKeyAndOrderFront(None);
    unsafe { app.activateIgnoringOtherApps(true) };

    // Set up CAMetalLayer on the content view
    let content_view = window.contentView().expect("No content view");
    unsafe {
        let _: () = msg_send![&content_view, setWantsLayer: true];
    }

    // Create wgpu surface and renderer
    // wgpu expects the NSView pointer (not NSWindow) for AppKit
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });

    let surface = unsafe {
        let view_ptr = Retained::as_ptr(&content_view) as *mut std::ffi::c_void;
        let handle = AppKitWindowHandle::new(
            std::ptr::NonNull::new(view_ptr).unwrap()
        );
        let raw = RawWindowHandle::AppKit(handle);
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: raw_window_handle::RawDisplayHandle::AppKit(raw_window_handle::AppKitDisplayHandle::new()),
            raw_window_handle: raw,
        }).expect("Failed to create surface")
    };

    let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("No adapter found");

    let (device, queue) = pollster_block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bloom_device"),
            ..Default::default()
        },
        None,
    )).expect("Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps.formats.iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width as u32,
        height: height as u32,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let renderer = Renderer::new(device, queue, surface, surface_config);
    let engine_state = EngineState::new(renderer);

    unsafe {
        let _ = ENGINE.set(engine_state);
        WINDOW = Some(window);
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
                    _ => {}
                }
                unsafe { app.sendEvent(&event) };
            }
            None => break,
        }
    }

    // Check if window was closed
    if unsafe { WINDOW.as_ref().map(|w| !w.isVisible()).unwrap_or(true) } {
        engine().should_close = true;
    }

    // Handle window resize
    if let Some(window) = unsafe { &WINDOW } {
        if let Some(content_view) = window.contentView() {
            let frame = content_view.frame();
            let new_w = frame.size.width as u32;
            let new_h = frame.size.height as u32;
            let eng = engine();
            if new_w > 0 && new_h > 0 && (new_w != eng.renderer.width() || new_h != eng.renderer.height()) {
                eng.renderer.resize(new_w, new_h);
            }
        }
    }

    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    engine().end_frame();
}

#[no_mangle]
pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.set_clear_color(r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_set_target_fps(fps: f64) {
    engine().target_fps = fps;
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
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::new());
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
            let renderer_ptr = &mut eng.renderer as *mut Renderer;
            eng.textures.load_texture(unsafe { &mut *renderer_ptr }, &data)
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_texture(handle: f64) {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut Renderer;
    eng.textures.unload_texture(handle, unsafe { &mut *renderer_ptr });
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
    let renderer_ptr = &mut eng.renderer as *mut Renderer;
    eng.textures.load_texture_from_image(handle, unsafe { &mut *renderer_ptr })
}

#[no_mangle]
pub extern "C" fn bloom_gen_texture_mipmaps(_handle: f64) {
    // Mipmap generation is handled by the GPU texture creation pipeline
    // This is a no-op for now as wgpu handles mipmaps internally
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

// ============================================================
// Models
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            let eng = engine();
            let renderer_ptr = &mut eng.renderer as *mut Renderer;
            eng.models.load_model_with_textures(&data, unsafe { &mut *renderer_ptr })
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
    match std::fs::read(path) {
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
            let c_str = std::ffi::CString::new(contents).unwrap_or_default();
            c_str.into_raw() as *const u8
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
