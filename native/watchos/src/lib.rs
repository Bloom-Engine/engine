//! bloom-watchos: Canvas + (future) SceneKit backend for watchOS.
//!
//! The Swift shell (`BloomWatchApp.swift`, compiled by Perry via
//! `--features watchos-swift-app`) owns `@main`. During init it spawns the
//! game thread calling `_perry_user_main`. The root view is a SwiftUI
//! `Canvas` that drains this crate's draw list each frame. Digital Crown
//! rotation, taps, and layout dimensions come back in through the
//! `bloom_watchos_*` inbound functions.

#![allow(non_upper_case_globals)]

mod ffi_stubs;
mod ffi_stubs_manual;
mod draw_list;
mod textures;
mod audio;
mod scene;
mod models;
mod postfx;

/// Perry StringHeader layout — mirrors bloom-shared's copy. Inlined here
/// because we don't depend on bloom-shared (keeps the watchos crate
/// wgpu-free).
#[repr(C)]
struct StringHeader {
    utf16_len: u32,
    byte_len: u32,
    capacity: u32,
    refcount: u32,
    // Perry 0.5.x's canonical StringHeader is 5×u32 = 20 bytes (data at +20).
    // Omitting `flags` made this 16 bytes, so `perry_str` read every incoming
    // string 4 bytes early — text rendered with a 4-null prefix + truncated
    // tail ("BLOOM JUMP" → "BLOOM"), and file paths came back corrupted so
    // levels never loaded.
    flags: u32,
}

/// Decode a Perry-side string pointer (i64 on this ABI) into a borrowed &str.
/// Returns "" for null / too-small pointers to keep the FFI boundary robust.
fn perry_str<'a>(ptr: i64) -> &'a str {
    if ptr == 0 { return ""; }
    let p = ptr as *const u8;
    if (p as usize) < 0x1000 { return ""; }
    unsafe {
        let header = p as *const StringHeader;
        let len = (*header).byte_len as usize;
        let data = p.add(std::mem::size_of::<StringHeader>());
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(data, len))
    }
}

/// Allocate a Perry-side heap string (StringHeader + UTF-8 payload) and return
/// its pointer as the i64 the FFI boundary expects. Mirrors `perry_str`'s
/// layout so a string we hand back is read identically to one Perry passed us.
/// Returning this (never 0) for a `returns:"string"` function is mandatory:
/// Perry's inline `.length` dereferences the pointer, so a null segfaults the
/// caller.
fn alloc_perry_string(s: &str) -> i64 {
    let bytes = s.as_bytes();
    let byte_len = bytes.len();
    let utf16_len = if bytes.iter().all(|&b| b < 0x80) {
        byte_len
    } else {
        s.encode_utf16().count()
    };
    // A value RETURNED across the FFI boundary is consumed by Perry's internal
    // string machinery, whose canonical StringHeader is 5×u32 = 20 bytes
    // (utf16_len, byte_len, capacity, refcount, flags) with data at +20.
    // (The local 16-byte `StringHeader`/`perry_str` above describe the *incoming*
    // arg representation, which omits `flags`; don't conflate the two.)
    const HEADER_SIZE: usize = 20;
    let total = HEADER_SIZE + byte_len;
    unsafe {
        let layout = std::alloc::Layout::from_size_align(total, 4).unwrap();
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            return 0;
        }
        *(ptr.add(0) as *mut u32) = utf16_len as u32;
        *(ptr.add(4) as *mut u32) = byte_len as u32;
        *(ptr.add(8) as *mut u32) = byte_len as u32; // capacity
        *(ptr.add(12) as *mut u32) = 1; // refcount = unique
        *(ptr.add(16) as *mut u32) = 0; // flags
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(HEADER_SIZE), byte_len);
        ptr as i64
    }
}

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use draw_list::{kind, DrawCmd};

// ============================================================
// Minimal engine state (input + timing). Rendering state lives
// in the draw_list and textures modules.
// ============================================================

const MAX_TOUCH: usize = 4;

struct WatchState {
    crown_bits: AtomicU64, // f64 bits
    language_code: AtomicU64, // packed 2-letter code (c0*256+c1), set by Swift at launch
    touch_x: [AtomicU64; MAX_TOUCH],
    touch_y: [AtomicU64; MAX_TOUCH],
    touch_active: [AtomicU64; MAX_TOUCH],
    screen_w: AtomicU64, // f64 bits
    screen_h: AtomicU64, // f64 bits
    target_fps: AtomicI64,
    start: OnceLock<Instant>,
    last_frame_ns: AtomicU64,
    frame_count: AtomicUsize,
}

fn state() -> &'static WatchState {
    static S: OnceLock<WatchState> = OnceLock::new();
    S.get_or_init(|| WatchState {
        crown_bits: AtomicU64::new(0),
        language_code: AtomicU64::new(25966), // "en" until Swift reports the locale
        touch_x: std::array::from_fn(|_| AtomicU64::new(0)),
        touch_y: std::array::from_fn(|_| AtomicU64::new(0)),
        touch_active: std::array::from_fn(|_| AtomicU64::new(0)),
        screen_w: AtomicU64::new(198f64.to_bits()),
        screen_h: AtomicU64::new(242f64.to_bits()),
        target_fps: AtomicI64::new(30),
        start: OnceLock::new(),
        last_frame_ns: AtomicU64::new(0),
        frame_count: AtomicUsize::new(0),
    })
}

fn now_nanos() -> u64 {
    let s = state();
    let start = s.start.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn add_crown(delta: f64) {
    let s = state();
    loop {
        let cur = s.crown_bits.load(Ordering::Acquire);
        let new = (f64::from_bits(cur) + delta).to_bits();
        if s.crown_bits
            .compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            break;
        }
    }
}

fn consume_crown() -> f64 {
    let s = state();
    let bits = s.crown_bits.swap(0, Ordering::AcqRel);
    f64::from_bits(bits)
}

// ============================================================
// Swift → Rust inbound hooks. Called from BloomWatchApp.swift.
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_watchos_crown_delta(delta: f64) { add_crown(delta); }

/// Swift reports the user's preferred language at launch as a packed
/// 2-letter ISO-639 primary subtag (c0*256 + c1, lowercased), matching
/// the bloom_get_language contract on the other platforms.
#[no_mangle]
pub extern "C" fn bloom_watchos_set_language(code: f64) {
    let c = code as u64;
    if c > 0 {
        state().language_code.store(c, Ordering::Release);
    }
}

#[no_mangle]
pub extern "C" fn bloom_watchos_touch(index: i64, x: f64, y: f64, active: i64) {
    let s = state();
    let i = index as usize;
    if i < MAX_TOUCH {
        s.touch_x[i].store(x.to_bits(), Ordering::Release);
        s.touch_y[i].store(y.to_bits(), Ordering::Release);
        s.touch_active[i].store(active as u64, Ordering::Release);
    }
}

#[no_mangle]
pub extern "C" fn bloom_watchos_set_screen(w: f64, h: f64) {
    let s = state();
    s.screen_w.store(w.to_bits(), Ordering::Release);
    s.screen_h.store(h.to_bits(), Ordering::Release);
}

#[no_mangle]
pub extern "C" fn bloom_watchos_set_bundle_path(path: *const c_char) {
    if path.is_null() { return; }
    let s = unsafe { std::ffi::CStr::from_ptr(path) };
    if let Ok(str) = s.to_str() {
        textures::set_bundle_path(str);
    }
}

// Draw-list snapshot accessors for Swift Canvas.

#[no_mangle]
pub extern "C" fn bloom_watchos_frame_count() -> u64 { draw_list::frame() }

#[no_mangle]
pub extern "C" fn bloom_watchos_copy_draw_list(dst: *mut DrawCmd, max: i64) -> i64 {
    draw_list::copy_into(dst, max)
}

/// Returns [r,g,b,a] of the current clear color, 0-255 per channel, into a
/// Swift-supplied 4-element f64 buffer.
#[no_mangle]
pub extern "C" fn bloom_watchos_clear_color(out: *mut f64) {
    if out.is_null() { return; }
    let c = draw_list::clear_rgba();
    unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), out, 4); }
}

#[no_mangle]
pub extern "C" fn bloom_watchos_texture_path(handle: u32) -> *const c_char {
    textures::path_ptr(handle)
}

/// Copy camera state [px,py,pz, tx,ty,tz, ux,uy,uz, fovy, proj] into a
/// Swift-supplied 11-element f64 buffer. Called by the Swift SceneView.
#[no_mangle]
pub extern "C" fn bloom_watchos_camera_state(out: *mut f64) {
    draw_list::camera_snapshot(out);
}

/// 1 if the game has ever opened a 3D mode section — hints to the Swift
/// shell that it should host a SceneView layer underneath the Canvas.
#[no_mangle]
pub extern "C" fn bloom_watchos_has_3d() -> f64 {
    if draw_list::has_3d() { 1.0 } else { 0.0 }
}

// ============================================================
// Perry game-loop handshake.
// ============================================================
//
// With --features watchos-swift-app, Perry doesn't generate its own main() —
// BloomWatchApp.swift is @main. Perry's watchos runtime still provides the
// `perry_main_init` C symbol which does global TS init; the Swift App calls
// it before spawning the game thread. We don't need to register any ObjC
// classes from this crate.

#[no_mangle]
pub extern "C" fn perry_register_native_classes() {}

#[no_mangle]
pub extern "C" fn perry_scene_will_connect(_scene: *const c_void) {}

// ============================================================
// Platform / timing / input overrides
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 8.0 }

/// Preferred OS language packed as `c0*256+c1`. TODO: real per-OS detection; returns "en" for now.
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    state().language_code.load(Ordering::Acquire) as f64
}

#[no_mangle]
pub extern "C" fn bloom_get_crown_rotation() -> f64 { consume_crown() }

#[no_mangle]
pub extern "C" fn bloom_get_screen_width() -> f64 {
    f64::from_bits(state().screen_w.load(Ordering::Acquire))
}

#[no_mangle]
pub extern "C" fn bloom_get_screen_height() -> f64 {
    f64::from_bits(state().screen_h.load(Ordering::Acquire))
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_x(index: f64) -> f64 {
    let s = state();
    let i = index as usize;
    if i < MAX_TOUCH { f64::from_bits(s.touch_x[i].load(Ordering::Acquire)) } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_y(index: f64) -> f64 {
    let s = state();
    let i = index as usize;
    if i < MAX_TOUCH { f64::from_bits(s.touch_y[i].load(Ordering::Acquire)) } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_count() -> f64 {
    let s = state();
    let mut n = 0.0;
    for i in 0..MAX_TOUCH {
        if s.touch_active[i].load(Ordering::Acquire) != 0 { n += 1.0; }
    }
    n
}

#[no_mangle]
pub extern "C" fn bloom_is_touch_active(index: f64) -> f64 {
    let s = state();
    let i = index as usize;
    if i < MAX_TOUCH && s.touch_active[i].load(Ordering::Acquire) != 0 { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_max_touch_points() -> f64 {
    MAX_TOUCH as f64
}

#[no_mangle]
pub extern "C" fn bloom_get_delta_time() -> f64 {
    let s = state();
    let now = now_nanos();
    let last = s.last_frame_ns.swap(now, Ordering::AcqRel);
    if last == 0 { 1.0 / 30.0 } else { (now - last) as f64 / 1_000_000_000.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_time() -> f64 {
    now_nanos() as f64 / 1_000_000_000.0
}

#[no_mangle]
pub extern "C" fn bloom_get_fps() -> f64 {
    state().target_fps.load(Ordering::Acquire) as f64
}

#[no_mangle]
pub extern "C" fn bloom_set_target_fps(fps: f64) {
    state().target_fps.store(fps as i64, Ordering::Release);
}

// No-op on watchOS (no deferred 3D pipeline); kept to satisfy the FFI
// surface so cross-platform game code can call setDirect2DMode() freely.
#[no_mangle]
pub extern "C" fn bloom_set_direct_2d_mode(_on: f64) {}

#[no_mangle]
pub extern "C" fn bloom_init_window(_w: f64, _h: f64, _title: i64, _fullscreen: f64) {}

#[no_mangle]
pub extern "C" fn bloom_close_window() {}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 { 0.0 }

// ============================================================
// Drawing lifecycle
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    // Pace the game thread to target fps. Canvas refresh is driven by the
    // frame counter change on the Swift side, so this sleep also controls
    // apparent frame rate in the view.
    let fps = state().target_fps.load(Ordering::Acquire);
    if fps > 0 {
        let frame_ns = 1_000_000_000u64 / fps as u64;
        std::thread::sleep(std::time::Duration::from_nanos(frame_ns));
    }
    draw_list::begin();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    draw_list::end();
    state().frame_count.fetch_add(1, Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
    draw_list::set_clear(r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_run_game(_callback: f64) {}

#[no_mangle]
pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
    let s = state();
    for i in 0..MAX_TOUCH {
        if s.touch_active[i].load(Ordering::Acquire) != 0 { return 1.0; }
    }
    if f64::from_bits(s.crown_bits.load(Ordering::Acquire)).abs() > 0.0 { return 1.0; }
    0.0
}

#[no_mangle] pub extern "C" fn bloom_is_key_pressed(_k: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_is_key_down(_k: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_is_key_released(_k: f64) -> f64 { 0.0 }

#[no_mangle] pub extern "C" fn bloom_inject_key_down(_k: f64) {}
#[no_mangle] pub extern "C" fn bloom_inject_key_up(_k: f64) {}
#[no_mangle] pub extern "C" fn bloom_inject_gamepad_axis(_a: f64, _v: f64) {}
#[no_mangle] pub extern "C" fn bloom_inject_gamepad_button_down(_b: f64) {}
#[no_mangle] pub extern "C" fn bloom_inject_gamepad_button_up(_b: f64) {}

#[no_mangle] pub extern "C" fn bloom_is_gamepad_available() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_get_gamepad_axis(_a: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_is_gamepad_button_pressed(_b: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_is_gamepad_button_down(_b: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_is_gamepad_button_released(_b: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_get_gamepad_axis_count() -> f64 { 0.0 }

// ============================================================
// 2D shape commands → draw list
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_draw_rect(x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
    let mut c = DrawCmd::zero();
    c.kind = kind::RECT;
    c.x = x; c.y = y; c.w = w; c.h = h;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_rect_lines(x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    let mut c = DrawCmd::zero();
    c.kind = kind::RECT_LINES;
    c.x = x; c.y = y; c.w = w; c.h = h;
    c.thickness = thickness;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle(cx: f64, cy: f64, rad: f64, r: f64, g: f64, b: f64, a: f64) {
    let mut c = DrawCmd::zero();
    c.kind = kind::CIRCLE;
    c.x = cx; c.y = cy; c.w = rad; c.h = rad;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

// 2D camera: emit marker commands carrying the camera so the Swift Canvas can
// apply the matching affine transform to every world-space draw until end.
// world→screen is (world - target) * zoom + offset. Without this, gameplay
// tiles/sprites (drawn between begin/end) render at raw world coords off-screen.
#[no_mangle]
pub extern "C" fn bloom_begin_mode_2d(ox: f64, oy: f64, tx: f64, ty: f64, _rot: f64, zoom: f64) {
    let mut c = DrawCmd::zero();
    c.kind = kind::BEGIN_2D;
    c.x = ox; c.y = oy;   // screen offset
    c.w = tx; c.h = ty;   // world target
    c.size = zoom;        // zoom
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_end_mode_2d() {
    let mut c = DrawCmd::zero();
    c.kind = kind::END_2D;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle_lines(cx: f64, cy: f64, rad: f64, r: f64, g: f64, b: f64, a: f64) {
    // ABI is (cx, cy, radius, r, g, b, a) — no thickness param (matches
    // package.json + src/shapes). A phantom 4th `thickness` param here
    // used to shift every color one slot right on watchOS.
    let mut c = DrawCmd::zero();
    c.kind = kind::CIRCLE_LINES;
    c.x = cx; c.y = cy; c.w = rad; c.h = rad;
    c.thickness = 1.0;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_line(x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    let mut c = DrawCmd::zero();
    c.kind = kind::LINE;
    c.x = x1; c.y = y1;
    c.src_x = x2; c.src_y = y2;
    c.thickness = thickness;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
    let mut c = DrawCmd::zero();
    c.kind = kind::TRIANGLE;
    c.x = x1; c.y = y1;
    c.src_x = x2; c.src_y = y2;
    c.src_w = x3; c.src_h = y3;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_poly(_cx: f64, _cy: f64, _sides: f64, _radius: f64, _rot: f64, _r: f64, _g: f64, _b: f64, _a: f64) {
    // Polygon fill — defer; Canvas handles via context.fill(Path.addArc).
    // Will land with the rest of the Canvas geometry work.
}

// ============================================================
// Texture commands
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_texture(path: i64) -> f64 {
    textures::load(perry_str(path)) as f64
}

#[no_mangle]
pub extern "C" fn bloom_unload_texture(_handle: f64) {
    // Retain by handle — unload is a no-op on watch (reloading isn't common
    // in Jump-class games; we keep cached CGImages on the Swift side for
    // the life of the process).
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_width(handle: f64) -> f64 {
    textures::width(handle as u32) as f64
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_height(handle: f64) -> f64 {
    textures::height(handle as u32) as f64
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture(handle: f64, x: f64, y: f64, r: f64, g: f64, b: f64, a: f64) {
    let h = handle as u32;
    if h == 0 { return; }
    let mut c = DrawCmd::zero();
    c.kind = kind::TEXTURE;
    c.tex = h;
    c.x = x; c.y = y;
    c.w = textures::width(h) as f64;
    c.h = textures::height(h) as f64;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_rec(handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let h = handle as u32;
    if h == 0 { return; }
    let mut c = DrawCmd::zero();
    c.kind = kind::TEXTURE_REC;
    c.tex = h;
    c.x = dst_x; c.y = dst_y;
    c.w = src_w; c.h = src_h;
    c.src_x = src_x; c.src_y = src_y; c.src_w = src_w; c.src_h = src_h;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_pro(handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64,
    ox: f64, oy: f64, rot: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let h = handle as u32;
    if h == 0 { return; }
    let mut c = DrawCmd::zero();
    c.kind = kind::TEXTURE_PRO;
    c.tex = h;
    c.x = dst_x; c.y = dst_y; c.w = dst_w; c.h = dst_h;
    c.src_x = src_x; c.src_y = src_y; c.src_w = src_w; c.src_h = src_h;
    c.ox = ox; c.oy = oy; c.rot = rot;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_set_texture_filter(_handle: f64, _mode: f64) {}

// ============================================================
// Text
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_draw_text(text: i64, x: f64, y: f64, size: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    push_text(perry_str(text), x, y, size, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_text_ex(_font: f64, text: i64,
    x: f64, y: f64, size: f64, _spacing: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    push_text(perry_str(text), x, y, size, r, g, b, a);
}

fn push_text(s: &str, x: f64, y: f64, size: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    if s.is_empty() { return; }
    let mut c = DrawCmd::zero();
    c.kind = kind::TEXT;
    c.x = x; c.y = y;
    c.size = size;
    c.r = r; c.g = g; c.b = b; c.a = a;
    c.set_text(s);
    draw_list::push(c);
}

/// Crude text width estimate — fontSize * 0.55 * char_count. Good enough for
/// HUD layout; real glyph metrics would require calling back into Swift
/// (CoreText) or bundling a font parser.
#[no_mangle]
pub extern "C" fn bloom_measure_text(text: i64, size: f64) -> f64 {
    perry_str(text).chars().count() as f64 * size * 0.55
}

#[no_mangle]
pub extern "C" fn bloom_measure_text_ex(_font: f64, text: i64, size: f64, _spacing: f64) -> f64 {
    bloom_measure_text(text, size)
}

#[no_mangle]
pub extern "C" fn bloom_load_font(_path: i64, _size: f64) -> f64 {
    1.0 // All text uses the system font on watch; single sentinel handle.
}

#[no_mangle]
pub extern "C" fn bloom_unload_font(_handle: f64) {}

// ============================================================
// File I/O — minimal std::fs-backed implementations
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_file_exists(path: i64) -> f64 {
    let p = perry_str(path);
    if p.is_empty() { return 0.0; }
    // Resolve bundle-relative paths the same way textures does.
    let full = if p.starts_with('/') { p.to_string() } else { textures::resolve_bundle_path(p) };
    if std::path::Path::new(&full).exists() { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_read_file(path: i64) -> i64 {
    let p = perry_str(path);
    if p.is_empty() {
        return alloc_perry_string("");
    }
    // Resolve bundle-relative paths the same way bloom_file_exists / textures do.
    let full = if p.starts_with('/') {
        p.to_string()
    } else {
        textures::resolve_bundle_path(p)
    };
    // Always hand back a valid StringHeader — empty on a miss, never null.
    match std::fs::read_to_string(&full) {
        Ok(contents) => alloc_perry_string(&contents),
        Err(_) => alloc_perry_string(""),
    }
}

#[no_mangle]
pub extern "C" fn bloom_write_file(_path: i64, _data: i64) -> f64 {
    0.0 // watchOS app sandbox — defer real write support.
}

// ============================================================
// 3D immediate-mode primitives
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_begin_mode_3d(
    px: f64, py: f64, pz: f64,
    tx: f64, ty: f64, tz: f64,
    ux: f64, uy: f64, uz: f64,
    fovy: f64, proj: f64,
) {
    draw_list::set_camera(px, py, pz, tx, ty, tz, ux, uy, uz, fovy, proj);
}

#[no_mangle]
pub extern "C" fn bloom_end_mode_3d() {
    // No-op: 3D commands are already in the list; the camera stays set until
    // the next begin_mode_3d. Swift's SceneView draws 3D cmds under the
    // Canvas layer each frame.
}

#[no_mangle]
pub extern "C" fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let mut c = DrawCmd::zero();
    c.kind = kind::CUBE;
    c.x = x; c.y = y; c.src_x = z; // src_x doubles as z on 3D cmds
    c.w = w; c.h = h; c.size = d;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let mut c = DrawCmd::zero();
    c.kind = kind::CUBE_WIRES;
    c.x = x; c.y = y; c.src_x = z; // src_x doubles as z on 3D cmds
    c.w = w; c.h = h; c.size = d;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_sphere(x: f64, y: f64, z: f64, radius: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let mut c = DrawCmd::zero();
    c.kind = kind::SPHERE;
    c.x = x; c.y = y; c.src_x = z; // src_x doubles as z on 3D cmds
    c.w = radius;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_sphere_wires(x: f64, y: f64, z: f64, radius: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let mut c = DrawCmd::zero();
    c.kind = kind::SPHERE_WIRES;
    c.x = x; c.y = y; c.src_x = z; // src_x doubles as z on 3D cmds
    c.w = radius;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_cylinder(x: f64, y: f64, z: f64,
    radius_top: f64, _radius_bottom: f64, height: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    // ABI is 10 params (no slices) — that's bloom_draw_cylinder_ex. The
    // phantom `slices` param shifted every color one slot right.
    let mut c = DrawCmd::zero();
    c.kind = kind::CYLINDER;
    c.x = x; c.y = y; c.src_x = z; // src_x doubles as z on 3D cmds
    c.w = radius_top; c.h = height;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

#[no_mangle]
pub extern "C" fn bloom_draw_plane(x: f64, y: f64, z: f64, w: f64, h: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let mut c = DrawCmd::zero();
    c.kind = kind::PLANE;
    c.x = x; c.y = y; c.src_x = z; // src_x doubles as z on 3D cmds
    c.w = w; c.h = h;
    c.r = r; c.g = g; c.b = b; c.a = a;
    draw_list::push(c);
}

// ============================================================
// Audio (forwards to Swift BloomAudioManager)
// ============================================================

#[no_mangle] pub extern "C" fn bloom_init_audio() { audio::init_audio(); }
#[no_mangle] pub extern "C" fn bloom_close_audio() { audio::close_audio(); }

#[no_mangle]
pub extern "C" fn bloom_load_sound(path: i64) -> f64 {
    audio::load_sound(perry_str(path)) as f64
}
#[no_mangle] pub extern "C" fn bloom_play_sound(handle: f64) { audio::play_sound(handle as u32); }
#[no_mangle] pub extern "C" fn bloom_stop_sound(handle: f64) { audio::stop_sound(handle as u32); }
#[no_mangle] pub extern "C" fn bloom_set_sound_volume(handle: f64, v: f64) {
    audio::set_sound_volume(handle as u32, v as f32);
}
#[no_mangle] pub extern "C" fn bloom_set_master_volume(v: f64) {
    audio::set_master_volume(v as f32);
}
#[no_mangle] pub extern "C" fn bloom_play_sound_3d(handle: f64, _x: f64, _y: f64, _z: f64) {
    // No spatial audio on watch — play at full 2D volume.
    audio::play_sound(handle as u32);
}

#[no_mangle]
pub extern "C" fn bloom_load_music(path: i64) -> f64 {
    audio::load_music(perry_str(path)) as f64
}
#[no_mangle] pub extern "C" fn bloom_play_music(handle: f64) { audio::play_music(handle as u32); }
#[no_mangle] pub extern "C" fn bloom_stop_music(handle: f64) { audio::stop_music(handle as u32); }
#[no_mangle] pub extern "C" fn bloom_set_music_volume(handle: f64, v: f64) {
    audio::set_music_volume(handle as u32, v as f32);
}
#[no_mangle] pub extern "C" fn bloom_is_music_playing(handle: f64) -> f64 {
    if audio::is_music_playing(handle as u32) { 1.0 } else { 0.0 }
}
#[no_mangle] pub extern "C" fn bloom_update_music_stream(_handle: f64) {
    // AVAudioPlayer owns stream pumping internally — no per-frame poke needed.
}

// ============================================================
// Retained scene graph (bloom_scene_*) — synced to SCNNodes in Swift
// ============================================================

#[no_mangle] pub extern "C" fn bloom_scene_create_node() -> f64 { scene::create() as f64 }
#[no_mangle] pub extern "C" fn bloom_scene_destroy_node(h: f64) { scene::destroy(h as u32); }
#[no_mangle] pub extern "C" fn bloom_scene_set_visible(h: f64, v: f64) {
    scene::set_visible(h as u32, v > 0.5);
}
#[no_mangle] pub extern "C" fn bloom_scene_set_cast_shadow(_h: f64, _v: f64) {
    // SceneKit manages shadow casting via per-node + light config; deferred.
}
#[no_mangle] pub extern "C" fn bloom_scene_set_receive_shadow(_h: f64, _v: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_parent(h: f64, parent: f64) {
    scene::set_parent(h as u32, parent as u32);
}

/// Transform is a raw pointer to 16 column-major f64s (128 bytes). No
/// StringHeader wrapping — bloom's TS side passes a `number[]` which Perry
/// forwards as a pointer to the Float64 data.
#[no_mangle]
pub extern "C" fn bloom_scene_set_transform(handle: f64, matrix_ptr: i64) {
    if matrix_ptr == 0 { return; }
    let src = unsafe { std::slice::from_raw_parts(matrix_ptr as *const f64, 16) };
    let mut arr = [0.0f32; 16];
    for i in 0..16 { arr[i] = src[i] as f32; }
    scene::set_transform(handle as u32, arr);
}

/// Geometry: vert_ptr points to `vertex_count * 12` f64s (12 floats per
/// vertex: xyz, nx ny nz, rgba, uv). idx_ptr points to `index_count` f64s
/// (each integer index boxed as f64 — matches bloom's TS `number[]` calls).
#[no_mangle]
pub extern "C" fn bloom_scene_update_geometry(
    handle: f64, verts_ptr: i64, vertex_count: f64,
    idx_ptr: i64, index_count: f64,
) {
    if verts_ptr == 0 || idx_ptr == 0 { return; }
    let vcount = vertex_count as usize;
    let icount = index_count as usize;
    let verts = unsafe { std::slice::from_raw_parts(verts_ptr as *const f64, vcount * 12) };
    let idx_f64 = unsafe { std::slice::from_raw_parts(idx_ptr as *const f64, icount) };
    scene::update_geometry_f64(handle as u32, verts, idx_f64);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_color(h: f64, r: f64, g: f64, b: f64, a: f64) {
    scene::set_color(h as u32, [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, a as f32 / 255.0]);
}
#[no_mangle]
pub extern "C" fn bloom_scene_set_material_pbr(h: f64, roughness: f64, metalness: f64) {
    scene::set_pbr(h as u32, roughness as f32, metalness as f32);
}
#[no_mangle]
pub extern "C" fn bloom_scene_set_material_texture(h: f64, tex: f64) {
    scene::set_texture(h as u32, tex as u32);
}
#[no_mangle] pub extern "C" fn bloom_scene_node_count() -> f64 { scene::node_count() as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_transform(handle: f64, out: f64) -> f64 {
    // `out` is a pointer to 16 f32s the caller has allocated. Write them.
    if out == 0.0 { return 0.0; }
    let t = scene::get_transform(handle as u32);
    unsafe {
        std::ptr::copy_nonoverlapping(t.as_ptr(), out as usize as *mut f32, 16);
    }
    1.0
}

// ============================================================
// Models — glTF (.glb) loading + scene attachment
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_model(path: i64) -> f64 {
    let p = perry_str(path);
    if p.is_empty() { return 0.0; }
    let full = if p.starts_with('/') { p.to_string() } else { textures::resolve_bundle_path(p) };
    models::load(&full) as f64
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_cube(w: f64, h: f64, d: f64) -> f64 {
    models::gen_cube_mesh(w as f32, h as f32, d as f32) as f64
}

/// Attach a loaded model to a scene node. If the model has a glTF scene
/// hierarchy, the full subtree is spawned as bloom scene-node descendants of
/// `node_handle` with each glTF node's TRS transform baked in. For
/// single-node models this collapses to the pre-#15 behavior (the target
/// node gets the one mesh's first primitive, sibling primitives become its
/// children).
///
/// The `mesh_idx` parameter is only honored when the model has no scene
/// hierarchy — it picks which mesh to instantiate into `node_handle`.
#[no_mangle]
pub extern "C" fn bloom_scene_attach_model(node_handle: f64, model_handle: f64, mesh_idx: f64) {
    let Some(model) = models::get(model_handle as u32) else { return; };
    let root = node_handle as u32;

    // If the model has a scene graph with non-trivial hierarchy (>1 node or a
    // transform on the root), walk it. Otherwise fall back to single-mesh
    // mode so mesh_idx still picks which mesh to attach.
    let single_mesh = model.nodes.len() <= 1
        && model.scene_roots.len() <= 1
        && model.nodes.first().map(|n| n.matrix.is_none()
            && n.translation == [0.0; 3]
            && n.rotation == [0.0, 0.0, 0.0, 1.0]
            && n.scale == [1.0; 3]).unwrap_or(true);

    if single_mesh {
        let mesh_i = mesh_idx as usize;
        let Some(mesh) = model.meshes.get(mesh_i) else { return; };
        attach_mesh(root, mesh);
        return;
    }

    // Scene-hierarchy path: one bloom scene node per glTF node. The attach
    // target `root` itself doesn't gain geometry — it's the hierarchy's
    // anchor. Children materialize via scene::create + scene::set_parent.
    // We build a gltf_node_idx → bloom_handle map during the walk, then
    // resolve skin joint references once everyone's in place.
    let mut gltf_to_bloom: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut skinned_nodes: Vec<(u32, usize)> = Vec::new();  // (bloom_handle, gltf_node_idx)
    for scene_root in &model.scene_roots {
        spawn_node_subtree(&model, *scene_root, root, &mut gltf_to_bloom, &mut skinned_nodes);
    }
    // Second pass: attach skin data to each skinned mesh-node. Joints are
    // resolved to the bloom handles we just created.
    for (bloom_handle, gltf_idx) in skinned_nodes {
        let gltf_node = &model.nodes[gltf_idx];
        let Some(skin_idx) = gltf_node.skin else { continue };
        let Some(skin) = model.skins.get(skin_idx) else { continue };
        let Some(mesh_idx) = gltf_node.mesh else { continue };
        let Some(mesh) = model.meshes.get(mesh_idx) else { continue };
        let Some(prim) = mesh.primitives.first() else { continue };

        let joint_handles: Vec<u32> = skin.joints.iter()
            .map(|&gi| *gltf_to_bloom.get(&gi).unwrap_or(&0))
            .collect();
        scene::set_skin(bloom_handle,
            joint_handles,
            skin.inverse_bind_matrices.clone(),
            prim.joint_indices.clone(),
            prim.weights.clone(),
        );

        // Animation tracks — resolve channel targets to bloom handles.
        // Auto-select animation 0 for now (typically the idle/survey clip).
        // Follow-up: bloom_update_model_animation to switch between them.
        if let Some(anim) = model.animations.first() {
            let mut tracks: Vec<scene::AnimTrack> = Vec::new();
            for ch in &anim.channels {
                let Some(&bone_h) = gltf_to_bloom.get(&ch.target_node) else { continue };
                if bone_h == 0 { continue; }
                tracks.push(scene::AnimTrack {
                    bone_handle: bone_h,
                    path: ch.path as u32,
                    times: ch.times.clone(),
                    values: ch.values.clone(),
                });
            }
            if !tracks.is_empty() {
                scene::set_anim_tracks(bloom_handle, tracks);
            }
        }
    }
}

fn attach_mesh(handle: u32, mesh: &models::Mesh) {
    if mesh.primitives.is_empty() { return; }
    apply_primitive(handle, &mesh.primitives[0]);
    for prim in mesh.primitives.iter().skip(1) {
        let child = scene::create();
        scene::set_parent(child, handle);
        apply_primitive(child, prim);
    }
}

fn spawn_node_subtree(
    model: &models::Model, node_idx: usize, parent: u32,
    gltf_to_bloom: &mut std::collections::HashMap<usize, u32>,
    skinned_nodes: &mut Vec<(u32, usize)>,
) {
    let Some(gltf_node) = model.nodes.get(node_idx) else { return; };
    let bloom_node = scene::create();
    gltf_to_bloom.insert(node_idx, bloom_node);
    scene::set_parent(bloom_node, parent);
    scene::set_transform(bloom_node, gltf_node.local_transform());
    if let Some(mi) = gltf_node.mesh {
        if let Some(mesh) = model.meshes.get(mi) {
            attach_mesh(bloom_node, mesh);
            if gltf_node.skin.is_some() {
                skinned_nodes.push((bloom_node, node_idx));
            }
        }
    }
    for &child_idx in &gltf_node.children {
        spawn_node_subtree(model, child_idx, bloom_node, gltf_to_bloom, skinned_nodes);
    }
}

fn apply_primitive(handle: u32, prim: &models::Primitive) {
    scene::set_geometry(handle,
        prim.positions.clone(), prim.normals.clone(),
        prim.uvs.clone(), prim.indices.clone());
    scene::set_color(handle, prim.color);
    scene::set_pbr(handle, prim.roughness, prim.metallic);
    scene::set_pbr_textures(handle,
        prim.tex_base_color, prim.tex_normal,
        prim.tex_metallic_roughness, prim.tex_emissive, prim.tex_occlusion);
}

// ============================================================
// Post-processing — stored for Swift to translate into SwiftUI modifiers
// ============================================================

#[no_mangle] pub extern "C" fn bloom_enable_postfx() { postfx::set_enabled(true); }
#[no_mangle] pub extern "C" fn bloom_disable_postfx() { postfx::set_enabled(false); }
#[no_mangle] pub extern "C" fn bloom_set_vignette(strength: f64, softness: f64) {
    postfx::set_vignette(strength, softness);
}
#[no_mangle] pub extern "C" fn bloom_set_chromatic_aberration(strength: f64) {
    postfx::set_chromatic_aberration(strength);
}
#[no_mangle] pub extern "C" fn bloom_set_film_grain(strength: f64) {
    postfx::set_film_grain(strength);
}
#[no_mangle] pub extern "C" fn bloom_set_manual_exposure(v: f64) { postfx::set_exposure(v); }
#[no_mangle] pub extern "C" fn bloom_set_auto_exposure(on: f64) {
    postfx::set_auto_exposure(on > 0.5);
}
#[no_mangle] pub extern "C" fn bloom_set_sun_shafts(
    strength: f64, decay: f64, r: f64, g: f64, b: f64,
) {
    postfx::set_sun_shafts(strength, decay, r, g, b);
}

#[no_mangle]
pub extern "C" fn bloom_watchos_postfx_state(out: *mut postfx::PostFxState) {
    postfx::snapshot(out);
}

#[no_mangle] pub extern "C" fn bloom_add_directional_light(
    dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64,
) {
    scene::add_directional_light(dx as f32, dy as f32, dz as f32,
                                 r as f32, g as f32, b as f32, intensity as f32);
}

#[no_mangle] pub extern "C" fn bloom_add_point_light(
    x: f64, y: f64, z: f64, range: f64, r: f64, g: f64, b: f64, intensity: f64,
) {
    scene::add_point_light(x as f32, y as f32, z as f32, range as f32,
                           r as f32, g as f32, b as f32, intensity as f32);
}

// ============================================================
// Scene snapshot accessors for Swift
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_watchos_scene_drain_dirty(dst: *mut scene::SceneNodeInfo, max: i64) -> i64 {
    scene::drain_dirty(dst, max)
}

#[no_mangle]
pub extern "C" fn bloom_watchos_scene_drain_destroyed(dst: *mut u32, max: i64) -> i64 {
    scene::drain_destroyed(dst, max)
}

#[no_mangle]
pub extern "C" fn bloom_watchos_scene_copy_lights(dst: *mut scene::Light, max: i64) -> i64 {
    scene::copy_lights(dst, max)
}

#[no_mangle]
pub extern "C" fn bloom_watchos_scene_geometry(handle: u32, out: *mut scene::GeometryPtrs) {
    if out.is_null() { return; }
    unsafe { *out = scene::geometry_ptrs(handle); }
}

#[no_mangle]
pub extern "C" fn bloom_watchos_scene_skin(handle: u32, out: *mut scene::SkinPtrs) {
    if out.is_null() { return; }
    unsafe { *out = scene::skin_ptrs(handle); }
}

#[no_mangle]
pub extern "C" fn bloom_watchos_scene_anim_track_count(handle: u32) -> i64 {
    scene::anim_track_count(handle)
}

#[no_mangle]
pub extern "C" fn bloom_watchos_scene_anim_track_info(handle: u32, idx: i64, out: *mut scene::AnimTrackInfo) {
    scene::anim_track_info(handle, idx, out);
}

#[no_mangle]
pub extern "C" fn bloom_draw_grid(slices: f64, spacing: f64) {
    let mut c = DrawCmd::zero();
    c.kind = kind::GRID;
    c.w = slices;
    c.h = spacing;
    // Grid color — light gray, matches bloom's default.
    c.r = 200.0; c.g = 200.0; c.b = 200.0; c.a = 255.0;
    draw_list::push(c);
}
