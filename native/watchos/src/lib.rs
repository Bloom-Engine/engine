//! bloom-watchos: proof-of-life backend for the watchOS target.
//!
//! This crate intentionally does not render. It provides every `bloom_*` FFI
//! symbol Perry's manifest declares so that `perry compile --target
//! watchos-simulator --features watchos-game-loop` can link a `.app`, launches
//! on the watch simulator, runs the user's TS on Perry's game thread, and
//! routes Digital Crown + touch input back through `getCrownRotation()` /
//! `getTouchX/Y`. Drawing calls are no-ops — actual rendering waits on a
//! `dlopen`-based Metal path since the watchOS SDK doesn't ship
//! `Metal.framework` for static linking.

#![allow(non_upper_case_globals)]

mod ffi_stubs;

use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

// ============================================================
// Minimal input state (standalone — no bloom-shared dep to keep
// this crate free of wgpu / Metal linkage for the first cut)
// ============================================================

const MAX_TOUCH: usize = 4;

struct WatchState {
    // Bit-packed keys (unused on watch but the FFI needs backing storage)
    keys_down: [AtomicU64; 8],
    // Crown accumulator — radians since last consume; i64 as f64::to_bits
    crown_bits: AtomicU64,
    // Touch state
    touch_x: [AtomicU64; MAX_TOUCH], // f64 bits
    touch_y: [AtomicU64; MAX_TOUCH], // f64 bits
    touch_active: [AtomicU64; MAX_TOUCH], // 0 or 1
    // Screen geometry
    screen_w: AtomicU64,
    screen_h: AtomicU64,
    // Timing
    target_fps: AtomicI64,
    start: OnceLock<Instant>,
    last_frame: AtomicU64, // nanoseconds since start
    frame_count: AtomicUsize,
}

fn state() -> &'static WatchState {
    static S: OnceLock<WatchState> = OnceLock::new();
    S.get_or_init(|| WatchState {
        keys_down: std::array::from_fn(|_| AtomicU64::new(0)),
        crown_bits: AtomicU64::new(0),
        touch_x: std::array::from_fn(|_| AtomicU64::new(0)),
        touch_y: std::array::from_fn(|_| AtomicU64::new(0)),
        touch_active: std::array::from_fn(|_| AtomicU64::new(0)),
        screen_w: AtomicU64::new(198f64.to_bits()), // 40mm Apple Watch default
        screen_h: AtomicU64::new(242f64.to_bits()),
        target_fps: AtomicI64::new(30),
        start: OnceLock::new(),
        last_frame: AtomicU64::new(0),
        frame_count: AtomicUsize::new(0),
    })
}

fn now_nanos() -> u64 {
    let s = state();
    let start = s.start.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

// Accumulate crown rotation (radians). Called from Swift via the
// accumulate_crown export below.
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
// Exports for the watchOS Swift shell (see PerryWatchGameLoop.swift
// embedded as a literal in perry_scene_will_connect below)
// ============================================================

/// Called from Swift whenever the Digital Crown rotates.
#[no_mangle]
pub extern "C" fn bloom_watchos_crown_delta(delta: f64) {
    add_crown(delta);
}

/// Called from Swift on each touch event. active=0 on end.
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

/// Called from Swift after the hosting controller lays out its view.
#[no_mangle]
pub extern "C" fn bloom_watchos_set_screen(w: f64, h: f64) {
    let s = state();
    s.screen_w.store(w.to_bits(), Ordering::Release);
    s.screen_h.store(h.to_bits(), Ordering::Release);
}

// ============================================================
// Perry game-loop handshake
// ============================================================
//
// Perry's watchos_game_loop.rs registers a fallback delegate whose
// applicationDidFinishLaunching calls perry_scene_will_connect(NULL). We don't
// need to override that delegate — the fallback is enough for proof-of-life.
// We leave perry_register_native_classes as a no-op and let the fallback run.

#[no_mangle]
pub extern "C" fn perry_register_native_classes() {
    // Intentionally empty. Perry's watchos_game_loop fallback delegate is
    // sufficient to signal applicationDidFinishLaunching into the TS thread.
    // A future revision will register a WKHostingController + Metal view here.
}

#[no_mangle]
pub extern "C" fn perry_scene_will_connect(_scene: *const c_void) {
    // Proof-of-life: we don't attach a view in this cut. The game thread is
    // already running the user's TS code (spawned by perry main() before
    // WKApplicationMain was entered). Input will flow once a future Swift
    // shell installs the WKCrownSequencer + touch handlers and calls the
    // bloom_watchos_* inbound functions above.
}

// ============================================================
// Overridden bloom_* FFI — the ones the stubs generator skipped
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 8.0 }

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
pub extern "C" fn bloom_get_delta_time() -> f64 {
    let s = state();
    let now = now_nanos();
    let last = s.last_frame.swap(now, Ordering::AcqRel);
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
pub extern "C" fn bloom_init_window(_w: f64, _h: f64, _title: i64, _fullscreen: f64) {
    // no-op — the Swift shell owns the window
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 { 0.0 }

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    // Throttle to target fps. Without a Metal presentation step, the game
    // loop would otherwise busy-spin. Sleep to roughly match target fps.
    let fps = state().target_fps.load(Ordering::Acquire);
    if fps > 0 {
        let frame_ns = 1_000_000_000u64 / fps as u64;
        std::thread::sleep(std::time::Duration::from_nanos(frame_ns));
    }
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    state().frame_count.fetch_add(1, Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn bloom_clear_background(_r: f64, _g: f64, _b: f64, _a: f64) {}

#[no_mangle]
pub extern "C" fn bloom_run_game(_callback: f64) {
    // No-op. runGame()'s native path blocks in a while loop calling
    // beginDrawing/update/endDrawing — that's what Perry's game thread will
    // actually execute. This symbol only exists to satisfy the FFI manifest.
}

#[no_mangle]
pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
    let s = state();
    for i in 0..MAX_TOUCH {
        if s.touch_active[i].load(Ordering::Acquire) != 0 { return 1.0; }
    }
    if f64::from_bits(s.crown_bits.load(Ordering::Acquire)).abs() > 0.0 { return 1.0; }
    0.0
}

#[no_mangle]
pub extern "C" fn bloom_is_key_pressed(_key: f64) -> f64 { 0.0 }
#[no_mangle]
pub extern "C" fn bloom_is_key_down(_key: f64) -> f64 { 0.0 }
#[no_mangle]
pub extern "C" fn bloom_is_key_released(_key: f64) -> f64 { 0.0 }

#[no_mangle]
pub extern "C" fn bloom_set_target_fps(fps: f64) {
    state().target_fps.store(fps as i64, Ordering::Release);
}

#[no_mangle]
pub extern "C" fn bloom_inject_key_down(_key: f64) {}
#[no_mangle]
pub extern "C" fn bloom_inject_key_up(_key: f64) {}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_axis(_axis: f64, _value: f64) {}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_down(_btn: f64) {}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_up(_btn: f64) {}

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_available() -> f64 { 0.0 }
#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis(_axis: f64) -> f64 { 0.0 }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_pressed(_btn: f64) -> f64 { 0.0 }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_down(_btn: f64) -> f64 { 0.0 }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_released(_btn: f64) -> f64 { 0.0 }
#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis_count() -> f64 { 0.0 }
