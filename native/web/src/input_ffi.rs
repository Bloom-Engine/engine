//! Input FFI surface for web: keyboard/mouse/gamepad/touch getters plus
//! the injection entry points the JS glue calls from DOM event
//! listeners. Split from lib.rs (2000-line file policy).

use crate::engine;
use wasm_bindgen::prelude::*;

// ============================================================
// Input - Keyboard
// ============================================================

#[wasm_bindgen]
pub fn bloom_is_key_pressed(key: f64) -> f64 {
    if engine().input.is_key_pressed(key as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_key_down(key: f64) -> f64 {
    if engine().input.is_key_down(key as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_key_released(key: f64) -> f64 {
    if engine().input.is_key_released(key as usize) { 1.0 } else { 0.0 }
}

// ============================================================
// Input - Mouse
// ============================================================

#[wasm_bindgen]
pub fn bloom_get_mouse_x() -> f64 {
    engine().input.mouse_x
}

#[wasm_bindgen]
pub fn bloom_get_mouse_y() -> f64 {
    engine().input.mouse_y
}

#[wasm_bindgen]
pub fn bloom_is_mouse_button_pressed(btn: f64) -> f64 {
    if engine().input.is_mouse_button_pressed(btn as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_mouse_button_down(btn: f64) -> f64 {
    if engine().input.is_mouse_button_down(btn as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_mouse_button_released(btn: f64) -> f64 {
    if engine().input.is_mouse_button_released(btn as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_mouse_delta_x() -> f64 {
    engine().input.mouse_delta_x
}

#[wasm_bindgen]
pub fn bloom_get_mouse_delta_y() -> f64 {
    engine().input.mouse_delta_y
}

// Accumulated scroll wheel delta since the last call. Reading consumes the
// value. Used by the editor's orbit camera and any scrollable UI panel.
#[wasm_bindgen]
pub fn bloom_get_mouse_wheel() -> f64 {
    engine().input.consume_mouse_wheel()
}

#[wasm_bindgen]
pub fn bloom_get_char_pressed() -> f64 {
    engine().input.pop_char() as f64
}

// Model bounds accessors. Return the axis-aligned bounding box of a loaded
// model in model-local coordinates.
#[wasm_bindgen]
pub fn bloom_get_model_bounds_min_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[0] as f64
}
#[wasm_bindgen]
pub fn bloom_get_model_bounds_min_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[1] as f64
}
#[wasm_bindgen]
pub fn bloom_get_model_bounds_min_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[2] as f64
}
#[wasm_bindgen]
pub fn bloom_get_model_bounds_max_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[0] as f64
}
#[wasm_bindgen]
pub fn bloom_get_model_bounds_max_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[1] as f64
}
#[wasm_bindgen]
pub fn bloom_get_model_bounds_max_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[2] as f64
}

// ============================================================
// Input - Gamepad
// ============================================================

// EN-063 — arity MUST match the manifest (and the shared macro): these
// carried a leading `gamepad` param the manifest never declared, so the
// game's single argument landed in it and the real one arrived undefined
// -> NaN -> `as usize` = 0. Every axis read axis 0 and every button read
// button 0 — silently, on the platform with no debugger. This is verbatim
// the argument-shift class ffi_core/mod.rs says the shared macro exists to
// prevent; the web crate hand-mirrors it and drifted.
#[wasm_bindgen]
pub fn bloom_is_gamepad_available() -> f64 {
    if engine().input.is_gamepad_available() { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_gamepad_axis(axis: f64) -> f64 {
    engine().input.get_gamepad_axis(axis as usize) as f64
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_pressed(button: f64) -> f64 {
    if engine().input.is_gamepad_button_pressed(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_down(button: f64) -> f64 {
    if engine().input.is_gamepad_button_down(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_released(button: f64) -> f64 {
    if engine().input.is_gamepad_button_released(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_gamepad_axis_count() -> f64 {
    engine().input.get_gamepad_axis_count() as f64
}

// ============================================================
// Input - Touch
// ============================================================

// EN-063 — take the slot index, like the manifest and the shared macro.
// Hardcoding slot 0 meant every finger reported the FIRST finger's
// coordinates while `is_touch_active(i)` honoured the index: on a phone
// both thumbsticks tracked one thumb.
#[wasm_bindgen]
pub fn bloom_get_touch_x(index: f64) -> f64 {
    engine().input.get_touch_x(index as usize)
}

#[wasm_bindgen]
pub fn bloom_get_touch_y(index: f64) -> f64 {
    engine().input.get_touch_y(index as usize)
}

#[wasm_bindgen]
pub fn bloom_get_touch_count() -> f64 {
    engine().input.get_touch_count() as f64
}

#[wasm_bindgen]
pub fn bloom_is_touch_active(index: f64) -> f64 {
    if engine().input.is_touch_active(index as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_max_touch_points() -> f64 {
    engine().input.max_touch_points() as f64
}

// ============================================================
// Input injection (called from JS event listeners)
// ============================================================

// Injection goes through inject_key_* (staged, applied at the top of
// begin_frame), NOT set_key_* (which writes keys_down directly). A direct
// write made outside the poll phase is folded into `prev` before the edge
// is computed, so `isKeyPressed` never fires — the exact bug input.rs's
// pending_key_down staging was added to cure. Mirrors the shared macro.
#[wasm_bindgen]
pub fn bloom_inject_key_down(key: f64) {
    engine().input.inject_key_down(key as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_key_up(key: f64) {
    engine().input.inject_key_up(key as usize);
}

// EN-063: mouse / touch / wheel / char injection. DOM event handlers run in
// their own tasks, which on a single-threaded wasm host always land BETWEEN
// frames — i.e. before the next begin_frame(), exactly where a native
// message pump writes. So the immediate setters are correct here (edges are
// computed at begin_frame from prev-frame state); only same-gap taps need
// care, which the JS glue handles by deferring the matching -up one frame.

#[wasm_bindgen]
pub fn bloom_inject_mouse_move(x: f64, y: f64) {
    engine().input.set_mouse_position(x, y);
}

/// Relative movement (pointer-lock `movementX/Y`). Feeds the raw-delta
/// accumulator that `begin_frame` prefers while the cursor is disabled —
/// the camera-orbit path.
#[wasm_bindgen]
pub fn bloom_inject_mouse_delta(dx: f64, dy: f64) {
    engine().input.accumulate_mouse_delta(dx, dy);
}

#[wasm_bindgen]
pub fn bloom_inject_mouse_button_down(button: f64) {
    engine().input.set_mouse_button_down(button as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_mouse_button_up(button: f64) {
    engine().input.set_mouse_button_up(button as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_mouse_wheel(dy: f64) {
    engine().input.accumulate_mouse_wheel(dy);
}

#[wasm_bindgen]
pub fn bloom_inject_touch(index: f64, x: f64, y: f64, active: f64) {
    engine().input.set_touch(index as usize, x, y, active != 0.0);
}

/// Deferred release: keeps the slot active for the current frame and clears
/// it at end_frame, so a touch that begins and ends inside one frame is
/// still seen by the game.
#[wasm_bindgen]
pub fn bloom_inject_touch_release(index: f64, x: f64, y: f64) {
    engine().input.release_touch(index as usize, x, y);
}

#[wasm_bindgen]
pub fn bloom_inject_char(codepoint: f64) {
    engine().input.push_char(codepoint as u32);
}

#[wasm_bindgen]
pub fn bloom_inject_gamepad_axis(axis: f64, value: f64) {
    engine().input.set_gamepad_axis(axis as usize, value as f32);
}

#[wasm_bindgen]
pub fn bloom_inject_gamepad_button_down(button: f64) {
    engine().input.set_gamepad_button_down(button as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_gamepad_button_up(button: f64) {
    engine().input.set_gamepad_button_up(button as usize);
}


/// EN-031 â€” gamepad rumble. The Gamepad API exposes vibrationActuator, but
/// only behind a user-gesture requirement and with patchy support, so the web
/// port records the request (keeping the symbol and the state consistent with
/// native) without driving a motor. Wire `playEffect` here when the browser
/// story settles.
#[wasm_bindgen]
pub fn bloom_gamepad_rumble(low: f64, high: f64, seconds: f64) {
    let inp = &mut engine().input;
    inp.rumble = [
        (low as f32).clamp(0.0, 1.0),
        (high as f32).clamp(0.0, 1.0),
        (seconds as f32).clamp(0.0, 10.0),
    ];
}

