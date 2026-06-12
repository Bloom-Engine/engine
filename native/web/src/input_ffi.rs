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

#[wasm_bindgen]
pub fn bloom_is_gamepad_available(gamepad: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_available() { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_gamepad_axis(gamepad: f64, axis: f64) -> f64 {
    let _ = gamepad;
    engine().input.get_gamepad_axis(axis as usize) as f64
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_pressed(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_pressed(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_down(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_down(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_released(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_released(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_gamepad_axis_count(gamepad: f64) -> f64 {
    let _ = gamepad;
    engine().input.get_gamepad_axis_count() as f64
}

// ============================================================
// Input - Touch
// ============================================================

#[wasm_bindgen]
pub fn bloom_get_touch_x() -> f64 {
    engine().input.get_touch_x(0)
}

#[wasm_bindgen]
pub fn bloom_get_touch_y() -> f64 {
    engine().input.get_touch_y(0)
}

#[wasm_bindgen]
pub fn bloom_get_touch_count() -> f64 {
    engine().input.get_touch_count() as f64
}

// ============================================================
// Input injection (called from JS event listeners)
// ============================================================

#[wasm_bindgen]
pub fn bloom_inject_key_down(key: f64) {
    engine().input.set_key_down(key as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_key_up(key: f64) {
    engine().input.set_key_up(key as usize);
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

