//! Keyboard, mouse, touch, and gamepad state (single-pad InputState model).
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_input {
    () => {

        // bloom_is_key_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_key_pressed(key: f64) -> f64 {
            $crate::ffi::guard("bloom_is_key_pressed", move || {
                if engine().input.is_key_pressed(key as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_key_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_key_down(key: f64) -> f64 {
            $crate::ffi::guard("bloom_is_key_down", move || {
                if engine().input.is_key_down(key as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_key_released  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_key_released(key: f64) -> f64 {
            $crate::ffi::guard("bloom_is_key_released", move || {
                if engine().input.is_key_released(key as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_mouse_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_x() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_x", move || {
                engine().input.mouse_x
        })
        }

        // bloom_get_mouse_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_y() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_y", move || {
                engine().input.mouse_y
        })
        }

        // bloom_is_mouse_button_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_mouse_button_pressed(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_mouse_button_pressed", move || {
                if engine().input.is_mouse_button_pressed(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_mouse_button_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_mouse_button_down(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_mouse_button_down", move || {
                if engine().input.is_mouse_button_down(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_mouse_button_released  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_mouse_button_released(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_mouse_button_released", move || {
                if engine().input.is_mouse_button_released(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_crown_rotation  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_crown_rotation() -> f64 {
            $crate::ffi::guard("bloom_get_crown_rotation", move || {
                engine().input.consume_crown_rotation()
        })
        }

        // bloom_is_gamepad_available  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_available() -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_available", move || {
                if engine().input.is_gamepad_available() { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_gamepad_axis  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_gamepad_axis(axis: f64) -> f64 {
            $crate::ffi::guard("bloom_get_gamepad_axis", move || {
                engine().input.get_gamepad_axis(axis as usize) as f64
        })
        }

        // bloom_gamepad_rumble  [EN-031]
        // low/high motor 0..1, duration in seconds. The platform's input poll
        // consumes this and drives its own vibration API; on platforms with no
        // vibration it is simply ignored (the state is written, nobody reads
        // it) — which is why the symbol exists everywhere and the behaviour
        // does not have to.
        #[no_mangle]
        pub extern "C" fn bloom_gamepad_rumble(low: f64, high: f64, seconds: f64) {
            $crate::ffi::guard("bloom_gamepad_rumble", move || {
                let inp = &mut engine().input;
                inp.rumble = [
                    (low as f32).clamp(0.0, 1.0),
                    (high as f32).clamp(0.0, 1.0),
                    (seconds as f32).clamp(0.0, 10.0),
                ];
        })
        }

        // bloom_is_gamepad_button_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_button_pressed(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_button_pressed", move || {
                if engine().input.is_gamepad_button_pressed(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_gamepad_button_down  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_button_down(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_button_down", move || {
                if engine().input.is_gamepad_button_down(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_is_gamepad_button_released  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_gamepad_button_released(btn: f64) -> f64 {
            $crate::ffi::guard("bloom_is_gamepad_button_released", move || {
                if engine().input.is_gamepad_button_released(btn as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_gamepad_axis_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_gamepad_axis_count() -> f64 {
            $crate::ffi::guard("bloom_get_gamepad_axis_count", move || {
                engine().input.get_gamepad_axis_count() as f64
        })
        }

        // bloom_get_mouse_delta_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_delta_x() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_delta_x", move || {
                engine().input.mouse_delta_x
        })
        }

        // bloom_get_mouse_delta_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_delta_y() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_delta_y", move || {
                engine().input.mouse_delta_y
        })
        }

        // bloom_get_mouse_wheel  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_mouse_wheel() -> f64 {
            $crate::ffi::guard("bloom_get_mouse_wheel", move || {
                engine().input.consume_mouse_wheel()
        })
        }

        // bloom_get_char_pressed  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_char_pressed() -> f64 {
            $crate::ffi::guard("bloom_get_char_pressed", move || {
                engine().input.pop_char() as f64
        })
        }

        // bloom_get_touch_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_touch_x(index: f64) -> f64 {
            $crate::ffi::guard("bloom_get_touch_x", move || {
                engine().input.get_touch_x(index as usize)
        })
        }

        // bloom_get_touch_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_touch_y(index: f64) -> f64 {
            $crate::ffi::guard("bloom_get_touch_y", move || {
                engine().input.get_touch_y(index as usize)
        })
        }

        // bloom_get_touch_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_touch_count() -> f64 {
            $crate::ffi::guard("bloom_get_touch_count", move || {
                engine().input.get_touch_count() as f64
        })
        }

        // bloom_is_touch_active  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_touch_active(index: f64) -> f64 {
            $crate::ffi::guard("bloom_is_touch_active", move || {
                if engine().input.is_touch_active(index as usize) { 1.0 } else { 0.0 }
        })
        }

        // bloom_get_max_touch_points  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_get_max_touch_points() -> f64 {
            $crate::ffi::guard("bloom_get_max_touch_points", move || {
                engine().input.max_touch_points() as f64
        })
        }

    };
}
