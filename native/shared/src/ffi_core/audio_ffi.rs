//! Sound and music playback control.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_audio_ffi {
    () => {

        // bloom_play_sound  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_play_sound(handle: f64) {
            $crate::ffi::guard("bloom_play_sound", move || {
                engine().audio.play_sound(handle);
        })
        }

        // bloom_stop_sound  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_stop_sound(handle: f64) {
            $crate::ffi::guard("bloom_stop_sound", move || {
                engine().audio.stop_sound(handle);
        })
        }

        // bloom_play_sound_3d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
            $crate::ffi::guard("bloom_play_sound_3d", move || {
                engine().audio.play_sound_3d(handle, x as f32, y as f32, z as f32);
        })
        }

        // bloom_play_music  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_play_music(handle: f64) {
            $crate::ffi::guard("bloom_play_music", move || {
                engine().audio.play_music(handle);
        })
        }

        // bloom_stop_music  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_stop_music(handle: f64) {
            $crate::ffi::guard("bloom_stop_music", move || {
                engine().audio.stop_music(handle);
        })
        }

        // bloom_is_music_playing  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_is_music_playing(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_is_music_playing", move || {
                if engine().audio.is_music_playing(handle) { 1.0 } else { 0.0 }
        })
        }

        // ---- EN-029: buses, reverb send, occlusion low-pass -------------
        //
        // The mixer was master + per-voice gain and nothing else. These three
        // additions are what separate "sounds are playing" from "the space
        // sounds like a place": a bus you can duck, a tail you can send to, and
        // a filter that makes a wall audible.

        // bloom_set_sound_bus — 0 = SFX, 1 = music, 2 = UI.
        #[no_mangle]
        pub extern "C" fn bloom_set_sound_bus(handle: f64, bus: f64) {
            $crate::ffi::guard("bloom_set_sound_bus", move || {
                engine().audio.set_sound_bus(handle, bus as u8);
        })
        }

        // bloom_set_sound_reverb_send — 0..1.
        #[no_mangle]
        pub extern "C" fn bloom_set_sound_reverb_send(handle: f64, send: f64) {
            $crate::ffi::guard("bloom_set_sound_reverb_send", move || {
                engine().audio.set_sound_reverb_send(handle, send as f32);
        })
        }

        // bloom_set_sound_lowpass — cutoff Hz; 0 = bypass.
        #[no_mangle]
        pub extern "C" fn bloom_set_sound_lowpass(handle: f64, cutoff: f64) {
            $crate::ffi::guard("bloom_set_sound_lowpass", move || {
                engine().audio.set_sound_lowpass(handle, cutoff as f32);
        })
        }

        // bloom_set_bus_gain
        #[no_mangle]
        pub extern "C" fn bloom_set_bus_gain(bus: f64, gain: f64) {
            $crate::ffi::guard("bloom_set_bus_gain", move || {
                engine().audio.set_bus_gain(bus as u8, gain as f32);
        })
        }

        // bloom_duck_bus — momentary attenuation with attack/hold/release.
        #[no_mangle]
        pub extern "C" fn bloom_duck_bus(bus: f64, amount: f64, attack: f64, release: f64, hold: f64) {
            $crate::ffi::guard("bloom_duck_bus", move || {
                engine().audio.duck_bus(
                    bus as u8, amount as f32, attack as f32, release as f32, hold as f32);
        })
        }

        // bloom_set_reverb — size / damp / wet, all 0..1. wet = 0 bypasses the
        // whole reverb path, so it costs nothing until a game asks for it.
        #[no_mangle]
        pub extern "C" fn bloom_set_reverb(size: f64, damp: f64, wet: f64) {
            $crate::ffi::guard("bloom_set_reverb", move || {
                engine().audio.set_reverb(size as f32, damp as f32, wet as f32);
        })
        }

    };
}
