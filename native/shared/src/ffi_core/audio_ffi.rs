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

    };
}
