//! Rust-side audio FFI that forwards to the Swift `BloomAudioManager`.
//!
//! The Swift shell owns AVAudioPlayer + AVAudioSession; this module is a
//! thin pass-through that decodes Perry string args + resolves bundle-relative
//! paths the same way textures does.

use std::ffi::CString;
use std::os::raw::c_char;

use crate::textures;

extern "C" {
    fn bloom_watchos_audio_init();
    fn bloom_watchos_audio_close();
    fn bloom_watchos_sound_load(path: *const c_char) -> u32;
    fn bloom_watchos_sound_play(handle: u32);
    fn bloom_watchos_sound_stop(handle: u32);
    fn bloom_watchos_sound_volume(handle: u32, volume: f32);
    fn bloom_watchos_master_volume(volume: f32);
    fn bloom_watchos_music_load(path: *const c_char) -> u32;
    fn bloom_watchos_music_play(handle: u32);
    fn bloom_watchos_music_stop(handle: u32);
    fn bloom_watchos_music_volume(handle: u32, volume: f32);
    fn bloom_watchos_music_is_playing(handle: u32) -> u32;
}

fn load_with<F>(path: &str, load_fn: F) -> u32
where F: FnOnce(*const c_char) -> u32
{
    if path.is_empty() { return 0; }
    let full = if path.starts_with('/') { path.to_string() } else { textures::resolve_bundle_path(path) };
    let Ok(c) = CString::new(full) else { return 0 };
    unsafe { load_fn(c.as_ptr()) }
}

pub fn init_audio() { unsafe { bloom_watchos_audio_init(); } }
pub fn close_audio() { unsafe { bloom_watchos_audio_close(); } }

pub fn load_sound(path: &str) -> u32 {
    load_with(path, |p| unsafe { bloom_watchos_sound_load(p) })
}
pub fn play_sound(handle: u32) { unsafe { bloom_watchos_sound_play(handle); } }
pub fn stop_sound(handle: u32) { unsafe { bloom_watchos_sound_stop(handle); } }
pub fn set_sound_volume(handle: u32, v: f32) { unsafe { bloom_watchos_sound_volume(handle, v); } }
pub fn set_master_volume(v: f32) { unsafe { bloom_watchos_master_volume(v); } }

pub fn load_music(path: &str) -> u32 {
    load_with(path, |p| unsafe { bloom_watchos_music_load(p) })
}
pub fn play_music(handle: u32) { unsafe { bloom_watchos_music_play(handle); } }
pub fn stop_music(handle: u32) { unsafe { bloom_watchos_music_stop(handle); } }
pub fn set_music_volume(handle: u32, v: f32) { unsafe { bloom_watchos_music_volume(handle, v); } }
pub fn is_music_playing(handle: u32) -> bool {
    unsafe { bloom_watchos_music_is_playing(handle) != 0 }
}
