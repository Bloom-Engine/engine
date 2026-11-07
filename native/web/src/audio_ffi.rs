//! Audio FFI surface for web: sounds, voices, music and the mixer pull.
//! Split from lib.rs (2000-line file policy).

#[allow(unused_imports)]
use crate::engine;
use wasm_bindgen::prelude::*;

// ============================================================
// Audio
// ============================================================

#[wasm_bindgen]
pub fn bloom_init_audio() {
    // Audio initialization is handled by JS glue (Web Audio API AudioContext).
    // The Rust AudioMixer is already initialized as part of EngineState.
}

#[wasm_bindgen]
pub fn bloom_close_audio() {
    // AudioContext cleanup handled by JS glue.
}

#[wasm_bindgen]
pub fn bloom_load_sound(_path: f64) -> f64 {
    0.0
}

/// Load a sound from raw file bytes (WAV or OGG). Fetched by JS glue via fetch().
#[wasm_bindgen]
pub fn bloom_load_sound_bytes(data: &[u8]) -> f64 {
    if let Some(sound) =
        bloom_shared::audio::parse_wav(data).or_else(|| bloom_shared::audio::parse_ogg(data))
    {
        engine().audio.load_sound(sound)
    } else {
        0.0
    }
}

#[wasm_bindgen]
pub fn bloom_play_sound(handle: f64) {
    engine().audio.play_sound(handle);
}

#[wasm_bindgen]
pub fn bloom_stop_sound(handle: f64) {
    engine().audio.stop_sound(handle);
}

#[wasm_bindgen]
pub fn bloom_set_sound_volume(handle: f64, volume: f64) {
    engine().audio.set_sound_volume(handle, volume as f32);
}

#[wasm_bindgen]
pub fn bloom_set_master_volume(volume: f64) {
    engine().audio.set_master_volume(volume as f32);
}

#[wasm_bindgen]
pub fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
    engine()
        .audio
        .play_sound_3d(handle, x as f32, y as f32, z as f32);
}

#[wasm_bindgen]
pub fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
    engine().audio.set_listener_position(
        x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32,
    );
}

// ---- EN-062: live spatial voices (see shared audio_ffi.rs for docs) --------

#[wasm_bindgen]
pub fn bloom_play_sound_3d_ex(
    handle: f64,
    x: f64,
    y: f64,
    z: f64,
    looping: f64,
    ref_dist: f64,
    max_dist: f64,
    rolloff: f64,
) -> f64 {
    engine().audio.play_sound_3d_ex(
        handle,
        x as f32,
        y as f32,
        z as f32,
        looping != 0.0,
        ref_dist as f32,
        max_dist as f32,
        rolloff as f32,
    )
}

#[wasm_bindgen]
pub fn bloom_voice_set_position(voice: f64, x: f64, y: f64, z: f64) {
    engine()
        .audio
        .set_voice_position(voice, x as f32, y as f32, z as f32);
}

#[wasm_bindgen]
pub fn bloom_voice_stop(voice: f64) {
    engine().audio.stop_voice(voice);
}

#[wasm_bindgen]
pub fn bloom_voice_set_volume(voice: f64, volume: f64) {
    engine().audio.set_voice_volume(voice, volume as f32);
}

#[wasm_bindgen]
pub fn bloom_voice_set_pitch(voice: f64, pitch: f64) {
    engine().audio.set_voice_pitch(voice, pitch as f32);
}

#[wasm_bindgen]
pub fn bloom_voice_set_lowpass(voice: f64, cutoff: f64) {
    engine().audio.set_voice_lowpass(voice, cutoff as f32);
}

// ---- Web Audio bridge (see shared audio_ffi.rs for docs) -------------------

#[wasm_bindgen]
pub fn bloom_play_sound_ex(handle: f64, looping: f64, volume: f64, pitch: f64, pan: f64) -> f64 {
    engine().audio.play_sound_ex(
        handle,
        looping != 0.0,
        volume as f32,
        pitch as f32,
        pan as f32,
    )
}

#[wasm_bindgen]
pub fn bloom_voice_set_pan(voice: f64, pan: f64) {
    engine().audio.set_voice_pan(voice, pan as f32);
}

#[wasm_bindgen]
pub fn bloom_set_music_looping(handle: f64, looping: f64) {
    engine().audio.set_music_looping(handle, looping != 0.0);
}

#[wasm_bindgen]
pub fn bloom_get_music_time(handle: f64) -> f64 {
    engine().audio.music_time(handle)
}

#[wasm_bindgen]
pub fn bloom_play_music_from(handle: f64, seconds: f64) {
    engine().audio.play_music_from(handle, seconds);
}

#[wasm_bindgen]
pub fn bloom_get_music_length(handle: f64) -> f64 {
    engine().audio.music_length(handle)
}

#[wasm_bindgen]
pub fn bloom_load_music(_path: f64) -> f64 {
    0.0
}

/// Load music from raw file bytes (WAV or OGG). Fetched by JS glue via fetch().
#[wasm_bindgen]
pub fn bloom_load_music_bytes(data: &[u8]) -> f64 {
    if let Some(sound) =
        bloom_shared::audio::parse_wav(data).or_else(|| bloom_shared::audio::parse_ogg(data))
    {
        engine().audio.load_music(sound)
    } else {
        0.0
    }
}

/// Mix audio into an interleaved stereo f32 buffer.
/// Called by JS Web Audio ScriptProcessorNode/AudioWorklet each audio frame.
#[wasm_bindgen]
pub fn bloom_audio_mix(output: &mut [f32]) {
    engine().audio.mix_output(output);
}

#[wasm_bindgen]
pub fn bloom_play_music(handle: f64) {
    engine().audio.play_music(handle);
}

#[wasm_bindgen]
pub fn bloom_stop_music(handle: f64) {
    engine().audio.stop_music(handle);
}

#[wasm_bindgen]
pub fn bloom_update_music_stream(handle: f64) {
    engine().audio.update_music_stream(handle);
}

#[wasm_bindgen]
pub fn bloom_set_music_volume(handle: f64, volume: f64) {
    engine().audio.set_music_volume(handle, volume as f32);
}

#[wasm_bindgen]
pub fn bloom_is_music_playing(handle: f64) -> f64 {
    if engine().audio.is_music_playing(handle) {
        1.0
    } else {
        0.0
    }
}
