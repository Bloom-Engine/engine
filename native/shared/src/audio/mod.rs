//! Audio system: a control half (this type, called from the FFI/main
//! thread) and a render half ([`AudioRenderer`], owned by the platform's
//! audio thread), connected by a lock-free SPSC command ring.
//!
//! # Why the split
//!
//! The old `AudioMixer` was one struct mutated from two threads: the FFI
//! surface pushed/removed `PlayingSound`s on the main thread while the
//! platform audio callback iterated and mutated the same Vec — a data
//! race that corrupted voice state under load, and a use-after-free
//! whenever `load_sound` reallocated the registry mid-mix. Audio threads
//! also must never block: a mutex held across a frame hitch is an audible
//! glitch.
//!
//! # Threading contract
//!
//! - `AudioMixer` (this type) lives in `EngineState` — main thread only.
//! - The platform's audio-init code calls [`AudioMixer::take_renderer`]
//!   once and moves the renderer into its audio thread/callback, which
//!   then calls [`AudioRenderer::mix`] exclusively.
//! - Web (single-threaded: ScriptProcessorNode fires on the JS main
//!   thread) never takes the renderer; [`AudioMixer::mix_output`] mixes
//!   inline through the same command path.
//! - Sample data crosses as `Arc<SoundData>`; unloading is graceful —
//!   live voices finish on the old data.
//! - Render → control feedback (is-music-playing, position) flows through
//!   per-music atomics ([`MusicShared`]), never shared mutable state.

mod decode;
mod render;
mod spsc;

pub use decode::{decode_audio, parse_ogg, parse_wav};
#[cfg(feature = "mp3")]
pub use decode::parse_mp3;
pub use render::AudioRenderer;

use crate::handles::HandleRegistry;
use render::Cmd;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// A loaded sound effect or music track (PCM samples).
pub struct SoundData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Render-side playback state readable from the control side.
pub struct MusicShared {
    pub playing: AtomicBool,
    pub position: AtomicUsize,
}

struct MusicEntry {
    data: Arc<SoundData>,
    shared: Arc<MusicShared>,
    volume: f32,
    looping: bool,
}

/// Control half of the audio system. All methods are main-thread.
pub struct AudioMixer {
    pub sounds: HandleRegistry<Arc<SoundData>>,
    music: HandleRegistry<MusicEntry>,
    /// Default volume per sound handle, applied to future plays.
    sound_volumes: Vec<(f64, f32)>,
    master_volume: f32,
    tx: spsc::Producer<Cmd>,
    /// Present until the platform takes it for its audio thread; used for
    /// inline mixing on single-threaded targets (web).
    renderer: Option<AudioRenderer>,
}

/// Command-ring capacity. 256 in-flight commands comfortably exceeds any
/// realistic burst (a command is one play/stop/volume change; the ring
/// drains every audio callback, i.e. every ~10ms).
const CMD_CAPACITY: usize = 256;

impl Default for AudioMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioMixer {
    pub fn new() -> Self {
        let (tx, rx) = spsc::channel(CMD_CAPACITY);
        Self {
            sounds: HandleRegistry::new(),
            music: HandleRegistry::new(),
            sound_volumes: Vec::new(),
            master_volume: 1.0,
            tx,
            renderer: Some(AudioRenderer::new(rx)),
        }
    }

    /// Hand the render half to the platform's audio thread. Call once
    /// from audio init, before the callback starts firing; the returned
    /// renderer must only be used from that one thread.
    pub fn take_renderer(&mut self) -> Option<AudioRenderer> {
        self.renderer.take()
    }

    fn send(&mut self, cmd: Cmd) {
        // Whether or not the renderer has been handed off, commands go
        // through the ring — the inline path (web) drains it on the next
        // mix_output, the threaded path on the next audio callback.
        if self.tx.push(cmd).is_err() {
            // Ring full: drop the command rather than block. 256 pending
            // commands between two audio callbacks means something is
            // very wrong upstream — say so once.
            crate::ffi::log_error(
                "bloom: audio command ring full — command dropped (is the audio callback running?)",
            );
        }
    }

    // ----------------------------------------------------------- sounds

    pub fn load_sound(&mut self, data: SoundData) -> f64 {
        self.sounds.alloc(Arc::new(data))
    }

    pub fn play_sound(&mut self, handle: f64) {
        let Some(data) = self.sounds.get(handle).cloned() else { return };
        let volume = self.get_sound_volume(handle);
        self.send(Cmd::PlaySound { sound_id: handle.to_bits(), data, volume, spatial: None });
    }

    pub fn play_sound_3d(&mut self, handle: f64, x: f32, y: f32, z: f32) {
        let Some(data) = self.sounds.get(handle).cloned() else { return };
        let volume = self.get_sound_volume(handle);
        self.send(Cmd::PlaySound {
            sound_id: handle.to_bits(),
            data,
            volume,
            spatial: Some([x, y, z]),
        });
    }

    pub fn stop_sound(&mut self, handle: f64) {
        self.send(Cmd::StopSound { sound_id: handle.to_bits() });
    }

    pub fn set_sound_volume(&mut self, handle: f64, volume: f32) {
        for entry in &mut self.sound_volumes {
            if entry.0 == handle {
                entry.1 = volume;
                self.send(Cmd::SetSoundVolume { sound_id: handle.to_bits(), volume });
                return;
            }
        }
        self.sound_volumes.push((handle, volume));
        self.send(Cmd::SetSoundVolume { sound_id: handle.to_bits(), volume });
    }

    fn get_sound_volume(&self, handle: f64) -> f32 {
        self.sound_volumes
            .iter()
            .find(|e| e.0 == handle)
            .map(|e| e.1)
            .unwrap_or(1.0)
    }

    pub fn unload_sound(&mut self, handle: f64) {
        // Voices already playing hold their own Arc and finish gracefully.
        self.sounds.free(handle);
        self.sound_volumes.retain(|e| e.0 != handle);
    }

    // ------------------------------------------------------------ music

    pub fn load_music(&mut self, data: SoundData) -> f64 {
        self.music.alloc(MusicEntry {
            data: Arc::new(data),
            shared: Arc::new(MusicShared {
                playing: AtomicBool::new(false),
                position: AtomicUsize::new(0),
            }),
            volume: 1.0,
            looping: true,
        })
    }

    pub fn play_music(&mut self, handle: f64) {
        let Some(m) = self.music.get(handle) else { return };
        // Optimistically flip the flag so is_music_playing is true the
        // moment play_music returns (the render thread confirms on its
        // next callback).
        m.shared.playing.store(true, Ordering::Relaxed);
        let cmd = Cmd::PlayMusic {
            music_id: handle.to_bits(),
            data: m.data.clone(),
            shared: m.shared.clone(),
            volume: m.volume,
            looping: m.looping,
        };
        self.send(cmd);
    }

    pub fn stop_music(&mut self, handle: f64) {
        if let Some(m) = self.music.get(handle) {
            m.shared.playing.store(false, Ordering::Relaxed);
        }
        self.send(Cmd::StopMusic { music_id: handle.to_bits() });
    }

    pub fn set_music_volume(&mut self, handle: f64, volume: f32) {
        if let Some(m) = self.music.get_mut(handle) {
            m.volume = volume;
        }
        self.send(Cmd::SetMusicVolume { music_id: handle.to_bits(), volume });
    }

    pub fn is_music_playing(&self, handle: f64) -> bool {
        self.music
            .get(handle)
            .map(|m| m.shared.playing.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    pub fn update_music_stream(&mut self, _handle: f64) {
        // No-op: tracks are fully decoded today. Exists for raylib API
        // compatibility; becomes meaningful with streamed decode.
    }

    // ----------------------------------------------------------- global

    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = volume;
        self.send(Cmd::SetMaster(volume));
    }

    pub fn master_volume(&self) -> f32 {
        self.master_volume
    }

    pub fn set_listener_position(&mut self, x: f32, y: f32, z: f32, fx: f32, fy: f32, fz: f32) {
        let len = (fx * fx + fy * fy + fz * fz).sqrt();
        let forward = if len > 0.0 {
            [fx / len, fy / len, fz / len]
        } else {
            [0.0, 0.0, -1.0]
        };
        self.send(Cmd::SetListener { pos: [x, y, z], forward });
    }

    /// Inline mixing for single-threaded targets (web) and tests: mixes
    /// through the internal renderer if it has not been taken by a
    /// platform audio thread.
    pub fn mix_output(&mut self, output: &mut [f32]) {
        match self.renderer.as_mut() {
            Some(r) => r.mix(output),
            None => {
                // A platform audio thread owns the renderer; mixing here
                // would be the exact data race this design removes.
                output.iter_mut().for_each(|s| *s = 0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(len: usize) -> SoundData {
        SoundData {
            samples: (0..len).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect(),
            sample_rate: 44_100,
            channels: 1,
        }
    }

    #[test]
    fn inline_play_and_mix() {
        let mut a = AudioMixer::new();
        let h = a.load_sound(tone(64));
        a.play_sound(h);
        let mut out = [0.0f32; 32];
        a.mix_output(&mut out);
        assert!(out.iter().any(|&s| s != 0.0), "voice produced no output");
    }

    #[test]
    fn music_playing_flag_round_trip() {
        let mut a = AudioMixer::new();
        let h = a.load_music(tone(16)); // tiny, non-looping after we set it
        // default looping=true → flip via the public surface used by FFI
        a.music.get_mut(h).unwrap().looping = false;
        assert!(!a.is_music_playing(h));
        a.play_music(h);
        assert!(a.is_music_playing(h), "flag not set synchronously");
        // Drain the whole track: 16 mono samples → 32 stereo out slots
        let mut out = [0.0f32; 64];
        a.mix_output(&mut out);
        assert!(!a.is_music_playing(h), "non-looping track did not end");
    }

    #[test]
    fn renderer_handoff_mixes_on_other_thread() {
        let mut a = AudioMixer::new();
        let h = a.load_sound(tone(1024));
        let mut r = a.take_renderer().expect("renderer already taken");
        a.play_sound(h);
        a.set_master_volume(0.5);
        let worker = std::thread::spawn(move || {
            let mut out = vec![0.0f32; 256];
            r.mix(&mut out);
            out
        });
        let out = worker.join().unwrap();
        assert!(out.iter().any(|&s| s != 0.0), "handed-off renderer produced no output");
        // Control side mixing is now inert (no double-mixing race).
        let mut silent = [1.0f32; 8];
        a.mix_output(&mut silent);
        assert!(silent.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn unload_mid_playback_is_graceful() {
        let mut a = AudioMixer::new();
        let h = a.load_sound(tone(64));
        a.play_sound(h);
        a.unload_sound(h);
        let mut out = [0.0f32; 32];
        a.mix_output(&mut out); // voice still holds its Arc
        assert!(out.iter().any(|&s| s != 0.0));
    }
}
