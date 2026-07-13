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
pub mod stream;

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

enum MusicSource {
    /// Fully decoded PCM (WAV always; everything on wasm32).
    Full(Arc<SoundData>),
    /// Compressed bytes decoded by a background worker at play time.
    #[cfg(not(target_arch = "wasm32"))]
    Streamed {
        kind: stream::StreamKind,
        bytes: Arc<Vec<u8>>,
        channels: u16,
    },
}

struct MusicEntry {
    source: MusicSource,
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
    /// EN-029 — per-sound routing: (bus, reverb send, low-pass cutoff Hz).
    /// A property of the sound, not of each play call.
    routes: std::collections::HashMap<u64, (u8, f32, f32)>,
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
            routes: std::collections::HashMap::new(),
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
        let (bus, send, lowpass) = self.routing(handle);
        self.send(Cmd::PlaySound {
            sound_id: handle.to_bits(), data, volume, spatial: None,
            bus, send, lowpass,
        });
    }

    pub fn play_sound_3d(&mut self, handle: f64, x: f32, y: f32, z: f32) {
        let Some(data) = self.sounds.get(handle).cloned() else { return };
        let volume = self.get_sound_volume(handle);
        let (bus, send, lowpass) = self.routing(handle);
        self.send(Cmd::PlaySound {
            sound_id: handle.to_bits(),
            data,
            volume,
            spatial: Some([x, y, z]),
            bus, send, lowpass,
        });
    }

    // ---- EN-029 routing ------------------------------------------------
    //
    // Routing is a property of the *sound*, not of the individual play call:
    // a footstep is always on the SFX bus, a menu blip is always UI. Setting
    // it once at load keeps the per-shot call sites unchanged.

    fn routing(&self, handle: f64) -> (u8, f32, f32) {
        match self.routes.get(&handle.to_bits()) {
            Some(r) => *r,
            None => (render::bus::SFX, 0.0, 0.0),
        }
    }

    /// Assign a sound to a mix bus (see `render::bus`).
    pub fn set_sound_bus(&mut self, handle: f64, bus: u8) {
        let e = self.routes.entry(handle.to_bits()).or_insert((render::bus::SFX, 0.0, 0.0));
        e.0 = bus;
    }

    /// Reverb send for this sound, 0..1. This is what gives a gunshot its tail.
    pub fn set_sound_reverb_send(&mut self, handle: f64, send: f32) {
        let send = send.clamp(0.0, 1.0);
        let e = self.routes.entry(handle.to_bits()).or_insert((render::bus::SFX, 0.0, 0.0));
        e.1 = send;
        // Also steer voices already in flight, so a zone change is audible on
        // the tail that is sounding right now rather than only the next one.
        self.send(Cmd::SetSoundSend { sound_id: handle.to_bits(), send });
    }

    /// Low-pass cutoff in Hz for this sound; 0 = bypass. The occlusion knob.
    pub fn set_sound_lowpass(&mut self, handle: f64, cutoff: f32) {
        let cutoff = cutoff.max(0.0);
        let e = self.routes.entry(handle.to_bits()).or_insert((render::bus::SFX, 0.0, 0.0));
        e.2 = cutoff;
        self.send(Cmd::SetSoundLowpass { sound_id: handle.to_bits(), cutoff });
    }

    pub fn set_bus_gain(&mut self, bus: u8, gain: f32) {
        self.send(Cmd::SetBusGain { bus, gain });
    }

    pub fn duck_bus(&mut self, bus: u8, amount: f32, attack: f32, release: f32, hold: f32) {
        self.send(Cmd::DuckBus { bus, amount, attack, release, hold });
    }

    pub fn set_reverb(&mut self, size: f32, damp: f32, wet: f32) {
        self.send(Cmd::SetReverbParams { size, damp, wet });
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
        self.alloc_music(MusicSource::Full(Arc::new(data)))
    }

    /// Load music from raw file bytes, streaming the decode when the
    /// format supports it (OGG/MP3 on native — keeps ~5 MB of compressed
    /// bytes resident instead of ~57 MB of PCM for a 5-minute track).
    /// WAV — and everything on wasm32, which has no threads — falls back
    /// to full decode. Returns 0 on undecodable data.
    pub fn load_music_bytes(&mut self, path: &str, data: Vec<u8>) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let lower = path.to_ascii_lowercase();
            let kind = if lower.ends_with(".ogg") {
                Some(stream::StreamKind::Ogg)
            } else {
                #[cfg(feature = "mp3")]
                if lower.ends_with(".mp3") {
                    Some(stream::StreamKind::Mp3)
                } else {
                    None
                }
                #[cfg(not(feature = "mp3"))]
                None
            };
            if let Some(kind) = kind {
                if let Some((_rate, channels)) = stream::probe(kind, &data) {
                    return self.alloc_music(MusicSource::Streamed {
                        kind,
                        bytes: Arc::new(data),
                        channels,
                    });
                }
                // Mislabelled file — fall through to sniffing decode.
            }
        }
        match decode_audio(path, &data) {
            Some(s) => self.load_music(s),
            None => 0.0,
        }
    }

    fn alloc_music(&mut self, source: MusicSource) -> f64 {
        self.music.alloc(MusicEntry {
            source,
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
        let payload = match &m.source {
            MusicSource::Full(data) => render::MusicPayload::Full(data.clone()),
            #[cfg(not(target_arch = "wasm32"))]
            MusicSource::Streamed { kind, bytes, channels } => render::MusicPayload::Stream {
                consumer: stream::start(*kind, bytes.clone(), m.looping),
                channels: *channels,
            },
        };
        let cmd = Cmd::PlayMusic {
            music_id: handle.to_bits(),
            payload,
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

    // ---- EN-029 -------------------------------------------------------

    fn peak(buf: &[f32]) -> f32 {
        buf.iter().fold(0.0f32, |m, s| m.max(s.abs()))
    }

    #[test]
    fn bus_gain_scales_only_its_own_bus() {
        let mut a = AudioMixer::new();
        let h = a.load_sound(tone(4096));
        a.play_sound(h);
        let mut loud = [0.0f32; 256];
        a.mix_output(&mut loud);

        let mut b = AudioMixer::new();
        let h2 = b.load_sound(tone(4096));
        b.set_bus_gain(render::bus::SFX, 0.25);
        b.play_sound(h2);
        let mut quiet = [0.0f32; 256];
        b.mix_output(&mut quiet);

        assert!(peak(&quiet) < peak(&loud) * 0.5,
            "SFX bus gain did not attenuate: {} vs {}", peak(&quiet), peak(&loud));

        // A sound on a *different* bus must be untouched by that gain.
        let mut c = AudioMixer::new();
        let h3 = c.load_sound(tone(4096));
        c.set_sound_bus(h3, render::bus::UI);
        c.set_bus_gain(render::bus::SFX, 0.0);
        c.play_sound(h3);
        let mut ui = [0.0f32; 256];
        c.mix_output(&mut ui);
        assert!(peak(&ui) > 0.1, "muting SFX also muted the UI bus");
    }

    #[test]
    fn duck_pulls_the_bus_down_then_recovers() {
        let mut a = AudioMixer::new();
        let h = a.load_sound(tone(1 << 16));
        a.play_sound(h);
        let mut out = [0.0f32; 512];
        a.mix_output(&mut out);
        let dry = peak(&out);

        // Duck hard, effectively instantly, and hold well past the block.
        a.duck_bus(render::bus::SFX, 0.9, 0.0001, 0.5, 1.0);
        let mut ducked = [0.0f32; 512];
        a.mix_output(&mut ducked);
        assert!(peak(&ducked) < dry * 0.5,
            "duck had no effect: {} vs {}", peak(&ducked), dry);
    }

    #[test]
    fn lowpass_attenuates_a_nyquist_tone() {
        // tone() alternates +0.5/-0.5 every sample: that is exactly Nyquist,
        // the highest frequency representable. A low cutoff must crush it.
        let mut a = AudioMixer::new();
        let h = a.load_sound(tone(4096));
        a.set_sound_lowpass(h, 200.0);
        a.play_sound(h);
        let mut out = [0.0f32; 512];
        a.mix_output(&mut out);
        assert!(peak(&out) < 0.1,
            "low-pass did not attenuate a Nyquist tone: peak {}", peak(&out));
    }

    /// Mix `blocks` × 256-sample blocks, returning the peak seen *after* the
    /// first block. The shortest comb delay is 1116 samples (~25 ms), so a
    /// reverb tail cannot appear inside one short block — you have to run the
    /// mixer past the delay length before asking whether it rang.
    fn peak_after_first_block(a: &mut AudioMixer, blocks: usize) -> f32 {
        let mut first = [0.0f32; 256];
        a.mix_output(&mut first);
        let mut p = 0.0f32;
        for _ in 1..blocks {
            let mut out = [0.0f32; 256];
            a.mix_output(&mut out);
            p = p.max(peak(&out));
        }
        p
    }

    #[test]
    fn reverb_rings_after_the_source_stops() {
        let mut a = AudioMixer::new();
        // A short click, fully sent to a long, bright reverb.
        let h = a.load_sound(tone(8));
        a.set_reverb(0.9, 0.1, 1.0);
        a.set_sound_reverb_send(h, 1.0);
        a.play_sound(h);
        // 40 blocks × 128 frames = 5120 frames, comfortably past the 1356-sample
        // longest comb.
        let tail = peak_after_first_block(&mut a, 40);
        assert!(tail > 0.0, "reverb produced no tail after the source ended");
    }

    #[test]
    fn reverb_is_bypassed_when_wet_is_zero() {
        let mut a = AudioMixer::new();
        let h = a.load_sound(tone(8));
        a.set_sound_reverb_send(h, 1.0); // sending, but nothing returns
        a.play_sound(h);
        let tail = peak_after_first_block(&mut a, 40);
        assert_eq!(tail, 0.0, "wet=0 must cost nothing and return nothing");
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
