//! The render half of the audio system: voice state + mixing.
//!
//! An [`AudioRenderer`] is owned exclusively by the platform's audio
//! thread (CoreAudio render callback on Apple platforms, the dedicated
//! ALSA/WASAPI/AAudio thread elsewhere, or inline on the main thread on
//! web). It never locks: control-side changes arrive as [`Cmd`]s over the
//! SPSC ring and are drained at the top of every [`AudioRenderer::mix`].
//!
//! Sample data is shared with the control side via `Arc<SoundData>` — a
//! sound unloaded mid-playback keeps its samples alive until the last
//! voice playing it finishes, then the Arc drops on this thread.

use super::spsc::Consumer;
use super::{MusicShared, SoundData};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Control → render commands. Everything the FFI surface can do to audio
/// state is expressed here; the render thread is the single writer of all
/// playback state.
pub enum Cmd {
    PlaySound {
        sound_id: u64,
        data: Arc<SoundData>,
        volume: f32,
        /// Some(world position) for spatial sounds.
        spatial: Option<[f32; 3]>,
    },
    StopSound { sound_id: u64 },
    SetSoundVolume { sound_id: u64, volume: f32 },
    PlayMusic {
        music_id: u64,
        data: Arc<SoundData>,
        shared: Arc<MusicShared>,
        volume: f32,
        looping: bool,
    },
    StopMusic { music_id: u64 },
    SetMusicVolume { music_id: u64, volume: f32 },
    SetMaster(f32),
    SetListener { pos: [f32; 3], forward: [f32; 3] },
}

struct Voice {
    sound_id: u64,
    data: Arc<SoundData>,
    position: usize,
    volume: f32,
    spatial: Option<[f32; 3]>,
}

struct MusicVoice {
    music_id: u64,
    data: Arc<SoundData>,
    shared: Arc<MusicShared>,
    position: usize,
    volume: f32,
    looping: bool,
}

pub struct AudioRenderer {
    rx: Consumer<Cmd>,
    voices: Vec<Voice>,
    music: Vec<MusicVoice>,
    master: f32,
    listener_pos: [f32; 3],
    listener_forward: [f32; 3],
}

impl AudioRenderer {
    pub(super) fn new(rx: Consumer<Cmd>) -> Self {
        Self {
            rx,
            voices: Vec::with_capacity(64),
            music: Vec::with_capacity(4),
            master: 1.0,
            listener_pos: [0.0; 3],
            listener_forward: [0.0, 0.0, -1.0],
        }
    }

    fn apply(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::PlaySound { sound_id, data, volume, spatial } => {
                self.voices.push(Voice { sound_id, data, position: 0, volume, spatial });
            }
            Cmd::StopSound { sound_id } => {
                self.voices.retain(|v| v.sound_id != sound_id);
            }
            Cmd::SetSoundVolume { sound_id, volume } => {
                for v in &mut self.voices {
                    if v.sound_id == sound_id {
                        v.volume = volume;
                    }
                }
            }
            Cmd::PlayMusic { music_id, data, shared, volume, looping } => {
                // Restart-from-zero semantics (matches the old mixer).
                self.music.retain(|m| m.music_id != music_id);
                shared.playing.store(true, Ordering::Relaxed);
                shared.position.store(0, Ordering::Relaxed);
                self.music.push(MusicVoice { music_id, data, shared, position: 0, volume, looping });
            }
            Cmd::StopMusic { music_id } => {
                if let Some(m) = self.music.iter().position(|m| m.music_id == music_id) {
                    let m = self.music.swap_remove(m);
                    m.shared.playing.store(false, Ordering::Relaxed);
                    m.shared.position.store(0, Ordering::Relaxed);
                }
            }
            Cmd::SetMusicVolume { music_id, volume } => {
                for m in &mut self.music {
                    if m.music_id == music_id {
                        m.volume = volume;
                    }
                }
            }
            Cmd::SetMaster(v) => self.master = v,
            Cmd::SetListener { pos, forward } => {
                self.listener_pos = pos;
                self.listener_forward = forward;
            }
        }
    }

    /// Mix all playing voices into `output` (interleaved stereo f32).
    /// Wait-free: drains pending commands, then mixes. Call only from the
    /// thread that owns this renderer.
    pub fn mix(&mut self, output: &mut [f32]) {
        // Bounded drain — the ring's capacity bounds this loop.
        while let Some(cmd) = self.rx.pop() {
            self.apply(cmd);
        }

        for sample in output.iter_mut() {
            *sample = 0.0;
        }

        // Spatial audio: listener-relative parameters, computed once.
        let [lx, ly, lz] = self.listener_pos;
        let [lfx, _lfy, lfz] = self.listener_forward; // "right" math projects out Y
        // Listener right vector (cross of forward and up=[0,1,0])
        let lrx = lfz;
        let lrz = -lfx;
        let lr_len = (lrx * lrx + lrz * lrz).sqrt().max(0.001);
        let master = self.master;

        // Sound effects
        self.voices.retain_mut(|v| {
            let sound = &v.data;

            let (gain_l, gain_r) = if let Some([sx, sy, sz]) = v.spatial {
                let dx = sx - lx;
                let dy = sy - ly;
                let dz = sz - lz;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                // Distance attenuation: 1/distance, clamped
                let attenuation = (1.0 / dist).min(1.0);
                // Pan: dot of source direction with listener right
                let pan = ((dx * lrx + dz * lrz) / (dist * lr_len)).clamp(-1.0, 1.0);
                (attenuation * (1.0 - pan) * 0.5, attenuation * (1.0 + pan) * 0.5)
            } else {
                (1.0, 1.0)
            };

            let vol_l = v.volume * master * gain_l;
            let vol_r = v.volume * master * gain_r;
            let mut i = 0;
            while i < output.len() && v.position < sound.samples.len() {
                if sound.channels == 1 {
                    let sample = sound.samples[v.position];
                    output[i] += sample * vol_l;
                    if i + 1 < output.len() {
                        output[i + 1] += sample * vol_r;
                    }
                    v.position += 1;
                    i += 2;
                } else {
                    output[i] += sound.samples[v.position] * vol_l;
                    v.position += 1;
                    if i + 1 < output.len() && v.position < sound.samples.len() {
                        output[i + 1] += sound.samples[v.position] * vol_r;
                        v.position += 1;
                    }
                    i += 2;
                }
            }
            v.position < sound.samples.len()
        });

        // Music
        self.music.retain_mut(|m| {
            let vol = m.volume * master;
            let mut i = 0;
            while i < output.len() && m.position < m.data.samples.len() {
                if m.data.channels == 1 {
                    let sample = m.data.samples[m.position] * vol;
                    output[i] += sample;
                    if i + 1 < output.len() {
                        output[i + 1] += sample;
                    }
                    m.position += 1;
                    i += 2;
                } else {
                    output[i] += m.data.samples[m.position] * vol;
                    m.position += 1;
                    i += 1;
                }
            }
            if m.position >= m.data.samples.len() {
                if m.looping {
                    m.position = 0;
                } else {
                    m.shared.playing.store(false, Ordering::Relaxed);
                    m.shared.position.store(0, Ordering::Relaxed);
                    return false;
                }
            }
            m.shared.position.store(m.position, Ordering::Relaxed);
            true
        });
    }
}
