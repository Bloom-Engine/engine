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
#[cfg(not(target_arch = "wasm32"))]
use super::stream::{StreamConsumer, StreamMsg};
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
        /// EN-029 — mix bus (see [`Bus`]), reverb send (0..1) and low-pass
        /// cutoff in Hz (<= 0 or >= NYQUIST = bypass).
        bus: u8,
        send: f32,
        lowpass: f32,
    },
    StopSound { sound_id: u64 },
    SetSoundVolume { sound_id: u64, volume: f32 },
    PlayMusic {
        music_id: u64,
        payload: MusicPayload,
        shared: Arc<MusicShared>,
        volume: f32,
        looping: bool,
    },
    StopMusic { music_id: u64 },
    SetMusicVolume { music_id: u64, volume: f32 },
    SetMaster(f32),
    SetListener { pos: [f32; 3], forward: [f32; 3] },

    // ---- EN-029 -------------------------------------------------------
    SetBusGain { bus: u8, gain: f32 },
    /// Momentary sidechain-style duck: pull `bus` down by `amount` (linear,
    /// 0..1) over `attack` seconds, hold, then release over `release`.
    DuckBus { bus: u8, amount: f32, attack: f32, release: f32, hold: f32 },
    SetReverbParams { size: f32, damp: f32, wet: f32 },
    /// Per-playing-voice sends — this is the occlusion primitive. The game
    /// raycasts and decides; the mixer just filters.
    SetSoundLowpass { sound_id: u64, cutoff: f32 },
    SetSoundSend { sound_id: u64, send: f32 },
}

/// Mix buses. Kept tiny and fixed: a general submix graph is a lot of
/// machinery for a game that needs exactly "duck the music when I'm hit" and
/// "don't let UI beeps ride the reverb".
pub mod bus {
    pub const SFX: u8 = 0;
    pub const MUSIC: u8 = 1;
    pub const UI: u8 = 2;
    pub const COUNT: usize = 3;
}

/// A duck envelope per bus. `gain` is the static level the game set; `duck` is
/// the momentary attenuation on top of it, and it is the one that moves.
#[derive(Clone, Copy)]
struct BusState {
    gain: f32,
    /// Current attenuation, 0 = untouched, 1 = fully ducked.
    duck: f32,
    /// Where `duck` is heading while the hold lasts.
    duck_target: f32,
    attack: f32,
    release: f32,
    hold_left: f32,
}

impl Default for BusState {
    fn default() -> Self {
        Self { gain: 1.0, duck: 0.0, duck_target: 0.0, attack: 0.01, release: 0.3, hold_left: 0.0 }
    }
}

impl BusState {
    fn set_duck(&mut self, amount: f32, attack: f32, release: f32, hold: f32) {
        self.duck_target = amount.clamp(0.0, 1.0);
        self.attack = attack.max(0.0);
        self.release = release.max(0.0);
        self.hold_left = hold.max(0.0);
    }

    /// Advance the duck envelope by one mix block.
    fn advance(&mut self, dt: f32) {
        let target = if self.hold_left > 0.0 {
            self.hold_left -= dt;
            self.duck_target
        } else {
            0.0
        };
        let rate = if target > self.duck { self.attack } else { self.release };
        if rate <= 0.0 {
            self.duck = target;
        } else {
            // One-pole toward the target — exponential, so a long block can
            // never overshoot the destination and ring.
            let k = (dt / rate).clamp(0.0, 1.0);
            self.duck += (target - self.duck) * k;
        }
    }

    fn current(&self) -> f32 { (self.gain * (1.0 - self.duck)).clamp(0.0, 4.0) }
}

struct Voice {
    sound_id: u64,
    data: Arc<SoundData>,
    position: usize,
    volume: f32,
    spatial: Option<[f32; 3]>,
    bus: u8,
    send: f32,
    /// Low-pass cutoff, Hz. <= 0 = bypass.
    lowpass: f32,
    /// One-pole filter memory, per output channel.
    lp_z: [f32; 2],
}

/// How a music voice gets its samples.
pub enum MusicPayload {
    Full(Arc<SoundData>),
    /// Chunks arrive from a background decode worker; the worker handles
    /// looping internally (End only arrives for finished non-loop tracks).
    #[cfg(not(target_arch = "wasm32"))]
    Stream { consumer: StreamConsumer, channels: u16 },
}

enum MusicSamples {
    Full { data: Arc<SoundData>, position: usize },
    #[cfg(not(target_arch = "wasm32"))]
    Stream {
        consumer: StreamConsumer,
        channels: u16,
        current: Vec<f32>,
        offset: usize,
        ended: bool,
    },
}

struct MusicVoice {
    music_id: u64,
    samples: MusicSamples,
    shared: Arc<MusicShared>,
    volume: f32,
    looping: bool,
    /// Total samples consumed (drives the shared position mirror).
    consumed: usize,
}

/// EN-029 — a Schroeder reverb: parallel comb filters (the density) into
/// series allpasses (the diffusion). Freeverb's topology, trimmed to 4+2 per
/// channel because this runs on the audio thread of a game, not a DAW.
///
/// Delay lengths are the classic 44.1 kHz tunings. At 48 kHz the room comes
/// out ~9% smaller, which is inaudible for a gunshot tail and not worth
/// resampling the delay lines for.
struct Reverb {
    combs: [Vec<f32>; 4],
    comb_idx: [usize; 4],
    comb_z: [f32; 4],
    allpass: [Vec<f32>; 2],
    ap_idx: [usize; 2],
    /// 0..1 — feedback in the combs. Bigger = longer tail.
    size: f32,
    /// 0..1 — high-frequency absorption in the tail.
    damp: f32,
    /// 0..1 — how much of the wet signal reaches the output.
    wet: f32,
}

impl Reverb {
    const COMB_LEN: [usize; 4] = [1116, 1188, 1277, 1356];
    const AP_LEN: [usize; 2] = [556, 441];

    fn new() -> Self {
        Self {
            combs: [
                vec![0.0; Self::COMB_LEN[0]],
                vec![0.0; Self::COMB_LEN[1]],
                vec![0.0; Self::COMB_LEN[2]],
                vec![0.0; Self::COMB_LEN[3]],
            ],
            comb_idx: [0; 4],
            comb_z: [0.0; 4],
            allpass: [vec![0.0; Self::AP_LEN[0]], vec![0.0; Self::AP_LEN[1]]],
            ap_idx: [0; 2],
            size: 0.7,
            damp: 0.5,
            wet: 0.0,
        }
    }

    /// One mono sample in, one wet sample out.
    fn process(&mut self, input: f32) -> f32 {
        let feedback = 0.7 + self.size * 0.28; // 0.70..0.98
        let damp = self.damp.clamp(0.0, 1.0);

        let mut out = 0.0;
        for c in 0..4 {
            let i = self.comb_idx[c];
            let delayed = self.combs[c][i];
            out += delayed;
            // Lowpass in the feedback path = the damping.
            self.comb_z[c] = delayed * (1.0 - damp) + self.comb_z[c] * damp;
            self.combs[c][i] = input + self.comb_z[c] * feedback;
            self.comb_idx[c] = (i + 1) % Self::COMB_LEN[c];
        }
        out *= 0.25;

        for a in 0..2 {
            let i = self.ap_idx[a];
            let delayed = self.allpass[a][i];
            let v = out - delayed * 0.5;
            self.allpass[a][i] = v;
            out = delayed + v * 0.5;
            self.ap_idx[a] = (i + 1) % Self::AP_LEN[a];
        }
        out
    }
}

pub struct AudioRenderer {
    rx: Consumer<Cmd>,
    voices: Vec<Voice>,
    music: Vec<MusicVoice>,
    master: f32,
    listener_pos: [f32; 3],
    listener_forward: [f32; 3],

    // EN-029
    buses: [BusState; bus::COUNT],
    reverb_l: Reverb,
    reverb_r: Reverb,
    /// Reverb input accumulator, one slot per output sample. Preallocated so
    /// the audio thread never allocates.
    send_buf: Vec<f32>,
    sample_rate: f32,
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
            buses: [BusState::default(); bus::COUNT],
            reverb_l: Reverb::new(),
            reverb_r: Reverb::new(),
            send_buf: vec![0.0; 8192],
            sample_rate: 44_100.0,
        }
    }

    pub fn set_sample_rate(&mut self, sr: f32) {
        if sr > 1000.0 { self.sample_rate = sr; }
    }

    fn apply(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::PlaySound { sound_id, data, volume, spatial, bus, send, lowpass } => {
                self.voices.push(Voice {
                    sound_id, data, position: 0, volume, spatial,
                    bus, send, lowpass, lp_z: [0.0; 2],
                });
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
            Cmd::PlayMusic { music_id, payload, shared, volume, looping } => {
                // Restart-from-zero semantics (matches the old mixer).
                self.music.retain(|m| m.music_id != music_id);
                shared.playing.store(true, Ordering::Relaxed);
                shared.position.store(0, Ordering::Relaxed);
                let samples = match payload {
                    MusicPayload::Full(data) => MusicSamples::Full { data, position: 0 },
                    #[cfg(not(target_arch = "wasm32"))]
                    MusicPayload::Stream { consumer, channels } => MusicSamples::Stream {
                        consumer,
                        channels,
                        current: Vec::new(),
                        offset: 0,
                        ended: false,
                    },
                };
                self.music.push(MusicVoice { music_id, samples, shared, volume, looping, consumed: 0 });
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
            Cmd::SetBusGain { bus, gain } => {
                if (bus as usize) < bus::COUNT {
                    self.buses[bus as usize].gain = gain.max(0.0);
                }
            }
            Cmd::DuckBus { bus, amount, attack, release, hold } => {
                if (bus as usize) < bus::COUNT {
                    self.buses[bus as usize].set_duck(amount, attack, release, hold);
                }
            }
            Cmd::SetReverbParams { size, damp, wet } => {
                for r in [&mut self.reverb_l, &mut self.reverb_r] {
                    r.size = size.clamp(0.0, 1.0);
                    r.damp = damp.clamp(0.0, 1.0);
                    r.wet = wet.clamp(0.0, 1.0);
                }
            }
            Cmd::SetSoundLowpass { sound_id, cutoff } => {
                for v in &mut self.voices {
                    if v.sound_id == sound_id { v.lowpass = cutoff; }
                }
            }
            Cmd::SetSoundSend { sound_id, send } => {
                for v in &mut self.voices {
                    if v.sound_id == sound_id { v.send = send.clamp(0.0, 1.0); }
                }
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

        // EN-029 — advance the per-bus duck envelopes once per block. Block
        // granularity is ~1-10 ms, far finer than any duck the ear resolves.
        let block_dt = (output.len() as f32 / 2.0) / self.sample_rate;
        for b in self.buses.iter_mut() {
            b.advance(block_dt);
        }
        let bus_gain = [
            self.buses[bus::SFX as usize].current(),
            self.buses[bus::MUSIC as usize].current(),
            self.buses[bus::UI as usize].current(),
        ];

        // Reverb send accumulator. Grows once if a host ever hands us a bigger
        // block than we sized for; steady-state it never allocates.
        if self.send_buf.len() < output.len() {
            self.send_buf.resize(output.len(), 0.0);
        }
        let reverb_active = self.reverb_l.wet > 0.0;
        if reverb_active {
            for s in self.send_buf[..output.len()].iter_mut() { *s = 0.0; }
        }

        // Spatial audio: listener-relative parameters, computed once.
        let [lx, ly, lz] = self.listener_pos;
        let [lfx, _lfy, lfz] = self.listener_forward; // "right" math projects out Y
        // Listener right vector (cross of forward and up=[0,1,0])
        let lrx = lfz;
        let lrz = -lfx;
        let lr_len = (lrx * lrx + lrz * lrz).sqrt().max(0.001);
        let master = self.master;
        let sample_rate = self.sample_rate;

        // Split the borrow: the voice loop needs `voices` and `send_buf`
        // mutably at once, and the compiler can only see they're disjoint if
        // we name them separately.
        let Self { voices, send_buf, reverb_l, reverb_r, music, .. } = self;

        // Sound effects
        voices.retain_mut(|v| {
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

            let bg = bus_gain[(v.bus as usize).min(bus::COUNT - 1)];
            let vol_l = v.volume * master * bg * gain_l;
            let vol_r = v.volume * master * bg * gain_r;

            // One-pole low-pass coefficient. This is the occlusion knob: a
            // muffled source is a source behind a wall, and it reads far more
            // like geometry than simply turning the volume down does.
            let lp_a = if v.lowpass > 0.0 && v.lowpass < sample_rate * 0.5 {
                let x = (-2.0 * std::f32::consts::PI * v.lowpass / sample_rate).exp();
                Some(x)
            } else {
                None
            };
            let send = v.send;

            let mut i = 0;
            while i < output.len() && v.position < sound.samples.len() {
                let (mut sl, mut sr) = if sound.channels == 1 {
                    let s = sound.samples[v.position];
                    v.position += 1;
                    (s, s)
                } else {
                    let l = sound.samples[v.position];
                    v.position += 1;
                    let r = if v.position < sound.samples.len() {
                        let r = sound.samples[v.position];
                        v.position += 1;
                        r
                    } else { l };
                    (l, r)
                };

                if let Some(a) = lp_a {
                    v.lp_z[0] = sl * (1.0 - a) + v.lp_z[0] * a;
                    v.lp_z[1] = sr * (1.0 - a) + v.lp_z[1] * a;
                    sl = v.lp_z[0];
                    sr = v.lp_z[1];
                }

                let ol = sl * vol_l;
                let or = sr * vol_r;
                output[i] += ol;
                if i + 1 < output.len() { output[i + 1] += or; }

                if reverb_active && send > 0.0 {
                    send_buf[i] += ol * send;
                    if i + 1 < output.len() { send_buf[i + 1] += or * send; }
                }
                i += 2;
            }
            v.position < sound.samples.len()
        });

        // Wet return. Processed after the dry voices so every send this block
        // contributed is in the tail.
        if reverb_active {
            let wet = reverb_l.wet;
            let mut i = 0;
            while i + 1 < output.len() {
                output[i] += reverb_l.process(send_buf[i]) * wet;
                output[i + 1] += reverb_r.process(send_buf[i + 1]) * wet;
                i += 2;
            }
        }

        // Music — on its own bus, which is the whole point: "duck the music
        // when the player takes a hit" is the single most-used mix move in the
        // genre and it needs music to be separable from SFX.
        let music_gain = bus_gain[bus::MUSIC as usize];
        music.retain_mut(|m| {
            let vol = m.volume * master * music_gain;
            let mut i = 0;
            let mut finished = false;
            match &mut m.samples {
                MusicSamples::Full { data, position } => {
                    while i < output.len() && *position < data.samples.len() {
                        if data.channels == 1 {
                            let sample = data.samples[*position] * vol;
                            output[i] += sample;
                            if i + 1 < output.len() {
                                output[i + 1] += sample;
                            }
                            *position += 1;
                            i += 2;
                        } else {
                            output[i] += data.samples[*position] * vol;
                            *position += 1;
                            i += 1;
                        }
                    }
                    if *position >= data.samples.len() {
                        if m.looping {
                            *position = 0;
                        } else {
                            finished = true;
                        }
                    }
                    m.consumed = *position;
                }
                #[cfg(not(target_arch = "wasm32"))]
                MusicSamples::Stream { consumer, channels, current, offset, ended } => {
                    let mono = *channels == 1;
                    while i < output.len() {
                        if *offset >= current.len() {
                            // Refill from the decode worker's ring.
                            match consumer.rx.pop() {
                                Some(StreamMsg::Chunk(c)) => {
                                    *current = c;
                                    *offset = 0;
                                }
                                Some(StreamMsg::End) => {
                                    *ended = true;
                                    break;
                                }
                                // Underrun: worker is behind (cold start
                                // or scheduling hiccup) — emit silence for
                                // the rest of this callback, resume next.
                                None => break,
                            }
                        }
                        if mono {
                            let sample = current[*offset] * vol;
                            output[i] += sample;
                            if i + 1 < output.len() {
                                output[i + 1] += sample;
                            }
                            *offset += 1;
                            m.consumed += 1;
                            i += 2;
                        } else {
                            output[i] += current[*offset] * vol;
                            *offset += 1;
                            m.consumed += 1;
                            i += 1;
                        }
                    }
                    if *ended && *offset >= current.len() {
                        finished = true;
                    }
                }
            }
            if finished {
                m.shared.playing.store(false, Ordering::Relaxed);
                m.shared.position.store(0, Ordering::Relaxed);
                return false;
            }
            m.shared.position.store(m.consumed, Ordering::Relaxed);
            true
        });
    }
}
