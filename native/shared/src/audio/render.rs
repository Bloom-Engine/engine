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
//!
//! # Spatialization (EN-062)
//!
//! Every voice carries a stable `voice_id`, so a playing voice can be
//! moved, re-volumed, re-pitched, filtered or stopped after the trigger —
//! that is what turns "a sound played at a point" into an *emitter* (a
//! river you walk along, a creature circling you). Spatial voices get:
//!
//! - **Inverse-clamped distance model** — `ref / (ref + rolloff·(d−ref))`,
//!   which with ref=1, rolloff=1 is exactly the old 1/d curve, so the
//!   pre-EN-062 API keeps its loudness. Past `max_dist` the voice is
//!   culled from the mix but keeps its playback head advancing.
//! - **Equal-power panning** (was linear, which dipped centered sources
//!   6 dB and made walk-bys "swell" at the ears). Width is 0.85: a real
//!   head leaks sound around itself; a 100%-one-ear source reads as a
//!   headphone artifact, not a position.
//! - **Air absorption** — a distance-driven low-pass (20 kHz at the ear
//!   falling exponentially with range). Distant gunfire is dull *before*
//!   it is quiet; that ordering is most of what "far away" sounds like.
//! - **Rear cue** — sources behind the listener are low-passed toward
//!   ~4.5 kHz and dipped ~1.5 dB. Cheap head-shadow approximation; it is
//!   the difference between "somewhere" and "behind you".
//! - **Doppler** — playback rate bends with radial velocity (343 m/s
//!   speed of sound, clamped, smoothed). Computed from the *distance
//!   delta* per block, so listener motion contributes symmetrically. A
//!   teleporting emitter (pool voice re-targeted to a new enemy) is
//!   detected by an impossible radial speed and resets cleanly instead
//!   of chirping.
//!
//! All per-voice gains ramp linearly across each mix block — a moving
//! emitter or a per-frame volume ride must never zipper or click.
//!
//! Voices resample with linear interpolation (`frame_pos` is fractional):
//! this is what makes doppler/pitch possible, and it also plays each
//! asset at its *authored* rate — previously a 44.1 kHz file on a 48 kHz
//! device played ~9% fast and sharp. Music streams are untouched.

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
        /// Stable per-play id — the handle for every SetVoice*/StopVoice.
        voice_id: u64,
        data: Arc<SoundData>,
        volume: f32,
        /// Some(world position) for spatial sounds.
        spatial: Option<[f32; 3]>,
        /// EN-062 — loop the sample seamlessly until StopVoice/StopSound.
        looping: bool,
        /// EN-062 — distance model. ref=1, rolloff=1 == the classic 1/d.
        ref_dist: f32,
        max_dist: f32,
        rolloff: f32,
        /// EN-062 — playback-rate multiplier (doppler multiplies on top).
        pitch: f32,
        /// EN-029 — mix bus (see [`bus`]), reverb send (0..1) and low-pass
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

    // ---- EN-062: live-voice control ------------------------------------
    SetVoicePosition { voice_id: u64, pos: [f32; 3] },
    /// Fades out over ~1 block and removes — a hard cut on a looping bed
    /// mid-waveform is an audible click.
    StopVoice { voice_id: u64 },
    SetVoiceVolume { voice_id: u64, volume: f32 },
    SetVoicePitch { voice_id: u64, pitch: f32 },
    /// Per-VOICE occlusion; SetSoundLowpass muffles every voice of a sound,
    /// which is wrong the moment two emitters share an asset.
    SetVoiceLowpass { voice_id: u64, cutoff: f32 },
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

/// Speed of sound, m/s — the doppler constant.
const SOUND_SPEED: f32 = 343.0;
/// A radial speed past this is a teleport (voice re-targeted), not motion:
/// reset doppler instead of bending pitch through the jump.
const DOPPLER_TELEPORT: f32 = 150.0;
/// Cutoffs above this bypass the one-pole entirely (inaudible + not free).
const LP_ENGAGE_HZ: f32 = 16_000.0;

struct Voice {
    sound_id: u64,
    voice_id: u64,
    data: Arc<SoundData>,
    /// Playback head in FRAMES, fractional — doppler and pitch resample.
    frame_pos: f64,
    volume: f32,
    spatial: Option<[f32; 3]>,
    looping: bool,
    ref_dist: f32,
    max_dist: f32,
    rolloff: f32,
    /// Game-set playback rate. Doppler multiplies on top of it.
    pitch: f32,
    /// Smoothed doppler rate.
    doppler: f32,
    /// Listener distance last block; < 0 = not yet seeded.
    prev_dist: f32,
    /// Current combined per-channel gains (spatial × volume × master × bus),
    /// ramped toward each block's target so moving emitters never zipper.
    g_l: f32,
    g_r: f32,
    /// False until the first block computes real targets (skip the ramp-in;
    /// the asset's own attack handles the onset).
    seeded: bool,
    /// StopVoice: fade to silence over a block, then drop.
    stopping: bool,
    bus: u8,
    send: f32,
    /// Low-pass cutoff, Hz. <= 0 = bypass.
    lowpass: f32,
    /// One-pole filter memory, per output channel.
    lp_z: [f32; 2],
}

impl Voice {
    /// Read frame `idx` as stereo (mono duplicates).
    #[inline]
    fn frame(&self, idx: usize) -> (f32, f32) {
        if self.data.channels <= 1 {
            let s = self.data.samples[idx];
            (s, s)
        } else {
            (self.data.samples[idx * 2], self.data.samples[idx * 2 + 1])
        }
    }
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
            Cmd::PlaySound {
                sound_id, voice_id, data, volume, spatial, looping,
                ref_dist, max_dist, rolloff, pitch, bus, send, lowpass,
            } => {
                self.voices.push(Voice {
                    sound_id, voice_id, data, frame_pos: 0.0, volume, spatial,
                    looping,
                    ref_dist: ref_dist.max(1e-3),
                    max_dist: max_dist.max(0.0),
                    rolloff: rolloff.max(0.0),
                    pitch: pitch.clamp(0.25, 4.0),
                    doppler: 1.0,
                    prev_dist: -1.0,
                    g_l: 0.0, g_r: 0.0, seeded: false, stopping: false,
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
            Cmd::SetVoicePosition { voice_id, pos } => {
                for v in &mut self.voices {
                    if v.voice_id == voice_id { v.spatial = Some(pos); }
                }
            }
            Cmd::StopVoice { voice_id } => {
                for v in &mut self.voices {
                    if v.voice_id == voice_id { v.stopping = true; }
                }
            }
            Cmd::SetVoiceVolume { voice_id, volume } => {
                for v in &mut self.voices {
                    if v.voice_id == voice_id { v.volume = volume.max(0.0); }
                }
            }
            Cmd::SetVoicePitch { voice_id, pitch } => {
                for v in &mut self.voices {
                    if v.voice_id == voice_id { v.pitch = pitch.clamp(0.25, 4.0); }
                }
            }
            Cmd::SetVoiceLowpass { voice_id, cutoff } => {
                for v in &mut self.voices {
                    if v.voice_id == voice_id { v.lowpass = cutoff.max(0.0); }
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
        let out_frames = output.len() / 2;
        let block_dt = (out_frames as f32).max(1.0) / self.sample_rate;
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
        // Listener right = cross(forward, up=[0,1,0]) = (-fz, 0, fx) — the SAME
        // vector mat4_look_at calls `s` (screen-right). EN-062 flipped the sign
        // here: this used to be (fz, -fx), which is screen-LEFT, so the whole
        // stereo field was mirrored — a shriek on screen-left panned right.
        let lrx = -lfz;
        let lrz = lfx;
        let lr_len = (lrx * lrx + lrz * lrz).sqrt().max(0.001);
        let master = self.master;
        let sample_rate = self.sample_rate;

        // Split the borrow: the voice loop needs `voices` and `send_buf`
        // mutably at once, and the compiler can only see they're disjoint if
        // we name them separately.
        let Self { voices, send_buf, reverb_l, reverb_r, music, .. } = self;

        // Sound effects
        voices.retain_mut(|v| {
            let channels = v.data.channels.max(1) as usize;
            let frames = v.data.samples.len() / channels;
            if frames == 0 { return false; }

            // ---- per-block spatial targets --------------------------------
            // Manual (occlusion) low-pass; spatial cues can only lower it.
            let mut cutoff = if v.lowpass > 0.0 { v.lowpass } else { f32::MAX };
            let (sg_l, sg_r, dist) = if let Some([sx, sy, sz]) = v.spatial {
                let dx = sx - lx;
                let dy = sy - ly;
                let dz = sz - lz;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-4);
                // Inverse-clamped distance model. ref=1, rolloff=1 == the old
                // 1/d exactly, so pre-EN-062 callers keep their loudness.
                let refd = v.ref_dist;
                let att = if dist > v.max_dist {
                    0.0
                } else {
                    (refd / (refd + v.rolloff * (dist.max(refd) - refd))).min(1.0)
                };
                // Equal-power pan from the horizontal azimuth, at 0.85 width
                // (a head is not an infinite baffle).
                let pan = ((dx * lrx + dz * lrz) / (dist * lr_len)).clamp(-1.0, 1.0) * 0.85;
                let theta = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
                let mut gl = att * theta.cos();
                let mut gr = att * theta.sin();
                // Air absorption: distance dulls before it silences.
                cutoff = cutoff.min(20_000.0 * (-0.006 * dist).exp());
                // Rear cue: head shadow behind the listener — darker and a
                // touch quieter. Horizontal only; overhead stays neutral.
                let f_len2 = lfx * lfx + lfz * lfz;
                if f_len2 > 1e-6 {
                    let back = (-(dx * lfx + dz * lfz) / (dist * f_len2.sqrt())).clamp(0.0, 1.0);
                    if back > 0.0 {
                        cutoff = cutoff.min(18_000.0 - 13_500.0 * back);
                        let dip = 1.0 - 0.15 * back;
                        gl *= dip;
                        gr *= dip;
                    }
                }
                (gl, gr, dist)
            } else {
                (1.0, 1.0, -1.0)
            };

            // ---- doppler ---------------------------------------------------
            // From the block-to-block distance delta, so listener motion and
            // source motion both count. Impossible speeds are teleports
            // (re-targeted pool voices) — snap, don't chirp.
            if dist >= 0.0 {
                if v.prev_dist >= 0.0 {
                    let vr = (dist - v.prev_dist) / block_dt.max(1e-4);
                    if vr.abs() > DOPPLER_TELEPORT {
                        v.doppler = 1.0;
                    } else {
                        let target = (SOUND_SPEED / (SOUND_SPEED + vr)).clamp(0.5, 2.0);
                        v.doppler += (target - v.doppler) * 0.25;
                    }
                }
                v.prev_dist = dist;
            }
            let step = (v.pitch * v.doppler) as f64
                * (v.data.sample_rate.max(8_000) as f64 / sample_rate as f64);

            // ---- combined gain targets, ramped across the block ------------
            let bg = bus_gain[(v.bus as usize).min(bus::COUNT - 1)];
            let (t_l, t_r) = if v.stopping {
                (0.0, 0.0)
            } else {
                (sg_l * v.volume * master * bg, sg_r * v.volume * master * bg)
            };
            if !v.seeded {
                v.g_l = t_l;
                v.g_r = t_r;
                v.seeded = true;
            }

            // Fully silent (culled by distance, or a zero-volume ambient bed):
            // advance the head arithmetically and skip the per-sample work.
            if t_l < 1e-5 && t_r < 1e-5 && v.g_l < 1e-5 && v.g_r < 1e-5 {
                if v.stopping { return false; }
                v.g_l = t_l;
                v.g_r = t_r;
                v.frame_pos += step * out_frames as f64;
                if v.frame_pos >= frames as f64 {
                    if v.looping {
                        v.frame_pos %= frames as f64;
                    } else {
                        return false;
                    }
                }
                return true;
            }

            let inv = 1.0 / (out_frames as f32).max(1.0);
            let dg_l = (t_l - v.g_l) * inv;
            let dg_r = (t_r - v.g_r) * inv;

            // One-pole low-pass coefficient: occlusion, air and rear cues all
            // fold into one cutoff. Muffling reads as geometry/distance in a
            // way that turning the volume down never does.
            let lp_a = if cutoff < LP_ENGAGE_HZ.min(sample_rate * 0.45) {
                Some((-2.0 * std::f32::consts::PI * cutoff / sample_rate).exp())
            } else {
                None
            };
            let send = v.send;

            let mut ended = false;
            let mut f = 0usize;
            while f < out_frames {
                let idx = v.frame_pos as usize;
                if idx >= frames { ended = !v.looping; break; }
                let frac = (v.frame_pos - idx as f64) as f32;
                let (s0l, s0r) = v.frame(idx);
                let nidx = if idx + 1 < frames { idx + 1 } else if v.looping { 0 } else { idx };
                let (s1l, s1r) = v.frame(nidx);
                let mut sl = s0l + (s1l - s0l) * frac;
                let mut sr = s0r + (s1r - s0r) * frac;

                if let Some(a) = lp_a {
                    v.lp_z[0] = sl * (1.0 - a) + v.lp_z[0] * a;
                    v.lp_z[1] = sr * (1.0 - a) + v.lp_z[1] * a;
                    sl = v.lp_z[0];
                    sr = v.lp_z[1];
                }

                let i = f * 2;
                let ol = sl * v.g_l;
                let or = sr * v.g_r;
                output[i] += ol;
                if i + 1 < output.len() { output[i + 1] += or; }

                if reverb_active && send > 0.0 {
                    send_buf[i] += ol * send;
                    if i + 1 < output.len() { send_buf[i + 1] += or * send; }
                }

                v.g_l += dg_l;
                v.g_r += dg_r;
                v.frame_pos += step;
                if v.frame_pos >= frames as f64 {
                    if v.looping {
                        v.frame_pos -= frames as f64;
                    } else {
                        ended = true;
                        break;
                    }
                }
                f += 1;
            }
            if !ended {
                // Land exactly on the target — no float drift across blocks.
                v.g_l = t_l;
                v.g_r = t_r;
            }
            if v.stopping && v.g_l < 1e-4 && v.g_r < 1e-4 {
                return false;
            }
            !ended
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
