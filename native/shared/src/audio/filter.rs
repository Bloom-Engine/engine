//! The mixer's low-pass: one-pole stages whose coefficient glides.
//!
//! A one-pole low-pass `z = x·(1−a) + z·a` has a property the mixer leans on:
//! at `a = 0` it is the identity, exactly, and its state then simply equals the
//! input. So "bypass" is not a separate code path the filter jumps in and out
//! of (which clicks: the state is stale the moment it re-engages), it is the
//! end of the same continuous range. Every change of cutoff, including on and
//! off, is a glide of `a`, ramped per sample, and nothing ever steps.
//!
//! Voices use one stage (6 dB/octave, the occlusion/air/rear cue they always
//! had). Music streams use two (12 dB/octave), tuned so the PAIR is 3 dB down
//! at the requested cutoff: a voice line heard from under water has to lose
//! its consonants, which one gentle stage does not do.

/// Above this a cutoff is inaudible and the filter is left out (`a = 0`).
pub(super) const LP_ENGAGE_HZ: f32 = 16_000.0;

/// How long a music stream's cutoff takes to settle after a change (one-pole
/// time constant across mix blocks, on top of the per-sample ramp inside each
/// block). 20 ms: fast enough to follow a dive, slow enough not to zipper.
pub(super) const LP_GLIDE_S: f32 = 0.02;

/// Two identical one-pole stages are 3 dB down at `0.6436 ×` their own corner
/// (`sqrt(sqrt(2) − 1)`); dividing by it puts the pair's −3 dB at the cutoff.
const TWO_STAGE_CORNER: f32 = 0.643_594_3;

/// The one-pole coefficient for `cutoff_hz` over `stages` stages at
/// `sample_rate`; 0 (transparent) when the cutoff is off (`<= 0`) or too high
/// to hear.
pub(super) fn lowpass_coeff(cutoff_hz: f32, sample_rate: f32, stages: usize) -> f32 {
    if cutoff_hz <= 0.0 || cutoff_hz >= LP_ENGAGE_HZ.min(sample_rate * 0.45) {
        return 0.0;
    }
    let corner = if stages >= 2 {
        cutoff_hz / TWO_STAGE_CORNER
    } else {
        cutoff_hz
    };
    let corner = corner.min(sample_rate * 0.45);
    (-2.0 * std::f32::consts::PI * corner / sample_rate).exp()
}

/// Stereo low-pass of up to two stages with a gliding coefficient.
#[derive(Clone, Copy, Default)]
pub(super) struct Lowpass {
    /// Coefficient in use at the end of the last block.
    pub a: f32,
    /// `[stage][channel]` filter memory.
    pub z: [[f32; 2]; 2],
}

impl Lowpass {
    /// Start a block that glides the coefficient from where it is to `a_end`
    /// over `frames` frames. Returns `None` when the filter is transparent for
    /// the whole block (both ends 0): the caller skips it, bit-exact.
    #[inline]
    pub fn begin(&mut self, a_end: f32, frames: usize) -> Option<(f32, f32)> {
        let a0 = self.a;
        self.a = a_end;
        if a0 <= 0.0 && a_end <= 0.0 {
            return None;
        }
        Some((a0, (a_end - a0) / (frames as f32).max(1.0)))
    }

    /// Filter one sample of `channel` (0 or 1) through `stages` stages.
    #[inline]
    pub fn tick(&mut self, channel: usize, x: f32, a: f32, stages: usize) -> f32 {
        let mut y = x;
        for s in 0..stages.min(2) {
            let z = &mut self.z[s][channel & 1];
            *z = y * (1.0 - a) + *z * a;
            y = *z;
        }
        y
    }
}

/// Where a music stream's coefficient goes this block: a one-pole glide
/// toward `target` with time constant [`LP_GLIDE_S`], snapped when close.
pub(super) fn glide(current: f32, target: f32, block_dt: f32) -> f32 {
    let k = 1.0 - (-block_dt / LP_GLIDE_S).exp();
    let next = current + (target - current) * k.clamp(0.0, 1.0);
    if (next - target).abs() < 1e-4 {
        target
    } else {
        next
    }
}

#[cfg(test)]
mod tests {
    //! The mixer cannot be listened to in CI, so these render it offline:
    //! a known signal goes in through the public control surface, the output
    //! buffer is measured.
    use super::super::{AudioMixer, SoundData};

    const SR: f32 = 44_100.0;
    const BLOCK: usize = 512; // stereo samples per mix call = 256 frames

    fn sine(freq: f32, seconds: f32) -> SoundData {
        let n = (seconds * SR) as usize;
        SoundData {
            samples: (0..n)
                .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
                .collect(),
            sample_rate: SR as u32,
            channels: 1,
        }
    }

    /// Mix `blocks` blocks; return the left channel of all of them.
    fn render(a: &mut AudioMixer, blocks: usize) -> Vec<f32> {
        let mut left = Vec::with_capacity(blocks * BLOCK / 2);
        for _ in 0..blocks {
            let mut out = [0.0f32; BLOCK];
            a.mix_output(&mut out);
            left.extend(out.chunks(2).map(|f| f[0]));
        }
        left
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    fn db(x: f32) -> f32 {
        20.0 * x.max(1e-9).log10()
    }

    /// Gain in dB of a music stream low-passed at `cutoff` for a sine at
    /// `freq`, measured after the filter and glide have settled.
    fn music_gain_db(freq: f32, cutoff: f32) -> f32 {
        let mut a = AudioMixer::new();
        let h = a.load_music(sine(freq, 1.0));
        a.set_music_lowpass(h, cutoff);
        a.play_music(h);
        let left = render(&mut a, 60); // ~0.35 s
        let tail = &left[left.len() / 2..];
        db(rms(tail) / (0.5 / std::f32::consts::SQRT_2))
    }

    #[test]
    fn music_lowpass_follows_its_curve() {
        // Two stages, -3 dB at the cutoff: the sweep must be flat well below
        // it, about -3 dB at it, and falling at 12 dB/octave above it (one stage
        // would fall ~6 dB from 2 to 4 kHz here, two fall ~11).
        let cut = 500.0;
        let pass = music_gain_db(60.0, cut);
        let at = music_gain_db(cut, cut);
        let oct2 = music_gain_db(2_000.0, cut);
        let oct3 = music_gain_db(4_000.0, cut);
        assert!(pass > -0.5, "60 Hz should pass: {pass:.2} dB");
        assert!((at + 3.0).abs() < 1.0, "-3 dB at the cutoff, got {at:.2}");
        assert!(oct2 < -15.0, "2 kHz (2 octaves up) only {oct2:.2} dB");
        assert!(oct3 < -26.0, "4 kHz (3 octaves up) only {oct3:.2} dB");
        assert!(
            oct3 < oct2 - 9.0,
            "slope too shallow: {oct2:.1} → {oct3:.1}"
        );
        // and a sweep is monotonic
        let mut last = 1.0;
        for f in [100.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0] {
            let g = music_gain_db(f, cut);
            assert!(g < last, "not monotonic at {f} Hz: {g:.2} after {last:.2}");
            last = g;
        }
    }

    #[test]
    fn music_lowpass_off_is_bit_exact() {
        // 0 = bypass must leave the stream untouched, sample for sample.
        let mut plain = AudioMixer::new();
        let h = plain.load_music(sine(3_000.0, 0.2));
        plain.play_music(h);
        let a = render(&mut plain, 8);

        let mut off = AudioMixer::new();
        let h = off.load_music(sine(3_000.0, 0.2));
        off.set_music_lowpass(h, 0.0);
        off.play_music(h);
        let b = render(&mut off, 8);
        assert_eq!(a, b);
    }

    #[test]
    fn music_lowpass_changes_while_playing_without_a_click() {
        // A 4 kHz line playing; the cutoff drops to 400 Hz mid-stream (he goes
        // under), then back off. The signal's largest sample-to-sample step is
        // bounded by its own slope (2π·f·A/sr ≈ 0.28); a filter that stepped
        // its state would jump far past it. And it must actually muffle.
        let mut a = AudioMixer::new();
        let h = a.load_music(sine(4_000.0, 2.0));
        a.play_music(h);
        let mut left = render(&mut a, 20);
        let before = rms(&left[left.len() - 1024..]);
        a.set_music_lowpass(h, 400.0);
        left.extend(render(&mut a, 40));
        let under = rms(&left[left.len() - 1024..]);
        a.set_music_lowpass(h, 0.0);
        left.extend(render(&mut a, 40));
        let after = rms(&left[left.len() - 1024..]);

        let slope = 2.0 * std::f32::consts::PI * 4_000.0 * 0.5 / SR;
        let max_step = left
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step <= slope * 1.05,
            "a step of {max_step} beats the signal's own slope {slope}"
        );
        assert!(
            db(under / before) < -25.0,
            "not muffled: {:.1} dB",
            db(under / before)
        );
        assert!((db(after / before)).abs() < 0.1, "did not come back");
    }

    #[test]
    fn music_lowpass_set_before_play_applies_from_the_start() {
        let mut a = AudioMixer::new();
        let h = a.load_music(sine(6_000.0, 0.5));
        a.set_music_lowpass(h, 300.0);
        a.play_music(h);
        let left = render(&mut a, 4);
        assert!(db(rms(&left[256..]) / 0.3535) < -25.0);
    }

    #[test]
    fn sound_lowpass_switching_on_mid_voice_does_not_click() {
        // The voice path used to jump in and out of its filter with stale
        // state. Now the coefficient glides from 0.
        let mut a = AudioMixer::new();
        let h = a.load_sound(sine(3_000.0, 1.0));
        a.play_sound(h);
        let mut left = render(&mut a, 10);
        a.set_sound_lowpass(h, 300.0);
        left.extend(render(&mut a, 10));
        a.set_sound_lowpass(h, 0.0);
        left.extend(render(&mut a, 10));
        let slope = 2.0 * std::f32::consts::PI * 3_000.0 * 0.5 / SR;
        let max_step = left
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_step <= slope * 1.05, "step {max_step} > slope {slope}");
    }
}
