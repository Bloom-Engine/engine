//! Morph weights of an animation instance: what clips author (glTF
//! `weights` channels), what the game sets, and how the two combine.
//!
//! Slots are model-level (see `morph_targets.rs`): `modelMorphIndex` finds a
//! slot by name once at load, and a frame then sets weights by index.
//!
//! Each `animUpdate` the mixer samples the clips' weight tracks exactly as
//! it blends their joints (base clip, crossfade from the previous clip,
//! blend partners at the base clip's phase, the masked layer by its
//! weight; a slot a clip does not animate counts as 0 there, except in the
//! layer, where it keeps the base's value). The weight drawn is
//! `clamp(clip + explicit, -1, 1)`: the game's weights ADD to the clips'
//! (a lip-sync curve over an authored expression clip), and `animClearMorphs`
//! drops the game's.

use crate::models::{AnimationData, ModelManager};

/// One glTF `weights` channel: the weights of one mesh's targets over time.
#[derive(Clone, Debug, Default)]
pub struct MorphTrack {
    /// Model-level slot of each of the mesh's targets.
    pub slots: Vec<u32>,
    pub times: Vec<f32>,
    /// `times.len() * slots.len()` values, key-major.
    pub values: Vec<f32>,
    /// STEP interpolation (LINEAR otherwise; CUBICSPLINE is read as LINEAR
    /// through its key values).
    pub step: bool,
}

impl MorphTrack {
    /// Add this track's weights at `t` into `out` (indexed by slot), and
    /// mark the slots it animates in `touched`.
    fn sample_into(&self, t: f32, out: &mut [f32], touched: &mut [bool]) {
        let n = self.slots.len();
        let keys = self.times.len();
        if n == 0 || keys == 0 || self.values.len() < keys * n {
            return;
        }
        let (a, b, f) = if t <= self.times[0] {
            (0, 0, 0.0)
        } else if t >= self.times[keys - 1] {
            (keys - 1, keys - 1, 0.0)
        } else {
            let hi = self.times.partition_point(|&k| k <= t).min(keys - 1);
            let lo = hi - 1;
            let span = self.times[hi] - self.times[lo];
            let f = if span > 0.0 {
                (t - self.times[lo]) / span
            } else {
                0.0
            };
            if self.step {
                (lo, lo, 0.0)
            } else {
                (lo, hi, f)
            }
        };
        for (i, &slot) in self.slots.iter().enumerate() {
            let s = slot as usize;
            if s >= out.len() {
                continue;
            }
            let va = self.values[a * n + i];
            let vb = self.values[b * n + i];
            out[s] = va + (vb - va) * f;
            touched[s] = true;
        }
    }
}

/// Per-instance morph weight state (lives in the `AnimMixer`).
#[derive(Clone, Debug, Default)]
pub struct MorphState {
    /// Weights the game set (`animSetMorphWeight`), by slot.
    pub explicit: Vec<f32>,
    /// Weights the clips gave at the last update, by slot.
    pub clip: Vec<f32>,
    /// `clamp(clip + explicit)`, what is drawn; rebuilt by `effective()`.
    pub effective: Vec<f32>,
}

impl MorphState {
    /// The weights to draw: the clips' plus the game's, clamped to [-1, 1].
    /// Empty when nothing has ever been set or animated.
    pub fn effective(&mut self) -> &[f32] {
        let n = self.clip.len().max(self.explicit.len());
        self.effective.clear();
        self.effective.resize(n, 0.0);
        for (i, w) in self.effective.iter_mut().enumerate() {
            let c = self.clip.get(i).copied().unwrap_or(0.0);
            let e = self.explicit.get(i).copied().unwrap_or(0.0);
            *w = (c + e).clamp(-1.0, 1.0);
        }
        &self.effective
    }
}

fn slot_count(clips: &[&AnimationData]) -> usize {
    clips
        .iter()
        .flat_map(|c| c.morph_tracks.iter())
        .flat_map(|t| t.slots.iter())
        .map(|&s| s as usize)
        .filter(|&s| s < crate::morph_targets::MAX_MORPH_SLOTS)
        .map(|s| s + 1)
        .max()
        .unwrap_or(0)
        .min(crate::morph_targets::MAX_MORPH_SLOTS)
}

fn sample_clip(clip: &AnimationData, t: f32, n: usize) -> (Vec<f32>, Vec<bool>) {
    let mut out = vec![0.0; n];
    let mut touched = vec![false; n];
    for track in &clip.morph_tracks {
        track.sample_into(t, &mut out, &mut touched);
    }
    (out, touched)
}

/// Sample the mixer's clips' weight tracks into `mixer.morph.clip`, blended
/// the way `advance_and_update` blends their joints. Cheap when no clip has
/// weight tracks (the usual case): one scan, nothing allocated.
pub fn evaluate_mixer(animations: &[AnimationData], mixer: &mut crate::models::AnimMixer) {
    if !animations.iter().any(|a| !a.morph_tracks.is_empty()) {
        mixer.morph.clip.clear();
        return;
    }
    let Some(cur) = animations.get(mixer.cur_clip) else {
        return;
    };
    let mut used: Vec<&AnimationData> = vec![cur];
    let prev = (mixer.fade_dur > 0.0)
        .then(|| animations.get(mixer.prev_clip))
        .flatten();
    used.extend(prev);
    let partners: Vec<(&AnimationData, f32)> = mixer
        .rig
        .blends
        .iter()
        .filter(|b| b.clip >= 0 && b.weight > 0.0)
        .filter_map(|b| animations.get(b.clip as usize).map(|c| (c, b.weight)))
        .collect();
    used.extend(partners.iter().map(|(c, _)| *c));
    let layer = (mixer.layer_clip >= 0 && mixer.layer_weight > 0.0)
        .then(|| animations.get(mixer.layer_clip as usize))
        .flatten();
    used.extend(layer);
    let n = slot_count(&used);

    let (mut out, _) = sample_clip(cur, mixer.cur_time, n);
    if let Some(prev) = prev {
        let w = (mixer.fade_t / mixer.fade_dur).clamp(0.0, 1.0);
        let w = w * w * (3.0 - 2.0 * w);
        let (p, _) = sample_clip(prev, mixer.prev_time, n);
        for (o, p) in out.iter_mut().zip(p) {
            *o += (p - *o) * (1.0 - w);
        }
    }
    if !partners.is_empty() {
        let cur_dur = cur.duration;
        let phase = if cur_dur > 0.0 {
            mixer.cur_time / cur_dur
        } else {
            0.0
        };
        let mut total = 1.0f32;
        for b in mixer.rig.blends.iter() {
            total -= b.weight.max(0.0) * ((b.clip >= 0) as i32 as f32);
        }
        let mut total = total.max(0.0);
        for (clip, weight) in &partners {
            let (p, _) = sample_clip(clip, phase * clip.duration, n);
            total += weight;
            let f = weight / total;
            for (o, p) in out.iter_mut().zip(p) {
                *o += (p - *o) * f;
            }
        }
    }
    if let Some(layer) = layer {
        let (l, touched) = sample_clip(layer, mixer.layer_time, n);
        let w = mixer.layer_weight;
        for i in 0..n {
            if touched[i] {
                out[i] += (l[i] - out[i]) * w;
            }
        }
    }
    mixer.morph.clip = out;
}

/// `evaluate_mixer` for the single-clip `updateModelAnimation`.
pub fn evaluate_single(clip: &AnimationData, time: f32, mixer: &mut crate::models::AnimMixer) {
    if clip.morph_tracks.is_empty() {
        mixer.morph.clip.clear();
        return;
    }
    let n = slot_count(&[clip]);
    mixer.morph.clip = sample_clip(clip, time, n).0;
}

/// After an animation update has staged the instance's pose: stage its
/// morph weights with it (`Renderer::stage_morph_weights`). Every FFI that
/// poses an instance calls this right after `set_joint_matrices_*`.
pub fn stage_weights(
    models: &mut ModelManager,
    renderer: &mut crate::renderer::Renderer,
    anim: f64,
) {
    if let Some(a) = models.animations.get_mut(anim) {
        renderer.stage_morph_weights(anim.to_bits(), a.mixer.morph.effective());
    }
}

/// The morph API over model and animation handles.
impl ModelManager {
    /// The model's slot names (`None` for a model without morph targets).
    pub fn model_morph_names(&self, model: f64) -> Option<std::sync::Arc<Vec<String>>> {
        self.get(model)?
            .meshes
            .iter()
            .find_map(|m| m.morph.as_ref().map(|t| t.names.clone()))
    }

    /// Number of morph slots of a model (0 without morph targets).
    pub fn model_morph_count(&self, model: f64) -> usize {
        self.model_morph_names(model).map_or(0, |n| n.len())
    }

    /// Slot of the target called `name`, or -1. A load-time call: it
    /// compares strings.
    pub fn model_morph_index(&self, model: f64, name: &str) -> i32 {
        self.model_morph_names(model)
            .and_then(|names| names.iter().position(|n| n == name))
            .map_or(-1, |i| i as i32)
    }

    /// Set the game's weight of one slot (added to the clips' at draw).
    /// Out-of-range slots and non-finite weights are ignored.
    pub fn anim_set_morph_weight(&mut self, anim: f64, slot: usize, weight: f32) {
        self.anim_set_morph_weights(anim, slot, &[weight]);
    }

    /// Set consecutive slots from `first`.
    pub fn anim_set_morph_weights(&mut self, anim: f64, first: usize, weights: &[f32]) {
        let Some(a) = self.animations.get_mut(anim) else {
            return;
        };
        let explicit = &mut a.mixer.morph.explicit;
        for (i, &w) in weights.iter().enumerate() {
            let slot = first + i;
            if slot >= crate::morph_targets::MAX_MORPH_SLOTS || !w.is_finite() {
                continue;
            }
            if explicit.len() <= slot {
                if w == 0.0 {
                    continue;
                }
                explicit.resize(slot + 1, 0.0);
            }
            explicit[slot] = w.clamp(-1.0, 1.0);
        }
    }

    /// Drop every weight the game set (the clips' stay).
    pub fn anim_clear_morphs(&mut self, anim: f64) {
        if let Some(a) = self.animations.get_mut(anim) {
            a.mixer.morph.explicit.clear();
        }
    }

    /// The weight a slot is drawn with: clips plus the game's, clamped, as of
    /// the last update (and the game's weights set since).
    pub fn anim_morph_weight(&mut self, anim: f64, slot: usize) -> f32 {
        let Some(a) = self.animations.get_mut(anim) else {
            return 0.0;
        };
        a.mixer.morph.effective().get(slot).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AnimationData, ModelAnimation};
    use std::sync::Arc;

    fn clip(name: &str, track: MorphTrack, duration: f32) -> AnimationData {
        AnimationData {
            channels: vec![],
            duration,
            name: name.into(),
            morph_tracks: vec![track],
        }
    }

    fn ramp(slots: Vec<u32>, from: f32, to: f32) -> MorphTrack {
        let n = slots.len();
        let mut values = vec![from; n];
        values.extend(std::iter::repeat_n(to, n));
        MorphTrack {
            slots,
            times: vec![0.0, 1.0],
            values,
            step: false,
        }
    }

    fn manager(clips: Vec<AnimationData>) -> (ModelManager, f64) {
        let mut m = ModelManager::new();
        let skel = crate::models::SkeletonData {
            joints: vec![crate::models::JointData {
                inverse_bind: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                children: vec![],
                name: "root".into(),
                rest_translation: [0.0; 3],
                rest_rotation: [0.0, 0.0, 0.0, 1.0],
                rest_scale: [1.0; 3],
            }],
            root_joints: vec![0],
        };
        let h = m.animations.alloc(ModelAnimation {
            skeleton: Some(Arc::new(skel)),
            animations: Arc::new(clips),
            ref_rest_rotations: None,
            joint_matrices: vec![],
            mixer: Default::default(),
            joint_world: vec![],
            mask_weights: vec![],
            mask_cached_root: -1,
        });
        (m, h)
    }

    #[test]
    fn linear_and_step_tracks_sample_between_keys() {
        let lin = ramp(vec![2], 0.0, 1.0);
        let mut out = vec![0.0; 3];
        let mut touched = vec![false; 3];
        lin.sample_into(0.25, &mut out, &mut touched);
        assert!((out[2] - 0.25).abs() < 1e-6);
        assert_eq!(touched, vec![false, false, true]);
        let mut step = ramp(vec![0], 0.0, 1.0);
        step.step = true;
        step.sample_into(0.99, &mut out, &mut touched);
        assert_eq!(out[0], 0.0);
        step.sample_into(1.5, &mut out, &mut touched);
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn clip_weights_follow_the_mixer_and_explicit_weights_add() {
        let (mut m, h) = manager(vec![
            clip("open", ramp(vec![0, 1], 0.0, 1.0), 1.0),
            clip("shut", ramp(vec![0], 0.0, 0.0), 1.0),
        ]);
        m.anim_play(h, 0, 0.0, 1.0, false);
        m.advance_and_update(h, 0.5);
        assert!((m.anim_morph_weight(h, 0) - 0.5).abs() < 1e-5);
        assert!((m.anim_morph_weight(h, 1) - 0.5).abs() < 1e-5);
        // the game's weight adds and the sum is clamped
        m.anim_set_morph_weight(h, 1, 0.75);
        assert!((m.anim_morph_weight(h, 1) - 1.0).abs() < 1e-6);
        m.anim_set_morph_weights(h, 3, &[0.25, -2.0]);
        assert!((m.anim_morph_weight(h, 3) - 0.25).abs() < 1e-6);
        assert!((m.anim_morph_weight(h, 4) + 1.0).abs() < 1e-6);
        m.anim_clear_morphs(h);
        assert!((m.anim_morph_weight(h, 1) - 0.5).abs() < 1e-5);
        assert_eq!(m.anim_morph_weight(h, 3), 0.0);
        // a crossfade blends the weights like the joints (smoothstepped)
        m.anim_play(h, 1, 1.0, 1.0, false);
        m.advance_and_update(h, 0.5);
        // prev clip at t=1.0 (weight 1), new clip 0; fade w=0.5 -> 0.5
        assert!((m.anim_morph_weight(h, 0) - 0.5).abs() < 1e-4);
        // slot 1 is animated only by the clip fading out: it fades too
        assert!((m.anim_morph_weight(h, 1) - 0.5).abs() < 1e-4);
        m.advance_and_update(h, 0.6);
        assert!(m.anim_morph_weight(h, 0).abs() < 1e-6);
    }

    #[test]
    fn a_layer_overrides_only_the_slots_it_animates() {
        let (mut m, h) = manager(vec![
            clip("base", ramp(vec![0, 1], 0.5, 0.5), 1.0),
            clip("blink", ramp(vec![1], 1.0, 1.0), 1.0),
        ]);
        m.anim_play(h, 0, 0.0, 1.0, true);
        m.anim_set_layer(h, 1, 0.5, 0, 1.0, true);
        m.advance_and_update(h, 0.1);
        assert!((m.anim_morph_weight(h, 0) - 0.5).abs() < 1e-5);
        assert!((m.anim_morph_weight(h, 1) - 0.75).abs() < 1e-5);
    }

    #[test]
    fn no_tracks_and_bad_input_cost_nothing() {
        let (mut m, h) = manager(vec![AnimationData {
            channels: vec![],
            duration: 1.0,
            name: "idle".into(),
            morph_tracks: vec![],
        }]);
        m.advance_and_update(h, 0.1);
        assert!(m.get_animation(h).unwrap().mixer.morph.clip.is_empty());
        m.anim_set_morph_weight(h, 5, f32::NAN);
        m.anim_set_morph_weight(h, 5, 0.0);
        m.anim_set_morph_weight(h, crate::morph_targets::MAX_MORPH_SLOTS, 1.0);
        assert!(m.get_animation(h).unwrap().mixer.morph.explicit.is_empty());
        assert_eq!(m.anim_morph_weight(h, 0), 0.0);
        assert_eq!(m.anim_morph_weight(-1.0, 0), 0.0);
        m.anim_set_morph_weight(-1.0, 0, 1.0);
        m.anim_clear_morphs(-1.0);
        assert_eq!(m.model_morph_count(-1.0), 0);
        assert_eq!(m.model_morph_index(-1.0, "jawOpen"), -1);
    }
}
