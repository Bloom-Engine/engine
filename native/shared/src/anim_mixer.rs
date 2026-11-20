//! EN-028 — per-model animation mixer state.
//!
//! Split out of `models.rs` (EN-052 line budget) when EN-055 landed: the
//! mixer is exactly the state that is PER-INSTANCE under animation
//! instancing, so it earns its own module. Re-exported from `models`, so
//! `crate::models::AnimMixer` paths are unchanged.
//!
//! Three things a single-clip sampler cannot do, all of which read as
//! "cheap" on screen: transitions pop, an attacking character has to stop
//! walking, and authored locomotion arcs (a pounce, a lunge) get replaced
//! by hand-tuned kinematics because the root is nailed to the rest pose.
//!
//! Layout: a base track that crossfades from `prev` to `cur` over
//! `fade_dur`, plus one optional additive-by-mask layer (an attack driving
//! the spine-up while the legs keep walking). All clocks advance in
//! `advance_animation`, so the game hands over a dt and never tracks clip
//! time itself.

#[derive(Clone)]
pub struct AnimMixer {
    pub cur_clip: usize,
    pub cur_time: f32,
    pub cur_speed: f32,
    pub cur_loop: bool,
    /// Clip we are fading *out* of. `fade_dur <= 0` means no fade in flight.
    pub prev_clip: usize,
    pub prev_time: f32,
    pub prev_speed: f32,
    pub prev_loop: bool,
    pub fade_t: f32,
    pub fade_dur: f32,
    /// Masked layer. `layer_clip < 0` = inactive.
    pub layer_clip: i32,
    pub layer_time: f32,
    pub layer_speed: f32,
    pub layer_loop: bool,
    pub layer_weight: f32,
    /// Root joint of the masked subtree (e.g. the spine). Every joint at or
    /// below it takes the layer pose; everything else keeps the base pose.
    pub layer_mask_root: i32,
    /// Opt-in root motion. Off by default so existing games are unchanged.
    pub root_motion: bool,
    pub root_delta: [f32; 3],
    /// True once a non-looping `cur_clip` has played past its duration.
    pub finished: bool,
    pub started: bool,
    /// EN-064 — blend partners, joint rotations and IK applied over the
    /// sampled pose. Empty by default, which costs nothing.
    pub rig: crate::anim_rig::AnimRig,
}

impl Default for AnimMixer {
    fn default() -> Self {
        Self {
            cur_clip: 0,
            cur_time: 0.0,
            cur_speed: 1.0,
            cur_loop: true,
            prev_clip: 0,
            prev_time: 0.0,
            prev_speed: 1.0,
            prev_loop: true,
            fade_t: 0.0,
            fade_dur: 0.0,
            layer_clip: -1,
            layer_time: 0.0,
            layer_speed: 1.0,
            layer_loop: false,
            layer_weight: 0.0,
            layer_mask_root: -1,
            root_motion: false,
            root_delta: [0.0; 3],
            finished: false,
            started: false,
            rig: crate::anim_rig::AnimRig::default(),
        }
    }
}

/// Where a clock lands when it is set to `t` on a clip of length `dur`:
/// wrapped on a loop, clamped on a one-shot. Also says whether a one-shot is
/// then at its end.
fn place(t: f32, dur: f32, looping: bool) -> (f32, bool) {
    if dur <= 0.0 {
        (0.0, false)
    } else if looping {
        (t.rem_euclid(dur), false)
    } else {
        let t = t.clamp(0.0, dur);
        (t, t >= dur)
    }
}

impl AnimMixer {
    /// Set a clock: `which` 0 the base clip's time in seconds, 1 the masked
    /// layer's, 2 the base clip's time as a 0..1 phase (the same selector
    /// as `anim_time`). `base_dur` / `layer_dur` are the clips' lengths.
    /// A fade in flight carries on; root motion does not count the jump
    /// (the next delta is measured from the new time).
    pub fn seek(&mut self, which: u32, t: f32, base_dur: f32, layer_dur: f32) {
        if !t.is_finite() {
            return;
        }
        match which {
            0 | 2 => {
                let secs = if which == 2 { t * base_dur } else { t };
                let (at, end) = place(secs, base_dur, self.cur_loop);
                self.cur_time = at;
                self.finished = end;
            }
            1 => {
                if self.layer_clip >= 0 {
                    self.layer_time = place(t, layer_dur, self.layer_loop).0;
                }
            }
            _ => {}
        }
    }

    /// Play the base clip again from its start, crossfading over `fade`
    /// seconds from the pose it is in now (0 = cut). The one thing
    /// `anim_play` will not do for the clip already playing: there it is a
    /// no-op by contract, so it can be called every frame.
    pub fn restart(&mut self, fade: f32) {
        if self.started && fade > 0.0 {
            self.prev_clip = self.cur_clip;
            self.prev_time = self.cur_time;
            self.prev_speed = self.cur_speed;
            self.prev_loop = self.cur_loop;
            self.fade_dur = fade;
            self.fade_t = 0.0;
        }
        self.cur_time = 0.0;
        self.finished = false;
    }
}

#[cfg(test)]
mod tests {
    use crate::models::{
        AnimationChannel, AnimationData, JointData, ModelAnimation, ModelManager, SkeletonData,
    };
    use std::sync::Arc;

    const ID4: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn joint(children: Vec<usize>) -> JointData {
        JointData {
            inverse_bind: ID4,
            children,
            name: String::new(),
            rest_translation: [0.0; 3],
            rest_rotation: [0.0, 0.0, 0.0, 1.0],
            rest_scale: [1.0; 3],
        }
    }

    /// `joint` slides 0 → 2 m along x over a 2 s clip.
    fn slide(joint: usize) -> AnimationData {
        AnimationData {
            channels: vec![AnimationChannel {
                joint_index: joint,
                timestamps: vec![0.0, 2.0],
                translations: vec![[0.0; 3], [2.0, 0.0, 0.0]],
                rotation_timestamps: vec![],
                rotations: vec![],
                scale_timestamps: vec![],
                scales: vec![],
            }],
            duration: 2.0,
            name: "slide".into(),
        }
    }

    /// A root with one child; clip 0 slides the child, clip 1 the root.
    fn rig() -> (ModelManager, f64) {
        let mut m = ModelManager::new();
        let h = m.animations.alloc(ModelAnimation {
            skeleton: Some(Arc::new(SkeletonData {
                joints: vec![joint(vec![1]), joint(vec![])],
                root_joints: vec![0],
            })),
            animations: Arc::new(vec![slide(1), slide(0)]),
            ref_rest_rotations: None,
            joint_matrices: vec![],
            mixer: Default::default(),
            joint_world: vec![],
            mask_weights: vec![],
            mask_cached_root: -1,
        });
        (m, h)
    }

    fn child_x(m: &ModelManager, h: f64) -> f32 {
        m.joint_world(h, 1).unwrap()[3][0]
    }

    #[test]
    fn restart_plays_the_clip_already_playing_again() {
        let (mut m, h) = rig();
        m.anim_play(h, 0, 0.0, 1.0, false);
        for _ in 0..30 {
            m.advance_and_update(h, 0.1);
        }
        assert!(m.anim_finished(h));
        // a one-shot that has finished holds its LAST frame
        assert!(
            (child_x(&m, h) - 2.0).abs() < 1e-4,
            "end pose {}",
            child_x(&m, h)
        );
        // animPlay of the same clip is a no-op by contract...
        m.anim_play(h, 0, 0.0, 1.0, false);
        m.advance_and_update(h, 0.1);
        assert!(m.anim_finished(h));
        // ...restart is the way to play it again
        m.anim_restart(h, 0.0);
        assert!(!m.anim_finished(h));
        m.advance_and_update(h, 0.5);
        assert!((m.anim_time(h, 0) - 0.5).abs() < 1e-5);
        assert!((child_x(&m, h) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn restart_with_a_fade_blends_out_of_the_old_pose() {
        let (mut m, h) = rig();
        m.anim_play(h, 0, 0.0, 1.0, true);
        m.advance_and_update(h, 1.5);
        m.anim_restart(h, 0.4);
        m.advance_and_update(h, 0.1);
        // halfway between prev (1.6) and the restarted clip (0.1), weighted
        // by the smoothstepped fade: strictly between, nearer the old pose
        let x = child_x(&m, h);
        assert!(x > 0.1 + 0.5 && x < 1.6, "x {x}");
        m.advance_and_update(h, 0.4); // fade done
        assert!((child_x(&m, h) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn seek_sets_base_layer_and_phase() {
        let (mut m, h) = rig();
        m.anim_play(h, 0, 0.0, 1.0, false);
        m.advance_and_update(h, 0.0);
        m.anim_seek(h, 0, 1.25);
        m.advance_and_update(h, 0.0);
        assert!((m.anim_time(h, 0) - 1.25).abs() < 1e-5);
        assert!((child_x(&m, h) - 1.25).abs() < 1e-4);
        m.anim_seek(h, 2, 0.25); // phase
        m.advance_and_update(h, 0.0);
        assert!((m.anim_time(h, 0) - 0.5).abs() < 1e-5);
        // a one-shot clamps; seeking to its end finishes it, back un-finishes
        m.anim_seek(h, 0, 9.0);
        assert!(m.anim_finished(h));
        assert!((m.anim_time(h, 0) - 2.0).abs() < 1e-5);
        m.anim_seek(h, 0, 0.3);
        assert!(!m.anim_finished(h));
        // a loop wraps
        m.anim_play(h, 1, 0.0, 1.0, true);
        m.anim_seek(h, 0, 4.5);
        assert!((m.anim_time(h, 0) - 0.5).abs() < 1e-5);
        // the layer clock, only with a layer on
        m.anim_seek(h, 1, 1.0);
        assert_eq!(m.anim_time(h, 1), 0.0);
        m.anim_set_layer(h, 0, 1.0, 1, 1.0, false);
        m.anim_seek(h, 1, 1.5);
        assert!((m.anim_time(h, 1) - 1.5).abs() < 1e-5);
        // nonsense is ignored
        m.anim_seek(h, 0, f32::NAN);
        assert!((m.anim_time(h, 0) - 0.5).abs() < 1e-5);
        m.anim_seek(-1.0, 0, 1.0);
        m.anim_restart(12345.0, 0.2);
    }

    #[test]
    fn seek_does_not_leak_into_root_motion() {
        let (mut m, h) = rig();
        m.anim_set_root_motion(h, true);
        m.anim_play(h, 1, 0.0, 1.0, true);
        m.advance_and_update(h, 0.2);
        m.anim_seek(h, 0, 1.8);
        m.advance_and_update(h, 0.1);
        let dx = m.anim_root_delta(h)[0];
        assert!((dx - 0.1).abs() < 1e-4, "root delta {dx} includes the jump");
    }
}
