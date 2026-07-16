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
}

impl Default for AnimMixer {
    fn default() -> Self {
        Self {
            cur_clip: 0, cur_time: 0.0, cur_speed: 1.0, cur_loop: true,
            prev_clip: 0, prev_time: 0.0, prev_speed: 1.0, prev_loop: true,
            fade_t: 0.0, fade_dur: 0.0,
            layer_clip: -1, layer_time: 0.0, layer_speed: 1.0, layer_loop: false,
            layer_weight: 0.0, layer_mask_root: -1,
            root_motion: false, root_delta: [0.0; 3],
            finished: false, started: false,
        }
    }
}
