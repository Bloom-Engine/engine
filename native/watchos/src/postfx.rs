//! Post-processing effect state. Atomic floats the Swift root view polls
//! each frame and translates into SwiftUI view modifiers.
//!
//! What maps cleanly onto built-in SwiftUI modifiers:
//!   - vignette → `.overlay(RadialGradient)` (strength + softness)
//!   - manual exposure → `.brightness(ev - 1.0)` approximation
//!   - auto exposure → identity (SceneKit's own tone mapping handles most of it)
//!
//! What would need a Metal shader via `.colorEffect(shader:)` (watchOS 10+)
//! and a Perry-side .metal compilation step — deferred:
//!   - chromatic aberration (per-channel position offset)
//!   - film grain (time-animated noise)
//!   - sun shafts (radial blur + light scattering)
//!
//! The corresponding bloom_set_* calls still store their values so a future
//! shader pipeline can pick them up without breaking TS code today.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(true);
static VIGNETTE_STRENGTH: AtomicU64 = AtomicU64::new(0);
static VIGNETTE_SOFTNESS: AtomicU64 = AtomicU64::new(0);
static CA_STRENGTH: AtomicU64 = AtomicU64::new(0);
static FILM_GRAIN: AtomicU64 = AtomicU64::new(0);
static EXPOSURE: AtomicU64 = AtomicU64::new(1.0f64.to_bits());
static AUTO_EXPOSURE: AtomicBool = AtomicBool::new(false);
static SUN_STRENGTH: AtomicU64 = AtomicU64::new(0);
static SUN_DECAY: AtomicU64 = AtomicU64::new(0.85f64.to_bits());
static SUN_R: AtomicU64 = AtomicU64::new(1.0f64.to_bits());
static SUN_G: AtomicU64 = AtomicU64::new(0.9f64.to_bits());
static SUN_B: AtomicU64 = AtomicU64::new(0.7f64.to_bits());

fn store(a: &AtomicU64, v: f64) { a.store(v.to_bits(), Ordering::Relaxed); }
fn load(a: &AtomicU64) -> f64 { f64::from_bits(a.load(Ordering::Relaxed)) }

pub fn set_enabled(on: bool) { ENABLED.store(on, Ordering::Relaxed); }
pub fn set_vignette(strength: f64, softness: f64) {
    store(&VIGNETTE_STRENGTH, strength);
    store(&VIGNETTE_SOFTNESS, softness);
}
pub fn set_chromatic_aberration(s: f64) { store(&CA_STRENGTH, s); }
pub fn set_film_grain(s: f64) { store(&FILM_GRAIN, s); }
pub fn set_exposure(v: f64) {
    store(&EXPOSURE, v);
    AUTO_EXPOSURE.store(false, Ordering::Relaxed);
}
pub fn set_auto_exposure(on: bool) { AUTO_EXPOSURE.store(on, Ordering::Relaxed); }

pub fn set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) {
    store(&SUN_STRENGTH, strength);
    store(&SUN_DECAY, decay);
    store(&SUN_R, r);
    store(&SUN_G, g);
    store(&SUN_B, b);
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PostFxState {
    pub enabled: u32,
    pub auto_exposure: u32,
    pub vignette_strength: f32,
    pub vignette_softness: f32,
    pub chromatic_aberration: f32,
    pub film_grain: f32,
    pub exposure: f32,
    pub sun_strength: f32,
    pub sun_decay: f32,
    pub sun_r: f32,
    pub sun_g: f32,
    pub sun_b: f32,
}

pub fn snapshot(out: *mut PostFxState) {
    if out.is_null() { return; }
    let s = PostFxState {
        enabled: if ENABLED.load(Ordering::Relaxed) { 1 } else { 0 },
        auto_exposure: if AUTO_EXPOSURE.load(Ordering::Relaxed) { 1 } else { 0 },
        vignette_strength: load(&VIGNETTE_STRENGTH) as f32,
        vignette_softness: load(&VIGNETTE_SOFTNESS) as f32,
        chromatic_aberration: load(&CA_STRENGTH) as f32,
        film_grain: load(&FILM_GRAIN) as f32,
        exposure: load(&EXPOSURE) as f32,
        sun_strength: load(&SUN_STRENGTH) as f32,
        sun_decay: load(&SUN_DECAY) as f32,
        sun_r: load(&SUN_R) as f32,
        sun_g: load(&SUN_G) as f32,
        sun_b: load(&SUN_B) as f32,
    };
    unsafe { *out = s; }
}
