//! What the platform knows about the display beyond the window's size: its
//! scale (physical pixels per point) and the safe-area insets (the part of the
//! window the Dynamic Island, a notch, rounded corners, the home indicator or
//! a foldable's status-bar column cover).
//!
//! Platforms that have these values report them every frame (iOS from the
//! view's trait collection and `safeAreaInsets`, which follow rotation, split
//! view and an iPhone Duo opening or closing); everywhere else the scale is 1
//! and the insets are 0. The insets are in the units of `getScreenWidth()`
//! (pixels on iOS, points on macOS — see EN-024).
//!
//! Plain atomics, not EngineState: iOS writes them from the frame, the FFI
//! reads them from the game thread, and neither needs the engine to exist.
use std::sync::atomic::{AtomicU64, Ordering};

static SCALE: AtomicU64 = AtomicU64::new(0);
static INSETS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Edges, in the order `getSafeAreaInset(edge)` takes them.
pub const TOP: usize = 0;
pub const RIGHT: usize = 1;
pub const BOTTOM: usize = 2;
pub const LEFT: usize = 3;

pub fn set_display_scale(scale: f64) {
    if scale.is_finite() && scale > 0.0 {
        SCALE.store(scale.to_bits(), Ordering::Relaxed);
    }
}

/// Physical pixels per point; 1 where the platform doesn't report it.
pub fn display_scale() -> f64 {
    let v = f64::from_bits(SCALE.load(Ordering::Relaxed));
    if v > 0.0 {
        v
    } else {
        1.0
    }
}

/// tvOS: whether the Siri Remote's Menu button leaves the app (hands the press to UIKit, which
/// goes back to the Home screen) instead of reaching the game. Apple requires that at the app's
/// top level (its title screen); everywhere else Menu is the game's "back".
static MENU_EXITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_menu_exits(on: bool) {
    MENU_EXITS.store(on, Ordering::Relaxed);
}

pub fn menu_exits() -> bool {
    MENU_EXITS.load(Ordering::Relaxed)
}

pub fn set_safe_area_insets(top: f64, right: f64, bottom: f64, left: f64) {
    for (slot, v) in INSETS.iter().zip([top, right, bottom, left]) {
        slot.store(
            if v.is_finite() && v > 0.0 { v } else { 0.0 }.to_bits(),
            Ordering::Relaxed,
        );
    }
}

/// One edge's inset (TOP, RIGHT, BOTTOM, LEFT); 0 for an unknown edge.
pub fn safe_area_inset(edge: usize) -> f64 {
    INSETS
        .get(edge)
        .map_or(0.0, |v| f64::from_bits(v.load(Ordering::Relaxed)))
}
