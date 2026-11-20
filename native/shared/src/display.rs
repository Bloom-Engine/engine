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

/// Gravity in the device's own axes, in g (iOS accelerometer; zero where there is none).
/// x: toward the right edge of a portrait screen, y: toward its top, z: out of the screen.
static GRAVITY: [std::sync::atomic::AtomicU32; 3] = [
    std::sync::atomic::AtomicU32::new(0), std::sync::atomic::AtomicU32::new(0), std::sync::atomic::AtomicU32::new(0),
];

pub fn set_device_gravity(x: f32, y: f32, z: f32) {
    GRAVITY[0].store(x.to_bits(), Ordering::Relaxed);
    GRAVITY[1].store(y.to_bits(), Ordering::Relaxed);
    GRAVITY[2].store(z.to_bits(), Ordering::Relaxed);
}

pub fn device_gravity(axis: usize) -> f32 {
    if axis > 2 { return 0.0; }
    f32::from_bits(GRAVITY[axis].load(Ordering::Relaxed))
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

// ---- monitors ---------------------------------------------------------------
//
// Which displays there are, which one the game is on, and moving it. Only a
// platform's windowing code can answer, so a platform crate that can registers
// a provider once at start-up; everywhere else the engine reports the one
// display it is drawing to. One shared FFI surface either way, so a game's
// "monitor" settings row needs no per-platform code.

/// One display, in physical pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct MonitorInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Refresh rate in Hz; 0 when the platform does not say.
    pub refresh_hz: f32,
}

/// A platform's answers. Plain function pointers: the providers read window
/// state that lives in the platform crate's own statics.
#[derive(Clone, Copy)]
pub struct MonitorProvider {
    pub list: fn() -> Vec<MonitorInfo>,
    /// Index into `list()` of the display the window is on.
    pub current: fn() -> usize,
    /// Ask for the window to move to that display. May complete over several
    /// frames (leaving and re-entering fullscreen). False if out of range.
    pub set: fn(usize) -> bool,
}

static MONITORS: std::sync::OnceLock<MonitorProvider> = std::sync::OnceLock::new();

/// Called once by a platform crate that can enumerate displays.
pub fn set_monitor_provider(provider: MonitorProvider) {
    let _ = MONITORS.set(provider);
}

/// The displays; `fallback` is the size the engine is drawing at, reported as
/// the single display when the platform registered nothing.
pub fn monitors(fallback: (u32, u32)) -> Vec<MonitorInfo> {
    let listed = MONITORS.get().map(|p| (p.list)()).unwrap_or_default();
    if listed.is_empty() {
        vec![MonitorInfo {
            name: String::from("Display"),
            width: fallback.0,
            height: fallback.1,
            refresh_hz: 0.0,
        }]
    } else {
        listed
    }
}

pub fn current_monitor() -> usize {
    MONITORS.get().map_or(0, |p| (p.current)())
}

pub fn set_monitor(index: usize) -> bool {
    match MONITORS.get() {
        Some(p) => (p.set)(index),
        // One display: asking for it is a success, anything else is not.
        None => index == 0,
    }
}
