//! Thread-safe draw-command list the Swift Canvas reads each frame.
//!
//! Game thread (spawned by the Swift shell calling `_perry_user_main`) pushes
//! commands via `push()` between `begin()` / `end()`. At `end()`, the frame
//! counter increments. Swift side calls `snapshot()` to atomically copy the
//! current commands into a static buffer it can then iterate via `cmd_at()`
//! without holding any lock. The snapshot is valid until the next `snapshot()`
//! call — Swift drains it synchronously inside a single Canvas closure.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Inline text capacity — long enough for HUD / menu strings; games needing
/// longer text should split across commands.
pub const TEXT_CAP: usize = 256;

/// Command kinds. Kept stable across Swift + Rust — don't renumber without
/// updating BloomWatchApp.swift.
#[allow(dead_code)]
pub mod kind {
    pub const CLEAR: i32 = 0;
    pub const RECT: i32 = 1;
    pub const RECT_LINES: i32 = 2;
    pub const CIRCLE: i32 = 3;
    pub const CIRCLE_LINES: i32 = 4;
    pub const LINE: i32 = 5;
    pub const TRIANGLE: i32 = 6;
    pub const TEXTURE: i32 = 7;      // whole texture at (x,y)
    pub const TEXTURE_REC: i32 = 8;  // source rect → dest rect, no rotation
    pub const TEXTURE_PRO: i32 = 9;  // full: source, dest, origin, rotation
    pub const TEXT: i32 = 10;

    // 3D immediate-mode primitives (pos = x,y,z; w,h = scale; src_x,y,z = secondary).
    pub const CUBE: i32 = 20;         // pos (x,y,z), size (w,h,size=depth)
    pub const CUBE_WIRES: i32 = 21;
    pub const SPHERE: i32 = 22;       // pos (x,y,z), radius (w)
    pub const SPHERE_WIRES: i32 = 23;
    pub const CYLINDER: i32 = 24;     // pos (x,y,z), radius (w), height (h)
    pub const PLANE: i32 = 25;        // pos (x,y,z), size (w,h)
    pub const GRID: i32 = 26;         // slices (w), spacing (h)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DrawCmd {
    pub kind: i32,
    pub _pad0: i32,
    pub tex: u32,
    pub _pad1: u32,

    // Destination geometry.
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,

    // Source sub-rectangle (textures) or secondary point (lines).
    pub src_x: f64,
    pub src_y: f64,
    pub src_w: f64,
    pub src_h: f64,

    // Origin / pivot for rotation + scale (relative to dest).
    pub ox: f64,
    pub oy: f64,

    // Color, 0-255 per channel.
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,

    // rotation (degrees), size (font), thickness (lines), pad.
    pub rot: f64,
    pub size: f64,
    pub thickness: f64,
    pub _pad2: f64,

    // Inline UTF-8 for text commands; text_len == 0 otherwise.
    pub text: [u8; TEXT_CAP],
    pub text_len: u64,
}

impl DrawCmd {
    pub const fn zero() -> Self {
        Self {
            kind: 0,
            _pad0: 0,
            tex: 0,
            _pad1: 0,
            x: 0.0, y: 0.0, w: 0.0, h: 0.0,
            src_x: 0.0, src_y: 0.0, src_w: 0.0, src_h: 0.0,
            ox: 0.0, oy: 0.0,
            r: 255.0, g: 255.0, b: 255.0, a: 255.0,
            rot: 0.0, size: 0.0, thickness: 1.0, _pad2: 0.0,
            text: [0; TEXT_CAP],
            text_len: 0,
        }
    }

    pub fn set_text(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let n = bytes.len().min(TEXT_CAP);
        self.text[..n].copy_from_slice(&bytes[..n]);
        self.text_len = n as u64;
    }
}

struct Inner {
    // The "building" list — game thread pushes into this between begin/end.
    building: Vec<DrawCmd>,
    // The "ready" list — snapshot target, what Swift reads.
    ready: Vec<DrawCmd>,
}

static INNER: Mutex<Inner> = Mutex::new(Inner {
    building: Vec::new(),
    ready: Vec::new(),
});

/// Monotonic frame counter — Swift polls this and invalidates Canvas when it
/// changes. Game thread bumps it on `end()`.
static FRAME: AtomicU64 = AtomicU64::new(0);

/// Clear color packed as f64 bits for the current frame (r,g,b,a).
static CLEAR: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(255f64.to_bits()),
];

/// 3D camera state — pos (xyz), target (xyz), up (xyz), fovy, proj (0=ortho, 1=persp).
/// Written by bloom_begin_mode_3d, read once per frame by the Swift SceneView.
static CAMERA: [AtomicU64; 11] = {
    const INIT: AtomicU64 = AtomicU64::new(0);
    [INIT; 11]
};
/// Non-zero when a 3D section has ever been opened — tells the Swift shell
/// it needs to host a SceneView, not just a Canvas.
static HAS_3D: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn begin() {
    let mut g = INNER.lock().unwrap();
    g.building.clear();
}

pub fn end() {
    {
        let mut g = INNER.lock().unwrap();
        // Move building → ready via take; clear the building list for the
        // next frame. Can't use mem::swap on two fields of the same borrow.
        let done = std::mem::take(&mut g.building);
        g.ready = done;
    }
    FRAME.fetch_add(1, Ordering::Release);
}

pub fn push(cmd: DrawCmd) {
    let mut g = INNER.lock().unwrap();
    g.building.push(cmd);
}

pub fn set_clear(r: f64, g: f64, b: f64, a: f64) {
    CLEAR[0].store(r.to_bits(), Ordering::Relaxed);
    CLEAR[1].store(g.to_bits(), Ordering::Relaxed);
    CLEAR[2].store(b.to_bits(), Ordering::Relaxed);
    CLEAR[3].store(a.to_bits(), Ordering::Relaxed);
}

pub fn clear_rgba() -> [f64; 4] {
    [
        f64::from_bits(CLEAR[0].load(Ordering::Relaxed)),
        f64::from_bits(CLEAR[1].load(Ordering::Relaxed)),
        f64::from_bits(CLEAR[2].load(Ordering::Relaxed)),
        f64::from_bits(CLEAR[3].load(Ordering::Relaxed)),
    ]
}

pub fn frame() -> u64 {
    FRAME.load(Ordering::Acquire)
}

pub fn set_camera(px: f64, py: f64, pz: f64,
                  tx: f64, ty: f64, tz: f64,
                  ux: f64, uy: f64, uz: f64,
                  fovy: f64, proj: f64) {
    let vals = [px, py, pz, tx, ty, tz, ux, uy, uz, fovy, proj];
    for (i, v) in vals.iter().enumerate() {
        CAMERA[i].store(v.to_bits(), Ordering::Relaxed);
    }
    HAS_3D.store(true, std::sync::atomic::Ordering::Release);
}

pub fn camera_snapshot(out: *mut f64) {
    if out.is_null() { return; }
    unsafe {
        for i in 0..11 {
            *out.add(i) = f64::from_bits(CAMERA[i].load(Ordering::Relaxed));
        }
    }
}

pub fn has_3d() -> bool {
    HAS_3D.load(std::sync::atomic::Ordering::Acquire)
}

/// Bulk-copy the ready list into a Swift-supplied buffer. Returns the number
/// of commands actually copied (never more than `max`). Swift allocates a
/// single buffer once and reuses it each frame.
pub fn copy_into(dst: *mut DrawCmd, max: i64) -> i64 {
    if dst.is_null() || max <= 0 {
        return 0;
    }
    let g = INNER.lock().unwrap();
    let n = g.ready.len().min(max as usize);
    // SAFETY: caller guarantees `dst` points to at least `max` DrawCmds of
    // writable memory with the same #[repr(C)] layout.
    unsafe {
        std::ptr::copy_nonoverlapping(g.ready.as_ptr(), dst, n);
    }
    n as i64
}
