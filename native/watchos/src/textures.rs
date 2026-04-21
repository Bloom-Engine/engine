//! Texture registry: paths from TS side → stable handles that the Swift
//! Canvas resolves to CGImages on first use and caches thereafter.
//!
//! Widths + heights are parsed out of the PNG IHDR here so `loadTexture`'s
//! synchronous `{ id, width, height }` contract works without round-tripping
//! to Swift. Only PNG is supported for this first cut — adequate for Jump
//! (all its sprites are PNG).

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Mutex;

struct TexEntry {
    /// Null-terminated path, kept alive for the program's lifetime — Swift
    /// holds the pointer across frames.
    path: CString,
    width: u32,
    height: u32,
}

struct Registry {
    entries: Vec<TexEntry>,
}

static REG: Mutex<Registry> = Mutex::new(Registry { entries: Vec::new() });

/// Load a texture — stores the path + parses PNG header for dimensions,
/// returns a 1-based handle. Returns 0 on failure.
pub fn load(path: &str) -> u32 {
    if path.is_empty() {
        return 0;
    }

    // Resolve bundle-relative paths against the main bundle's resource path.
    let full = resolve_bundle_path(path);

    let (w, h) = match parse_png_size(&full) {
        Some(v) => v,
        None => (0, 0), // unknown — Swift will skip/draw placeholder
    };

    let cpath = match CString::new(full) {
        Ok(c) => c,
        Err(_) => return 0,
    };

    let mut reg = REG.lock().unwrap();
    reg.entries.push(TexEntry { path: cpath, width: w, height: h });
    reg.entries.len() as u32
}

pub fn width(handle: u32) -> u32 {
    let reg = REG.lock().unwrap();
    if handle == 0 || handle as usize > reg.entries.len() {
        return 0;
    }
    reg.entries[handle as usize - 1].width
}

pub fn height(handle: u32) -> u32 {
    let reg = REG.lock().unwrap();
    if handle == 0 || handle as usize > reg.entries.len() {
        return 0;
    }
    reg.entries[handle as usize - 1].height
}

/// Get the null-terminated resolved path for a handle. Returns null for 0.
/// Pointer is stable for the program's lifetime.
pub fn path_ptr(handle: u32) -> *const c_char {
    let reg = REG.lock().unwrap();
    if handle == 0 || handle as usize > reg.entries.len() {
        return std::ptr::null();
    }
    reg.entries[handle as usize - 1].path.as_ptr()
}

/// Resolve a possibly-bundle-relative path to an absolute filesystem path.
/// On watchOS the Swift shell populates `BUNDLE_RESOURCE_PATH` on startup
/// via `bloom_watchos_set_bundle_path`; if it hasn't, paths are returned
/// unchanged.
pub fn resolve_bundle_path(p: &str) -> String {
    if p.starts_with('/') {
        return p.to_string();
    }
    let bundle = BUNDLE_PATH.lock().unwrap();
    if bundle.is_empty() {
        return p.to_string();
    }
    format!("{}/{}", bundle, p)
}

static BUNDLE_PATH: Mutex<String> = Mutex::new(String::new());

pub fn set_bundle_path(p: &str) {
    let mut b = BUNDLE_PATH.lock().unwrap();
    b.clear();
    b.push_str(p);
}

/// Read a PNG file's IHDR chunk and return (width, height). Returns None for
/// any parse failure (wrong magic, truncated, etc.).
fn parse_png_size(path: &str) -> Option<(u32, u32)> {
    let bytes = std::fs::read(path).ok()?;
    // PNG signature (8 bytes) + IHDR length (4) + "IHDR" (4) + width (4) + height (4) = 24 bytes min
    if bytes.len() < 24 {
        return None;
    }
    const PNG_SIG: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes[..8] != PNG_SIG {
        return None;
    }
    if &bytes[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Some((w, h))
}
