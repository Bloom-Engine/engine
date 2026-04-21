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

/// Register in-memory image bytes (PNG or JPEG) by writing them to a temp
/// file inside the app's sandbox and returning a handle. Used by the glTF
/// loader for embedded textures. Returns 0 on failure.
pub fn register_bytes(bytes: &[u8]) -> u32 {
    if bytes.len() < 8 { return 0; }
    let (ext, (w, h)) = detect_image(bytes);
    if ext.is_empty() { return 0; }

    // Write to a stable path inside the temp dir. We include a counter so
    // distinct blobs don't collide.
    let mut reg = REG.lock().unwrap();
    let idx = reg.entries.len();
    let dir = std::env::temp_dir().join("bloom_watchos_tex");
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join(format!("tex_{}.{}", idx, ext));
    if std::fs::write(&file, bytes).is_err() { return 0; }

    let path_str = file.to_string_lossy().to_string();
    let Ok(cpath) = CString::new(path_str) else { return 0; };
    reg.entries.push(TexEntry { path: cpath, width: w, height: h });
    reg.entries.len() as u32
}

/// Detect PNG or JPEG by magic bytes, returning extension + (width, height).
/// Both formats are common in glTF .glb embeds.
fn detect_image(b: &[u8]) -> (&'static str, (u32, u32)) {
    // PNG signature: 89 50 4E 47 0D 0A 1A 0A
    if b.len() >= 24 && b[0] == 0x89 && &b[1..4] == b"PNG" && &b[12..16] == b"IHDR" {
        let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
        let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
        return ("png", (w, h));
    }
    // JPEG signature: FF D8 ... look for SOF0 / SOF2 marker for dimensions.
    if b.len() >= 4 && b[0] == 0xFF && b[1] == 0xD8 {
        if let Some(dims) = parse_jpeg_size(b) {
            return ("jpg", dims);
        }
        return ("jpg", (0, 0));
    }
    ("", (0, 0))
}

/// Walk JPEG segments until a Start-Of-Frame marker, read (h, w).
fn parse_jpeg_size(b: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2;
    while i + 9 < b.len() {
        if b[i] != 0xFF { return None; }
        let marker = b[i + 1];
        i += 2;
        // Standalone markers (no length field)
        if marker == 0xD8 || marker == 0xD9 { continue; }
        if i + 2 > b.len() { return None; }
        let seg_len = u16::from_be_bytes([b[i], b[i+1]]) as usize;
        // SOF0 (0xC0), SOF1 (0xC1), SOF2 (0xC2) — baseline + progressive.
        if marker == 0xC0 || marker == 0xC1 || marker == 0xC2 {
            if i + 7 > b.len() { return None; }
            let h = u16::from_be_bytes([b[i+3], b[i+4]]) as u32;
            let w = u16::from_be_bytes([b[i+5], b[i+6]]) as u32;
            return Some((w, h));
        }
        i += seg_len;
    }
    None
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
