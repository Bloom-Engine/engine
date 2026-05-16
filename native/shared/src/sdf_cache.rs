//! Disk cache for per-mesh signed distance fields baked by ticket 014.
//!
//! Each mesh content-hashes (positions + indices) to a 64-bit key; the
//! 32³ R32Float voxel data (~128 KB) lives under the platform cache
//! directory. Cold launches that hit the cache skip the GPU
//! brute-force point-triangle bake entirely and `queue.write_texture`
//! the bytes directly. Misses fall through to the existing in-process
//! bake; the renderer reads the texture back on the same frame and
//! writes the cache entry so the next launch hits.
//!
//! Caching is best-effort by design — every fallible operation
//! (hashing aside) returns `Option`/`Result` and the renderer treats
//! an error or `None` as "no cache, just bake."
//!
//! Web isn't supported here. The wasm32 build needs IndexedDB plumbing
//! before it can store anything; until then `cache_dir()` returns
//! `None` on wasm and the bake falls through normally.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

/// File header written before the raw R32Float voxel bytes. Fixed
/// 16 bytes so a future version can extend without breaking layout
/// readers (the `version` byte is the gate).
const FILE_MAGIC: [u8; 6] = *b"BLSDF\0";
const FILE_VERSION: u8 = 1;

/// Voxel resolution we bake at. Mirrors `renderer::formats::MESH_SDF_RES`.
/// Hardcoded here rather than imported to keep the cache module
/// dependency-free of the renderer.
pub const VOXEL_RES: u32 = 32;

/// Total payload size: 32³ × f32. Exposed for callers sizing a staging
/// buffer or a `queue.write_texture` source slice.
pub const VOXEL_BYTES: usize = (VOXEL_RES as usize).pow(3) * 4;

/// Content hash of a mesh's geometry. Stable across Rust versions
/// because the underlying mix is FNV-1a over fixed little-endian bytes;
/// renaming the type or adding fields is fine, just don't change the
/// hash math without bumping `FILE_VERSION`.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct MeshHash(pub u64);

impl MeshHash {
    fn to_filename(self) -> String {
        format!("{:016x}.sdf", self.0)
    }
}

/// FNV-1a 64-bit. Algorithmically frozen — changes here invalidate
/// every existing cache entry without warning.
fn fnv1a(input: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in input {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compute a stable 64-bit content hash for the SDF input. Only
/// position bits and indices feed in — meshes that share a surface
/// but differ in normals/uv/colour share a cache entry, which is
/// correct: SDF is geometry-only.
///
/// `positions` is the contiguous `[f32; 3]` slice (one entry per
/// vertex). Callers with an interleaved vertex stride extract just
/// the position component before calling.
pub fn compute_mesh_hash(positions: &[[f32; 3]], indices: &[u32]) -> MeshHash {
    let mut h: u64 = 0xcbf29ce484222325;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    };
    for p in positions {
        for c in p {
            for b in c.to_bits().to_le_bytes() {
                mix(&mut h, b);
            }
        }
    }
    for i in indices {
        for b in i.to_le_bytes() {
            mix(&mut h, b);
        }
    }
    // Fold the vertex/index counts in so two meshes that share a
    // prefix can't accidentally collide via positions-as-suffix.
    for b in (positions.len() as u64).to_le_bytes() {
        mix(&mut h, b);
    }
    for b in (indices.len() as u64).to_le_bytes() {
        mix(&mut h, b);
    }
    let _ = fnv1a; // expose the helper for any future direct use
    MeshHash(h)
}

/// Platform cache root. Returns `None` when the host has no usable
/// cache directory (wasm) or when the env vars used to derive the
/// path aren't set.
///
/// Resolution order:
///   - macOS / iOS / tvOS / watchOS: `$HOME/Library/Caches/bloom/sdf`
///   - Linux / Android:              `${XDG_CACHE_HOME:-$HOME/.cache}/bloom/sdf`
///   - Windows:                      `%LOCALAPPDATA%\bloom\cache\sdf`
///   - wasm32:                       `None`
pub fn cache_dir() -> Option<PathBuf> {
    #[cfg(target_arch = "wasm32")]
    { return None; }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = if cfg!(target_vendor = "apple") {
            let home = std::env::var_os("HOME")?;
            PathBuf::from(home).join("Library").join("Caches").join("bloom").join("sdf")
        } else if cfg!(target_os = "windows") {
            let local = std::env::var_os("LOCALAPPDATA")?;
            PathBuf::from(local).join("bloom").join("cache").join("sdf")
        } else {
            // Linux + Android (XDG-style).
            let base = std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
            base.join("bloom").join("sdf")
        };
        Some(dir)
    }
}

/// Resolve a hash to its cache file path, creating the cache root if
/// needed. Returns `None` if the cache dir can't be created.
fn cache_path(hash: MeshHash) -> Option<PathBuf> {
    let dir = cache_dir()?;
    if !dir.exists() {
        fs::create_dir_all(&dir).ok()?;
    }
    Some(dir.join(hash.to_filename()))
}

/// Look up cached voxel bytes. Returns `Some(bytes)` only when the
/// file exists, parses, and matches the expected magic + version +
/// resolution + payload size. Any failure is silently treated as a
/// miss — the caller falls through to the GPU bake.
pub fn load(hash: MeshHash) -> Option<Vec<u8>> {
    let path = cache_path(hash)?;
    let mut f = fs::File::open(&path).ok()?;

    let mut header = [0u8; 16];
    f.read_exact(&mut header).ok()?;
    if header[..6] != FILE_MAGIC { return None; }
    if header[6] != FILE_VERSION { return None; }
    // header[7] reserved (alignment pad / future flags).
    let res = u32::from_le_bytes(header[8..12].try_into().ok()?);
    if res != VOXEL_RES { return None; }
    // header[12..16] reserved.

    let mut bytes = Vec::with_capacity(VOXEL_BYTES);
    f.read_to_end(&mut bytes).ok()?;
    if bytes.len() != VOXEL_BYTES { return None; }
    Some(bytes)
}

/// Write voxel bytes for a mesh hash. Best-effort; an `Err` return
/// means the cache wasn't updated but rendering can continue.
pub fn store(hash: MeshHash, voxel_bytes: &[u8]) -> std::io::Result<()> {
    if voxel_bytes.len() != VOXEL_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "voxel payload size mismatch",
        ));
    }
    let path = cache_path(hash).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "cache directory unavailable")
    })?;

    // Write to a temp file and rename so a crash mid-write can never
    // leave a partial entry that survives validation.
    let tmp = path.with_extension("sdf.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        let mut header = [0u8; 16];
        header[..6].copy_from_slice(&FILE_MAGIC);
        header[6] = FILE_VERSION;
        // header[7] reserved.
        header[8..12].copy_from_slice(&VOXEL_RES.to_le_bytes());
        // header[12..16] reserved.
        f.write_all(&header)?;
        f.write_all(voxel_bytes)?;
        f.sync_data()?;
    }
    fs::rename(&tmp, &path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_for_identical_input() {
        let pos = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let idx = vec![0_u32, 1, 2];
        assert_eq!(compute_mesh_hash(&pos, &idx), compute_mesh_hash(&pos, &idx));
    }

    #[test]
    fn hash_changes_when_position_changes() {
        let pos1 = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let pos2 = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.001, 0.0]];
        let idx = vec![0_u32, 1, 2];
        assert_ne!(compute_mesh_hash(&pos1, &idx), compute_mesh_hash(&pos2, &idx));
    }

    #[test]
    fn hash_changes_when_index_changes() {
        let pos = vec![[0.0_f32; 3]; 3];
        let idx1 = vec![0_u32, 1, 2];
        let idx2 = vec![0_u32, 2, 1];
        assert_ne!(compute_mesh_hash(&pos, &idx1), compute_mesh_hash(&pos, &idx2));
    }

    #[test]
    fn hash_distinguishes_count_from_value() {
        // Two empty inputs must hash distinctly from a single-zero
        // input — guards against the count-fold being load-bearing.
        let h_empty = compute_mesh_hash(&[], &[]);
        let h_one = compute_mesh_hash(&[[0.0_f32; 3]], &[0_u32]);
        assert_ne!(h_empty, h_one);
    }

    #[test]
    fn store_then_load_roundtrips() {
        // Skip when the env doesn't expose a cache dir (CI sandbox can do this).
        let Some(_) = cache_dir() else { return; };
        // Use a hash unlikely to collide with anything else's tests.
        let h = MeshHash(0xfeed_cafe_dead_beef);
        let bytes: Vec<u8> = (0..VOXEL_BYTES).map(|i| (i * 7 + 13) as u8).collect();
        store(h, &bytes).expect("store");
        let got = load(h).expect("load hit");
        assert_eq!(got, bytes);
        // Cleanup so the test is repeatable.
        if let Some(p) = cache_path(h) { let _ = fs::remove_file(p); }
    }

    #[test]
    fn load_miss_returns_none() {
        let Some(_) = cache_dir() else { return; };
        let h = MeshHash(0x0000_0000_dead_dead);
        if let Some(p) = cache_path(h) { let _ = fs::remove_file(p); }
        assert!(load(h).is_none());
    }

    #[test]
    fn store_rejects_wrong_size() {
        let h = MeshHash(0);
        assert!(store(h, &[0u8; 100]).is_err());
    }

    #[test]
    fn load_rejects_wrong_magic() {
        let Some(dir) = cache_dir() else { return; };
        let _ = fs::create_dir_all(&dir);
        let h = MeshHash(0xbad_0_bad_1);
        let p = dir.join(h.to_filename());
        // Hand-write a file with the wrong magic.
        let mut bad = vec![0u8; 16 + VOXEL_BYTES];
        bad[..6].copy_from_slice(b"NOTBLM");
        std::fs::write(&p, &bad).unwrap();
        assert!(load(h).is_none());
        let _ = fs::remove_file(p);
    }
}
