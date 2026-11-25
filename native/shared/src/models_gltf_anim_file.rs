//! Reading a GLB's skeleton and clips without reading its textures.
//!
//! `loadModelAnimation(path)` is called on the same file as `loadModel`, and
//! a character's GLB is mostly textures (Jonah's: 91 of 124 MB). This reads
//! the GLB header, the JSON chunk and only the buffer views that the skin's
//! inverse bind matrices and the animation samplers use, into an otherwise
//! untouched zeroed buffer of the binary chunk's size (fresh zero pages
//! cost no memory until written).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::ModelAnimation;

/// `None`: not a case handled here (read the whole file instead).
/// `Some(result)`: the result of the load, read from the file's ranges.
pub(super) fn load(path: &Path) -> Option<Option<ModelAnimation>> {
    let mut f = File::open(path).ok()?;
    let mut hdr = [0u8; 20];
    f.read_exact(&mut hdr).ok()?;
    if &hdr[0..4] != b"glTF" || &hdr[16..20] != b"JSON" {
        return None;
    }
    let json_len = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]) as u64;
    let mut json = vec![0u8; json_len as usize];
    f.read_exact(&mut json).ok()?;
    let mut bin_hdr = [0u8; 8];
    let (bin_off, bin_len) = match f.read_exact(&mut bin_hdr) {
        Ok(()) if &bin_hdr[4..8] == b"BIN\0" => (
            20 + json_len + 8,
            u32::from_le_bytes([bin_hdr[0], bin_hdr[1], bin_hdr[2], bin_hdr[3]]) as usize,
        ),
        Ok(()) => return None,
        Err(_) => (0, 0),
    };
    let gltf = super::parse::parse_gltf(&json)?;
    // one buffer, the GLB's own binary chunk
    let mut buffers = gltf.buffers();
    match (buffers.next(), buffers.next()) {
        (Some(b), None) if matches!(b.source(), gltf::buffer::Source::Bin) => {}
        (None, None) => {}
        _ => return None,
    }
    let mut accessors: Vec<gltf::Accessor> = Vec::new();
    for skin in gltf.skins() {
        accessors.extend(skin.inverse_bind_matrices());
    }
    for anim in gltf.animations() {
        for sampler in anim.samplers() {
            accessors.push(sampler.input());
            accessors.push(sampler.output());
        }
    }
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for a in &accessors {
        if a.sparse().is_some() {
            return None;
        }
        if let Some(view) = a.view() {
            let (o, l) = (view.offset(), view.length());
            // the accessor's last element must lie inside its view (else the
            // full read would see bytes this one leaves at zero)
            let stride = view.stride().unwrap_or(a.size());
            let span = a.offset() + a.count().saturating_sub(1) * stride + a.size();
            if o + l > bin_len || (a.count() > 0 && span > l) {
                return None;
            }
            ranges.push((o, l));
        }
    }
    ranges.sort_unstable();
    ranges.dedup();
    let mut bin = vec![0u8; bin_len];
    for (o, l) in ranges {
        f.seek(SeekFrom::Start(bin_off + o as u64)).ok()?;
        f.read_exact(&mut bin[o..o + l]).ok()?;
    }
    let buffer_data = if bin_len > 0 { vec![bin] } else { Vec::new() };
    Some(super::animation_from(&gltf, &buffer_data))
}

#[cfg(test)]
mod tests {
    use super::super::{animation_from, load_gltf_animation};
    use crate::models::ModelAnimation;

    /// Every number and name of a loaded clip set, bit for bit.
    fn digest(a: &ModelAnimation) -> (Vec<u32>, Vec<String>) {
        let mut v: Vec<u32> = Vec::new();
        let mut names = Vec::new();
        if let Some(s) = &a.skeleton {
            for j in &s.joints {
                v.extend(j.inverse_bind.iter().flatten().map(|x| x.to_bits()));
                v.extend(
                    j.rest_translation
                        .iter()
                        .chain(j.rest_rotation.iter())
                        .chain(j.rest_scale.iter())
                        .map(|x| x.to_bits()),
                );
                v.extend(j.children.iter().map(|&c| c as u32));
                names.push(j.name.clone());
            }
            v.extend(s.root_joints.iter().map(|&r| r as u32));
        }
        for clip in a.animations.iter() {
            names.push(clip.name.clone());
            v.push(clip.duration.to_bits());
            // (channels come out of a HashMap: their order is not fixed)
            let mut chans: Vec<_> = clip.channels.iter().collect();
            chans.sort_by_key(|c| c.joint_index);
            for c in chans {
                v.push(c.joint_index as u32);
                let f = c
                    .timestamps
                    .iter()
                    .chain(c.rotation_timestamps.iter())
                    .chain(c.scale_timestamps.iter());
                v.extend(f.map(|x| x.to_bits()));
                v.extend(
                    c.translations
                        .iter()
                        .flatten()
                        .chain(c.scales.iter().flatten())
                        .map(|x| x.to_bits()),
                );
                v.extend(c.rotations.iter().flatten().map(|x| x.to_bits()));
            }
            for m in &clip.morph_tracks {
                v.extend(m.slots.iter().copied());
                v.extend(m.times.iter().chain(m.values.iter()).map(|x| x.to_bits()));
            }
        }
        (v, names)
    }

    fn same_as_full_read(path: &std::path::Path) {
        let full = load_gltf_animation(&std::fs::read(path).unwrap()).expect("full read loads");
        let ranged = super::load(path)
            .expect("a GLB is read by ranges")
            .expect("ranged read loads");
        let (a, an) = digest(&full);
        let (b, bn) = digest(&ranged);
        assert!(!a.is_empty(), "{} has clip data", path.display());
        assert_eq!(an, bn);
        assert!(
            a == b,
            "{}: ranged read differs from the full read",
            path.display()
        );
        let _ = animation_from; // (the shared body both paths use)
    }

    #[test]
    fn reading_only_the_clip_ranges_gives_the_full_reads_clips() {
        let fox = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/test-gltf-watch/assets/Fox.glb");
        same_as_full_read(&fox);
    }

    /// BLOOM_ANIM_COMPARE=a.glb:b.glb cargo test -- --ignored anim_compare
    #[test]
    #[ignore]
    fn anim_compare_files_from_env() {
        let list = std::env::var("BLOOM_ANIM_COMPARE").unwrap_or_default();
        for p in list.split(':').filter(|p| !p.is_empty()) {
            let p = std::path::Path::new(p);
            same_as_full_read(p);
            let t = std::time::Instant::now();
            let _ = load_gltf_animation(&std::fs::read(p).unwrap());
            let full = t.elapsed();
            let t = std::time::Instant::now();
            let _ = super::load(p);
            eprintln!("{}: full {:?}, ranges {:?}", p.display(), full, t.elapsed());
        }
    }
}
