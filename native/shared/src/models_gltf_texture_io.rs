//! Texture-byte and URI helpers for the glTF loader.

/// Replace the extension on a URI (keeps directories / query strings
/// untouched). Used to fall back from `foo.png` → `foo.dds` when a
/// glTF references a PNG URI that isn't on disk but the DDS sibling is.
pub(super) fn swap_extension(uri: &str, new_ext: &str) -> String {
    let q = uri.find('?').unwrap_or(uri.len());
    let (path, query) = uri.split_at(q);
    let new_path = match path.rfind('.') {
        Some(dot) if dot > path.rfind('/').unwrap_or(0) => {
            format!("{}.{}", &path[..dot], new_ext)
        }
        _ => format!("{}.{}", path, new_ext),
    };
    format!("{}{}", new_path, query)
}

/// Decode a texture byte slice into RGBA8 pixels + dimensions. Tries
/// DDS first when the URI extension suggests it (for asset packs like
/// Lumberyard Bistro that ship BC-compressed textures), falling back
/// to the `image` crate for PNG/JPEG/etc. Returns None on failure.
pub(super) fn decode_texture_bytes(bytes: &[u8], uri: &str) -> Option<(Vec<u8>, u32, u32)> {
    let is_dds =
        uri.to_ascii_lowercase().ends_with(".dds") || bytes.len() >= 4 && &bytes[..4] == b"DDS ";
    if is_dds {
        if let Ok(dds) = image_dds::ddsfile::Dds::read(bytes) {
            // Decode mip 0 → RGBA8. image_from_dds handles the common
            // BC1–BC7 formats; anything it can't decode falls through
            // to the image crate which will almost certainly fail too.
            if let Ok(rgba) = image_dds::image_from_dds(&dds, 0) {
                let (w, h) = (rgba.width(), rgba.height());
                return Some((rgba.into_raw(), w, h));
            }
        }
    }
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Some((rgba.into_raw(), w, h))
}

pub(super) fn base64_decode(input: &str, output: &mut Vec<u8>) {
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in input.as_bytes() {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' => continue,
            _ => continue,
        };
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
}
