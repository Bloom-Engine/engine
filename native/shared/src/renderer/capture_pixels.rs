//! Convert captured texture bytes to the RGB PNG contract without changing
//! transfer functions or the raw surface bytes exposed to screenshot callers.

pub(super) fn rgba8_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        for column in 0..width {
            let offset = row_start + (column * 4) as usize;
            rgb.extend_from_slice(&data[offset..offset + 3]);
        }
    }
    rgb
}

pub(super) fn frame_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    format: wgpu::TextureFormat,
) -> Vec<u8> {
    let mut rgb = rgba8_rgb(data, width, height, padded_bytes_per_row);
    if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in rgb.chunks_exact_mut(3) {
            pixel.swap(0, 2);
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::frame_rgb;
    use crate::renderer::util::encode_png_simple;
    use wgpu::TextureFormat;

    #[test]
    fn png_preserves_colors_for_rgba_and_bgra_with_padded_rows() {
        let expected = vec![255, 16, 32, 8, 64, 192];
        for format in [
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
        ] {
            let mut bytes = vec![99; 512];
            let bgra = matches!(
                format,
                TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
            );
            bytes[0..4].copy_from_slice(if bgra {
                &[32, 16, 255, 255]
            } else {
                &[255, 16, 32, 255]
            });
            bytes[256..260].copy_from_slice(if bgra {
                &[192, 64, 8, 127]
            } else {
                &[8, 64, 192, 127]
            });
            let rgb = frame_rgb(&bytes, 1, 2, 256, format);
            let png = encode_png_simple(1, 2, &rgb).unwrap();
            let decoded = image::load_from_memory(&png).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (1, 2));
            assert_eq!(decoded.as_raw(), &expected, "{format:?}");
        }
    }
}
