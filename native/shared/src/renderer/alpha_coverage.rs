//! Coverage-preserving color mip generation for alpha-tested materials.
//!
//! Level zero remains byte-identical to the authored texture. Lower levels
//! store the fraction of source texels that survive the material's effective
//! alpha reference in A, while RGB averages only surviving coverage in linear
//! light. Empty texels receive a nearest visible color so filtering cannot
//! pull transparent border colors into a surviving silhouette. The scene and
//! shadow shaders interpret lower-level alpha as a deterministic subpixel
//! coverage probability.

use std::collections::VecDeque;

/// Build an RGBA8 color mip chain.
///
/// `coverage_reference` is expressed in texture-alpha space. When present,
/// level zero is left untouched and lower mip alpha stores coverage rather
/// than an averaged opacity. When absent, this is the renderer's established
/// four-texel byte average exactly.
pub(super) fn build_color_mip_chain(
    width: u32,
    height: u32,
    data: &[u8],
    mip_count: u32,
    coverage_reference: Option<f32>,
    srgb_rgb: bool,
) -> (Vec<u8>, Vec<usize>) {
    let base_len = width as usize * height as usize * 4;
    assert!(
        data.len() >= base_len,
        "RGBA texture upload is shorter than its declared extent"
    );
    let mut mip_data = Vec::with_capacity(base_len.saturating_mul(2));
    mip_data.extend_from_slice(&data[..base_len]);
    let mut mip_offsets = vec![0usize];
    let mut previous_width = width.max(1);
    let mut previous_height = height.max(1);
    let reference = coverage_reference.map(|value| value.max(0.0));
    let srgb_decode = (reference.is_some() || srgb_rgb)
        .then(|| std::array::from_fn::<_, 256, _>(|value| srgb_u8_to_linear(value as u8)));
    let srgb_encode = (reference.is_some() || srgb_rgb).then(|| {
        (0..=u16::MAX)
            .map(|value| linear_to_srgb_u8(value as f32 / u16::MAX as f32))
            .collect::<Vec<_>>()
    });

    for level in 1..mip_count {
        let previous_offset = mip_offsets[level as usize - 1];
        let width = (previous_width / 2).max(1);
        let height = (previous_height / 2).max(1);
        mip_offsets.push(mip_data.len());
        if let Some(reference) = reference {
            append_coverage_mip_level(
                &mut mip_data,
                previous_offset,
                previous_width,
                previous_height,
                width,
                height,
                reference,
                level == 1,
                srgb_decode.as_ref().expect("coverage decode table exists"),
                srgb_encode.as_ref().expect("coverage encode table exists"),
            );
        } else if srgb_rgb {
            append_ordinary_srgb_mip_level(
                &mut mip_data,
                previous_offset,
                previous_width,
                previous_height,
                width,
                height,
                srgb_decode.as_ref().expect("sRGB decode table exists"),
                srgb_encode.as_ref().expect("sRGB encode table exists"),
            );
        } else {
            append_ordinary_color_mip_level(
                &mut mip_data,
                previous_offset,
                previous_width,
                previous_height,
                width,
                height,
            );
        }
        previous_width = width;
        previous_height = height;
    }

    (mip_data, mip_offsets)
}

#[allow(clippy::too_many_arguments)]
fn append_ordinary_srgb_mip_level(
    mip_data: &mut Vec<u8>,
    previous_offset: usize,
    previous_width: u32,
    previous_height: u32,
    width: u32,
    height: u32,
    srgb_decode: &[f32; 256],
    srgb_encode: &[u8],
) {
    let pw = previous_width as usize;
    let ph = previous_height as usize;
    let index = |x: usize, y: usize| previous_offset + (y * pw + x) * 4;

    for y in 0..height as usize {
        for x in 0..width as usize {
            let sx = x * 2;
            let sy = y * 2;
            let sx1 = (sx + 1).min(pw - 1);
            let sy1 = (sy + 1).min(ph - 1);
            let children = [
                index(sx, sy),
                index(sx1, sy),
                index(sx, sy1),
                index(sx1, sy1),
            ];
            for channel in 0..3 {
                let linear = children
                    .iter()
                    .map(|child| srgb_decode[mip_data[*child + channel] as usize])
                    .sum::<f32>()
                    * 0.25;
                let table_index = (linear.clamp(0.0, 1.0) * u16::MAX as f32).round() as usize;
                mip_data.push(srgb_encode[table_index]);
            }
            let alpha_sum: u32 = children
                .iter()
                .map(|child| mip_data[*child + 3] as u32)
                .sum();
            mip_data.push(((alpha_sum + 2) / 4) as u8);
        }
    }
}

fn append_ordinary_color_mip_level(
    mip_data: &mut Vec<u8>,
    previous_offset: usize,
    previous_width: u32,
    previous_height: u32,
    width: u32,
    height: u32,
) {
    let pw = previous_width as usize;
    let ph = previous_height as usize;
    let index = |x: usize, y: usize| previous_offset + (y * pw + x) * 4;

    for y in 0..height as usize {
        for x in 0..width as usize {
            let sx = x * 2;
            let sy = y * 2;
            let sx1 = (sx + 1).min(pw - 1);
            let sy1 = (sy + 1).min(ph - 1);
            let children = [
                index(sx, sy),
                index(sx1, sy),
                index(sx, sy1),
                index(sx1, sy1),
            ];
            for channel in 0..4 {
                let sum: u32 = children
                    .iter()
                    .map(|child| mip_data[*child + channel] as u32)
                    .sum();
                mip_data.push(((sum + 2) / 4) as u8);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_coverage_mip_level(
    mip_data: &mut Vec<u8>,
    previous_offset: usize,
    previous_width: u32,
    previous_height: u32,
    width: u32,
    height: u32,
    reference: f32,
    previous_alpha_is_authored: bool,
    srgb_decode: &[f32; 256],
    srgb_encode: &[u8],
) {
    let pw = previous_width as usize;
    let ph = previous_height as usize;
    let width = width as usize;
    let height = height as usize;
    let level_offset = mip_data.len();
    let index = |x: usize, y: usize| previous_offset + (y * pw + x) * 4;
    let scale_x = pw as f32 / width as f32;
    let scale_y = ph as f32 / height as f32;

    // Integrate the exact normalized footprint rather than dropping an odd
    // source row/column. A halving step touches at most 3x3 source texels.
    for y in 0..height {
        let y0 = y as f32 * scale_y;
        let y1 = (y + 1) as f32 * scale_y;
        let source_y_end = y1.ceil().min(ph as f32) as usize;
        for x in 0..width {
            let x0 = x as f32 * scale_x;
            let x1 = (x + 1) as f32 * scale_x;
            let source_x_end = x1.ceil().min(pw as f32) as usize;
            let mut area_sum = 0.0f32;
            let mut visible_area = 0.0f32;
            let mut linear_rgb = [0.0f32; 3];

            for source_y in y0.floor() as usize..source_y_end {
                let overlap_y = (y1.min((source_y + 1) as f32) - y0.max(source_y as f32)).max(0.0);
                for source_x in x0.floor() as usize..source_x_end {
                    let overlap_x =
                        (x1.min((source_x + 1) as f32) - x0.max(source_x as f32)).max(0.0);
                    let area = overlap_x * overlap_y;
                    let source = index(source_x.min(pw - 1), source_y.min(ph - 1));
                    let alpha = mip_data[source + 3] as f32 / 255.0;
                    let coverage = if previous_alpha_is_authored {
                        if alpha >= reference {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        alpha
                    };
                    let visible_weight = area * coverage;
                    area_sum += area;
                    visible_area += visible_weight;
                    for channel in 0..3 {
                        linear_rgb[channel] +=
                            srgb_decode[mip_data[source + channel] as usize] * visible_weight;
                    }
                }
            }

            for value in linear_rgb {
                let linear = if visible_area > 1e-8 {
                    value / visible_area
                } else {
                    0.0
                };
                let table_index = (linear.clamp(0.0, 1.0) * u16::MAX as f32).round() as usize;
                mip_data.push(srgb_encode[table_index]);
            }
            let coverage = if area_sum > 1e-8 {
                visible_area / area_sum
            } else {
                0.0
            };
            mip_data.push((coverage * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }

    dilate_visible_rgb(mip_data, level_offset, width, height);
}

fn dilate_visible_rgb(mip_data: &mut [u8], level_offset: usize, width: usize, height: usize) {
    let pixel_count = width * height;
    let mut visited = vec![false; pixel_count];
    let mut queue = VecDeque::new();
    for pixel in 0..pixel_count {
        if mip_data[level_offset + pixel * 4 + 3] != 0 {
            visited[pixel] = true;
            queue.push_back(pixel);
        }
    }
    if queue.is_empty() {
        return;
    }

    while let Some(pixel) = queue.pop_front() {
        let x = pixel % width;
        let y = pixel / width;
        let mut visit = |neighbor: usize| {
            if visited[neighbor] {
                return;
            }
            let source = level_offset + pixel * 4;
            let target = level_offset + neighbor * 4;
            let rgb = [mip_data[source], mip_data[source + 1], mip_data[source + 2]];
            mip_data[target..target + 3].copy_from_slice(&rgb);
            visited[neighbor] = true;
            queue.push_back(neighbor);
        };
        if x > 0 {
            visit(pixel - 1);
        }
        if x + 1 < width {
            visit(pixel + 1);
        }
        if y > 0 {
            visit(pixel - width);
        }
        if y + 1 < height {
            visit(pixel + width);
        }
    }
}

fn srgb_u8_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha_mean(level: &[u8]) -> f32 {
        level
            .chunks_exact(4)
            .map(|pixel| pixel[3] as f32 / 255.0)
            .sum::<f32>()
            / (level.len() / 4) as f32
    }

    #[test]
    fn ordinary_color_mips_keep_the_established_byte_average() {
        let pixels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let (chain, offsets) = build_color_mip_chain(2, 2, &pixels, 2, None, false);
        assert_eq!(offsets, [0, 16]);
        assert_eq!(&chain[16..], &[7, 8, 9, 10]);
    }

    #[test]
    fn srgb_color_mips_average_rgb_in_linear_light_and_alpha_as_data() {
        let pixels = [
            0, 0, 0, 0, 255, 255, 255, 64, 0, 0, 0, 128, 255, 255, 255, 255,
        ];
        let (chain, offsets) = build_color_mip_chain(2, 2, &pixels, 2, None, true);
        assert_eq!(offsets, [0, 16]);
        // 50% linear light encodes to sRGB 188, not the byte-space mean 128.
        assert_eq!(&chain[16..19], &[188, 188, 188]);
        assert_eq!(chain[19], 112);
    }

    #[test]
    fn lower_mips_retain_authored_mask_area_and_visible_color() {
        let mut pixels = Vec::new();
        for index in 0..64 {
            let visible = index % 4 != 0;
            pixels.extend_from_slice(if visible {
                &[40, 180, 30, 255]
            } else {
                &[255, 0, 255, 0]
            });
        }
        let (chain, offsets) = build_color_mip_chain(8, 8, &pixels, 4, Some(0.5), true);
        let expected_coverage = 0.75;
        for (level, extent) in [(1usize, 4usize), (2, 2), (3, 1)] {
            let start = offsets[level];
            let end = start + extent * extent * 4;
            assert!(
                (alpha_mean(&chain[start..end]) - expected_coverage).abs() <= 1.0 / 255.0,
                "level {level} lost source coverage"
            );
            for pixel in chain[start..end].chunks_exact(4) {
                assert!(pixel[1] > pixel[0], "transparent magenta bled into RGB");
            }
        }
        assert_eq!(&chain[..pixels.len()], pixels.as_slice());
    }

    #[test]
    fn odd_extent_coverage_keeps_the_authored_border_area() {
        let mut pixels = vec![0u8; 5 * 3 * 4];
        let visible = (2 * 5 + 4) * 4;
        pixels[visible..visible + 4].copy_from_slice(&[30, 220, 20, 255]);
        let (chain, offsets) = build_color_mip_chain(5, 3, &pixels, 3, Some(0.5), true);
        let level_one = &chain[offsets[1]..offsets[1] + 2 * 4];
        let level_two = &chain[offsets[2]..offsets[2] + 4];
        let authored = 1.0 / 15.0;
        assert!((alpha_mean(level_one) - authored).abs() <= 1.5 / 255.0);
        assert!((alpha_mean(level_two) - authored).abs() <= 1.5 / 255.0);
    }

    #[test]
    fn empty_coverage_texels_receive_visible_edge_color() {
        let mut pixels = Vec::new();
        for _y in 0..8 {
            for x in 0..8 {
                pixels.extend_from_slice(if x < 4 {
                    &[20, 210, 25, 255]
                } else {
                    &[255, 0, 255, 0]
                });
            }
        }
        let (chain, offsets) = build_color_mip_chain(8, 8, &pixels, 2, Some(0.5), true);
        for pixel in chain[offsets[1]..].chunks_exact(4) {
            assert!(
                pixel[1] > pixel[0] && pixel[1] > pixel[2],
                "transparent magenta remained available to bilinear filtering"
            );
        }
    }

    #[test]
    fn coverage_rgb_is_filtered_in_linear_light() {
        let pixels = [
            0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255,
        ];
        let (chain, offsets) = build_color_mip_chain(2, 2, &pixels, 2, Some(0.5), true);
        let output = &chain[offsets[1]..];
        assert!(
            (output[0] as i16 - 188).abs() <= 1,
            "50% linear luminance must encode near sRGB 188, got {}",
            output[0]
        );
        assert_eq!(output[0], output[1]);
        assert_eq!(output[1], output[2]);
        assert_eq!(output[3], 255);
    }
}
