//! CPU-side source mip preparation for environment-map convolution.
//!
//! The GGX prefilter samples these ordinary radiance mips according to each
//! importance sample's solid-angle footprint. This keeps tiny, high-energy HDR
//! emitters (notably the sun) energy-preserving without turning individual
//! Monte Carlo hits into visible fireflies.

const F16_MAX: f32 = 65_504.0;

#[derive(Debug)]
pub(super) struct RadianceMip {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) rgba16: Vec<u16>,
}

fn finite_f16(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-F16_MAX, F16_MAX)
    } else {
        0.0
    }
}

fn pack_rgba16(pixels: &[[f32; 3]]) -> Vec<u16> {
    let mut packed = Vec::with_capacity(pixels.len() * 4);
    for pixel in pixels {
        packed.push(half::f16::from_f32(finite_f16(pixel[0])).to_bits());
        packed.push(half::f16::from_f32(finite_f16(pixel[1])).to_bits());
        packed.push(half::f16::from_f32(finite_f16(pixel[2])).to_bits());
        packed.push(half::f16::from_f32(1.0).to_bits());
    }
    packed
}

fn pack_rgb_f32(rgb_f32: &[f32]) -> Vec<u16> {
    let mut packed = Vec::with_capacity(rgb_f32.len() / 3 * 4);
    for pixel in rgb_f32.chunks_exact(3) {
        packed.push(half::f16::from_f32(finite_f16(pixel[0])).to_bits());
        packed.push(half::f16::from_f32(finite_f16(pixel[1])).to_bits());
        packed.push(half::f16::from_f32(finite_f16(pixel[2])).to_bits());
        packed.push(half::f16::from_f32(1.0).to_bits());
    }
    packed
}

fn row_solid_angle_weight(row: u32, height: u32) -> f64 {
    let theta0 = std::f64::consts::PI * row as f64 / height as f64;
    let theta1 = std::f64::consts::PI * (row + 1) as f64 / height as f64;
    (theta0.cos() - theta1.cos()).max(f64::EPSILON)
}

/// Build an energy-preserving equirectangular radiance pyramid.
///
/// Vertical reduction uses each source row's exact spherical area rather than
/// an image-space box average. Horizontal texels all span the same azimuth.
/// Odd extents are partitioned so every source texel contributes exactly once.
pub(super) fn build_radiance_mip_chain(
    width: u32,
    height: u32,
    rgb_f32: &[f32],
) -> Vec<RadianceMip> {
    let Some(texel_count) = (width as usize).checked_mul(height as usize) else {
        return Vec::new();
    };
    let Some(required_values) = texel_count.checked_mul(3) else {
        return Vec::new();
    };
    if width == 0 || height == 0 || rgb_f32.len() < required_values {
        return Vec::new();
    }

    let mip_count = u32::BITS - width.max(height).leading_zeros();
    let mut mips = Vec::with_capacity(mip_count as usize);
    mips.push(RadianceMip {
        width,
        height,
        rgba16: pack_rgb_f32(&rgb_f32[..required_values]),
    });
    if mip_count == 1 {
        return mips;
    }

    // Keep only reduced float pixels. Mip zero is packed directly from the
    // caller's slice, avoiding a second full-resolution f32 panorama.
    let mut current: Option<Vec<[f32; 3]>> = None;
    let mut current_width = width;
    let mut current_height = height;
    let mut column_weights = vec![1.0_f64; width as usize];
    let mut row_weights = (0..height)
        .map(|row| row_solid_angle_weight(row, height))
        .collect::<Vec<_>>();

    for _level in 1..mip_count {
        let next_width = (current_width / 2).max(1);
        let next_height = (current_height / 2).max(1);
        let mut next = Vec::with_capacity((next_width * next_height) as usize);
        let mut next_column_weights = Vec::with_capacity(next_width as usize);
        for x in 0..next_width {
            let source_x0 = x * current_width / next_width;
            let source_x1 = ((x + 1) * current_width / next_width).max(source_x0 + 1);
            next_column_weights.push(
                column_weights[source_x0 as usize..source_x1 as usize]
                    .iter()
                    .sum(),
            );
        }
        let mut next_row_weights = Vec::with_capacity(next_height as usize);
        for y in 0..next_height {
            let source_y0 = y * current_height / next_height;
            let source_y1 = ((y + 1) * current_height / next_height).max(source_y0 + 1);
            next_row_weights.push(
                row_weights[source_y0 as usize..source_y1 as usize]
                    .iter()
                    .sum(),
            );
            for x in 0..next_width {
                let source_x0 = x * current_width / next_width;
                let source_x1 = ((x + 1) * current_width / next_width).max(source_x0 + 1);
                let mut sum = [0.0_f64; 3];
                let mut weight_sum = 0.0_f64;
                for source_y in source_y0..source_y1 {
                    for source_x in source_x0..source_x1 {
                        let source_index = (source_y * current_width + source_x) as usize;
                        let pixel = if let Some(current) = current.as_ref() {
                            current[source_index]
                        } else {
                            let rgb_index = source_index * 3;
                            [
                                finite_f16(rgb_f32[rgb_index]),
                                finite_f16(rgb_f32[rgb_index + 1]),
                                finite_f16(rgb_f32[rgb_index + 2]),
                            ]
                        };
                        let weight =
                            row_weights[source_y as usize] * column_weights[source_x as usize];
                        sum[0] += pixel[0] as f64 * weight;
                        sum[1] += pixel[1] as f64 * weight;
                        sum[2] += pixel[2] as f64 * weight;
                        weight_sum += weight;
                    }
                }
                next.push([
                    finite_f16((sum[0] / weight_sum) as f32),
                    finite_f16((sum[1] / weight_sum) as f32),
                    finite_f16((sum[2] / weight_sum) as f32),
                ]);
            }
        }
        mips.push(RadianceMip {
            width: next_width,
            height: next_height,
            rgba16: pack_rgba16(&next),
        });
        current = Some(next);
        current_width = next_width;
        current_height = next_height;
        column_weights = next_column_weights;
        row_weights = next_row_weights;
    }

    mips
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack_rgb(mip: &RadianceMip, index: usize) -> [f32; 3] {
        [
            half::f16::from_bits(mip.rgba16[index * 4]).to_f32(),
            half::f16::from_bits(mip.rgba16[index * 4 + 1]).to_f32(),
            half::f16::from_bits(mip.rgba16[index * 4 + 2]).to_f32(),
        ]
    }

    #[test]
    fn constant_radiance_survives_every_mip() {
        let source = vec![4.25_f32; 8 * 4 * 3];
        let mips = build_radiance_mip_chain(8, 4, &source);
        assert_eq!(
            mips.iter().map(|m| (m.width, m.height)).collect::<Vec<_>>(),
            [(8, 4), (4, 2), (2, 1), (1, 1)]
        );
        for mip in &mips {
            for index in 0..(mip.width * mip.height) as usize {
                assert_eq!(unpack_rgb(mip, index), [4.25; 3]);
            }
        }
    }

    #[test]
    fn spherical_average_preserves_polar_energy() {
        let mut source = vec![0.0_f32; 8 * 4 * 3];
        for x in 0..8 {
            source[x * 3] = 1_000.0;
        }
        let mips = build_radiance_mip_chain(8, 4, &source);
        let final_rgb = unpack_rgb(mips.last().unwrap(), 0);
        let polar_band_fraction = row_solid_angle_weight(0, 4) / row_solid_angle_weight(0, 1);
        let expected = 1_000.0 * polar_band_fraction as f32;
        assert!((final_rgb[0] - expected).abs() < 0.1);
        assert_eq!(final_rgb[1], 0.0);
        assert_eq!(final_rgb[2], 0.0);
    }

    #[test]
    fn odd_extents_cover_all_texels_and_non_finite_values_are_bounded() {
        let mut source = vec![2.0_f32; 5 * 3 * 3];
        source[0] = f32::NAN;
        source[1] = f32::INFINITY;
        source[2] = -f32::INFINITY;
        source[3] = 100_000.0;
        let mips = build_radiance_mip_chain(5, 3, &source);
        assert_eq!(
            mips.iter().map(|m| (m.width, m.height)).collect::<Vec<_>>(),
            [(5, 3), (2, 1), (1, 1)]
        );
        for mip in &mips {
            for bits in &mip.rgba16 {
                assert!(half::f16::from_bits(*bits).to_f32().is_finite());
            }
        }
        assert_eq!(unpack_rgb(&mips[0], 0), [0.0; 3]);
        assert_eq!(unpack_rgb(&mips[0], 1)[0], F16_MAX);
    }

    #[test]
    fn odd_extents_preserve_the_spherical_average_across_reductions() {
        let width = 5;
        let height = 5;
        let mut source = vec![0.0_f32; width * height * 3];
        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for y in 0..height {
            for x in 0..width {
                let value = (y * width + x + 1) as f32;
                source[(y * width + x) * 3] = value;
                let weight = row_solid_angle_weight(y as u32, height as u32);
                weighted_sum += value as f64 * weight;
                weight_sum += weight;
            }
        }

        let mips = build_radiance_mip_chain(width as u32, height as u32, &source);
        let expected = (weighted_sum / weight_sum) as f32;
        let final_value = unpack_rgb(mips.last().unwrap(), 0)[0];
        assert!((final_value - expected).abs() < 0.01);
    }
}
