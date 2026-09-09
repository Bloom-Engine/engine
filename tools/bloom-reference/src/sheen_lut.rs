//! Deterministic directional-albedo oracle for the Khronos Charlie sheen BRDF.
//!
//! The checked R16F table is generated from these equations and consumed by
//! both the CPU contract and Bloom's lazy realtime layered-material path.

#![allow(dead_code)]

use rayon::prelude::*;

pub const DEFAULT_LUT_SIZE: usize = 128;
pub const DEFAULT_SAMPLE_COUNT: u32 = 4096;

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

fn lambda_numeric_helper(x: f32, alpha_g: f32) -> f32 {
    let one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);
    let mix = |a: f32, b: f32| a + (b - a) * one_minus_alpha_sq;
    let a = mix(21.5473, 25.3245);
    let b = mix(3.82987, 3.32435);
    let c = mix(0.19823, 0.16801);
    let d = mix(-1.97760, -1.27393);
    let e = mix(-4.32054, -4.85967);
    a / (1.0 + b * x.max(0.0).powf(c)) + d * x + e
}

fn lambda_sheen(cos_theta: f32, alpha_g: f32) -> f32 {
    let cosine = cos_theta.abs().clamp(0.0, 1.0);
    if cosine < 0.5 {
        lambda_numeric_helper(cosine, alpha_g).exp()
    } else {
        (2.0 * lambda_numeric_helper(0.5, alpha_g) - lambda_numeric_helper(1.0 - cosine, alpha_g))
            .exp()
    }
}

pub fn visibility_sheen(n_dot_l: f32, n_dot_v: f32, perceptual_roughness: f32) -> f32 {
    let n_dot_l = n_dot_l.max(1e-6);
    let n_dot_v = n_dot_v.max(1e-6);
    let alpha_g = perceptual_roughness.max(1e-3).powi(2);
    1.0 / ((1.0 + lambda_sheen(n_dot_v, alpha_g) + lambda_sheen(n_dot_l, alpha_g))
        * (4.0 * n_dot_v * n_dot_l))
}

pub fn distribution_charlie(n_dot_h: f32, perceptual_roughness: f32) -> f32 {
    let alpha_g = perceptual_roughness.max(1e-3).powi(2);
    let inverse_alpha = 1.0 / alpha_g;
    let sin2_h = (1.0 - n_dot_h.clamp(0.0, 1.0).powi(2)).max(0.0);
    (2.0 + inverse_alpha) * sin2_h.powf(0.5 * inverse_alpha) / (2.0 * std::f32::consts::PI)
}

/// Integrate `E(NdotV, roughness)` by importance-sampling the normalized
/// Charlie microfacet-normal distribution. The estimator cancels `D`
/// analytically and remains stable for the grazing, near-delta cloth lobe.
pub fn directional_albedo(n_dot_v: f32, perceptual_roughness: f32, sample_count: u32) -> f32 {
    let n_dot_v = n_dot_v.clamp(1e-4, 1.0);
    let roughness = perceptual_roughness.clamp(1e-3, 1.0);
    let alpha_g = roughness * roughness;
    let view = [(1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v];
    let mut sum = 0.0;
    for index in 0..sample_count {
        let u = (index as f32 + 0.5) / sample_count as f32;
        let v = radical_inverse_vdc(index);
        let sin_theta = u.powf(alpha_g / (2.0 * alpha_g + 1.0));
        let cos_theta = (1.0 - sin_theta * sin_theta).max(0.0).sqrt();
        let phi = std::f32::consts::TAU * v;
        let half = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
        let v_dot_h = (view[0] * half[0] + view[1] * half[1] + view[2] * half[2]).max(0.0);
        if v_dot_h <= 0.0 || cos_theta <= 0.0 {
            continue;
        }
        let light = [
            2.0 * v_dot_h * half[0] - view[0],
            2.0 * v_dot_h * half[1] - view[1],
            2.0 * v_dot_h * half[2] - view[2],
        ];
        let n_dot_l = light[2];
        if n_dot_l <= 0.0 {
            continue;
        }
        // p(H)=D(H)NdotH and p(L)=p(H)/(4 VdotH).
        sum += visibility_sheen(n_dot_l, n_dot_v, roughness) * n_dot_l * 4.0 * v_dot_h / cos_theta;
    }
    (sum / sample_count.max(1) as f32).clamp(0.0, 1.0)
}

pub fn build_r16f_lut(size: usize, sample_count: u32) -> Vec<u16> {
    (0..size * size)
        .into_par_iter()
        .map(|index| {
            let x = index % size;
            let y = index / size;
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let roughness = (y as f32 + 0.5) / size as f32;
            half::f16::from_f32(directional_albedo(n_dot_v, roughness, sample_count)).to_bits()
        })
        .collect()
}

pub fn sample_r16f_lut(bytes: &[u8], size: usize, n_dot_v: f32, roughness: f32) -> f32 {
    assert_eq!(bytes.len(), size * size * 2);
    let coordinate = |value: f32| value.clamp(0.0, 1.0) * size as f32 - 0.5;
    let x = coordinate(n_dot_v);
    let y = coordinate(roughness);
    let x0 = x.floor().clamp(0.0, (size - 1) as f32) as usize;
    let y0 = y.floor().clamp(0.0, (size - 1) as f32) as usize;
    let x1 = (x0 + 1).min(size - 1);
    let y1 = (y0 + 1).min(size - 1);
    let tx = (x - x.floor()).clamp(0.0, 1.0);
    let ty = (y - y.floor()).clamp(0.0, 1.0);
    let load = |x: usize, y: usize| {
        let offset = (y * size + x) * 2;
        half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
    };
    let top = load(x0, y0) + (load(x1, y0) - load(x0, y0)) * tx;
    let bottom = load(x0, y1) + (load(x1, y1) - load(x0, y1)) * tx;
    top + (bottom - top) * ty
}
