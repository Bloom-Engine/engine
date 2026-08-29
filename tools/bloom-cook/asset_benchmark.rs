//! Repeatable cooked-vs-source asset qualification measurements.

use crate::geometry_format::decode_geometry;
use crate::texture_cook::{cook_prepared_texture, PreparedTexture, TextureSettings};
use image_dds::ddsfile::Dds;
use serde_json::{json, Value};
use std::hint::black_box;
use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

const DEFAULT_ITERATIONS: usize = 25;
const MAX_ITERATIONS: usize = 1_000;
const WARMUP_ITERATIONS: usize = 3;

pub(crate) fn benchmark_texture_command(input: &Path, flags: &[String]) -> Result<String, String> {
    let (iterations, texture_flags) = split_iterations(flags)?;
    let settings = TextureSettings::parse(texture_flags.iter().map(String::as_str))?;
    let prepared = PreparedTexture::read(input, settings)?;

    let encode_start = Instant::now();
    let cooked = cook_prepared_texture(input, &prepared)?;
    let offline_encode_ms = encode_start.elapsed().as_secs_f64() * 1_000.0;
    let dds = Dds::read(Cursor::new(&cooked.bytes))
        .map_err(|error| format!("parse cooked DDS: {error}"))?;
    let decoded = image_dds::image_from_dds(&dds, 0)
        .map_err(|error| format!("decode cooked DDS mip zero: {error}"))?;
    let source = image::load_from_memory(&prepared.source_bytes)
        .map_err(|error| format!("decode source texture: {error}"))?
        .to_rgba8();
    if source.dimensions() != decoded.dimensions() {
        return Err("cooked DDS dimensions differ from the source texture".to_string());
    }

    let raw_decode = benchmark(iterations, || {
        let image = image::load_from_memory(&prepared.source_bytes)
            .map_err(|error| format!("decode source texture: {error}"))?
            .to_rgba8();
        black_box((image.width(), image.height(), image.as_raw().len()));
        Ok(())
    })?;
    let cooked_parse = benchmark(iterations, || {
        let dds = Dds::read(Cursor::new(&cooked.bytes))
            .map_err(|error| format!("parse cooked DDS: {error}"))?;
        black_box((dds.get_width(), dds.get_height(), dds.data.len()));
        Ok(())
    })?;
    let cooked_fallback_decode = benchmark(iterations, || {
        let dds = Dds::read(Cursor::new(&cooked.bytes))
            .map_err(|error| format!("parse cooked DDS: {error}"))?;
        let image = image_dds::image_from_dds(&dds, 0)
            .map_err(|error| format!("decode cooked DDS mip zero: {error}"))?;
        black_box((image.width(), image.height(), image.as_raw().len()));
        Ok(())
    })?;

    let mip_levels = dds.get_num_mipmap_levels();
    let raw_gpu_bytes = rgba8_mip_bytes(source.width(), source.height(), mip_levels);
    let cooked_gpu_bytes = dds.data.len() as u64;
    let expected_cooked_bytes = if settings.is_normal_map() {
        raw_gpu_bytes
    } else {
        bc_mip_bytes(source.width(), source.height(), mip_levels)
    };
    if cooked_gpu_bytes != expected_cooked_bytes {
        return Err(format!(
            "DDS payload is {cooked_gpu_bytes} bytes, expected {expected_cooked_bytes} for {}",
            settings.format_name()
        ));
    }
    let quality = texture_quality(
        source.as_raw(),
        decoded.as_raw(),
        source.width(),
        source.height(),
        settings,
    );
    let report = json!({
        "dimensions": {
            "height": source.height(),
            "mip_levels": mip_levels,
            "width": source.width(),
        },
        "disk": {
            "cooked_dds_bytes": cooked.bytes.len(),
            "cooked_to_source_ratio": cooked.bytes.len() as f64 / prepared.source_bytes.len() as f64,
            "source_bytes": prepared.source_bytes.len(),
        },
        "gpu_memory": {
            "cooked_adapter_bytes": cooked_gpu_bytes,
            "cooked_memory_reduction_percent":
                (1.0 - cooked_gpu_bytes as f64 / raw_gpu_bytes as f64) * 100.0,
            "cooked_to_raw_ratio": cooked_gpu_bytes as f64 / raw_gpu_bytes as f64,
            "raw_rgba8_bytes": raw_gpu_bytes,
        },
        "input": input.display().to_string(),
        "iterations": iterations,
        "kind": "texture",
        "quality": quality,
        "schema": "bloom-asset-benchmark-v1",
        "settings": settings.as_json(),
        "artifact_format": settings.format_name(),
        "timing_ms": {
            "cooked_dds_cpu_fallback_parse_and_decode": cooked_fallback_decode.as_json(),
            "cooked_dds_parse_for_direct_upload": cooked_parse.as_json(),
            "offline_texture_encode": offline_encode_ms,
            "raw_source_decode": raw_decode.as_json(),
            "source_decode_to_cooked_parse_mean_speedup":
                raw_decode.mean / cooked_parse.mean.max(f64::MIN_POSITIVE),
        },
        "timing_scope": "warm OS cache; CPU parse/decode only; excludes GPU creation and upload",
    });
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize texture benchmark: {error}"))
}

pub(crate) fn benchmark_geometry_command(
    source: &Path,
    cooked: &Path,
    flags: &[String],
) -> Result<String, String> {
    let (iterations, remaining) = split_iterations(flags)?;
    if let Some(flag) = remaining.first() {
        return Err(format!("unknown geometry benchmark option {flag:?}"));
    }
    let source_bytes = std::fs::metadata(source)
        .map_err(|error| format!("inspect {}: {error}", source.display()))?
        .len();
    let cooked_bytes =
        std::fs::read(cooked).map_err(|error| format!("read {}: {error}", cooked.display()))?;
    let archive = decode_geometry(&cooked_bytes)?;

    let source_import = benchmark(iterations, || {
        let (document, buffers, images) = gltf::import(source)
            .map_err(|error| format!("import source glTF {}: {error}", source.display()))?;
        let buffer_bytes = buffers.iter().map(|buffer| buffer.0.len()).sum::<usize>();
        let image_bytes = images.iter().map(|image| image.pixels.len()).sum::<usize>();
        black_box((
            document.meshes().count(),
            document.materials().count(),
            buffer_bytes,
            image_bytes,
        ));
        Ok(())
    })?;
    let cooked_validation = benchmark(iterations, || {
        let bytes = std::fs::read(cooked)
            .map_err(|error| format!("read cooked geometry {}: {error}", cooked.display()))?;
        let archive = decode_geometry(&bytes)?;
        black_box((
            archive.clusters.len(),
            archive.pages.len(),
            archive.payload_bytes(),
        ));
        Ok(())
    })?;

    let report = json!({
        "artifact": {
            "clusters": archive.clusters.len(),
            "compatibility_records": archive.compatibility.len(),
            "format_version": archive.format_version,
            "pages": archive.pages.len(),
            "payload_bytes": archive.payload_bytes(),
            "triangles": archive.triangle_count(),
        },
        "bytes": {
            "cooked_bgeo": cooked_bytes.len(),
            "cooked_to_source_ratio": cooked_bytes.len() as f64 / source_bytes as f64,
            "source_gltf": source_bytes,
        },
        "cooked": cooked.display().to_string(),
        "input": source.display().to_string(),
        "iterations": iterations,
        "kind": "geometry",
        "schema": "bloom-asset-benchmark-v1",
        "timing_ms": {
            "cooked_full_read_and_validation": cooked_validation.as_json(),
            "source_gltf_import": source_import.as_json(),
            "source_import_to_cooked_validation_mean_speedup":
                source_import.mean / cooked_validation.mean.max(f64::MIN_POSITIVE),
        },
        "timing_scope": "warm OS cache; source includes glTF parse plus buffer/image import; cooked includes full file read, structure validation, and hashes; excludes GPU creation/upload",
    });
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize geometry benchmark: {error}"))
}

#[derive(Debug)]
struct TimingSummary {
    mean: f64,
    median: f64,
    min: f64,
    p95: f64,
    max: f64,
}

impl TimingSummary {
    fn as_json(&self) -> Value {
        json!({
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "p95": self.p95,
        })
    }
}

fn benchmark(
    iterations: usize,
    mut operation: impl FnMut() -> Result<(), String>,
) -> Result<TimingSummary, String> {
    for _ in 0..WARMUP_ITERATIONS {
        operation()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation()?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = if samples.len().is_multiple_of(2) {
        let upper = samples.len() / 2;
        (samples[upper - 1] + samples[upper]) * 0.5
    } else {
        samples[samples.len() / 2]
    };
    let p95_index = (samples.len() * 95).div_ceil(100).saturating_sub(1);
    Ok(TimingSummary {
        mean,
        median,
        min: samples[0],
        p95: samples[p95_index],
        max: samples[samples.len() - 1],
    })
}

fn split_iterations(flags: &[String]) -> Result<(usize, Vec<String>), String> {
    let mut iterations = None;
    let mut remaining = Vec::new();
    let mut index = 0;
    while index < flags.len() {
        if flags[index] != "--iterations" {
            remaining.push(flags[index].clone());
            index += 1;
            continue;
        }
        let value = flags
            .get(index + 1)
            .ok_or("--iterations requires a value")?
            .parse::<usize>()
            .map_err(|_| "--iterations must be an integer".to_string())?;
        if !(1..=MAX_ITERATIONS).contains(&value) {
            return Err(format!(
                "--iterations must be between 1 and {MAX_ITERATIONS}"
            ));
        }
        if iterations.replace(value).is_some() {
            return Err("--iterations may only be specified once".to_string());
        }
        index += 2;
    }
    Ok((iterations.unwrap_or(DEFAULT_ITERATIONS), remaining))
}

fn rgba8_mip_bytes(mut width: u32, mut height: u32, mip_levels: u32) -> u64 {
    let mut bytes = 0u64;
    for _ in 0..mip_levels {
        bytes += u64::from(width) * u64::from(height) * 4;
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    bytes
}

fn bc_mip_bytes(mut width: u32, mut height: u32, mip_levels: u32) -> u64 {
    let mut bytes = 0u64;
    for _ in 0..mip_levels {
        bytes += u64::from(width.div_ceil(4)) * u64::from(height.div_ceil(4)) * 16;
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    bytes
}

fn texture_quality(
    reference: &[u8],
    candidate: &[u8],
    width: u32,
    height: u32,
    settings: TextureSettings,
) -> Value {
    let mut sum_abs_rgb = 0u64;
    let mut sum_sq_rgb = 0.0;
    let mut sum_sq_linear_rgb = 0.0;
    let mut max_rgb = 0u8;
    let mut sum_abs_alpha = 0u64;
    let mut max_alpha = 0u8;
    let mut normal_angle_sum = 0.0;
    let mut normal_angle_max = 0.0f64;
    let pixels = (width as usize) * (height as usize);
    for (reference, candidate) in reference.chunks_exact(4).zip(candidate.chunks_exact(4)) {
        for channel in 0..3 {
            let delta = reference[channel].abs_diff(candidate[channel]);
            sum_abs_rgb += u64::from(delta);
            sum_sq_rgb += (f64::from(delta) / 255.0).powi(2);
            max_rgb = max_rgb.max(delta);
            let to_linear = |value: u8| {
                let value = f64::from(value) / 255.0;
                if settings.is_srgb() {
                    srgb_to_linear(value)
                } else {
                    value
                }
            };
            sum_sq_linear_rgb +=
                (to_linear(reference[channel]) - to_linear(candidate[channel])).powi(2);
        }
        let expected_alpha = if settings.is_normal_map() {
            0
        } else {
            reference[3]
        };
        let alpha_delta = expected_alpha.abs_diff(candidate[3]);
        sum_abs_alpha += u64::from(alpha_delta);
        max_alpha = max_alpha.max(alpha_delta);
        if settings.is_normal_map() {
            let reference_normal = decode_normal(reference);
            let candidate_normal = decode_normal(candidate);
            let dot = reference_normal
                .iter()
                .zip(candidate_normal)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let angle = dot.acos().to_degrees();
            normal_angle_sum += angle;
            normal_angle_max = normal_angle_max.max(angle);
        }
    }
    let rgb_channels = (pixels * 3) as f64;
    let rmse_rgb = (sum_sq_rgb / rgb_channels).sqrt();
    let rmse_linear_rgb = (sum_sq_linear_rgb / rgb_channels).sqrt();
    let mut quality = json!({
        "alpha_max_abs_byte": max_alpha,
        "alpha_mean_abs_byte": sum_abs_alpha as f64 / pixels as f64,
        "luminance_ssim": ssim_luminance(reference, candidate, width, height),
        "rgb_linear_rmse": rmse_linear_rgb,
        "rgb_max_abs_byte": max_rgb,
        "rgb_mean_abs_byte": sum_abs_rgb as f64 / rgb_channels,
        "rgb_psnr_db": psnr(rmse_rgb),
        "rgb_rmse": rmse_rgb,
    });
    if settings.is_normal_map() {
        quality["normal_mean_angular_error_degrees"] = json!(normal_angle_sum / pixels as f64);
        quality["normal_max_angular_error_degrees"] = json!(normal_angle_max);
    }
    quality
}

fn psnr(rmse: f64) -> Value {
    if rmse == 0.0 {
        Value::Null
    } else {
        json!(20.0 * (1.0 / rmse).log10())
    }
}

fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn decode_normal(pixel: &[u8]) -> [f64; 3] {
    let mut normal = [
        f64::from(pixel[0]) * (2.0 / 255.0) - 1.0,
        f64::from(pixel[1]) * (2.0 / 255.0) - 1.0,
        f64::from(pixel[2]) * (2.0 / 255.0) - 1.0,
    ];
    let length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    if length <= 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    for value in &mut normal {
        *value /= length;
    }
    normal
}

fn ssim_luminance(reference: &[u8], candidate: &[u8], width: u32, height: u32) -> f64 {
    const WINDOW: usize = 8;
    const C1: f64 = 0.0001;
    const C2: f64 = 0.0009;
    let width = width as usize;
    let height = height as usize;
    if width < WINDOW || height < WINDOW {
        return 1.0;
    }
    let luminance = |pixels: &[u8], x: usize, y: usize| {
        let index = (y * width + x) * 4;
        (0.2126 * f64::from(pixels[index])
            + 0.7152 * f64::from(pixels[index + 1])
            + 0.0722 * f64::from(pixels[index + 2]))
            / 255.0
    };
    let mut total = 0.0;
    let mut windows = 0usize;
    for y0 in (0..=height - WINDOW).step_by(WINDOW) {
        for x0 in (0..=width - WINDOW).step_by(WINDOW) {
            let mut mean_reference = 0.0;
            let mut mean_candidate = 0.0;
            for y in y0..y0 + WINDOW {
                for x in x0..x0 + WINDOW {
                    mean_reference += luminance(reference, x, y);
                    mean_candidate += luminance(candidate, x, y);
                }
            }
            let count = (WINDOW * WINDOW) as f64;
            mean_reference /= count;
            mean_candidate /= count;
            let mut variance_reference = 0.0;
            let mut variance_candidate = 0.0;
            let mut covariance = 0.0;
            for y in y0..y0 + WINDOW {
                for x in x0..x0 + WINDOW {
                    let reference_delta = luminance(reference, x, y) - mean_reference;
                    let candidate_delta = luminance(candidate, x, y) - mean_candidate;
                    variance_reference += reference_delta * reference_delta;
                    variance_candidate += candidate_delta * candidate_delta;
                    covariance += reference_delta * candidate_delta;
                }
            }
            variance_reference /= count;
            variance_candidate /= count;
            covariance /= count;
            total += ((2.0 * mean_reference * mean_candidate + C1) * (2.0 * covariance + C2))
                / ((mean_reference * mean_reference + mean_candidate * mean_candidate + C1)
                    * (variance_reference + variance_candidate + C2));
            windows += 1;
        }
    }
    total / windows as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_memory_accounts_for_odd_bc_blocks_and_rgba_texels() {
        assert_eq!(rgba8_mip_bytes(5, 3, 3), 72);
        assert_eq!(bc_mip_bytes(5, 3, 3), 64);
        assert_eq!(rgba8_mip_bytes(1024, 1024, 11), 5_592_404);
        assert_eq!(bc_mip_bytes(1024, 1024, 11), 1_398_128);
    }

    #[test]
    fn quality_metrics_are_exact_for_identical_pixels() {
        let pixels = vec![17u8; 8 * 8 * 4];
        let settings = TextureSettings::parse(std::iter::empty()).unwrap();
        let metrics = texture_quality(&pixels, &pixels, 8, 8, settings);
        assert_eq!(metrics["rgb_rmse"], 0.0);
        assert!(metrics["rgb_psnr_db"].is_null());
        assert_eq!(metrics["rgb_max_abs_byte"], 0);
        assert_eq!(metrics["luminance_ssim"], 1.0);
    }

    #[test]
    fn iteration_parser_is_bounded_and_order_independent() {
        let flags = vec![
            "--normal".to_string(),
            "--iterations".to_string(),
            "7".to_string(),
        ];
        let (iterations, remaining) = split_iterations(&flags).unwrap();
        assert_eq!(iterations, 7);
        assert_eq!(remaining, ["--normal"]);
        assert!(split_iterations(&["--iterations".to_string(), "0".to_string()]).is_err());
        assert!(split_iterations(&[
            "--iterations".to_string(),
            "2".to_string(),
            "--iterations".to_string(),
            "3".to_string(),
        ])
        .is_err());
    }
}
