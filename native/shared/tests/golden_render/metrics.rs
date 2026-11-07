fn luminance(px: &[u8]) -> f64 {
    (0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64) / 255.0
}

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert display-referred sRGB bytes to OKLab using the same reference
/// transform as `tools/bloom-diff`. Keeping this metric in the in-process GPU
/// goldens makes chroma regressions visible without writing temporary PNGs or
/// spawning a second process.
fn srgb_to_oklab(px: &[u8]) -> [f64; 3] {
    let r = srgb_to_linear(px[0] as f64 / 255.0);
    let g = srgb_to_linear(px[1] as f64 / 255.0);
    let b = srgb_to_linear(px[2] as f64 / 255.0);
    let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn sobel_luminance(pixels: &[u8], width: usize, x: usize, y: usize) -> f64 {
    let lum = |xx: usize, yy: usize| {
        let index = (yy * width + xx) * 4;
        luminance(&pixels[index..index + 4])
    };
    let gx = -lum(x - 1, y - 1) + lum(x + 1, y - 1) - 2.0 * lum(x - 1, y) + 2.0 * lum(x + 1, y)
        - lum(x - 1, y + 1)
        + lum(x + 1, y + 1);
    let gy = -lum(x - 1, y - 1) - 2.0 * lum(x, y - 1) - lum(x + 1, y - 1)
        + lum(x - 1, y + 1)
        + 2.0 * lum(x, y + 1)
        + lum(x + 1, y + 1);
    (gx * gx + gy * gy).sqrt() * 0.25
}

fn mean_sobel_edge_delta(expected: &[u8], actual: &[u8], width: u32, height: u32) -> f64 {
    let width = width as usize;
    let height = height as usize;
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            total += (sobel_luminance(expected, width, x, y)
                - sobel_luminance(actual, width, x, y))
            .abs();
            count += 1;
        }
    }
    total / count as f64
}

/// Single-scale luminance SSIM over non-overlapping 8x8 windows. This is
/// intentionally identical in shape to tools/bloom-diff's regression metric.
fn ssim_luminance(reference: &[u8], candidate: &[u8], width: u32, height: u32) -> f64 {
    const WINDOW: usize = 8;
    const C1: f64 = 0.0001;
    const C2: f64 = 0.0009;
    let width = width as usize;
    let height = height as usize;
    if width < WINDOW || height < WINDOW {
        return 1.0;
    }
    let mut total = 0.0;
    let mut windows = 0usize;
    for y0 in (0..=height - WINDOW).step_by(WINDOW) {
        for x0 in (0..=width - WINDOW).step_by(WINDOW) {
            let mut mean_r = 0.0;
            let mut mean_c = 0.0;
            for y in y0..y0 + WINDOW {
                for x in x0..x0 + WINDOW {
                    let i = (y * width + x) * 4;
                    mean_r += luminance(&reference[i..i + 4]);
                    mean_c += luminance(&candidate[i..i + 4]);
                }
            }
            let n = (WINDOW * WINDOW) as f64;
            mean_r /= n;
            mean_c /= n;
            let mut var_r = 0.0;
            let mut var_c = 0.0;
            let mut covariance = 0.0;
            for y in y0..y0 + WINDOW {
                for x in x0..x0 + WINDOW {
                    let i = (y * width + x) * 4;
                    let dr = luminance(&reference[i..i + 4]) - mean_r;
                    let dc = luminance(&candidate[i..i + 4]) - mean_c;
                    var_r += dr * dr;
                    var_c += dc * dc;
                    covariance += dr * dc;
                }
            }
            var_r /= n;
            var_c /= n;
            covariance /= n;
            total += ((2.0 * mean_r * mean_c + C1) * (2.0 * covariance + C2))
                / ((mean_r * mean_r + mean_c * mean_c + C1) * (var_r + var_c + C2));
            windows += 1;
        }
    }
    total / windows as f64
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DiffMetrics {
    pub(super) mean_rgba: f64,
    pub(super) mean_rgb: f64,
    /// Display-referred luminance RMSE in normalized 0..1 units.
    pub(super) rmse_luminance: f64,
    /// Mean perceptual sRGB colour distance in OKLab.
    pub(super) mean_oklab_delta: f64,
    /// Mean absolute difference between normalized Sobel edge magnitudes.
    pub(super) mean_edge_delta: f64,
    pub(super) max_diff: u8,
    pub(super) outlier_pixel_fraction: f64,
    pub(super) outlier_channel_fraction: f64,
    pub(super) ssim: f64,
}

pub(super) fn calculate_diff_metrics(
    expected: &[u8],
    actual: &[u8],
    width: u32,
    height: u32,
) -> DiffMetrics {
    assert_eq!(expected.len(), actual.len());
    assert_eq!(actual.len(), width as usize * height as usize * 4);
    let mut sum_abs = 0.0;
    let mut sum_abs_rgb = 0.0;
    let mut sum_sq_luminance = 0.0;
    let mut sum_oklab_delta = 0.0;
    let mut outlier_pixels = 0usize;
    let mut outlier_channels = 0usize;
    let mut max_diff = 0u8;
    for (actual, expected) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let mut pixel_max = 0u8;
        for channel in 0..4 {
            let diff = actual[channel].abs_diff(expected[channel]);
            sum_abs += diff as f64;
            if channel < 3 {
                sum_abs_rgb += diff as f64;
                pixel_max = pixel_max.max(diff);
            }
            if diff > 32 {
                outlier_channels += 1;
            }
            max_diff = max_diff.max(diff);
        }
        if pixel_max > 32 {
            outlier_pixels += 1;
        }
        let luminance_delta = luminance(expected) - luminance(actual);
        sum_sq_luminance += luminance_delta * luminance_delta;
        let expected_oklab = srgb_to_oklab(expected);
        let actual_oklab = srgb_to_oklab(actual);
        let lightness_delta = expected_oklab[0] - actual_oklab[0];
        let a_delta = expected_oklab[1] - actual_oklab[1];
        let b_delta = expected_oklab[2] - actual_oklab[2];
        sum_oklab_delta +=
            (lightness_delta * lightness_delta + a_delta * a_delta + b_delta * b_delta).sqrt();
    }
    let pixel_count = width as f64 * height as f64;
    DiffMetrics {
        mean_rgba: sum_abs / actual.len() as f64,
        mean_rgb: sum_abs_rgb / (pixel_count * 3.0),
        rmse_luminance: (sum_sq_luminance / pixel_count).sqrt(),
        mean_oklab_delta: sum_oklab_delta / pixel_count,
        mean_edge_delta: mean_sobel_edge_delta(expected, actual, width, height),
        max_diff,
        outlier_pixel_fraction: outlier_pixels as f64 / pixel_count,
        outlier_channel_fraction: outlier_channels as f64 / actual.len() as f64,
        ssim: ssim_luminance(expected, actual, width, height),
    }
}

pub(super) fn select_outlier_gate(metrics: DiffMetrics, is_pt_oracle: bool) -> (&'static str, f64) {
    if is_pt_oracle {
        ("pixel", metrics.outlier_pixel_fraction)
    } else {
        ("channel", metrics.outlier_channel_fraction)
    }
}

#[test]
fn diff_metrics_keep_raster_gate_and_detect_coherent_pt_regions() {
    let expected = [0u8, 0, 0, 255, 0, 0, 0, 255];
    let actual = [64u8, 0, 0, 255, 0, 0, 0, 255];
    let metrics = calculate_diff_metrics(&expected, &actual, 2, 1);
    assert_eq!(select_outlier_gate(metrics, false), ("channel", 0.125));
    assert_eq!(select_outlier_gate(metrics, true), ("pixel", 0.5));
    assert_eq!(metrics.max_diff, 64);
    assert!(metrics.rmse_luminance > 0.0);
    assert!(metrics.mean_oklab_delta > 0.0);
    assert_eq!(metrics.mean_edge_delta, 0.0);
}

#[test]
fn perceptual_diff_metrics_detect_edge_and_chroma_regressions() {
    let mut expected = vec![0u8; 5 * 5 * 4];
    let mut actual = vec![0u8; 5 * 5 * 4];
    for pixel in expected.chunks_exact_mut(4) {
        pixel[3] = 255;
    }
    for pixel in actual.chunks_exact_mut(4) {
        pixel[3] = 255;
    }
    let center = (2 * 5 + 2) * 4;
    expected[center..center + 4].copy_from_slice(&[255, 0, 0, 255]);
    actual[center..center + 4].copy_from_slice(&[0, 255, 0, 255]);

    let metrics = calculate_diff_metrics(&expected, &actual, 5, 5);
    assert!(metrics.rmse_luminance > 0.0);
    assert!(metrics.mean_oklab_delta > 0.0);
    assert!(metrics.mean_edge_delta > 0.0);
}
