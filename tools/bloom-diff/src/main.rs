//! bloom-diff — pixel-accurate image comparison.
//!
//! Purpose: given a ground-truth image (from bloom-reference) and a
//! candidate image (from the Bloom realtime renderer), quantify how
//! different they are and produce a visual diff.
//!
//! This is the piece that lets the reference path tracer actually do
//! its job: every renderer change is a question of "does this PR make
//! the realtime image closer to or farther from the reference?". Now
//! that question has a numerical answer.
//!
//! Output:
//!   - Console: per-channel RMSE, max error, SSIM score, percentage of
//!     pixels exceeding the tolerance.
//!   - --heatmap PATH (optional): per-pixel difference magnitude as a
//!     false-color image (black = identical, bright = big diff).
//!   - --composite PATH (optional): side-by-side 2-up (reference +
//!     candidate + heatmap) for quick eyeballing.
//!   - Exit code: 0 if mean RMSE ≤ --tolerance, 1 otherwise. Lets CI
//!     fail a build when a change regresses visual correctness.
//!
//! Usage:
//!   bloom-diff --reference ref.png --candidate shot.png
//!              [--heatmap diff.png] [--composite side.png]
//!              [--tolerance 0.02]

use image::{ImageBuffer, Rgb, RgbImage};
use std::env;
use std::path::Path;
use std::process::ExitCode;

// ============================================================
// Metrics
// ============================================================

#[derive(Debug, Clone, Copy)]
struct DiffStats {
    /// Per-channel root-mean-squared error (0–1 linear scale).
    rmse_r: f32,
    rmse_g: f32,
    rmse_b: f32,
    /// Luminance RMSE — often the most meaningful single number since
    /// humans are more sensitive to brightness than chroma shifts.
    rmse_luminance: f32,
    /// Max absolute per-channel difference across the whole image.
    max_abs_error: f32,
    /// Percentage of pixels where the absolute-difference magnitude
    /// exceeds the supplied tolerance.
    percent_above_tolerance: f32,
    /// Structural-similarity index over the luminance channel. 1.0 is
    /// identical; 0.0 is "nothing in common". Tends to correlate well
    /// with perceptual similarity, unlike raw RMSE.
    ssim: f32,
    /// Mean Euclidean colour distance in OKLab. OKLab is approximately
    /// perceptually uniform, so this catches chroma/hue errors that a
    /// luminance-only metric can miss. It is intentionally named rather
    /// than presented as FLIP: FLIP includes a display model that we cannot
    /// infer from an arbitrary PNG.
    mean_oklab_delta: f32,
    /// Mean absolute difference between Sobel edge magnitudes. This gives
    /// small, displaced shadow and geometry edges a targeted signal even
    /// when they occupy too little of the frame to move global SSIM.
    mean_edge_delta: f32,
    /// Number of pixels selected by the optional mask.
    selected_pixels: u64,
    width: u32,
    height: u32,
}

fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert an sRGB triple to OKLab using Björn Ottosson's reference
/// matrices. PNG bytes are display-referred sRGB, so conversion to linear
/// light is part of the transform.
fn srgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let r = srgb_to_linear(rgb[0]);
    let g = srgb_to_linear(rgb[1]);
    let b = srgb_to_linear(rgb[2]);
    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let l = l.cbrt();
    let m = m.cbrt();
    let s = s.cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn mask_includes(mask: Option<&[u8]>, index: usize) -> bool {
    mask.map(|m| m[index] != 0).unwrap_or(true)
}

/// Load an RGB image as normalized linear f32 triples. We operate in
/// gamma-encoded (sRGB byte) space rather than re-linearizing — the
/// reference renderer already wrote sRGB-encoded output, so a byte-
/// for-byte comparison is what we want. Future work could add an
/// optional --linear flag for tonemapper-bypassing comparisons.
fn load_rgb_normalized(path: &Path) -> Result<Vec<[f32; 3]>, String> {
    let img = image::open(path).map_err(|e| format!("open {:?}: {e}", path))?;
    let rgb = img.to_rgb8();
    Ok(rgb
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect())
}

fn compute_stats(
    reference: &[[f32; 3]],
    candidate: &[[f32; 3]],
    width: u32,
    height: u32,
    tolerance: f32,
    mask: Option<&[u8]>,
) -> DiffStats {
    let mut sum_sq = [0f64; 3];
    let mut sum_sq_lum = 0f64;
    let mut sum_oklab_delta = 0f64;
    let mut max_abs = 0f32;
    let mut n_above = 0u64;
    let mut selected = 0u64;

    for (i, (r, c)) in reference.iter().zip(candidate.iter()).enumerate() {
        if !mask_includes(mask, i) {
            continue;
        }
        selected += 1;
        let dr = r[0] - c[0];
        let dg = r[1] - c[1];
        let db = r[2] - c[2];
        sum_sq[0] += (dr as f64) * (dr as f64);
        sum_sq[1] += (dg as f64) * (dg as f64);
        sum_sq[2] += (db as f64) * (db as f64);

        let lum_r = luminance(*r);
        let lum_c = luminance(*c);
        let dl = lum_r - lum_c;
        sum_sq_lum += (dl as f64) * (dl as f64);

        let mag = dr.abs().max(dg.abs()).max(db.abs());
        if mag > max_abs {
            max_abs = mag;
        }
        if mag > tolerance {
            n_above += 1;
        }
        let lab_r = srgb_to_oklab(*r);
        let lab_c = srgb_to_oklab(*c);
        let dl = lab_r[0] - lab_c[0];
        let da = lab_r[1] - lab_c[1];
        let db = lab_r[2] - lab_c[2];
        sum_oklab_delta += ((dl * dl + da * da + db * db).sqrt()) as f64;
    }

    let n = selected.max(1) as f64;

    let rmse_r = (sum_sq[0] / n as f64).sqrt() as f32;
    let rmse_g = (sum_sq[1] / n as f64).sqrt() as f32;
    let rmse_b = (sum_sq[2] / n as f64).sqrt() as f32;
    let rmse_luminance = (sum_sq_lum / n as f64).sqrt() as f32;

    let ssim = compute_ssim_luminance(reference, candidate, width, height, mask);
    let mean_edge_delta = compute_edge_delta(reference, candidate, width, height, mask);

    DiffStats {
        rmse_r,
        rmse_g,
        rmse_b,
        rmse_luminance,
        max_abs_error: max_abs,
        percent_above_tolerance: 100.0 * (n_above as f32) / n as f32,
        ssim,
        mean_oklab_delta: (sum_oklab_delta / n) as f32,
        mean_edge_delta,
        selected_pixels: selected,
        width,
        height,
    }
}

fn sobel_luminance(pixels: &[[f32; 3]], width: usize, x: usize, y: usize) -> f32 {
    let lum = |xx: usize, yy: usize| luminance(pixels[yy * width + xx]);
    let gx = -lum(x - 1, y - 1) + lum(x + 1, y - 1) - 2.0 * lum(x - 1, y) + 2.0 * lum(x + 1, y)
        - lum(x - 1, y + 1)
        + lum(x + 1, y + 1);
    let gy = -lum(x - 1, y - 1) - 2.0 * lum(x, y - 1) - lum(x + 1, y - 1)
        + lum(x - 1, y + 1)
        + 2.0 * lum(x, y + 1)
        + lum(x + 1, y + 1);
    (gx * gx + gy * gy).sqrt() * 0.25
}

fn compute_edge_delta(
    reference: &[[f32; 3]],
    candidate: &[[f32; 3]],
    width: u32,
    height: u32,
    mask: Option<&[u8]>,
) -> f32 {
    let width = width as usize;
    let height = height as usize;
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut total = 0.0f64;
    let mut count = 0u64;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = y * width + x;
            if !mask_includes(mask, i) {
                continue;
            }
            total += (sobel_luminance(reference, width, x, y)
                - sobel_luminance(candidate, width, x, y))
            .abs() as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (total / count as f64) as f32
    }
}

/// SSIM over the luminance channel with a single-scale 8×8 window.
/// Not as good as MS-SSIM but fast and plenty accurate for our
/// "did this PR move the image closer to truth" check.
fn compute_ssim_luminance(
    reference: &[[f32; 3]],
    candidate: &[[f32; 3]],
    width: u32,
    height: u32,
    mask: Option<&[u8]>,
) -> f32 {
    const WINDOW: usize = 8;
    // SSIM's stability constants (from the original Wang et al. paper,
    // scaled to the 0..1 luminance range we use).
    const K1: f32 = 0.01;
    const K2: f32 = 0.03;
    const L: f32 = 1.0; // dynamic range for normalized images
    let c1 = (K1 * L) * (K1 * L);
    let c2 = (K2 * L) * (K2 * L);

    let w = width as usize;
    let h = height as usize;
    if w < WINDOW || h < WINDOW {
        return 1.0; // too small to analyze meaningfully; treat as identical
    }

    let mut sum = 0f64;
    let mut count = 0u64;

    // Non-overlapping 8×8 windows. Sliding windows would be more
    // accurate but 8× slower; for regression testing the blocky
    // version is plenty — we care about directional signal, not a
    // perfect Wang-et-al reproduction.
    let mut y = 0usize;
    while y + WINDOW <= h {
        let mut x = 0usize;
        while x + WINDOW <= w {
            let selected = (y..y + WINDOW)
                .flat_map(|yy| (x..x + WINDOW).map(move |xx| yy * w + xx))
                .filter(|&i| mask_includes(mask, i))
                .count();
            if selected == 0 {
                x += WINDOW;
                continue;
            }
            let (mean_r, mean_c, var_r, var_c, cov) =
                window_luminance_stats(reference, candidate, w, x, y, WINDOW, mask);
            let num = (2.0 * mean_r * mean_c + c1) * (2.0 * cov + c2);
            let den = (mean_r * mean_r + mean_c * mean_c + c1) * (var_r + var_c + c2);
            sum += (num / den) as f64;
            count += 1;
            x += WINDOW;
        }
        y += WINDOW;
    }

    if count == 0 {
        1.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Mean and variance of luminance in an N×N window plus the
/// covariance between reference and candidate. Returned as f32s.
fn window_luminance_stats(
    reference: &[[f32; 3]],
    candidate: &[[f32; 3]],
    width: usize,
    x0: usize,
    y0: usize,
    size: usize,
    mask: Option<&[u8]>,
) -> (f32, f32, f32, f32, f32) {
    let mut sum_r = 0f32;
    let mut sum_c = 0f32;
    let mut selected = 0usize;
    for yy in 0..size {
        for xx in 0..size {
            let i = (y0 + yy) * width + (x0 + xx);
            if !mask_includes(mask, i) {
                continue;
            }
            sum_r += luminance(reference[i]);
            sum_c += luminance(candidate[i]);
            selected += 1;
        }
    }
    let n = selected.max(1) as f32;
    let mean_r = sum_r / n;
    let mean_c = sum_c / n;

    let mut var_r = 0f32;
    let mut var_c = 0f32;
    let mut cov = 0f32;
    for yy in 0..size {
        for xx in 0..size {
            let i = (y0 + yy) * width + (x0 + xx);
            if !mask_includes(mask, i) {
                continue;
            }
            let dr = luminance(reference[i]) - mean_r;
            let dc = luminance(candidate[i]) - mean_c;
            var_r += dr * dr;
            var_c += dc * dc;
            cov += dr * dc;
        }
    }
    var_r /= n;
    var_c /= n;
    cov /= n;
    (mean_r, mean_c, var_r, var_c, cov)
}

// ============================================================
// Heatmap + composite output
// ============================================================

/// Map a difference magnitude (0..1) to a false-color hot palette so
/// small errors stay dark and big errors scream. Goes black → red →
/// yellow → white, amplified so typical-tolerance errors (1-2%) are
/// visible without manually setting a gain.
fn heatmap_color(magnitude: f32) -> Rgb<u8> {
    let m = (magnitude * 16.0).clamp(0.0, 1.0);
    let r = (m * 3.0).clamp(0.0, 1.0);
    let g = ((m - 0.33) * 3.0).clamp(0.0, 1.0);
    let b = ((m - 0.66) * 3.0).clamp(0.0, 1.0);
    Rgb([
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8
    ])
}

fn write_heatmap(
    reference: &[[f32; 3]],
    candidate: &[[f32; 3]],
    width: u32,
    height: u32,
    path: &Path,
) -> Result<(), String> {
    let mut img: RgbImage = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let dr = (reference[i][0] - candidate[i][0]).abs();
            let dg = (reference[i][1] - candidate[i][1]).abs();
            let db = (reference[i][2] - candidate[i][2]).abs();
            let mag = dr.max(dg).max(db);
            img.put_pixel(x, y, heatmap_color(mag));
        }
    }
    img.save(path).map_err(|e| format!("save {:?}: {e}", path))
}

fn write_composite(
    reference_img: &RgbImage,
    candidate_img: &RgbImage,
    heatmap: &RgbImage,
    path: &Path,
) -> Result<(), String> {
    let w = reference_img.width();
    let h = reference_img.height();
    // Three panels side by side with a 1-pixel divider.
    let pad = 1u32;
    let total_w = w * 3 + pad * 2;
    let mut out: RgbImage = ImageBuffer::from_pixel(total_w, h, Rgb([80, 80, 80]));

    for y in 0..h {
        for x in 0..w {
            out.put_pixel(x, y, *reference_img.get_pixel(x, y));
            out.put_pixel(w + pad + x, y, *candidate_img.get_pixel(x, y));
            out.put_pixel(2 * (w + pad) + x, y, *heatmap.get_pixel(x, y));
        }
    }
    out.save(path).map_err(|e| format!("save {:?}: {e}", path))
}

/// Test-only negative controls for the quality harness. These are deliberately
/// simple, deterministic image-space stand-ins for common renderer failures;
/// they prove that the configured metric mix and thresholds reject the class
/// of error without adding fault branches to production shaders.

fn apply_seeded_fault(image: &RgbImage, fault: &str) -> Result<RgbImage, String> {
    let (width, height) = image.dimensions();
    let mut out = image.clone();
    match fault {
        "brdf-energy" => {
            for pixel in out.pixels_mut() {
                for channel in & mut pixel.0 {
                    *channel = ((*channel as f32 * 1.18).round() as u16).min(255) as u8;
                }
            }
        }
        "shadow-placement" => {
            let shift = (width / 32).clamp(2, 16);
    for y in 0..height {
        for x in 0..width {
            let sx = x.saturating_sub(shift);
                    out.put_pixel(x,y, *image.get_pixel(sx, y));
                }
            }
        }
        "gi-leakage" => {
            for pixel in out.pixels_mut() {
                let luma =
                    0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
                if luma < 128.0 {
                    pixel[0] = pixel[0].saturating_add(24);
                    pixel[1] = pixel[1].saturating_add(38);
                    pixel[2] = pixel[2].saturating_add(20);
                }
            }
        }
        "motion-history" => {
            let shift = ( width / 48).clamp(2, 12);
            for y in 0..height {
                for x in 0..width {
                    let history = image.get_pixel(x.saturating_sub(shift), y);
                    let current = image.get_pixel(x, y);
                    out.put_pixel(
                        x,
                        y,
                        Rgb([
                            ((current[0] as u16 * 3 + history[0] as u16) / 4) as u8,
                            ((current[1] as u16 * 3 + history[1] as u16) / 4) as u8,
                            ((current[2] as u16 * 3 + history[2] as u16) / 4) as u8,
                        ]),
                    );
                }
            }
        }
        "texture-orientation" => {
            for y in 0..height {
                for x in 0..width {
                    out.put_pixel(x, y, *image.get_pixel(x, height - 1 - y));
                }
            }
        }
        other => {
            return Err(format!(
                "unknown seeded fault {other:?}; expected brdf-energy, shadow-placement, \
                 gi-leakage, motion-history, or texture-orientation"
            ));
        }
    }
    Ok(out)
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn write_metrics_json(
    path: &Path,
    args: &Args,
    stats: DiffStats,
    passed: bool,
    failures: &[String],
) -> Result<(), String> {
    let optional = |value: Option<f32>| {
        value
            .map(|v| format!("{v:.9}"))
            .unwrap_or_else(|| "null".to_owned())
    };
    let failure_json = failures
        .iter()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .collect::<Vec<_>>()
        .join(", ");
    let mask_json = args
        .mask_path
        .as_ref()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .unwrap_or_else(|| "null".to_owned());
    let fault_json = args
        .seed_fault
        .as_ref()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .unwrap_or_else(|| "null".to_owned());
    let json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"bloom-diff-result-v2\",\n",
            "  \"reference\": \"{}\",\n",
            "  \"candidate\": \"{}\",\n",
            "  \"mask\": {},\n",
            "  \"seeded_fault\": {},\n",
            "  \"width\": {},\n",
            "  \"height\": {},\n",
            "  \"selected_pixels\": {},\n",
            "  \"metrics\": {{\n",
            "    \"rmse_luminance\": {:.9},\n",
            "    \"rmse_r\": {:.9},\n",
            "    \"rmse_g\": {:.9},\n",
            "    \"rmse_b\": {:.9},\n",
            "    \"max_abs_error\": {:.9},\n",
            "    \"percent_above_tolerance\": {:.9},\n",
            "    \"ssim_luminance\": {:.9},\n",
            "    \"mean_oklab_delta\": {:.9},\n",
            "    \"mean_edge_delta\": {:.9}\n",
            "  }},\n",
            "  \"thresholds\": {{\n",
            "    \"pixel_tolerance\": {:.9},\n",
            "    \"min_ssim\": {},\n",
            "    \"max_rmse\": {},\n",
            "    \"max_oklab_delta\": {},\n",
            "    \"max_edge_delta\": {}\n",
            "  }},\n",
            "  \"report_only\": {},\n",
            "  \"passed\": {},\n",
            "  \"failures\": [{}]\n",
            "}}\n"
        ),
        json_escape(&args.reference_path),
        json_escape(&args.candidate_path),
        mask_json,
        fault_json,
        stats.width,
        stats.height,
        stats.selected_pixels,
        stats.rmse_luminance,
        stats.rmse_r,
        stats.rmse_g,
        stats.rmse_b,
        stats.max_abs_error,
        stats.percent_above_tolerance,
        stats.ssim,
        stats.mean_oklab_delta,
        stats.mean_edge_delta,
        args.tolerance,
        optional(args.min_ssim),
        optional(args.max_rmse),
        optional(args.max_oklab_delta),
        optional(args.max_edge_delta),
        args.report_only,
        passed,
        failure_json,
    );
    std::fs::write(path, json).map_err(|e| format!("write {:?}: {e}", path))
}

// ============================================================
// CLI
// ============================================================

struct Args {
    reference_path: String,
    candidate_path: String,
    mask_path: Option<String>,
    heatmap_path: Option<String>,
    composite_path: Option<String>,
    metrics_json_path: Option<String>,
    tolerance: f32,
    min_ssim: Option<f32>,
    max_rmse: Option<f32>,
    max_oklab_delta: Option<f32>,
    max_edge_delta: Option<f32>,
    seed_fault: Option<String>,
    fault_output_path: Option<String>,
    report_only: bool,
    quiet: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut reference_path: Option<String> = None;
    let mut candidate_path: Option<String> = None;
    let mut mask_path: Option<String> = None;
    let mut heatmap_path: Option<String> = None;
    let mut composite_path: Option<String> = None;
    let mut metrics_json_path: Option<String> = None;
    let mut tolerance: f32 = 0.02;
    let mut min_ssim: Option<f32> = None;
    let mut max_rmse: Option<f32> = None;
    let mut max_oklab_delta: Option<f32> = None;
    let mut max_edge_delta: Option<f32> = None;
    let mut seed_fault: Option<String> = None;
    let mut fault_output_path: Option<String> = None;
    let mut report_only = false;
    let mut quiet = false;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--reference" | "-r" => reference_path = iter.next(),
            "--candidate" | "-c" => candidate_path = iter.next(),
            "--mask" => mask_path = iter.next(),
            "--heatmap" => heatmap_path = iter.next(),
            "--composite" => composite_path = iter.next(),
            "--metrics-json" => metrics_json_path = iter.next(),
            "--tolerance" => {
                tolerance = iter
                    .next()
                    .ok_or("--tolerance needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --tolerance: {e}"))?;
            }
            "--min-ssim" => {
                min_ssim = Some(
                    iter.next()
                        .ok_or("--min-ssim needs a value")?
                        .parse()
                        .map_err(|e| format!("invalid --min-ssim: {e}"))?,
                );
            }
            "--max-rmse" => {
                max_rmse = Some(
                    iter.next()
                        .ok_or("--max-rmse needs a value")?
                        .parse()
                        .map_err(|e| format!("invalid --max-rmse: {e}"))?,
                );
            }
            "--max-oklab-delta" => {
                max_oklab_delta = Some(
                    iter.next()
                        .ok_or("--max-oklab-delta needs a value")?
                        .parse()
                        .map_err(|e| format!("invalid --max-oklab-delta: {e}"))?,
                );
            }
            "--max-edge-delta" => {
                max_edge_delta = Some(
                    iter.next()
                        .ok_or("--max-edge-delta needs a value")?
                        .parse()
                        .map_err(|e| format!("invalid --max-edge-delta: {e}"))?,
                );
            }
            "--seed-fault" => seed_fault = iter.next(),
            "--fault-output" => fault_output_path = iter.next(),
            "--report-only" => report_only = true,
            "--quiet" | "-q" => quiet = true,
            "-h" | "--help" => {
                println!("bloom-diff — compare two PNG images");
                println!();
                println!("  --reference PATH  ground-truth image (from bloom-reference)");
                println!("  --candidate PATH  image to compare (e.g. realtime screenshot)");
                println!("  --mask PATH       optional non-black inclusion mask");
                println!("  --heatmap PATH    write per-pixel false-color diff");
                println!("  --composite PATH  write 3-up side-by-side (ref|cand|heat)");
                println!("  --metrics-json P  write stable machine-readable metrics");
                println!("  --tolerance F     per-pixel diff threshold for 'differs' %");
                println!("                    (default 0.02 = 2/255 on any channel)");
                println!("  --min-ssim F      explicit SSIM hard gate");
                println!("  --max-rmse F      explicit luminance-RMSE hard gate");
                println!("  --max-oklab-delta F  perceptual colour hard gate");
                println!("  --max-edge-delta F   Sobel edge hard gate");
                println!("  --seed-fault NAME  test-only brdf-energy|shadow-placement|");
                println!("                     gi-leakage|motion-history|texture-orientation");
                println!("  --fault-output P   write the intentionally corrupted candidate");
                println!("  --report-only     produce evidence but always exit zero");
                println!("  --quiet           suppress stdout output");
                println!();
                println!("Exit code:0 if max(RMSE_luminance, (1 - SSIM)) ≤ tolerance,");
                println!("           1 otherwise. Intended for use in CI.");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        reference_path: reference_path.ok_or("--reference is required")?,
        candidate_path: candidate_path.ok_or("- - candidate is required")?,
        mask_path,
        heatmap_path,
        composite_path,
        metrics_json_path,
        tolerance,
        min_ssim,
        max_rmse,
        max_oklab_delta,
        max_edge_delta,
        seed_fault,
        fault_output_path,
        report_only,
        quiet,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let reference_pixels = match load_rgb_normalized(Path::new(&args.reference_path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error loading reference: {e}");
            return ExitCode::from(1);
        }
    };
    // We need the dimensions explicitly for SSIM windowing + heatmap
    // output — load the raw images once more (cheap; we already have
    // them in memory via the decoder).
    let reference_img = match image::open(&args.reference_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("error re-opening reference: {e}");
            return ExitCode::from(1);
        }
    };
    let mut candidate_img = match image::open(&args.candidate_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("error re-opening candidate: {e}");
            return ExitCode::from(1);
        }
    };
    if let Some(fault) = &args.seed_fault {
        candidate_img = match apply_seeded_fault(&candidate_img, fault) {
            Ok(image) => image,
            Err(e) => {
                eprintln!("error applying seeded fault: {e}");
                return ExitCode::from(2);
            }
        };
        if let Some(path) = &args.fault_output_path {
            if let Some(parent) = Path::new(path).parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("error creating fault-output directory {:?}: {e}", parent);
                    return ExitCode::from(1);
                }
            }
            if let Err(e) = candidate_img.save(path) {
                eprintln!("error writing fault output {path}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    let candidate_pixels: Vec<[f32; 3]> = candidate_img
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    if reference_img.dimensions() != candidate_img.dimensions() {
        eprintln!(
            "error: image dimensions mismatch — reference {}x{}, candidate {}x{}",
            reference_img.width(),
            reference_img.height(),
            candidate_img.width(),
            candidate_img.height()
        );
        return ExitCode::from(1);
    }
    if reference_pixels.len() != candidate_pixels.len() {
        eprintln!("error: pixel count mismatch despite matching dimensions?");
        return ExitCode::from(1);
    }

    let (width, height) = reference_img.dimensions();
    let mask = if let Some(path) = &args.mask_path {
        let image = match image::open(path) {
            Ok(image) => image.to_luma8(),
            Err(e) => {
                eprintln!("error loading mask {path}: {e}");
                return ExitCode::from(1);
            }
        };
        if image.dimensions() != (width, height) {
            eprintln!(
                "error: mask dimensions mismatch — expected {}x{}, got {}x{}",
                width,
                height,
                image.width(),
                image.height()
            );
            return ExitCode::from(1);
        }
        Some(image.into_raw())
    } else {
        None
    };
    let stats = compute_stats(
        &reference_pixels,
        &candidate_pixels,
        width,
        height,
        args.tolerance,
        mask.as_deref(),
    );
    if stats.selected_pixels == 0 {
        eprintln!("error: mask selects zero pixels");
        return ExitCode::from(1);
    }

    if !args.quiet {
        println!("reference: {} ({}×{})", args.reference_path, width, height);
        println!("candidate: {}", args.candidate_path);
        println!();
        println!(
            "RMSE (luminance):  {:.5}   (0 = identical, 1 = max)",
            stats.rmse_luminance
        );
        println!(
            "RMSE (R/G/B):      {:.5} / {:.5} / {:.5}",
            stats.rmse_r, stats.rmse_g, stats.rmse_b
        );
        println!("maxabs error:     {:.5}", stats.max_abs_error);
        println!(
            "% above tolerance: {:.2}%   (tolerance = {})",
            stats.percent_above_tolerance, args.tolerance
        );
        println!(
            "SSIM (luminance):  {:.5}   (1 = identical, 0 = nothing in common)",
            stats.ssim
        );
        println!("OKLab mean delta:  {:.5}", stats.mean_oklab_delta);
        println!("edge mean delta:   {:.5}", stats.mean_edge_delta);
        if args.mask_path.is_some() {
            println!("selected pixels:   {}", stats.selected_pixels);
        }
    }

    if let Some(path) = &args.heatmap_path {
        match write_heatmap(
            &reference_pixels,
            &candidate_pixels,
            width,
            height,
            Path::new(path),
        ) {
            Ok(()) => {
                if !args.quiet {
                    println!("wrote heatmap: {path}");
                }
            }
            Err(e) => {
                eprintln!("error writing heatmap: {e}");
                return ExitCode::from(1);
            }
        }
    }

    if let Some(path) = &args.composite_path {
        let heatmap_buf = make_heatmap_buffer(&reference_pixels, &candidate_pixels, width, height);
        if let Err(e) = write_composite(
            &reference_img,
            &candidate_img,
            &heatmap_buf,
            Path::new(path),
        ) {
            eprintln!("error writing composite: {e}");
            return ExitCode::from(1);
        }
        if !args.quiet {
            println!("wrote composite: {path}");
        }
    }

    let has_explicit_thresholds = args.min_ssim.is_some()
        || args.max_rmse.is_some()
        || args.max_oklab_delta.is_some()
        || args.max_edge_delta.is_some();
    let mut failures = Vec::new();
    if let Some(limit) = args.min_ssim {
        if stats.ssim < limit {
            failures.push(format!("ssim {:.9} < {:.9}", stats.ssim, limit));
        }
    }
    if let Some(limit) = args.max_rmse {
        if stats.rmse_luminance > limit {
            failures.push(format!(
                "rmse_luminance {:.9} > {:.9}",
                stats.rmse_luminance, limit
            ));
        }
    }
    if let Some(limit) = args.max_oklab_delta {
        if stats.mean_oklab_delta > limit {
            failures.push(format!(
                "mean_oklab_delta {:.9} > {:.9}",
                stats.mean_oklab_delta, limit
            ));
        }
    }
    if let Some(limit) = args.max_edge_delta {
        if stats.mean_edge_delta > limit {
            failures.push(format!(
                "mean_edge_delta {:.9} > {:.9}",
                stats.mean_edge_delta, limit
            ));
        }
    }
    if !has_explicit_thresholds {
        // Backwards-compatible legacy policy.
        let ssim_deficit = (1.0 - stats.ssim).max(0.0);
        if stats.rmse_luminance > args.tolerance || ssim_deficit > args.tolerance {
            failures.push(format!(
                "legacy combined error {:.9} > {:.9}",
                stats.rmse_luminance.max(ssim_deficit),
                args.tolerance
            ));
        }
    }
    let passed = failures.is_empty();
    if let Some(path) = &args.metrics_json_path {
        if let Some(parent) = Path::new(path).parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("error creating metrics directory {:?}: {e}", parent);
                return ExitCode::from(1);
            }
        }
        if let Err(e) = write_metrics_json(Path::new(path), &args, stats, passed, &failures) {
            eprintln!("error writing metrics JSON: {e}");
            return ExitCode::from(1);
        }
        if !args.quiet {
            println!("wrote metrics: {path}");
        }
    }
    let fail = !passed;
    if fail {
        if !args.quiet {
            println!();
            println!("FAIL: {}", failures.join("; "));
        }
        if args.report_only {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        }
    } else {
        if !args.quiet {
            println!();
            println!("PASS: within tolerance");
        }
        ExitCode::SUCCESS
    }
}

/// In-memory heatmap used for the composite output — same algorithm
/// as `write_heatmap` but returns a buffer instead of writing it.
fn make_heatmap_buffer(
    reference: &[[f32; 3]],
    candidate: &[[f32; 3]],
    width: u32,
    height: u32,
) -> RgbImage {
    let mut img: RgbImage = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let dr = (reference[i][0] - candidate[i][0]).abs();
            let dg = (reference[i][1] - candidate[i][1]).abs();
            let db = (reference[i][2] - candidate[i][2]).abs();
            let mag = dr.max(dg).max(db);
            img.put_pixel(x, y, heatmap_color(mag));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(width: u32, height: u32) -> RgbImage {
        ImageBuffer::from_fn(width, height, |x, y| {
            if ((x / 4) + (y / 4)) % 2 == 0 {
                Rgb([32, 64, 96])
            } else {
                Rgb([224, 192, 128])
            }
        })
    }

    fn normalized(image: &RgbImage) -> Vec<[f32; 3]> {
        image
            .pixels()
            .map(|p| {
                [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect()
    }

    #[test]
    fn identical_images_are_exact_under_every_metric() {
        let image = checkerboard(32, 32);
        let pixels = normalized(&image);
        let stats = compute_stats(&pixels, &pixels, 32, 32, 0.02, None);
        assert_eq!(stats.rmse_luminance, 0.0);
        assert_eq!(stats.mean_oklab_delta, 0.0);
        assert_eq!(stats.mean_edge_delta, 0.0);
        assert_eq!(stats.percent_above_tolerance, 0.0);
        assert!((stats.ssim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mask_excludes_faults_outside_the_target_region() {
        let reference = checkerboard(16, 16);
        let mut candidate = reference.clone();
        candidate.put_pixel(15, 15, Rgb([255, 0, 255]));
        let mut mask = vec![0u8; 16 * 16];
        mask[0] = 255;
        let stats = compute_stats(
            &normalized(&reference),
            &normalized(&candidate),
            16,
            16,
            0.02,
            Some(&mask),
        );
        assert_eq!(stats.selected_pixels, 1);
        assert_eq!(stats.rmse_luminance, 0.0);
    }

    #[test]
    fn every_quality_negative_control_changes_a_signal() {
        let reference = checkerboard(64, 64);
        let reference_pixels = normalized(&reference);
        for fault in [
            "brdf-energy",
            "shadow-placement",
            "gi-leakage",
            "motion-history",
            "texture-orientation",
        ] {
            let candidate = apply_seeded_fault(&reference, fault).unwrap();
            let stats = compute_stats(
                &reference_pixels,
                &normalized(&candidate),
                64,
                64,
                0.02,
                None,
            );
            assert!(
                stats.rmse_luminance > 0.001
                    || stats.mean_oklab_delta > 0.001
                    || stats.mean_edge_delta > 0.001
                    || stats.ssim < 0.999,
                "{fault} did not move any regression signal: {stats:?}"
            );
        }
    }
}
