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
    width: u32,
    height: u32,
}

fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
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
) -> DiffStats {
    let n = reference.len() as f32;
    let mut sum_sq = [0f64; 3];
    let mut sum_sq_lum = 0f64;
    let mut max_abs = 0f32;
    let mut n_above = 0u64;

    for (r, c) in reference.iter().zip(candidate.iter()) {
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
    }

    let rmse_r = (sum_sq[0] / n as f64).sqrt() as f32;
    let rmse_g = (sum_sq[1] / n as f64).sqrt() as f32;
    let rmse_b = (sum_sq[2] / n as f64).sqrt() as f32;
    let rmse_luminance = (sum_sq_lum / n as f64).sqrt() as f32;

    let ssim = compute_ssim_luminance(reference, candidate, width, height);

    DiffStats {
        rmse_r,
        rmse_g,
        rmse_b,
        rmse_luminance,
        max_abs_error: max_abs,
        percent_above_tolerance: 100.0 * (n_above as f32) / n,
        ssim,
        width,
        height,
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
            let (mean_r, mean_c, var_r, var_c, cov) =
                window_luminance_stats(reference, candidate, w, x, y, WINDOW);
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
) -> (f32, f32, f32, f32, f32) {
    let mut sum_r = 0f32;
    let mut sum_c = 0f32;
    for yy in 0..size {
        for xx in 0..size {
            let i = (y0 + yy) * width + (x0 + xx);
            sum_r += luminance(reference[i]);
            sum_c += luminance(candidate[i]);
        }
    }
    let n = (size * size) as f32;
    let mean_r = sum_r / n;
    let mean_c = sum_c / n;

    let mut var_r = 0f32;
    let mut var_c = 0f32;
    let mut cov = 0f32;
    for yy in 0..size {
        for xx in 0..size {
            let i = (y0 + yy) * width + (x0 + xx);
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
        (b * 255.0) as u8,
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

// ============================================================
// CLI
// ============================================================

struct Args {
    reference_path: String,
    candidate_path: String,
    heatmap_path: Option<String>,
    composite_path: Option<String>,
    tolerance: f32,
    quiet: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut reference_path: Option<String> = None;
    let mut candidate_path: Option<String> = None;
    let mut heatmap_path: Option<String> = None;
    let mut composite_path: Option<String> = None;
    let mut tolerance: f32 = 0.02;
    let mut quiet = false;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--reference" | "-r" => reference_path = iter.next(),
            "--candidate" | "-c" => candidate_path = iter.next(),
            "--heatmap" => heatmap_path = iter.next(),
            "--composite" => composite_path = iter.next(),
            "--tolerance" => {
                tolerance = iter
                    .next()
                    .ok_or("--tolerance needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --tolerance: {e}"))?;
            }
            "--quiet" | "-q" => quiet = true,
            "-h" | "--help" => {
                println!("bloom-diff — compare two PNG images");
                println!();
                println!("  --reference PATH  ground-truth image (from bloom-reference)");
                println!("  --candidate PATH  image to compare (e.g. realtime screenshot)");
                println!("  --heatmap PATH    write per-pixel false-color diff");
                println!("  --composite PATH  write 3-up side-by-side (ref|cand|heat)");
                println!("  --tolerance F     per-pixel diff threshold for 'differs' %");
                println!("                    (default 0.02 = 2/255 on any channel)");
                println!("  --quiet           suppress stdout output");
                println!();
                println!("Exit code: 0 if max(RMSE_luminance, (1 - SSIM)) ≤ tolerance,");
                println!("           1 otherwise. Intended for use in CI.");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        reference_path: reference_path.ok_or("--reference is required")?,
        candidate_path: candidate_path.ok_or("--candidate is required")?,
        heatmap_path,
        composite_path,
        tolerance,
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
    let candidate_pixels = match load_rgb_normalized(Path::new(&args.candidate_path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error loading candidate: {e}");
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
    let candidate_img = match image::open(&args.candidate_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("error re-opening candidate: {e}");
            return ExitCode::from(1);
        }
    };

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
    let stats = compute_stats(
        &reference_pixels,
        &candidate_pixels,
        width,
        height,
        args.tolerance,
    );

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
        println!("max abs error:     {:.5}", stats.max_abs_error);
        println!(
            "% above tolerance: {:.2}%   (tolerance = {})",
            stats.percent_above_tolerance, args.tolerance
        );
        println!(
            "SSIM (luminance):  {:.5}   (1 = identical, 0 = nothing in common)",
            stats.ssim
        );
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
        if let Err(e) = write_composite(&reference_img, &candidate_img, &heatmap_buf, Path::new(path))
        {
            eprintln!("error writing composite: {e}");
            return ExitCode::from(1);
        }
        if !args.quiet {
            println!("wrote composite: {path}");
        }
    }

    // Pass/fail policy: fail if luminance RMSE OR the complement of
    // SSIM exceeds tolerance. Combining both catches cases where one
    // metric is fooled (e.g. uniform offset passes RMSE but fails
    // SSIM; speckle noise the other way around).
    let fail_threshold = args.tolerance;
    let ssim_deficit = (1.0 - stats.ssim).max(0.0);
    let fail = stats.rmse_luminance > fail_threshold || ssim_deficit > fail_threshold;
    if fail {
        if !args.quiet {
            println!();
            println!(
                "FAIL: exceeds tolerance ({} > {})",
                stats.rmse_luminance.max(ssim_deficit),
                fail_threshold
            );
        }
        ExitCode::from(1)
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
