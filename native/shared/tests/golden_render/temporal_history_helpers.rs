use super::*;

pub(crate) fn severe_pixel_fraction(reference: &[u8], candidate: &[u8]) -> f64 {
    let severe = reference
        .chunks_exact(4)
        .zip(candidate.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|channel| a[channel].abs_diff(b[channel]) > 64))
        .count();
    severe as f64 / (reference.len() / 4) as f64
}

pub(crate) fn average_rgba(frames: &[Vec<u8>]) -> Vec<u8> {
    assert!(!frames.is_empty());
    let mut sum = vec![0u32; frames[0].len()];
    for frame in frames {
        for (sum, value) in sum.iter_mut().zip(frame) {
            *sum += u32::from(*value);
        }
    }
    let count = frames.len() as u32;
    sum.into_iter()
        .map(|sum| ((sum + count / 2) / count) as u8)
        .collect()
}

pub(crate) fn evaluate_motion_recovery(label: &str, old_pose: &[u8], frames: &[Vec<u8>]) {
    let stable = average_rgba(&frames[8..]);
    let movement = calculate_diff_metrics(old_pose, &stable, W, H);
    let recovery = frames[..8]
        .iter()
        .map(|frame| calculate_diff_metrics(&stable, frame, W, H))
        .collect::<Vec<_>>();
    let severe = frames[..8]
        .iter()
        .map(|frame| severe_pixel_fraction(&stable, frame))
        .collect::<Vec<_>>();
    let trail_frames = severe
        .iter()
        .enumerate()
        .find(|(index, _)| severe[*index..].iter().all(|fraction| *fraction <= 0.005))
        .map(|(index, _)| index)
        .unwrap_or(severe.len());
    let stable_mean = frames[8..]
        .iter()
        .map(|frame| calculate_diff_metrics(&stable, frame, W, H).mean_rgb)
        .sum::<f64>()
        / (frames.len() - 8) as f64;
    eprintln!(
        "temporal-corpus {label} movement_mean={:.4} initial_mean={:.4} frame4_mean={:.4} \
         frame4_outliers={:.4}% trail_frames={trail_frames} \
         stable_flicker={stable_mean:.4}",
        movement.mean_rgb,
        recovery[0].mean_rgb,
        recovery[4].mean_rgb,
        recovery[4].outlier_pixel_fraction * 100.0,
    );
    assert!(
        trail_frames <= 4,
        "{label} left severe motion trails beyond four frames"
    );
    assert!(
        movement.mean_rgb >= 1.0 && movement.outlier_pixel_fraction >= 0.01,
        "{label} negative control did not produce visible object motion"
    );
    assert!(
        recovery[4].outlier_pixel_fraction <= 0.02,
        "{label} coherent trail covered over 2% after four frames"
    );
    assert!(
        stable_mean <= 2.0,
        "{label} did not settle to a stable jitter-cycle estimate"
    );
}

pub(crate) fn configure_taa_motion_corpus(renderer: &mut Renderer) {
    renderer.set_taa_enabled(true);
    renderer.set_render_scale(1.0);
    renderer.set_ssao_enabled(false);
    renderer.set_ssr_enabled(false);
    renderer.set_ssgi_enabled(false);
    renderer.set_bloom_enabled(false);
    renderer.set_auto_exposure(false);
    renderer.set_motion_blur_enabled(false);
    renderer.set_shadows_enabled(false);
}
