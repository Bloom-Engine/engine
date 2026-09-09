//! Common temporal-history invalidation and camera-cut ownership.

use super::*;

/// Native rendering already shades every display pixel, so a full-pixel
/// Halton aperture behaves like an unnecessarily broad box filter after
/// convergence. A half-pixel aperture retains sub-pixel edge coverage while
/// tracking the 2x supersampled reference more closely than either the former
/// 0.75-pixel aperture or a narrower 0.25-pixel candidate. Fractional
/// reconstruction still needs the full footprint to cover display samples
/// that were not shaded in the current frame.
pub(super) fn taa_jitter_spread(render_scale: f32) -> f32 {
    if render_scale >= 0.999 {
        0.5
    } else {
        1.0
    }
}

/// Reapply the current frame's projection jitter to the previous unjittered
/// projection. Both current and previous clip positions then contain the same
/// sampling offset, so the velocity target represents scene/camera motion
/// only rather than the Halton phase delta.
pub(super) fn velocity_reference_projection(
    mut previous_projection_unjittered: [[f32; 4]; 4],
    current_jitter_ndc: [f32; 2],
) -> [[f32; 4]; 4] {
    previous_projection_unjittered[2][0] += current_jitter_ndc[0];
    previous_projection_unjittered[2][1] += current_jitter_ndc[1];
    previous_projection_unjittered
}

impl Renderer {
    /// Invalidate every temporal consumer before a deliberate camera cut,
    /// teleport, FOV discontinuity, or world load. Call before the next
    /// `begin_mode_3d`; existing GPU allocations are retained.
    pub fn reset_temporal_history(&mut self) {
        self.taa_current_idx = 0;
        self.taa_frame_index = 0;
        self.taa_history_valid = false;
        self.taa_history_written = false;
        self.ssao_history_idx = 0;
        self.ssao_history_frame = 0;
        self.ssr_history_idx = 0;
        self.ssr_history_valid = false;
        self.probe_history_idx = 0;
        self.probe_frame_index = 0;
        self.probe_history_valid = false;
        self.exposure_current_idx = 0;
        self.exposure_history_valid = false;
        self.exposure_history_written = false;
        self.reset_path_tracing_history(0);
        self.immediate_motion.reset();
        #[cfg(feature = "models3d")]
        self.invalidate_registered_virtual_hiz(false);
        self.temporal_camera_cut_pending = true;
    }
}

#[cfg(test)]
mod tests {
    use super::{taa_jitter_spread, velocity_reference_projection};
    use crate::renderer::{mat4_mul_vec4, mat4_perspective};

    fn assert_vec2_close(actual: [f32; 2], expected: [f32; 2]) {
        for axis in 0..2 {
            assert!(
                (actual[axis] - expected[axis]).abs() <= 1.0e-6,
                "axis {axis}: {} != {}",
                actual[axis],
                expected[axis]
            );
        }
    }

    fn ndc_to_uv(ndc: [f32; 2]) -> [f32; 2] {
        [ndc[0] * 0.5 + 0.5, 0.5 - ndc[1] * 0.5]
    }

    fn velocity_from_clip(current: [f32; 4], previous: [f32; 4]) -> [f32; 2] {
        [
            (current[0] / current[3] - previous[0] / previous[3]) * 0.5,
            (current[1] / current[3] - previous[1] / previous[3]) * 0.5,
        ]
    }

    fn previous_uv_from_velocity(current_uv: [f32; 2], velocity: [f32; 2]) -> [f32; 2] {
        [current_uv[0] - velocity[0], current_uv[1] + velocity[1]]
    }

    #[test]
    fn native_jitter_uses_a_narrower_aperture_without_reducing_upscale_coverage() {
        assert_eq!(taa_jitter_spread(1.0), 0.5);
        assert_eq!(taa_jitter_spread(0.999), 0.5);
        assert_eq!(taa_jitter_spread(0.998), 1.0);
        assert_eq!(taa_jitter_spread(0.75), 1.0);
        assert_eq!(taa_jitter_spread(0.15), 1.0);
    }

    #[test]
    fn current_minus_previous_velocity_reprojects_with_texture_y_orientation() {
        let current_ndc = [0.6, -0.2];
        let previous_ndc = [-0.2, 0.4];
        let current_clip = [current_ndc[0], current_ndc[1], 0.5, 1.0];
        let previous_clip = [previous_ndc[0], previous_ndc[1], 0.5, 1.0];
        let velocity = velocity_from_clip(current_clip, previous_clip);

        assert_vec2_close(velocity, [0.4, -0.3]);
        assert_vec2_close(
            previous_uv_from_velocity(ndc_to_uv(current_ndc), velocity),
            ndc_to_uv(previous_ndc),
        );
        assert_vec2_close(ndc_to_uv(previous_ndc), [0.4, 0.3]);
    }

    #[test]
    fn current_jitter_on_previous_projection_makes_static_velocity_zero() {
        let projection = mat4_perspective(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let jitter = [0.00325, -0.0045];
        // Match the independent current-projection construction in
        // begin_mode_3d; the helper owns only the previous-frame reference.
        let mut current_projection = projection;
        current_projection[2][0] += jitter[0];
        current_projection[2][1] += jitter[1];
        let previous_projection = velocity_reference_projection(projection, jitter);
        let point = [1.25, -0.75, -5.0, 1.0];

        let current_clip = mat4_mul_vec4(&current_projection, &point);
        let previous_clip = mat4_mul_vec4(&previous_projection, &point);
        assert_vec2_close(velocity_from_clip(current_clip, previous_clip), [0.0, 0.0]);

        let uncorrected_previous_clip = mat4_mul_vec4(&projection, &point);
        let jitter_leak = velocity_from_clip(current_clip, uncorrected_previous_clip);
        assert!(
            jitter_leak[0].abs() > 1.0e-4 && jitter_leak[1].abs() > 1.0e-4,
            "negative control did not expose the jitter delta: {jitter_leak:?}"
        );
    }
}
