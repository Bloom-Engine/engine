//! Common temporal-history invalidation and camera-cut ownership.

use super::*;

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
        self.probe_history_valid = false;
        self.exposure_current_idx = 0;
        self.exposure_history_valid = false;
        self.exposure_history_written = false;
        self.reset_path_tracing_history(0);
        self.temporal_camera_cut_pending = true;
    }
}
