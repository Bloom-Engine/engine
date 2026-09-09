//! Dirty-range planning for the renderer's large lighting uniform buffer.
//!
//! Lighting setters mutate the CPU snapshot throughout a frame. Uploading the
//! complete ~9 KiB block after every setter made one ordinary frame enqueue the
//! same data many times. This tracker compares the final snapshot against the
//! last submitted bytes and emits at most one aligned range for each logical
//! region: fixed/directional fields, point lights, and view/shadow/frame data.

use std::ops::Range;

use super::types::LightingUniforms;

const POINT_LIGHTS_OFFSET: usize = std::mem::offset_of!(LightingUniforms, point_lights);
const VIEW_DATA_OFFSET: usize = std::mem::offset_of!(LightingUniforms, camera_pos);
const LIGHTING_BYTES: usize = std::mem::size_of::<LightingUniforms>();
const REGIONS: [Range<usize>; 3] = [
    0..POINT_LIGHTS_OFFSET,
    POINT_LIGHTS_OFFSET..VIEW_DATA_OFFSET,
    VIEW_DATA_OFFSET..LIGHTING_BYTES,
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct LightingUploadStats {
    pub(super) write_count: u32,
    pub(super) byte_count: u64,
}

pub(super) struct LightingUploadBatch {
    ranges: [Option<Range<usize>>; 3],
}

impl LightingUploadBatch {
    pub(super) fn ranges(&self) -> impl Iterator<Item = &Range<usize>> {
        self.ranges.iter().flatten()
    }
}

pub(super) struct LightingUploadTracker {
    last_uploaded: LightingUniforms,
    frame_stats: LightingUploadStats,
}

impl LightingUploadTracker {
    pub(super) fn new(initial: LightingUniforms) -> Self {
        Self {
            last_uploaded: initial,
            frame_stats: LightingUploadStats::default(),
        }
    }

    pub(super) fn begin_frame(&mut self) {
        self.frame_stats = LightingUploadStats::default();
    }

    pub(super) fn plan(&mut self, current: LightingUniforms) -> LightingUploadBatch {
        let current_bytes = bytemuck::bytes_of(&current);
        let previous_bytes = bytemuck::bytes_of(&self.last_uploaded);
        let ranges = REGIONS
            .clone()
            .map(|region| changed_word_range(current_bytes, previous_bytes, region));
        for range in ranges.iter().flatten() {
            self.frame_stats.write_count = self.frame_stats.write_count.saturating_add(1);
            self.frame_stats.byte_count = self
                .frame_stats
                .byte_count
                .saturating_add((range.end - range.start) as u64);
        }
        self.last_uploaded = current;
        LightingUploadBatch { ranges }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn frame_stats(&self) -> LightingUploadStats {
        self.frame_stats
    }
}

fn changed_word_range(
    current: &[u8],
    previous: &[u8],
    region: Range<usize>,
) -> Option<Range<usize>> {
    debug_assert_eq!(region.start % wgpu::COPY_BUFFER_ALIGNMENT as usize, 0);
    debug_assert_eq!(region.end % wgpu::COPY_BUFFER_ALIGNMENT as usize, 0);
    let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
    let first = (region.start..region.end)
        .step_by(alignment)
        .find(|&offset| {
            current[offset..offset + alignment] != previous[offset..offset + alignment]
        })?;
    let last = (region.start..region.end)
        .step_by(alignment)
        .rev()
        .find(|&offset| current[offset..offset + alignment] != previous[offset..offset + alignment])
        .expect("the first changed word proves a last changed word exists");
    Some(first..last + alignment)
}

#[cfg(test)]
mod tests {
    use super::{LightingUploadTracker, LIGHTING_BYTES};
    use crate::renderer::types::LightingUniforms;

    #[test]
    fn unchanged_snapshot_schedules_no_upload() {
        let lighting = LightingUniforms::defaults();
        let mut tracker = LightingUploadTracker::new(lighting);
        tracker.begin_frame();

        assert_eq!(tracker.plan(lighting).ranges().count(), 0);
        assert_eq!(tracker.frame_stats().write_count, 0);
        assert_eq!(tracker.frame_stats().byte_count, 0);
    }

    #[test]
    fn changes_are_bounded_to_three_non_overlapping_regions() {
        let initial = LightingUniforms::defaults();
        let mut changed = initial;
        changed.ambient[0] = 0.25;
        changed.point_lights[7].position[2] = 42.0;
        changed.camera_pos[0] = 3.0;

        let mut tracker = LightingUploadTracker::new(initial);
        tracker.begin_frame();
        let batch = tracker.plan(changed);
        let ranges: Vec<_> = batch.ranges().cloned().collect();

        assert_eq!(ranges.len(), 3);
        assert!(ranges.windows(2).all(|pair| pair[0].end <= pair[1].start));
        assert_eq!(tracker.frame_stats().write_count, 3);
        assert!(tracker.frame_stats().byte_count < LIGHTING_BYTES as u64);
    }

    #[test]
    fn repeated_setter_mutations_coalesce_before_planning() {
        let initial = LightingUniforms::defaults();
        let mut changed = initial;
        changed.point_lights[3].position = [1.0, 2.0, 3.0, 4.0];
        changed.point_lights[3].color = [0.2, 0.4, 0.6, 8.0];

        let mut tracker = LightingUploadTracker::new(initial);
        tracker.begin_frame();
        let batch = tracker.plan(changed);

        assert_eq!(batch.ranges().count(), 1);
        assert_eq!(tracker.frame_stats().write_count, 1);
        assert_eq!(tracker.frame_stats().byte_count, 32);
    }
}
