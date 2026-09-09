//! Previous-position ownership for raylib-style immediate 3D primitives.
//!
//! Immediate vertices are rebuilt in world space every frame, so a previous
//! model matrix cannot describe their motion. Stable submission slots own a
//! compact CPU copy of the prior positions instead. The existing tangent lane
//! is unused by `pipeline_3d`; xyz carries the previous world position and w
//! marks that payload for the vertex shader. This changes neither the vertex
//! stride nor GPU upload size.

use super::{Renderer, Vertex3D};

pub(super) const PREVIOUS_POSITION_MARKER: f32 = 2.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PrimitiveKind {
    Cube,
    CubeWires,
    Sphere,
    SphereWires,
    Cylinder,
    Plane,
    Grid,
    Ray,
}

#[derive(Clone, Copy, Debug)]
struct Range {
    kind: PrimitiveKind,
    start: usize,
    count: usize,
}

#[derive(Default)]
pub(super) struct History {
    previous_positions: Vec<[f32; 3]>,
    current_positions: Vec<[f32; 3]>,
    previous_ranges: Vec<Range>,
    current_ranges: Vec<Range>,
    next_slot: usize,
}

impl History {
    pub(super) fn begin_frame(&mut self) {
        std::mem::swap(&mut self.previous_positions, &mut self.current_positions);
        self.current_positions.clear();
        std::mem::swap(&mut self.previous_ranges, &mut self.current_ranges);
        self.current_ranges.clear();
        self.next_slot = 0;
    }

    pub(super) fn reset(&mut self) {
        self.previous_positions.clear();
        self.current_positions.clear();
        self.previous_ranges.clear();
        self.current_ranges.clear();
        self.next_slot = 0;
    }

    /// Attach the matching prior-frame positions to one completed primitive.
    ///
    /// Kind and vertex count are a topology fence. A first appearance, a
    /// reordered primitive of another kind, or a changed tessellation seeds
    /// previous=current so it cannot create a false motion vector.
    pub(super) fn record(&mut self, kind: PrimitiveKind, vertices: &mut [Vertex3D]) {
        let previous = self
            .previous_ranges
            .get(self.next_slot)
            .copied()
            .filter(|range| {
                range.kind == kind
                    && range.count == vertices.len()
                    && range.start.saturating_add(range.count) <= self.previous_positions.len()
            });
        let current_start = self.current_positions.len();
        for (index, vertex) in vertices.iter_mut().enumerate() {
            let prior = previous
                .map(|range| self.previous_positions[range.start + index])
                .unwrap_or(vertex.position);
            vertex.tangent = [prior[0], prior[1], prior[2], PREVIOUS_POSITION_MARKER];
            self.current_positions.push(vertex.position);
        }
        self.current_ranges.push(Range {
            kind,
            start: current_start,
            count: vertices.len(),
        });
        self.next_slot += 1;
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(super) fn stats(&self) -> (usize, usize) {
        let entries = self.current_ranges.len();
        let bytes = (self.previous_positions.capacity() + self.current_positions.capacity())
            * std::mem::size_of::<[f32; 3]>()
            + (self.previous_ranges.capacity() + self.current_ranges.capacity())
                * std::mem::size_of::<Range>();
        (entries, bytes)
    }
}

impl Renderer {
    pub(super) fn record_immediate_motion(&mut self, kind: PrimitiveKind, vertex_start: usize) {
        self.immediate_motion
            .record(kind, &mut self.vertices_3d[vertex_start..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(position: [f32; 3]) -> Vertex3D {
        Vertex3D {
            position,
            tangent: [9.0; 4],
            ..Default::default()
        }
    }

    fn prior(vertex: &Vertex3D) -> [f32; 3] {
        [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]]
    }

    #[test]
    fn first_appearance_seeds_zero_motion_then_stable_slot_uses_prior_position() {
        let mut history = History::default();
        history.begin_frame();
        let mut first = [vertex([1.0, 2.0, 3.0])];
        history.record(PrimitiveKind::Cube, &mut first);
        assert_eq!(prior(&first[0]), first[0].position);
        assert_eq!(first[0].tangent[3], PREVIOUS_POSITION_MARKER);

        history.begin_frame();
        let mut moved = [vertex([4.0, 5.0, 6.0])];
        history.record(PrimitiveKind::Cube, &mut moved);
        assert_eq!(prior(&moved[0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn kind_and_topology_mismatches_cannot_inherit_unrelated_motion() {
        let mut history = History::default();
        history.begin_frame();
        history.record(
            PrimitiveKind::Cube,
            &mut [vertex([1.0, 0.0, 0.0]), vertex([2.0, 0.0, 0.0])],
        );

        history.begin_frame();
        let mut wrong_kind = [vertex([8.0, 0.0, 0.0]), vertex([9.0, 0.0, 0.0])];
        history.record(PrimitiveKind::Sphere, &mut wrong_kind);
        assert_eq!(prior(&wrong_kind[0]), wrong_kind[0].position);

        history.begin_frame();
        let mut wrong_count = [vertex([12.0, 0.0, 0.0])];
        history.record(PrimitiveKind::Sphere, &mut wrong_count);
        assert_eq!(prior(&wrong_count[0]), wrong_count[0].position);
    }

    #[test]
    fn an_empty_frame_breaks_submission_history() {
        let mut history = History::default();
        history.begin_frame();
        history.record(PrimitiveKind::Ray, &mut [vertex([1.0, 0.0, 0.0])]);
        history.begin_frame();
        history.begin_frame();

        let mut reappeared = [vertex([7.0, 0.0, 0.0])];
        history.record(PrimitiveKind::Ray, &mut reappeared);
        assert_eq!(prior(&reappeared[0]), reappeared[0].position);
    }
}
