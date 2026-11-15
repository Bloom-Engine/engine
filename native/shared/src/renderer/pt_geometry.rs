//! Path-tracing geometry megabuffer uploads.
//!
//! UV1 is a separate, independently lazy stream so `Vertex3D`, BLAS input,
//! and base-only PT keep their established ABI and bandwidth.

use super::*;

fn buffer_allocation_size(required: u64, maximum: u64) -> Option<u64> {
    if required > maximum {
        return None;
    }
    Some(
        required
            .checked_next_power_of_two()
            .unwrap_or(required)
            .min(maximum),
    )
}

/// Bytes of one megabuffer vertex (`Vertex3D`, 24 f32 words).
const PT_VERTEX_BYTES: u64 = 96;

/// How much geometry the megabuffers may hold on this device (EN-070).
///
/// The buffers are BLAS input and the pre-skin pass binds one mesh's window
/// of the vertex buffer, so without the path tracer a buffer may be as large
/// as the device allows a buffer to be. The path-trace kernel binds both
/// buffers whole as storage, which has its own, usually smaller, limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PtGeometryBudget {
    /// Largest size of the vertex buffer and of the index buffer.
    pub(super) buffer_bytes: u64,
    /// Largest storage binding: one skinned mesh's window must fit in it.
    binding_bytes: u64,
    /// Storage binding offsets are multiples of this.
    offset_alignment: u64,
}

impl PtGeometryBudget {
    pub(super) fn new(limits: &wgpu::Limits, bound_whole: bool) -> Self {
        let binding_bytes = u64::from(limits.max_storage_buffer_binding_size);
        Self {
            buffer_bytes: if bound_whole {
                limits.max_buffer_size.min(binding_bytes)
            } else {
                limits.max_buffer_size
            },
            binding_bytes,
            offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment).max(1),
        }
    }

    /// Whether a skinned mesh of `vertex_count` / `index_count` still fits
    /// behind `vertex_cursor` vertices and `index_cursor` indices. One that
    /// does not is left out of the ray scene; it is still drawn.
    pub(super) fn admits(
        &self,
        vertex_cursor: u32,
        index_cursor: u32,
        vertex_count: usize,
        index_count: usize,
    ) -> bool {
        let vertex_end = u64::from(vertex_cursor) + vertex_count as u64;
        let index_end = u64::from(index_cursor) + index_count as u64;
        let (_, window_bytes, _) = self.skin_window(vertex_cursor, vertex_count as u64);
        vertex_end <= u64::from(u32::MAX)
            && index_end <= u64::from(u32::MAX)
            && vertex_end * PT_VERTEX_BYTES <= self.buffer_bytes
            && index_end * 4 <= self.buffer_bytes
            && window_bytes <= self.binding_bytes
    }

    /// The part of the vertex buffer the pre-skin pass binds to pose the
    /// mesh at `vertex_base`: byte offset, byte size, and the mesh's first
    /// vertex counted from that offset. The offset is the last whole vertex
    /// at or before the mesh that is also a legal binding offset, so the
    /// window never reaches past the mesh's own last vertex.
    pub(super) fn skin_window(&self, vertex_base: u32, vertex_count: u64) -> (u64, u64, u32) {
        let step = lcm(PT_VERTEX_BYTES, self.offset_alignment) / PT_VERTEX_BYTES;
        let local_base = u64::from(vertex_base) % step;
        let offset = (u64::from(vertex_base) - local_base) * PT_VERTEX_BYTES;
        let size = (local_base + vertex_count) * PT_VERTEX_BYTES;
        (offset, size, local_base as u32)
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

/// Says once per process that skinned meshes were left out of the ray scene.
pub(super) fn report_pt_geometry_overflow(left_out: usize, budget: &PtGeometryBudget) {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if left_out > 0 && !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "bloom: ray scene full: {left_out} skinned mesh(es) this frame do not fit the \
             {} MiB traced-geometry buffer and are left out of ray-traced lighting (they \
             are still drawn); said once",
            budget.buffer_bytes >> 20
        );
    }
}

pub(super) fn append_pt_secondary_uvs(
    output: &mut Option<Vec<[f32; 2]>>,
    vertex_base: usize,
    vertex_count: usize,
    secondary_uvs: Option<&[[f32; 2]]>,
    uses_uv1: bool,
) {
    debug_assert!(output.as_ref().is_none_or(|uvs| uvs.len() == vertex_base));
    debug_assert!(!uses_uv1 || secondary_uvs.is_some_and(|uvs| uvs.len() == vertex_count));
    if output.is_none() && uses_uv1 {
        *output = Some(vec![[0.0; 2]; vertex_base]);
    }
    if let Some(output) = output {
        if uses_uv1 {
            output.extend_from_slice(secondary_uvs.unwrap());
        } else {
            output.resize(output.len() + vertex_count, [0.0; 2]);
        }
    }
}

impl Renderer {
    pub(super) fn pt_geometry_budget(&self, bound_whole: bool) -> PtGeometryBudget {
        PtGeometryBudget::new(&self.device.limits(), bound_whole)
    }

    pub(super) fn upload_pt_geometry(
        &mut self,
        vertices: &[f32],
        indices: &[u32],
        secondary_uvs: Option<&[[f32; 2]]>,
    ) -> bool {
        let vertex_bytes = (std::mem::size_of_val(vertices)).max(16) as u64;
        let index_bytes = (std::mem::size_of_val(indices)).max(16) as u64;
        let maximum = self.pt_geometry_budget(self.pt_active()).buffer_bytes;
        let Some(vertex_allocation) = buffer_allocation_size(vertex_bytes, maximum) else {
            return false;
        };
        let Some(index_allocation) = buffer_allocation_size(index_bytes, maximum) else {
            return false;
        };
        let vertex_recreate = self
            .pt_geo_vertex_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < vertex_bytes || buffer.size() > maximum);
        let index_recreate = self
            .pt_geo_index_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < index_bytes || buffer.size() > maximum);
        // Dynamic PT windows also serve as BLAS input after pre-skinning.
        let geometry_usage = if self.hw_rt_enabled {
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::BLAS_INPUT
        } else {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        };
        if vertex_recreate {
            self.pt_geo_vertex_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_geo_vertices"),
                size: vertex_allocation,
                usage: geometry_usage,
                mapped_at_creation: false,
            }));
        }
        if index_recreate {
            self.pt_geo_index_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_geo_indices"),
                size: index_allocation,
                usage: geometry_usage,
                mapped_at_creation: false,
            }));
        }
        if vertex_recreate || index_recreate {
            self.pt_bg = [None, None];
        }
        if !vertices.is_empty() {
            self.queue.write_buffer(
                self.pt_geo_vertex_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(vertices),
            );
        }
        if !indices.is_empty() {
            self.queue.write_buffer(
                self.pt_geo_index_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(indices),
            );
        }

        let Some(secondary_uvs) = secondary_uvs else {
            return true;
        };
        let secondary_bytes = std::mem::size_of_val(secondary_uvs).max(8) as u64;
        let Some(secondary_allocation) = buffer_allocation_size(secondary_bytes, maximum) else {
            return false;
        };
        let secondary_recreate = self
            .pt_layered
            .uv1_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < secondary_bytes);
        if secondary_recreate {
            self.pt_layered.uv1_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_layered_uv1_vertices"),
                size: secondary_allocation,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.pt_layered
                .bind_groups
                .iter_mut()
                .for_each(|group| *group = None);
        }
        self.queue.write_buffer(
            self.pt_layered.uv1_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(secondary_uvs),
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{append_pt_secondary_uvs, buffer_allocation_size, PtGeometryBudget};

    fn budget(
        buffer_mib: u64,
        binding_mib: u64,
        alignment: u32,
        bound_whole: bool,
    ) -> PtGeometryBudget {
        let limits = wgpu::Limits {
            max_buffer_size: buffer_mib << 20,
            max_storage_buffer_binding_size: (binding_mib << 20) as _,
            min_storage_buffer_offset_alignment: alignment,
            ..Default::default()
        };
        PtGeometryBudget::new(&limits, bound_whole)
    }

    #[test]
    fn skinned_meshes_fill_the_buffer_and_no_further() {
        // 1 MiB of vertices is 10922 whole vertices of 96 bytes.
        let b = budget(1, 128, 256, false);
        assert!(b.admits(0, 0, 10_922, 3));
        assert!(!b.admits(0, 0, 10_923, 3));
        assert!(b.admits(10_000, 0, 922, 3));
        assert!(!b.admits(10_000, 0, 923, 3));
        // Indices have the same ceiling, at four bytes each.
        assert!(b.admits(0, 262_000, 3, 144));
        assert!(!b.admits(0, 262_000, 3, 145));
        // Cursors are u32 in the instance records.
        let huge = budget(1 << 20, 128, 256, false);
        assert!(!huge.admits(u32::MAX - 5, 0, 6, 3));
    }

    #[test]
    fn path_tracer_binds_whole_buffers_so_the_binding_limit_caps_them() {
        assert_eq!(budget(1024, 128, 256, false).buffer_bytes, 1024 << 20);
        assert_eq!(budget(1024, 128, 256, true).buffer_bytes, 128 << 20);
        assert_eq!(budget(64, 128, 256, true).buffer_bytes, 64 << 20);
    }

    #[test]
    fn skin_window_is_aligned_covers_the_mesh_and_fits_a_binding() {
        for alignment in [1_u32, 16, 32, 64, 256] {
            let b = budget(1024, 128, alignment, false);
            for base in [0_u32, 1, 7, 8, 9, 31_337, 2_800_000] {
                let (offset, size, local) = b.skin_window(base, 5000);
                assert_eq!(offset % u64::from(alignment), 0);
                assert_eq!(offset % 96, 0);
                assert_eq!(offset / 96 + u64::from(local), u64::from(base));
                assert_eq!(size, (u64::from(local) + 5000) * 96);
                assert!(local < 8);
            }
        }
        // A window is one mesh plus at most seven vertices of padding in
        // front; a mesh too large for one binding is not admitted.
        let b = budget(1024, 1, 256, false);
        assert!(b.admits(7, 0, 10_915, 3));
        assert!(!b.admits(7, 0, 10_916, 3));
    }

    #[test]
    fn allocation_growth_never_overshoots_device_limit() {
        let mib = 1024 * 1024;
        assert_eq!(
            buffer_allocation_size(129 * mib, 256 * mib),
            Some(256 * mib)
        );
        assert_eq!(
            buffer_allocation_size(200 * mib, 224 * mib),
            Some(224 * mib)
        );
        assert_eq!(buffer_allocation_size(225 * mib, 224 * mib), None);
    }

    #[test]
    fn secondary_uv_stream_backfills_and_stays_vertex_aligned() {
        let mut output = None;
        append_pt_secondary_uvs(&mut output, 0, 3, None, false);
        assert!(output.is_none());
        append_pt_secondary_uvs(&mut output, 3, 2, Some(&[[0.25, 0.5], [0.75, 1.0]]), true);
        append_pt_secondary_uvs(&mut output, 5, 1, None, false);
        assert_eq!(
            output.unwrap(),
            vec![
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.25, 0.5],
                [0.75, 1.0],
                [0.0, 0.0],
            ]
        );
    }
}
