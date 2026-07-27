//! Path-tracing geometry megabuffer uploads.
//!
//! UV1 is a separate, independently lazy stream so `Vertex3D`, BLAS input,
//! and base-only PT keep their established ABI and bandwidth.

use super::*;

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
    pub(super) fn upload_pt_geometry(
        &mut self,
        vertices: &[f32],
        indices: &[u32],
        secondary_uvs: Option<&[[f32; 2]]>,
    ) {
        let vertex_bytes = (std::mem::size_of_val(vertices)).max(16) as u64;
        let index_bytes = (std::mem::size_of_val(indices)).max(16) as u64;
        let vertex_recreate = self
            .pt_geo_vertex_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < vertex_bytes);
        let index_recreate = self
            .pt_geo_index_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < index_bytes);
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
                size: vertex_bytes.next_power_of_two(),
                usage: geometry_usage,
                mapped_at_creation: false,
            }));
        }
        if index_recreate {
            self.pt_geo_index_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_geo_indices"),
                size: index_bytes.next_power_of_two(),
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
            return;
        };
        let secondary_bytes = std::mem::size_of_val(secondary_uvs).max(8) as u64;
        let secondary_recreate = self
            .pt_layered
            .uv1_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < secondary_bytes);
        if secondary_recreate {
            self.pt_layered.uv1_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pt_layered_uv1_vertices"),
                size: secondary_bytes.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.pt_layered.bind_groups = std::array::from_fn(|_| None);
        }
        self.queue.write_buffer(
            self.pt_layered.uv1_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(secondary_uvs),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::append_pt_secondary_uvs;

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
