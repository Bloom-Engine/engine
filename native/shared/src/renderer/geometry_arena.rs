//! The shared static-geometry arena of the GPU-driven path: one vertex and one
//! index buffer for every static mesh, with a free list. It grows by doubling
//! up to the device's buffer limit; past that an upload is refused and the
//! caller keeps the mesh on buffers of its own.

use super::super::{material_indirection::GpuCompletionTracker, Vertex3D};
use super::GeometrySlice;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct FreeRange {
    offset: u64,
    size: u64,
}

struct RetiredGeometry {
    epoch: u64,
    slice: GeometrySlice,
}

pub(super) struct GeometryArena {
    pub(super) vertex: wgpu::Buffer,
    pub(super) index: wgpu::Buffer,
    vertex_capacity: u64,
    index_capacity: u64,
    pub(super) vertex_end: u64,
    index_end: u64,
    free_vertices: Vec<FreeRange>,
    free_indices: Vec<FreeRange>,
    retired: Vec<RetiredGeometry>,
    completion: GpuCompletionTracker,
    pub(super) generation: u64,
    /// Largest size either arena buffer may reach on this device. The vertex
    /// arena is also bound whole as a storage buffer by visibility shading,
    /// so the storage-binding limit applies as well as the buffer limit.
    max_capacity: u64,
    full_warned: bool,
}

impl GeometryArena {
    const INITIAL_CAPACITY: u64 = 64 * 1024;

    pub(super) fn new(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            max_capacity: arena_max_capacity(
                limits.max_buffer_size,
                limits.max_storage_buffer_binding_size as u64,
            ),
            full_warned: false,
            vertex: create_vertex_arena(device, Self::INITIAL_CAPACITY),
            index: create_index_arena(device, Self::INITIAL_CAPACITY),
            vertex_capacity: Self::INITIAL_CAPACITY,
            index_capacity: Self::INITIAL_CAPACITY,
            vertex_end: 0,
            index_end: 0,
            free_vertices: Vec::new(),
            free_indices: Vec::new(),
            retired: Vec::new(),
            completion: GpuCompletionTracker::default(),
            generation: 0,
        }
    }

    pub(super) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex3D],
        indices: &[u32],
    ) -> Option<GeometrySlice> {
        self.collect();
        let vertex_size = std::mem::size_of_val(vertices) as u64;
        let index_size = std::mem::size_of_val(indices) as u64;
        let previous_vertex_end = self.vertex_end;
        let previous_index_end = self.index_end;
        let vertex_offset = allocate_range(
            &mut self.free_vertices,
            &mut self.vertex_end,
            vertex_size,
            std::mem::align_of::<Vertex3D>() as u64,
        );
        let index_offset =
            allocate_range(&mut self.free_indices, &mut self.index_end, index_size, 4);
        if vertex_offset + vertex_size > self.max_capacity
            || index_offset + index_size > self.max_capacity
        {
            // The device cannot hold a larger arena. Hand the ranges back and
            // let the caller keep this mesh on its own buffers.
            release_range(
                &mut self.free_vertices,
                &mut self.vertex_end,
                previous_vertex_end,
                vertex_offset,
                vertex_size,
            );
            release_range(
                &mut self.free_indices,
                &mut self.index_end,
                previous_index_end,
                index_offset,
                index_size,
            );
            if !self.full_warned {
                self.full_warned = true;
                log::warn!(
                    "bloom: gpu-driven geometry arena is full at {} MB (device limit); \
                     further static meshes keep dedicated buffers",
                    self.max_capacity / (1024 * 1024)
                );
            }
            return None;
        }
        self.grow_if_needed(
            device,
            queue,
            vertex_offset + vertex_size,
            index_offset + index_size,
            previous_vertex_end,
            previous_index_end,
        );
        if vertex_size > 0 {
            queue.write_buffer(&self.vertex, vertex_offset, bytemuck::cast_slice(vertices));
        }
        if index_size > 0 {
            queue.write_buffer(&self.index, index_offset, bytemuck::cast_slice(indices));
        }
        Some(GeometrySlice {
            vertex_offset,
            vertex_size,
            index_offset,
            index_size,
            first_index: (index_offset / 4) as u32,
            base_vertex: (vertex_offset / std::mem::size_of::<Vertex3D>() as u64) as i32,
        })
    }

    pub(super) fn retire_many(
        &mut self,
        queue: &wgpu::Queue,
        slices: impl IntoIterator<Item = GeometrySlice>,
    ) {
        let slices: Vec<_> = slices.into_iter().collect();
        if slices.is_empty() {
            return;
        }
        let epoch = self.completion.track_submitted_work(queue);
        self.retired.extend(
            slices
                .into_iter()
                .map(|slice| RetiredGeometry { epoch, slice }),
        );
    }

    fn collect(&mut self) {
        let completed = self.completion.completed_epoch();
        let mut keep = Vec::with_capacity(self.retired.len());
        for retired in self.retired.drain(..) {
            if retired.epoch > completed {
                keep.push(retired);
                continue;
            }
            if retired.slice.vertex_size > 0 {
                self.free_vertices.push(FreeRange {
                    offset: retired.slice.vertex_offset,
                    size: retired.slice.vertex_size,
                });
            }
            if retired.slice.index_size > 0 {
                self.free_indices.push(FreeRange {
                    offset: retired.slice.index_offset,
                    size: retired.slice.index_size,
                });
            }
        }
        self.retired = keep;
        merge_ranges(&mut self.free_vertices);
        merge_ranges(&mut self.free_indices);
    }

    fn grow_if_needed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        required_vertex: u64,
        required_index: u64,
        copy_vertex_end: u64,
        copy_index_end: u64,
    ) {
        let new_vertex_capacity =
            grown_capacity(self.vertex_capacity, required_vertex, self.max_capacity);
        let new_index_capacity =
            grown_capacity(self.index_capacity, required_index, self.max_capacity);
        if new_vertex_capacity == self.vertex_capacity && new_index_capacity == self.index_capacity
        {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_geometry_arena_grow"),
        });
        if new_vertex_capacity != self.vertex_capacity {
            let next = create_vertex_arena(device, new_vertex_capacity);
            if copy_vertex_end > 0 {
                encoder.copy_buffer_to_buffer(&self.vertex, 0, &next, 0, copy_vertex_end);
            }
            self.vertex = next;
            self.vertex_capacity = new_vertex_capacity;
        }
        if new_index_capacity != self.index_capacity {
            let next = create_index_arena(device, new_index_capacity);
            if copy_index_end > 0 {
                encoder.copy_buffer_to_buffer(&self.index, 0, &next, 0, copy_index_end);
            }
            self.index = next;
            self.index_capacity = new_index_capacity;
        }
        self.generation = self.generation.wrapping_add(1);
        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn create_vertex_arena(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_driven_shared_vertices"),
        size,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_index_arena(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_driven_shared_indices"),
        size,
        usage: wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Capacity doubles to the next power of two, but never past `max`: the
/// caller has already checked that `required` fits under it.
fn grown_capacity(current: u64, required: u64, max: u64) -> u64 {
    if required <= current {
        current
    } else {
        required
            .next_power_of_two()
            .max(GeometryArena::INITIAL_CAPACITY)
            .min(max.max(required))
    }
}

/// Both limits, rounded down to whole vertices and indices.
fn arena_max_capacity(max_buffer_size: u64, max_storage_binding_size: u64) -> u64 {
    let limit = max_buffer_size.min(max_storage_binding_size);
    let unit = std::mem::size_of::<Vertex3D>() as u64 * 4;
    limit / unit * unit
}

/// Undo one `allocate_range`: an allocation taken from the end of the arena
/// moves the end back, one taken from the free list returns to it.
fn release_range(
    free: &mut Vec<FreeRange>,
    end: &mut u64,
    previous_end: u64,
    offset: u64,
    size: u64,
) {
    if size == 0 {
        return;
    }
    if *end != previous_end {
        *end = previous_end;
    } else {
        free.push(FreeRange { offset, size });
        merge_ranges(free);
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn allocate_range(free: &mut Vec<FreeRange>, end: &mut u64, size: u64, alignment: u64) -> u64 {
    if size == 0 {
        return 0;
    }
    for index in 0..free.len() {
        let aligned = align_up(free[index].offset, alignment);
        let padding = aligned - free[index].offset;
        if padding + size > free[index].size {
            continue;
        }
        let original = free.swap_remove(index);
        if padding > 0 {
            free.push(FreeRange {
                offset: original.offset,
                size: padding,
            });
        }
        let tail_offset = aligned + size;
        let tail_size = original.size - padding - size;
        if tail_size > 0 {
            free.push(FreeRange {
                offset: tail_offset,
                size: tail_size,
            });
        }
        return aligned;
    }
    let offset = align_up(*end, alignment);
    *end = offset + size;
    offset
}

fn merge_ranges(ranges: &mut Vec<FreeRange>) {
    ranges.sort_by_key(|range| range.offset);
    let mut write = 0usize;
    for read in 0..ranges.len() {
        let current = ranges[read];
        if write > 0 {
            let previous = &mut ranges[write - 1];
            if previous.offset + previous.size == current.offset {
                previous.size += current.size;
                continue;
            }
        }
        ranges[write] = current;
        write += 1;
    }
    ranges.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_growth_stops_at_the_device_limit() {
        let mb = 1024 * 1024;
        assert_eq!(grown_capacity(64 * mb, 60 * mb, 256 * mb), 64 * mb);
        assert_eq!(grown_capacity(64 * mb, 100 * mb, 256 * mb), 128 * mb);
        // 300 MB would double to 512 MB; a 400 MB device gets 400 MB.
        assert_eq!(grown_capacity(256 * mb, 300 * mb, 400 * mb), 400 * mb);
        let unit = std::mem::size_of::<Vertex3D>() as u64 * 4;
        assert_eq!(arena_max_capacity(1000 * mb, 300 * mb) % unit, 0);
        assert!(arena_max_capacity(1000 * mb, 300 * mb) <= 300 * mb);
    }

    #[test]
    fn refused_allocation_is_handed_back() {
        // Taken from the end of the arena: the end moves back.
        let mut free = Vec::new();
        let mut end = 64;
        let offset = allocate_range(&mut free, &mut end, 32, 4);
        assert_eq!((offset, end), (64, 96));
        release_range(&mut free, &mut end, 64, offset, 32);
        assert_eq!(end, 64);
        assert!(free.is_empty());
        // Taken from the free list: the range is free again, and whole.
        let mut free = vec![FreeRange {
            offset: 16,
            size: 32,
        }];
        let offset = allocate_range(&mut free, &mut end, 8, 4);
        assert_eq!((offset, end), (16, 64));
        release_range(&mut free, &mut end, 64, offset, 8);
        assert_eq!(
            free,
            vec![FreeRange {
                offset: 16,
                size: 32
            }]
        );
    }

    #[test]
    fn allocator_reuses_and_merges_ranges() {
        let mut free = vec![FreeRange {
            offset: 16,
            size: 16,
        }];
        let mut end = 64;
        assert_eq!(allocate_range(&mut free, &mut end, 8, 4), 16);
        free.push(FreeRange {
            offset: 24,
            size: 8,
        });
        free.push(FreeRange {
            offset: 16,
            size: 8,
        });
        merge_ranges(&mut free);
        assert!(free.iter().any(|range| *range
            == FreeRange {
                offset: 16,
                size: 16
            }));
    }
}
