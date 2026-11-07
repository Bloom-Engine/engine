//! Shared allocation/upload primitive for independently lazy PT sidecars.

pub(super) fn ensure_pt_sidecar<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    records: &[T],
    buffer: &mut Option<wgpu::Buffer>,
    dirty: &mut bool,
    record_bytes: u64,
    label: &'static str,
) -> bool {
    debug_assert!(!records.is_empty());
    let needed = record_bytes * records.len() as u64;
    let recreate = buffer.as_ref().is_none_or(|buffer| buffer.size() < needed);
    if recreate {
        let capacity = records.len().next_power_of_two() as u64;
        *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: record_bytes * capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        *dirty = true;
    }
    if *dirty {
        queue.write_buffer(
            buffer.as_ref().expect("PT sidecar buffer is initialized"),
            0,
            bytemuck::cast_slice(records),
        );
        *dirty = false;
    }
    recreate
}
