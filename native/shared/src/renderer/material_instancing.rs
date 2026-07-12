//! EN-001 instance-buffer creation/destruction, including the
//! spatial tiling that lets the dispatcher frustum-cull instanced
//! draws per tile. Split from material_system.rs (2000-line file
//! policy).

use super::material_system::{MaterialSystem, InstanceBuffer, InstanceTile};

impl MaterialSystem {
    /// EN-001 — create a persistent instance buffer from CPU-side
    /// floats. The data layout matches `InstanceData3D` (9 floats per
    /// instance: pos.xyz, rot_y, scale, tint.rgba); this method pads
    /// each instance to 12 floats at upload time so the GPU side gets
    /// the correct 48-byte stride. Returns a 1-based handle to use
    /// with `submit_draw_instanced`. Pair with `destroy_instance_buffer`
    /// when the buffer's no longer needed.
    pub fn create_instance_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raw: &[f32],
        instance_count: u32,
    ) -> u32 {
        let count = (instance_count as usize).min(raw.len() / 9);

        // Spatial tiling: reorder instances into an XZ grid so each tile
        // is a contiguous range the dispatcher can frustum-cull as a
        // unit. Reordering is invisible to opaque/cutout draws (depth
        // tested) and the per-instance attributes travel with the
        // instance, so @builtin(instance_index) consumers stay
        // consistent with their data. Small buffers stay untiled.
        const TILE_TARGET: usize = 128;
        const TILE_MIN_COUNT: usize = 512;
        let mut order: Vec<usize> = (0..count).collect();
        let mut tiles: Vec<InstanceTile> = Vec::new();
        if count >= TILE_MIN_COUNT {
            let mut xz_min = [f32::MAX; 2];
            let mut xz_max = [f32::MIN; 2];
            for i in 0..count {
                let p = &raw[i * 9..i * 9 + 3];
                if p[0] < xz_min[0] { xz_min[0] = p[0]; }
                if p[0] > xz_max[0] { xz_max[0] = p[0]; }
                if p[2] < xz_min[1] { xz_min[1] = p[2]; }
                if p[2] > xz_max[1] { xz_max[1] = p[2]; }
            }
            let grid = ((count as f32 / TILE_TARGET as f32).sqrt().ceil() as usize).max(1);
            let ext_x = (xz_max[0] - xz_min[0]).max(1e-3);
            let ext_z = (xz_max[1] - xz_min[1]).max(1e-3);
            let cell_of = |i: usize| -> usize {
                let p = &raw[i * 9..i * 9 + 3];
                let cx = (((p[0] - xz_min[0]) / ext_x * grid as f32) as usize).min(grid - 1);
                let cz = (((p[2] - xz_min[1]) / ext_z * grid as f32) as usize).min(grid - 1);
                cz * grid + cx
            };
            order.sort_by_key(|&i| cell_of(i));
            // Emit one tile per non-empty cell (contiguous after the sort).
            let mut start = 0usize;
            while start < count {
                let cell = cell_of(order[start]);
                let mut end = start + 1;
                while end < count && cell_of(order[end]) == cell { end += 1; }
                let mut pmin = [f32::MAX; 3];
                let mut pmax = [f32::MIN; 3];
                let mut max_scale = 0.0f32;
                for &i in &order[start..end] {
                    let inst = &raw[i * 9..i * 9 + 9];
                    for a in 0..3 {
                        if inst[a] < pmin[a] { pmin[a] = inst[a]; }
                        if inst[a] > pmax[a] { pmax[a] = inst[a]; }
                    }
                    if inst[4].abs() > max_scale { max_scale = inst[4].abs(); }
                }
                tiles.push(InstanceTile {
                    first: start as u32,
                    count: (end - start) as u32,
                    pmin,
                    pmax,
                    max_scale,
                });
                start = end;
            }
        }

        let mut packed: Vec<f32> = Vec::with_capacity(count * 12);
        for &i in order.iter() {
            let off = i * 9;
            packed.extend_from_slice(&raw[off..off + 3]);     // pos.xyz
            packed.push(raw[off + 3]);                        // rot_y
            packed.push(raw[off + 4]);                        // scale
            packed.extend_from_slice(&raw[off + 5..off + 9]); // tint.rgba
            packed.extend_from_slice(&[0.0, 0.0, 0.0]);       // pad to 48 bytes
        }
        let size = (packed.len() * std::mem::size_of::<f32>()) as u64;
        // Empty buffers can't be created (size 0 is invalid in wgpu).
        // Reserve at least one stride so the BG/binding remains valid.
        let buffer_size = size.max(48);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_instance_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !packed.is_empty() {
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&packed));
        }
        self.instance_buffers.push(Some(InstanceBuffer {
            buffer,
            count: count as u32,
            tiles,
        }));
        self.instance_buffers.len() as u32
    }

    /// EN-026 — a *dynamic* instance buffer: fixed capacity, rewritten every
    /// frame, never tiled.
    ///
    /// The static path above reorders instances into XZ tiles so the
    /// dispatcher can frustum-cull them, which is right for a 20k-blade grass
    /// field that never moves and wrong for particles, which move every frame
    /// and would have to be re-tiled (a sort) each time. Here the caller
    /// simply writes the live prefix of the buffer and draws that many
    /// instances.
    pub fn create_dynamic_instance_buffer(
        &mut self,
        device: &wgpu::Device,
        capacity: u32,
    ) -> u32 {
        let size = (capacity.max(1) as u64) * 48;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamic_instance_buffer"),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_buffers.push(Some(InstanceBuffer {
            buffer,
            count: capacity,
            tiles: Vec::new(),
        }));
        self.instance_buffers.len() as u32
    }

    /// EN-026 — overwrite the first `count` instances of a dynamic buffer.
    /// `packed` is already at the 12-float GPU stride (pos.xyz, rot_y, scale,
    /// tint.rgba, extra.xyz), so this is a straight memcpy into GPU memory
    /// with no per-instance work on the CPU.
    pub fn update_instance_buffer(
        &mut self,
        queue: &wgpu::Queue,
        handle: u32,
        packed: &[f32],
        count: u32,
    ) {
        if handle == 0 || count == 0 { return; }
        let idx = handle as usize - 1;
        let Some(Some(ib)) = self.instance_buffers.get(idx) else { return };
        let n = (count as usize).min(packed.len() / 12);
        if n == 0 { return; }
        queue.write_buffer(&ib.buffer, 0, bytemuck::cast_slice(&packed[..n * 12]));
    }

    /// EN-001 — drop an instance buffer slot. The slot is left as
    /// `None` so previously-issued handles never alias a future
    /// allocation. No-op for `handle == 0` or out-of-range handles.
    pub fn destroy_instance_buffer(&mut self, handle: u32) {
        if handle == 0 { return; }
        let idx = handle as usize - 1;
        if idx < self.instance_buffers.len() {
            self.instance_buffers[idx] = None;
        }
    }
}
