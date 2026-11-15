//! EN-001 instance-buffer creation/destruction, including the
//! spatial tiling that lets the dispatcher frustum-cull instanced
//! draws per tile. Split from material_system.rs (2000-line file
//! policy).

use super::material_system::{InstanceBuffer, InstanceDrawInfo, InstanceTile, MaterialSystem};

/// EN-060: distance thinning of a static instance buffer.
///
/// Every instance carries a `rank` in (0, 1) in `instance_extra.x`
/// (location 11), spread evenly inside its tile. A thinning `level`
/// falls with eye distance: `1 + band` up to `start`, a straight line
/// down to `min_fraction` at `end`, constant beyond. An instance is
/// whole while the level is above `rank + band`, and gone once the
/// level is at or below its rank. Two halves make that invisible:
///
/// * The MATERIAL's vertex stage decides per instance, shrinking an
///   instance away as the level comes down through its band (helpers
///   in `common/instance_lod.wgsl`). That is continuous in the camera
///   position, so nothing pops.
/// * The ENGINE only skips work: inside a tile the instances are stored
///   in rank order, so a tile whose NEAREST point is at distance `d`
///   draws its first `ceil(count * level(d))` instances. Every instance
///   it leaves out is one the shader has already shrunk to nothing.
///
/// `min_density` (instances per square metre of ground, 0 = no limit)
/// keeps thinning away from where the scatter is already sparse and
/// every instance can be told apart: a tile is never thinned below that
/// density, and a tile at or under it is never thinned at all. Each
/// tile's resulting floor on the level goes to its instances in
/// `instance_extra.y`.
///
/// A shader that ignores the rank still draws correctly but thins in
/// tile-sized steps, so only opt in with a material that uses it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InstanceLod {
    pub start: f32,
    pub end: f32,
    pub min_fraction: f32,
    pub band: f32,
    pub min_density: f32,
}

impl InstanceLod {
    /// None when the parameters describe "no thinning".
    pub fn new(
        start: f32,
        end: f32,
        min_fraction: f32,
        band: f32,
        min_density: f32,
    ) -> Option<Self> {
        let ok = start.is_finite() && end.is_finite() && min_fraction.is_finite();
        if !ok || min_fraction >= 1.0 {
            return None;
        }
        let start = start.max(0.0);
        Some(Self {
            start,
            end: end.max(start + 1e-3),
            min_fraction: min_fraction.max(0.0),
            band: if band.is_finite() {
                band.clamp(1e-4, 1.0)
            } else {
                1e-4
            },
            min_density: if min_density.is_finite() {
                min_density.max(0.0)
            } else {
                0.0
            },
        })
    }

    /// The thinning level at eye distance `d` in a tile whose floor is
    /// `floor`. The WGSL twin is `instance_lod_level`; the two must stay
    /// the same function.
    pub fn level(&self, d: f32, floor: f32) -> f32 {
        let t = ((d - self.start) / (self.end - self.start)).clamp(0.0, 1.0);
        let top = 1.0 + self.band;
        (top + (self.min_fraction - top) * t).max(floor)
    }

    /// The floor of a tile of `count` instances over `area` square metres:
    /// the level at which `min_density` of them per square metre are still
    /// whole. `1 + band` (never thinned) when the tile is that sparse already.
    pub fn tile_floor(&self, count: u32, area: f32) -> f32 {
        if self.min_density <= 0.0 || count == 0 {
            return 0.0;
        }
        (self.min_density * area / count as f32).clamp(0.0, 1.0) + self.band
    }

    /// How many leading instances of a `count`-instance tile can still be
    /// visible when the tile's nearest point is `d` from the eye. Instance
    /// `k` has rank `(k + 0.5) / count` and is gone once `rank >= level`.
    pub fn prefix(&self, count: u32, d: f32, floor: f32) -> u32 {
        ((count as f32 * self.level(d, floor).min(1.0)).ceil() as u32).min(count)
    }
}

/// Distance from `eye` to the nearest point of an axis-aligned box.
fn distance_to_aabb(eye: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> f32 {
    let mut d2 = 0.0f32;
    for a in 0..3 {
        let c = eye[a].clamp(bmin[a], bmax[a]);
        d2 += (eye[a] - c) * (eye[a] - c);
    }
    d2.sqrt()
}

/// Deterministic scramble of an instance's authored index: the order
/// inside a thinned tile, so a prefix of the tile is spread over it
/// whatever order the game scattered in.
fn lod_shuffle_key(i: usize) -> u32 {
    let mut x = (i as u32).wrapping_add(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^ (x >> 16)
}

impl MaterialSystem {
    /// Issue the draw(s) of one opaque/cutout command whose mesh and
    /// instance buffer are bound. Untiled: one draw of `instance_range`.
    /// Tiled: `commands` holds only the opaque + cutout buckets, so
    /// emitting the visible tile ranges is output-identical to the
    /// full-range draw. Tiles' position AABBs are inflated by
    /// max_scale × the mesh's local half-diagonal (rotation-safe).
    /// Adjacent visible tiles merge into one draw. With `lod_eye` and a
    /// buffer that opted in, a far tile draws only its leading instances
    /// (see [`InstanceLod`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_instance_tiles<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        mesh: &super::MeshDrawRef<'pass>,
        instance: Option<&InstanceDrawInfo>,
        instance_range: std::ops::Range<u32>,
        planes: Option<&[[f32; 4]; 6]>,
        lod_eye: Option<[f32; 3]>,
        (lmin, lmax): ([f32; 3], [f32; 3]),
    ) {
        let buffer = instance
            .filter(|_| planes.is_some() && lmin[0] <= lmax[0])
            .and_then(|inst| {
                self.instance_buffers
                    .get((inst.buffer_handle as usize).wrapping_sub(1))
                    .and_then(|s| s.as_ref())
            })
            .filter(|b| !b.tiles.is_empty());
        let (Some(buffer), Some(planes)) = (buffer, planes) else {
            pass.draw_indexed(mesh.index_range(), mesh.base_vertex, instance_range);
            return;
        };
        let lod = buffer.lod.zip(lod_eye);
        let half_diag = 0.5
            * ((lmax[0] - lmin[0]).powi(2)
                + (lmax[1] - lmin[1]).powi(2)
                + (lmax[2] - lmin[2]).powi(2))
            .sqrt();
        let mut run: Option<(u32, u32)> = None;
        for tile in buffer.tiles.iter() {
            let r = tile.max_scale * half_diag;
            let bmin = [tile.pmin[0] - r, tile.pmin[1] - r, tile.pmin[2] - r];
            let bmax = [tile.pmax[0] + r, tile.pmax[1] + r, tile.pmax[2] + r];
            if crate::scene::aabb_outside_frustum(planes, bmin, bmax) {
                if let Some((s, e)) = run.take() {
                    pass.draw_indexed(mesh.index_range(), mesh.base_vertex, s..e);
                }
                continue;
            }
            // The distance is to the box of instance POSITIONS (the shader
            // measures to the instance position), which is never farther
            // than any instance in the tile.
            let end = tile.first
                + match lod {
                    Some((lod, eye)) => lod.prefix(
                        tile.count,
                        distance_to_aabb(eye, tile.pmin, tile.pmax),
                        tile.lod_floor,
                    ),
                    None => tile.count,
                };
            run = match run {
                Some((s, e)) if e == tile.first => Some((s, end)),
                Some((s, e)) => {
                    pass.draw_indexed(mesh.index_range(), mesh.base_vertex, s..e);
                    Some((tile.first, end))
                }
                None => Some((tile.first, end)),
            };
        }
        if let Some((s, e)) = run {
            if e > s {
                pass.draw_indexed(mesh.index_range(), mesh.base_vertex, s..e);
            }
        }
    }

    /// EN-001 — see [`Self::create_instance_buffer_lod`]; this is the
    /// plain form, without distance thinning.
    pub fn create_instance_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raw: &[f32],
        instance_count: u32,
    ) -> u32 {
        self.create_instance_buffer_lod(device, queue, raw, instance_count, None)
    }

    /// EN-001 — create a persistent instance buffer from CPU-side
    /// floats. The data layout matches `InstanceData3D` (9 floats per
    /// instance: pos.xyz, rot_y, scale, tint.rgba); this method pads
    /// each instance to 12 floats at upload time so the GPU side gets
    /// the correct 48-byte stride. Returns a 1-based handle to use
    /// with `submit_draw_instanced`. Pair with `destroy_instance_buffer`
    /// when the buffer's no longer needed.
    ///
    /// EN-060 — with `lod`, the buffer is always tiled, each tile's
    /// instances are stored in a scrambled order and their rank goes
    /// into `extra.x`; see [`InstanceLod`]. Without it the upload is
    /// what it always was.
    pub fn create_instance_buffer_lod(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raw: &[f32],
        instance_count: u32,
        lod: Option<InstanceLod>,
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
        // Per-instance (rank, tile floor) for `extra.xy`, by position in `order`.
        let mut ranks: Vec<[f32; 2]> = Vec::new();
        if count >= TILE_MIN_COUNT || (lod.is_some() && count > 0) {
            let mut xz_min = [f32::MAX; 2];
            let mut xz_max = [f32::MIN; 2];
            for i in 0..count {
                let p = &raw[i * 9..i * 9 + 3];
                if p[0] < xz_min[0] {
                    xz_min[0] = p[0];
                }
                if p[0] > xz_max[0] {
                    xz_max[0] = p[0];
                }
                if p[2] < xz_min[1] {
                    xz_min[1] = p[2];
                }
                if p[2] > xz_max[1] {
                    xz_max[1] = p[2];
                }
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
                while end < count && cell_of(order[end]) == cell {
                    end += 1;
                }
                let mut pmin = [f32::MAX; 3];
                let mut pmax = [f32::MIN; 3];
                let mut max_scale = 0.0f32;
                for &i in &order[start..end] {
                    let inst = &raw[i * 9..i * 9 + 9];
                    for a in 0..3 {
                        if inst[a] < pmin[a] {
                            pmin[a] = inst[a];
                        }
                        if inst[a] > pmax[a] {
                            pmax[a] = inst[a];
                        }
                    }
                    if inst[4].abs() > max_scale {
                        max_scale = inst[4].abs();
                    }
                }
                let mut lod_floor = 0.0;
                if let Some(lod) = lod {
                    order[start..end].sort_by_key(|&i| lod_shuffle_key(i));
                    let n = (end - start) as f32;
                    let cell_area = (ext_x / grid as f32) * (ext_z / grid as f32);
                    lod_floor = lod.tile_floor((end - start) as u32, cell_area);
                    ranks.extend((0..end - start).map(|k| [(k as f32 + 0.5) / n, lod_floor]));
                }
                tiles.push(InstanceTile {
                    first: start as u32,
                    count: (end - start) as u32,
                    pmin,
                    pmax,
                    max_scale,
                    lod_floor,
                });
                start = end;
            }
        }

        let mut packed: Vec<f32> = Vec::with_capacity(count * 12);
        for (k, &i) in order.iter().enumerate() {
            let off = i * 9;
            packed.extend_from_slice(&raw[off..off + 3]); // pos.xyz
            packed.push(raw[off + 3]); // rot_y
            packed.push(raw[off + 4]); // scale
            packed.extend_from_slice(&raw[off + 5..off + 9]); // tint.rgba
                                                              // extra.xyz: zero, or the EN-060 rank and tile floor in x, y.
            let [rank, floor] = ranks.get(k).copied().unwrap_or([0.0; 2]);
            packed.extend_from_slice(&[rank, floor, 0.0]);
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
            lod,
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
    pub fn create_dynamic_instance_buffer(&mut self, device: &wgpu::Device, capacity: u32) -> u32 {
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
            lod: None,
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
        if handle == 0 || count == 0 {
            return;
        }
        let idx = handle as usize - 1;
        let Some(Some(ib)) = self.instance_buffers.get(idx) else {
            return;
        };
        let n = (count as usize).min(packed.len() / 12);
        if n == 0 {
            return;
        }
        queue.write_buffer(&ib.buffer, 0, bytemuck::cast_slice(&packed[..n * 12]));
    }

    /// EN-001 — drop an instance buffer slot. The slot is left as
    /// `None` so previously-issued handles never alias a future
    /// allocation. No-op for `handle == 0` or out-of-range handles.
    pub fn destroy_instance_buffer(&mut self, handle: u32) {
        if handle == 0 {
            return;
        }
        let idx = handle as usize - 1;
        if idx < self.instance_buffers.len() {
            self.instance_buffers[idx] = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's half, transcribed: 0 = the instance is gone.
    fn visible(rank: f32, level: f32, band: f32) -> f32 {
        ((level - rank) / band).clamp(0.0, 1.0)
    }

    #[test]
    fn lod_parameters_that_mean_no_thinning_give_none() {
        assert!(InstanceLod::new(10.0, 40.0, 1.0, 0.1, 0.0).is_none());
        assert!(InstanceLod::new(f32::NAN, 40.0, 0.5, 0.1, 0.0).is_none());
        let lod = InstanceLod::new(40.0, 10.0, 0.5, 0.1, 0.0).unwrap();
        assert!(lod.end > lod.start);
    }

    #[test]
    fn every_instance_is_whole_up_to_start() {
        let lod = InstanceLod::new(20.0, 60.0, 0.4, 0.15, 0.0).unwrap();
        for d in [0.0, 5.0, 20.0] {
            assert_eq!(lod.prefix(128, d, 0.0), 128);
            // the highest rank a 128-instance tile holds
            assert_eq!(visible(127.5 / 128.0, lod.level(d, 0.0), lod.band), 1.0);
        }
    }

    #[test]
    fn level_falls_to_min_fraction_and_stays() {
        let lod = InstanceLod::new(20.0, 60.0, 0.4, 0.15, 0.0).unwrap();
        assert!((lod.level(60.0, 0.0) - 0.4).abs() < 1e-6);
        assert!((lod.level(500.0, 0.0) - 0.4).abs() < 1e-6);
        assert_eq!(lod.prefix(100, 500.0, 0.0), 40);
        let mut last = f32::MAX;
        for i in 0..200 {
            let l = lod.level(i as f32 * 0.5, 0.0);
            assert!(l <= last);
            last = l;
        }
    }

    /// The contract between the two halves: an instance the engine does
    /// not draw is one the shader has shrunk to nothing, at the tile's
    /// nearest distance and so at every instance's own (larger) distance.
    #[test]
    fn engine_only_drops_instances_the_shader_already_hid() {
        let lod = InstanceLod::new(18.0, 55.0, 0.35, 0.2, 0.0).unwrap();
        for floor in [0.0f32, 0.5, 0.9, 1.2] {
            for count in [1u32, 2, 7, 128, 300] {
                for step in 0..400 {
                    let d_tile = step as f32 * 0.25;
                    let n = lod.prefix(count, d_tile, floor);
                    assert!(n >= 1, "a tile never empties while min_fraction > 0");
                    for k in n..count {
                        let rank = (k as f32 + 0.5) / count as f32;
                        for further in [0.0f32, 0.3, 4.0] {
                            let level = lod.level(d_tile + further, floor);
                            let v = visible(rank, level, lod.band);
                            assert_eq!(v, 0.0, "count {count} k {k} d {d_tile}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_sparse_tile_is_never_thinned() {
        // 4 per square metre wanted; a 16 m2 tile.
        let lod = InstanceLod::new(10.0, 40.0, 0.3, 0.2, 4.0).unwrap();
        // 40 instances is 2.5 per m2, under the limit: every rank whole at any distance.
        let sparse = lod.tile_floor(40, 16.0);
        assert_eq!(lod.prefix(40, 1000.0, sparse), 40);
        assert_eq!(
            visible(39.5 / 40.0, lod.level(1000.0, sparse), lod.band),
            1.0
        );
        // 160 instances is 10 per m2: down to 4 per m2 (64 whole) and no further.
        let dense = lod.tile_floor(160, 16.0);
        let level = lod.level(1000.0, dense);
        let whole = (0..160)
            .filter(|&k| visible((k as f32 + 0.5) / 160.0, level, lod.band) == 1.0)
            .count();
        assert!((63..=65).contains(&whole), "{whole}");
        // and without a limit the tile goes down to min_fraction
        let free = InstanceLod::new(10.0, 40.0, 0.3, 0.2, 0.0).unwrap();
        assert_eq!(free.prefix(160, 1000.0, free.tile_floor(160, 16.0)), 48);
    }

    #[test]
    fn shuffle_spreads_a_prefix_over_the_authored_order() {
        // Half of a 128-instance tile, taken in shuffled order, should hit
        // all four quarters of the authored index range about evenly.
        let mut order: Vec<usize> = (0..128).collect();
        order.sort_by_key(|&i| lod_shuffle_key(i));
        let mut quarters = [0u32; 4];
        for &i in &order[..64] {
            quarters[i / 32] += 1;
        }
        for q in quarters {
            assert!((8..=24).contains(&q), "{quarters:?}");
        }
    }

    #[test]
    fn nearest_point_of_a_box() {
        let d = distance_to_aabb([0.0, 0.0, 0.0], [3.0, -1.0, -1.0], [5.0, 1.0, 1.0]);
        assert!((d - 3.0).abs() < 1e-6);
        assert_eq!(
            distance_to_aabb([4.0, 0.0, 0.0], [3.0, -1.0, -1.0], [5.0, 1.0, 1.0]),
            0.0
        );
    }
}
