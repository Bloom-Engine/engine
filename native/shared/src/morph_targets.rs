//! Morph targets (glTF blend shapes): the data a mesh keeps, and the CPU
//! reference of what the GPU pass computes. `docs/morph-targets.md` is the
//! overview.
//!
//! A face rig moves a few thousand of a character's vertices with each of
//! its ~50 shapes; most of the mesh (the body, the clothes) is touched by
//! none. So the deltas are stored sparse, as a per-vertex list (CSR):
//!
//! - `ranges[v]..ranges[v + 1]` are vertex `v`'s entries in `entries`, so an
//!   untouched vertex costs one `u32` and a touched one its entries only;
//! - an entry is one target's delta on one vertex: the model-level morph
//!   slot it belongs to and the position, normal and tangent deltas as half
//!   floats (24 bytes). Positions are stored divided by `pos_scale` (the
//!   largest position delta component of the mesh) so the halves always lie
//!   in [-1, 1]: no overflow whatever the units, and the precision is a
//!   fixed fraction (2^-11) of the largest delta.
//!
//! The per-vertex layout is what the GPU wants: one thread per vertex sums
//! its own entries, no atomics, no scatter. Entries of a vertex are sorted
//! by slot.
//!
//! Slots are model-level: every mesh of a model numbers its targets by the
//! model's name table, so "jawOpen" on the face, the teeth and the tongue is
//! one slot and one weight drives all three.

use crate::renderer::Vertex3D;
use std::sync::Arc;

/// Below this a delta component counts as zero (an exporter writes the
/// whole mesh for every target, untouched vertices included).
pub const MORPH_DELTA_EPSILON: f32 = 1e-6;

/// Most slots a model or an animation instance may have.
pub const MAX_MORPH_SLOTS: usize = 1024;

/// One target's delta on one vertex, as the GPU reads it (6 x u32).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MorphEntry {
    /// Model-level morph slot.
    pub slot: u32,
    /// Half floats, low half first (WGSL `unpack2x16float` order):
    /// [pos.x pos.y] [pos.z nrm.x] [nrm.y nrm.z] [tan.x tan.y] [tan.z 0].
    /// Positions are divided by `MorphTargets::pos_scale`.
    pub packed: [u32; 5],
}

/// Sparse morph deltas of one primitive. Immutable after load and shared
/// (via `Arc`) between every placement and every instance of the mesh;
/// the weights are per animation instance.
#[derive(Debug)]
pub struct MorphTargets {
    /// The model's slot names (`targetNames`), the same `Arc` on every
    /// mesh of one load. Unnamed targets are called `"<mesh>#<n>"`.
    pub names: Arc<Vec<String>>,
    pub vertex_count: u32,
    /// `vertex_count + 1` offsets into `entries`.
    pub ranges: Vec<u32>,
    pub entries: Vec<MorphEntry>,
    pub pos_scale: f32,
    /// Per axis, the most any vertex can move with every weight in [-1, 1]:
    /// max over vertices of the sum of |position delta|. Bounds grow by it.
    pub bound_growth: [f32; 3],
    /// Vertices with at least one entry.
    pub touched_vertices: u32,
    pub has_normals: bool,
    pub has_tangents: bool,
}

/// Decoded delta of one entry: position (already times `pos_scale`), normal,
/// tangent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphDelta {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
}

fn half_bits(value: f32) -> u32 {
    u32::from(half::f16::from_f32(value).to_bits())
}

fn half_value(bits: u32) -> f32 {
    half::f16::from_bits((bits & 0xffff) as u16).to_f32()
}

fn pack2(lo: f32, hi: f32) -> u32 {
    half_bits(lo) | (half_bits(hi) << 16)
}

fn unpack2(word: u32) -> (f32, f32) {
    (half_value(word), half_value(word >> 16))
}

/// Collects targets of one primitive, then packs them.
pub struct MorphTargetsBuilder {
    vertex_count: usize,
    /// Per vertex: (slot, position, normal, tangent) in f32.
    per_vertex: Vec<Vec<(u32, [f32; 9])>>,
    has_normals: bool,
    has_tangents: bool,
}

impl MorphTargetsBuilder {
    pub fn new(vertex_count: usize) -> Self {
        Self {
            vertex_count,
            per_vertex: vec![Vec::new(); vertex_count],
            has_normals: false,
            has_tangents: false,
        }
    }

    /// Add one target. Each stream is per vertex (missing = no delta);
    /// a stream of the wrong length is ignored. `transform` maps a vertex's
    /// delta (position, normal, tangent as 9 floats) into the space the
    /// loader put the vertex in (skin scale, baked node transform).
    pub fn add_target(
        &mut self,
        slot: u32,
        positions: Option<&[[f32; 3]]>,
        normals: Option<&[[f32; 3]]>,
        tangents: Option<&[[f32; 3]]>,
        transform: impl Fn(usize, [f32; 9]) -> [f32; 9],
    ) {
        let n = self.vertex_count;
        let positions = positions.filter(|s| s.len() == n);
        let normals = normals.filter(|s| s.len() == n);
        let tangents = tangents.filter(|s| s.len() == n);
        self.has_normals |= normals.is_some();
        self.has_tangents |= tangents.is_some();
        for v in 0..n {
            let mut delta = [0.0f32; 9];
            if let Some(p) = positions {
                delta[0..3].copy_from_slice(&p[v]);
            }
            if let Some(nr) = normals {
                delta[3..6].copy_from_slice(&nr[v]);
            }
            if let Some(t) = tangents {
                delta[6..9].copy_from_slice(&t[v]);
            }
            if delta
                .iter()
                .all(|c| c.abs() < MORPH_DELTA_EPSILON || !c.is_finite())
            {
                continue;
            }
            let mut delta = transform(v, delta);
            for c in delta.iter_mut() {
                if !c.is_finite() || c.abs() < MORPH_DELTA_EPSILON {
                    *c = 0.0;
                }
            }
            if delta.iter().all(|c| *c == 0.0) {
                continue;
            }
            let list = &mut self.per_vertex[v];
            // Two targets of one primitive with the same name share a slot:
            // their deltas add, as two shapes at the same weight would.
            if let Some(existing) = list.iter_mut().find(|(s, _)| *s == slot) {
                for (a, b) in existing.1.iter_mut().zip(delta) {
                    *a += b;
                }
            } else {
                list.push((slot, delta));
            }
        }
    }

    /// Pack. `None` when no target moved any vertex.
    pub fn build(mut self, names: Arc<Vec<String>>) -> Option<MorphTargets> {
        let mut pos_scale = 0.0f32;
        let mut count = 0usize;
        for list in &mut self.per_vertex {
            list.sort_by_key(|(slot, _)| *slot);
            count += list.len();
            for (_, d) in list.iter() {
                for c in &d[0..3] {
                    pos_scale = pos_scale.max(c.abs());
                }
            }
        }
        if count == 0 {
            return None;
        }
        if pos_scale <= 0.0 {
            pos_scale = 1.0;
        }
        // Normal and tangent deltas of unit vectors lie in [-2, 2]; anything
        // past the half range is broken data, clamped rather than packed as
        // infinity.
        let clamp_dir = |c: f32| c.clamp(-60000.0, 60000.0);
        let mut ranges = Vec::with_capacity(self.vertex_count + 1);
        let mut entries = Vec::with_capacity(count);
        let mut bound_growth = [0.0f32; 3];
        let mut touched = 0u32;
        for list in &self.per_vertex {
            ranges.push(entries.len() as u32);
            if !list.is_empty() {
                touched += 1;
            }
            let mut sum = [0.0f32; 3];
            for (slot, d) in list {
                let p = [d[0] / pos_scale, d[1] / pos_scale, d[2] / pos_scale];
                let entry = MorphEntry {
                    slot: *slot,
                    packed: [
                        pack2(p[0], p[1]),
                        pack2(p[2], clamp_dir(d[3])),
                        pack2(clamp_dir(d[4]), clamp_dir(d[5])),
                        pack2(clamp_dir(d[6]), clamp_dir(d[7])),
                        pack2(clamp_dir(d[8]), 0.0),
                    ],
                };
                let decoded = decode_entry(&entry, pos_scale);
                for a in 0..3 {
                    sum[a] += decoded.position[a].abs();
                }
                entries.push(entry);
            }
            for a in 0..3 {
                bound_growth[a] = bound_growth[a].max(sum[a]);
            }
        }
        ranges.push(entries.len() as u32);
        Some(MorphTargets {
            names,
            vertex_count: self.vertex_count as u32,
            ranges,
            entries,
            pos_scale,
            bound_growth,
            touched_vertices: touched,
            has_normals: self.has_normals,
            has_tangents: self.has_tangents,
        })
    }
}

/// Decode one entry.
pub fn decode_entry(entry: &MorphEntry, pos_scale: f32) -> MorphDelta {
    let (px, py) = unpack2(entry.packed[0]);
    let (pz, nx) = unpack2(entry.packed[1]);
    let (ny, nz) = unpack2(entry.packed[2]);
    let (tx, ty) = unpack2(entry.packed[3]);
    let (tz, _) = unpack2(entry.packed[4]);
    MorphDelta {
        position: [px * pos_scale, py * pos_scale, pz * pos_scale],
        normal: [nx, ny, nz],
        tangent: [tx, ty, tz],
    }
}

fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

impl MorphTargets {
    /// Entries of vertex `v`.
    pub fn vertex_entries(&self, v: usize) -> &[MorphEntry] {
        let lo = self.ranges[v] as usize;
        let hi = self.ranges[v + 1] as usize;
        &self.entries[lo..hi]
    }

    /// Bytes this mesh's deltas take (CPU copy; the GPU copy is the same).
    pub fn byte_size(&self) -> usize {
        self.ranges.len() * 4 + self.entries.len() * std::mem::size_of::<MorphEntry>()
    }

    /// The morphed vertex `v` of `base` at `weights` (indexed by slot,
    /// missing slots are 0): exactly what the GPU pass writes. Position adds
    /// the weighted deltas; normal and tangent add theirs and are
    /// renormalised (a tangent that was the zero "no tangent" sentinel stays
    /// zero; the tangent's handedness `w` is kept).
    pub fn morph_vertex(&self, base: &Vertex3D, v: usize, weights: &[f32]) -> Vertex3D {
        let mut out = *base;
        let (dp, dn, dt) = self.weighted(v, weights);
        for a in 0..3 {
            out.position[a] += dp[a];
        }
        if self.has_normals {
            let n = [
                base.normal[0] + dn[0],
                base.normal[1] + dn[1],
                base.normal[2] + dn[2],
            ];
            out.normal = normalize_or(n, base.normal);
        }
        let bt = [base.tangent[0], base.tangent[1], base.tangent[2]];
        if self.has_tangents && bt.iter().any(|c| *c != 0.0) {
            let t = normalize_or([bt[0] + dt[0], bt[1] + dt[1], bt[2] + dt[2]], bt);
            out.tangent = [t[0], t[1], t[2], base.tangent[3]];
        }
        out
    }

    /// The weighted position delta of vertex `v`.
    pub fn position_offset(&self, v: usize, weights: &[f32]) -> [f32; 3] {
        self.weighted(v, weights).0
    }

    fn weighted(&self, v: usize, weights: &[f32]) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let mut dp = [0.0f32; 3];
        let mut dn = [0.0f32; 3];
        let mut dt = [0.0f32; 3];
        for entry in self.vertex_entries(v) {
            let w = weights.get(entry.slot as usize).copied().unwrap_or(0.0);
            if w == 0.0 {
                continue;
            }
            let d = decode_entry(entry, self.pos_scale);
            for a in 0..3 {
                dp[a] += w * d.position[a];
                dn[a] += w * d.normal[a];
                dt[a] += w * d.tangent[a];
            }
        }
        (dp, dn, dt)
    }

    /// Grow an AABB of the rest mesh so it holds every pose the targets can
    /// make with weights in [-1, 1].
    pub fn grow_bounds(&self, min: &mut [f32; 3], max: &mut [f32; 3]) {
        if min[0] > max[0] {
            return;
        }
        for a in 0..3 {
            min[a] -= self.bound_growth[a];
            max[a] += self.bound_growth[a];
        }
    }

    /// A copy with the deltas of some vertices mapped by `transform` (the
    /// loader's bake of a node transform into rigid vertices).
    pub fn mapped(
        &self,
        mut transform: impl FnMut(usize, MorphDelta) -> Option<MorphDelta>,
    ) -> Option<MorphTargets> {
        let mut builder = MorphTargetsBuilder::new(self.vertex_count as usize);
        builder.has_normals = self.has_normals;
        builder.has_tangents = self.has_tangents;
        for v in 0..self.vertex_count as usize {
            for entry in self.vertex_entries(v) {
                let d = decode_entry(entry, self.pos_scale);
                let d = transform(v, d).unwrap_or(d);
                let mut packed = [0.0f32; 9];
                packed[0..3].copy_from_slice(&d.position);
                packed[3..6].copy_from_slice(&d.normal);
                packed[6..9].copy_from_slice(&d.tangent);
                builder.per_vertex[v].push((entry.slot, packed));
            }
        }
        builder.build(self.names.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(position: [f32; 3]) -> Vertex3D {
        Vertex3D {
            position,
            normal: [0.0, 0.0, 1.0],
            color: [1.0; 4],
            uv: [0.0; 2],
            joints: [0.0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
    }

    fn names(n: usize) -> Arc<Vec<String>> {
        Arc::new((0..n).map(|i| format!("t{i}")).collect())
    }

    #[test]
    fn untouched_vertices_cost_no_entries_and_entries_sort_by_slot() {
        let mut b = MorphTargetsBuilder::new(5);
        // Target 1 touches vertices 1 and 3; target 0 only vertex 3. The
        // exporter wrote every vertex: the zeros must be dropped.
        let t1 = [
            [0.0; 3],
            [0.0, 0.25, 0.0],
            [0.0; 3],
            [0.5, 0.0, 0.0],
            [1e-8, 0.0, 0.0],
        ];
        let t0 = [[0.0; 3], [0.0; 3], [0.0; 3], [0.0, 0.0, -0.125], [0.0; 3]];
        b.add_target(1, Some(&t1), None, None, |_, d| d);
        b.add_target(0, Some(&t0), None, None, |_, d| d);
        let m = b.build(names(2)).expect("targets moved vertices");
        assert_eq!(m.ranges, vec![0, 0, 1, 1, 3, 3]);
        assert_eq!(m.touched_vertices, 2);
        assert_eq!(m.entries.len(), 3);
        let slots: Vec<u32> = m.vertex_entries(3).iter().map(|e| e.slot).collect();
        assert_eq!(slots, vec![0, 1]);
        assert_eq!(m.pos_scale, 0.5);
        // 24 B an entry, 4 B a vertex.
        assert_eq!(m.byte_size(), 6 * 4 + 3 * 24);
    }

    #[test]
    fn nothing_moved_builds_nothing() {
        let mut b = MorphTargetsBuilder::new(3);
        b.add_target(
            0,
            Some(&[[0.0; 3]; 3]),
            Some(&[[0.0; 3]; 3]),
            None,
            |_, d| d,
        );
        assert!(b.build(names(1)).is_none());
    }

    #[test]
    fn half_packing_keeps_a_fixed_fraction_of_the_largest_delta() {
        // Centimetre-scale units (deltas of tens) and metre-scale (hundredths)
        // must both survive: positions are normalised before packing.
        for scale in [0.01f32, 1.0, 100.0, 30000.0] {
            let mut b = MorphTargetsBuilder::new(2);
            let p = [
                [0.3 * scale, -0.7 * scale, 0.0123 * scale],
                [scale, 0.0, 0.0],
            ];
            b.add_target(0, Some(&p), None, None, |_, d| d);
            let m = b.build(names(1)).unwrap();
            let d = decode_entry(&m.vertex_entries(0)[0], m.pos_scale);
            for a in 0..3 {
                let err = (d.position[a] - p[0][a]).abs();
                assert!(err <= scale * 1.0 / 2048.0, "scale {scale} axis {a}: {err}");
            }
        }
    }

    #[test]
    fn cpu_morph_adds_weighted_deltas_and_renormalises() {
        let mut b = MorphTargetsBuilder::new(1);
        b.add_target(
            0,
            Some(&[[0.0, 1.0, 0.0]]),
            Some(&[[0.0, 1.0, -1.0]]),
            None,
            |_, d| d,
        );
        b.add_target(2, Some(&[[0.5, 0.0, 0.0]]), None, None, |_, d| d);
        let m = b.build(names(3)).unwrap();
        let base = vertex([1.0, 2.0, 3.0]);
        let out = m.morph_vertex(&base, 0, &[0.5, 9.0, 1.0]);
        assert!((out.position[0] - 1.5).abs() < 1e-3);
        assert!((out.position[1] - 2.5).abs() < 1e-3);
        assert!((out.position[2] - 3.0).abs() < 1e-6);
        // normal (0,0,1) + 0.5*(0,1,-1) = (0,0.5,0.5) -> normalised
        let r = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out.normal[1] - r).abs() < 1e-3 && (out.normal[2] - r).abs() < 1e-3);
        // no tangent stream: tangent untouched
        assert_eq!(out.tangent, base.tangent);
        // all weights zero: the bind pose exactly
        let rest = m.morph_vertex(&base, 0, &[0.0; 3]);
        assert_eq!(rest.position, base.position);
        assert_eq!(rest.normal, base.normal);
        // slots past the weights are 0
        assert_eq!(m.morph_vertex(&base, 0, &[]).position, base.position);
    }

    #[test]
    fn bounds_grow_by_the_most_any_vertex_can_move() {
        let mut b = MorphTargetsBuilder::new(2);
        b.add_target(
            0,
            Some(&[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]),
            None,
            None,
            |_, d| d,
        );
        b.add_target(
            1,
            Some(&[[-0.3, 0.0, 0.0], [0.0, 0.2, 0.05]]),
            None,
            None,
            |_, d| d,
        );
        let m = b.build(names(2)).unwrap();
        let g = m.bound_growth;
        assert!((g[0] - 0.4).abs() < 1e-3, "{g:?}");
        assert!((g[1] - 0.4).abs() < 1e-3, "{g:?}");
        assert!((g[2] - 0.05).abs() < 1e-3, "{g:?}");
        let (mut lo, mut hi) = ([0.0; 3], [1.0; 3]);
        m.grow_bounds(&mut lo, &mut hi);
        assert!(lo[0] < -0.39 && hi[0] > 1.39);
        // the empty-AABB sentinel stays empty
        let (mut lo, mut hi) = ([1.0; 3], [-1.0; 3]);
        m.grow_bounds(&mut lo, &mut hi);
        assert!(lo[0] > hi[0]);
    }

    #[test]
    fn a_transform_maps_the_deltas_of_the_vertices_it_names() {
        let mut b = MorphTargetsBuilder::new(2);
        b.add_target(
            0,
            Some(&[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            None,
            None,
            |v, mut d| {
                if v == 1 {
                    for c in &mut d[0..3] {
                        *c *= 100.0;
                    }
                }
                d
            },
        );
        let m = b.build(names(1)).unwrap();
        assert!((m.position_offset(0, &[1.0])[0] - 1.0).abs() < 0.05);
        assert!((m.position_offset(1, &[1.0])[0] - 100.0).abs() < 0.05);
        let swapped = m
            .mapped(|v, d| {
                (v == 0).then_some(MorphDelta {
                    position: [0.0, d.position[0], 0.0],
                    ..d
                })
            })
            .unwrap();
        assert!((swapped.position_offset(0, &[1.0])[1] - 1.0).abs() < 0.05);
        assert!((swapped.position_offset(1, &[1.0])[0] - 100.0).abs() < 0.05);
    }
}
