//! CPU preparation for a scene SDF clipmap bake: the world-triangle gather
//! and the per-cell triangle binning the bake shader reads.
//!
//! The first bake of a scene runs this in one go. Every later bake is a
//! rebake of a live clipmap, and spreads the work over frames with the
//! per-frame budgets below, so a rebake costs a bounded slice of each frame
//! instead of one long frame. A scene whose transforms change every frame
//! (a rocking ship, a skinned crowd) rebakes continuously: before this, each
//! frame restarted the whole gather and binning and dropped the bake it had
//! started the frame before, so the clipmap cost its full preparation every
//! frame and was never published.

/// Vertices a rebake transforms to world space per frame.
pub(super) const REBAKE_GATHER_VERTICES_PER_FRAME: usize = 40_000;
/// Triangle-to-cell references a rebake counts or writes per frame.
pub(super) const REBAKE_BIN_REFS_PER_FRAME: usize = 400_000;

/// A scene SDF clipmap bake whose CPU preparation is still in progress.
pub(super) struct ClipmapPrep {
    /// Scene version when the gather began. The published clipmap carries
    /// it, so a scene that changed while the bake ran is baked again.
    pub(super) scene_version: u64,
    pub(super) transparent_gi: bool,
    pub(super) next_node: usize,
    pub(super) gathered: bool,
    pub(super) vertices: Vec<f32>,
    pub(super) indices: Vec<u32>,
    pub(super) tri_count: u32,
    /// Set with the binner, when the gather is complete.
    pub(super) origin: [f32; 3],
    pub(super) binner: Option<TriangleBinner>,
}

pub(super) fn clipmap_prep() -> &'static std::sync::Mutex<Option<ClipmapPrep>> {
    static INSTANCE: std::sync::OnceLock<std::sync::Mutex<Option<ClipmapPrep>>> =
        std::sync::OnceLock::new();
    INSTANCE.get_or_init(|| std::sync::Mutex::new(None))
}

/// Bins triangles into `cells`³ cells, each triangle's list expanded by one
/// cell (the bake shader's narrow band) so the per-cell clamp stays a
/// conservative lower bound for sphere tracing. A two-pass counting sort
/// (count, prefix-sum, fill) that can stop and resume between triangles.
pub(super) struct TriangleBinner {
    grid_min: [f32; 3],
    cell_size: f32,
    cells: usize,
    /// Per-cell counts in the first pass, per-cell write cursors in the second.
    counts: Vec<u32>,
    pub(super) offsets: Vec<u32>,
    pub(super) tri_refs: Vec<u32>,
    next_tri: usize,
    filling: bool,
}

impl TriangleBinner {
    pub(super) fn new(grid_min: [f32; 3], extent: f32, cells: usize) -> Self {
        Self {
            grid_min,
            cell_size: extent / cells as f32,
            cells,
            counts: vec![0u32; cells * cells * cells],
            offsets: Vec::new(),
            tri_refs: Vec::new(),
            next_tri: 0,
            filling: false,
        }
    }

    fn cell_range(
        &self,
        vertices: &[f32],
        indices: &[u32],
        tri: usize,
    ) -> Option<([usize; 3], [usize; 3])> {
        // Written out per corner: filling `[f32::MAX; 3]` first compiles to a
        // memset_pattern16 call per array, which was most of this function.
        let p0 = indices[tri * 3] as usize * 12;
        let p1 = indices[tri * 3 + 1] as usize * 12;
        let p2 = indices[tri * 3 + 2] as usize * 12;
        let (a, b, c) = (
            &vertices[p0..p0 + 3],
            &vertices[p1..p1 + 3],
            &vertices[p2..p2 + 3],
        );
        let lo = [
            a[0].min(b[0]).min(c[0]),
            a[1].min(b[1]).min(c[1]),
            a[2].min(b[2]).min(c[2]),
        ];
        let hi = [
            a[0].max(b[0]).max(c[0]),
            a[1].max(b[1]).max(c[1]),
            a[2].max(b[2]).max(c[2]),
        ];
        let cells = self.cells as f32;
        let mut c_lo = [0usize; 3];
        let mut c_hi = [0usize; 3];
        for a in 0..3 {
            // Expand by one cell width (= the shader's band).
            let lo_c = ((lo[a] - self.cell_size - self.grid_min[a]) / self.cell_size).floor();
            let hi_c = ((hi[a] + self.cell_size - self.grid_min[a]) / self.cell_size).floor();
            if hi_c < 0.0 || lo_c >= cells {
                return None; // entirely outside the clipmap volume
            }
            c_lo[a] = lo_c.max(0.0) as usize;
            c_hi[a] = hi_c.min(cells - 1.0) as usize;
        }
        Some((c_lo, c_hi))
    }

    /// Advance by about `budget` cell references. Returns true once `offsets`
    /// and `tri_refs` are complete.
    pub(super) fn step(
        &mut self,
        vertices: &[f32],
        indices: &[u32],
        tri_count: usize,
        budget: usize,
    ) -> bool {
        let cells = self.cells;
        let mut spent = 0usize;
        if !self.filling {
            while self.next_tri < tri_count {
                if spent >= budget {
                    return false;
                }
                if let Some((lo, hi)) = self.cell_range(vertices, indices, self.next_tri) {
                    for z in lo[2]..=hi[2] {
                        for y in lo[1]..=hi[1] {
                            for x in lo[0]..=hi[0] {
                                self.counts[(z * cells + y) * cells + x] += 1;
                            }
                        }
                    }
                    spent += (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1);
                }
                spent += 1;
                self.next_tri += 1;
            }
            let cell_count = self.counts.len();
            self.offsets = vec![0u32; cell_count + 1];
            for i in 0..cell_count {
                self.offsets[i + 1] = self.offsets[i] + self.counts[i];
            }
            self.counts.copy_from_slice(&self.offsets[..cell_count]);
            // wgpu rejects zero-sized buffers — keep one dummy entry when no
            // triangle touches the volume (all cells then read empty ranges).
            self.tri_refs = vec![0u32; (self.offsets[cell_count] as usize).max(1)];
            self.filling = true;
            self.next_tri = 0;
        }
        while self.next_tri < tri_count {
            if spent >= budget {
                return false;
            }
            if let Some((lo, hi)) = self.cell_range(vertices, indices, self.next_tri) {
                for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            let ci = (z * cells + y) * cells + x;
                            self.tri_refs[self.counts[ci] as usize] = self.next_tri as u32;
                            self.counts[ci] += 1;
                        }
                    }
                }
                spent += (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1);
            }
            spent += 1;
            self.next_tri += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A few triangles of different sizes, one of them outside the grid.
    fn soup() -> (Vec<f32>, Vec<u32>, usize) {
        let corners: [[f32; 3]; 12] = [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
            [-6.0, -6.0, -6.0],
            [6.0, -6.0, -6.0],
            [6.0, 6.0, 6.0],
            [3.0, -2.0, 1.0],
            [3.2, -2.0, 1.0],
            [3.0, -1.8, 1.2],
            [90.0, 90.0, 90.0],
            [91.0, 90.0, 90.0],
            [90.0, 91.0, 90.0],
        ];
        let mut vertices = Vec::new();
        for c in corners {
            vertices.extend_from_slice(&[c[0], c[1], c[2]]);
            vertices.extend_from_slice(&[0.0; 9]);
        }
        (vertices, (0..12).collect(), 4)
    }

    #[test]
    fn stepped_binning_matches_one_shot() {
        let (vertices, indices, tri_count) = soup();
        let mut whole = TriangleBinner::new([-8.0; 3], 16.0, 8);
        assert!(whole.step(&vertices, &indices, tri_count, usize::MAX));
        assert_eq!(whole.offsets.len(), 8 * 8 * 8 + 1);
        assert!(whole.tri_refs.len() > 27 * 2);
        assert!(
            !whole.tri_refs.contains(&3),
            "the far triangle is binned nowhere"
        );

        let mut stepped = TriangleBinner::new([-8.0; 3], 16.0, 8);
        let mut steps = 1;
        while !stepped.step(&vertices, &indices, tri_count, 5) {
            steps += 1;
            assert!(steps < 1000, "binning must terminate");
        }
        assert!(steps > 2, "a small budget spreads the work over calls");
        assert_eq!(stepped.offsets, whole.offsets);
        assert_eq!(stepped.tri_refs, whole.tri_refs);
    }
}
