//! World-space triangle soup for the scene distance field.
//!
//! `min_feature` (world units, 0 = keep everything) thins triangles that are
//! far smaller than the field's voxels. A field sampled every 62 cm gains
//! nothing from a character's 8 mm triangles, yet the bake tested each voxel
//! against every one of them. Of the triangles with area below
//! `min_feature`², only the first to fall in each `min_feature`-sized world
//! cell is kept, so a dense surface stays covered by samples about
//! `min_feature` apart and the distance to it moves by at most that. Larger
//! triangles are untouched.

use crate::renderer::Vertex3D;

/// Floats per soup vertex: the world position and nothing else. The bake
/// shader wants the `Vertex3D` stride, and gets it when the triangles a bake
/// uses are packed for upload (`sdf_clipmap_prep::pack_binned_triangles`); a
/// soup kept at that stride was three quarters zero padding, 216 MB for the
/// streets of a city.
pub(crate) const STRIDE: usize = 3;

fn push_position(vbuf: &mut Vec<f32>, t: &[[f32; 4]; 4], p: [f32; 3]) {
    vbuf.extend_from_slice(&[
        t[0][0] * p[0] + t[1][0] * p[1] + t[2][0] * p[2] + t[3][0],
        t[0][1] * p[0] + t[1][1] * p[1] + t[2][1] * p[2] + t[3][1],
        t[0][2] * p[0] + t[1][2] * p[1] + t[2][2] * p[2] + t[3][2],
    ]);
}

/// True when a node's world box (`bmin`..`bmax`) lies wholly outside the
/// region `lo`..`hi`. An invalid box (`bmin[0] > bmax[0]`: the node has not
/// been through `prepare()` yet) is never outside, so the node is kept.
pub(crate) fn bounds_outside(bmin: [f32; 3], bmax: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> bool {
    if bmin[0] > bmax[0] {
        return false;
    }
    (0..3).any(|a| bmax[a] < lo[a] || bmin[a] > hi[a])
}

/// The region a gather for a field centred on `centre` should cover: the
/// field's own box (`extent` wide) grown by `margin` on every side, so the
/// camera can travel `margin` before the soup has to be gathered again.
pub(crate) fn gather_region(centre: [f32; 3], extent: f32, margin: f32) -> ([f32; 3], [f32; 3]) {
    let h = extent * 0.5 + margin;
    (
        [centre[0] - h, centre[1] - h, centre[2] - h],
        [centre[0] + h, centre[1] + h, centre[2] + h],
    )
}

/// True when the field box (`extent` wide, centred on `centre`) fits inside `region`.
pub(crate) fn region_covers(region: ([f32; 3], [f32; 3]), centre: [f32; 3], extent: f32) -> bool {
    let h = extent * 0.5;
    (0..3).all(|a| centre[a] - h >= region.0[a] && centre[a] + h <= region.1[a])
}

/// True when the box `bmin`..`bmax` lies wholly inside the region `lo`..`hi`.
/// An invalid box (see `bounds_outside`) is not inside anything.
fn bounds_inside(bmin: [f32; 3], bmax: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> bool {
    bmin[0] <= bmax[0] && (0..3).all(|a| bmin[a] >= lo[a] && bmax[a] <= hi[a])
}

/// Append the part of one mesh that a gather of `region` wants, in world
/// space; returns the triangles appended. `bounds` is the mesh's world box.
/// A mesh wholly outside the region adds nothing, one wholly inside goes in
/// as `append_node_triangles` emits it, and one that straddles the edge gives
/// only the triangles that reach into the region: a 500 m ground tile under a
/// 120 m region would otherwise put all of itself into every soup.
pub(crate) fn append_node_triangles_in(
    vertices: &[Vertex3D],
    indices: &[u32],
    world_transform: &[[f32; 4]; 4],
    min_feature: f32,
    region: Option<([f32; 3], [f32; 3])>,
    bounds: ([f32; 3], [f32; 3]),
    vbuf: &mut Vec<f32>,
    ibuf: &mut Vec<u32>,
) -> u32 {
    let clip = match region {
        Some((lo, hi)) if bounds_outside(bounds.0, bounds.1, lo, hi) => return 0,
        Some((lo, hi)) if !bounds_inside(bounds.0, bounds.1, lo, hi) => Some((lo, hi)),
        _ => None,
    };
    append_triangles(
        vertices,
        indices,
        world_transform,
        min_feature,
        clip,
        vbuf,
        ibuf,
    )
}

/// Append one mesh in world space; returns the triangles appended.
#[cfg(test)]
pub(crate) fn append_node_triangles(
    vertices: &[Vertex3D],
    indices: &[u32],
    world_transform: &[[f32; 4]; 4],
    min_feature: f32,
    vbuf: &mut Vec<f32>,
    ibuf: &mut Vec<u32>,
) -> u32 {
    append_triangles(
        vertices,
        indices,
        world_transform,
        min_feature,
        None,
        vbuf,
        ibuf,
    )
}

fn append_triangles(
    vertices: &[Vertex3D],
    indices: &[u32],
    world_transform: &[[f32; 4]; 4],
    min_feature: f32,
    clip: Option<([f32; 3], [f32; 3])>,
    vbuf: &mut Vec<f32>,
    ibuf: &mut Vec<u32>,
) -> u32 {
    let base = (vbuf.len() / STRIDE) as u32;
    let thinned = if min_feature > 0.0 || clip.is_some() {
        select_triangles(vertices, indices, world_transform, min_feature, clip)
    } else {
        None
    };
    if let Some(kept) = thinned {
        // Only the surviving triangles' corners are emitted.
        vbuf.reserve(kept.len() * STRIDE);
        for (slot, &index) in kept.iter().enumerate() {
            push_position(vbuf, world_transform, vertices[index as usize].position);
            ibuf.push(base + slot as u32);
        }
        return (kept.len() / 3) as u32;
    }
    vbuf.reserve(vertices.len() * STRIDE);
    for vertex in vertices {
        push_position(vbuf, world_transform, vertex.position);
    }
    ibuf.extend(indices.iter().map(|&index| base + index));
    (indices.len() / 3) as u32
}

/// The corner indices of the triangles that survive clipping to `clip` and
/// thinning, or `None` when nothing was dropped and the mesh should be
/// emitted as it is.
fn select_triangles(
    vertices: &[Vertex3D],
    indices: &[u32],
    world_transform: &[[f32; 4]; 4],
    min_feature: f32,
    clip: Option<([f32; 3], [f32; 3])>,
) -> Option<Vec<u32>> {
    // Areas are measured in object space and scaled by the transform's mean
    // area scale; shear and non-uniform scale only move the threshold a little.
    let t = world_transform;
    let axis = |c: usize| (t[c][0] * t[c][0] + t[c][1] * t[c][1] + t[c][2] * t[c][2]).sqrt();
    let (sx, sy, sz) = (axis(0), axis(1), axis(2));
    let area_scale = (sx * sy + sy * sz + sz * sx) / 3.0;
    // Without a `min_feature` no triangle counts as small.
    let (small_area, inv_cell) = if min_feature > 0.0 {
        (min_feature * min_feature / area_scale, 1.0 / min_feature)
    } else {
        (0.0, 0.0)
    };
    if !small_area.is_finite() || !inv_cell.is_finite() {
        return None;
    }
    let world = |p: [f32; 3]| {
        [
            t[0][0] * p[0] + t[1][0] * p[1] + t[2][0] * p[2] + t[3][0],
            t[0][1] * p[0] + t[1][1] * p[1] + t[2][1] * p[2] + t[3][1],
            t[0][2] * p[0] + t[1][2] * p[1] + t[2][2] * p[2] + t[3][2],
        ]
    };
    let mut occupied: std::collections::HashSet<[i32; 3]> = Default::default();
    let mut kept: Vec<u32> = Vec::new();
    let mut dropped_any = false;
    for (triangle, corners) in indices.chunks_exact(3).enumerate() {
        let position = |corner: u32| vertices.get(corner as usize).map(|v| v.position);
        let (Some(a), Some(b), Some(c)) = (
            position(corners[0]),
            position(corners[1]),
            position(corners[2]),
        ) else {
            return None;
        };
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        // A triangle whose box misses the clip region is left out.
        let reaches_in = clip.is_none_or(|(lo, hi)| {
            let (wa, wb, wc) = (world(a), world(b), world(c));
            (0..3).all(|k| {
                wa[k].max(wb[k]).max(wc[k]) >= lo[k] && wa[k].min(wb[k]).min(wc[k]) <= hi[k]
            })
        });
        // A small triangle survives only as the first one in its world cell.
        let keep = reaches_in
            && (area >= small_area || {
                let m = [
                    (a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0,
                ];
                let cell = |row: usize| {
                    let world = t[0][row] * m[0] + t[1][row] * m[1] + t[2][row] * m[2] + t[3][row];
                    (world * inv_cell).floor() as i32
                };
                occupied.insert([cell(0), cell(1), cell(2)])
            });
        if keep {
            if dropped_any {
                kept.extend_from_slice(corners);
            }
        } else if !dropped_any {
            // First drop: everything before it was kept.
            dropped_any = true;
            kept.reserve(indices.len() / 4);
            kept.extend_from_slice(&indices[..triangle * 3]);
        }
    }
    dropped_any.then_some(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn a_node_is_left_out_only_when_wholly_outside_the_region() {
        let (lo, hi) = gather_region([100.0, 0.0, 100.0], 40.0, 40.0);
        assert_eq!(lo, [40.0, -60.0, 40.0]);
        assert_eq!(hi, [160.0, 60.0, 160.0]);
        // far away
        assert!(bounds_outside(
            [900.0, 0.0, 0.0],
            [910.0, 5.0, 10.0],
            lo,
            hi
        ));
        // reaching into the region
        assert!(!bounds_outside(
            [20.0, 0.0, 90.0],
            [45.0, 5.0, 110.0],
            lo,
            hi
        ));
        // a great ground plane that contains it
        assert!(!bounds_outside(
            [-5000.0, -1.0, -5000.0],
            [5000.0, 0.0, 5000.0],
            lo,
            hi
        ));
        // not yet through prepare(): kept
        assert!(!bounds_outside([f32::MAX; 3], [f32::MIN; 3], lo, hi));
    }

    #[test]
    fn a_region_covers_the_field_until_the_camera_leaves_the_margin() {
        let region = gather_region([0.0, 0.0, 0.0], 40.0, 40.0);
        assert!(region_covers(region, [0.0, 0.0, 0.0], 40.0));
        assert!(region_covers(region, [39.0, 0.0, -39.0], 40.0));
        assert!(!region_covers(region, [41.0, 0.0, 0.0], 40.0));
    }

    /// A flat `cells`×`cells` grid of side `size`, two triangles per cell.
    fn grid(cells: u32, size: f32) -> (Vec<Vertex3D>, Vec<u32>) {
        let mut vertices = Vec::new();
        for y in 0..=cells {
            for x in 0..=cells {
                vertices.push(Vertex3D {
                    position: [
                        x as f32 * size / cells as f32,
                        0.0,
                        y as f32 * size / cells as f32,
                    ],
                    ..Default::default()
                });
            }
        }
        let mut indices = Vec::new();
        let row = cells + 1;
        for y in 0..cells {
            for x in 0..cells {
                let a = y * row + x;
                indices.extend_from_slice(&[a, a + 1, a + row, a + 1, a + row + 1, a + row]);
            }
        }
        (vertices, indices)
    }

    #[test]
    fn large_triangles_are_emitted_unchanged() {
        let (vertices, indices) = grid(4, 4.0); // 0.5 m² triangles
        let (mut vbuf, mut ibuf) = (Vec::new(), Vec::new());
        let count =
            append_node_triangles(&vertices, &indices, &IDENTITY, 0.08, &mut vbuf, &mut ibuf);
        assert_eq!(count, 32);
        assert_eq!(ibuf, indices);
        assert_eq!(vbuf.len(), vertices.len() * STRIDE);
    }

    #[test]
    fn dense_mesh_is_thinned_to_one_sample_per_cell_and_stays_covered() {
        // 1 m² in 20 000 triangles of 0.5 cm² each; samples 10 cm apart.
        let (vertices, indices) = grid(100, 1.0);
        let (mut vbuf, mut ibuf) = (Vec::new(), Vec::new());
        let count =
            append_node_triangles(&vertices, &indices, &IDENTITY, 0.1, &mut vbuf, &mut ibuf);
        assert!((100..=121).contains(&count), "kept {count}");
        assert_eq!(ibuf.len() as u32, count * 3);
        assert_eq!(vbuf.len(), ibuf.len() * STRIDE);
        // Every point of the sheet is still within one sample spacing of a
        // surviving triangle.
        let mut worst = 0.0f32;
        for gy in 0..=10 {
            for gx in 0..=10 {
                let (px, pz) = (gx as f32 * 0.1, gy as f32 * 0.1);
                let nearest = vbuf
                    .chunks_exact(STRIDE)
                    .map(|v| ((v[0] - px).powi(2) + (v[2] - pz).powi(2)).sqrt())
                    .fold(f32::MAX, f32::min);
                worst = worst.max(nearest);
            }
        }
        assert!(worst <= 0.15, "largest hole {worst} m");
    }

    #[test]
    fn a_mesh_across_the_region_edge_gives_only_the_triangles_that_reach_in() {
        // A 400 m sheet of 10 m cells; the region is the 100 m square in its middle.
        let (vertices, indices) = grid(40, 400.0);
        let region = ([150.0, -50.0, 150.0], [250.0, 50.0, 250.0]);
        let bounds = ([0.0, 0.0, 0.0], [400.0, 0.0, 400.0]);
        let (mut vbuf, mut ibuf) = (Vec::new(), Vec::new());
        let count = append_node_triangles_in(
            &vertices,
            &indices,
            &IDENTITY,
            0.08,
            Some(region),
            bounds,
            &mut vbuf,
            &mut ibuf,
        );
        // 10 cells inside and the ring of cells touching the edge: 12 x 12.
        assert_eq!(count, 12 * 12 * 2);
        assert_eq!(vbuf.len(), ibuf.len() * STRIDE);
        for v in vbuf.chunks_exact(STRIDE) {
            assert!((140.0..=260.0).contains(&v[0]) && (140.0..=260.0).contains(&v[2]));
        }
        // Wholly inside: emitted as it is. Wholly outside: nothing.
        let whole = ([-10.0, -50.0, -10.0], [410.0, 50.0, 410.0]);
        let (mut vbuf, mut ibuf) = (Vec::new(), Vec::new());
        let count = append_node_triangles_in(
            &vertices,
            &indices,
            &IDENTITY,
            0.08,
            Some(whole),
            bounds,
            &mut vbuf,
            &mut ibuf,
        );
        assert_eq!((count, ibuf.len()), (3200, indices.len()));
        let away = ([900.0, -50.0, 900.0], [950.0, 50.0, 950.0]);
        let count = append_node_triangles_in(
            &vertices,
            &indices,
            &IDENTITY,
            0.08,
            Some(away),
            bounds,
            &mut vbuf,
            &mut ibuf,
        );
        assert_eq!((count, ibuf.len()), (0, indices.len()));
        // No thinning asked for: the clip still applies.
        let (mut vbuf, mut ibuf) = (Vec::new(), Vec::new());
        let count = append_node_triangles_in(
            &vertices,
            &indices,
            &IDENTITY,
            0.0,
            Some(region),
            bounds,
            &mut vbuf,
            &mut ibuf,
        );
        assert_eq!(count, 12 * 12 * 2);
    }

    #[test]
    fn scale_is_taken_from_the_transform() {
        let (vertices, indices) = grid(100, 1.0);
        let mut big = IDENTITY;
        big[0][0] = 100.0;
        big[1][1] = 100.0;
        big[2][2] = 100.0; // triangles are now 0.5 m² each
        let (mut vbuf, mut ibuf) = (Vec::new(), Vec::new());
        let count = append_node_triangles(&vertices, &indices, &big, 0.1, &mut vbuf, &mut ibuf);
        assert_eq!(count, 20_000);
    }
}
