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

/// 12 floats per vertex, the `Vertex3D` stride the bake shader indexes with:
/// position, then zero padding.
const STRIDE: usize = 12;

fn push_position(vbuf: &mut Vec<f32>, t: &[[f32; 4]; 4], p: [f32; 3]) {
    vbuf.extend_from_slice(&[
        t[0][0] * p[0] + t[1][0] * p[1] + t[2][0] * p[2] + t[3][0],
        t[0][1] * p[0] + t[1][1] * p[1] + t[2][1] * p[2] + t[3][1],
        t[0][2] * p[0] + t[1][2] * p[1] + t[2][2] * p[2] + t[3][2],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
}

/// Append one mesh in world space; returns the triangles appended.
pub(crate) fn append_node_triangles(
    vertices: &[Vertex3D],
    indices: &[u32],
    world_transform: &[[f32; 4]; 4],
    min_feature: f32,
    vbuf: &mut Vec<f32>,
    ibuf: &mut Vec<u32>,
) -> u32 {
    let base = (vbuf.len() / STRIDE) as u32;
    let thinned = if min_feature > 0.0 {
        thin_small_triangles(vertices, indices, world_transform, min_feature)
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

/// The corner indices of the triangles that survive thinning, or `None` when
/// nothing was dropped and the mesh should be emitted as it is.
fn thin_small_triangles(
    vertices: &[Vertex3D],
    indices: &[u32],
    world_transform: &[[f32; 4]; 4],
    min_feature: f32,
) -> Option<Vec<u32>> {
    // Areas are measured in object space and scaled by the transform's mean
    // area scale; shear and non-uniform scale only move the threshold a little.
    let t = world_transform;
    let axis = |c: usize| (t[c][0] * t[c][0] + t[c][1] * t[c][1] + t[c][2] * t[c][2]).sqrt();
    let (sx, sy, sz) = (axis(0), axis(1), axis(2));
    let area_scale = (sx * sy + sy * sz + sz * sx) / 3.0;
    let small_area = min_feature * min_feature / area_scale;
    let inv_cell = 1.0 / min_feature;
    if !small_area.is_finite() || !inv_cell.is_finite() {
        return None;
    }
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
        // A small triangle survives only as the first one in its world cell.
        let keep = area >= small_area || {
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
        };
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
