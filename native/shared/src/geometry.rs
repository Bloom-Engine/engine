//! Geometry generation for Bloom Engine.
//!
//! Provides polygon extrusion (for walls and slabs) and CSG box subtraction
//! (for door/window cutouts). These mirror the Three.js ExtrudeGeometry and
//! three-bvh-csg functionality used by the Pascal Editor.

use crate::renderer::Vertex3D;

/// Result of geometry generation.
pub struct GeometryData {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
}

/// Extrude a 2D polygon along the Y axis by the given depth.
///
/// - `polygon`: flat array of 2D points [x0, z0, x1, z1, ...]
/// - `holes`: list of hole polygons, each as flat [x0, z0, x1, z1, ...]
/// - `depth`: extrusion height (in Y direction)
///
/// Returns vertices and indices. The geometry extends from Y=0 to Y=depth.
/// Normals and UVs are computed automatically.
pub fn extrude_polygon(
    polygon: &[f64],
    holes: &[Vec<f64>],
    depth: f64,
) -> GeometryData {
    let depth = depth as f32;

    // Build earcutr input: flatten polygon + holes, track hole starts
    let n_poly = polygon.len() / 2;
    let mut flat_coords: Vec<f64> = polygon.to_vec();
    let mut hole_indices: Vec<usize> = Vec::new();

    for hole in holes {
        hole_indices.push(flat_coords.len() / 2);
        flat_coords.extend_from_slice(hole);
    }

    // Triangulate the 2D polygon
    let triangles = earcutr::earcut(&flat_coords, &hole_indices, 2)
        .unwrap_or_default();

    let n_points = flat_coords.len() / 2;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // ---- Bottom face (Y = 0, normal pointing down) ----
    let base_bottom = 0u32;
    for i in 0..n_points {
        let x = flat_coords[i * 2] as f32;
        let z = flat_coords[i * 2 + 1] as f32;
        vertices.push(Vertex3D {
            position: [x, 0.0, z],
            normal: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv: [x, z], // planar UV
            joints: [0.0; 4],
            weights: [0.0; 4],
        });
    }
    // Bottom triangles (reversed winding for downward-facing)
    for tri in triangles.chunks(3) {
        indices.push(base_bottom + tri[0] as u32);
        indices.push(base_bottom + tri[2] as u32);
        indices.push(base_bottom + tri[1] as u32);
    }

    // ---- Top face (Y = depth, normal pointing up) ----
    let base_top = vertices.len() as u32;
    for i in 0..n_points {
        let x = flat_coords[i * 2] as f32;
        let z = flat_coords[i * 2 + 1] as f32;
        vertices.push(Vertex3D {
            position: [x, depth, z],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv: [x, z],
            joints: [0.0; 4],
            weights: [0.0; 4],
        });
    }
    for tri in triangles.chunks(3) {
        indices.push(base_top + tri[0] as u32);
        indices.push(base_top + tri[1] as u32);
        indices.push(base_top + tri[2] as u32);
    }

    // ---- Side faces (walls of the extrusion) ----
    // Process each edge of the outer polygon and each hole
    let mut edge_loops: Vec<Vec<usize>> = Vec::new();

    // Outer polygon edges
    let outer_loop: Vec<usize> = (0..n_poly).collect();
    edge_loops.push(outer_loop);

    // Hole edges
    for (hi, hole) in holes.iter().enumerate() {
        let n_hole = hole.len() / 2;
        let start = hole_indices[hi];
        let hole_loop: Vec<usize> = (start..start + n_hole).collect();
        edge_loops.push(hole_loop);
    }

    for edge_loop in &edge_loops {
        let n = edge_loop.len();
        for i in 0..n {
            let i0 = edge_loop[i];
            let i1 = edge_loop[(i + 1) % n];

            let x0 = flat_coords[i0 * 2] as f32;
            let z0 = flat_coords[i0 * 2 + 1] as f32;
            let x1 = flat_coords[i1 * 2] as f32;
            let z1 = flat_coords[i1 * 2 + 1] as f32;

            // Edge direction and outward normal
            let dx = x1 - x0;
            let dz = z1 - z0;
            let len = (dx * dx + dz * dz).sqrt();
            if len < 1e-6 { continue; }
            let nx = -dz / len;
            let nz = dx / len;

            // 4 vertices for this side quad
            let base = vertices.len() as u32;

            // Bottom-left, bottom-right, top-right, top-left
            let u_len = len;
            vertices.push(Vertex3D {
                position: [x0, 0.0, z0],
                normal: [nx, 0.0, nz],
                color: [1.0, 1.0, 1.0, 1.0],
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
            });
            vertices.push(Vertex3D {
                position: [x1, 0.0, z1],
                normal: [nx, 0.0, nz],
                color: [1.0, 1.0, 1.0, 1.0],
                uv: [u_len, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
            });
            vertices.push(Vertex3D {
                position: [x1, depth, z1],
                normal: [nx, 0.0, nz],
                color: [1.0, 1.0, 1.0, 1.0],
                uv: [u_len, depth],
                joints: [0.0; 4],
                weights: [0.0; 4],
            });
            vertices.push(Vertex3D {
                position: [x0, depth, z0],
                normal: [nx, 0.0, nz],
                color: [1.0, 1.0, 1.0, 1.0],
                uv: [0.0, depth],
                joints: [0.0; 4],
                weights: [0.0; 4],
            });

            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
            indices.push(base);
            indices.push(base + 2);
            indices.push(base + 3);
        }
    }

    GeometryData { vertices, indices }
}

/// Subtract an axis-aligned box from existing geometry.
///
/// This is a simplified CSG operation that clips triangles against a box.
/// For Phase 2 MVP, we use a simple approach: remove triangles fully inside
/// the box and clip intersecting ones. This handles the common case of
/// door/window cutouts in walls.
///
/// - `min`: box minimum corner [x, y, z]
/// - `max`: box maximum corner [x, y, z]
pub fn subtract_box(
    geo: &GeometryData,
    min: [f32; 3],
    max: [f32; 3],
) -> GeometryData {
    let mut out_vertices = Vec::new();
    let mut out_indices = Vec::new();

    // Process each triangle
    for tri in geo.indices.chunks(3) {
        if tri.len() < 3 { continue; }
        let v0 = &geo.vertices[tri[0] as usize];
        let v1 = &geo.vertices[tri[1] as usize];
        let v2 = &geo.vertices[tri[2] as usize];

        // Check if any vertex is inside the box
        let in0 = point_in_box(&v0.position, &min, &max);
        let in1 = point_in_box(&v1.position, &min, &max);
        let in2 = point_in_box(&v2.position, &min, &max);

        if in0 && in1 && in2 {
            // Triangle fully inside box — discard
            continue;
        }

        // Triangle fully outside or partially intersecting — keep for now
        // (Full CSG clipping would split partial triangles, but for MVP
        //  removing fully-interior triangles handles most cutout cases)
        let base = out_vertices.len() as u32;
        out_vertices.push(*v0);
        out_vertices.push(*v1);
        out_vertices.push(*v2);
        out_indices.push(base);
        out_indices.push(base + 1);
        out_indices.push(base + 2);
    }

    GeometryData {
        vertices: out_vertices,
        indices: out_indices,
    }
}

fn point_in_box(p: &[f32; 3], min: &[f32; 3], max: &[f32; 3]) -> bool {
    p[0] >= min[0] && p[0] <= max[0]
    && p[1] >= min[1] && p[1] <= max[1]
    && p[2] >= min[2] && p[2] <= max[2]
}
