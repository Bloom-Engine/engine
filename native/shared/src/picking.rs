//! Scene graph picking — raycast from screen coordinates against scene nodes.
//!
//! Converts 2D screen position to a 3D world ray via inverse VP matrix,
//! then tests against all visible scene node triangles using Moller-Trumbore.

use crate::renderer::Vertex3D;
use crate::scene::SceneGraph;

/// Result of a scene pick operation.
#[derive(Clone, Debug)]
pub struct PickResult {
    pub hit: bool,
    pub handle: f64,       // scene node handle that was hit
    pub distance: f32,
    pub point: [f32; 3],   // world-space hit point
    pub normal: [f32; 3],  // face normal at hit point
}

impl PickResult {
    pub fn miss() -> Self {
        Self {
            hit: false,
            handle: 0.0,
            distance: 0.0,
            point: [0.0; 3],
            normal: [0.0; 3],
        }
    }
}

/// Unproject screen coordinates to a world-space ray.
/// screen_x, screen_y: pixel coordinates (0,0 = top-left)
/// width, height: viewport dimensions
/// inv_vp: inverse view-projection matrix
/// camera_pos: camera world position
pub fn screen_to_ray(
    screen_x: f32, screen_y: f32,
    width: f32, height: f32,
    inv_vp: &[[f32; 4]; 4],
    camera_pos: &[f32; 3],
) -> ([f32; 3], [f32; 3]) {
    // Convert screen coords to NDC (-1 to 1)
    let ndc_x = (screen_x / width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_y / height) * 2.0; // flip Y

    // Unproject near point (z = -1 in NDC)
    let near_ndc = [ndc_x, ndc_y, -1.0, 1.0];
    let near_world = mat4_mul_vec4(inv_vp, &near_ndc);

    // Unproject far point (z = 1 in NDC)
    let far_ndc = [ndc_x, ndc_y, 1.0, 1.0];
    let far_world = mat4_mul_vec4(inv_vp, &far_ndc);

    // Perspective divide
    let near = [
        near_world[0] / near_world[3],
        near_world[1] / near_world[3],
        near_world[2] / near_world[3],
    ];
    let far = [
        far_world[0] / far_world[3],
        far_world[1] / far_world[3],
        far_world[2] / far_world[3],
    ];

    // Ray direction
    let dx = far[0] - near[0];
    let dy = far[1] - near[1];
    let dz = far[2] - near[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let dir = if len > 1e-8 {
        [dx / len, dy / len, dz / len]
    } else {
        [0.0, 0.0, -1.0]
    };

    (near, dir)
}

/// Raycast against all visible scene nodes. Returns the closest hit.
pub fn raycast_scene(
    scene: &SceneGraph,
    origin: &[f32; 3],
    direction: &[f32; 3],
) -> PickResult {
    let mut best = PickResult::miss();
    let mut best_dist = f32::MAX;

    for (handle, node) in scene.nodes.iter() {
        if !node.visible || node.indices.is_empty() {
            continue;
        }

        // Transform ray into node's local space via inverse transform
        let inv_transform = mat4_inverse_local(&node.transform);
        let local_origin = mat4_transform_point(&inv_transform, origin);
        let local_dir = mat4_transform_dir(&inv_transform, direction);

        // Test against all triangles
        for tri in node.indices.chunks(3) {
            if tri.len() < 3 { continue; }
            let v0 = &node.vertices[tri[0] as usize];
            let v1 = &node.vertices[tri[1] as usize];
            let v2 = &node.vertices[tri[2] as usize];

            if let Some((t, u, v)) = ray_triangle_intersection(
                &local_origin, &local_dir,
                &v0.position, &v1.position, &v2.position,
            ) {
                if t > 0.0 && t < best_dist {
                    best_dist = t;

                    // Compute hit point in world space
                    let hit_local = [
                        local_origin[0] + local_dir[0] * t,
                        local_origin[1] + local_dir[1] * t,
                        local_origin[2] + local_dir[2] * t,
                    ];
                    let hit_world = mat4_transform_point(&node.transform, &hit_local);

                    // Interpolate normal
                    let w = 1.0 - u - v;
                    let normal = [
                        v0.normal[0] * w + v1.normal[0] * u + v2.normal[0] * v,
                        v0.normal[1] * w + v1.normal[1] * u + v2.normal[1] * v,
                        v0.normal[2] * w + v1.normal[2] * u + v2.normal[2] * v,
                    ];
                    let nl = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
                    let normal = if nl > 1e-6 {
                        [normal[0]/nl, normal[1]/nl, normal[2]/nl]
                    } else {
                        [0.0, 1.0, 0.0]
                    };

                    best = PickResult {
                        hit: true,
                        handle,
                        distance: t,
                        point: hit_world,
                        normal,
                    };
                }
            }
        }
    }

    best
}

// ============================================================
// Moller-Trumbore ray-triangle intersection
// ============================================================

fn ray_triangle_intersection(
    origin: &[f32; 3], dir: &[f32; 3],
    v0: &[f32; 3], v1: &[f32; 3], v2: &[f32; 3],
) -> Option<(f32, f32, f32)> {
    const EPSILON: f32 = 1e-7;

    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];

    let h = cross(dir, &e2);
    let a = dot(&e1, &h);
    if a.abs() < EPSILON { return None; }

    let f = 1.0 / a;
    let s = [origin[0]-v0[0], origin[1]-v0[1], origin[2]-v0[2]];
    let u = f * dot(&s, &h);
    if u < 0.0 || u > 1.0 { return None; }

    let q = cross(&s, &e1);
    let v = f * dot(dir, &q);
    if v < 0.0 || u + v > 1.0 { return None; }

    let t = f * dot(&e2, &q);
    if t > EPSILON {
        Some((t, u, v))
    } else {
        None
    }
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
    [
        m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2] + m[3][0]*v[3],
        m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2] + m[3][1]*v[3],
        m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2] + m[3][2]*v[3],
        m[0][3]*v[0] + m[1][3]*v[1] + m[2][3]*v[2] + m[3][3]*v[3],
    ]
}

fn mat4_transform_point(m: &[[f32; 4]; 4], p: &[f32; 3]) -> [f32; 3] {
    let v = mat4_mul_vec4(m, &[p[0], p[1], p[2], 1.0]);
    if v[3].abs() > 1e-8 {
        [v[0]/v[3], v[1]/v[3], v[2]/v[3]]
    } else {
        [v[0], v[1], v[2]]
    }
}

fn mat4_transform_dir(m: &[[f32; 4]; 4], d: &[f32; 3]) -> [f32; 3] {
    // Transform direction (no translation, no perspective divide)
    [
        m[0][0]*d[0] + m[1][0]*d[1] + m[2][0]*d[2],
        m[0][1]*d[0] + m[1][1]*d[1] + m[2][1]*d[2],
        m[0][2]*d[0] + m[1][2]*d[1] + m[2][2]*d[2],
    ]
}

fn mat4_inverse_local(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    crate::renderer::mat4_invert(*m)
}
