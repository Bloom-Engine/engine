//! Immediate-mode 3D primitive tessellation.
//!
//! These methods append geometry to the renderer-owned transient 3D streams
//! and register each primitive with the common temporal-motion producer.

use super::*;

impl Renderer {
    // ============================================================
    // 3D drawing
    // ============================================================

    fn add_line_3d(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4], thickness: f32) {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 0.0001 {
            return;
        }
        let (dx, dy, dz) = (dx / len, dy / len, dz / len);

        // Find perpendicular using cross product with best reference axis
        let (px, py, pz) = if dy.abs() > 0.9 {
            // Cross with X axis: (0, dz, -dy)
            (0.0, dz, -dy)
        } else {
            // Cross with Y axis: (-dz, 0, dx)
            (-dz, 0.0, dx)
        };
        let plen = (px * px + py * py + pz * pz).sqrt();
        let ht = thickness * 0.5;
        let (px, py, pz) = (px / plen * ht, py / plen * ht, pz / plen * ht);
        let normal = [px / ht, py / ht, pz / ht];

        let base = self.vertices_3d.len() as u32;
        self.vertices_3d.push(Vertex3D {
            position: [start[0] + px, start[1] + py, start[2] + pz],
            normal,
            color,
            uv: [0.0, 0.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        self.vertices_3d.push(Vertex3D {
            position: [start[0] - px, start[1] - py, start[2] - pz],
            normal,
            color,
            uv: [0.0, 0.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        self.vertices_3d.push(Vertex3D {
            position: [end[0] - px, end[1] - py, end[2] - pz],
            normal,
            color,
            uv: [0.0, 0.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        self.vertices_3d.push(Vertex3D {
            position: [end[0] + px, end[1] + py, end[2] + pz],
            normal,
            color,
            uv: [0.0, 0.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        self.indices_3d
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn draw_cube(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        w: f64,
        h: f64,
        d: f64,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let motion_start = self.vertices_3d.len();
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let (hw, hh, hd) = (w as f32 * 0.5, h as f32 * 0.5, d as f32 * 0.5);

        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            (
                [0.0, 0.0, -1.0],
                [
                    [x - hw, y - hh, z - hd],
                    [x + hw, y - hh, z - hd],
                    [x + hw, y + hh, z - hd],
                    [x - hw, y + hh, z - hd],
                ],
            ), // front
            (
                [0.0, 0.0, 1.0],
                [
                    [x + hw, y - hh, z + hd],
                    [x - hw, y - hh, z + hd],
                    [x - hw, y + hh, z + hd],
                    [x + hw, y + hh, z + hd],
                ],
            ), // back
            (
                [-1.0, 0.0, 0.0],
                [
                    [x - hw, y - hh, z + hd],
                    [x - hw, y - hh, z - hd],
                    [x - hw, y + hh, z - hd],
                    [x - hw, y + hh, z + hd],
                ],
            ), // left
            (
                [1.0, 0.0, 0.0],
                [
                    [x + hw, y - hh, z - hd],
                    [x + hw, y - hh, z + hd],
                    [x + hw, y + hh, z + hd],
                    [x + hw, y + hh, z - hd],
                ],
            ), // right
            (
                [0.0, 1.0, 0.0],
                [
                    [x - hw, y + hh, z - hd],
                    [x + hw, y + hh, z - hd],
                    [x + hw, y + hh, z + hd],
                    [x - hw, y + hh, z + hd],
                ],
            ), // top
            (
                [0.0, -1.0, 0.0],
                [
                    [x - hw, y - hh, z + hd],
                    [x + hw, y - hh, z + hd],
                    [x + hw, y - hh, z - hd],
                    [x - hw, y - hh, z - hd],
                ],
            ), // bottom
        ];

        for (normal, verts) in &faces {
            let base = self.vertices_3d.len() as u32;
            for v in verts {
                self.vertices_3d.push(Vertex3D {
                    position: *v,
                    normal: *normal,
                    color,
                    uv: [0.0, 0.0],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    tangent: [0.0; 4],
                });
            }
            // Outward winding (matches the declared normals). The old
            // order wound every face inward: with back-face culling you
            // saw each cube's interior — same bug that made draw_plane
            // invisible from above.
            self.indices_3d.extend_from_slice(&[
                base,
                base + 2,
                base + 1,
                base,
                base + 3,
                base + 2,
            ]);
        }
        self.record_immediate_motion(immediate_motion::PrimitiveKind::Cube, motion_start);
    }

    pub fn draw_cube_wires(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        w: f64,
        h: f64,
        d: f64,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let motion_start = self.vertices_3d.len();
        let color = Self::color_to_f32(r, g, b, a);
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let (hw, hh, hd) = (w as f32 * 0.5, h as f32 * 0.5, d as f32 * 0.5);
        let t = 0.02f32;

        let corners = [
            [x - hw, y - hh, z - hd],
            [x + hw, y - hh, z - hd],
            [x + hw, y + hh, z - hd],
            [x - hw, y + hh, z - hd],
            [x - hw, y - hh, z + hd],
            [x + hw, y - hh, z + hd],
            [x + hw, y + hh, z + hd],
            [x - hw, y + hh, z + hd],
        ];
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // front
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4), // back
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // connecting
        ];
        for (a_idx, b_idx) in &edges {
            self.add_line_3d(corners[*a_idx], corners[*b_idx], color, t);
        }
        self.record_immediate_motion(immediate_motion::PrimitiveKind::CubeWires, motion_start);
    }

    pub fn draw_sphere(
        &mut self,
        cx: f64,
        cy: f64,
        cz: f64,
        radius: f64,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let motion_start = self.vertices_3d.len();
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (cx, cy, cz, radius) = (cx as f32, cy as f32, cz as f32, radius as f32);
        let rings = 8u32;
        let slices = 8u32;

        for i in 0..rings {
            let theta1 = (i as f32) / (rings as f32) * std::f32::consts::PI;
            let theta2 = ((i + 1) as f32) / (rings as f32) * std::f32::consts::PI;
            for j in 0..slices {
                let phi1 = (j as f32) / (slices as f32) * std::f32::consts::TAU;
                let phi2 = ((j + 1) as f32) / (slices as f32) * std::f32::consts::TAU;

                let p = |theta: f32, phi: f32| -> ([f32; 3], [f32; 3]) {
                    let nx = theta.sin() * phi.cos();
                    let ny = theta.cos();
                    let nz = theta.sin() * phi.sin();
                    (
                        [cx + radius * nx, cy + radius * ny, cz + radius * nz],
                        [nx, ny, nz],
                    )
                };

                let (p00, n00) = p(theta1, phi1);
                let (p10, n10) = p(theta2, phi1);
                let (p11, n11) = p(theta2, phi2);
                let (p01, n01) = p(theta1, phi2);

                let base = self.vertices_3d.len() as u32;
                self.vertices_3d.push(Vertex3D {
                    position: p00,
                    normal: n00,
                    color,
                    uv: [0.0, 0.0],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    tangent: [0.0; 4],
                });
                self.vertices_3d.push(Vertex3D {
                    position: p10,
                    normal: n10,
                    color,
                    uv: [0.0, 0.0],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    tangent: [0.0; 4],
                });
                self.vertices_3d.push(Vertex3D {
                    position: p11,
                    normal: n11,
                    color,
                    uv: [0.0, 0.0],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    tangent: [0.0; 4],
                });
                self.vertices_3d.push(Vertex3D {
                    position: p01,
                    normal: n01,
                    color,
                    uv: [0.0, 0.0],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    tangent: [0.0; 4],
                });
                self.indices_3d.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base,
                    base + 2,
                    base + 3,
                ]);
            }
        }
        self.record_immediate_motion(immediate_motion::PrimitiveKind::Sphere, motion_start);
    }

    pub fn draw_sphere_wires(
        &mut self,
        cx: f64,
        cy: f64,
        cz: f64,
        radius: f64,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let motion_start = self.vertices_3d.len();
        let color = Self::color_to_f32(r, g, b, a);
        let (cx, cy, cz, radius) = (cx as f32, cy as f32, cz as f32, radius as f32);
        let segments = 16u32;

        for i in 0..segments {
            let a1 = (i as f32) / (segments as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32) / (segments as f32) * std::f32::consts::TAU;
            // XY ring
            self.add_line_3d(
                [cx + radius * a1.cos(), cy + radius * a1.sin(), cz],
                [cx + radius * a2.cos(), cy + radius * a2.sin(), cz],
                color,
                0.02,
            );
            // XZ ring
            self.add_line_3d(
                [cx + radius * a1.cos(), cy, cz + radius * a1.sin()],
                [cx + radius * a2.cos(), cy, cz + radius * a2.sin()],
                color,
                0.02,
            );
            // YZ ring
            self.add_line_3d(
                [cx, cy + radius * a1.cos(), cz + radius * a1.sin()],
                [cx, cy + radius * a2.cos(), cz + radius * a2.sin()],
                color,
                0.02,
            );
        }
        self.record_immediate_motion(immediate_motion::PrimitiveKind::SphereWires, motion_start);
    }

    pub fn draw_cylinder(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        radius_top: f64,
        radius_bottom: f64,
        height: f64,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let motion_start = self.vertices_3d.len();
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let (rt, rb, h) = (radius_top as f32, radius_bottom as f32, height as f32);
        let slices = 16u32;

        for i in 0..slices {
            let a1 = (i as f32) / (slices as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32) / (slices as f32) * std::f32::consts::TAU;
            let (c1, s1) = (a1.cos(), a1.sin());
            let (c2, s2) = (a2.cos(), a2.sin());

            // Side face
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D {
                position: [x + rb * c1, y, z + rb * s1],
                normal: [c1, 0.0, s1],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.vertices_3d.push(Vertex3D {
                position: [x + rb * c2, y, z + rb * s2],
                normal: [c2, 0.0, s2],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.vertices_3d.push(Vertex3D {
                position: [x + rt * c2, y + h, z + rt * s2],
                normal: [c2, 0.0, s2],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.vertices_3d.push(Vertex3D {
                position: [x + rt * c1, y + h, z + rt * s1],
                normal: [c1, 0.0, s1],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.indices_3d.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base,
                base + 2,
                base + 3,
            ]);

            // Top cap
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D {
                position: [x, y + h, z],
                normal: [0.0, 1.0, 0.0],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.vertices_3d.push(Vertex3D {
                position: [x + rt * c1, y + h, z + rt * s1],
                normal: [0.0, 1.0, 0.0],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.vertices_3d.push(Vertex3D {
                position: [x + rt * c2, y + h, z + rt * s2],
                normal: [0.0, 1.0, 0.0],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.indices_3d
                .extend_from_slice(&[base, base + 1, base + 2]);

            // Bottom cap
            let base = self.vertices_3d.len() as u32;
            self.vertices_3d.push(Vertex3D {
                position: [x, y, z],
                normal: [0.0, -1.0, 0.0],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.vertices_3d.push(Vertex3D {
                position: [x + rb * c2, y, z + rb * s2],
                normal: [0.0, -1.0, 0.0],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.vertices_3d.push(Vertex3D {
                position: [x + rb * c1, y, z + rb * s1],
                normal: [0.0, -1.0, 0.0],
                color,
                uv: [0.0, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
            self.indices_3d
                .extend_from_slice(&[base, base + 1, base + 2]);
        }
        self.record_immediate_motion(immediate_motion::PrimitiveKind::Cylinder, motion_start);
    }

    pub fn draw_plane(
        &mut self,
        cx: f64,
        cy: f64,
        cz: f64,
        w: f64,
        d: f64,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let motion_start = self.vertices_3d.len();
        self.ensure_draw_state_3d(self.current_texture_3d);
        let color = Self::color_to_f32(r, g, b, a);
        let (cx, cy, cz) = (cx as f32, cy as f32, cz as f32);
        let (hw, hd) = (w as f32 * 0.5, d as f32 * 0.5);
        let normal = [0.0f32, 1.0, 0.0];

        let base = self.vertices_3d.len() as u32;
        self.vertices_3d.push(Vertex3D {
            position: [cx - hw, cy, cz - hd],
            normal,
            color,
            uv: [0.0, 0.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        self.vertices_3d.push(Vertex3D {
            position: [cx + hw, cy, cz - hd],
            normal,
            color,
            uv: [1.0, 0.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        self.vertices_3d.push(Vertex3D {
            position: [cx + hw, cy, cz + hd],
            normal,
            color,
            uv: [1.0, 1.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        self.vertices_3d.push(Vertex3D {
            position: [cx - hw, cy, cz + hd],
            normal,
            color,
            uv: [0.0, 1.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
        // Wind so the +Y-normal side is the front face when seen from
        // above — the previous order back-face-culled the plane from
        // every camera above it (only visible from underneath).
        self.indices_3d
            .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        self.record_immediate_motion(immediate_motion::PrimitiveKind::Plane, motion_start);
    }

    pub fn draw_grid(&mut self, slices: i32, spacing: f64) {
        let motion_start = self.vertices_3d.len();
        let color = [0.5f32, 0.5, 0.5, 1.0];
        let spacing = spacing as f32;
        let half = slices as f32 * spacing / 2.0;

        for i in 0..=slices {
            let pos = -half + i as f32 * spacing;
            self.add_line_3d([-half, 0.0, pos], [half, 0.0, pos], color, 0.01);
            self.add_line_3d([pos, 0.0, -half], [pos, 0.0, half], color, 0.01);
        }
        self.record_immediate_motion(immediate_motion::PrimitiveKind::Grid, motion_start);
    }

    pub fn draw_ray(
        &mut self,
        origin_x: f64,
        origin_y: f64,
        origin_z: f64,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let motion_start = self.vertices_3d.len();
        let color = Self::color_to_f32(r, g, b, a);
        let start = [origin_x as f32, origin_y as f32, origin_z as f32];
        let end = [
            (origin_x + dir_x) as f32,
            (origin_y + dir_y) as f32,
            (origin_z + dir_z) as f32,
        ];
        self.add_line_3d(start, end, color, 0.02);
        self.record_immediate_motion(immediate_motion::PrimitiveKind::Ray, motion_start);
    }
}
