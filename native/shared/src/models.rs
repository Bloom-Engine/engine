use crate::handles::HandleRegistry;
use crate::renderer::Vertex3D;

pub struct MeshData {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    pub texture_idx: Option<u32>,
    pub normal_texture_idx: Option<u32>,
    pub metallic_roughness_texture_idx: Option<u32>,
    pub emissive_texture_idx: Option<u32>,
    pub occlusion_texture_idx: Option<u32>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    /// glTF alpha cutoff for MASK mode — fragments with base-colour
    /// alpha below this are discarded. 0.0 means OPAQUE mode (no
    /// discard); glTF spec default for MASK is 0.5. BLEND mode is
    /// currently treated as MASK @ 0.5 since we don't have a sorted
    /// transparent pipeline yet.
    pub alpha_cutoff: f32,
}

pub struct ModelData {
    pub meshes: Vec<MeshData>,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
}

pub struct JointData {
    pub inverse_bind: [[f32; 4]; 4],
    pub children: Vec<usize>,
    pub name: String,
    pub rest_translation: [f32; 3],
    pub rest_rotation: [f32; 4],
    pub rest_scale: [f32; 3],
}

pub struct AnimationChannel {
    pub joint_index: usize,
    pub timestamps: Vec<f32>,
    pub translations: Vec<[f32; 3]>,
    pub rotation_timestamps: Vec<f32>,
    pub rotations: Vec<[f32; 4]>,
    pub scale_timestamps: Vec<f32>,
    pub scales: Vec<[f32; 3]>,
}

pub struct AnimationData {
    pub channels: Vec<AnimationChannel>,
    pub duration: f32,
    pub name: String,
}

pub struct SkeletonData {
    pub joints: Vec<JointData>,
    pub root_joints: Vec<usize>,
}

pub struct ModelAnimation {
    pub skeleton: Option<SkeletonData>,
    pub animations: Vec<AnimationData>,
    pub joint_matrices: Vec<[[f32; 4]; 4]>,
    /// Reference rest-pose rotations (from first animation, sampled at t=0).
    /// Used for retargeting when multiple armatures have different rest orientations.
    pub ref_rest_rotations: Option<Vec<[f32; 4]>>,
}

pub struct ModelManager {
    pub models: HandleRegistry<ModelData>,
    pub animations: HandleRegistry<ModelAnimation>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: HandleRegistry::new(),
            animations: HandleRegistry::new(),
        }
    }

    pub fn load_model(&mut self, file_data: &[u8]) -> f64 {
        match load_gltf(file_data) {
            Some(model) => self.models.alloc(model),
            None => 0.0,
        }
    }

    pub fn load_model_with_textures(&mut self, file_data: &[u8], renderer: &mut crate::renderer::Renderer) -> f64 {
        match load_gltf_with_textures(file_data, renderer, None) {
            Some(model) => self.models.alloc(model),
            None => 0.0,
        }
    }

    /// Like `load_model_with_textures` but also resolves external `.bin`
    /// and image URIs relative to `base_dir` — required for loose glTF
    /// files (as opposed to single-file .glb). Intel Sponza etc.
    pub fn load_model_with_textures_from_path(
        &mut self,
        file_data: &[u8],
        base_dir: &std::path::Path,
        renderer: &mut crate::renderer::Renderer,
    ) -> f64 {
        match load_gltf_with_textures(file_data, renderer, Some(base_dir)) {
            Some(model) => self.models.alloc(model),
            None => 0.0,
        }
    }

    pub fn get(&self, handle: f64) -> Option<&ModelData> {
        self.models.get(handle)
    }

    /// Return the axis-aligned bounding box of a loaded model as
    /// `(min_xyz, max_xyz)`. Used by editors to size move/rotate gizmos,
    /// auto-frame the camera on selection, and snap placed entities onto
    /// terrain. Returns the origin for unknown handles so callers can read
    /// without checking for None.
    pub fn get_bounds(&self, handle: f64) -> ([f32; 3], [f32; 3]) {
        match self.models.get(handle) {
            Some(model) => (model.bbox_min, model.bbox_max),
            None => ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        }
    }

    pub fn get_animation(&self, handle: f64) -> Option<&ModelAnimation> {
        self.animations.get(handle)
    }

    pub fn unload_model(&mut self, handle: f64) {
        self.models.free(handle);
    }

    pub fn gen_mesh_cube(&mut self, w: f32, h: f32, d: f32) -> f64 {
        let hw = w * 0.5;
        let hh = h * 0.5;
        let hd = d * 0.5;
        let white = [1.0, 1.0, 1.0, 1.0];

        #[rustfmt::skip]
        let faces: &[([f32; 3], [f32; 3], [f32; 2])] = &[
            // Front face (+Z)
            ([-hw, -hh,  hd], [0.0, 0.0, 1.0], [0.0, 1.0]),
            ([ hw, -hh,  hd], [0.0, 0.0, 1.0], [1.0, 1.0]),
            ([ hw,  hh,  hd], [0.0, 0.0, 1.0], [1.0, 0.0]),
            ([-hw,  hh,  hd], [0.0, 0.0, 1.0], [0.0, 0.0]),
            // Back face (-Z)
            ([ hw, -hh, -hd], [0.0, 0.0, -1.0], [0.0, 1.0]),
            ([-hw, -hh, -hd], [0.0, 0.0, -1.0], [1.0, 1.0]),
            ([-hw,  hh, -hd], [0.0, 0.0, -1.0], [1.0, 0.0]),
            ([ hw,  hh, -hd], [0.0, 0.0, -1.0], [0.0, 0.0]),
            // Right face (+X)
            ([ hw, -hh,  hd], [1.0, 0.0, 0.0], [0.0, 1.0]),
            ([ hw, -hh, -hd], [1.0, 0.0, 0.0], [1.0, 1.0]),
            ([ hw,  hh, -hd], [1.0, 0.0, 0.0], [1.0, 0.0]),
            ([ hw,  hh,  hd], [1.0, 0.0, 0.0], [0.0, 0.0]),
            // Left face (-X)
            ([-hw, -hh, -hd], [-1.0, 0.0, 0.0], [0.0, 1.0]),
            ([-hw, -hh,  hd], [-1.0, 0.0, 0.0], [1.0, 1.0]),
            ([-hw,  hh,  hd], [-1.0, 0.0, 0.0], [1.0, 0.0]),
            ([-hw,  hh, -hd], [-1.0, 0.0, 0.0], [0.0, 0.0]),
            // Top face (+Y)
            ([-hw,  hh,  hd], [0.0, 1.0, 0.0], [0.0, 1.0]),
            ([ hw,  hh,  hd], [0.0, 1.0, 0.0], [1.0, 1.0]),
            ([ hw,  hh, -hd], [0.0, 1.0, 0.0], [1.0, 0.0]),
            ([-hw,  hh, -hd], [0.0, 1.0, 0.0], [0.0, 0.0]),
            // Bottom face (-Y)
            ([-hw, -hh, -hd], [0.0, -1.0, 0.0], [0.0, 1.0]),
            ([ hw, -hh, -hd], [0.0, -1.0, 0.0], [1.0, 1.0]),
            ([ hw, -hh,  hd], [0.0, -1.0, 0.0], [1.0, 0.0]),
            ([-hw, -hh,  hd], [0.0, -1.0, 0.0], [0.0, 0.0]),
        ];

        let vertices: Vec<Vertex3D> = faces.iter().map(|(pos, norm, uv)| Vertex3D {
            position: *pos,
            normal: *norm,
            color: white,
            uv: *uv,
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        }).collect();

        let mut indices = Vec::with_capacity(36);
        for face in 0..6u32 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let model = ModelData {
            meshes: vec![MeshData { vertices, indices, texture_idx: None, normal_texture_idx: None, metallic_roughness_texture_idx: None, emissive_texture_idx: None, occlusion_texture_idx: None, metallic_factor: 0.0, roughness_factor: 1.0, emissive_factor: [0.0; 3], alpha_cutoff: 0.0 }],
            bbox_min: [-hw, -hh, -hd],
            bbox_max: [hw, hh, hd],
        };
        self.models.alloc(model)
    }

    pub fn gen_mesh_heightmap(&mut self, image_data: &[u8], img_w: u32, img_h: u32, size_x: f32, size_y: f32, size_z: f32) -> f64 {
        let cols = img_w as usize;
        let rows = img_h as usize;
        if cols < 2 || rows < 2 { return 0.0; }

        let mut vertices = Vec::with_capacity(cols * rows);
        let white = [1.0, 1.0, 1.0, 1.0];

        for z in 0..rows {
            for x in 0..cols {
                let pixel_idx = (z * cols + x) * 4;
                let luminance = if pixel_idx + 2 < image_data.len() {
                    (image_data[pixel_idx] as f32 * 0.299
                        + image_data[pixel_idx + 1] as f32 * 0.587
                        + image_data[pixel_idx + 2] as f32 * 0.114) / 255.0
                } else {
                    0.0
                };

                let px = (x as f32 / (cols - 1) as f32 - 0.5) * size_x;
                let py = luminance * size_y;
                let pz = (z as f32 / (rows - 1) as f32 - 0.5) * size_z;
                let u = x as f32 / (cols - 1) as f32;
                let v = z as f32 / (rows - 1) as f32;

                vertices.push(Vertex3D {
                    position: [px, py, pz],
                    normal: [0.0, 1.0, 0.0],
                    color: white,
                    uv: [u, v],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    tangent: [0.0; 4],
                });
            }
        }

        // Compute normals from neighboring heights
        for z in 0..rows {
            for x in 0..cols {
                let idx = z * cols + x;
                let left = if x > 0 { vertices[z * cols + x - 1].position[1] } else { vertices[idx].position[1] };
                let right = if x < cols - 1 { vertices[z * cols + x + 1].position[1] } else { vertices[idx].position[1] };
                let up = if z > 0 { vertices[(z - 1) * cols + x].position[1] } else { vertices[idx].position[1] };
                let down = if z < rows - 1 { vertices[(z + 1) * cols + x].position[1] } else { vertices[idx].position[1] };
                let sx = size_x / (cols - 1) as f32;
                let sz = size_z / (rows - 1) as f32;
                let nx = (left - right) / (2.0 * sx);
                let nz = (up - down) / (2.0 * sz);
                let len = (nx * nx + 1.0 + nz * nz).sqrt();
                vertices[idx].normal = [nx / len, 1.0 / len, nz / len];
            }
        }

        let mut indices = Vec::with_capacity((cols - 1) * (rows - 1) * 6);
        for z in 0..rows - 1 {
            for x in 0..cols - 1 {
                let tl = (z * cols + x) as u32;
                let tr = tl + 1;
                let bl = ((z + 1) * cols + x) as u32;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        }

        let model = ModelData {
            meshes: vec![MeshData { vertices, indices, texture_idx: None, normal_texture_idx: None, metallic_roughness_texture_idx: None, emissive_texture_idx: None, occlusion_texture_idx: None, metallic_factor: 0.0, roughness_factor: 1.0, emissive_factor: [0.0; 3], alpha_cutoff: 0.0 }],
            bbox_min: [-size_x * 0.5, 0.0, -size_z * 0.5],
            bbox_max: [size_x * 0.5, size_y, size_z * 0.5],
        };
        self.models.alloc(model)
    }

    /// Create a mesh from raw float data passed from TS.
    /// vertex_data layout: [x,y,z, nx,ny,nz, r,g,b,a, u,v] per vertex (12 floats each)
    pub fn create_mesh(&mut self, vertex_data: &[f32], index_data: &[u32]) -> f64 {
        let floats_per_vert = 12;
        let vert_count = vertex_data.len() / floats_per_vert;
        if vert_count == 0 { return 0.0; }

        let mut vertices = Vec::with_capacity(vert_count);
        let mut bbox_min = [f32::MAX; 3];
        let mut bbox_max = [f32::MIN; 3];

        for i in 0..vert_count {
            let o = i * floats_per_vert;
            let pos = [vertex_data[o], vertex_data[o+1], vertex_data[o+2]];
            for k in 0..3 {
                if pos[k] < bbox_min[k] { bbox_min[k] = pos[k]; }
                if pos[k] > bbox_max[k] { bbox_max[k] = pos[k]; }
            }
            vertices.push(Vertex3D {
                position: pos,
                normal: [vertex_data[o+3], vertex_data[o+4], vertex_data[o+5]],
                color: [vertex_data[o+6], vertex_data[o+7], vertex_data[o+8], vertex_data[o+9]],
                uv: [vertex_data[o+10], vertex_data[o+11]],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
        }

        let indices = index_data.to_vec();
        let model = ModelData {
            meshes: vec![MeshData { vertices, indices, texture_idx: None, normal_texture_idx: None, metallic_roughness_texture_idx: None, emissive_texture_idx: None, occlusion_texture_idx: None, metallic_factor: 0.0, roughness_factor: 1.0, emissive_factor: [0.0; 3], alpha_cutoff: 0.0 }],
            bbox_min,
            bbox_max,
        };
        self.models.alloc(model)
    }

    /// Q9: Generate a ribbon mesh along a Catmull-Rom spline. Used by the
    /// editor's river tool. `points` is flat [x0,y0,z0, x1,y1,z1, ...],
    /// `widths` has one width per control point.
    pub fn gen_mesh_spline_ribbon(&mut self, points: &[f32], widths: &[f32]) -> f64 {
        let n = points.len() / 3;
        if n < 2 || widths.len() < n { return 0.0; }

        // Evaluate Catmull-Rom at fine intervals.
        let segments = (n - 1) * 8; // 8 subdivisions per segment.
        let mut center_pts: Vec<[f32; 3]> = Vec::with_capacity(segments + 1);
        let mut center_widths: Vec<f32> = Vec::with_capacity(segments + 1);

        for i in 0..n - 1 {
            for sub in 0..8 {
                let t = sub as f32 / 8.0;
                let p = catmull_rom_point(points, n, i, t);
                let w = widths[i] * (1.0 - t) + widths[i + 1] * t;
                center_pts.push(p);
                center_widths.push(w);
            }
        }
        // Add the last point.
        let last = n - 1;
        center_pts.push([points[last * 3], points[last * 3 + 1], points[last * 3 + 2]]);
        center_widths.push(widths[last]);

        // Build ribbon vertices (two per center point: left and right).
        let ribbon_len = center_pts.len();
        let mut vertices = Vec::with_capacity(ribbon_len * 2);
        let mut bbox_min = [f32::MAX; 3];
        let mut bbox_max = [f32::MIN; 3];
        let white = [0.3, 0.5, 0.8, 0.7]; // Water-blue tint.

        for i in 0..ribbon_len {
            // Tangent direction.
            let tangent = if i < ribbon_len - 1 {
                let dx = center_pts[i + 1][0] - center_pts[i][0];
                let dz = center_pts[i + 1][2] - center_pts[i][2];
                let len = (dx * dx + dz * dz).sqrt().max(1e-6);
                [dx / len, dz / len]
            } else if i > 0 {
                let dx = center_pts[i][0] - center_pts[i - 1][0];
                let dz = center_pts[i][2] - center_pts[i - 1][2];
                let len = (dx * dx + dz * dz).sqrt().max(1e-6);
                [dx / len, dz / len]
            } else {
                [0.0, 1.0]
            };

            // Perpendicular in XZ plane (rotate tangent 90 degrees).
            let perp = [-tangent[1], tangent[0]];
            let hw = center_widths[i] * 0.5;
            let cp = center_pts[i];
            let u = i as f32 / (ribbon_len - 1).max(1) as f32;

            // Left vertex.
            let lx = cp[0] + perp[0] * hw;
            let ly = cp[1];
            let lz = cp[2] + perp[1] * hw;
            update_bounds(&mut bbox_min, &mut bbox_max, lx, ly, lz);
            vertices.push(Vertex3D {
                position: [lx, ly, lz],
                normal: [0.0, 1.0, 0.0],
                color: white,
                uv: [u, 0.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });

            // Right vertex.
            let rx = cp[0] - perp[0] * hw;
            let ry = cp[1];
            let rz = cp[2] - perp[1] * hw;
            update_bounds(&mut bbox_min, &mut bbox_max, rx, ry, rz);
            vertices.push(Vertex3D {
                position: [rx, ry, rz],
                normal: [0.0, 1.0, 0.0],
                color: white,
                uv: [u, 1.0],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            });
        }

        // Triangle strip indices.
        let mut indices = Vec::with_capacity((ribbon_len - 1) * 6);
        for i in 0..(ribbon_len - 1) as u32 {
            let bl = i * 2;
            let br = bl + 1;
            let tl = bl + 2;
            let tr = bl + 3;
            indices.extend_from_slice(&[bl, tl, br, br, tl, tr]);
        }

        if vertices.is_empty() {
            bbox_min = [0.0; 3];
            bbox_max = [0.0; 3];
        }

        let model = ModelData {
            meshes: vec![MeshData { vertices, indices, texture_idx: None, normal_texture_idx: None, metallic_roughness_texture_idx: None, emissive_texture_idx: None, occlusion_texture_idx: None, metallic_factor: 0.0, roughness_factor: 1.0, emissive_factor: [0.0; 3], alpha_cutoff: 0.0 }],
            bbox_min,
            bbox_max,
        };
        self.models.alloc(model)
    }

    pub fn load_model_animation(&mut self, file_data: &[u8]) -> f64 {
        match load_gltf_animation(file_data) {
            Some(anim) => self.animations.alloc(anim),
            None => 0.0,
        }
    }

    pub fn update_model_animation(&mut self, handle: f64, anim_index: usize, time: f32) {
        if let Some(model_anim) = self.animations.get_mut(handle) {
            let skeleton = match &model_anim.skeleton {
                Some(s) => s,
                None => return,
            };
            if anim_index >= model_anim.animations.len() { return; }

            let joint_count = skeleton.joints.len();
            if model_anim.joint_matrices.len() != joint_count {
                model_anim.joint_matrices = vec![mat4_identity(); joint_count];
            }

            // Initialize from rest-pose transforms (fallback for non-animated joints)
            let mut local_translations: Vec<[f32; 3]> = skeleton.joints.iter()
                .map(|j| j.rest_translation).collect();
            let mut local_rotations: Vec<[f32; 4]> = skeleton.joints.iter()
                .map(|j| j.rest_rotation).collect();
            let mut local_scales: Vec<[f32; 3]> = skeleton.joints.iter()
                .map(|j| j.rest_scale).collect();

            let anim = &model_anim.animations[anim_index];
            let t = if anim.duration > 0.0 { time % anim.duration } else { 0.0 };

            #[cfg(debug_assertions)]
            let mut channels_applied = 0usize;
            for channel in &anim.channels {
                let ji = channel.joint_index;
                if ji >= joint_count { continue; }
                #[cfg(debug_assertions)]
                { channels_applied += 1; }

                if !channel.translations.is_empty() && !channel.timestamps.is_empty() {
                    local_translations[ji] = sample_vec3(&channel.timestamps, &channel.translations, t);
                }
                if !channel.rotations.is_empty() {
                    let rot_ts = if !channel.rotation_timestamps.is_empty() { &channel.rotation_timestamps } else { &channel.timestamps };
                    if !rot_ts.is_empty() {
                        local_rotations[ji] = sample_quat(rot_ts, &channel.rotations, t);
                    }
                }
                if !channel.scales.is_empty() {
                    let scale_ts = if !channel.scale_timestamps.is_empty() { &channel.scale_timestamps } else { &channel.timestamps };
                    if !scale_ts.is_empty() {
                        local_scales[ji] = sample_vec3(scale_ts, &channel.scales, t);
                    }
                }
            }

            // Lock root translation to rest pose (strip all root motion)
            local_translations[0] = skeleton.joints[0].rest_translation;

            // Build world transforms by walking the hierarchy from roots
            let mut world_transforms = vec![mat4_identity(); joint_count];

            let root_joints = skeleton.root_joints.clone();
            for &root in &root_joints {
                compute_joint_transforms(
                    skeleton, root, &mat4_identity(),
                    &local_translations, &local_rotations, &local_scales,
                    &mut world_transforms,
                );
            }

            // Multiply by inverse bind matrices to get final joint matrices
            for i in 0..joint_count {
                model_anim.joint_matrices[i] = mat4_mul(&world_transforms[i], &skeleton.joints[i].inverse_bind);
            }

            #[cfg(debug_assertions)]
            {
                static mut DEBUG_PRINTED: bool = false;
                unsafe {
                    if !DEBUG_PRINTED {
                        DEBUG_PRINTED = true;
                        eprintln!("[anim] channels_applied={}, t={:.3}, anim_index={}", channels_applied, t, anim_index);
                        eprintln!("[anim] Joint0 local: t=[{:.2},{:.2},{:.2}] r=[{:.4},{:.4},{:.4},{:.4}]",
                            local_translations[0][0], local_translations[0][1], local_translations[0][2],
                            local_rotations[0][0], local_rotations[0][1], local_rotations[0][2], local_rotations[0][3]);
                        let m = &model_anim.joint_matrices[0];
                        eprintln!("[anim] Joint0 final diag=[{:.4},{:.4},{:.4}] trans=[{:.4},{:.4},{:.4}]",
                            m[0][0], m[1][1], m[2][2], m[3][0], m[3][1], m[3][2]);
                    }
                }
            }
        }
    }
}

// ============================================================
// Matrix / quaternion helpers for skeletal animation
// ============================================================

fn mat4_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row]*b[col][0] + a[1][row]*b[col][1] + a[2][row]*b[col][2] + a[3][row]*b[col][3];
        }
    }
    out
}

/// Walk the scene graph and collect EVERY world-space transform that
/// references each mesh. Unlike `walk_scene_for_mesh_transforms` which
/// records only the first occurrence, this version captures every
/// instance — so glTF scenes with heavy mesh reuse (Bistro: 5910 nodes
/// referencing 551 unique meshes) render every chair / bollard / chain
/// / bush instead of collapsing to a single copy each.
fn walk_scene_collect_instances(
    node: &gltf::Node,
    parent: &[[f32; 4]; 4],
    out: &mut [Vec<[[f32; 4]; 4]>],
) {
    let local = node.transform().matrix();
    let world = mat4_mul(parent, &local);
    if let Some(mesh) = node.mesh() {
        let idx = mesh.index();
        if idx < out.len() {
            out[idx].push(world);
        }
    }
    for child in node.children() {
        walk_scene_collect_instances(&child, &world, out);
    }
}

/// Transform a 3D point by a 4x4 matrix (column-major). Treats the
/// point as having w=1 and drops w from the result.
fn mat4_transform_point(m: &[[f32; 4]; 4], p: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0]*p[0] + m[1][0]*p[1] + m[2][0]*p[2] + m[3][0],
        m[0][1]*p[0] + m[1][1]*p[1] + m[2][1]*p[2] + m[3][1],
        m[0][2]*p[0] + m[1][2]*p[1] + m[2][2]*p[2] + m[3][2],
    ]
}

/// Transform a direction vector by a 3x3 matrix (extracted from a 4x4
/// column-major stored as the top-left 3x3). Used for normals under
/// the inverse-transpose matrix.
fn mat3_transform_vec(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2],
        m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2],
        m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2],
    ]
}

/// Inverse-transpose of the 3x3 rotation+scale part of a 4x4 matrix.
/// Correct way to transform normals when the matrix has non-uniform
/// scale; falls back to identity if the 3x3 block isn't invertible.
fn mat4_inverse_transpose_3x3(m: &[[f32; 4]; 4]) -> [[f32; 3]; 3] {
    let a = m[0][0]; let b = m[1][0]; let c = m[2][0];
    let d = m[0][1]; let e = m[1][1]; let f = m[2][1];
    let g = m[0][2]; let h = m[1][2]; let i = m[2][2];

    let inv00 =  e*i - f*h;
    let inv01 =  f*g - d*i;
    let inv02 =  d*h - e*g;
    let inv10 =  c*h - b*i;
    let inv11 =  a*i - c*g;
    let inv12 =  b*g - a*h;
    let inv20 =  b*f - c*e;
    let inv21 =  c*d - a*f;
    let inv22 =  a*e - b*d;

    let det = a*inv00 + b*inv01 + c*inv02;
    if det.abs() < 1e-10 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv_det = 1.0 / det;
    // Store in column-major like the rest of the file (columns first).
    // The result is the inverse-transpose, so rows/cols are swapped
    // from the plain inverse.
    [
        [inv00 * inv_det, inv01 * inv_det, inv02 * inv_det],
        [inv10 * inv_det, inv11 * inv_det, inv12 * inv_det],
        [inv20 * inv_det, inv21 * inv_det, inv22 * inv_det],
    ]
}

fn mat4_from_trs(t: &[f32; 3], r: &[f32; 4], s: &[f32; 3]) -> [[f32; 4]; 4] {
    let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;

    // Column-major: m[col][row]
    [
        [(1.0 - (yy + zz)) * s[0], (xy + wz) * s[0],         (xz - wy) * s[0],         0.0],  // column 0
        [(xy - wz) * s[1],         (1.0 - (xx + zz)) * s[1], (yz + wx) * s[1],         0.0],  // column 1
        [(xz + wy) * s[2],         (yz - wx) * s[2],         (1.0 - (xx + yy)) * s[2], 0.0],  // column 2
        [t[0],                      t[1],                      t[2],                      1.0],  // column 3 (translation)
    ]
}

fn quat_slerp(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    let mut b2 = *b;
    if dot < 0.0 {
        dot = -dot;
        b2 = [-b[0], -b[1], -b[2], -b[3]];
    }
    if dot > 0.9995 {
        let mut out = [
            a[0] + t * (b2[0] - a[0]),
            a[1] + t * (b2[1] - a[1]),
            a[2] + t * (b2[2] - a[2]),
            a[3] + t * (b2[3] - a[3]),
        ];
        let len = (out[0]*out[0] + out[1]*out[1] + out[2]*out[2] + out[3]*out[3]).sqrt();
        if len > 0.0 { for v in &mut out { *v /= len; } }
        return out;
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;
    [
        wa * a[0] + wb * b2[0],
        wa * a[1] + wb * b2[1],
        wa * a[2] + wb * b2[2],
        wa * a[3] + wb * b2[3],
    ]
}

fn lerp_vec3(a: &[f32; 3], b: &[f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

fn find_keyframe_pair(timestamps: &[f32], time: f32) -> (usize, usize, f32) {
    if timestamps.len() <= 1 {
        return (0, 0, 0.0);
    }
    if time <= timestamps[0] {
        return (0, 0, 0.0);
    }
    if time >= timestamps[timestamps.len() - 1] {
        let last = timestamps.len() - 1;
        return (last, last, 0.0);
    }
    for i in 0..timestamps.len() - 1 {
        if time >= timestamps[i] && time < timestamps[i + 1] {
            let dt = timestamps[i + 1] - timestamps[i];
            let t = if dt > 0.0 { (time - timestamps[i]) / dt } else { 0.0 };
            return (i, i + 1, t);
        }
    }
    let last = timestamps.len() - 1;
    (last, last, 0.0)
}

fn sample_vec3(timestamps: &[f32], values: &[[f32; 3]], time: f32) -> [f32; 3] {
    if values.is_empty() { return [0.0; 3]; }
    if values.len() == 1 { return values[0]; }
    let (i0, i1, t) = find_keyframe_pair(timestamps, time);
    if i0 >= values.len() { return values[values.len() - 1]; }
    if i1 >= values.len() { return values[values.len() - 1]; }
    lerp_vec3(&values[i0], &values[i1], t)
}

fn sample_quat(timestamps: &[f32], values: &[[f32; 4]], time: f32) -> [f32; 4] {
    if values.is_empty() { return [0.0, 0.0, 0.0, 1.0]; }
    if values.len() == 1 { return values[0]; }
    let (i0, i1, t) = find_keyframe_pair(timestamps, time);
    if i0 >= values.len() { return values[values.len() - 1]; }
    if i1 >= values.len() { return values[values.len() - 1]; }
    quat_slerp(&values[i0], &values[i1], t)
}

fn compute_joint_transforms(
    skeleton: &SkeletonData,
    joint_idx: usize,
    parent_transform: &[[f32; 4]; 4],
    translations: &[[f32; 3]],
    rotations: &[[f32; 4]],
    scales: &[[f32; 3]],
    world_transforms: &mut [[[f32; 4]; 4]],
) {
    if joint_idx >= skeleton.joints.len() { return; }
    let local = mat4_from_trs(&translations[joint_idx], &rotations[joint_idx], &scales[joint_idx]);
    let world = mat4_mul(parent_transform, &local);
    world_transforms[joint_idx] = world;
    let children = skeleton.joints[joint_idx].children.clone();
    for &child in &children {
        compute_joint_transforms(skeleton, child, &world, translations, rotations, scales, world_transforms);
    }
}

// ============================================================
// glTF animation loader
// ============================================================

fn read_accessor_f32(_gltf: &gltf::Gltf, buffer_data: &[Vec<u8>], accessor: &gltf::Accessor) -> Vec<f32> {
    let view = match accessor.view() {
        Some(v) => v,
        None => return Vec::new(),
    };
    let buf_idx = view.buffer().index();
    if buf_idx >= buffer_data.len() { return Vec::new(); }
    let buf = &buffer_data[buf_idx];
    let offset = view.offset() + accessor.offset();
    let count = accessor.count();
    let stride = view.stride().unwrap_or(accessor.size());
    let component_count = match accessor.dimensions() {
        gltf::accessor::Dimensions::Scalar => 1,
        gltf::accessor::Dimensions::Vec2 => 2,
        gltf::accessor::Dimensions::Vec3 => 3,
        gltf::accessor::Dimensions::Vec4 => 4,
        gltf::accessor::Dimensions::Mat4 => 16,
        _ => 1,
    };

    let mut result = Vec::with_capacity(count * component_count);
    for i in 0..count {
        let base = offset + i * stride;
        for c in 0..component_count {
            let byte_offset = base + c * 4;
            if byte_offset + 4 <= buf.len() {
                let val = f32::from_le_bytes([buf[byte_offset], buf[byte_offset+1], buf[byte_offset+2], buf[byte_offset+3]]);
                result.push(val);
            } else {
                result.push(0.0);
            }
        }
    }
    result
}

fn load_gltf_animation(data: &[u8]) -> Option<ModelAnimation> {
    let gltf = gltf::Gltf::from_slice(data).ok()?;

    // Get buffer data
    let mut buffer_data: Vec<Vec<u8>> = Vec::new();
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                if let Some(blob) = gltf.blob.as_ref() {
                    buffer_data.push(blob.clone());
                }
            }
            gltf::buffer::Source::Uri(uri) => {
                if let Some(encoded) = uri.strip_prefix("data:application/octet-stream;base64,") {
                    let mut decoded = Vec::new();
                    let _ = base64_decode(encoded, &mut decoded);
                    buffer_data.push(decoded);
                } else {
                    buffer_data.push(Vec::new());
                }
            }
        }
    }

    // Parse skeleton from the first skin
    let skeleton = if let Some(skin) = gltf.skins().next() {
        let joints_nodes: Vec<_> = skin.joints().collect();
        let joint_count = joints_nodes.len();

        // Build a mapping from node index to joint index
        let mut node_to_joint = std::collections::HashMap::new();
        for (ji, node) in joints_nodes.iter().enumerate() {
            node_to_joint.insert(node.index(), ji);
        }

        // Read inverse bind matrices
        let ibm_data = if let Some(accessor) = skin.inverse_bind_matrices() {
            read_accessor_f32(&gltf, &buffer_data, &accessor)
        } else {
            let mut default_ibm = Vec::with_capacity(joint_count * 16);
            for _ in 0..joint_count {
                default_ibm.extend_from_slice(&[
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ]);
            }
            default_ibm
        };

        let mut joints = Vec::with_capacity(joint_count);
        let mut root_joints = Vec::new();

        for (ji, node) in joints_nodes.iter().enumerate() {
            let mut ibm = [[0.0f32; 4]; 4];
            let base = ji * 16;
            if base + 16 <= ibm_data.len() {
                // glTF stores column-major; read directly (we also use column-major)
                for a in 0..4 {
                    for b in 0..4 {
                        ibm[a][b] = ibm_data[base + a * 4 + b];
                    }
                }
            } else {
                ibm = mat4_identity();
            }

            // Blender FBX export bakes 100x scale into IBMs (converts m→cm for bone space).
            // This is NEEDED because Blender also pre-scales vertex positions to meters.
            // The 100x in IBMs converts meter-space vertices to cm-space bone transforms.
            // DO NOT normalize — the scale is intentional and required.

            let children: Vec<usize> = node.children()
                .filter_map(|child| node_to_joint.get(&child.index()).copied())
                .collect();

            let name = node.name().unwrap_or("").to_string();
            let (t, r, s) = node.transform().decomposed();

            joints.push(JointData {
                inverse_bind: ibm, children, name,
                rest_translation: t,
                rest_rotation: r,
                rest_scale: s,
            });
        }

        // Find root joints (joints that are not children of any other joint)
        let mut is_child = vec![false; joint_count];
        for joint in &joints {
            for &child in &joint.children {
                if child < joint_count { is_child[child] = true; }
            }
        }
        for i in 0..joint_count {
            if !is_child[i] { root_joints.push(i); }
        }

        #[cfg(debug_assertions)]
        {
            eprintln!("[anim] Skeleton: {} joints, {} roots", joints.len(), root_joints.len());
            for (i, j) in joints.iter().enumerate() {
                if i < 5 || i == joints.len() - 1 {
                    eprintln!("[anim]   joint {}: '{}' children={:?}", i, j.name, j.children);
                }
            }
        }

        Some(SkeletonData { joints, root_joints })
    } else {
        #[cfg(debug_assertions)]
        eprintln!("[anim] No skin found in glTF!");
        None
    };

    // Parse animations
    let mut animations = Vec::new();
    for anim in gltf.animations() {
        let mut channels = Vec::new();
        let mut duration: f32 = 0.0;

        // Build node-to-joint mapping for channel resolution
        let node_to_joint: std::collections::HashMap<usize, usize> = if let Some(skin) = gltf.skins().next() {
            skin.joints().enumerate().map(|(ji, node)| (node.index(), ji)).collect()
        } else {
            std::collections::HashMap::new()
        };

        // Group channels by target node: (trans_ts, translations, rot_ts, rotations, scale_ts, scales)
        let mut node_channels: std::collections::HashMap<usize, (Vec<f32>, Vec<[f32; 3]>, Vec<f32>, Vec<[f32; 4]>, Vec<f32>, Vec<[f32; 3]>)> = std::collections::HashMap::new();

        #[cfg(debug_assertions)]
        let mut skipped_channels = 0usize;
        #[cfg(debug_assertions)]
        let mut mapped_channels = 0usize;
        #[cfg(debug_assertions)]
        {
            eprintln!("[anim] Animation '{}' has {} channels, node_to_joint map has {} entries",
                anim.name().unwrap_or("?"), anim.channels().count(), node_to_joint.len());
            for (ci, ch) in anim.channels().enumerate() {
                if ci < 5 {
                    let tn = ch.target().node();
                    eprintln!("[anim]   channel {} targets node {} '{}'  mapped={}",
                        ci, tn.index(), tn.name().unwrap_or("?"),
                        node_to_joint.contains_key(&tn.index()));
                }
            }
        }
        for channel in anim.channels() {
            let target_node = channel.target().node().index();
            let joint_index = match node_to_joint.get(&target_node) {
                Some(&ji) => {
                    #[cfg(debug_assertions)]
                    { mapped_channels += 1; }
                    ji
                },
                None => {
                    #[cfg(debug_assertions)]
                    { skipped_channels += 1; }
                    continue;
                },
            };

            let sampler = channel.sampler();
            let input_accessor = sampler.input();
            let output_accessor = sampler.output();

            let timestamps = read_accessor_f32(&gltf, &buffer_data, &input_accessor);
            let values = read_accessor_f32(&gltf, &buffer_data, &output_accessor);

            if let Some(&last) = timestamps.last() {
                if last > duration { duration = last; }
            }

            let entry = node_channels.entry(joint_index).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));

            match channel.target().property() {
                gltf::animation::Property::Translation => {
                    entry.0 = timestamps;
                    entry.1 = values.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                }
                gltf::animation::Property::Rotation => {
                    entry.2 = timestamps;
                    entry.3 = values.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
                }
                gltf::animation::Property::Scale => {
                    entry.4 = timestamps;
                    entry.5 = values.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                }
                _ => {}
            }
        }

        for (joint_index, (trans_ts, translations, rot_ts, rotations, scale_ts, scales)) in node_channels {
            // Use the longest timestamp array as the primary (for backward compat)
            let timestamps = if rot_ts.len() >= trans_ts.len() && rot_ts.len() >= scale_ts.len() {
                rot_ts.clone()
            } else if trans_ts.len() >= scale_ts.len() {
                trans_ts.clone()
            } else {
                scale_ts.clone()
            };
            channels.push(AnimationChannel {
                joint_index,
                timestamps,
                translations,
                rotation_timestamps: rot_ts,
                rotations,
                scale_timestamps: scale_ts,
                scales,
            });
        }

        let name = anim.name().unwrap_or("").to_string();
        #[cfg(debug_assertions)]
        {
            let total_kf: usize = channels.iter().map(|c| c.timestamps.len()).sum();
            let avg_kf = if !channels.is_empty() { total_kf / channels.len() } else { 0 };
            eprintln!("[anim] Animation '{}': {} channels mapped, {} skipped, duration={:.2}s, avg {}/ch keyframes",
                name, mapped_channels, skipped_channels, duration, avg_kf);
        }
        animations.push(AnimationData { channels, duration, name });
    }

    let joint_count = skeleton.as_ref().map(|s| s.joints.len()).unwrap_or(0);
    // Build reference rest rotations from the first animation at t=0
    let ref_rest_rotations = if animations.len() > 1 {
        if let Some(ref skel) = skeleton {
            let joint_count_s = skel.joints.len();
            let mut rest_rots = vec![[0.0f32, 0.0, 0.0, 1.0]; joint_count_s];
            // Sample first animation at t=0 to get reference rest rotations
            let anim0 = &animations[0];
            for ch in &anim0.channels {
                if ch.joint_index < joint_count_s && !ch.rotations.is_empty() {
                    rest_rots[ch.joint_index] = if ch.rotations.len() > 0 { ch.rotations[0] } else { [0.0, 0.0, 0.0, 1.0] };
                }
            }
            #[cfg(debug_assertions)]
            eprintln!("[retarget] Built reference rest rotations from anim 0 for {} joints", joint_count_s);
            Some(rest_rots)
        } else { None }
    } else { None };

    Some(ModelAnimation {
        skeleton,
        animations,
        joint_matrices: vec![mat4_identity(); joint_count],
        ref_rest_rotations,
    })
}

fn load_gltf_with_textures(
    data: &[u8],
    renderer: &mut crate::renderer::Renderer,
    base_dir: Option<&std::path::Path>,
) -> Option<ModelData> {
    let gltf = gltf::Gltf::from_slice(data).ok()?;

    // Get buffer data
    let mut buffer_data: Vec<Vec<u8>> = Vec::new();
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                if let Some(blob) = gltf.blob.as_ref() { buffer_data.push(blob.clone()); }
            }
            gltf::buffer::Source::Uri(uri) => {
                if let Some(encoded) = uri.strip_prefix("data:application/octet-stream;base64,") {
                    let mut decoded = Vec::new();
                    let _ = base64_decode(encoded, &mut decoded);
                    buffer_data.push(decoded);
                } else if let Some(dir) = base_dir {
                    // External .bin file alongside the .gltf.
                    let path = dir.join(uri);
                    match std::fs::read(&path) {
                        Ok(bytes) => buffer_data.push(bytes),
                        Err(_) => buffer_data.push(Vec::new()),
                    }
                } else {
                    buffer_data.push(Vec::new());
                }
            }
        }
    }

    // Pre-walk materials to identify which image indices are normal
    // maps. They need LEADR-style vector-space mip generation and per-
    // mip variance baked into alpha; see register_texture_kind.
    let mut normal_image_set: std::collections::HashSet<usize> = Default::default();
    for mat in gltf.materials() {
        if let Some(nt) = mat.normal_texture() {
            normal_image_set.insert(nt.texture().source().index());
        }
    }

    // Extract and register textures
    let mut texture_indices: Vec<u32> = Vec::new(); // maps glTF image index -> renderer texture index
    for (image_idx, image) in gltf.images().enumerate() {
        let is_normal = normal_image_set.contains(&image_idx);
        match image.source() {
            gltf::image::Source::View { view, .. } => {
                let buf_idx = view.buffer().index();
                if buf_idx < buffer_data.len() {
                    let offset = view.offset();
                    let length = view.length();
                    if offset + length <= buffer_data[buf_idx].len() {
                        let img_data = &buffer_data[buf_idx][offset..offset + length];
                        // Decode image (PNG/JPEG)
                        if let Ok(img) = image::load_from_memory(img_data) {
                            let rgba = img.to_rgba8();
                            let (w, h) = (rgba.width(), rgba.height());
                            let tex_idx = renderer.register_texture_kind(w, h, &rgba, is_normal);
                            texture_indices.push(tex_idx);
                        } else {
                            texture_indices.push(0); // fallback to white
                        }
                    } else {
                        texture_indices.push(0);
                    }
                } else {
                    texture_indices.push(0);
                }
            }
            gltf::image::Source::Uri { uri, .. } => {
                // External image file (loose glTF). Resolve relative to
                // the .gltf file's directory.
                let (bytes, effective_uri): (Option<Vec<u8>>, String) =
                    if let Some(encoded) = uri.strip_prefix("data:") {
                        let decoded = encoded.find(";base64,").map(|pos| {
                            let b64 = &encoded[pos + 8..];
                            let mut out = Vec::new();
                            let _ = base64_decode(b64, &mut out);
                            out
                        });
                        (decoded, uri.to_string())
                    } else if let Some(dir) = base_dir {
                        let primary = dir.join(uri);
                        if let Ok(b) = std::fs::read(&primary) {
                            (Some(b), uri.to_string())
                        } else {
                            // Asset packs sometimes ship DDS-only while the
                            // glTF still references a .png URI (Lumberyard
                            // Bistro does this). Retry with a .dds
                            // sibling before giving up.
                            let swapped = swap_extension(uri, "dds");
                            let alt = dir.join(&swapped);
                            match std::fs::read(&alt) {
                                Ok(b) => (Some(b), swapped),
                                Err(_) => (None, uri.to_string()),
                            }
                        }
                    } else {
                        (None, uri.to_string())
                    };
                match bytes.and_then(|b| decode_texture_bytes(&b, &effective_uri)) {
                    Some((rgba, w, h)) => {
                        texture_indices.push(renderer.register_texture_kind(w, h, &rgba, is_normal));
                    }
                    None => texture_indices.push(0),
                }
            }
        }
    }

    // Detect armature scale for skinned meshes.
    // Blender FBX imports set armature scale to 0.01 (cm→m conversion).
    // Vertex positions inherit this scale but bone transforms don't,
    // creating a unit mismatch. We apply the inverse to vertex positions.
    let skin_vertex_scale: f32 = {
        let mut scale = 1.0f32;
        for node in gltf.nodes() {
            if node.mesh().is_some() && node.skin().is_some() {
                // Found a skinned mesh node — look for parent with scale
                for parent in gltf.nodes() {
                    for child in parent.children() {
                        if child.index() == node.index() {
                            let (_, _, s) = parent.transform().decomposed();
                            let avg_scale = (s[0] + s[1] + s[2]) / 3.0;
                            if avg_scale > 0.001 && (avg_scale - 1.0).abs() > 0.01 {
                                scale = 1.0 / avg_scale;
                            }
                        }
                    }
                }
            }
        }
        // Fallback: check IBMs for large scale (Blender FBX baked 100x)
        if (scale - 1.0).abs() < 0.01 {
            if let Some(skin) = gltf.skins().next() {
                if let Some(accessor) = skin.inverse_bind_matrices() {
                    let view = accessor.view().unwrap();
                    let buf_idx = view.buffer().index();
                    if buf_idx < buffer_data.len() {
                        let offset = view.offset() + accessor.offset();
                        let data = &buffer_data[buf_idx];
                        if offset + 12 <= data.len() {
                            // Read first 3 floats (first column of first IBM)
                            let f0 = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
                            let f1 = f32::from_le_bytes([data[offset+4], data[offset+5], data[offset+6], data[offset+7]]);
                            let f2 = f32::from_le_bytes([data[offset+8], data[offset+9], data[offset+10], data[offset+11]]);
                            let diag = (f0*f0 + f1*f1 + f2*f2).sqrt();
                            if diag > 10.0 {
                                scale = diag;
                                #[cfg(debug_assertions)]
                                eprintln!("[skin] IBM col0 len={:.1}, applying {:.0}x vertex scale", diag, scale);
                            }
                        }
                    }
                }
            }
        }
        if (scale - 1.0).abs() > 0.01 {
            #[cfg(debug_assertions)]
            eprintln!("[skin] Applying {:.0}x vertex scale to compensate armature transform", scale);
        }
        scale
    };

    let mut meshes = Vec::new();
    let mut bbox_min = [f32::MAX; 3];
    let mut bbox_max = [f32::MIN; 3];

    // Walk the scene node tree to collect world-space transforms for
    // each mesh-referencing node. glTF supports instancing by having
    // multiple nodes reference the same mesh at different transforms
    // — Bistro uses this heavily (5910 nodes, 551 meshes: chairs,
    // bollards, chains, foliage repeated everywhere). We emit one
    // MeshData PER (mesh, transform) pair so every instance actually
    // shows up in the scene. Memory cost is linear in node count;
    // not great for deep instancing but correct. Animated / skinned
    // meshes are unaffected — the armature transforms apply on top.
    let mesh_count = gltf.meshes().count();
    let mut mesh_instances: Vec<Vec<[[f32; 4]; 4]>> = vec![Vec::new(); mesh_count];
    let identity = [[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            walk_scene_collect_instances(&node, &identity, &mut mesh_instances);
        }
    }

    for mesh in gltf.meshes() {
        let instances = mesh_instances[mesh.index()].clone();
        // Meshes reachable from no scene node would have no instances;
        // fall back to a single identity transform so orphan meshes
        // still render (matches prior behaviour for simple models).
        let instance_transforms: Vec<Option<[[f32; 4]; 4]>> = if instances.is_empty() {
            vec![None]
        } else {
            instances.into_iter().map(Some).collect()
        };

        for mesh_world in &instance_transforms {
            let mesh_world = *mesh_world;
            // Inverse-transpose 3×3 for normals under non-uniform scale.
            let normal_xform = mesh_world.map(|m| mat4_inverse_transpose_3x3(&m));
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| buffer_data.get(buf.index()).map(|d| d.as_slice()));
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            let normals: Vec<[f32; 3]> = reader.read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let tex_coords: Vec<[f32; 2]> = reader.read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            // Tangents (vec4: xyz = tangent, w = bitangent sign ±1).
            // If absent, we leave them as zero so the shader knows to
            // skip normal-map perturbation for this mesh.
            let tangents: Vec<[f32; 4]> = reader.read_tangents()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0; 4]; positions.len()]);

            // Get vertex colors if available
            let vert_colors: Option<Vec<[f32; 4]>> = reader.read_colors(0)
                .map(|iter| iter.into_rgba_f32().collect());

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let emissive_factor = mat.emissive_factor();

            let tex_idx_of = |img_idx: usize| -> Option<u32> {
                texture_indices.get(img_idx).copied()
            };

            let normal_tex_idx = mat.normal_texture()
                .and_then(|info| tex_idx_of(info.texture().source().index()));
            let emissive_tex_idx = mat.emissive_texture()
                .and_then(|info| tex_idx_of(info.texture().source().index()));
            let occlusion_tex_idx = mat.occlusion_texture()
                .and_then(|info| tex_idx_of(info.texture().source().index()));

            // Metallic-roughness first; fall back to
            // KHR_materials_pbrSpecularGlossiness when only that's
            // authored (Lumberyard Bistro + many FBX exports).
            // Conversion matches the load_gltf_staged path — see
            // specgloss_to_metalrough for the algorithm.
            let (mut base_color, mut metallic_factor, mut roughness_factor, tex_idx, mr_tex_idx) =
                if pbr.base_color_texture().is_none() {
                    if let Some(sg) = mat.pbr_specular_glossiness() {
                        let diffuse = sg.diffuse_factor();
                        let spec = sg.specular_factor();
                        let (base_color, metallic) =
                            specgloss_to_metalrough(diffuse, spec);
                        let roughness = 1.0 - sg.glossiness_factor();
                        let diffuse_tex = sg.diffuse_texture()
                            .and_then(|info| tex_idx_of(info.texture().source().index()));
                        (base_color, metallic, roughness, diffuse_tex, None)
                    } else {
                        (pbr.base_color_factor(), pbr.metallic_factor(), pbr.roughness_factor(), None, None)
                    }
                } else {
                    let tex = pbr.base_color_texture()
                        .and_then(|info| tex_idx_of(info.texture().source().index()));
                    let mr = pbr.metallic_roughness_texture()
                        .and_then(|info| tex_idx_of(info.texture().source().index()));
                    (pbr.base_color_factor(), pbr.metallic_factor(), pbr.roughness_factor(), tex, mr)
                };

            if let Some(t) = mat.transmission() {
                apply_transmission_hack(
                    t.transmission_factor(),
                    &mut base_color,
                    &mut metallic_factor,
                    &mut roughness_factor,
                );
            }

            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                let p = positions[i];
                for k in 0..3 {
                    if p[k] < bbox_min[k] { bbox_min[k] = p[k]; }
                    if p[k] > bbox_max[k] { bbox_max[k] = p[k]; }
                }
                let color = if let Some(ref vc) = vert_colors {
                    vc[i]
                } else {
                    [base_color[0], base_color[1], base_color[2], base_color[3]]
                };
                // Skin data (joints + weights)
                let joint_vals: Option<Vec<[u16; 4]>> = reader.read_joints(0)
                    .map(|iter| iter.into_u16().collect());
                let weight_vals: Option<Vec<[f32; 4]>> = reader.read_weights(0)
                    .map(|iter| iter.into_f32().collect());

                let jv = if let Some(ref j) = joint_vals {
                    [j[i][0] as f32, j[i][1] as f32, j[i][2] as f32, j[i][3] as f32]
                } else {
                    [0.0; 4]
                };
                let wv = if let Some(ref w) = weight_vals {
                    w[i]
                } else {
                    [0.0; 4]
                };
                // Apply inverse armature scale to skinned vertex positions
                let is_skinned = wv[0] + wv[1] + wv[2] + wv[3] > 0.01;
                let base_pos = if is_skinned && (skin_vertex_scale - 1.0).abs() > 0.01 {
                    [p[0] * skin_vertex_scale, p[1] * skin_vertex_scale, p[2] * skin_vertex_scale]
                } else {
                    p
                };
                // Bake the mesh's scene node transform into world-space
                // position/normal. Skinned meshes are NOT world-baked:
                // their node transform is expected to be consumed by the
                // armature, and the pose is driven by joint matrices at
                // draw time. Static (non-skinned) meshes get the baked
                // transform so drawModel's position/scale arguments
                // apply on top of the correct base pose.
                let (final_pos, final_normal, final_tangent) = if is_skinned {
                    (base_pos, normals[i], tangents[i])
                } else if let Some(xform) = mesh_world {
                    let t_in = [tangents[i][0], tangents[i][1], tangents[i][2]];
                    let t_out = match normal_xform {
                        // Tangents transform like positions (as directions)
                        // under the linear part of the transform — we use
                        // the upper 3×3 of the model matrix, not its
                        // inverse-transpose. But since our mesh_world is
                        // rigid-ish (no shear), the normal_xform gets us
                        // close enough for the common case. For a purely
                        // orthonormal node transform these are identical.
                        Some(ref n) => mat3_transform_vec(n, &t_in),
                        None => t_in,
                    };
                    (
                        mat4_transform_point(&xform, &base_pos),
                        match normal_xform {
                            Some(ref n) => mat3_transform_vec(n, &normals[i]),
                            None => normals[i],
                        },
                        [t_out[0], t_out[1], t_out[2], tangents[i][3]],
                    )
                } else {
                    (base_pos, normals[i], tangents[i])
                };
                // Update bbox to reflect the final (possibly transformed)
                // position so the camera auto-framing still works right.
                for k in 0..3 {
                    if final_pos[k] < bbox_min[k] { bbox_min[k] = final_pos[k]; }
                    if final_pos[k] > bbox_max[k] { bbox_max[k] = final_pos[k]; }
                }
                vertices.push(Vertex3D {
                    position: final_pos,
                    normal: final_normal,
                    color,
                    uv: tex_coords[i],
                    joints: jv,
                    weights: wv,
                    tangent: final_tangent,
                });
            }
            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            meshes.push(MeshData {
                vertices,
                indices,
                texture_idx: tex_idx,
                normal_texture_idx: normal_tex_idx,
                metallic_roughness_texture_idx: mr_tex_idx,
                emissive_texture_idx: emissive_tex_idx,
                occlusion_texture_idx: occlusion_tex_idx,
                metallic_factor,
                roughness_factor,
                emissive_factor,
                alpha_cutoff: alpha_cutoff_from_material(&mat),
            });
        }
        } // end instance loop
    }

    if meshes.is_empty() { return None; }
    Some(ModelData { meshes, bbox_min, bbox_max })
}

/// Like load_gltf_with_textures but decodes textures to RGBA without GPU registration.
/// Returns a StagedModel with decoded textures that can later be committed on the main thread.
pub fn load_gltf_staged(data: &[u8]) -> Option<crate::staging::StagedModel> {
    use crate::staging::{StagedTexture, StagedModel};

    let gltf = gltf::Gltf::from_slice(data).ok()?;

    let mut buffer_data: Vec<Vec<u8>> = Vec::new();
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                if let Some(blob) = gltf.blob.as_ref() { buffer_data.push(blob.clone()); }
            }
            gltf::buffer::Source::Uri(uri) => {
                if let Some(encoded) = uri.strip_prefix("data:application/octet-stream;base64,") {
                    let mut decoded = Vec::new();
                    let _ = base64_decode(encoded, &mut decoded);
                    buffer_data.push(decoded);
                } else {
                    buffer_data.push(Vec::new());
                }
            }
        }
    }

    // Decode textures to RGBA without GPU registration.
    // staged_textures[i] corresponds to glTF image index i.
    // texture_indices maps glTF image index -> 1-based index into staged_textures (0 = no texture).
    let mut staged_textures: Vec<StagedTexture> = Vec::new();
    let mut texture_indices: Vec<u32> = Vec::new();
    for image in gltf.images() {
        match image.source() {
            gltf::image::Source::View { view, .. } => {
                let buf_idx = view.buffer().index();
                if buf_idx < buffer_data.len() {
                    let offset = view.offset();
                    let length = view.length();
                    if offset + length <= buffer_data[buf_idx].len() {
                        let img_data = &buffer_data[buf_idx][offset..offset + length];
                        if let Ok(img) = image::load_from_memory(img_data) {
                            let rgba = img.to_rgba8();
                            let (w, h) = (rgba.width(), rgba.height());
                            staged_textures.push(StagedTexture {
                                data: rgba.into_raw(),
                                width: w,
                                height: h,
                            });
                            // 1-based index into staged_textures
                            texture_indices.push(staged_textures.len() as u32);
                        } else {
                            texture_indices.push(0);
                        }
                    } else {
                        texture_indices.push(0);
                    }
                } else {
                    texture_indices.push(0);
                }
            }
            _ => { texture_indices.push(0); }
        }
    }

    // Detect armature scale (same logic as load_gltf_with_textures)
    let skin_vertex_scale: f32 = {
        let mut scale = 1.0f32;
        for node in gltf.nodes() {
            if node.mesh().is_some() && node.skin().is_some() {
                for parent in gltf.nodes() {
                    for child in parent.children() {
                        if child.index() == node.index() {
                            let (_, _, s) = parent.transform().decomposed();
                            let avg_scale = (s[0] + s[1] + s[2]) / 3.0;
                            if avg_scale > 0.001 && (avg_scale - 1.0).abs() > 0.01 {
                                scale = 1.0 / avg_scale;
                            }
                        }
                    }
                }
            }
        }
        if (scale - 1.0).abs() < 0.01 {
            if let Some(skin) = gltf.skins().next() {
                if let Some(accessor) = skin.inverse_bind_matrices() {
                    let view = accessor.view().unwrap();
                    let buf_idx = view.buffer().index();
                    if buf_idx < buffer_data.len() {
                        let offset = view.offset() + accessor.offset();
                        let data = &buffer_data[buf_idx];
                        if offset + 12 <= data.len() {
                            let f0 = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
                            let f1 = f32::from_le_bytes([data[offset+4], data[offset+5], data[offset+6], data[offset+7]]);
                            let f2 = f32::from_le_bytes([data[offset+8], data[offset+9], data[offset+10], data[offset+11]]);
                            let diag = (f0*f0 + f1*f1 + f2*f2).sqrt();
                            if diag > 10.0 {
                                scale = diag;
                            }
                        }
                    }
                }
            }
        }
        scale
    };

    let mut meshes = Vec::new();
    let mut bbox_min = [f32::MAX; 3];
    let mut bbox_max = [f32::MIN; 3];

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| buffer_data.get(buf.index()).map(|d| d.as_slice()));
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            let normals: Vec<[f32; 3]> = reader.read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let tex_coords: Vec<[f32; 2]> = reader.read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            let tangents: Vec<[f32; 4]> = reader.read_tangents()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0; 4]; positions.len()]);
            let vert_colors: Option<Vec<[f32; 4]>> = reader.read_colors(0)
                .map(|iter| iter.into_rgba_f32().collect());

            let mat = primitive.material();
            let pbr = mat.pbr_metallic_roughness();
            let emissive_factor = mat.emissive_factor();

            let tex_idx_of = |img_idx: usize| -> Option<u32> {
                texture_indices.get(img_idx).copied()
            };

            let normal_tex_idx = mat.normal_texture()
                .and_then(|info| tex_idx_of(info.texture().source().index()));
            let emissive_tex_idx = mat.emissive_texture()
                .and_then(|info| tex_idx_of(info.texture().source().index()));
            let occlusion_tex_idx = mat.occlusion_texture()
                .and_then(|info| tex_idx_of(info.texture().source().index()));

            // Prefer the glTF 2.0 metallic-roughness model. Fall back
            // to KHR_materials_pbrSpecularGlossiness when the material
            // only ships the legacy spec-gloss extension (Lumberyard
            // Bistro and many FBX-exported scenes do). Conversion
            // follows the reference Khronos algorithm: pick metallic
            // that best explains the diffuse/specular split under the
            // assumption of a 0.04 dielectric baseline, then blend
            // base_color between diffuse and specular weighted by
            // metallic² (metals tint their reflection, dielectrics
            // show their diffuse).
            let (mut base_color, mut metallic_factor, mut roughness_factor, tex_idx, mr_tex_idx) =
                if pbr.base_color_texture().is_none() {
                    if let Some(sg) = mat.pbr_specular_glossiness() {
                        let diffuse = sg.diffuse_factor();
                        let spec = sg.specular_factor();
                        let (base_color, metallic) =
                            specgloss_to_metalrough(diffuse, spec);
                        let roughness = 1.0 - sg.glossiness_factor();
                        let diffuse_tex = sg.diffuse_texture()
                            .and_then(|info| tex_idx_of(info.texture().source().index()));
                        (base_color, metallic, roughness, diffuse_tex, None)
                    } else {
                        (pbr.base_color_factor(), pbr.metallic_factor(), pbr.roughness_factor(), None, None)
                    }
                } else {
                    let tex = pbr.base_color_texture()
                        .and_then(|info| tex_idx_of(info.texture().source().index()));
                    let mr = pbr.metallic_roughness_texture()
                        .and_then(|info| tex_idx_of(info.texture().source().index()));
                    (pbr.base_color_factor(), pbr.metallic_factor(), pbr.roughness_factor(), tex, mr)
                };

            if let Some(t) = mat.transmission() {
                apply_transmission_hack(
                    t.transmission_factor(),
                    &mut base_color,
                    &mut metallic_factor,
                    &mut roughness_factor,
                );
            }

            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                let p = positions[i];
                for k in 0..3 {
                    if p[k] < bbox_min[k] { bbox_min[k] = p[k]; }
                    if p[k] > bbox_max[k] { bbox_max[k] = p[k]; }
                }
                let color = if let Some(ref vc) = vert_colors {
                    vc[i]
                } else {
                    [base_color[0], base_color[1], base_color[2], base_color[3]]
                };
                let joint_vals: Option<Vec<[u16; 4]>> = reader.read_joints(0)
                    .map(|iter| iter.into_u16().collect());
                let weight_vals: Option<Vec<[f32; 4]>> = reader.read_weights(0)
                    .map(|iter| iter.into_f32().collect());
                let jv = if let Some(ref j) = joint_vals {
                    [j[i][0] as f32, j[i][1] as f32, j[i][2] as f32, j[i][3] as f32]
                } else {
                    [0.0; 4]
                };
                let wv = if let Some(ref w) = weight_vals { w[i] } else { [0.0; 4] };
                let is_skinned = wv[0] + wv[1] + wv[2] + wv[3] > 0.01;
                let final_pos = if is_skinned && (skin_vertex_scale - 1.0).abs() > 0.01 {
                    [p[0] * skin_vertex_scale, p[1] * skin_vertex_scale, p[2] * skin_vertex_scale]
                } else {
                    p
                };
                vertices.push(Vertex3D {
                    position: final_pos,
                    normal: normals[i],
                    color,
                    uv: tex_coords[i],
                    joints: jv,
                    weights: wv,
                    tangent: tangents[i],
                });
            }
            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            meshes.push(MeshData {
                vertices,
                indices,
                texture_idx: tex_idx,
                normal_texture_idx: normal_tex_idx,
                metallic_roughness_texture_idx: mr_tex_idx,
                emissive_texture_idx: emissive_tex_idx,
                occlusion_texture_idx: occlusion_tex_idx,
                metallic_factor,
                roughness_factor,
                emissive_factor,
                alpha_cutoff: alpha_cutoff_from_material(&mat),
            });
        }
    }

    if meshes.is_empty() { return None; }
    Some(StagedModel {
        model: ModelData { meshes, bbox_min, bbox_max },
        textures: staged_textures,
    })
}

fn load_gltf(data: &[u8]) -> Option<ModelData> {
    let gltf = gltf::Gltf::from_slice(data).ok()?;

    // Get buffer data (for .glb, embedded; for .gltf, inline base64)
    let mut buffer_data: Vec<Vec<u8>> = Vec::new();
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                if let Some(blob) = gltf.blob.as_ref() {
                    buffer_data.push(blob.clone());
                }
            }
            gltf::buffer::Source::Uri(uri) => {
                if let Some(encoded) = uri.strip_prefix("data:application/octet-stream;base64,") {
                    // Try to decode base64 inline data
                    let mut decoded = Vec::new();
                    let _ = base64_decode(encoded, &mut decoded);
                    buffer_data.push(decoded);
                } else {
                    buffer_data.push(Vec::new());
                }
            }
        }
    }

    let mut meshes = Vec::new();
    let mut bbox_min = [f32::MAX; 3];
    let mut bbox_max = [f32::MIN; 3];

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| buffer_data.get(buf.index()).map(|d| d.as_slice()));

            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            let normals: Vec<[f32; 3]> = reader.read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            let tex_coords: Vec<[f32; 2]> = reader.read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            // Material base color
            let base_color = primitive.material().pbr_metallic_roughness()
                .base_color_factor();
            let color = [base_color[0], base_color[1], base_color[2], base_color[3]];

            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                let p = positions[i];
                for k in 0..3 {
                    if p[k] < bbox_min[k] { bbox_min[k] = p[k]; }
                    if p[k] > bbox_max[k] { bbox_max[k] = p[k]; }
                }
                // Read skin data if available
                let joint_vals: Option<Vec<[u16; 4]>> = reader.read_joints(0)
                    .map(|iter| iter.into_u16().collect());
                let weight_vals: Option<Vec<[f32; 4]>> = reader.read_weights(0)
                    .map(|iter| iter.into_f32().collect());
                let jv = if let Some(ref j) = joint_vals {
                    [j[i][0] as f32, j[i][1] as f32, j[i][2] as f32, j[i][3] as f32]
                } else { [0.0; 4] };
                let wv = if let Some(ref w) = weight_vals { w[i] } else { [0.0; 4] };

                vertices.push(Vertex3D {
                    position: p,
                    normal: normals[i],
                    color,
                    uv: tex_coords[i],
                    joints: jv,
                    weights: wv,
                    tangent: [0.0; 4],
                });
            }

            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            meshes.push(MeshData { vertices, indices, texture_idx: None, normal_texture_idx: None, metallic_roughness_texture_idx: None, emissive_texture_idx: None, occlusion_texture_idx: None, metallic_factor: 0.0, roughness_factor: 1.0, emissive_factor: [0.0; 3], alpha_cutoff: 0.0 });
        }
    }

    if meshes.is_empty() { return None; }
    Some(ModelData { meshes, bbox_min, bbox_max })
}

/// Convert a KHR_materials_pbrSpecularGlossiness (diffuse + specular
/// + glossiness) material to the metallic-roughness model. Uses the
/// reference Khronos two-path formula so materials authored in
/// Substance/3ds Max/FBX pipelines (Lumberyard Bistro, many ORCA
/// assets) render correctly on a metal-rough pipeline.
///
/// High-level idea: assume a 0.04 dielectric reflectance baseline,
/// solve for the metallic factor that best reconciles the authored
/// diffuse and specular colors, then blend base_color between the
/// two weighted by metallic². Metals have specular ≈ albedo, so the
/// specular color becomes their base_color; dielectrics carry their
/// diffuse color through at metallic ≈ 0.
///
/// Map glTF alpha mode + cutoff to a single shader cutoff value.
/// OPAQUE → 0.0 (fragment shader's `< cutoff` discard never fires).
/// MASK   → material-authored cutoff (default 0.5 per glTF spec).
/// BLEND  → treated as MASK @ 0.5 since we don't yet have a sorted
///          transparent pipeline. Better than silently rendering
///          foliage + fabric as fully opaque — an alpha-cutout leaf
///          card is at least the right *shape*.
fn alpha_cutoff_from_material(mat: &gltf::Material) -> f32 {
    match mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque => 0.0,
        gltf::material::AlphaMode::Mask => mat.alpha_cutoff().unwrap_or(0.5),
        gltf::material::AlphaMode::Blend => 0.5,
    }
}

/// Fake KHR_materials_transmission as a near-mirror dielectric.
///
/// We don't implement real refractive transmission (no back-buffer
/// refraction pass, no thin-walled/volume distinction), so a
/// transmission=0.9 glass pane loads as a plain diffuse white surface
/// that drowns the 4% Fresnel specular in a bright diffuse term — the
/// classic "painted white window" look.
///
/// As a stand-in we force heavy transmission materials to behave like
/// chrome: metallic=1 so f0=base_color (not 0.04), roughness ≤ 0.05
/// so reflections stay crisp, and a mild (0.85×) tint on base_color
/// so pure-white glass doesn't read as perfectly reflective chrome.
/// Not physically correct for glass — real glass only reflects 4% at
/// normal incidence — but it matches how windows *read* in photos
/// (reflecting sky/buildings) far better than a flat diffuse surface.
fn apply_transmission_hack(
    transmission: f32,
    base_color: &mut [f32; 4],
    metallic: &mut f32,
    roughness: &mut f32,
) {
    if transmission > 0.5 {
        *metallic = 1.0;
        *roughness = roughness.min(0.05);
        base_color[0] *= 0.85;
        base_color[1] *= 0.85;
        base_color[2] *= 0.85;
        base_color[3] = 1.0;
    }
}

/// Reference: Khronos glTF sample specGloss→metallicRoughness
/// converter (https://github.com/KhronosGroup/glTF/pull/1355).
fn specgloss_to_metalrough(diffuse: [f32; 4], specular: [f32; 3]) -> ([f32; 4], f32) {
    let dielectric_specular = 0.04_f32;
    let epsilon = 1e-6_f32;

    let one_minus_dielectric = 1.0 - dielectric_specular;
    let diffuse_max = diffuse[0].max(diffuse[1]).max(diffuse[2]);
    let specular_max = specular[0].max(specular[1]).max(specular[2]);

    // Solve a quadratic for metallic. Coefficients from the Khronos
    // reference: mapping perceived brightness split between diffuse
    // and specular back to a single metallic parameter.
    let a = dielectric_specular;
    let b = diffuse_max * one_minus_dielectric / dielectric_specular.max(epsilon)
          + specular_max
          - 2.0 * dielectric_specular;
    let c = dielectric_specular - specular_max;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let metallic = if specular_max < dielectric_specular {
        0.0
    } else {
        (((-b + discriminant.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)).min(1.0)
    };

    // base_color = mix(diffuse, specular, metallic²) with the diffuse
    // branch scaled to undo the dielectric energy split.
    let diffuse_branch_scale = one_minus_dielectric
        / (1.0 - metallic * dielectric_specular).max(epsilon);
    let metal_weight = metallic * metallic;
    let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let r = lerp(diffuse[0] * diffuse_branch_scale, specular[0], metal_weight);
    let g = lerp(diffuse[1] * diffuse_branch_scale, specular[1], metal_weight);
    let bl = lerp(diffuse[2] * diffuse_branch_scale, specular[2], metal_weight);
    ([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), bl.clamp(0.0, 1.0), diffuse[3]], metallic)
}

/// Replace the extension on a URI (keeps directories / query strings
/// untouched). Used to fall back from `foo.png` → `foo.dds` when a
/// glTF references a PNG URI that isn't on disk but the DDS sibling is.
fn swap_extension(uri: &str, new_ext: &str) -> String {
    let q = uri.find('?').unwrap_or(uri.len());
    let (path, query) = uri.split_at(q);
    let new_path = match path.rfind('.') {
        Some(dot) if dot > path.rfind('/').unwrap_or(0) => {
            format!("{}.{}", &path[..dot], new_ext)
        }
        _ => format!("{}.{}", path, new_ext),
    };
    format!("{}{}", new_path, query)
}

/// Decode a texture byte slice into RGBA8 pixels + dimensions. Tries
/// DDS first when the URI extension suggests it (for asset packs like
/// Lumberyard Bistro that ship BC-compressed textures), falling back
/// to the `image` crate for PNG/JPEG/etc. Returns None on failure.
fn decode_texture_bytes(bytes: &[u8], uri: &str) -> Option<(Vec<u8>, u32, u32)> {
    let is_dds = uri.to_ascii_lowercase().ends_with(".dds")
        || bytes.len() >= 4 && &bytes[..4] == b"DDS ";
    if is_dds {
        if let Ok(dds) = image_dds::ddsfile::Dds::read(bytes) {
            // Decode mip 0 → RGBA8. image_from_dds handles the common
            // BC1–BC7 formats; anything it can't decode falls through
            // to the image crate which will almost certainly fail too.
            if let Ok(rgba) = image_dds::image_from_dds(&dds, 0) {
                let (w, h) = (rgba.width(), rgba.height());
                return Some((rgba.into_raw(), w, h));
            }
        }
    }
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Some((rgba.into_raw(), w, h))
}

fn base64_decode(input: &str, output: &mut Vec<u8>) {
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in input.as_bytes() {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' => continue,
            _ => continue,
        };
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
}

// ---- Catmull-Rom spline helpers (Q9) ----

fn catmull_rom_point(points: &[f32], n: usize, segment: usize, t: f32) -> [f32; 3] {
    // Indices: p0 = segment - 1, p1 = segment, p2 = segment + 1, p3 = segment + 2.
    // Clamp at boundaries.
    let i0 = if segment > 0 { segment - 1 } else { 0 };
    let i1 = segment;
    let i2 = if segment + 1 < n { segment + 1 } else { n - 1 };
    let i3 = if segment + 2 < n { segment + 2 } else { n - 1 };

    let p0 = [points[i0 * 3], points[i0 * 3 + 1], points[i0 * 3 + 2]];
    let p1 = [points[i1 * 3], points[i1 * 3 + 1], points[i1 * 3 + 2]];
    let p2 = [points[i2 * 3], points[i2 * 3 + 1], points[i2 * 3 + 2]];
    let p3 = [points[i3 * 3], points[i3 * 3 + 1], points[i3 * 3 + 2]];

    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        out[k] = 0.5 * (
            (2.0 * p1[k]) +
            (-p0[k] + p2[k]) * t +
            (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2 +
            (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3
        );
    }
    out
}

fn update_bounds(bmin: &mut [f32; 3], bmax: &mut [f32; 3], x: f32, y: f32, z: f32) {
    if x < bmin[0] { bmin[0] = x; }
    if y < bmin[1] { bmin[1] = y; }
    if z < bmin[2] { bmin[2] = z; }
    if x > bmax[0] { bmax[0] = x; }
    if y > bmax[1] { bmax[1] = y; }
    if z > bmax[2] { bmax[2] = z; }
}
