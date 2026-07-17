use crate::handles::HandleRegistry;
use crate::renderer::Vertex3D;
use std::sync::Arc;

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

pub use crate::anim_mixer::AnimMixer;

// glTF/GLB parsing lives in models_gltf.rs, models3d-gated so builds
// without the feature drop the gltf/image_dds deps (EN-014, re-fixed by
// EN-063). Everything else in this file compiles in every configuration.
#[cfg(feature = "models3d")]
#[path = "models_gltf.rs"]
mod gltf_load;
#[cfg(feature = "models3d")]
pub use gltf_load::load_gltf_staged;

pub struct ModelAnimation {
    /// EN-055 — the parsed clip data (skeleton, keyframe tracks, rest
    /// rotations) is IMMUTABLE after load and shared between instances via
    /// `Arc`: `instantiate_animation` clones the handles, not the data. Only
    /// the fields below the shared block are per-instance state.
    pub skeleton: Option<Arc<SkeletonData>>,
    pub animations: Arc<Vec<AnimationData>>,
    /// Reference rest-pose rotations (from first animation, sampled at t=0).
    /// Used for retargeting when multiple armatures have different rest orientations.
    pub ref_rest_rotations: Option<Arc<Vec<[f32; 4]>>>,
    // ---- per-instance state from here down ---------------------------------
    pub joint_matrices: Vec<[[f32; 4]; 4]>,
    /// EN-028 mixer state.
    pub mixer: AnimMixer,
    /// EN-033 — joint world transforms *before* the inverse-bind multiply.
    /// `joint_matrices` is skinning-space and useless for attaching props;
    /// this is the model-space transform a socket actually wants.
    pub joint_world: Vec<[[f32; 4]; 4]>,
    /// Cached per-joint layer weights, rebuilt when `layer_mask_root` changes.
    pub mask_weights: Vec<f32>,
    pub mask_cached_root: i32,
}

pub struct ModelManager {
    pub models: HandleRegistry<ModelData>,
    pub animations: HandleRegistry<ModelAnimation>,
    /// Scratch buffers for the array-free mesh-upload path. Perry 0.5.1171
    /// rejects passing a JS `number[]` to a native `i64` pointer param
    /// (strict safe-integer check), so `createMesh` instead pushes vertex
    /// floats / indices one scalar at a time through `mesh_scratch_push_*`
    /// (all `f64` ABI) and then builds the mesh from these. Mirrors the
    /// physics subsystem's `scratch_*` shape-upload path.
    pub scratch_f32: Vec<f32>,
    pub scratch_u32: Vec<u32>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: HandleRegistry::new(),
            animations: HandleRegistry::new(),
            scratch_f32: Vec::new(),
            scratch_u32: Vec::new(),
        }
    }

    pub fn mesh_scratch_reset(&mut self) {
        self.scratch_f32.clear();
        self.scratch_u32.clear();
    }
    pub fn mesh_scratch_push_f32(&mut self, v: f32) { self.scratch_f32.push(v); }
    pub fn mesh_scratch_push_u32(&mut self, v: u32) { self.scratch_u32.push(v); }

    /// Build a mesh from the scratch buffers: `vertex_count` vertices of 12
    /// floats each in `scratch_f32`, `index_count` indices in `scratch_u32`.
    pub fn create_mesh_from_scratch(&mut self, vertex_count: u32, index_count: u32) -> f64 {
        let need_f = vertex_count as usize * 12;
        let need_u = index_count as usize;
        if vertex_count == 0 || self.scratch_f32.len() < need_f || self.scratch_u32.len() < need_u {
            return 0.0;
        }
        // Clone out so create_mesh's &self borrow doesn't alias scratch.
        let verts: Vec<f32> = self.scratch_f32[..need_f].to_vec();
        let inds: Vec<u32> = self.scratch_u32[..need_u].to_vec();
        self.create_mesh(&verts, &inds)
    }

    /// Read-only view of the f32 scratch buffer, for consumers that lay their
    /// own data out in it (the spline ribbon packs positions then widths).
    pub fn scratch_floats(&self) -> &[f32] {
        &self.scratch_f32
    }

    /// Take the scratch buffers as raw vertex floats + indices, for callers
    /// that build something other than a Model out of them (the scene graph's
    /// `update_geometry`). Same 12-floats-per-vertex layout as
    /// `create_mesh_from_scratch`; returns None if the scratch is short.
    pub fn take_scratch_geometry(
        &self,
        vertex_count: u32,
        index_count: u32,
    ) -> Option<(Vec<f32>, Vec<u32>)> {
        let need_f = vertex_count as usize * 12;
        let need_u = index_count as usize;
        if vertex_count == 0 || self.scratch_f32.len() < need_f || self.scratch_u32.len() < need_u {
            return None;
        }
        Some((
            self.scratch_f32[..need_f].to_vec(),
            self.scratch_u32[..need_u].to_vec(),
        ))
    }

    #[cfg(feature = "models3d")]
    pub fn load_model(&mut self, file_data: &[u8]) -> f64 {
        match gltf_load::load_gltf(file_data) {
            Some(model) => self.models.alloc(model),
            None => 0.0,
        }
    }

    #[cfg(feature = "models3d")]
    pub fn load_model_with_textures(&mut self, file_data: &[u8], renderer: &mut crate::renderer::Renderer) -> f64 {
        match gltf_load::load_gltf_with_textures(file_data, renderer, None) {
            Some(model) => self.models.alloc(model),
            None => 0.0,
        }
    }

    /// Like `load_model_with_textures` but also resolves external `.bin`
    /// and image URIs relative to `base_dir` — required for loose glTF
    /// files (as opposed to single-file .glb). Intel Sponza etc.
    #[cfg(feature = "models3d")]
    pub fn load_model_with_textures_from_path(
        &mut self,
        file_data: &[u8],
        base_dir: &std::path::Path,
        renderer: &mut crate::renderer::Renderer,
    ) -> f64 {
        match gltf_load::load_gltf_with_textures(file_data, renderer, Some(base_dir)) {
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

    /// EN-025 — the ragdoll writes joint matrices directly, bypassing the
    /// sampler entirely: once a thing is dead, physics owns its pose.
    pub fn get_animation_mut(&mut self, handle: f64) -> Option<&mut ModelAnimation> {
        self.animations.get_mut(handle)
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

    #[cfg(feature = "models3d")]
    pub fn load_model_animation(&mut self, file_data: &[u8]) -> f64 {
        match gltf_load::load_gltf_animation(file_data) {
            Some(anim) => self.animations.alloc(anim),
            None => 0.0,
        }
    }

    /// EN-055 — a new animation INSTANCE over an already-loaded clip set.
    /// The parsed data (skeleton, keyframe tracks, rest rotations) is shared
    /// via `Arc`; the mixer, joint matrices and mask cache are fresh — so N
    /// characters get independent clocks/fades without N GLB re-parses
    /// (which was 5.5 s of the shooter's 8 s boot). Returns 0 for a dead
    /// source handle, mirroring load's failure convention.
    pub fn instantiate_animation(&mut self, src: f64) -> f64 {
        let inst = match self.animations.get(src) {
            Some(a) => ModelAnimation {
                skeleton: a.skeleton.clone(),
                animations: a.animations.clone(),
                ref_rest_rotations: a.ref_rest_rotations.clone(),
                joint_matrices: a.joint_matrices.clone(),
                mixer: AnimMixer::default(),
                joint_world: a.joint_world.clone(),
                mask_weights: vec![0.0; a.mask_weights.len()],
                mask_cached_root: -1,
            },
            None => return 0.0,
        };
        self.animations.alloc(inst)
    }

    pub fn update_model_animation(&mut self, handle: f64, anim_index: usize, time: f32) {
        if let Some(model_anim) = self.animations.get_mut(handle) {
            let skeleton = match &model_anim.skeleton {
                Some(s) => s,
                None => return,
            };
            if anim_index >= model_anim.animations.len() { return; }
            #[cfg(debug_assertions)]
            let joint_count = skeleton.joints.len();

            let pose = sample_local_pose(skeleton, &model_anim.animations[anim_index], time, true);
            model_anim.apply_pose(&pose);

            #[cfg(debug_assertions)]
            {
                static mut DEBUG_PRINTED: bool = false;
                unsafe {
                    if !DEBUG_PRINTED && joint_count > 0 {
                        DEBUG_PRINTED = true;
                        eprintln!("[anim] joints={}, t={:.3}, anim_index={}", joint_count, time, anim_index);
                    }
                }
            }
        }
    }

    // ---- EN-028 mixer -----------------------------------------------------

    /// Start a transition to `clip`. Re-requesting the clip already playing is
    /// a no-op, so game code can call this unconditionally every frame ("I
    /// want to be walking") instead of tracking edges itself — which is how
    /// this gets used in practice and where the pops came from before.
    pub fn anim_play(&mut self, handle: f64, clip: usize, fade: f32, speed: f32, looping: bool) {
        if let Some(ma) = self.animations.get_mut(handle) {
            if clip >= ma.animations.len() { return; }
            let m = &mut ma.mixer;
            // Re-requesting the clip that is ALREADY current is a no-op by
            // contract ("safe to call every frame with the clip you want"), and
            // that has to hold DURING a fade too — a fade is a fade *to*
            // cur_clip, so the right thing is to let it finish.
            //
            // This used to also require `m.fade_dur <= 0.0`. A caller doing the
            // documented thing then fell through to the restart path on every
            // frame of the fade: it re-seeded the fade (fade_t = 0, so the fade
            // could never complete) and reset cur_time = 0. The base clip was
            // pinned at t≈0 forever, from the first clip change onward — every
            // animated character in a game froze the moment it stopped idling.
            if m.started && m.cur_clip == clip {
                m.cur_speed = speed;
                m.cur_loop = looping;
                return;
            }
            if m.started && fade > 0.0 {
                m.prev_clip = m.cur_clip;
                m.prev_time = m.cur_time;
                m.prev_speed = m.cur_speed;
                m.prev_loop = m.cur_loop;
                m.fade_dur = fade;
                m.fade_t = 0.0;
            } else {
                m.fade_dur = 0.0;
                m.fade_t = 0.0;
            }
            m.cur_clip = clip;
            m.cur_time = 0.0;
            m.cur_speed = speed;
            m.cur_loop = looping;
            m.finished = false;
            m.started = true;
        }
    }

    /// Masked layer: `clip` drives every joint at or below `mask_root`,
    /// blended in by `weight`. weight <= 0 (or clip < 0) turns it off.
    pub fn anim_set_layer(&mut self, handle: f64, clip: i32, weight: f32, mask_root: i32, speed: f32, looping: bool) {
        if let Some(ma) = self.animations.get_mut(handle) {
            let m = &mut ma.mixer;
            let off = clip < 0 || weight <= 0.0 || (clip as usize) >= ma.animations.len();
            if off {
                m.layer_clip = -1;
                m.layer_weight = 0.0;
                return;
            }
            if m.layer_clip != clip {
                m.layer_time = 0.0;
                m.layer_clip = clip;
            }
            m.layer_weight = weight.clamp(0.0, 1.0);
            m.layer_mask_root = mask_root;
            m.layer_speed = speed;
            m.layer_loop = looping;
        }
    }

    pub fn anim_set_root_motion(&mut self, handle: f64, on: bool) {
        if let Some(ma) = self.animations.get_mut(handle) {
            ma.mixer.root_motion = on;
            ma.mixer.root_delta = [0.0; 3];
        }
    }

    pub fn anim_finished(&self, handle: f64) -> bool {
        self.animations.get(handle).map(|m| m.mixer.finished).unwrap_or(true)
    }

    pub fn anim_clip_duration(&self, handle: f64, clip: usize) -> f32 {
        self.animations.get(handle)
            .and_then(|m| m.animations.get(clip))
            .map(|a| a.duration)
            .unwrap_or(0.0)
    }

    pub fn anim_root_delta(&self, handle: f64) -> [f32; 3] {
        self.animations.get(handle).map(|m| m.mixer.root_delta).unwrap_or([0.0; 3])
    }

    pub fn find_joint(&self, handle: f64, name: &str) -> i32 {
        if let Some(ma) = self.animations.get(handle) {
            if let Some(sk) = &ma.skeleton {
                for (i, j) in sk.joints.iter().enumerate() {
                    if j.name == name { return i as i32; }
                }
                // Fall back to a case-insensitive contains match: exporters
                // decorate names ("mixamorig:Hand_R", "Bip01 R Hand") often
                // enough that an exact match is the exception, not the rule.
                let want = name.to_ascii_lowercase();
                for (i, j) in sk.joints.iter().enumerate() {
                    if j.name.to_ascii_lowercase().contains(&want) { return i as i32; }
                }
            }
        }
        -1
    }

    /// Model-space transform of a joint (EN-033 sockets). Valid after the
    /// frame's `advance_and_update`.
    pub fn joint_world(&self, handle: f64, joint: usize) -> Option<[[f32; 4]; 4]> {
        self.animations.get(handle)
            .and_then(|m| m.joint_world.get(joint))
            .copied()
    }

    /// Advance every mixer clock by `dt` and rebuild the pose. One call per
    /// model per frame; the game never touches clip time.
    pub fn advance_and_update(&mut self, handle: f64, dt: f32) {
        if let Some(ma) = self.animations.get_mut(handle) {
            if ma.skeleton.is_none() || ma.animations.is_empty() { return; }
            if !ma.mixer.started { ma.mixer.started = true; }

            // --- advance clocks
            let cur_dur = ma.animations.get(ma.mixer.cur_clip).map(|a| a.duration).unwrap_or(0.0);
            let t_before = ma.mixer.cur_time;
            let t_raw = ma.mixer.cur_time + dt * ma.mixer.cur_speed;
            let mut wrapped = false;
            ma.mixer.cur_time = if cur_dur <= 0.0 {
                0.0
            } else if ma.mixer.cur_loop {
                if t_raw >= cur_dur { wrapped = true; }
                t_raw.rem_euclid(cur_dur)
            } else if t_raw >= cur_dur {
                ma.mixer.finished = true;
                cur_dur
            } else {
                t_raw
            };

            if ma.mixer.fade_dur > 0.0 {
                let prev_dur = ma.animations.get(ma.mixer.prev_clip).map(|a| a.duration).unwrap_or(0.0);
                let pt = ma.mixer.prev_time + dt * ma.mixer.prev_speed;
                ma.mixer.prev_time = if prev_dur <= 0.0 {
                    0.0
                } else if ma.mixer.prev_loop {
                    pt.rem_euclid(prev_dur)
                } else {
                    pt.min(prev_dur)
                };
                ma.mixer.fade_t += dt;
                if ma.mixer.fade_t >= ma.mixer.fade_dur {
                    ma.mixer.fade_dur = 0.0;
                    ma.mixer.fade_t = 0.0;
                }
            }

            if ma.mixer.layer_clip >= 0 {
                let li = ma.mixer.layer_clip as usize;
                let ldur = ma.animations.get(li).map(|a| a.duration).unwrap_or(0.0);
                let lt = ma.mixer.layer_time + dt * ma.mixer.layer_speed;
                ma.mixer.layer_time = if ldur <= 0.0 {
                    0.0
                } else if ma.mixer.layer_loop {
                    lt.rem_euclid(ldur)
                } else {
                    lt.min(ldur)
                };
            }

            // --- sample + blend the base track
            let skel = ma.skeleton.as_ref().unwrap();
            let strip_root = !ma.mixer.root_motion;
            let mut pose = sample_local_pose(skel, &ma.animations[ma.mixer.cur_clip], ma.mixer.cur_time, strip_root);

            if ma.mixer.fade_dur > 0.0 {
                let w = (ma.mixer.fade_t / ma.mixer.fade_dur).clamp(0.0, 1.0);
                // Smoothstep the fade — a linear pose blend still reads as a
                // slope discontinuity at both ends of the transition.
                let w = w * w * (3.0 - 2.0 * w);
                let prev = sample_local_pose(skel, &ma.animations[ma.mixer.prev_clip], ma.mixer.prev_time, strip_root);
                blend_pose(&mut pose, &prev, 1.0 - w, None);
            }

            // --- root motion: delta of the root joint's authored translation
            if ma.mixer.root_motion && !skel.joints.is_empty() {
                let cd = cur_dur;
                let anim = &ma.animations[ma.mixer.cur_clip];
                let p_now = root_translation_at(skel, anim, ma.mixer.cur_time);
                let p_old = root_translation_at(skel, anim, t_before);
                let d = if wrapped && cd > 0.0 {
                    let p_end = root_translation_at(skel, anim, cd);
                    let p_start = root_translation_at(skel, anim, 0.0);
                    [
                        (p_end[0] - p_old[0]) + (p_now[0] - p_start[0]),
                        (p_end[1] - p_old[1]) + (p_now[1] - p_start[1]),
                        (p_end[2] - p_old[2]) + (p_now[2] - p_start[2]),
                    ]
                } else {
                    [p_now[0] - p_old[0], p_now[1] - p_old[1], p_now[2] - p_old[2]]
                };
                ma.mixer.root_delta = d;
                // The delta is handed to the character controller, so the pose
                // itself must not also carry it or the model double-moves.
                pose.0[0] = skel.joints[0].rest_translation;
            } else {
                ma.mixer.root_delta = [0.0; 3];
            }

            // --- masked layer over the top
            if ma.mixer.layer_clip >= 0 && ma.mixer.layer_weight > 0.0 {
                let root = ma.mixer.layer_mask_root;
                if ma.mask_cached_root != root {
                    ma.mask_weights = build_mask_weights(skel, root);
                    ma.mask_cached_root = root;
                }
                let li = ma.mixer.layer_clip as usize;
                let lpose = sample_local_pose(skel, &ma.animations[li], ma.mixer.layer_time, true);
                let w = ma.mixer.layer_weight;
                let mask = ma.mask_weights.clone();
                blend_pose(&mut pose, &lpose, w, Some(&mask));
            }

            ma.apply_pose(&pose);
        }
    }
}

type LocalPose = (Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>);

impl ModelAnimation {
    /// Local TRS pose -> world transforms -> skinning matrices. Keeps the
    /// world transforms around too, because that is what sockets read.
    fn apply_pose(&mut self, pose: &LocalPose) {
        let skeleton = match &self.skeleton { Some(s) => s, None => return };
        let joint_count = skeleton.joints.len();
        if self.joint_matrices.len() != joint_count {
            self.joint_matrices = vec![mat4_identity(); joint_count];
        }
        if self.joint_world.len() != joint_count {
            self.joint_world = vec![mat4_identity(); joint_count];
        }
        let mut world = vec![mat4_identity(); joint_count];
        for &root in &skeleton.root_joints {
            compute_joint_transforms(skeleton, root, &mat4_identity(), &pose.0, &pose.1, &pose.2, &mut world);
        }
        for i in 0..joint_count {
            self.joint_matrices[i] = mat4_mul(&world[i], &skeleton.joints[i].inverse_bind);
        }
        self.joint_world.copy_from_slice(&world);
    }
}

/// Sample one clip into a local TRS pose, rest pose as the fallback for
/// joints the clip does not animate.
fn sample_local_pose(skeleton: &SkeletonData, anim: &AnimationData, time: f32, strip_root: bool) -> LocalPose {
    let joint_count = skeleton.joints.len();
    let mut t: Vec<[f32; 3]> = skeleton.joints.iter().map(|j| j.rest_translation).collect();
    let mut r: Vec<[f32; 4]> = skeleton.joints.iter().map(|j| j.rest_rotation).collect();
    let mut s: Vec<[f32; 3]> = skeleton.joints.iter().map(|j| j.rest_scale).collect();

    let time = if anim.duration > 0.0 { time.rem_euclid(anim.duration) } else { 0.0 };

    for channel in &anim.channels {
        let ji = channel.joint_index;
        if ji >= joint_count { continue; }
        if !channel.translations.is_empty() && !channel.timestamps.is_empty() {
            t[ji] = sample_vec3(&channel.timestamps, &channel.translations, time);
        }
        if !channel.rotations.is_empty() {
            let ts = if !channel.rotation_timestamps.is_empty() { &channel.rotation_timestamps } else { &channel.timestamps };
            if !ts.is_empty() { r[ji] = sample_quat(ts, &channel.rotations, time); }
        }
        if !channel.scales.is_empty() {
            let ts = if !channel.scale_timestamps.is_empty() { &channel.scale_timestamps } else { &channel.timestamps };
            if !ts.is_empty() { s[ji] = sample_vec3(ts, &channel.scales, time); }
        }
    }

    if strip_root && joint_count > 0 {
        t[0] = skeleton.joints[0].rest_translation;
    }
    (t, r, s)
}

/// The root joint's authored translation at `time` — the raw channel value,
/// *not* the rest-locked one, which is the whole point of root motion.
fn root_translation_at(skeleton: &SkeletonData, anim: &AnimationData, time: f32) -> [f32; 3] {
    if skeleton.joints.is_empty() { return [0.0; 3]; }
    let time = if anim.duration > 0.0 { time.clamp(0.0, anim.duration) } else { 0.0 };
    for channel in &anim.channels {
        if channel.joint_index == 0 && !channel.translations.is_empty() && !channel.timestamps.is_empty() {
            return sample_vec3(&channel.timestamps, &channel.translations, time);
        }
    }
    skeleton.joints[0].rest_translation
}

/// `dst = lerp(dst, src, w * mask[j])`. Rotations use nlerp with a
/// hemisphere fix — without the dot-sign flip, two clips whose quaternions
/// land on opposite hemispheres blend the *long* way round and the limb
/// visibly swings through the body.
fn blend_pose(dst: &mut LocalPose, src: &LocalPose, w: f32, mask: Option<&[f32]>) {
    let n = dst.0.len().min(src.0.len());
    for j in 0..n {
        let jw = match mask {
            Some(m) => w * m.get(j).copied().unwrap_or(0.0),
            None => w,
        };
        if jw <= 0.0 { continue; }
        let jw = jw.min(1.0);
        for k in 0..3 {
            dst.0[j][k] = dst.0[j][k] + (src.0[j][k] - dst.0[j][k]) * jw;
            dst.2[j][k] = dst.2[j][k] + (src.2[j][k] - dst.2[j][k]) * jw;
        }
        let a = dst.1[j];
        let mut b = src.1[j];
        let dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
        if dot < 0.0 { b = [-b[0], -b[1], -b[2], -b[3]]; }
        let mut q = [
            a[0] + (b[0] - a[0]) * jw,
            a[1] + (b[1] - a[1]) * jw,
            a[2] + (b[2] - a[2]) * jw,
            a[3] + (b[3] - a[3]) * jw,
        ];
        let len = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
        if len > 1e-6 {
            q = [q[0]/len, q[1]/len, q[2]/len, q[3]/len];
        } else {
            q = a;
        }
        dst.1[j] = q;
    }
}

/// 1.0 for every joint at or below `root`, 0.0 elsewhere. `root < 0` means
/// "whole skeleton" so a layer with no mask is a plain full-body override.
fn build_mask_weights(skeleton: &SkeletonData, root: i32) -> Vec<f32> {
    let n = skeleton.joints.len();
    if root < 0 || (root as usize) >= n { return vec![1.0; n]; }
    let mut w = vec![0.0f32; n];
    let mut stack = vec![root as usize];
    while let Some(j) = stack.pop() {
        if j >= n || w[j] > 0.0 { continue; }
        w[j] = 1.0;
        for &c in &skeleton.joints[j].children { stack.push(c); }
    }
    w
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
