use crate::handles::HandleRegistry;
use crate::renderer::Vertex3D;

pub struct MeshData {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    pub texture_idx: Option<u32>,
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
    pub rotations: Vec<[f32; 4]>,
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
        match load_gltf_with_textures(file_data, renderer) {
            Some(model) => self.models.alloc(model),
            None => 0.0,
        }
    }

    pub fn get(&self, handle: f64) -> Option<&ModelData> {
        self.models.get(handle)
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
        }).collect();

        let mut indices = Vec::with_capacity(36);
        for face in 0..6u32 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let model = ModelData {
            meshes: vec![MeshData { vertices, indices, texture_idx: None }],
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
            meshes: vec![MeshData { vertices, indices, texture_idx: None }],
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
            });
        }

        let indices = index_data.to_vec();
        let model = ModelData {
            meshes: vec![MeshData { vertices, indices, texture_idx: None }],
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
                if !channel.rotations.is_empty() && !channel.timestamps.is_empty() {
                    local_rotations[ji] = sample_quat(&channel.timestamps, &channel.rotations, t);
                }
                if !channel.scales.is_empty() && !channel.timestamps.is_empty() {
                    local_scales[ji] = sample_vec3(&channel.timestamps, &channel.scales, t);
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

fn quat_mul(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    // Hamilton product: a * b where q = [x, y, z, w]
    [
        a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
        a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
        a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
        a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
    ]
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

fn read_accessor_f32(gltf: &gltf::Gltf, buffer_data: &[Vec<u8>], accessor: &gltf::Accessor) -> Vec<f32> {
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

        // Group channels by target node
        let mut node_channels: std::collections::HashMap<usize, (Vec<f32>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)> = std::collections::HashMap::new();

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

            let entry = node_channels.entry(joint_index).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new(), Vec::new()));

            match channel.target().property() {
                gltf::animation::Property::Translation => {
                    entry.0 = timestamps;
                    entry.1 = values.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                }
                gltf::animation::Property::Rotation => {
                    if entry.0.is_empty() { entry.0 = timestamps; }
                    entry.2 = values.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
                }
                gltf::animation::Property::Scale => {
                    if entry.0.is_empty() { entry.0 = timestamps; }
                    entry.3 = values.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                }
                _ => {}
            }
        }

        for (joint_index, (timestamps, translations, rotations, scales)) in node_channels {
            channels.push(AnimationChannel {
                joint_index,
                timestamps,
                translations,
                rotations,
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

fn load_gltf_with_textures(data: &[u8], renderer: &mut crate::renderer::Renderer) -> Option<ModelData> {
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
                } else {
                    buffer_data.push(Vec::new());
                }
            }
        }
    }

    // Extract and register textures
    let mut texture_indices: Vec<u32> = Vec::new(); // maps glTF image index -> renderer texture index
    for image in gltf.images() {
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
                            let tex_idx = renderer.register_texture(w, h, &rgba);
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
            _ => { texture_indices.push(0); }
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

            // Get vertex colors if available
            let vert_colors: Option<Vec<[f32; 4]>> = reader.read_colors(0)
                .map(|iter| iter.into_rgba_f32().collect());

            let base_color = primitive.material().pbr_metallic_roughness().base_color_factor();

            // Determine texture index for this mesh
            let tex_idx = primitive.material().pbr_metallic_roughness()
                .base_color_texture()
                .and_then(|info| {
                    let img_idx = info.texture().source().index();
                    texture_indices.get(img_idx).copied()
                });

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
                });
            }
            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            meshes.push(MeshData { vertices, indices, texture_idx: tex_idx });
        }
    }

    if meshes.is_empty() { return None; }
    Some(ModelData { meshes, bbox_min, bbox_max })
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
                });
            }

            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            meshes.push(MeshData { vertices, indices, texture_idx: None });
        }
    }

    if meshes.is_empty() { return None; }
    Some(ModelData { meshes, bbox_min, bbox_max })
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
