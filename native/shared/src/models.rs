use crate::handles::HandleRegistry;
use crate::renderer::Vertex3D;

pub struct MeshData {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
}

pub struct ModelData {
    pub meshes: Vec<MeshData>,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
}

pub struct ModelManager {
    pub models: HandleRegistry<ModelData>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self { models: HandleRegistry::new() }
    }

    pub fn load_model(&mut self, file_data: &[u8]) -> f64 {
        match load_gltf(file_data) {
            Some(model) => self.models.alloc(model),
            None => 0.0,
        }
    }

    pub fn get(&self, handle: f64) -> Option<&ModelData> {
        self.models.get(handle)
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
        }).collect();

        let mut indices = Vec::with_capacity(36);
        for face in 0..6u32 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let model = ModelData {
            meshes: vec![MeshData { vertices, indices }],
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
            meshes: vec![MeshData { vertices, indices }],
            bbox_min: [-size_x * 0.5, 0.0, -size_z * 0.5],
            bbox_max: [size_x * 0.5, size_y, size_z * 0.5],
        };
        self.models.alloc(model)
    }
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
                vertices.push(Vertex3D {
                    position: p,
                    normal: normals[i],
                    color,
                    uv: tex_coords[i],
                });
            }

            let indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            meshes.push(MeshData { vertices, indices });
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
