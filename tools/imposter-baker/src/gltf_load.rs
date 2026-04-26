//! Minimal GLB loader for the imposter baker. V1: one mesh, all
//! primitives concatenated, positions + normals + indices only. No
//! textures, no skinning, no node transforms beyond identity.

use std::path::Path;

#[derive(Debug)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals:   Vec<[f32; 3]>,
    pub indices:   Vec<u32>,
    pub aabb_min:  [f32; 3],
    pub aabb_max:  [f32; 3],
}

impl MeshData {
    pub fn center(&self) -> [f32; 3] {
        [
            0.5 * (self.aabb_min[0] + self.aabb_max[0]),
            0.5 * (self.aabb_min[1] + self.aabb_max[1]),
            0.5 * (self.aabb_min[2] + self.aabb_max[2]),
        ]
    }

    pub fn radius(&self) -> f32 {
        let c = self.center();
        let dx = self.aabb_max[0] - c[0];
        let dy = self.aabb_max[1] - c[1];
        let dz = self.aabb_max[2] - c[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

pub fn load_glb<P: AsRef<Path>>(path: P) -> Result<MeshData, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("read {}: {e}", path.as_ref().display()))?;
    let g = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("parse gltf: {e}"))?;

    // GLB: a single embedded BIN buffer.
    let bin = g
        .blob
        .as_ref()
        .ok_or("expected GLB with embedded BIN buffer")?;

    let mut positions = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut indices = Vec::<u32>::new();
    let mut aabb_min = [f32::INFINITY; 3];
    let mut aabb_max = [f32::NEG_INFINITY; 3];

    for mesh in g.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| {
                if buf.index() == 0 {
                    Some(bin.as_slice())
                } else {
                    None
                }
            });

            let base = positions.len() as u32;

            let pos_iter = reader.read_positions().ok_or("primitive missing POSITION")?;
            for p in pos_iter {
                aabb_min[0] = aabb_min[0].min(p[0]);
                aabb_min[1] = aabb_min[1].min(p[1]);
                aabb_min[2] = aabb_min[2].min(p[2]);
                aabb_max[0] = aabb_max[0].max(p[0]);
                aabb_max[1] = aabb_max[1].max(p[1]);
                aabb_max[2] = aabb_max[2].max(p[2]);
                positions.push(p);
            }

            // Normals: optional in spec; if missing, fill with +Y so the
            // shader still gets something sensible (lambert will look
            // flat but not crash).
            if let Some(n_iter) = reader.read_normals() {
                normals.extend(n_iter);
            } else {
                normals.resize(positions.len(), [0.0, 1.0, 0.0]);
            }
            // Make sure normals length matches positions length even
            // when the primitive supplies fewer than expected.
            normals.resize(positions.len(), [0.0, 1.0, 0.0]);

            if let Some(idx_iter) = reader.read_indices() {
                for i in idx_iter.into_u32() {
                    indices.push(base + i);
                }
            } else {
                // Non-indexed: emit sequential indices for this prim.
                let count = (positions.len() as u32) - base;
                indices.extend(base..base + count);
            }
        }
    }

    if positions.is_empty() {
        return Err("no geometry in GLB".to_string());
    }

    Ok(MeshData { positions, normals, indices, aabb_min, aabb_max })
}
