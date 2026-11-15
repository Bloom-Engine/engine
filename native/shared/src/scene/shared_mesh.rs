//! What the scene graph remembers about an imported model primitive that many
//! nodes share: its geometry key, and the GPU buffers it has when used as a
//! reduced-detail level.
//!
//! Placing one model mesh on a thousand nodes used to hash its vertices a
//! thousand times (twice each) and, for level-of-detail meshes, copy and
//! upload them a thousand times. Both are properties of the mesh, so they are
//! computed once per mesh here.

use super::*;
use std::sync::Weak;

struct SharedMeshEntry {
    /// Guards the address key: an entry whose mesh has been dropped is stale
    /// even if a new mesh was allocated at the same address.
    mesh: Weak<MeshData>,
    key: Option<SceneGeometryKey>,
    lod_buffers: Option<(wgpu::Buffer, wgpu::Buffer)>,
}

#[derive(Default)]
pub(super) struct SharedMeshCache {
    entries: HashMap<usize, SharedMeshEntry>,
    prune_at: usize,
}

impl SharedMeshCache {
    fn entry(&mut self, mesh: &Arc<MeshData>) -> &mut SharedMeshEntry {
        if self.entries.len() >= self.prune_at.max(256) {
            self.entries
                .retain(|_, entry| entry.mesh.strong_count() > 0);
            self.prune_at = self.entries.len() * 2;
        }
        let entry = self
            .entries
            .entry(Arc::as_ptr(mesh) as usize)
            .or_insert_with(|| SharedMeshEntry {
                mesh: Arc::downgrade(mesh),
                key: None,
                lod_buffers: None,
            });
        let same = entry
            .mesh
            .upgrade()
            .is_some_and(|live| Arc::ptr_eq(&live, mesh));
        if !same {
            *entry = SharedMeshEntry {
                mesh: Arc::downgrade(mesh),
                key: None,
                lod_buffers: None,
            };
        }
        entry
    }

    /// The geometry key of a shared mesh, hashed on first use.
    pub(super) fn key(&mut self, mesh: &Arc<MeshData>) -> SceneGeometryKey {
        let entry = self.entry(mesh);
        *entry
            .key
            .get_or_insert_with(|| scene_geometry_key(&mesh.vertices, &mesh.indices))
    }

    /// Vertex and index buffers of a shared mesh used as a reduced-detail
    /// level, created on first use and shared by every node after that.
    pub(super) fn lod_buffers(
        &mut self,
        device: &wgpu::Device,
        mesh: &Arc<MeshData>,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let entry = self.entry(mesh);
        entry
            .lod_buffers
            .get_or_insert_with(|| {
                (
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("scene_shared_lod_vb"),
                        contents: bytemuck::cast_slice(&mesh.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("scene_shared_lod_ib"),
                        contents: bytemuck::cast_slice(&mesh.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                )
            })
            .clone()
    }
}

/// The geometry key of a node: remembered per mesh when the node shows a
/// shared model primitive, hashed directly for geometry the node owns.
pub(super) fn node_geometry_key(node: &SceneNode, cache: &mut SharedMeshCache) -> SceneGeometryKey {
    match node.shared_model_mesh.as_ref() {
        Some(mesh) => cache.key(mesh),
        None => scene_geometry_key(node.vertices(), node.indices()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(x: f32) -> Arc<MeshData> {
        let mut vertex = Vertex3D::default();
        vertex.position = [x, 0.0, 0.0];
        Arc::new(MeshData {
            vertices: vec![vertex; 3],
            secondary_tex_coords: None,
            indices: vec![0, 1, 2],
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3],
            alpha_mode: crate::models::MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission: crate::models::MaterialTransmission::default(),
            layered_pbr: crate::models::MaterialLayeredPbr::default(),
        })
    }

    #[test]
    fn shared_mesh_key_matches_the_direct_hash_and_is_remembered() {
        let mut cache = SharedMeshCache::default();
        let a = mesh(1.0);
        let key = cache.key(&a);
        assert_eq!(key, scene_geometry_key(&a.vertices, &a.indices));
        assert_eq!(cache.key(&a), key);
        assert_eq!(cache.entries.len(), 1);
        assert_ne!(cache.key(&mesh(2.0)), key);
    }

    #[test]
    fn a_dropped_mesh_does_not_lend_its_key_to_a_new_one() {
        let mut cache = SharedMeshCache::default();
        let a = mesh(1.0);
        let address = Arc::as_ptr(&a) as usize;
        let key_a = cache.key(&a);
        drop(a);
        // Put a different mesh under the dead entry's address, as the
        // allocator may do, and check the stale key is not returned.
        let b = mesh(2.0);
        let stale = cache.entries.remove(&address).unwrap();
        cache.entries.insert(Arc::as_ptr(&b) as usize, stale);
        let key_b = cache.key(&b);
        assert_ne!(key_b, key_a);
        assert_eq!(key_b, scene_geometry_key(&b.vertices, &b.indices));
    }
}
