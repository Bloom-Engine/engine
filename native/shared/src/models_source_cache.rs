//! Loading the same model file twice shares one import.
//!
//! `loadModel(path)` used to parse the file, decode every image and register
//! a second set of GPU textures on each call, so a game that loads one
//! character for two actors paid for its textures twice (66 MB for a model
//! with three 2048x2048 maps). A repeated load of an unchanged file now
//! returns a new model handle over the first import's meshes: the geometry
//! is `Arc`-shared and the materials point at the textures already
//! registered.
//!
//! The cache holds the meshes weakly. Once every handle of a file has been
//! unloaded, the next load imports it again.

use super::{MeshData, ModelData, ModelPrimitiveSource};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Weak};

struct Entry {
    byte_len: usize,
    fingerprint: u64,
    meshes: Vec<Weak<MeshData>>,
    mesh_transforms: Vec<[[f32; 4]; 4]>,
    mesh_cast_shadows: Vec<bool>,
    mesh_sources: Vec<Option<ModelPrimitiveSource>>,
    source_geometry_sha256: Option<[u8; 32]>,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
}

#[derive(Default)]
pub struct SourceModelCache {
    entries: HashMap<String, Entry>,
}

fn fingerprint(file_data: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    file_data.hash(&mut hasher);
    hasher.finish()
}

impl SourceModelCache {
    /// A model sharing the earlier import of `source_path`, if that file was
    /// loaded before with the same bytes and its meshes are still alive.
    pub fn share(&mut self, source_path: &Path, file_data: &[u8]) -> Option<ModelData> {
        let key = source_path.display().to_string();
        let entry = self.entries.get(&key)?;
        if entry.byte_len != file_data.len() || entry.fingerprint != fingerprint(file_data) {
            return None;
        }
        let meshes: Option<Vec<Arc<MeshData>>> = entry.meshes.iter().map(Weak::upgrade).collect();
        let Some(meshes) = meshes else {
            self.entries.remove(&key);
            return None;
        };
        Some(ModelData {
            meshes,
            mesh_transforms: entry.mesh_transforms.clone(),
            mesh_cast_shadows: entry.mesh_cast_shadows.clone(),
            mesh_sources: entry.mesh_sources.clone(),
            source_geometry_sha256: entry.source_geometry_sha256,
            bbox_min: entry.bbox_min,
            bbox_max: entry.bbox_max,
        })
    }

    pub fn remember(&mut self, source_path: &Path, file_data: &[u8], model: &ModelData) {
        self.entries.insert(
            source_path.display().to_string(),
            Entry {
                byte_len: file_data.len(),
                fingerprint: fingerprint(file_data),
                meshes: model.meshes.iter().map(Arc::downgrade).collect(),
                mesh_transforms: model.mesh_transforms.clone(),
                mesh_cast_shadows: model.mesh_cast_shadows.clone(),
                mesh_sources: model.mesh_sources.clone(),
                source_geometry_sha256: model.source_geometry_sha256,
                bbox_min: model.bbox_min,
                bbox_max: model.bbox_max,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelManager;

    #[test]
    fn repeated_load_shares_meshes_until_the_file_changes_or_is_unloaded() {
        let mut models = ModelManager::new();
        let handle = models.gen_mesh_cube(1.0, 1.0, 1.0);
        let path = Path::new("assets/actor.glb");
        let mut cache = SourceModelCache::default();
        cache.remember(path, b"file bytes", models.get(handle).unwrap());

        let shared = cache.share(path, b"file bytes").expect("same file shares");
        let original = models.get(handle).unwrap();
        assert_eq!(shared.meshes.len(), original.meshes.len());
        assert!(Arc::ptr_eq(&shared.meshes[0], &original.meshes[0]));
        assert_eq!(shared.bbox_max, original.bbox_max);

        assert!(
            cache.share(path, b"edited byt").is_none(),
            "changed bytes reimport"
        );
        assert!(cache
            .share(Path::new("assets/other.glb"), b"file bytes")
            .is_none());

        drop(shared);
        models.unload_model(handle);
        assert!(
            cache.share(path, b"file bytes").is_none(),
            "unloaded model reimports"
        );
    }
}
