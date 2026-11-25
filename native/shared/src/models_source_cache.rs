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

/// A fast fingerprint of the file's bytes (four independent 64-bit lanes,
/// several GB/s). It only has to notice a file that changed on disk, not
/// resist an adversary. SipHash (`DefaultHasher`) ran at about 1.5 GB/s, so
/// it cost about 60 ms per 100 MB model, on every load and every repeat.
fn fingerprint(file_data: &[u8]) -> u64 {
    const K: [u64; 4] = [
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
        0x85eb_ca77_c2b2_ae63,
    ];
    let mut lane = [K[1], K[2], K[3], K[0]];
    let mut chunks = file_data.chunks_exact(32);
    for c in &mut chunks {
        for (i, l) in lane.iter_mut().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&c[i * 8..i * 8 + 8]);
            *l = (*l ^ u64::from_le_bytes(w))
                .wrapping_mul(K[i])
                .rotate_left(31);
        }
    }
    let mut h = file_data.len() as u64 ^ K[0];
    for (i, b) in chunks.remainder().iter().enumerate() {
        h = (h ^ (u64::from(*b) << ((i % 8) * 8)))
            .wrapping_mul(K[1])
            .rotate_left(27);
    }
    for (i, l) in lane.iter().enumerate() {
        h = (h ^ *l).wrapping_mul(K[i]).rotate_left(29);
    }
    h ^ (h >> 32)
}

/// What tells a changed file from the one loaded before: its size, time of
/// change and (on Unix) inode and device when the file can be asked, else the
/// bytes. Hashing the bytes read every page of a mapped file, most of them
/// texture levels the load would otherwise never touch.
fn file_identity(source_path: &Path, file_data: &[u8]) -> u64 {
    let Ok(meta) = std::fs::metadata(source_path) else {
        return fingerprint(file_data);
    };
    if meta.len() != file_data.len() as u64 {
        return fingerprint(file_data);
    }
    let changed = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    #[cfg(unix)]
    let (ino, dev) = {
        use std::os::unix::fs::MetadataExt;
        (meta.ino(), meta.dev())
    };
    #[cfg(not(unix))]
    let (ino, dev) = (0u64, 0u64);
    let mut id = Vec::with_capacity(32);
    for v in [meta.len(), changed, ino, dev] {
        id.extend_from_slice(&v.to_le_bytes());
    }
    fingerprint(&id)
}

impl SourceModelCache {
    /// A model sharing the earlier import of `source_path`, if that file was
    /// loaded before with the same bytes and its meshes are still alive.
    pub fn share(&mut self, source_path: &Path, file_data: &[u8]) -> Option<ModelData> {
        let key = source_path.display().to_string();
        let entry = self.entries.get(&key)?;
        if entry.byte_len != file_data.len()
            || entry.fingerprint != file_identity(source_path, file_data)
        {
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
                fingerprint: file_identity(source_path, file_data),
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
    fn a_models_textures_are_its_own_only_while_no_other_model_shares_its_primitives() {
        let mut models = ModelManager::new();
        let handle = models.gen_mesh_cube(1.0, 1.0, 1.0);
        let mut mesh = (*models.get(handle).unwrap().meshes[0]).clone();
        mesh.texture_idx = Some(7);
        mesh.normal_texture_idx = Some(8);
        mesh.occlusion_texture_idx = Some(8);
        mesh.emissive_texture_idx = Some(0);
        let mesh = Arc::new(mesh);
        // the same primitive placed twice, as repeated glTF nodes do
        let meshes = vec![mesh.clone(), mesh];
        assert_eq!(
            crate::models::exclusive_texture_indices(&meshes),
            vec![7, 8]
        );
        // a second load of the file shares the primitive: nothing may be freed
        let other_model = meshes[0].clone();
        assert!(crate::models::exclusive_texture_indices(&meshes).is_empty());
        drop(other_model);
        assert_eq!(
            crate::models::exclusive_texture_indices(&meshes),
            vec![7, 8]
        );
    }

    #[test]
    fn the_fingerprint_notices_any_changed_byte() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        let base = fingerprint(&data);
        assert_eq!(base, fingerprint(&data.clone()));
        for i in [0, 31, 32, 500, 991, 999] {
            let mut d = data.clone();
            d[i] ^= 1;
            assert_ne!(base, fingerprint(&d), "byte {i}");
        }
        assert_ne!(fingerprint(&data[..999]), fingerprint(&data[..998]));
        assert_ne!(fingerprint(&[0u8; 64]), fingerprint(&[0u8; 65]));
    }

    #[test]
    fn a_file_on_disk_is_known_by_its_size_and_time_of_change() {
        let mut models = ModelManager::new();
        let handle = models.gen_mesh_cube(1.0, 1.0, 1.0);
        let dir = std::env::temp_dir().join(format!("bloom-source-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("actor.glb");
        std::fs::write(&path, b"file bytes").unwrap();
        let mut cache = SourceModelCache::default();
        cache.remember(&path, b"file bytes", models.get(handle).unwrap());
        assert!(
            cache.share(&path, b"file bytes").is_some(),
            "same file shares"
        );
        // the same length, rewritten later: a new import
        std::fs::write(&path, b"edited byt").unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert!(
            cache.share(&path, b"edited byt").is_none(),
            "changed file reimports"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

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
