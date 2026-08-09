//! Scene graph resource-key, culling, matrix, and invariant helpers.
//!
//! Kept as a child module so it can operate on the graph's private cache
//! records without widening their public API.

use super::*;

pub(super) fn scene_node_gpu_driven_ready(
    node: &SceneNode,
    imported_refraction_enabled: bool,
) -> bool {
    node.active_lod < 0
        // Retained MASK/transparent nodes historically blend in submission
        // order while depth-writing in the same pass. A depth prepass would
        // collapse overlapping translucent layers to the nearest surface,
        // changing foliage colour even when its cutout silhouette matched.
        // Keep those nodes on the compatibility path until they have a
        // dedicated order-independent transparency bucket.
        && node.material.alpha_cutoff <= 0.0
        && node.material.alpha_mode != MaterialAlphaMode::Blend
        && !(imported_refraction_enabled && node.material.transmission.is_active())
        && !node.material.layered_pbr.is_active()
        && (!node.material.opacity.is_finite() || node.material.opacity >= 1.0)
        && node.gpu_geometry.is_some()
        && node.gpu_material_id != MaterialId::FALLBACK
        && node.uniform_slot.is_some()
}

pub(super) fn retained_order_is_gpu_safe(
    nodes: &HandleRegistry<SceneNode>,
    allow_cutout_compatibility: bool,
) -> bool {
    !nodes.iter().any(|(_, node)| {
        node.visible
            && !node.gi_only
            && !node.indices().is_empty()
            && ((node.material.alpha_cutoff > 0.0 && !allow_cutout_compatibility)
                || (node.material.opacity.is_finite() && node.material.opacity < 1.0))
    })
}

pub(super) fn scene_geometry_key(vertices: &[Vertex3D], indices: &[u32]) -> SceneGeometryKey {
    const FNV_PRIME: u64 = 0x100000001b3;
    fn hash(mut value: u64, bytes: &[u8]) -> u64 {
        for &byte in bytes {
            value ^= byte as u64;
            value = value.wrapping_mul(FNV_PRIME);
        }
        value
    }
    let vertex_bytes = bytemuck::cast_slice(vertices);
    let index_bytes = bytemuck::cast_slice(indices);
    SceneGeometryKey {
        hash_a: hash(hash(0xcbf29ce484222325, vertex_bytes), index_bytes),
        hash_b: hash(hash(0x84222325cbf29ce4, index_bytes), vertex_bytes),
        vertex_count: vertices.len() as u32,
        index_count: indices.len() as u32,
    }
}

pub(super) fn scene_material_key(material: &PbrMaterial) -> SceneMaterialKey {
    SceneMaterialKey {
        metal_rough: [
            material.metalness.to_bits(),
            material.roughness.to_bits(),
            if material.specular_glossiness_factor.is_some() {
                2
            } else {
                (material.metallic_roughness_texture_idx != 0) as u32
            },
            material
                .alpha_mode
                .shader_alpha_value(material.alpha_cutoff)
                .to_bits(),
        ],
        emissive: [
            material.emissive[0].to_bits(),
            material.emissive[1].to_bits(),
            material.emissive[2].to_bits(),
            u32::from(material.alpha_coverage_mips),
        ],
        spec_gloss: material
            .specular_glossiness_factor
            .unwrap_or([1.0; 4])
            .map(f32::to_bits),
        textures: [
            material.texture_idx,
            material.normal_texture_idx,
            material.metallic_roughness_texture_idx,
            material.emissive_texture_idx,
            material.occlusion_texture_idx,
        ],
    }
}

pub(super) fn release_scene_geometry(
    cache: &mut HashMap<SceneGeometryKey, SharedSceneGeometry>,
    retired: &mut Vec<GeometrySlice>,
    key: SceneGeometryKey,
) {
    let Some(entry) = cache.get_mut(&key) else {
        return;
    };
    if entry.references > 1 {
        entry.references -= 1;
        return;
    }
    if let Some(entry) = cache.remove(&key) {
        retired.push(entry.slice);
    }
}

pub(super) fn release_scene_gpu_resources(
    cache: &mut HashMap<SceneGeometryKey, SharedSceneGpuResources>,
    key: SceneGeometryKey,
) {
    let Some(entry) = cache.get_mut(&key) else {
        return;
    };
    if entry.references > 1 {
        entry.references -= 1;
    } else {
        cache.remove(&key);
    }
}

pub(super) fn release_scene_material(
    cache: &mut HashMap<SceneMaterialKey, SharedSceneMaterial>,
    retired: &mut Vec<MaterialId>,
    key: SceneMaterialKey,
) {
    let Some(entry) = cache.get_mut(&key) else {
        return;
    };
    if entry.references > 1 {
        entry.references -= 1;
        return;
    }
    if let Some(entry) = cache.remove(&key) {
        retired.push(entry.id);
    }
}

// ============================================================
// Matrix math (4x4, column-major)
// ============================================================

// ============================================================
// Frustum culling
// ============================================================
// Gribb-Hartmann plane extraction: for a column-major clip matrix M,
// each plane = ±row_i + row_3. We build 6 planes (left/right/bottom/
// top/near/far) in world space directly from the VP matrix, so every
// plane-test below is a world-space dot product.
//
// A node's world-space AABB is outside the frustum if ALL 8 of its
// corners are on the negative side of ANY single plane. The standard
// "positive-vertex-only" optimization is skipped here — testing 8
// corners is still a few dozen multiplies per node, trivial compared
// to the per-node GPU cost we skip on a cull hit.
//
// Plane format: [nx, ny, nz, d] where `nx*x + ny*y + nz*z + d >= 0`
// means the point is inside that plane's half-space. No normalization
// — we only care about the sign.

/// Longest NDC-extent of a world AABB under `vp` — the "screen coverage"
/// that drives LOD selection (1.0 = spans the full viewport). Corners at
/// or behind the near plane return 1.0 (force the finest level).
pub(super) fn aabb_screen_coverage(vp: &[[f32; 4]; 4], wmin: [f32; 3], wmax: [f32; 3]) -> f32 {
    let mut lo = [f32::MAX, f32::MAX];
    let mut hi = [f32::MIN, f32::MIN];
    for ix in 0..2 {
        for iy in 0..2 {
            for iz in 0..2 {
                let x = if ix == 0 { wmin[0] } else { wmax[0] };
                let y = if iy == 0 { wmin[1] } else { wmax[1] };
                let z = if iz == 0 { wmin[2] } else { wmax[2] };
                let cw = vp[0][3] * x + vp[1][3] * y + vp[2][3] * z + vp[3][3];
                if cw <= 1e-3 {
                    return 1.0;
                }
                let cx = (vp[0][0] * x + vp[1][0] * y + vp[2][0] * z + vp[3][0]) / cw;
                let cy = (vp[0][1] * x + vp[1][1] * y + vp[2][1] * z + vp[3][1]) / cw;
                lo[0] = lo[0].min(cx);
                lo[1] = lo[1].min(cy);
                hi[0] = hi[0].max(cx);
                hi[1] = hi[1].max(cy);
            }
        }
    }
    // NDC spans -1..1, so extent/2 = fraction of the viewport.
    (((hi[0] - lo[0]).max(hi[1] - lo[1])) * 0.5).clamp(0.0, 1.0)
}

pub(crate) fn extract_frustum_planes(vp: &[[f32; 4]; 4]) -> [[f32; 4]; 6] {
    // Row vectors of the column-major matrix: row_i[col] = vp[col][i].
    let row = |i: usize| [vp[0][i], vp[1][i], vp[2][i], vp[3][i]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    [
        add(r3, r0), // left
        sub(r3, r0), // right
        add(r3, r1), // bottom
        sub(r3, r1), // top
        r2,          // near (wgpu uses 0..1 depth → near = row_2)
        sub(r3, r2), // far
    ]
}

pub(crate) fn aabb_outside_frustum(planes: &[[f32; 4]; 6], bmin: [f32; 3], bmax: [f32; 3]) -> bool {
    for p in planes.iter() {
        let mut all_outside = true;
        for ix in 0..2 {
            let x = if ix == 0 { bmin[0] } else { bmax[0] };
            for iy in 0..2 {
                let y = if iy == 0 { bmin[1] } else { bmax[1] };
                for iz in 0..2 {
                    let z = if iz == 0 { bmin[2] } else { bmax[2] };
                    if p[0] * x + p[1] * y + p[2] * z + p[3] >= 0.0 {
                        all_outside = false;
                        break;
                    }
                }
                if !all_outside {
                    break;
                }
            }
            if !all_outside {
                break;
            }
        }
        if all_outside {
            return true;
        }
    }
    false
}

pub(super) fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    result
}

#[cfg(test)]
#[path = "../scene_visibility_routing_tests.rs"]
mod visibility_routing_tests;

#[cfg(test)]
mod gpu_driven_cache_tests {
    use super::*;

    fn test_slice() -> GeometrySlice {
        GeometrySlice {
            vertex_offset: 96,
            vertex_size: 192,
            index_offset: 24,
            index_size: 48,
            first_index: 6,
            base_vertex: 1,
        }
    }

    #[test]
    fn geometry_keys_are_content_addressed() {
        let mut vertices = vec![Vertex3D::default()];
        let first = scene_geometry_key(&vertices, &[0, 1, 2]);
        assert_eq!(first, scene_geometry_key(&vertices, &[0, 1, 2]));
        vertices[0].position[0] = 1.0;
        assert_ne!(first, scene_geometry_key(&vertices, &[0, 1, 2]));
        assert_ne!(
            first,
            scene_geometry_key(&[Vertex3D::default()], &[0, 2, 1])
        );
    }

    #[test]
    fn material_key_ignores_per_draw_tint_but_tracks_pbr_state() {
        let first = PbrMaterial::default();
        let mut tint_only = first.clone();
        tint_only.color = [0.1, 0.2, 0.3];
        tint_only.opacity = 0.25;
        assert_eq!(scene_material_key(&first), scene_material_key(&tint_only));

        let mut changed = first.clone();
        changed.roughness = 0.35;
        assert_ne!(scene_material_key(&first), scene_material_key(&changed));

        let mut spec_gloss = first.clone();
        spec_gloss.specular_glossiness_factor = Some([0.2, 0.4, 0.8, 0.65]);
        spec_gloss.metallic_roughness_texture_idx = 7;
        assert_ne!(scene_material_key(&first), scene_material_key(&spec_gloss));

        let mut other_spec_gloss = spec_gloss.clone();
        other_spec_gloss.specular_glossiness_factor = Some([0.2, 0.4, 0.7, 0.65]);
        assert_ne!(
            scene_material_key(&spec_gloss),
            scene_material_key(&other_spec_gloss)
        );
    }

    #[test]
    fn physical_metadata_round_trips_and_dirties_only_on_change() {
        let mut scene = SceneGraph::new();
        let node = scene.create_node();
        scene.nodes.get_mut(node).unwrap().mat_dirty = false;
        let initial_tlas_version = scene.tlas_version;
        let transmission = MaterialTransmission {
            authored: true,
            factor: 0.8,
            ior_authored: true,
            ior: 1.45,
            volume_authored: true,
            thickness_factor: 0.25,
            attenuation_distance: 2.0,
            attenuation_color: [0.8, 0.9, 1.0],
            thickness_source: crate::models::MaterialThicknessSource::Authored,
            ..Default::default()
        };
        scene.set_material_transmission(node, transmission);
        let retained = scene.nodes.get(node).unwrap();
        assert_eq!(retained.material.transmission, transmission);
        assert!(
            retained.mat_dirty,
            "the physical material bind group must rebuild when transmission changes"
        );
        assert_ne!(
            scene.tlas_version, initial_tlas_version,
            "visible transmission changes must invalidate GI transport and TLAS masks"
        );
        let changed_tlas_version = scene.tlas_version;
        scene.nodes.get_mut(node).unwrap().mat_dirty = false;
        scene.set_material_transmission(node, transmission);
        assert!(
            !scene.nodes.get(node).unwrap().mat_dirty,
            "setting identical physical metadata must remain allocation-free"
        );
        assert_eq!(
            scene.tlas_version, changed_tlas_version,
            "identical physical metadata must not trigger a redundant GI rebuild"
        );

        scene.set_material_pbr(node, 0.4, 1.0);
        assert_ne!(
            scene.tlas_version, changed_tlas_version,
            "metallic suppression must invalidate transparent-GI membership"
        );
        let metallic_tlas_version = scene.tlas_version;
        scene.nodes.get_mut(node).unwrap().mat_dirty = false;
        scene.set_material_pbr(node, 0.4, 1.0);
        assert_eq!(scene.tlas_version, metallic_tlas_version);
        assert!(
            !scene.nodes.get(node).unwrap().mat_dirty,
            "identical PBR factors must remain allocation-free"
        );
        scene.set_material_emissive_factor(node, 0.0, 0.0, 0.0);
        scene.set_material_texture(node, 0);
        assert!(
            !scene.nodes.get(node).unwrap().mat_dirty,
            "default descriptor fields must not rebuild an unchanged material"
        );
    }

    #[test]
    fn shared_geometry_retires_only_after_last_reference() {
        let key = SceneGeometryKey {
            hash_a: 1,
            hash_b: 2,
            vertex_count: 3,
            index_count: 3,
        };
        let mut cache = HashMap::from([(
            key,
            SharedSceneGeometry {
                slice: test_slice(),
                references: 2,
            },
        )]);
        let mut retired = Vec::new();
        release_scene_geometry(&mut cache, &mut retired, key);
        assert_eq!(cache[&key].references, 1);
        assert!(retired.is_empty());
        release_scene_geometry(&mut cache, &mut retired, key);
        assert!(!cache.contains_key(&key));
        assert_eq!(retired, vec![test_slice()]);
    }
}
