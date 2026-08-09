use super::transform::{
    mat3_transform_vec, mat4_inverse_transpose_3x3, mat4_mean_scale, mat4_transform_direction,
    mat4_transform_point,
};
use super::{walk_scene_collect_instances, MeshData, Vertex3D};
use std::sync::Arc;

fn vertex_is_skinned(vertex: &Vertex3D) -> bool {
    vertex.weights.iter().sum::<f32>() > 0.01
}

fn mesh_is_skinned(mesh: &MeshData) -> bool {
    mesh.vertices.iter().any(vertex_is_skinned)
}

pub(super) fn shared_or_owned_instance(
    primitive: &Arc<MeshData>,
    world: [[f32; 4]; 4],
) -> (Arc<MeshData>, [[f32; 4]; 4]) {
    if mesh_is_skinned(primitive) {
        (
            Arc::new(bake_owned_mesh_instance(primitive.as_ref().clone(), &world)),
            crate::renderer::IDENTITY_MAT4,
        )
    } else {
        (Arc::clone(primitive), world)
    }
}

/// Bake one static glTF node transform into an owned compatibility instance.
///
/// This path is deliberately retained only for skinned/mixed primitives: the
/// current animation contract already places weighted vertices through the
/// joint palette while rigid vertices in the same primitive still need their
/// authored node transform. Ordinary static primitives never call this and
/// remain immutable/shared.
fn bake_owned_mesh_instance(mut instance: MeshData, world: &[[f32; 4]; 4]) -> MeshData {
    let normal_transform = mat4_inverse_transpose_3x3(world);
    let has_skinning = mesh_is_skinned(&instance);
    for vertex in &mut instance.vertices {
        if vertex_is_skinned(vertex) {
            continue;
        }
        vertex.position = mat4_transform_point(world, &vertex.position);
        vertex.normal = mat3_transform_vec(&normal_transform, &vertex.normal);
        let tangent = mat4_transform_direction(
            world,
            &[vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]],
        );
        vertex.tangent[0] = tangent[0];
        vertex.tangent[1] = tangent[1];
        vertex.tangent[2] = tangent[2];
    }
    if !has_skinning {
        instance.transmission.baked_thickness_scale *= mat4_mean_scale(world);
    }
    instance
}

fn selected_scene_transforms(gltf: &gltf::Gltf) -> Vec<Vec<[[f32; 4]; 4]>> {
    let mesh_count = gltf.meshes().count();
    let mut transforms = vec![Vec::new(); mesh_count];
    let identity = crate::renderer::IDENTITY_MAT4;
    // glTF defines one active/default scene. Walking every scene duplicates
    // placements from authoring variants and makes scene selection depend on
    // exporter ordering. Fall back deterministically to the first scene.
    if let Some(scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
        for node in scene.nodes() {
            walk_scene_collect_instances(&node, &identity, &mut transforms);
        }
    }
    transforms
}

/// Expand scene placements while sharing every immutable static primitive.
///
/// The returned vectors are parallel: `meshes[i]` is drawn with
/// `transforms[i]`. Multiple nodes that reference the same glTF primitive
/// therefore clone only an `Arc`; no vertex/index payload is copied or baked.
pub(super) fn share_scene_mesh_instances(
    gltf: &gltf::Gltf,
    source_meshes: Vec<Vec<MeshData>>,
) -> (Vec<Arc<MeshData>>, Vec<[[f32; 4]; 4]>) {
    let transforms = selected_scene_transforms(gltf);
    let identity = crate::renderer::IDENTITY_MAT4;
    let mut output = Vec::new();
    let mut output_transforms = Vec::new();

    for (mesh_index, primitives) in source_meshes.into_iter().enumerate() {
        let worlds: &[[[f32; 4]; 4]] = if transforms[mesh_index].is_empty() {
            std::slice::from_ref(&identity)
        } else {
            &transforms[mesh_index]
        };
        let primitives: Vec<Arc<MeshData>> = primitives.into_iter().map(Arc::new).collect();

        for world in worlds {
            for primitive in &primitives {
                let (instance, transform) = shared_or_owned_instance(primitive, *world);
                output.push(instance);
                output_transforms.push(transform);
            }
        }
    }
    (output, output_transforms)
}

pub(super) fn model_bounds(
    meshes: &[Arc<MeshData>],
    transforms: &[[[f32; 4]; 4]],
) -> ([f32; 3], [f32; 3]) {
    let mut minimum = [f32::MAX; 3];
    let mut maximum = [f32::MIN; 3];
    for (index, mesh) in meshes.iter().enumerate() {
        let transform = transforms
            .get(index)
            .copied()
            .unwrap_or(crate::renderer::IDENTITY_MAT4);
        for vertex in &mesh.vertices {
            let position = mat4_transform_point(&transform, &vertex.position);
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(position[axis]);
                maximum[axis] = maximum[axis].max(position[axis]);
            }
        }
    }
    if meshes.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (minimum, maximum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformed_bounds_do_not_require_baked_vertices() {
        let mesh = Arc::new(MeshData {
            vertices: vec![Vertex3D {
                position: [1.0, 2.0, 3.0],
                normal: [0.0, 1.0, 0.0],
                color: [1.0; 4],
                uv: [0.0; 2],
                joints: [0.0; 4],
                weights: [0.0; 4],
                tangent: [0.0; 4],
            }],
            secondary_tex_coords: None,
            indices: vec![0],
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
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
        });
        let mut transform = crate::renderer::IDENTITY_MAT4;
        transform[3] = [10.0, -2.0, 4.0, 1.0];
        let (minimum, maximum) = model_bounds(&[mesh], &[transform]);
        assert_eq!(minimum, [11.0, 0.0, 7.0]);
        assert_eq!(maximum, minimum);
    }
}
