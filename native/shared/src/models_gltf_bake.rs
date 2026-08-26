use super::transform::{
    mat3_transform_vec, mat4_inverse_transpose_3x3, mat4_mean_scale, mat4_transform_direction,
    mat4_transform_point,
};
use super::{walk_scene_collect_instances, MeshData, ModelPrimitiveSource, Vertex3D};
use std::sync::Arc;

pub(super) fn complete_geometry_source_hash(
    source_bytes: &[u8],
    gltf: &gltf::Gltf,
    buffers: &[Vec<u8>],
) -> Option<[u8; 32]> {
    let descriptors = gltf.buffers().collect::<Vec<_>>();
    if descriptors.len() != buffers.len()
        || descriptors
            .iter()
            .zip(buffers)
            .any(|(descriptor, bytes)| bytes.len() < descriptor.length())
    {
        return None;
    }
    let slices = buffers.iter().map(Vec::as_slice).collect::<Vec<&[u8]>>();
    Some(bloom_geometry_format::geometry_source_sha256(
        source_bytes,
        &slices,
    ))
}

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

fn selected_scene_instances(gltf: &gltf::Gltf) -> Vec<Vec<super::MeshInstance>> {
    let mesh_count = gltf.meshes().count();
    let mut instances = vec![Vec::new(); mesh_count];
    let identity = crate::renderer::IDENTITY_MAT4;
    // glTF defines one active/default scene. Walking every scene duplicates
    // placements from authoring variants and makes scene selection depend on
    // exporter ordering. Fall back deterministically to the first scene.
    if let Some(scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
        for node in scene.nodes() {
            walk_scene_collect_instances(&node, &identity, &mut instances);
        }
    }
    instances
}

/// Expand scene placements while sharing every immutable static primitive.
///
/// The returned vectors are parallel: `meshes[i]` is drawn with
/// `transforms[i]`. Multiple nodes that reference the same glTF primitive
/// therefore clone only an `Arc`; no vertex/index payload is copied or baked.
pub(super) fn share_scene_mesh_instances(
    gltf: &gltf::Gltf,
    source_meshes: Vec<Vec<(u32, MeshData)>>,
) -> (
    Vec<Arc<MeshData>>,
    Vec<[[f32; 4]; 4]>,
    Vec<bool>,
    Vec<Option<ModelPrimitiveSource>>,
) {
    let instances = selected_scene_instances(gltf);
    let identity = crate::renderer::IDENTITY_MAT4;
    let mut output = Vec::new();
    let mut output_transforms = Vec::new();
    let mut output_cast_shadows = Vec::new();
    let mut output_sources = Vec::new();

    for (mesh_index, primitives) in source_meshes.into_iter().enumerate() {
        let fallback = super::MeshInstance {
            transform: identity,
            cast_shadow: true,
        };
        let placements: &[super::MeshInstance] = if instances[mesh_index].is_empty() {
            std::slice::from_ref(&fallback)
        } else {
            &instances[mesh_index]
        };
        let primitives: Vec<(u32, Arc<MeshData>)> = primitives
            .into_iter()
            .map(|(primitive_index, primitive)| (primitive_index, Arc::new(primitive)))
            .collect();

        for (placement_index, placement) in placements.iter().enumerate() {
            for (primitive_index, primitive) in &primitives {
                let (instance, transform) =
                    shared_or_owned_instance(primitive, placement.transform);
                output.push(instance);
                output_transforms.push(transform);
                output_cast_shadows.push(placement.cast_shadow);
                output_sources.push(Some(ModelPrimitiveSource {
                    mesh_index: mesh_index as u32,
                    primitive_index: *primitive_index,
                    placement_index: placement_index as u32,
                }));
            }
        }
    }
    (
        output,
        output_transforms,
        output_cast_shadows,
        output_sources,
    )
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

    fn one_vertex_mesh() -> MeshData {
        MeshData {
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
        }
    }

    #[test]
    fn transformed_bounds_do_not_require_baked_vertices() {
        let mesh = Arc::new(one_vertex_mesh());
        let mut transform = crate::renderer::IDENTITY_MAT4;
        transform[3] = [10.0, -2.0, 4.0, 1.0];
        let (minimum, maximum) = model_bounds(&[mesh], &[transform]);
        assert_eq!(minimum, [11.0, 0.0, 7.0]);
        assert_eq!(maximum, minimum);
    }

    #[test]
    fn scene_instances_preserve_authored_shadow_intent() {
        let document = br#"{
            "asset":{"version":"2.0"},
            "scene":0,
            "scenes":[{"nodes":[0,1]}],
            "nodes":[
                {"mesh":0,"extras":{"BLOOM_cast_shadow":false}},
                {"mesh":0,"translation":[4.0,0.0,0.0]}
            ],
            "buffers":[{"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA","byteLength":12}],
            "bufferViews":[{"buffer":0,"byteLength":12}],
            "accessors":[{
                "bufferView":0,"componentType":5126,"count":1,"type":"VEC3",
                "min":[0.0,0.0,0.0],"max":[0.0,0.0,0.0]
            }],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]
        }"#;
        let gltf = gltf::Gltf::from_slice(document).expect("minimal glTF parses");
        let (meshes, transforms, cast_shadows, sources) =
            share_scene_mesh_instances(&gltf, vec![vec![(0, one_vertex_mesh())]]);

        assert_eq!(meshes.len(), 2);
        assert_eq!(transforms.len(), 2);
        assert_eq!(cast_shadows, vec![false, true]);
        assert_eq!(transforms[1][3][0], 4.0);
        assert_eq!(sources[0].unwrap().placement_index, 0);
        assert_eq!(sources[1].unwrap().placement_index, 1);
    }
}
