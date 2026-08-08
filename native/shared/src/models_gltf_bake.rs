use super::{
    mat3_transform_vec, mat4_inverse_transpose_3x3, mat4_mean_scale, mat4_transform_direction,
    mat4_transform_point, walk_scene_collect_instances, MeshData, Vertex3D,
};

fn vertex_is_skinned(vertex: &Vertex3D) -> bool {
    vertex.weights.iter().sum::<f32>() > 0.01
}

/// Bake one static glTF node transform into a mesh instance.
///
/// Skinned vertices remain in armature space. Static geometry carries the
/// transform's mean scale into volume thickness because its node matrix will
/// no longer exist when transmission is shaded.
fn bake_mesh_instance(mut instance: MeshData, world: &[[f32; 4]; 4]) -> MeshData {
    let normal_transform = mat4_inverse_transpose_3x3(world);
    let has_skinning = instance.vertices.iter().any(vertex_is_skinned);
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

/// Expand raw mesh primitives into every concrete scene instance.
pub(super) fn bake_scene_mesh_instances(
    gltf: &gltf::Gltf,
    source_meshes: Vec<Vec<MeshData>>,
) -> Vec<MeshData> {
    let mesh_count = gltf.meshes().count();
    let mut transforms: Vec<Vec<[[f32; 4]; 4]>> = vec![Vec::new(); mesh_count];
    let identity = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            walk_scene_collect_instances(&node, &identity, &mut transforms);
        }
    }

    let mut output = Vec::new();
    for (mesh_index, primitives) in source_meshes.into_iter().enumerate() {
        let Some((last_world, preceding_worlds)) = transforms[mesh_index].split_last() else {
            output.extend(primitives);
            continue;
        };
        for world in preceding_worlds {
            output.extend(
                primitives
                    .iter()
                    .cloned()
                    .map(|primitive| bake_mesh_instance(primitive, world)),
            );
        }
        output.extend(
            primitives
                .into_iter()
                .map(|primitive| bake_mesh_instance(primitive, last_world)),
        );
    }
    output
}

pub(super) fn model_bounds(meshes: &[MeshData]) -> ([f32; 3], [f32; 3]) {
    let mut minimum = [f32::MAX; 3];
    let mut maximum = [f32::MIN; 3];
    for vertex in meshes.iter().flat_map(|mesh| &mesh.vertices) {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex.position[axis]);
            maximum[axis] = maximum[axis].max(vertex.position[axis]);
        }
    }
    (minimum, maximum)
}
