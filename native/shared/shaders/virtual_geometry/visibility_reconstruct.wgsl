// Shared validated triangle reconstruction used by the full-PBR consumer and
// its real-GPU oracle. Bindings must use the virtual_* names declared by the
// consumer before this source is appended.

struct BloomVirtualVisibilityTriangle {
    valid: bool,
    selection: GpuSelectedVirtualCluster,
    instance: GpuVirtualInstance,
    vertex0: BloomVirtualVertex,
    vertex1: BloomVirtualVertex,
    vertex2: BloomVirtualVertex,
    local0: vec4<f32>,
    local1: vec4<f32>,
    local2: vec4<f32>,
    world0: vec4<f32>,
    world1: vec4<f32>,
    world2: vec4<f32>,
    clip0: vec4<f32>,
    clip1: vec4<f32>,
    clip2: vec4<f32>,
};

fn bloom_virtual_visibility_triangle(
    draw_index: u32,
    primitive_id: u32,
) -> BloomVirtualVisibilityTriangle {
    var result: BloomVirtualVisibilityTriangle;
    result.valid = false;
    if (draw_index >= arrayLength(&virtual_selected.records)) {
        return result;
    }
    let selection = virtual_selected.records[draw_index];
    if ((selection.flags & BLOOM_VIRTUAL_FLAG_ALPHA_MASKED) != 0u
        || selection.material_id == 0u
        || selection.instance_index >= arrayLength(&virtual_instances.records)) {
        return result;
    }
    let instance = virtual_instances.records[selection.instance_index];
    if (instance.instance_info.x != selection.mesh_id) {
        return result;
    }
    if (selection.cluster_table_index >= arrayLength(&virtual_clusters.records)) {
        return result;
    }
    let cluster = virtual_clusters.records[selection.cluster_table_index];
    if (cluster.payload.w != selection.mesh_id
        || selection.triangle_count != cluster.page_lod_counts.w
        || primitive_id >= selection.triangle_count) {
        return result;
    }

    let page_base = selection.physical_page_base;
    let corner_base = page_base + cluster.payload.y + primitive_id * 3u;
    let local_index0 = bloom_virtual_load_local_index(corner_base);
    let local_index1 = bloom_virtual_load_local_index(corner_base + 1u);
    let local_index2 = bloom_virtual_load_local_index(corner_base + 2u);
    let vertex_count = cluster.page_lod_counts.z;
    if (local_index0 >= vertex_count
        || local_index1 >= vertex_count
        || local_index2 >= vertex_count) {
        return result;
    }
    let vertex_base = page_base + cluster.payload.x;
    let vertex_stride = cluster.payload.z;
    let vertex_encoding = (selection.flags >> BLOOM_VIRTUAL_VERTEX_ENCODING_SHIFT)
        & BLOOM_VIRTUAL_VERTEX_ENCODING_MASK;
    let vertex0 = bloom_virtual_decode_vertex(
        vertex_base + local_index0 * vertex_stride,
        vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    let vertex1 = bloom_virtual_decode_vertex(
        vertex_base + local_index1 * vertex_stride,
        vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    let vertex2 = bloom_virtual_decode_vertex(
        vertex_base + local_index2 * vertex_stride,
        vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    let local0 = vec4<f32>(vertex0.position, 1.0);
    let local1 = vec4<f32>(vertex1.position, 1.0);
    let local2 = vec4<f32>(vertex2.position, 1.0);
    let world0 = instance.model * local0;
    let world1 = instance.model * local1;
    let world2 = instance.model * local2;

    result.valid = true;
    result.selection = selection;
    result.instance = instance;
    result.vertex0 = vertex0;
    result.vertex1 = vertex1;
    result.vertex2 = vertex2;
    result.local0 = local0;
    result.local1 = local1;
    result.local2 = local2;
    result.world0 = world0;
    result.world1 = world1;
    result.world2 = world2;
    result.clip0 = virtual_frame.view_projection * world0;
    result.clip1 = virtual_frame.view_projection * world1;
    result.clip2 = virtual_frame.view_projection * world2;
    return result;
}

fn bloom_virtual_world_normal(
    instance: GpuVirtualInstance,
    local_normal: vec3<f32>,
) -> vec3<f32> {
    return normalize(vec3<f32>(
        dot(instance.normal_rows[0].xyz, local_normal),
        dot(instance.normal_rows[1].xyz, local_normal),
        dot(instance.normal_rows[2].xyz, local_normal),
    ));
}

fn bloom_virtual_safe_direction(direction: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(direction, direction);
    return select(
        vec3<f32>(0.0),
        direction * inverseSqrt(max(length_squared, 1e-20)),
        length_squared > 1e-8,
    );
}

fn bloom_virtual_world_tangent(
    instance: GpuVirtualInstance,
    local_tangent: vec4<f32>,
) -> vec4<f32> {
    let model_handedness = select(
        1.0,
        -1.0,
        (instance.instance_info.z & BLOOM_VIRTUAL_INSTANCE_NEGATIVE_DETERMINANT) != 0u,
    );
    return vec4<f32>(
        bloom_virtual_safe_direction(
            (instance.model * vec4<f32>(local_tangent.xyz, 0.0)).xyz,
        ),
        local_tangent.w * model_handedness,
    );
}
