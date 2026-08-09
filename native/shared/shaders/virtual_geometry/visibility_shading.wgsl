// Full-PBR reconstruction for IDs produced by visibility_raster.wgsl. This
// entry point is a disjoint fullscreen pass: compatibility IDs discard here,
// and virtual IDs discard in the compatibility visibility shader.

struct VirtualVisibilityShadeVertexOut {
    @builtin(position) position: vec4<f32>,
};

@group(4) @binding(0) var virtual_shade_ids: texture_2d<u32>;
@group(4) @binding(1) var<storage, read> virtual_page_words: BloomVirtualRawWords;
@group(4) @binding(2) var<storage, read> virtual_meshes: VirtualMeshTable;
@group(4) @binding(3) var<storage, read> virtual_clusters: VirtualClusterTable;
@group(4) @binding(4) var<storage, read> virtual_selected: VirtualSelectedTable;
@group(4) @binding(5) var<storage, read> virtual_instances: VirtualInstanceTable;
@group(4) @binding(6) var<uniform> virtual_frame: GpuVirtualVisibilityFrame;

@vertex
fn vs_virtual_visibility_shade(
    @builtin(vertex_index) vertex_index: u32,
) -> VirtualVisibilityShadeVertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VirtualVisibilityShadeVertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

fn virtual_visibility_shade_fault() -> SceneOut {
    return SceneOut(
        vec4<f32>(8.0, 0.0, 8.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0),
        vec4<f32>(1.0, 0.0, 1.0, 0.0),
    );
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
        safe_scene_tangent(
            (instance.model * vec4<f32>(local_tangent.xyz, 0.0)).xyz,
        ),
        local_tangent.w * model_handedness,
    );
}

@fragment
fn fs_virtual_visibility_shade(
    in: VirtualVisibilityShadeVertexOut,
) -> SceneOut {
    let pixel = vec2<i32>(in.position.xy);
    let raw_visibility = textureLoad(virtual_shade_ids, pixel, 0).xy;
    if (!bloom_visibility_valid(raw_visibility)) { discard; }
    let visibility = bloom_decode_visibility(raw_visibility);
    if (!visibility.virtual_geometry) { discard; }
    if (visibility.draw_id >= arrayLength(&virtual_selected.records)) {
        return virtual_visibility_shade_fault();
    }
    let selection = virtual_selected.records[visibility.draw_id];
    if ((selection.flags & BLOOM_VIRTUAL_FLAG_ALPHA_MASKED) != 0u
        || selection.material_id == 0u
        || selection.instance_index >= arrayLength(&virtual_instances.records)) {
        return virtual_visibility_shade_fault();
    }
    let instance = virtual_instances.records[selection.instance_index];
    if (instance.instance_info.x != selection.mesh_id) {
        return virtual_visibility_shade_fault();
    }
    let mesh_slot_plus_one = selection.mesh_id & BLOOM_VIRTUAL_MESH_SLOT_MASK;
    if (mesh_slot_plus_one == 0u) {
        return virtual_visibility_shade_fault();
    }
    let mesh_index = mesh_slot_plus_one - 1u;
    if (mesh_index >= arrayLength(&virtual_meshes.records)) {
        return virtual_visibility_shade_fault();
    }
    let mesh = virtual_meshes.records[mesh_index];
    if (mesh.mesh_id != selection.mesh_id
        || selection.cluster_index >= mesh.cluster_count) {
        return virtual_visibility_shade_fault();
    }
    let cluster_index = mesh.cluster_table_base + selection.cluster_index;
    if (cluster_index >= arrayLength(&virtual_clusters.records)) {
        return virtual_visibility_shade_fault();
    }
    let cluster = virtual_clusters.records[cluster_index];
    if (selection.triangle_count != cluster.page_lod_counts.w
        || visibility.primitive_id >= selection.triangle_count) {
        return virtual_visibility_shade_fault();
    }

    let page_base = selection.physical_slot * mesh.page_stride_bytes;
    let corner_base = page_base + cluster.payload.y + visibility.primitive_id * 3u;
    let local_index0 = bloom_virtual_load_local_index(corner_base);
    let local_index1 = bloom_virtual_load_local_index(corner_base + 1u);
    let local_index2 = bloom_virtual_load_local_index(corner_base + 2u);
    let vertex_count = cluster.page_lod_counts.z;
    if (local_index0 >= vertex_count
        || local_index1 >= vertex_count
        || local_index2 >= vertex_count) {
        return virtual_visibility_shade_fault();
    }
    let vertex_base = page_base + cluster.payload.x;
    let vertex_stride = cluster.payload.z;
    let vertex0 = bloom_virtual_decode_vertex(
        vertex_base + local_index0 * vertex_stride,
        mesh.vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    let vertex1 = bloom_virtual_decode_vertex(
        vertex_base + local_index1 * vertex_stride,
        mesh.vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    let vertex2 = bloom_virtual_decode_vertex(
        vertex_base + local_index2 * vertex_stride,
        mesh.vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );

    let local0 = vec4<f32>(vertex0.position, 1.0);
    let local1 = vec4<f32>(vertex1.position, 1.0);
    let local2 = vec4<f32>(vertex2.position, 1.0);
    let world0 = instance.model * local0;
    let world1 = instance.model * local1;
    let world2 = instance.model * local2;
    let clip0 = virtual_frame.view_projection * world0;
    let clip1 = virtual_frame.view_projection * world1;
    let clip2 = virtual_frame.view_projection * world2;
    let dimensions = textureDimensions(virtual_shade_ids);
    let point_ndc = vec2<f32>(
        in.position.x / f32(dimensions.x) * 2.0 - 1.0,
        1.0 - in.position.y / f32(dimensions.y) * 2.0,
    );
    let bary = bloom_perspective_barycentrics(point_ndc, clip0, clip1, clip2);
    let current_clip = bloom_interpolate4(clip0, clip1, clip2, bary);

    // Reconstruct the two adjacent helper lanes from this same triangle so
    // texture gradients and specular AA match raster helper invocations.
    let x_step = select(-1.0, 1.0, (u32(pixel.x) & 1u) == 0u);
    let y_step = select(-1.0, 1.0, (u32(pixel.y) & 1u) == 0u);
    let point_x_ndc = vec2<f32>(
        (in.position.x + x_step) / f32(dimensions.x) * 2.0 - 1.0,
        point_ndc.y,
    );
    let point_y_ndc = vec2<f32>(
        point_ndc.x,
        1.0 - (in.position.y + y_step) / f32(dimensions.y) * 2.0,
    );
    let bary_x = bloom_perspective_barycentrics(point_x_ndc, clip0, clip1, clip2);
    let bary_y = bloom_perspective_barycentrics(point_y_ndc, clip0, clip1, clip2);

    let normal0 = bloom_virtual_world_normal(instance, vertex0.normal);
    let normal1 = bloom_virtual_world_normal(instance, vertex1.normal);
    let normal2 = bloom_virtual_world_normal(instance, vertex2.normal);
    let tangent0 = bloom_virtual_world_tangent(instance, vertex0.tangent);
    let tangent1 = bloom_virtual_world_tangent(instance, vertex1.tangent);
    let tangent2 = bloom_virtual_world_tangent(instance, vertex2.tangent);
    let fragment_normal = bloom_interpolate3(normal0, normal1, normal2, bary);
    let fragment_uv = bloom_interpolate2(vertex0.uv0, vertex1.uv0, vertex2.uv0, bary);
    let fragment_world = bloom_interpolate3(world0.xyz, world1.xyz, world2.xyz, bary);
    let fragment_tangent = bloom_interpolate4(tangent0, tangent1, tangent2, bary);
    let normal_x = bloom_interpolate3(normal0, normal1, normal2, bary_x);
    let normal_y = bloom_interpolate3(normal0, normal1, normal2, bary_y);
    let tangent_x = bloom_interpolate4(tangent0, tangent1, tangent2, bary_x);
    let tangent_y = bloom_interpolate4(tangent0, tangent1, tangent2, bary_y);
    let uv_x = bloom_interpolate2(vertex0.uv0, vertex1.uv0, vertex2.uv0, bary_x);
    let uv_y = bloom_interpolate2(vertex0.uv0, vertex1.uv0, vertex2.uv0, bary_y);
    let world_x = bloom_interpolate3(world0.xyz, world1.xyz, world2.xyz, bary_x);
    let world_y = bloom_interpolate3(world0.xyz, world1.xyz, world2.xyz, bary_y);
    let visibility_gradients = BloomVisibilityGradients(
        (world_x - fragment_world) * x_step,
        (world_y - fragment_world) * y_step,
        (uv_x - fragment_uv) * x_step,
        (uv_y - fragment_uv) * y_step,
        normal_x,
        normal_y,
        tangent_x,
        tangent_y,
        uv_x,
        uv_y,
        x_step,
        y_step,
    );

    var fragment: VertexOutputScene;
    fragment.clip_position = in.position;
    fragment.normal = fragment_normal;
    fragment.color = bloom_interpolate4(
        vertex0.color,
        vertex1.color,
        vertex2.color,
        bary,
    ) * instance.model_tint;
    fragment.uv = fragment_uv;
    fragment.world_pos = fragment_world;
    fragment.tangent = fragment_tangent;
    fragment.curr_clip = current_clip;
    fragment.prev_clip = bloom_interpolate4(
        virtual_frame.previous_view_projection * (instance.previous_model * local0),
        virtual_frame.previous_view_projection * (instance.previous_model * local1),
        virtual_frame.previous_view_projection * (instance.previous_model * local2),
        bary,
    );
    fragment.material_id = selection.material_id;
    fragment.draw_flags = selection.flags & BLOOM_VIRTUAL_FLAG_DOUBLE_SIDED;
    return shade_main_scene(
        fragment,
        visibility.front_facing,
        visibility_gradients,
    );
}
