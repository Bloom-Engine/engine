struct BloomVirtualShadingProbeRecord {
    result_info: vec4<u32>,
    identity: vec4<u32>,
    barycentrics: vec4<f32>,
    current_clip: vec4<f32>,
    previous_clip: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec4<f32>,
    world_tangent: vec4<f32>,
    uv: vec4<f32>,
    color: vec4<f32>,
};
struct BloomVirtualShadingProbeTable {
    records: array<BloomVirtualShadingProbeRecord>,
};

@group(0) @binding(0) var virtual_shade_ids: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> virtual_page_words: BloomVirtualRawWords;
@group(0) @binding(2) var<storage, read> virtual_clusters: VirtualClusterTable;
@group(0) @binding(3) var<storage, read> virtual_selected: VirtualSelectedTable;
@group(0) @binding(4) var<storage, read> virtual_instances: VirtualInstanceTable;
@group(0) @binding(5) var<uniform> virtual_frame: GpuVirtualVisibilityFrame;
@group(0) @binding(6) var<storage, read_write> virtual_probe: BloomVirtualShadingProbeTable;

@compute @workgroup_size(8, 8, 1)
fn cs_virtual_visibility_shading_probe(
    @builtin(global_invocation_id) invocation: vec3<u32>,
) {
    let dimensions = textureDimensions(virtual_shade_ids);
    if (invocation.x >= dimensions.x || invocation.y >= dimensions.y) {
        return;
    }
    let output_index = invocation.y * dimensions.x + invocation.x;
    if (output_index >= arrayLength(&virtual_probe.records)) {
        return;
    }
    let pixel = vec2<i32>(invocation.xy);
    let raw_visibility = textureLoad(virtual_shade_ids, pixel, 0).xy;
    if (!bloom_visibility_valid(raw_visibility)) {
        return;
    }
    let visibility = bloom_decode_visibility(raw_visibility);
    if (!visibility.virtual_geometry) {
        return;
    }
    let triangle = bloom_virtual_visibility_triangle(
        visibility.draw_id,
        visibility.primitive_id,
    );
    if (!triangle.valid) {
        return;
    }
    let point_ndc = vec2<f32>(
        (f32(invocation.x) + 0.5) / f32(dimensions.x) * 2.0 - 1.0,
        1.0 - (f32(invocation.y) + 0.5) / f32(dimensions.y) * 2.0,
    );
    let bary = bloom_perspective_barycentrics(
        point_ndc,
        triangle.clip0,
        triangle.clip1,
        triangle.clip2,
    );
    let normal0 = bloom_virtual_world_normal(triangle.instance, triangle.vertex0.normal);
    let normal1 = bloom_virtual_world_normal(triangle.instance, triangle.vertex1.normal);
    let normal2 = bloom_virtual_world_normal(triangle.instance, triangle.vertex2.normal);
    let tangent0 = bloom_virtual_world_tangent(triangle.instance, triangle.vertex0.tangent);
    let tangent1 = bloom_virtual_world_tangent(triangle.instance, triangle.vertex1.tangent);
    let tangent2 = bloom_virtual_world_tangent(triangle.instance, triangle.vertex2.tangent);
    virtual_probe.records[output_index] = BloomVirtualShadingProbeRecord(
        vec4<u32>(
            1u,
            visibility.draw_id,
            visibility.primitive_id,
            triangle.selection.material_id,
        ),
        vec4<u32>(
            select(0u, 1u, visibility.front_facing),
            triangle.selection.instance_index,
            triangle.instance.instance_info.z,
            triangle.selection.flags,
        ),
        vec4<f32>(bary, 1.0),
        bloom_interpolate4(triangle.clip0, triangle.clip1, triangle.clip2, bary),
        bloom_interpolate4(
            virtual_frame.previous_view_projection
                * (triangle.instance.previous_model * triangle.local0),
            virtual_frame.previous_view_projection
                * (triangle.instance.previous_model * triangle.local1),
            virtual_frame.previous_view_projection
                * (triangle.instance.previous_model * triangle.local2),
            bary,
        ),
        vec4<f32>(
            bloom_interpolate3(
                triangle.world0.xyz,
                triangle.world1.xyz,
                triangle.world2.xyz,
                bary,
            ),
            1.0,
        ),
        vec4<f32>(bloom_interpolate3(normal0, normal1, normal2, bary), 0.0),
        bloom_interpolate4(tangent0, tangent1, tangent2, bary),
        vec4<f32>(
            bloom_interpolate2(
                triangle.vertex0.uv0,
                triangle.vertex1.uv0,
                triangle.vertex2.uv0,
                bary,
            ),
            0.0,
            0.0,
        ),
        bloom_interpolate4(
            triangle.vertex0.color,
            triangle.vertex1.color,
            triangle.vertex2.color,
            bary,
        ) * triangle.instance.model_tint,
    );
}
