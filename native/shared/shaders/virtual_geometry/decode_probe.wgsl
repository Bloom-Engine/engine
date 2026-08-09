// Test-only compute consumer for the shared cooked-page decoder. Its ABI
// mirrors the production virtual cluster and selection tables exactly.

const BLOOM_VIRTUAL_VERTEX_ENCODING_SHIFT: u32 = 28u;
const BLOOM_VIRTUAL_VERTEX_ENCODING_MASK: u32 = 3u;

struct GpuVirtualClusterEntry {
    aabb_min_error: vec4<f32>,
    aabb_max_radius: vec4<f32>,
    sphere: vec4<f32>,
    normal_cone: vec4<f32>,
    identity: vec4<u32>,
    page_lod_counts: vec4<u32>,
    payload: vec4<u32>,
    relations: vec4<u32>,
};
struct ClusterTable { records: array<GpuVirtualClusterEntry>, };

struct GpuSelectedVirtualCluster {
    mesh_id: u32,
    instance_index: u32,
    cluster_table_index: u32,
    physical_page_base: u32,
    lod_level: u32,
    triangle_count: u32,
    material_id: u32,
    flags: u32,
};
struct SelectedTable { records: array<GpuSelectedVirtualCluster>, };

struct GpuVirtualInstance {
    model: mat4x4<f32>,
    normal_rows: array<vec4<f32>, 3>,
    instance_info: vec4<u32>,
    previous_model: mat4x4<f32>,
    model_tint: vec4<f32>,
};
struct InstanceTable { records: array<GpuVirtualInstance>, };

struct GpuDecodedVirtualVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    tangent: vec4<f32>,
    uv0_uv1: vec4<f32>,
    color: vec4<f32>,
    current_world: vec4<f32>,
    previous_world: vec4<f32>,
    tinted_color: vec4<f32>,
    world_normal: vec4<f32>,
    info: vec4<u32>,
};
struct DecodedTable { records: array<GpuDecodedVirtualVertex>, };

struct DecodeParams {
    selected_count: u32,
    max_corners: u32,
    output_capacity: u32,
    reserved: u32,
};

@group(0) @binding(1) var<storage, read> clusters: ClusterTable;
@group(0) @binding(2) var<storage, read> selected: SelectedTable;
@group(0) @binding(3) var<storage, read_write> decoded: DecodedTable;
@group(0) @binding(4) var<uniform> params: DecodeParams;
@group(0) @binding(5) var<storage, read> instances: InstanceTable;

@compute @workgroup_size(32, 1, 1)
fn decode_selected_corners(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let corner = invocation.x;
    let selected_index = invocation.y;
    if (selected_index >= params.selected_count || corner >= params.max_corners) {
        return;
    }
    let output_index = selected_index * params.max_corners + corner;
    if (output_index >= params.output_capacity
        || selected_index >= arrayLength(&selected.records)) {
        return;
    }
    let selection = selected.records[selected_index];
    if (selection.instance_index >= arrayLength(&instances.records)) {
        return;
    }
    let instance = instances.records[selection.instance_index];
    if (selection.cluster_table_index >= arrayLength(&clusters.records)) {
        return;
    }
    let cluster = clusters.records[selection.cluster_table_index];
    let corner_count = cluster.page_lod_counts.w * 3u;
    if (cluster.payload.w != selection.mesh_id
        || corner >= corner_count
        || selection.triangle_count != cluster.page_lod_counts.w) {
        return;
    }
    let page_base = selection.physical_page_base;
    let local_vertex = bloom_virtual_load_local_index(page_base + cluster.payload.y + corner);
    if (local_vertex >= cluster.page_lod_counts.z) {
        return;
    }
    let vertex_offset = page_base + cluster.payload.x + local_vertex * cluster.payload.z;
    let vertex = bloom_virtual_decode_vertex(
        vertex_offset,
        (selection.flags >> BLOOM_VIRTUAL_VERTEX_ENCODING_SHIFT)
            & BLOOM_VIRTUAL_VERTEX_ENCODING_MASK,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    let local_position = vec4<f32>(vertex.position, 1.0);
    let world_normal = normalize(vec3<f32>(
        dot(instance.normal_rows[0].xyz, vertex.normal),
        dot(instance.normal_rows[1].xyz, vertex.normal),
        dot(instance.normal_rows[2].xyz, vertex.normal),
    ));
    decoded.records[output_index] = GpuDecodedVirtualVertex(
        local_position,
        vec4<f32>(vertex.normal, 0.0),
        vertex.tangent,
        vec4<f32>(vertex.uv0, vertex.uv1),
        vertex.color,
        instance.model * local_position,
        instance.previous_model * local_position,
        vertex.color * instance.model_tint,
        vec4<f32>(world_normal, 0.0),
        vec4<u32>(selected_index, selection.cluster_table_index, corner, local_vertex),
    );
}
