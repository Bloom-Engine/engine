// Test-only compute consumer for the shared cooked-page decoder. Its ABI
// mirrors the production virtual mesh, cluster, and selection tables exactly.

struct GpuVirtualMeshEntry {
    mesh_id: u32,
    page_table_base: u32,
    page_count: u32,
    cluster_table_base: u32,
    cluster_count: u32,
    root_cluster_count: u32,
    page_stride_bytes: u32,
    vertex_encoding: u32,
    format_version: u32,
    flags: u32,
    reserved: vec2<u32>,
};
struct MeshTable { records: array<GpuVirtualMeshEntry>, };

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
    instance_id: u32,
    cluster_index: u32,
    physical_slot: u32,
    lod_level: u32,
    triangle_count: u32,
    material_index: u32,
    flags: u32,
};
struct SelectedTable { records: array<GpuSelectedVirtualCluster>, };

struct GpuDecodedVirtualVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    tangent: vec4<f32>,
    uv0_uv1: vec4<f32>,
    color: vec4<f32>,
    info: vec4<u32>,
};
struct DecodedTable { records: array<GpuDecodedVirtualVertex>, };

struct DecodeParams {
    selected_count: u32,
    max_corners: u32,
    output_capacity: u32,
    reserved: u32,
};

@group(0) @binding(1) var<storage, read> meshes: MeshTable;
@group(0) @binding(2) var<storage, read> clusters: ClusterTable;
@group(0) @binding(3) var<storage, read> selected: SelectedTable;
@group(0) @binding(4) var<storage, read_write> decoded: DecodedTable;
@group(0) @binding(5) var<uniform> params: DecodeParams;

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
    let mesh_slot_plus_one = selection.mesh_id & 0xfffffu;
    if (mesh_slot_plus_one == 0u) {
        return;
    }
    let mesh_index = mesh_slot_plus_one - 1u;
    if (mesh_index >= arrayLength(&meshes.records)) {
        return;
    }
    let mesh = meshes.records[mesh_index];
    if (mesh.mesh_id != selection.mesh_id || selection.cluster_index >= mesh.cluster_count) {
        return;
    }
    let cluster_index = mesh.cluster_table_base + selection.cluster_index;
    if (cluster_index >= arrayLength(&clusters.records)) {
        return;
    }
    let cluster = clusters.records[cluster_index];
    let corner_count = cluster.page_lod_counts.w * 3u;
    if (corner >= corner_count || selection.triangle_count != cluster.page_lod_counts.w) {
        return;
    }
    let page_base = selection.physical_slot * mesh.page_stride_bytes;
    let local_vertex = bloom_virtual_load_local_index(page_base + cluster.payload.y + corner);
    if (local_vertex >= cluster.page_lod_counts.z) {
        return;
    }
    let vertex_offset = page_base + cluster.payload.x + local_vertex * cluster.payload.z;
    let vertex = bloom_virtual_decode_vertex(
        vertex_offset,
        mesh.vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    decoded.records[output_index] = GpuDecodedVirtualVertex(
        vec4<f32>(vertex.position, 1.0),
        vec4<f32>(vertex.normal, 0.0),
        vertex.tangent,
        vec4<f32>(vertex.uv0, vertex.uv1),
        vertex.color,
        vec4<u32>(selected_index, selection.cluster_index, corner, local_vertex),
    );
}
