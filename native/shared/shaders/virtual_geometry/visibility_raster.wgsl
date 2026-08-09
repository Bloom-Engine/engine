const BLOOM_VIRTUAL_MESH_SLOT_MASK: u32 = 0x000fffffu;
const BLOOM_VIRTUAL_FLAG_DOUBLE_SIDED: u32 = 1u;
const BLOOM_VIRTUAL_FLAG_ALPHA_MASKED: u32 = 2u;

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
struct GpuSelectedVirtualCluster {
    mesh_id: u32,
    instance_index: u32,
    cluster_index: u32,
    physical_slot: u32,
    lod_level: u32,
    triangle_count: u32,
    material_id: u32,
    flags: u32,
};
struct GpuVirtualInstance {
    model: mat4x4<f32>,
    normal_rows: array<vec4<f32>, 3>,
    instance_info: vec4<u32>,
    previous_model: mat4x4<f32>,
    model_tint: vec4<f32>,
};
struct GpuVirtualVisibilityFrame {
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
};
struct VirtualMeshTable { records: array<GpuVirtualMeshEntry>, };
struct VirtualClusterTable { records: array<GpuVirtualClusterEntry>, };
struct VirtualSelectedTable { records: array<GpuSelectedVirtualCluster>, };
struct VirtualInstanceTable { records: array<GpuVirtualInstance>, };

struct VirtualVisibilityVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) draw_index: u32,
    @location(1) @interpolate(flat) flags: u32,
};

@group(0) @binding(0) var<storage, read> virtual_page_words: BloomVirtualRawWords;
@group(0) @binding(1) var<storage, read> virtual_meshes: VirtualMeshTable;
@group(0) @binding(2) var<storage, read> virtual_clusters: VirtualClusterTable;
@group(0) @binding(3) var<storage, read> virtual_selected: VirtualSelectedTable;
@group(0) @binding(4) var<storage, read> virtual_instances: VirtualInstanceTable;
@group(0) @binding(5) var<uniform> virtual_frame: GpuVirtualVisibilityFrame;

fn bloom_invalid_virtual_vertex() -> VirtualVisibilityVertexOut {
    return VirtualVisibilityVertexOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), 0u, 0u);
}

@vertex
fn vs_virtual_visibility(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) selected_index: u32,
) -> VirtualVisibilityVertexOut {
    if (selected_index >= arrayLength(&virtual_selected.records)) {
        return bloom_invalid_virtual_vertex();
    }
    let selection = virtual_selected.records[selected_index];
    if (selection.instance_index >= arrayLength(&virtual_instances.records)) {
        return bloom_invalid_virtual_vertex();
    }
    let instance = virtual_instances.records[selection.instance_index];
    if (instance.instance_info.x != selection.mesh_id) {
        return bloom_invalid_virtual_vertex();
    }
    let mesh_slot_plus_one = selection.mesh_id & BLOOM_VIRTUAL_MESH_SLOT_MASK;
    if (mesh_slot_plus_one == 0u) {
        return bloom_invalid_virtual_vertex();
    }
    let mesh_index = mesh_slot_plus_one - 1u;
    if (mesh_index >= arrayLength(&virtual_meshes.records)) {
        return bloom_invalid_virtual_vertex();
    }
    let mesh = virtual_meshes.records[mesh_index];
    if (mesh.mesh_id != selection.mesh_id || selection.cluster_index >= mesh.cluster_count) {
        return bloom_invalid_virtual_vertex();
    }
    let cluster_index = mesh.cluster_table_base + selection.cluster_index;
    if (cluster_index >= arrayLength(&virtual_clusters.records)) {
        return bloom_invalid_virtual_vertex();
    }
    let cluster = virtual_clusters.records[cluster_index];
    let corner_count = cluster.page_lod_counts.w * 3u;
    if (corner >= corner_count || selection.triangle_count != cluster.page_lod_counts.w) {
        return bloom_invalid_virtual_vertex();
    }
    let page_base = selection.physical_slot * mesh.page_stride_bytes;
    let local_vertex = bloom_virtual_load_local_index(page_base + cluster.payload.y + corner);
    if (local_vertex >= cluster.page_lod_counts.z) {
        return bloom_invalid_virtual_vertex();
    }
    let vertex_offset = page_base + cluster.payload.x + local_vertex * cluster.payload.z;
    let vertex = bloom_virtual_decode_vertex(
        vertex_offset,
        mesh.vertex_encoding,
        cluster.aabb_min_error.xyz,
        cluster.aabb_max_radius.xyz,
    );
    let world = instance.model * vec4<f32>(vertex.position, 1.0);
    return VirtualVisibilityVertexOut(
        virtual_frame.view_projection * world,
        selected_index,
        selection.flags,
    );
}

@fragment
fn fs_virtual_visibility(
    in: VirtualVisibilityVertexOut,
    @builtin(primitive_index) primitive_id: u32,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec2<u32> {
    // Masked clusters remain on compatibility rendering until this pass owns
    // the exact alpha-coverage texture/sampler and cutoff contract.
    if ((in.flags & BLOOM_VIRTUAL_FLAG_ALPHA_MASKED) != 0u) { discard; }
    if ((in.flags & BLOOM_VIRTUAL_FLAG_DOUBLE_SIDED) == 0u && !front_facing) { discard; }
    return bloom_encode_virtual_visibility(in.draw_index, primitive_id, front_facing);
}
