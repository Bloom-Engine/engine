// Shared raw-page render ABI. Keep this declaration block common to the
// visibility raster and PBR reconstruction stages so selected-record and
// transform addressing cannot drift between passes.

const BLOOM_VIRTUAL_MESH_SLOT_MASK: u32 = 0x000fffffu;
const BLOOM_VIRTUAL_FLAG_DOUBLE_SIDED: u32 = 1u;
const BLOOM_VIRTUAL_FLAG_ALPHA_MASKED: u32 = 2u;
const BLOOM_VIRTUAL_INSTANCE_NEGATIVE_DETERMINANT: u32 = 2u;

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
