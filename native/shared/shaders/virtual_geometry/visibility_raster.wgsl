struct VirtualVisibilityVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) draw_index: u32,
    @location(1) @interpolate(flat) flags: u32,
};

@group(0) @binding(0) var<storage, read> virtual_page_words: BloomVirtualRawWords;
@group(0) @binding(1) var<storage, read> virtual_clusters: VirtualClusterTable;
@group(0) @binding(2) var<storage, read> virtual_selected: VirtualSelectedTable;
@group(0) @binding(3) var<storage, read> virtual_instances: VirtualInstanceTable;
@group(0) @binding(4) var<uniform> virtual_frame: GpuVirtualVisibilityFrame;

fn bloom_invalid_virtual_vertex() -> VirtualVisibilityVertexOut {
    return VirtualVisibilityVertexOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), 0u, 0u);
}

fn bloom_virtual_visibility_vertex(
    selected_index: u32,
    corner: u32,
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
    if (selection.cluster_table_index >= arrayLength(&virtual_clusters.records)) {
        return bloom_invalid_virtual_vertex();
    }
    let cluster = virtual_clusters.records[selection.cluster_table_index];
    let corner_count = cluster.page_lod_counts.w * 3u;
    if (cluster.payload.w != selection.mesh_id
        || corner >= corner_count
        || selection.triangle_count != cluster.page_lod_counts.w) {
        return bloom_invalid_virtual_vertex();
    }
    let page_base = selection.physical_page_base;
    let local_vertex = bloom_virtual_load_local_index(page_base + cluster.payload.y + corner);
    if (local_vertex >= cluster.page_lod_counts.z) {
        return bloom_invalid_virtual_vertex();
    }
    let vertex_offset = page_base + cluster.payload.x + local_vertex * cluster.payload.z;
    let vertex = bloom_virtual_decode_vertex(
        vertex_offset,
        (selection.flags >> BLOOM_VIRTUAL_VERTEX_ENCODING_SHIFT)
            & BLOOM_VIRTUAL_VERTEX_ENCODING_MASK,
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

@vertex
fn vs_virtual_visibility(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) selected_index: u32,
) -> VirtualVisibilityVertexOut {
    return bloom_virtual_visibility_vertex(selected_index, corner);
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
