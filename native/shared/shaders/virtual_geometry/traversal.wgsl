
const NO_RELATION: u32 = 0xffffffffu;
const ALL_SOURCE_MESHES: u32 = 0xffffffffu;
// Keep private per-invocation storage bounded tightly enough to remain in fast
// GPU local storage. Overflow retains the complete resident parent, so this is
// a quality fallback rather than a coverage failure.
const TRAVERSAL_GROUP_STACK_CAPACITY: u32 = 32u;
const VIRTUAL_CLUSTER_UNIFORM_CHILD_RANGE: u32 = 0x08000000u;
const INSTANCE_CONE_CULL_SAFE: u32 = 1u;
const INSTANCE_PREVIOUS_HIZ_ELIGIBLE: u32 = 4u;

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
struct GpuVirtualPageEntry {
    slot_plus_one: u32,
    payload_bytes: u32,
    mesh_id: u32,
    flags: u32,
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
struct GpuVirtualInstance {
    model: mat4x4<f32>,
    normal_rows: array<vec4<f32>, 3>,
    instance_info: vec4<u32>,
    root_span: vec4<u32>,
    previous_model: mat4x4<f32>,
    model_tint: vec4<f32>,
};
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
struct GpuVirtualPageRequest {
    mesh_id: u32,
    page_index: u32,
    priority_bits: u32,
    source_cluster: u32,
};
struct GpuVirtualPageUse {
    mesh_id: u32,
    source_cluster: u32,
    priority_bits: u32,
};
struct MeshTable { records: array<GpuVirtualMeshEntry>, };
struct PageTable { records: array<GpuVirtualPageEntry>, };
struct ClusterTable { records: array<GpuVirtualClusterEntry>, };
struct InstanceTable { records: array<GpuVirtualInstance>, };
struct SelectedTable { records: array<GpuSelectedVirtualCluster>, };
struct RequestTable { records: array<GpuVirtualPageRequest>, };
struct PageUseTable { records: array<GpuVirtualPageUse>, };
struct TraversalCounters {
    selected_count: atomic<u32>,
    page_request_count: atomic<u32>,
    visible_groups: atomic<u32>,
    frustum_culled_groups: atomic<u32>,
    cone_culled_clusters: atomic<u32>,
    refined_groups: atomic<u32>,
    fallback_groups: atomic<u32>,
    missing_current_pages: atomic<u32>,
    selected_overflow: atomic<u32>,
    request_overflow: atomic<u32>,
    invalid_records: atomic<u32>,
    depth_limit_fallbacks: atomic<u32>,
    occlusion_culled_groups: atomic<u32>,
    occlusion_uncertain_groups: atomic<u32>,
    page_use_count: atomic<u32>,
    page_use_overflow: atomic<u32>,
};
struct TraversalParams {
    planes: array<vec4<f32>, 6>,
    view_projection: mat4x4<f32>,
    camera_projection: vec4<f32>,
    thresholds: vec4<f32>,
    dispatch: vec4<u32>,
    limits: vec4<u32>,
};
struct WorldSphere {
    center: vec3<f32>,
    radius: f32,
};
struct HiZParams {
    previous_view_projection: mat4x4<f32>,
    previous_view: mat4x4<f32>,
    current_view_projection: mat4x4<f32>,
    current_view: mat4x4<f32>,
    extent: vec4<u32>,
    thresholds: vec4<f32>,
};
struct ProjectedBounds {
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    nearest_depth: f32,
    valid: u32,
};

@group(0) @binding(0) var<storage, read> meshes: MeshTable;
@group(0) @binding(1) var<storage, read> pages: PageTable;
@group(0) @binding(2) var<storage, read> clusters: ClusterTable;
@group(0) @binding(3) var<storage, read> instances: InstanceTable;
@group(0) @binding(4) var<storage, read_write> selected: SelectedTable;
@group(0) @binding(5) var<storage, read_write> requests: RequestTable;
@group(0) @binding(6) var<storage, read_write> counters: TraversalCounters;
@group(0) @binding(7) var<storage, read_write> page_uses: PageUseTable;
@group(0) @binding(8) var<uniform> params: TraversalParams;
@group(1) @binding(0) var<uniform> hiz_params: HiZParams;
@group(1) @binding(1) var hiz_0: texture_2d<f32>;
@group(1) @binding(2) var hiz_1: texture_2d<f32>;
@group(1) @binding(3) var hiz_2: texture_2d<f32>;
@group(1) @binding(4) var hiz_3: texture_2d<f32>;
@group(1) @binding(5) var hiz_4: texture_2d<f32>;
@group(1) @binding(6) var hiz_5: texture_2d<f32>;
@group(1) @binding(7) var hiz_6: texture_2d<f32>;
@group(1) @binding(8) var hiz_7: texture_2d<f32>;
@group(1) @binding(9) var hiz_8: texture_2d<f32>;

fn valid_cluster(mesh: GpuVirtualMeshEntry, local_index: u32) -> bool {
    return local_index < mesh.cluster_count
        && mesh.cluster_table_base + local_index < arrayLength(&clusters.records);
}

fn valid_page(mesh: GpuVirtualMeshEntry, local_index: u32) -> bool {
    return local_index < mesh.page_count
        && mesh.page_table_base + local_index < arrayLength(&pages.records);
}

fn scale_bound(model: mat4x4<f32>) -> f32 {
    let c0 = model[0].xyz;
    let c1 = model[1].xyz;
    let c2 = model[2].xyz;
    let g0 = vec3<f32>(dot(c0, c0), dot(c0, c1), dot(c0, c2));
    let g1 = vec3<f32>(g0.y, dot(c1, c1), dot(c1, c2));
    let g2 = vec3<f32>(g0.z, g1.z, dot(c2, c2));
    let eigen_upper = max(
        dot(abs(g0), vec3<f32>(1.0)),
        max(dot(abs(g1), vec3<f32>(1.0)), dot(abs(g2), vec3<f32>(1.0)))
    );
    return sqrt(max(eigen_upper, 0.0));
}

fn world_sphere(
    cluster: GpuVirtualClusterEntry,
    instance: GpuVirtualInstance,
    scale: f32,
) -> WorldSphere {
    return WorldSphere(
        (instance.model * vec4<f32>(cluster.sphere.xyz, 1.0)).xyz,
        cluster.aabb_max_radius.w * scale
    );
}

fn project_hiz_bounds(
    local_min: vec3<f32>,
    local_max: vec3<f32>,
    model: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
) -> ProjectedBounds {
    var uv_min = vec2<f32>(1.0e30);
    var uv_max = vec2<f32>(-1.0e30);
    var nearest_depth = 1.0e30;
    for (var corner = 0u; corner < 8u; corner++) {
        let local = vec3<f32>(
            select(local_min.x, local_max.x, (corner & 1u) != 0u),
            select(local_min.y, local_max.y, (corner & 2u) != 0u),
            select(local_min.z, local_max.z, (corner & 4u) != 0u)
        );
        let world = model * vec4<f32>(local, 1.0);
        let clip = view_projection * world;
        if (clip.w <= 0.05 || clip.w != clip.w || any(abs(clip.xyz) > vec3<f32>(1.0e30))) {
            return ProjectedBounds(uv_min, uv_max, 0.0, 0u);
        }
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        uv_min = min(uv_min, uv);
        uv_max = max(uv_max, uv);
        nearest_depth = min(nearest_depth, -(view * world).z);
    }
    if (nearest_depth != nearest_depth || nearest_depth <= 0.0) {
        return ProjectedBounds(uv_min, uv_max, nearest_depth, 0u);
    }
    return ProjectedBounds(uv_min, uv_max, nearest_depth, 1u);
}

fn hiz_depth(mip: u32, coordinate: vec2<i32>) -> f32 {
    switch mip {
        case 0u: { return textureLoad(hiz_0, coordinate, 0).r; }
        case 1u: { return textureLoad(hiz_1, coordinate, 0).r; }
        case 2u: { return textureLoad(hiz_2, coordinate, 0).r; }
        case 3u: { return textureLoad(hiz_3, coordinate, 0).r; }
        case 4u: { return textureLoad(hiz_4, coordinate, 0).r; }
        case 5u: { return textureLoad(hiz_5, coordinate, 0).r; }
        case 6u: { return textureLoad(hiz_6, coordinate, 0).r; }
        case 7u: { return textureLoad(hiz_7, coordinate, 0).r; }
        default: { return textureLoad(hiz_8, coordinate, 0).r; }
    }
}

// 0 = proven occluded, 1 = sampled and visible, 2 = uncertain/visible.
fn previous_hiz_group_result(
    local_min: vec3<f32>,
    local_max: vec3<f32>,
    instance: GpuVirtualInstance,
) -> u32 {
    if (hiz_params.extent.w == 0u
        || (instance.instance_info.z & INSTANCE_PREVIOUS_HIZ_ELIGIBLE) == 0u) {
        return 2u;
    }
    let previous = project_hiz_bounds(
        local_min,
        local_max,
        instance.previous_model,
        hiz_params.previous_view_projection,
        hiz_params.previous_view
    );
    let current = project_hiz_bounds(
        local_min,
        local_max,
        instance.model,
        hiz_params.current_view_projection,
        hiz_params.current_view
    );
    if (previous.valid == 0u || current.valid == 0u) { return 2u; }
    if (previous.uv_max.x <= 0.0 || previous.uv_min.x >= 1.0
        || previous.uv_max.y <= 0.0 || previous.uv_min.y >= 1.0
        || current.uv_max.x <= 0.0 || current.uv_min.x >= 1.0
        || current.uv_max.y <= 0.0 || current.uv_min.y >= 1.0) {
        return 2u;
    }
    let minimum_delta = abs(previous.uv_min - current.uv_min);
    let maximum_delta = abs(previous.uv_max - current.uv_max);
    let screen_delta = max(
        max(minimum_delta.x, minimum_delta.y),
        max(maximum_delta.x, maximum_delta.y)
    );
    if (screen_delta > max(hiz_params.thresholds.x, hiz_params.thresholds.y)) {
        return 2u;
    }

    let expansion = hiz_params.thresholds.xy * 2.0;
    let uv_min = clamp(min(previous.uv_min, current.uv_min) - expansion, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_max = clamp(max(previous.uv_max, current.uv_max) + expansion, vec2<f32>(0.0), vec2<f32>(1.0));
    let base_span = max(
        (uv_max.x - uv_min.x) * f32(hiz_params.extent.x),
        (uv_max.y - uv_min.y) * f32(hiz_params.extent.y)
    );
    var mip = 0u;
    var span = base_span;
    while (span > 2.0 && mip + 1u < hiz_params.extent.z) {
        span *= 0.5;
        mip++;
    }
    let divisor = 1u << mip;
    let dimensions = max(
        vec2<u32>(1u),
        (hiz_params.extent.xy + vec2<u32>(divisor - 1u)) / divisor
    );
    let maximum_coordinate = vec2<i32>(dimensions - vec2<u32>(1u));
    let first = clamp(vec2<i32>(floor(uv_min * vec2<f32>(dimensions))), vec2<i32>(0), maximum_coordinate);
    let last = clamp(vec2<i32>(floor(uv_max * vec2<f32>(dimensions))), vec2<i32>(0), maximum_coordinate);
    var maximum_depth = 0.0;
    for (var y = first.y; y <= last.y; y++) {
        for (var x = first.x; x <= last.x; x++) {
            maximum_depth = max(maximum_depth, hiz_depth(mip, vec2<i32>(x, y)));
        }
    }
    let nearest_depth = min(previous.nearest_depth, current.nearest_depth);
    let occluded = nearest_depth
        > maximum_depth * (1.0 + hiz_params.thresholds.z) + hiz_params.thresholds.w;
    return select(1u, 0u, occluded);
}

fn cluster_frustum_outside_mask(
    cluster: GpuVirtualClusterEntry,
    instance: GpuVirtualInstance,
    sphere: WorldSphere,
) -> u32 {
    let local_center = (cluster.aabb_min_error.xyz + cluster.aabb_max_radius.xyz) * 0.5;
    let local_extent = (cluster.aabb_max_radius.xyz - cluster.aabb_min_error.xyz) * 0.5;
    let world_center = (instance.model * vec4<f32>(local_center, 1.0)).xyz;
    var outside_mask = 0u;
    for (var plane_index = 0u; plane_index < 6u; plane_index++) {
        let plane = params.planes[plane_index];
        let projected_radius = dot(
            abs(vec3<f32>(
                dot(plane.xyz, instance.model[0].xyz),
                dot(plane.xyz, instance.model[1].xyz),
                dot(plane.xyz, instance.model[2].xyz)
            )),
            local_extent
        );
        if (dot(plane.xyz, world_center) + plane.w < -projected_radius) {
            outside_mask |= 1u << plane_index;
        }
    }
    let clip = params.view_projection * vec4<f32>(sphere.center, 1.0);
    let w_gradient = vec3<f32>(
        params.view_projection[0].w,
        params.view_projection[1].w,
        params.view_projection[2].w
    );
    let nearest_w = clip.w - sphere.radius * length(w_gradient);
    if (nearest_w <= params.thresholds.z) {
        outside_mask &= 0x30u;
    }
    return outside_mask;
}

fn projected_error(
    cluster: GpuVirtualClusterEntry,
    sphere: WorldSphere,
    scale: f32,
) -> f32 {
    let world_error = cluster.aabb_min_error.w * scale;
    if (world_error <= 0.0) {
        return 0.0;
    }
    let clip = params.view_projection * vec4<f32>(sphere.center, 1.0);
    let clip_w_gradient = vec3<f32>(
        params.view_projection[0].w,
        params.view_projection[1].w,
        params.view_projection[2].w
    );
    let nearest_w = clip.w - sphere.radius * length(clip_w_gradient);
    // Preserve refinement at the near plane without collapsing every
    // near-intersecting group onto one source-order tie.
    return world_error * params.camera_projection.w / max(nearest_w, params.thresholds.z);
}

fn cone_culled(
    cluster: GpuVirtualClusterEntry,
    instance: GpuVirtualInstance,
    sphere: WorldSphere,
) -> bool {
    let cutoff = cluster.normal_cone.w;
    if (cutoff <= 0.0 || (instance.instance_info.z & INSTANCE_CONE_CULL_SAFE) == 0u) {
        return false;
    }
    var axis = vec3<f32>(
        dot(instance.normal_rows[0].xyz, cluster.normal_cone.xyz),
        dot(instance.normal_rows[1].xyz, cluster.normal_cone.xyz),
        dot(instance.normal_rows[2].xyz, cluster.normal_cone.xyz)
    );
    let axis_length = length(axis);
    let to_camera = params.camera_projection.xyz - sphere.center;
    let distance = length(to_camera);
    if (axis_length <= 1.0e-8 || distance <= sphere.radius || distance <= 1.0e-8) {
        return false;
    }
    axis /= axis_length;
    let view_direction = to_camera / distance;
    let sin_theta = sqrt(max(1.0 - cutoff * cutoff, 0.0));
    let sin_phi = clamp(sphere.radius / distance, 0.0, 1.0);
    let cos_phi = sqrt(max(1.0 - sin_phi * sin_phi, 0.0));
    let conservative_threshold = -(sin_theta * cos_phi + cutoff * sin_phi);
    return dot(axis, view_direction) <= conservative_threshold;
}

fn group_is_resident(mesh: GpuVirtualMeshEntry, first: u32, count: u32) -> bool {
    for (var offset = 0u; offset < count; offset++) {
        let local_cluster = first + offset;
        if (!valid_cluster(mesh, local_cluster)) {
            return false;
        }
        let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
        let page_index = cluster.page_lod_counts.x;
        if (!valid_page(mesh, page_index)) {
            return false;
        }
        let page = pages.records[mesh.page_table_base + page_index];
        if (page.slot_plus_one == 0u || page.mesh_id != mesh.mesh_id || (page.flags & 1u) == 0u) {
            return false;
        }
    }
    return true;
}

fn emit_missing_requests(
    mesh: GpuVirtualMeshEntry,
    first: u32,
    count: u32,
    priority_bits: u32,
) {
    for (var offset = 0u; offset < count; offset++) {
        let local_cluster = first + offset;
        if (!valid_cluster(mesh, local_cluster)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
        let page_index = cluster.page_lod_counts.x;
        if (!valid_page(mesh, page_index)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let page = pages.records[mesh.page_table_base + page_index];
        if (page.slot_plus_one != 0u && page.mesh_id == mesh.mesh_id && (page.flags & 1u) != 0u) {
            continue;
        }
        var duplicate = false;
        for (var previous = 0u; previous < offset; previous++) {
            let previous_cluster = clusters.records[mesh.cluster_table_base + first + previous];
            if (previous_cluster.page_lod_counts.x == page_index) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            let output_index = atomicAdd(&counters.page_request_count, 1u);
            if (output_index < params.dispatch.w) {
                requests.records[output_index] = GpuVirtualPageRequest(
                    mesh.mesh_id,
                    page_index,
                    priority_bits,
                    first
                );
            } else {
                atomicAdd(&counters.request_overflow, 1u);
            }
        }
    }
}

fn select_group(
    mesh: GpuVirtualMeshEntry,
    instance_index: u32,
    instance: GpuVirtualInstance,
    first: u32,
    count: u32,
    scale: f32,
    priority_bits: u32,
) {
    // One final group identifies the entire selected hierarchy path. The CPU
    // owns the validated archive metadata and protects this group plus every
    // ancestor, avoiding an atomic feedback write for every intermediate group.
    let page_use_index = atomicAdd(&counters.page_use_count, 1u);
    if (page_use_index < params.dispatch.w) {
        page_uses.records[page_use_index] = GpuVirtualPageUse(
            mesh.mesh_id,
            first,
            priority_bits
        );
    } else {
        atomicAdd(&counters.page_use_overflow, 1u);
    }
    for (var offset = 0u; offset < count; offset++) {
        let local_cluster = first + offset;
        if (!valid_cluster(mesh, local_cluster)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
        let sphere = world_sphere(cluster, instance, scale);
        if (cluster_frustum_outside_mask(cluster, instance, sphere) != 0u) {
            continue;
        }
        if (cone_culled(cluster, instance, sphere)) {
            atomicAdd(&counters.cone_culled_clusters, 1u);
            continue;
        }
        let page_index = cluster.page_lod_counts.x;
        if (!valid_page(mesh, page_index)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let page = pages.records[mesh.page_table_base + page_index];
        if (page.slot_plus_one == 0u || page.mesh_id != mesh.mesh_id || (page.flags & 1u) == 0u) {
            atomicAdd(&counters.missing_current_pages, 1u);
            emit_missing_requests(mesh, local_cluster, 1u, priority_bits);
            continue;
        }
        let output_index = atomicAdd(&counters.selected_count, 1u);
        if (output_index < params.dispatch.z) {
            selected.records[output_index] = GpuSelectedVirtualCluster(
                mesh.mesh_id,
                instance_index,
                mesh.cluster_table_base + local_cluster,
                (page.slot_plus_one - 1u) * mesh.page_stride_bytes,
                cluster.page_lod_counts.y,
                cluster.page_lod_counts.w,
                cluster.identity.z,
                (cluster.identity.w & ~VIRTUAL_CLUSTER_UNIFORM_CHILD_RANGE)
                    | (mesh.vertex_encoding << 28u)
            );
        } else {
            atomicAdd(&counters.selected_overflow, 1u);
        }
    }
}

@compute @workgroup_size(64)
fn select_virtual_clusters(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_index = gid.y;
    let root_ordinal = gid.x;
    if (instance_index >= params.dispatch.x || root_ordinal >= params.dispatch.y) {
        return;
    }
    if (instance_index >= arrayLength(&instances.records)) {
        atomicAdd(&counters.invalid_records, 1u);
        return;
    }
    let instance = instances.records[instance_index];
    if (root_ordinal >= instance.root_span.y) {
        return;
    }
    let root_index = instance.root_span.x + root_ordinal;
    let descriptor_index = instance.instance_info.x & 0xfffffu;
    if (descriptor_index == 0u || descriptor_index - 1u >= arrayLength(&meshes.records)) {
        atomicAdd(&counters.invalid_records, 1u);
        return;
    }
    let mesh = meshes.records[descriptor_index - 1u];
    if (mesh.mesh_id != instance.instance_info.x || root_index >= mesh.root_cluster_count) {
        return;
    }
    if (!valid_cluster(mesh, root_index)) {
        atomicAdd(&counters.invalid_records, 1u);
        return;
    }

    let root = clusters.records[mesh.cluster_table_base + root_index];
    let source_mesh_filter = instance.instance_info.w;
    if (source_mesh_filter != ALL_SOURCE_MESHES && root.identity.x != source_mesh_filter) {
        return;
    }
    var root_group_first = root_index;
    var root_group_count = 1u;
    if (root.relations.w != 0u && valid_cluster(mesh, root.relations.z)) {
        let first_child = clusters.records[mesh.cluster_table_base + root.relations.z];
        if (first_child.relations.x != NO_RELATION && first_child.relations.y != 0u) {
            root_group_first = first_child.relations.x;
            root_group_count = first_child.relations.y;
        }
    }
    if (root_index != root_group_first) {
        return;
    }

    let scale = scale_bound(instance.model);
    // Hierarchies branch whenever one coarse group replaces several lower
    // atomic ranges. A bounded depth-first stack follows every branch in this
    // invocation. Stack overflow fails closed to the complete resident parent,
    // preserving coverage without another pass or allocation.
    var group_stack: array<vec4<u32>, TRAVERSAL_GROUP_STACK_CAPACITY>;
    var stack_count = 1u;
    group_stack[0] = vec4<u32>(
        root_group_first,
        root_group_count,
        bitcast<u32>(1.0e30),
        0u
    );
    loop {
        if (stack_count == 0u) {
            break;
        }
        stack_count -= 1u;
        let pending = group_stack[stack_count];
        let group_first = pending.x;
        let group_count = pending.y;
        let group_priority_bits = pending.z;
        let depth = pending.w;
        if (depth >= params.limits.x) {
            atomicAdd(&counters.depth_limit_fallbacks, 1u);
            select_group(
                mesh,
                instance_index,
                instance,
                group_first,
                group_count,
                scale,
                group_priority_bits
            );
            continue;
        }
        if (group_count == 0u || group_count > params.limits.y
            || group_first + group_count > mesh.cluster_count) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }

        var common_outside_mask = 0x3fu;
        var has_intersecting_cluster = false;
        var occlusion_visible = hiz_params.extent.w == 0u;
        var occlusion_uncertain = false;
        var intersecting_error = 0.0;
        var group_error = 0.0;
        var intersecting_min = vec3<f32>(1.0e30);
        var intersecting_max = vec3<f32>(-1.0e30);
        for (var offset = 0u; offset < group_count; offset++) {
            let local_cluster = group_first + offset;
            if (!valid_cluster(mesh, local_cluster)) {
                atomicAdd(&counters.invalid_records, 1u);
                continue;
            }
            let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
            let sphere = world_sphere(cluster, instance, scale);
            let outside_mask = cluster_frustum_outside_mask(cluster, instance, sphere);
            let error = projected_error(cluster, sphere, scale);
            common_outside_mask &= outside_mask;
            group_error = max(group_error, error);
            if (outside_mask == 0u) {
                has_intersecting_cluster = true;
                intersecting_error = max(intersecting_error, error);
                intersecting_min = min(intersecting_min, cluster.aabb_min_error.xyz);
                intersecting_max = max(intersecting_max, cluster.aabb_max_radius.xyz);
            }
        }
        if (common_outside_mask != 0u) {
            atomicAdd(&counters.frustum_culled_groups, 1u);
            continue;
        }
        let maximum_error = select(group_error, intersecting_error, has_intersecting_cluster);
        let refinement_priority_bits = bitcast<u32>(max(maximum_error, 0.0));
        if (hiz_params.extent.w != 0u) {
            if (has_intersecting_cluster) {
                let hiz_result = previous_hiz_group_result(
                    intersecting_min,
                    intersecting_max,
                    instance
                );
                occlusion_visible = hiz_result != 0u;
                occlusion_uncertain = hiz_result == 2u;
            } else {
                occlusion_visible = true;
                occlusion_uncertain = true;
            }
        }
        if (!occlusion_visible) {
            atomicAdd(&counters.occlusion_culled_groups, 1u);
            continue;
        }
        if (occlusion_uncertain) {
            atomicAdd(&counters.occlusion_uncertain_groups, 1u);
        }
        atomicAdd(&counters.visible_groups, 1u);
        let first_cluster = clusters.records[mesh.cluster_table_base + group_first];
        let uniform_child_range =
            (first_cluster.identity.w & VIRTUAL_CLUSTER_UNIFORM_CHILD_RANGE) != 0u;
        var child_group_count = 0u;
        var previous_child_first = NO_RELATION;
        var previous_child_count = 0u;
        var has_children = false;
        var has_terminal_clusters = false;
        var children_valid = true;
        var children_resident = true;
        if (uniform_child_range) {
            let child_first = first_cluster.relations.z;
            let child_count = first_cluster.relations.w;
            if (child_first == NO_RELATION || child_count == 0u) {
                has_terminal_clusters = true;
            } else if (child_count > params.limits.y
                || child_first + child_count > mesh.cluster_count) {
                children_valid = false;
            } else {
                has_children = true;
                child_group_count += 1u;
                children_resident = group_is_resident(mesh, child_first, child_count);
                previous_child_first = child_first;
                previous_child_count = child_count;
            }
        } else {
            for (var offset = 0u; offset < group_count; offset++) {
                let cluster = clusters.records[mesh.cluster_table_base + group_first + offset];
                let child_first = cluster.relations.z;
                let child_count = cluster.relations.w;
                if (child_first == NO_RELATION || child_count == 0u) {
                    has_terminal_clusters = true;
                    continue;
                }
                has_children = true;
                if (child_count > params.limits.y
                    || child_first + child_count > mesh.cluster_count) {
                    children_valid = false;
                    continue;
                }
                if (child_first != previous_child_first || child_count != previous_child_count) {
                    child_group_count += 1u;
                    children_resident = children_resident
                        && group_is_resident(mesh, child_first, child_count);
                    previous_child_first = child_first;
                    previous_child_count = child_count;
                }
            }
        }
        if (!children_valid || (has_children && has_terminal_clusters)) {
            atomicAdd(&counters.invalid_records, 1u);
            select_group(
                mesh,
                instance_index,
                instance,
                group_first,
                group_count,
                scale,
                group_priority_bits
            );
            continue;
        }
        let wants_refinement = maximum_error > params.thresholds.x && has_children;
        if (!wants_refinement) {
            select_group(
                mesh,
                instance_index,
                instance,
                group_first,
                group_count,
                scale,
                group_priority_bits
            );
            continue;
        }
        if (!children_resident) {
            atomicAdd(&counters.fallback_groups, 1u);
            if (uniform_child_range) {
                emit_missing_requests(
                    mesh,
                    first_cluster.relations.z,
                    first_cluster.relations.w,
                    refinement_priority_bits
                );
            } else {
                previous_child_first = NO_RELATION;
                previous_child_count = 0u;
                for (var offset = 0u; offset < group_count; offset++) {
                    let cluster = clusters.records[mesh.cluster_table_base + group_first + offset];
                    let child_first = cluster.relations.z;
                    let child_count = cluster.relations.w;
                    if (child_first != NO_RELATION && child_count != 0u
                        && (child_first != previous_child_first
                            || child_count != previous_child_count)) {
                        emit_missing_requests(
                            mesh,
                            child_first,
                            child_count,
                            refinement_priority_bits
                        );
                        previous_child_first = child_first;
                        previous_child_count = child_count;
                    }
                }
            }
            select_group(
                mesh,
                instance_index,
                instance,
                group_first,
                group_count,
                scale,
                group_priority_bits
            );
            continue;
        }
        if (stack_count + child_group_count > TRAVERSAL_GROUP_STACK_CAPACITY) {
            atomicAdd(&counters.depth_limit_fallbacks, 1u);
            select_group(
                mesh,
                instance_index,
                instance,
                group_first,
                group_count,
                scale,
                group_priority_bits
            );
            continue;
        }
        atomicAdd(&counters.refined_groups, 1u);
        if (uniform_child_range) {
            group_stack[stack_count] = vec4<u32>(
                first_cluster.relations.z,
                first_cluster.relations.w,
                refinement_priority_bits,
                depth + 1u
            );
            stack_count += 1u;
        } else {
            previous_child_first = NO_RELATION;
            previous_child_count = 0u;
            for (var offset = 0u; offset < group_count; offset++) {
                let cluster = clusters.records[mesh.cluster_table_base + group_first + offset];
                let child_first = cluster.relations.z;
                let child_count = cluster.relations.w;
                if (child_first != NO_RELATION && child_count != 0u
                    && (child_first != previous_child_first
                        || child_count != previous_child_count)) {
                    group_stack[stack_count] = vec4<u32>(
                        child_first,
                        child_count,
                        refinement_priority_bits,
                        depth + 1u
                    );
                    stack_count += 1u;
                    previous_child_first = child_first;
                    previous_child_count = child_count;
                }
            }
        }
    }
}
