//! Deterministic, crack-safe coarse hierarchy construction.
//!
//! A hierarchy edge replaces one contiguous child group with one contiguous
//! parent group. Simplification locks the topological border of the complete
//! group, so traversal can switch the group atomically without opening cracks.
//! Parents keep the full child-group bounds and accumulated absolute error for
//! conservative culling and future projected-pixel selection.

use crate::meshlet::{
    build_leaf_meshlets, Meshlet, MeshletBounds, MeshletLimits, StaticPrimitive, StaticVertex,
    FLAG_COARSE_ROOT, NO_RELATION,
};
use meshopt::{SimplifyOptions, VertexDataAdapter};
use std::cmp::Reverse;
use std::collections::BTreeMap;

const GROUP_FANOUT: usize = 8;
const ATTRIBUTE_STRIDE_FLOATS: usize = 15;
const ATTRIBUTE_WEIGHTS: [f32; ATTRIBUTE_STRIDE_FLOATS] = [
    0.5, 0.5, 0.5, // normal
    0.25, 0.25, 0.25, 0.25, // tangent + handedness
    10.0, 10.0, // UV0
    10.0, 10.0, // UV1
    1.0, 1.0, 1.0, 1.0, // color
];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HierarchyStats {
    pub leaf_clusters: u32,
    pub leaf_triangles: u64,
    pub leaf_payload_bytes: u64,
    pub parent_clusters: u32,
    pub root_clusters: u32,
    pub root_triangles: u64,
    pub maximum_level: u32,
    pub maximum_error: f32,
    pub root_payload_bytes: u64,
    pub root_clusters_by_level: [u32; 17],
    pub root_payload_bytes_by_level: [u64; 17],
}

impl HierarchyStats {
    pub fn merge(&mut self, other: Self) {
        self.leaf_clusters += other.leaf_clusters;
        self.leaf_triangles += other.leaf_triangles;
        self.leaf_payload_bytes += other.leaf_payload_bytes;
        self.parent_clusters += other.parent_clusters;
        self.root_clusters += other.root_clusters;
        self.root_triangles += other.root_triangles;
        self.maximum_level = self.maximum_level.max(other.maximum_level);
        self.maximum_error = self.maximum_error.max(other.maximum_error);
        self.root_payload_bytes += other.root_payload_bytes;
        for level in 0..self.root_clusters_by_level.len() {
            self.root_clusters_by_level[level] += other.root_clusters_by_level[level];
            self.root_payload_bytes_by_level[level] += other.root_payload_bytes_by_level[level];
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct VertexKey([u32; 18]);

struct CombinedMesh {
    vertices: Vec<StaticVertex>,
    indices: Vec<u32>,
}

#[derive(Clone, Copy)]
struct ClusterGroup {
    start: usize,
    count: usize,
}

impl ClusterGroup {
    fn end(self) -> Result<usize, String> {
        self.start
            .checked_add(self.count)
            .ok_or("cluster group range overflow".to_string())
    }
}

pub fn build_spatial_leaf_meshlets(
    primitive: &StaticPrimitive,
    limits: MeshletLimits,
) -> Result<Vec<Meshlet>, String> {
    let limits = limits.validate()?;
    let maximum_triangles = (limits.max_triangles / 4) * 4;
    if maximum_triangles < 4 {
        return build_leaf_meshlets(primitive, limits);
    }
    let position_bytes = encode_positions(&primitive.vertices);
    let adapter = VertexDataAdapter::new(&position_bytes, 12, 0)
        .map_err(|error| format!("create meshoptimizer leaf adapter: {error}"))?;
    let optimized = meshopt::build_meshlets(
        &primitive.indices,
        &adapter,
        limits.max_vertices as usize,
        maximum_triangles as usize,
        0.5,
    );
    if optimized.is_empty() {
        return Err(format!(
            "meshoptimizer produced no leaf clusters for mesh {} primitive {}",
            primitive.mesh_index, primitive.primitive_index
        ));
    }
    let mut leaves = Vec::with_capacity(optimized.len());
    for optimized_meshlet in optimized.iter() {
        let vertices: Vec<_> = optimized_meshlet
            .vertices
            .iter()
            .map(|index| primitive.vertices[*index as usize])
            .collect();
        let local = StaticPrimitive {
            mesh_index: primitive.mesh_index,
            primitive_index: primitive.primitive_index,
            material_index: primitive.material_index,
            double_sided: primitive.double_sided,
            alpha_masked: primitive.alpha_masked,
            vertices,
            indices: optimized_meshlet
                .triangles
                .iter()
                .map(|index| *index as u32)
                .collect(),
        };
        let mut built = build_leaf_meshlets(&local, limits)?;
        if built.len() != 1 {
            return Err("optimized leaf exceeded the configured meshlet limits".to_string());
        }
        leaves.push(built.remove(0));
    }
    Ok(leaves)
}

pub fn build_meshlet_hierarchy(
    leaves: Vec<Meshlet>,
    limits: MeshletLimits,
    maximum_levels: u32,
) -> Result<(Vec<Meshlet>, HierarchyStats), String> {
    let limits = limits.validate()?;
    if leaves.is_empty() {
        return Ok((leaves, HierarchyStats::default()));
    }
    validate_same_primitive(&leaves)?;

    let leaf_count = leaves.len();
    let leaf_triangles = leaves
        .iter()
        .map(|meshlet| meshlet.triangle_count() as u64)
        .sum();
    let leaf_payload_bytes = leaves
        .iter()
        .map(|meshlet| meshlet.encoded_payload_bytes() as u64)
        .sum();
    let mut nodes = leaves;
    let mut current: Vec<_> = (0..leaf_count)
        .map(|start| ClusterGroup { start, count: 1 })
        .collect();
    let mut roots = Vec::<usize>::new();
    let mut maximum_level = 0;

    for level in 1..=maximum_levels {
        if current.is_empty() {
            break;
        }
        let mut next = Vec::<ClusterGroup>::new();
        let partitions = partition_atomic_groups(&current, &nodes)?;
        for child_group in partitions {
            if child_group.count == 1 {
                roots.push(child_group.start);
                continue;
            }
            let child_end = child_group.end()?;
            let children = &nodes[child_group.start..child_end];
            let Some(mut parents) = simplify_group(children, limits, level)? else {
                roots.extend(child_group.start..child_end);
                continue;
            };
            if parents.len() >= child_group.count {
                roots.extend(child_group.start..child_end);
                continue;
            }

            coalesce_atomic_replacement(&mut nodes, child_group)?;

            let parent_start = nodes.len();
            let parent_count =
                u32::try_from(parents.len()).map_err(|_| "parent group count exceeds u32")?;
            let child_start =
                u32::try_from(child_group.start).map_err(|_| "child group start exceeds u32")?;
            let child_count =
                u32::try_from(child_group.count).map_err(|_| "child group count exceeds u32")?;
            let parent_start_u32 =
                u32::try_from(parent_start).map_err(|_| "parent group start exceeds u32")?;
            for child in &mut nodes[child_group.start..child_end] {
                child.parent = parent_start_u32;
                child.parent_count = parent_count;
                child.flags &= !FLAG_COARSE_ROOT;
            }
            for parent in &mut parents {
                parent.first_child = child_start;
                parent.child_count = child_count;
            }
            let count = parents.len();
            nodes.extend(parents);
            next.push(ClusterGroup {
                start: parent_start,
                count,
            });
            maximum_level = level;
        }
        current = next;
    }

    for group in current {
        roots.extend(group.start..group.end()?);
    }
    roots.sort_unstable();
    roots.dedup();
    for root in &roots {
        nodes[*root].flags |= FLAG_COARSE_ROOT;
        debug_assert_eq!(nodes[*root].parent, NO_RELATION);
        debug_assert_eq!(nodes[*root].parent_count, 0);
    }

    let maximum_error = nodes
        .iter()
        .map(|meshlet| meshlet.geometric_error)
        .fold(0.0, f32::max);
    let root_payload_bytes = roots
        .iter()
        .map(|index| nodes[*index].encoded_payload_bytes() as u64)
        .sum();
    let root_triangles = roots
        .iter()
        .map(|index| nodes[*index].triangle_count() as u64)
        .sum();
    let mut root_clusters_by_level = [0; 17];
    let mut root_payload_bytes_by_level = [0; 17];
    for root in &roots {
        let level = nodes[*root].lod_level as usize;
        root_clusters_by_level[level] += 1;
        root_payload_bytes_by_level[level] += nodes[*root].encoded_payload_bytes() as u64;
    }
    let parent_clusters = (nodes.len() - leaf_count) as u32;
    Ok((
        nodes,
        HierarchyStats {
            leaf_clusters: leaf_count as u32,
            leaf_triangles,
            leaf_payload_bytes,
            parent_clusters,
            root_clusters: roots.len() as u32,
            root_triangles,
            maximum_level,
            maximum_error,
            root_payload_bytes,
            root_clusters_by_level,
            root_payload_bytes_by_level,
        },
    ))
}

/// Once several lower groups become the atomic child side of one hierarchy
/// edge, refining that side must replace the complete group again. Preserve
/// the union of their lower replacement ranges on every sibling and point
/// those grandchildren back at the complete atomic group. Without this
/// coalescing, multi-level traversal follows the first sibling's children and
/// silently drops the remaining siblings during the next refinement.
fn coalesce_atomic_replacement(nodes: &mut [Meshlet], group: ClusterGroup) -> Result<(), String> {
    let group_end = group.end()?;
    let mut ranges = nodes[group.start..group_end]
        .iter()
        .filter_map(|node| {
            (node.first_child != NO_RELATION && node.child_count != 0)
                .then_some((node.first_child as usize, node.child_count as usize))
        })
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return Ok(());
    }
    if ranges.len() != group.count {
        return Err("atomic child group mixes terminal and refinable clusters".to_string());
    }
    ranges.sort_unstable();
    ranges.dedup();
    let replacement_start = ranges[0].0;
    let mut replacement_end = replacement_start;
    for (start, count) in ranges {
        if start != replacement_end {
            return Err("atomic child replacement ranges are not contiguous".to_string());
        }
        replacement_end = start
            .checked_add(count)
            .ok_or("atomic child replacement range overflow".to_string())?;
    }
    if replacement_end > nodes.len() {
        return Err("atomic child replacement exceeds hierarchy nodes".to_string());
    }
    let replacement_count = replacement_end - replacement_start;
    let replacement_start_u32 = u32::try_from(replacement_start)
        .map_err(|_| "atomic child replacement start exceeds u32")?;
    let replacement_count_u32 = u32::try_from(replacement_count)
        .map_err(|_| "atomic child replacement count exceeds u32")?;
    let group_start_u32 =
        u32::try_from(group.start).map_err(|_| "atomic child group start exceeds u32")?;
    let group_count_u32 =
        u32::try_from(group.count).map_err(|_| "atomic child group count exceeds u32")?;
    let shared_error = nodes[group.start..group_end]
        .iter()
        .map(|node| node.geometric_error)
        .fold(0.0, f32::max);
    let shared_bounds = hierarchy_bounds(&nodes[group.start..group_end]);

    for node in &mut nodes[group.start..group_end] {
        node.first_child = replacement_start_u32;
        node.child_count = replacement_count_u32;
        node.geometric_error = shared_error;
        node.bounds.aabb_min = shared_bounds.aabb_min;
        node.bounds.aabb_max = shared_bounds.aabb_max;
        node.bounds.sphere_center = shared_bounds.sphere_center;
        node.bounds.sphere_radius = shared_bounds.sphere_radius;
    }
    for child in &mut nodes[replacement_start..replacement_end] {
        child.parent = group_start_u32;
        child.parent_count = group_count_u32;
    }
    Ok(())
}

/// Offset local relation indices after appending one primitive hierarchy to a
/// file-global cluster array.
pub fn offset_relations(meshlets: &mut [Meshlet], base: usize) -> Result<(), String> {
    let base = u32::try_from(base).map_err(|_| "global cluster base exceeds u32")?;
    for meshlet in meshlets {
        if meshlet.parent != NO_RELATION {
            meshlet.parent = meshlet
                .parent
                .checked_add(base)
                .ok_or("global parent index overflow")?;
        }
        if meshlet.first_child != NO_RELATION {
            meshlet.first_child = meshlet
                .first_child
                .checked_add(base)
                .ok_or("global child index overflow")?;
        }
    }
    Ok(())
}

/// Put all coarse roots first and group them by source mesh so filtered runtime
/// instances can dispatch only their compact root range. Non-root clusters stay
/// coarse-to-fine, and every atomic relation range remains contiguous.
pub fn order_for_streaming(meshlets: &mut Vec<Meshlet>) -> Result<(), String> {
    if meshlets.len() <= 1 {
        return Ok(());
    }
    let mut order: Vec<_> = (0..meshlets.len()).collect();
    order.sort_by_key(|index| {
        let meshlet = &meshlets[*index];
        let coarse_root = meshlet.flags & FLAG_COARSE_ROOT != 0;
        (
            u8::from(!coarse_root),
            if coarse_root { meshlet.mesh_index } else { 0 },
            Reverse(meshlet.lod_level),
            if coarse_root { 0 } else { meshlet.mesh_index },
            meshlet.primitive_index,
            meshlet.material_index,
            *index,
        )
    });
    let mut new_index = vec![0usize; meshlets.len()];
    for (new, old) in order.iter().enumerate() {
        new_index[*old] = new;
    }
    let mut reordered: Vec<_> = order.iter().map(|old| meshlets[*old].clone()).collect();
    for (new, old) in order.iter().enumerate() {
        let source = &meshlets[*old];
        if source.parent != NO_RELATION {
            reordered[new].parent =
                remap_contiguous_range(source.parent, source.parent_count, &new_index, "parent")?;
        }
        if source.first_child != NO_RELATION {
            reordered[new].first_child = remap_contiguous_range(
                source.first_child,
                source.child_count,
                &new_index,
                "child",
            )?;
        }
    }
    *meshlets = reordered;
    Ok(())
}

fn remap_contiguous_range(
    old_start: u32,
    count: u32,
    new_index: &[usize],
    label: &str,
) -> Result<u32, String> {
    let old_start = old_start as usize;
    let old_end = old_start
        .checked_add(count as usize)
        .ok_or_else(|| format!("{label} relation range overflow during streaming order"))?;
    if count == 0 || old_end > new_index.len() {
        return Err(format!(
            "{label} relation range exceeds cluster table during streaming order"
        ));
    }
    let new_start = new_index[old_start];
    for (ordinal, mapped) in new_index[old_start..old_end].iter().enumerate() {
        if *mapped != new_start + ordinal {
            return Err(format!(
                "{label} relation group lost contiguity during streaming order"
            ));
        }
    }
    u32::try_from(new_start).map_err(|_| format!("{label} relation start exceeds u32"))
}

fn partition_atomic_groups(
    groups: &[ClusterGroup],
    nodes: &[Meshlet],
) -> Result<Vec<ClusterGroup>, String> {
    let mut partitions = Vec::new();
    let mut begin = 0usize;
    while begin < groups.len() {
        let mut end = begin + 1;
        while end < groups.len() && end - begin < GROUP_FANOUT {
            let previous = group_replacement_range(groups[end - 1], nodes)?;
            let next = group_replacement_range(groups[end], nodes)?;
            let replacement_contiguous = match (previous, next) {
                (None, None) => true,
                (Some((_, previous_end)), Some((next_start, _))) => previous_end == next_start,
                _ => false,
            };
            if !replacement_contiguous {
                break;
            }
            end += 1;
        }
        partitions.push(merge_contiguous_groups(&groups[begin..end])?);
        begin = end;
    }
    Ok(partitions)
}

fn group_replacement_range(
    group: ClusterGroup,
    nodes: &[Meshlet],
) -> Result<Option<(usize, usize)>, String> {
    let end = group.end()?;
    let first = nodes
        .get(group.start)
        .ok_or("atomic group start exceeds hierarchy nodes".to_string())?;
    if first.first_child == NO_RELATION || first.child_count == 0 {
        if nodes[group.start..end]
            .iter()
            .any(|node| node.first_child != NO_RELATION || node.child_count != 0)
        {
            return Err("atomic group mixes terminal and refinable siblings".to_string());
        }
        return Ok(None);
    }
    if nodes[group.start..end]
        .iter()
        .any(|node| node.first_child != first.first_child || node.child_count != first.child_count)
    {
        return Err("atomic siblings disagree on their replacement range".to_string());
    }
    let start = first.first_child as usize;
    let replacement_end = start
        .checked_add(first.child_count as usize)
        .ok_or("atomic replacement range overflow".to_string())?;
    if replacement_end > nodes.len() {
        return Err("atomic replacement range exceeds hierarchy nodes".to_string());
    }
    Ok(Some((start, replacement_end)))
}

fn merge_contiguous_groups(groups: &[ClusterGroup]) -> Result<ClusterGroup, String> {
    let first = *groups
        .first()
        .ok_or("cannot merge an empty cluster-group set")?;
    let mut end = first.end()?;
    for group in &groups[1..] {
        if group.start != end {
            return Err("hierarchy child groups are not contiguous".to_string());
        }
        end = group.end()?;
    }
    Ok(ClusterGroup {
        start: first.start,
        count: end - first.start,
    })
}

fn validate_same_primitive(meshlets: &[Meshlet]) -> Result<(), String> {
    let first = &meshlets[0];
    if meshlets.iter().any(|meshlet| {
        meshlet.mesh_index != first.mesh_index
            || meshlet.primitive_index != first.primitive_index
            || meshlet.material_index != first.material_index
            || (meshlet.flags & !FLAG_COARSE_ROOT) != (first.flags & !FLAG_COARSE_ROOT)
    }) {
        return Err(
            "a hierarchy may not cross mesh, primitive, material, or flag boundaries".to_string(),
        );
    }
    Ok(())
}

fn simplify_group(
    children: &[Meshlet],
    limits: MeshletLimits,
    level: u32,
) -> Result<Option<Vec<Meshlet>>, String> {
    let combined = combine_children(children);
    let target_triangles = (combined.indices.len() / 6).max(1);
    let position_bytes = encode_positions(&combined.vertices);
    let adapter = VertexDataAdapter::new(&position_bytes, 12, 0)
        .map_err(|error| format!("create meshoptimizer position adapter: {error}"))?;
    let attributes = encode_attributes(&combined.vertices);
    let vertex_locks = vec![false; combined.vertices.len()];
    let mut added_error = 0.0;
    let simplified = meshopt::simplify_with_attributes_and_locks(
        &combined.indices,
        &adapter,
        &attributes,
        &ATTRIBUTE_WEIGHTS,
        ATTRIBUTE_STRIDE_FLOATS * std::mem::size_of::<f32>(),
        &vertex_locks,
        target_triangles * 3,
        f32::MAX,
        SimplifyOptions::LockBorder | SimplifyOptions::ErrorAbsolute,
        Some(&mut added_error),
    );
    if simplified.is_empty()
        || !simplified.len().is_multiple_of(3)
        || simplified.len() >= combined.indices.len()
        || !added_error.is_finite()
        || added_error < 0.0
    {
        return Ok(None);
    }

    let first = &children[0];
    let primitive = StaticPrimitive {
        mesh_index: first.mesh_index,
        primitive_index: first.primitive_index,
        material_index: first.material_index,
        double_sided: first.flags & crate::meshlet::FLAG_DOUBLE_SIDED != 0,
        alpha_masked: first.flags & crate::meshlet::FLAG_ALPHA_MASKED != 0,
        vertices: combined.vertices,
        indices: simplified,
    };
    let mut parents = build_spatial_leaf_meshlets(&primitive, limits)?;
    if parents.is_empty() {
        return Ok(None);
    }
    let error = children
        .iter()
        .map(|child| child.geometric_error)
        .fold(0.0, f32::max)
        + added_error;
    let bounds = hierarchy_bounds(children);
    for parent in &mut parents {
        parent.lod_level = level;
        parent.geometric_error = error;
        parent.bounds.aabb_min = bounds.aabb_min;
        parent.bounds.aabb_max = bounds.aabb_max;
        parent.bounds.sphere_center = bounds.sphere_center;
        parent.bounds.sphere_radius = bounds.sphere_radius;
    }
    Ok(Some(parents))
}

fn combine_children(children: &[Meshlet]) -> CombinedMesh {
    let mut vertices = Vec::<StaticVertex>::new();
    let mut indices = Vec::<u32>::new();
    let mut remap = BTreeMap::<VertexKey, u32>::new();
    for child in children {
        for local_index in &child.local_indices {
            let vertex = child.vertices[*local_index as usize];
            let key = vertex_key(vertex);
            let global_index = match remap.get(&key) {
                Some(index) => *index,
                None => {
                    let index = vertices.len() as u32;
                    vertices.push(vertex);
                    remap.insert(key, index);
                    index
                }
            };
            indices.push(global_index);
        }
    }
    CombinedMesh { vertices, indices }
}

fn encode_positions(vertices: &[StaticVertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vertices.len() * 12);
    for vertex in vertices {
        for value in vertex.position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn encode_attributes(vertices: &[StaticVertex]) -> Vec<f32> {
    let mut attributes = Vec::with_capacity(vertices.len() * ATTRIBUTE_STRIDE_FLOATS);
    for vertex in vertices {
        attributes.extend_from_slice(&vertex.normal);
        attributes.extend_from_slice(&vertex.tangent);
        attributes.extend_from_slice(&vertex.uv0);
        attributes.extend_from_slice(&vertex.uv1);
        attributes.extend_from_slice(&vertex.color);
    }
    attributes
}

fn hierarchy_bounds(children: &[Meshlet]) -> MeshletBounds {
    let mut bounds = children[0].bounds;
    bounds.aabb_min = [f32::INFINITY; 3];
    bounds.aabb_max = [f32::NEG_INFINITY; 3];
    for child in children {
        for axis in 0..3 {
            bounds.aabb_min[axis] = bounds.aabb_min[axis].min(child.bounds.aabb_min[axis]);
            bounds.aabb_max[axis] = bounds.aabb_max[axis].max(child.bounds.aabb_max[axis]);
        }
    }
    bounds.sphere_center = [
        (bounds.aabb_min[0] + bounds.aabb_max[0]) * 0.5,
        (bounds.aabb_min[1] + bounds.aabb_max[1]) * 0.5,
        (bounds.aabb_min[2] + bounds.aabb_max[2]) * 0.5,
    ];
    bounds.sphere_radius = children
        .iter()
        .map(|child| {
            distance3(bounds.sphere_center, child.bounds.sphere_center) + child.bounds.sphere_radius
        })
        .fold(0.0, f32::max);
    bounds
}

fn vertex_key(vertex: StaticVertex) -> VertexKey {
    let mut bits = [0u32; 18];
    for (cursor, value) in vertex
        .position
        .iter()
        .chain(vertex.normal.iter())
        .chain(vertex.tangent.iter())
        .chain(vertex.uv0.iter())
        .chain(vertex.uv1.iter())
        .chain(vertex.color.iter())
        .enumerate()
    {
        bits[cursor] = value.to_bits();
    }
    VertexKey(bits)
}

fn distance3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let x = a[0] - b[0];
    let y = a[1] - b[1];
    let z = a[2] - b[2];
    (x * x + y * y + z * z).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry_format::{decode_geometry, encode_geometry, sha256, DEFAULT_PAGE_BYTES};

    fn grid_primitive(width: usize, height: usize) -> StaticPrimitive {
        let mut vertices = Vec::new();
        for y in 0..=height {
            for x in 0..=width {
                let xf = x as f32 / width as f32;
                let yf = y as f32 / height as f32;
                vertices.push(StaticVertex {
                    position: [xf, yf, (xf * 4.0).sin() * (yf * 3.0).cos() * 0.05],
                    normal: [0.0, 0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv0: [xf, yf],
                    uv1: [xf, yf],
                    color: [xf, yf, 1.0 - xf, 1.0],
                });
            }
        }
        let stride = width + 1;
        let mut indices = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let a = (y * stride + x) as u32;
                let b = a + 1;
                let c = a + stride as u32;
                let d = c + 1;
                indices.extend([a, b, d, a, d, c]);
            }
        }
        StaticPrimitive {
            mesh_index: 0,
            primitive_index: 0,
            material_index: Some(0),
            double_sided: false,
            alpha_masked: false,
            vertices,
            indices,
        }
    }

    #[test]
    fn hierarchy_is_deterministic_grouped_and_monotonic() {
        let primitive = grid_primitive(32, 32);
        let limits = MeshletLimits {
            max_vertices: 64,
            max_triangles: 64,
        };
        let leaves = build_spatial_leaf_meshlets(&primitive, limits).unwrap();
        let leaf_count = leaves.len();
        let (mut a, stats) = build_meshlet_hierarchy(leaves.clone(), limits, 8).unwrap();
        let (mut b, other_stats) = build_meshlet_hierarchy(leaves, limits, 8).unwrap();
        assert_eq!(stats, other_stats);
        assert!(stats.parent_clusters > 0);
        assert!(stats.root_clusters < leaf_count as u32);
        assert!(stats.maximum_level >= 2);

        order_for_streaming(&mut a).unwrap();
        order_for_streaming(&mut b).unwrap();
        let encoded_a = encode_geometry(&a, &[], sha256(b"grid"), DEFAULT_PAGE_BYTES).unwrap();
        let encoded_b = encode_geometry(&b, &[], sha256(b"grid"), DEFAULT_PAGE_BYTES).unwrap();
        assert_eq!(encoded_a, encoded_b);
        let archive = decode_geometry(&encoded_a).unwrap();
        assert!(archive.coarse_root_page_count() > 0);
        let root_cluster_count: usize = archive.pages[..archive.coarse_root_page_count()]
            .iter()
            .map(|page| page.cluster_count as usize)
            .sum();
        assert_eq!(root_cluster_count, stats.root_clusters as usize);

        for (index, node) in a.iter().enumerate() {
            if node.parent == NO_RELATION {
                assert_eq!(node.parent_count, 0);
                assert_ne!(node.flags & FLAG_COARSE_ROOT, 0);
                continue;
            }
            assert!(node.parent_count > 0);
            let parents =
                &a[node.parent as usize..node.parent as usize + node.parent_count as usize];
            let atomic_children = parents[0].first_child as usize
                ..parents[0].first_child as usize + parents[0].child_count as usize;
            if a[atomic_children.clone()]
                .iter()
                .all(|child| child.child_count != 0)
            {
                let lower = (
                    a[atomic_children.start].first_child,
                    a[atomic_children.start].child_count,
                );
                assert!(a[atomic_children.clone()]
                    .iter()
                    .all(|child| (child.first_child, child.child_count) == lower));
                let lower_range = lower.0 as usize..(lower.0 + lower.1) as usize;
                assert!(a[lower_range].iter().all(|grandchild| {
                    grandchild.parent == parents[0].first_child
                        && grandchild.parent_count == parents[0].child_count
                }));
            }
            for parent in parents {
                assert!(parent.lod_level > node.lod_level);
                let children = parent.first_child as usize
                    ..parent.first_child as usize + parent.child_count as usize;
                assert!(children.contains(&index));
                assert!(parent.geometric_error >= node.geometric_error);
                for axis in 0..3 {
                    assert!(parent.bounds.aabb_min[axis] <= node.bounds.aabb_min[axis]);
                    assert!(parent.bounds.aabb_max[axis] >= node.bounds.aabb_max[axis]);
                }
            }
        }
        assert_eq!(
            a.iter()
                .filter(|node| node.flags & FLAG_COARSE_ROOT != 0)
                .count(),
            stats.root_clusters as usize
        );
    }

    #[test]
    fn streaming_order_groups_source_roots_even_when_lod_levels_differ() {
        let limits = MeshletLimits {
            max_vertices: 64,
            max_triangles: 64,
        };
        let leaves = build_spatial_leaf_meshlets(&grid_primitive(16, 16), limits).unwrap();
        let (mut first, _) = build_meshlet_hierarchy(leaves, limits, 8).unwrap();
        let mut second = first.clone();
        for meshlet in &mut second {
            meshlet.mesh_index = 1;
            if meshlet.flags & FLAG_COARSE_ROOT != 0 {
                meshlet.lod_level += 1;
            }
        }
        offset_relations(&mut second, first.len()).unwrap();
        first.extend(second);
        order_for_streaming(&mut first).unwrap();

        let root_meshes = first
            .iter()
            .take_while(|meshlet| meshlet.flags & FLAG_COARSE_ROOT != 0)
            .map(|meshlet| meshlet.mesh_index)
            .collect::<Vec<_>>();
        assert!(!root_meshes.is_empty());
        assert!(root_meshes.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(root_meshes.contains(&0));
        assert!(root_meshes.contains(&1));
    }

    #[test]
    fn locked_outer_border_survives_coarse_roots() {
        let primitive = grid_primitive(24, 24);
        let limits = MeshletLimits {
            max_vertices: 64,
            max_triangles: 64,
        };
        let leaves = build_spatial_leaf_meshlets(&primitive, limits).unwrap();
        let (nodes, _) = build_meshlet_hierarchy(leaves, limits, 8).unwrap();
        let root_positions: Vec<_> = nodes
            .iter()
            .filter(|node| node.flags & FLAG_COARSE_ROOT != 0)
            .flat_map(|node| node.vertices.iter().map(|vertex| vertex.position))
            .collect();
        let boundary_positions: Vec<_> = primitive
            .vertices
            .iter()
            .filter(|vertex| {
                vertex.position[0] == 0.0
                    || vertex.position[0] == 1.0
                    || vertex.position[1] == 0.0
                    || vertex.position[1] == 1.0
            })
            .map(|vertex| vertex.position)
            .collect();
        for boundary in boundary_positions {
            assert!(
                root_positions.contains(&boundary),
                "coarse hierarchy dropped locked boundary vertex {boundary:?}"
            );
        }
    }
}
