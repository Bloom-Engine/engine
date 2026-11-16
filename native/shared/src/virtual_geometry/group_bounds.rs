//! Which cluster groups can be judged from one cluster.

use std::collections::BTreeSet;

/// For each cluster: whether it starts a group (a parent range or a child
/// range some cluster names) of more than one cluster whose bounds, bounding
/// sphere and error are bit-identical.
pub(super) fn uniform_group_bounds_flags(
    archive: &bloom_geometry_format::GeometryArchive,
) -> Vec<bool> {
    let clusters = &archive.clusters;
    let mut groups = BTreeSet::<(u32, u32)>::new();
    for cluster in clusters {
        if cluster.first_child != bloom_geometry_format::NO_RELATION && cluster.child_count > 1 {
            groups.insert((cluster.first_child, cluster.child_count));
        }
        if cluster.parent != bloom_geometry_format::NO_RELATION && cluster.parent_count > 1 {
            groups.insert((cluster.parent, cluster.parent_count));
        }
    }
    let key = |cluster: &bloom_geometry_format::ClusterRecord| {
        let mut bits = [0u32; 11];
        for axis in 0..3 {
            bits[axis] = cluster.aabb_min[axis].to_bits();
            bits[3 + axis] = cluster.aabb_max[axis].to_bits();
            bits[6 + axis] = cluster.sphere_center[axis].to_bits();
        }
        bits[9] = cluster.sphere_radius.to_bits();
        bits[10] = cluster.geometric_error.to_bits();
        bits
    };
    // The flag sits on a start index, and a parent range and a child range
    // may start at the same cluster with different lengths. It holds only if
    // every range starting there is uniform.
    let mut flags = vec![false; clusters.len()];
    let mut blocked = vec![false; clusters.len()];
    for (first, count) in groups {
        let first = first as usize;
        let Some(group) = first
            .checked_add(count as usize)
            .and_then(|end| clusters.get(first..end))
        else {
            if let Some(start) = blocked.get_mut(first) {
                *start = true;
            }
            continue;
        };
        let reference = key(&group[0]);
        if group.iter().all(|cluster| key(cluster) == reference) {
            flags[first] = true;
        } else {
            blocked[first] = true;
        }
    }
    for (flag, blocked) in flags.iter_mut().zip(blocked) {
        *flag &= !blocked;
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::uniform_group_bounds_flags;
    use bloom_geometry_format::{ClusterRecord, GeometryArchive, VertexEncoding, NO_RELATION};

    fn cluster(
        parent: u32,
        parent_count: u32,
        first_child: u32,
        child_count: u32,
        error: f32,
    ) -> ClusterRecord {
        ClusterRecord {
            mesh_index: 0,
            primitive_index: 0,
            material_index: None,
            flags: 0,
            page_index: 0,
            vertex_count: 3,
            triangle_count: 1,
            lod_level: 0,
            vertex_offset: 0,
            index_offset: 0,
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            sphere_center: [0.5; 3],
            sphere_radius: 1.0,
            normal_cone_axis: [0.0, 0.0, 1.0],
            normal_cone_cutoff: -1.0,
            geometric_error: error,
            parent,
            parent_count,
            first_child,
            child_count,
            vertex_stride: 72,
        }
    }

    fn archive(clusters: Vec<ClusterRecord>) -> GeometryArchive {
        GeometryArchive {
            format_version: 1,
            vertex_encoding: VertexEncoding::Float32,
            source_sha256: [0; 32],
            payload_sha256: [0; 32],
            page_budget_bytes: 4096,
            file_payload_offset: 0,
            clusters,
            pages: Vec::new(),
            compatibility: Vec::new(),
        }
    }

    #[test]
    fn a_group_is_marked_only_when_every_cluster_agrees() {
        // Clusters 0 and 1 form a parent group over children 2 and 3.
        let uniform = archive(vec![
            cluster(NO_RELATION, 0, 2, 2, 1.0),
            cluster(NO_RELATION, 0, 2, 2, 1.0),
            cluster(0, 2, NO_RELATION, 0, 0.0),
            cluster(0, 2, NO_RELATION, 0, 0.5),
        ]);
        assert_eq!(
            uniform_group_bounds_flags(&uniform),
            [true, false, false, false],
            "the parent pair agrees; the child pair differs in error"
        );
    }

    #[test]
    fn a_shared_start_is_marked_only_if_every_range_from_it_is_uniform() {
        // Cluster 0 starts a uniform parent range of two (named by cluster
        // 3) and a child range of three (named by cluster 4) whose third
        // cluster differs.
        let shared = archive(vec![
            cluster(NO_RELATION, 0, NO_RELATION, 0, 1.0),
            cluster(NO_RELATION, 0, NO_RELATION, 0, 1.0),
            cluster(NO_RELATION, 0, NO_RELATION, 0, 2.0),
            cluster(0, 2, NO_RELATION, 0, 0.0),
            cluster(NO_RELATION, 0, 0, 3, 3.0),
        ]);
        assert!(!uniform_group_bounds_flags(&shared)[0]);
    }
}
