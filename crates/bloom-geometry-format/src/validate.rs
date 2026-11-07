use crate::hash::{hex_hash, sha256};
use crate::types::*;
use crate::vertex::validate_cluster_vertices;
use std::collections::BTreeSet;

pub fn validate_page_budget(page_budget_bytes: u32) -> Result<(), String> {
    if !(MIN_PAGE_BYTES..=MAX_PAGE_BYTES).contains(&page_budget_bytes)
        || !page_budget_bytes.is_power_of_two()
    {
        return Err(format!(
            "geometry page budget must be a power of two in \
             {MIN_PAGE_BYTES}..={MAX_PAGE_BYTES}, got {page_budget_bytes}"
        ));
    }
    Ok(())
}

pub(crate) fn validate_pages(
    pages: &[PageRecord],
    clusters: &[ClusterRecord],
    payload: &[u8],
    page_budget_bytes: u32,
) -> Result<(), String> {
    if pages.is_empty() {
        return if clusters.is_empty() && payload.is_empty() {
            Ok(())
        } else {
            Err("non-empty cooked geometry contains no pages".to_string())
        };
    }
    let mut expected_payload_offset = 0u64;
    let mut expected_cluster = 0u32;
    let mut reached_non_root_pages = false;
    for (page_index, page) in pages.iter().enumerate() {
        if page.payload_offset != expected_payload_offset
            || page.first_cluster != expected_cluster
            || page.cluster_count == 0
        {
            return Err(format!(
                "page {page_index} has a gap, overlap, or empty cluster range"
            ));
        }
        if page.payload_bytes == 0 || page.payload_bytes > page_budget_bytes {
            return Err(format!(
                "page {page_index} length {} violates budget {page_budget_bytes}",
                page.payload_bytes
            ));
        }
        let cluster_start = page.first_cluster as usize;
        let cluster_end = cluster_start
            .checked_add(page.cluster_count as usize)
            .ok_or_else(|| format!("page {page_index} cluster range overflow"))?;
        let page_clusters = clusters
            .get(cluster_start..cluster_end)
            .ok_or_else(|| format!("page {page_index} cluster range exceeds cluster table"))?;
        let first_root = page_clusters[0].flags & FLAG_COARSE_ROOT != 0;
        let first_class = (
            first_root,
            if first_root {
                0
            } else {
                page_clusters[0].lod_level
            },
        );
        if page_clusters.iter().any(|cluster| {
            let coarse_root = cluster.flags & FLAG_COARSE_ROOT != 0;
            (coarse_root, if coarse_root { 0 } else { cluster.lod_level }) != first_class
        }) {
            return Err(format!(
                "page {page_index} mixes streamable hierarchy levels or root residency classes"
            ));
        }
        if first_class.0 {
            if reached_non_root_pages {
                return Err(format!(
                    "page {page_index} places coarse roots after streamable pages"
                ));
            }
        } else {
            reached_non_root_pages = true;
        }
        let start = usize::try_from(page.payload_offset)
            .map_err(|_| format!("page {page_index} offset exceeds host address space"))?;
        let end = start
            .checked_add(page.payload_bytes as usize)
            .ok_or_else(|| format!("page {page_index} range overflow"))?;
        let page_payload = payload
            .get(start..end)
            .ok_or_else(|| format!("page {page_index} exceeds payload"))?;
        let actual_hash = sha256(page_payload);
        if actual_hash != page.sha256 {
            return Err(format!(
                "page {page_index} hash mismatch: expected {}, actual {}",
                hex_hash(page.sha256),
                hex_hash(actual_hash)
            ));
        }
        expected_payload_offset = end as u64;
        expected_cluster = expected_cluster
            .checked_add(page.cluster_count)
            .ok_or("page cluster range overflow")?;
    }
    if expected_payload_offset != payload.len() as u64 {
        return Err("page ranges do not cover the complete payload".to_string());
    }
    if expected_cluster as usize != clusters.len() {
        return Err("page cluster ranges do not cover the cluster table".to_string());
    }
    Ok(())
}

pub(crate) fn validate_clusters(
    clusters: &[ClusterRecord],
    pages: &[PageRecord],
    payload: &[u8],
    vertex_encoding: VertexEncoding,
) -> Result<(), String> {
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let known_flags = FLAG_DOUBLE_SIDED | FLAG_ALPHA_MASKED | FLAG_COARSE_ROOT;
        if !(3..=u8::MAX as u32).contains(&cluster.vertex_count)
            || cluster.triangle_count == 0
            || cluster.flags & !known_flags != 0
        {
            return Err(format!(
                "cluster {cluster_index} has invalid counts, stride, or flags"
            ));
        }
        let page = pages
            .get(cluster.page_index as usize)
            .ok_or_else(|| format!("cluster {cluster_index} references missing page"))?;
        let first = page.first_cluster as usize;
        let end = first
            .checked_add(page.cluster_count as usize)
            .ok_or_else(|| format!("cluster {cluster_index} page range overflow"))?;
        if cluster_index < first || cluster_index >= end {
            return Err(format!(
                "cluster {cluster_index} is outside its page cluster range"
            ));
        }
        validate_cluster_payload(cluster_index, cluster, page, payload, vertex_encoding)?;
        validate_cluster_bounds(cluster_index, cluster)?;
        validate_cluster_relations(cluster_index, cluster, clusters.len())?;
    }
    validate_hierarchy(clusters)
}

pub(crate) fn validate_compatibility_partition(
    clusters: &[ClusterRecord],
    compatibility: &[CompatibilityRecord],
) -> Result<(), String> {
    let eligible = clusters
        .iter()
        .map(|cluster| (cluster.mesh_index, cluster.primitive_index))
        .collect::<BTreeSet<_>>();
    let mut routed = BTreeSet::new();
    for (record_index, record) in compatibility.iter().enumerate() {
        let identity = (record.mesh_index, record.primitive_index);
        if eligible.contains(&identity) {
            return Err(format!(
                "compatibility record {record_index} overlaps eligible mesh {} primitive {}",
                record.mesh_index, record.primitive_index
            ));
        }
        if !routed.insert(identity) {
            return Err(format!(
                "compatibility record {record_index} duplicates mesh {} primitive {}",
                record.mesh_index, record.primitive_index
            ));
        }
    }
    Ok(())
}

fn validate_cluster_payload(
    cluster_index: usize,
    cluster: &ClusterRecord,
    page: &PageRecord,
    payload: &[u8],
    vertex_encoding: VertexEncoding,
) -> Result<(), String> {
    let page_start = page.payload_offset;
    let page_end = page_start
        .checked_add(page.payload_bytes as u64)
        .ok_or_else(|| format!("cluster {cluster_index} page range overflow"))?;
    let vertex_bytes = (cluster.vertex_count as u64)
        .checked_mul(cluster.vertex_stride as u64)
        .ok_or_else(|| format!("cluster {cluster_index} vertex range overflow"))?;
    let index_bytes = (cluster.triangle_count as u64)
        .checked_mul(3)
        .ok_or_else(|| format!("cluster {cluster_index} index range overflow"))?;
    let vertex_end = cluster
        .vertex_offset
        .checked_add(vertex_bytes)
        .ok_or_else(|| format!("cluster {cluster_index} vertex range overflow"))?;
    let index_end = cluster
        .index_offset
        .checked_add(index_bytes)
        .ok_or_else(|| format!("cluster {cluster_index} index range overflow"))?;
    if cluster.vertex_offset < page_start
        || !cluster.vertex_offset.is_multiple_of(16)
        || cluster.index_offset != vertex_end
        || index_end > page_end
    {
        return Err(format!(
            "cluster {cluster_index} payload offsets exceed or overlap its page"
        ));
    }
    validate_cluster_vertices(cluster_index, cluster, payload, vertex_encoding)?;
    let index_start = usize::try_from(cluster.index_offset)
        .map_err(|_| format!("cluster {cluster_index} index offset exceeds host space"))?;
    let index_end = usize::try_from(index_end)
        .map_err(|_| format!("cluster {cluster_index} index end exceeds host space"))?;
    if payload[index_start..index_end]
        .iter()
        .any(|index| *index as u32 >= cluster.vertex_count)
    {
        return Err(format!(
            "cluster {cluster_index} local index exceeds vertex count"
        ));
    }
    Ok(())
}

fn validate_cluster_bounds(cluster_index: usize, cluster: &ClusterRecord) -> Result<(), String> {
    let finite = cluster
        .aabb_min
        .iter()
        .chain(cluster.aabb_max.iter())
        .chain(cluster.sphere_center.iter())
        .chain(std::iter::once(&cluster.sphere_radius))
        .chain(cluster.normal_cone_axis.iter())
        .chain(std::iter::once(&cluster.normal_cone_cutoff))
        .chain(std::iter::once(&cluster.geometric_error))
        .all(|value| value.is_finite());
    if !finite
        || cluster
            .aabb_min
            .iter()
            .zip(cluster.aabb_max)
            .any(|(min, max)| *min > max)
        || cluster.sphere_radius < 0.0
        || !(-1.0..=1.0).contains(&cluster.normal_cone_cutoff)
        || cluster.geometric_error < 0.0
    {
        return Err(format!("cluster {cluster_index} has invalid bounds/error"));
    }
    Ok(())
}

fn validate_cluster_relations(
    cluster_index: usize,
    cluster: &ClusterRecord,
    cluster_count: usize,
) -> Result<(), String> {
    validate_relation(cluster_index, "parent", cluster.parent, cluster_count)?;
    if cluster.parent == NO_RELATION {
        if cluster.parent_count != 0 {
            return Err(format!(
                "cluster {cluster_index} has no parent but a non-zero parent count"
            ));
        }
    } else {
        let parent_end = (cluster.parent as usize)
            .checked_add(cluster.parent_count as usize)
            .ok_or_else(|| format!("cluster {cluster_index} parent range overflow"))?;
        if cluster.parent_count == 0 || parent_end > cluster_count {
            return Err(format!(
                "cluster {cluster_index} parent range exceeds cluster table"
            ));
        }
    }
    if cluster.child_count == 0 {
        if cluster.first_child != NO_RELATION {
            return Err(format!(
                "cluster {cluster_index} has no children but a first-child index"
            ));
        }
    } else {
        let first = cluster.first_child as usize;
        let end = first
            .checked_add(cluster.child_count as usize)
            .ok_or_else(|| format!("cluster {cluster_index} child range overflow"))?;
        if first >= cluster_count || end > cluster_count {
            return Err(format!(
                "cluster {cluster_index} child range exceeds cluster table"
            ));
        }
    }
    Ok(())
}

fn validate_hierarchy(clusters: &[ClusterRecord]) -> Result<(), String> {
    let hierarchy_present = clusters.iter().any(|cluster| {
        cluster.parent_count != 0
            || cluster.child_count != 0
            || cluster.lod_level != 0
            || cluster.flags & FLAG_COARSE_ROOT != 0
    });
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        if cluster.parent == NO_RELATION {
            if hierarchy_present && cluster.flags & FLAG_COARSE_ROOT == 0 {
                return Err(format!(
                    "hierarchy cluster {cluster_index} has no parent and is not a coarse root"
                ));
            }
            continue;
        }
        if cluster.flags & FLAG_COARSE_ROOT != 0 {
            return Err(format!(
                "cluster {cluster_index} is both a hierarchy child and coarse root"
            ));
        }
        validate_parent_group(cluster_index, cluster, clusters)?;
    }
    for (parent_index, parent) in clusters.iter().enumerate() {
        if parent.child_count != 0 {
            validate_child_group(parent_index, parent, clusters)?;
        }
    }
    Ok(())
}

fn validate_parent_group(
    cluster_index: usize,
    cluster: &ClusterRecord,
    clusters: &[ClusterRecord],
) -> Result<(), String> {
    let parent_start = cluster.parent as usize;
    let parent_end = parent_start + cluster.parent_count as usize;
    let first_parent = &clusters[parent_start];
    for (offset, parent) in clusters[parent_start..parent_end].iter().enumerate() {
        let parent_index = parent_start + offset;
        let child_start = parent.first_child as usize;
        let child_end = child_start
            .checked_add(parent.child_count as usize)
            .ok_or_else(|| format!("cluster {parent_index} child range overflow"))?;
        if !(child_start..child_end).contains(&cluster_index)
            || parent.lod_level <= cluster.lod_level
            || parent.first_child != first_parent.first_child
            || parent.child_count != first_parent.child_count
            || parent.lod_level != first_parent.lod_level
            || parent.geometric_error != first_parent.geometric_error
        {
            return Err(format!(
                "cluster {cluster_index} has a non-reciprocal or inconsistent parent group"
            ));
        }
    }
    Ok(())
}

fn validate_child_group(
    parent_index: usize,
    parent: &ClusterRecord,
    clusters: &[ClusterRecord],
) -> Result<(), String> {
    let child_start = parent.first_child as usize;
    let child_end = child_start + parent.child_count as usize;
    let first_child = &clusters[child_start];
    let parent_start = first_child.parent as usize;
    let parent_end = parent_start
        .checked_add(first_child.parent_count as usize)
        .ok_or_else(|| format!("cluster {parent_index} sibling range overflow"))?;
    if first_child.parent == NO_RELATION
        || first_child.parent_count == 0
        || !(parent_start..parent_end).contains(&parent_index)
    {
        return Err(format!(
            "parent {parent_index} is outside its reciprocal sibling group"
        ));
    }
    for (offset, child) in clusters[child_start..child_end].iter().enumerate() {
        let child_index = child_start + offset;
        if child.parent != first_child.parent
            || child.parent_count != first_child.parent_count
            || child.lod_level >= parent.lod_level
            || child.mesh_index != parent.mesh_index
            || child.primitive_index != parent.primitive_index
            || child.material_index != parent.material_index
            || (child.flags & !FLAG_COARSE_ROOT) != (parent.flags & !FLAG_COARSE_ROOT)
            || parent.geometric_error < child.geometric_error
        {
            return Err(format!(
                "parent {parent_index} and child {child_index} violate hierarchy identity/error"
            ));
        }
    }
    Ok(())
}

fn validate_relation(
    cluster_index: usize,
    label: &str,
    relation: u32,
    cluster_count: usize,
) -> Result<(), String> {
    if relation != NO_RELATION && relation as usize >= cluster_count {
        return Err(format!(
            "cluster {cluster_index} {label} index {relation} exceeds cluster table"
        ));
    }
    Ok(())
}
