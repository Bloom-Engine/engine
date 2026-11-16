//! A reduced ordinary mesh cut from a cooked archive's own hierarchy.
//!
//! Virtual clusters are drawn to the camera only. Shadow maps, the ray
//! scene, mesh cards and the distance field all consume ordinary triangles,
//! so each virtual model also gets a stand-in for them: the clusters of one
//! consistent cut through the hierarchy at a chosen geometric error, decoded
//! on the CPU. An error of zero selects the leaves, which is the source mesh.

use super::VirtualGeometryAsset;
use bloom_geometry_format::{ClusterRecord, VertexEncoding, FLAG_ALPHA_MASKED, NO_RELATION};
use std::collections::BTreeMap;

/// One decoded vertex, in the archive's attribute set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProxyVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// The stand-in triangles of one source glTF primitive.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProxyPrimitive {
    pub vertices: Vec<ProxyVertex>,
    pub indices: Vec<u32>,
}

/// Largest error over the group of sibling clusters `cluster` belongs to.
/// Siblings replace the same child range and are selected together, so they
/// must be judged by one number.
fn group_error(clusters: &[ClusterRecord], index: usize) -> f32 {
    let cluster = &clusters[index];
    if cluster.first_child == NO_RELATION {
        return cluster.geometric_error;
    }
    let Some(child) = clusters.get(cluster.first_child as usize) else {
        return cluster.geometric_error;
    };
    if child.parent == NO_RELATION {
        return cluster.geometric_error;
    }
    let first = child.parent as usize;
    let end = (first + child.parent_count as usize).min(clusters.len());
    clusters[first..end]
        .iter()
        .map(|sibling| sibling.geometric_error)
        .fold(cluster.geometric_error, f32::max)
}

/// Whether `index` lies on the cut at `max_error`: its own group is within
/// the error and the group that would replace it is not.
fn on_cut(clusters: &[ClusterRecord], index: usize, max_error: f32) -> bool {
    let cluster = &clusters[index];
    if cluster.first_child != NO_RELATION && group_error(clusters, index) > max_error {
        return false;
    }
    if cluster.parent == NO_RELATION {
        return true;
    }
    let parent = cluster.parent as usize;
    parent >= clusters.len() || group_error(clusters, parent) > max_error
}

fn f32_at(bytes: &[u8], offset: usize) -> Option<f32> {
    let raw = bytes.get(offset..offset + 4)?;
    Some(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    let raw = bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([raw[0], raw[1]]))
}

fn snorm16(value: u16) -> f32 {
    (f32::from(value as i16) / 32767.0).max(-1.0)
}

fn octahedral(x: u16, y: u16) -> [f32; 3] {
    let x = snorm16(x);
    let y = snorm16(y);
    let mut n = [x, y, 1.0 - x.abs() - y.abs()];
    if n[2] < 0.0 {
        let old_x = n[0];
        let sign = |v: f32| if v < 0.0 { -1.0 } else { 1.0 };
        n[0] = (1.0 - n[1].abs()) * sign(old_x);
        n[1] = (1.0 - old_x.abs()) * sign(n[1]);
    }
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        .sqrt()
        .max(1.0e-20);
    [n[0] / length, n[1] / length, n[2] / length]
}

fn decode_vertex(
    page: &[u8],
    offset: usize,
    cluster: &ClusterRecord,
    encoding: VertexEncoding,
) -> Option<ProxyVertex> {
    match encoding {
        VertexEncoding::Float32 => {
            let f = |component: usize| f32_at(page, offset + component * 4);
            Some(ProxyVertex {
                position: [f(0)?, f(1)?, f(2)?],
                normal: [f(3)?, f(4)?, f(5)?],
                tangent: [f(6)?, f(7)?, f(8)?, f(9)?],
                uv: [f(10)?, f(11)?],
                color: [f(14)?, f(15)?, f(16)?, f(17)?],
            })
        }
        VertexEncoding::Quantized => {
            let mut position = [0.0; 3];
            for (axis, value) in position.iter_mut().enumerate() {
                let encoded = f32::from(u16_at(page, offset + axis * 2)?) / 65535.0;
                let (lo, hi) = (cluster.aabb_min[axis], cluster.aabb_max[axis]);
                *value = lo + (hi - lo) * encoded;
            }
            let normal = octahedral(u16_at(page, offset + 6)?, u16_at(page, offset + 8)?);
            let flags = u16_at(page, offset + 28)?;
            let mut tangent = [0.0; 4];
            if flags & 1 != 0 {
                let t = octahedral(u16_at(page, offset + 10)?, u16_at(page, offset + 12)?);
                tangent = [t[0], t[1], t[2], snorm16(u16_at(page, offset + 26)?)];
            }
            let half = |at: usize| Some(half::f16::from_bits(u16_at(page, at)?).to_f32());
            let color = page.get(offset + 22..offset + 26)?;
            Some(ProxyVertex {
                position,
                normal,
                tangent,
                uv: [half(offset + 14)?, half(offset + 16)?],
                color: [
                    f32::from(color[0]) / 255.0,
                    f32::from(color[1]) / 255.0,
                    f32::from(color[2]) / 255.0,
                    f32::from(color[3]) / 255.0,
                ],
            })
        }
    }
}

/// Decode the cut at `max_error` (object units) into one mesh per source
/// primitive, keyed by `(mesh_index, primitive_index)`. Alpha-masked clusters
/// are left out: the ordinary renderer draws those primitives itself. The
/// asset must be memory-backed (as `VirtualGeometryAsset::from_bytes` makes).
pub fn build_proxy(
    asset: &VirtualGeometryAsset,
    max_error: f32,
) -> Result<BTreeMap<(u32, u32), ProxyPrimitive>, String> {
    let archive = asset.archive();
    let clusters = &archive.clusters;
    let mut output = BTreeMap::<(u32, u32), ProxyPrimitive>::new();
    for (index, cluster) in clusters.iter().enumerate() {
        if cluster.flags & FLAG_ALPHA_MASKED != 0 || !on_cut(clusters, index, max_error) {
            continue;
        }
        let page_index = cluster.page_index as usize;
        let page_record = archive
            .pages
            .get(page_index)
            .ok_or_else(|| format!("cluster {index} names missing page {page_index}"))?;
        let page = asset
            .page_bytes(page_index)
            .ok_or_else(|| format!("page {page_index} is not in memory"))?;
        let local = |offset: u64| {
            offset
                .checked_sub(page_record.payload_offset)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("cluster {index} lies outside page {page_index}"))
        };
        let vertex_start = local(cluster.vertex_offset)?;
        let index_start = local(cluster.index_offset)?;
        let primitive = output
            .entry((cluster.mesh_index, cluster.primitive_index))
            .or_default();
        let base = u32::try_from(primitive.vertices.len())
            .map_err(|_| "stand-in mesh exceeds 32-bit indices".to_string())?;
        for vertex in 0..cluster.vertex_count as usize {
            let offset = vertex_start + vertex * cluster.vertex_stride as usize;
            primitive.vertices.push(
                decode_vertex(page, offset, cluster, archive.vertex_encoding)
                    .ok_or_else(|| format!("cluster {index} vertex {vertex} is truncated"))?,
            );
        }
        let index_bytes = page
            .get(index_start..index_start + cluster.triangle_count as usize * 3)
            .ok_or_else(|| format!("cluster {index} indices are truncated"))?;
        primitive
            .indices
            .extend(index_bytes.iter().map(|local| base + u32::from(*local)));
    }
    Ok(output)
}

/// Diagonal of the bounds of every cluster: the scale a relative error is
/// measured against.
pub fn bounds_diagonal(asset: &VirtualGeometryAsset) -> f32 {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for cluster in &asset.archive().clusters {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(cluster.aabb_min[axis]);
            hi[axis] = hi[axis].max(cluster.aabb_max[axis]);
        }
    }
    if lo[0] > hi[0] {
        return 0.0;
    }
    let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}
