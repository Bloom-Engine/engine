use super::*;

fn triangles(asset: &VirtualGeometryAsset, max_error: f32) -> usize {
    build_proxy(asset, max_error)
        .unwrap()
        .values()
        .map(|primitive| primitive.indices.len() / 3)
        .sum()
}

#[test]
fn proxy_cut_follows_the_error_threshold() {
    // Errors in the test hierarchy equal the level: leaves 0, middle 1, roots 2.
    let asset = hierarchy_asset(hierarchy_archive());
    assert_eq!(triangles(&asset, 0.0), 4, "zero error keeps every leaf");
    assert_eq!(
        triangles(&asset, 1.0),
        2,
        "the middle level replaces the leaves"
    );
    assert_eq!(triangles(&asset, 1.5), 2);
    assert_eq!(
        triangles(&asset, 2.0),
        2,
        "the roots replace the middle level"
    );
    assert_eq!(triangles(&asset, f32::MAX), 2);
}

#[test]
fn proxy_cut_takes_sibling_groups_whole() {
    // Two roots replace four middle clusters, which replace two leaf pairs.
    let asset = hierarchy_asset(branching_hierarchy_archive());
    assert_eq!(triangles(&asset, 0.0), 4);
    assert_eq!(triangles(&asset, 1.0), 4);
    assert_eq!(triangles(&asset, 2.0), 2);
}

#[test]
fn proxy_vertices_decode_both_encodings() {
    for encoding in [VertexEncoding::Float32, VertexEncoding::Quantized] {
        let mut archive = hierarchy_archive();
        if encoding == VertexEncoding::Quantized {
            archive.format_version = QUANTIZED_VERSION;
            archive.vertex_encoding = encoding;
            for cluster in &mut archive.clusters {
                cluster.vertex_stride = 32;
                cluster.index_offset = 96;
            }
        }
        let asset = hierarchy_asset(archive);
        let proxy = build_proxy(&asset, 0.0).unwrap();
        let primitive = &proxy[&(0, 0)];
        assert_eq!(primitive.vertices.len(), 12);
        assert_eq!(primitive.indices.len(), 12);
        assert!(primitive
            .indices
            .iter()
            .all(|index| (*index as usize) < primitive.vertices.len()));
        for vertex in &primitive.vertices {
            assert!((vertex.normal[2] - 1.0).abs() < 1.0e-3, "{encoding:?}");
            assert!((vertex.color[0] - 1.0).abs() < 1.0e-2);
            assert!((vertex.color[1] - 0.5).abs() < 1.0e-2);
            assert!((vertex.uv[0] - vertex.position[0]).abs() < 1.0e-3);
            assert!(vertex.position.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }
}

#[test]
fn proxy_leaves_alpha_masked_clusters_to_the_ordinary_renderer() {
    let mut archive = hierarchy_archive();
    for cluster in &mut archive.clusters {
        cluster.flags |= FLAG_ALPHA_MASKED;
    }
    let asset = hierarchy_asset(archive);
    assert!(build_proxy(&asset, 0.0).unwrap().is_empty());
}

#[test]
fn zero_proxy_error_keeps_the_leaves_even_under_a_lossless_parent() {
    let mut archive = hierarchy_archive();
    for cluster in &mut archive.clusters {
        cluster.geometric_error = 0.0;
    }
    let asset = hierarchy_asset(archive);
    assert_eq!(triangles(&asset, 0.0), 4);
    assert_eq!(triangles(&asset, 0.5), 2, "any allowance takes the roots");
}
