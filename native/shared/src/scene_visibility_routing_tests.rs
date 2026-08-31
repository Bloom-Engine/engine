use super::*;

#[test]
fn visibility_routing_admits_opaque_subset_beside_cutout_compatibility() {
    let mut scene = SceneGraph::new();
    let opaque = scene.create_node();
    scene.nodes.get_mut(opaque).unwrap().indices = vec![0, 1, 2];
    assert!(retained_order_is_gpu_safe(&scene.nodes, false));

    let cutout = scene.create_node();
    let cutout_node = scene.nodes.get_mut(cutout).unwrap();
    cutout_node.indices = vec![0, 1, 2];
    cutout_node.material.alpha_cutoff = 0.5;
    assert!(!retained_order_is_gpu_safe(&scene.nodes, false));
    assert!(
        retained_order_is_gpu_safe(&scene.nodes, true),
        "explicit visibility routing keeps MASK on compatibility without disabling opaques"
    );

    scene.nodes.get_mut(cutout).unwrap().visible = false;
    assert!(retained_order_is_gpu_safe(&scene.nodes, false));

    let fractional = scene.create_node();
    let fractional_node = scene.nodes.get_mut(fractional).unwrap();
    fractional_node.indices = vec![0, 1, 2];
    fractional_node.material.opacity = 0.5;
    assert!(!retained_order_is_gpu_safe(&scene.nodes, false));
    assert!(
        !retained_order_is_gpu_safe(&scene.nodes, true),
        "legacy in-pass fractional opacity remains globally order-sensitive"
    );
    scene.nodes.get_mut(fractional).unwrap().visible = false;

    let blend = scene.create_node();
    let blend_node = scene.nodes.get_mut(blend).unwrap();
    blend_node.indices = vec![0, 1, 2];
    blend_node.material.alpha_mode = MaterialAlphaMode::Blend;
    assert!(
        retained_order_is_gpu_safe(&scene.nodes, false),
        "dedicated translucent submission must not disable the opaque GPU-driven subset"
    );
}
