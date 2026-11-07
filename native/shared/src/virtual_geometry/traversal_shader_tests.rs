use super::*;

#[test]
fn traversal_shader_parses_and_uses_bounded_depth_and_branch_contracts() {
    wgpu::naga::front::wgsl::parse_str(TRAVERSAL_SHADER)
        .unwrap_or_else(|error| panic!("virtual hierarchy WGSL failed: {error:?}"));
    assert!(TRAVERSAL_SHADER.contains("depth >= params.limits.x"));
    assert!(TRAVERSAL_SHADER
        .contains("stack_count + child_group_count > TRAVERSAL_GROUP_STACK_CAPACITY"));
    assert_eq!(MAX_GPU_HIERARCHY_LEVELS, 32);
    assert_eq!(TRAVERSAL_GROUP_STACK_CAPACITY, 32);
}

#[test]
fn non_uniform_instances_disable_only_cone_culling() {
    let model = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let instance = GpuVirtualInstance::new(VirtualMeshId::from_raw(1), 7, model).unwrap();
    assert!(!instance.cone_cull_safe());
    assert!(!instance.negative_determinant());
    assert_eq!(instance.normal_rows[0][0], 0.5);
    assert_eq!(instance.normal_rows[1][1], 1.0);
    assert_eq!(instance.normal_rows[2][2], 2.0);
}

#[test]
fn mirrored_instances_preserve_tangent_handedness_state() {
    let model = [
        [-2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let instance = GpuVirtualInstance::new(VirtualMeshId::from_raw(1), 8, model).unwrap();
    assert!(instance.cone_cull_safe());
    assert!(instance.negative_determinant());
}
