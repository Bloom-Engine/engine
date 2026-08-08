use super::{pt_fault_constants, pt_kernel_variant};

#[test]
fn production_variant_removes_query_heavy_debug_locals() {
    let production = pt_kernel_variant(false);
    assert!(!production.contains("var rq8: ray_query"));
    assert!(!production.contains("PT_QUERY_DIAGNOSTICS"));
    assert!(production.contains("var rq: ray_query"));
    assert_eq!(production.matches(": ray_query").count(), 2);
    assert!(production.contains("debug == 24.0"));
    assert!(production.contains("debug == 25.0"));
    assert!(production.contains("@binding(8) var<storage, read> accum"));
    assert!(production.contains("@binding(18) var<storage, read> moments"));
    assert!(production.contains("@binding(20) var<storage, read> resv"));

    let diagnostics = pt_kernel_variant(true);
    assert!(diagnostics.contains("var rq8: ray_query"));
    assert_eq!(diagnostics.matches(": ray_query").count(), 12);
}

#[test]
fn negative_control_constants_change_only_the_targeted_input() {
    let production = pt_fault_constants(None);
    assert!(production.contains("ENERGY_SCALE: f32 = 1.0"));
    assert!(production.contains("REPROJECTION_OFFSET: f32 = 0.0"));
    assert!(production.contains("BYPASS_VALIDATION: bool = false"));

    let energy = pt_fault_constants(Some("brdf-energy"));
    assert!(energy.contains("ENERGY_SCALE: f32 = 1.25"));
    assert!(energy.contains("REPROJECTION_OFFSET: f32 = 0.0"));
    assert!(energy.contains("BYPASS_VALIDATION: bool = false"));

    let reprojection = pt_fault_constants(Some("reprojection"));
    assert!(reprojection.contains("ENERGY_SCALE: f32 = 1.0"));
    assert!(reprojection.contains("REPROJECTION_OFFSET: f32 = 0.05"));
    assert!(reprojection.contains("BYPASS_VALIDATION: bool = true"));
}

#[test]
fn base_transport_uses_bounded_reciprocal_layered_contract() {
    let production = pt_kernel_variant(false);
    assert!(production.contains("sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2)"));
    assert!(production.contains("sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2)"));
    assert!(!production.contains("sqrt((n_dot_v * (1.0 - a2) + a2) * n_dot_v)"));
    assert!(production.contains("let energy_factor = mix(1.0, 1.0 / 1.51, roughness)"));
    assert!(production
        .contains("base_color * (1.0 - metallic) * view_transmission * light_transmission"));
    assert!(
        production.contains("full_alb * (1.0 - metal) * view_transmission * light_transmission")
    );
    assert!(production.contains("return diffuse + spec;"));
    assert!(!production.contains("return alb * lit + spec;"));
}

#[test]
fn zero_sample_count_cannot_seed_from_retained_moments() {
    let production = pt_kernel_variant(false);
    let gate = production
        .find("if (u.size.w > 0u) {")
        .expect("neighbor seeding must have a global-history gate");
    let neighbor_read = production[gate..]
        .find("let m = moments[qidx];")
        .expect("gate must enclose the disocclusion neighbor read");
    let gate_end = production[gate..]
        .find("if (seed_w > 0.0) {")
        .expect("seed reduction follows the gated search");
    assert!(neighbor_read < gate_end);
}
