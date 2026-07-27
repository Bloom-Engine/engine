use super::*;

#[test]
fn record_abi_is_six_vec4s_and_default_is_inactive() {
    assert_eq!(std::mem::size_of::<PtLayeredMaterialCpu>(), 96);
    let record = PtLayeredMaterialCpu::default();
    assert_eq!(record.header, [PT_LAYERED_RECORD_VERSION, 0, 0, 0]);
    assert!(!record.active());
}

#[test]
fn scalar_record_preserves_every_current_lobe_bit() {
    let mask = crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE
        | crate::models::MaterialLayeredPbr::SHEEN_LOBE
        | crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE
        | crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE
        | crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE;
    let material = crate::models::MaterialLayeredPbr::from_authoring_factors(
        mask,
        0.8,
        0.2,
        1.0,
        0.7,
        [0.9, 0.8, 0.7],
        1.45,
        [0.2, 0.1, 0.05],
        0.4,
        0.6,
        0.3,
        0.75,
        1.3,
        120.0,
        360.0,
    );
    let record = PtLayeredMaterialCpu::from_material(material);
    assert_eq!(record.header[1], mask);
    assert_eq!(record.clearcoat_ior, [0.8, 0.2, 1.0, 1.45]);
    assert_eq!(record.iridescence, [0.75, 1.3, 120.0, 360.0]);
    assert!(record.has_iridescence() && record.has_qualified_transport());
}

#[test]
fn only_qualified_lobes_select_layered_transport() {
    let sheen = crate::models::MaterialLayeredPbr::from_authoring_factors(
        crate::models::MaterialLayeredPbr::SHEEN_LOBE,
        0.0,
        0.0,
        1.0,
        1.0,
        [1.0; 3],
        1.5,
        [0.3, 0.1, 0.05],
        0.4,
        0.0,
        0.0,
        0.0,
        1.3,
        100.0,
        400.0,
    );
    let clearcoat = crate::models::MaterialLayeredPbr::from_authoring_factors(
        crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE,
        0.8,
        0.2,
        1.0,
        1.0,
        [1.0; 3],
        1.5,
        [0.0; 3],
        0.0,
        0.0,
        0.0,
        0.0,
        1.3,
        100.0,
        400.0,
    );
    let specular = crate::models::MaterialLayeredPbr::from_authoring_factors(
        crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE,
        0.0,
        0.0,
        1.0,
        0.7,
        [0.8, 0.6, 0.4],
        1.8,
        [0.0; 3],
        0.0,
        0.0,
        0.0,
        0.0,
        1.3,
        100.0,
        400.0,
    );
    let anisotropy = crate::models::MaterialLayeredPbr::from_authoring_factors(
        crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE,
        0.0,
        0.0,
        1.0,
        1.0,
        [1.0; 3],
        1.5,
        [0.0; 3],
        0.0,
        0.6,
        0.3,
        0.0,
        1.3,
        100.0,
        400.0,
    );
    let texture = crate::models::MaterialTextureBinding {
        source_texture_index: 1,
        source_image_index: 2,
        runtime_texture_idx: Some(3),
        transform: Default::default(),
    };
    let textured_material = crate::models::MaterialLayeredPbr {
        clearcoat_authored: true,
        clearcoat_factor: 0.8,
        clearcoat_texture: Some(texture),
        specular_authored: true,
        specular_factor: 0.7,
        specular_texture: Some(texture),
        sheen_authored: true,
        sheen_color_factor: [0.3, 0.1, 0.05],
        sheen_color_texture: Some(texture),
        anisotropy_authored: true,
        anisotropy_strength: 0.6,
        anisotropy_texture: Some(texture),
        iridescence_authored: true,
        iridescence_factor: 0.75,
        iridescence_texture: Some(texture),
        ..Default::default()
    };
    let sheen = PtLayeredMaterialCpu::from_material(sheen);
    let clearcoat = PtLayeredMaterialCpu::from_material(clearcoat);
    let specular = PtLayeredMaterialCpu::from_material(specular);
    let anisotropy = PtLayeredMaterialCpu::from_material(anisotropy);
    let textured = PtLayeredMaterialCpu::from_material(textured_material);
    let textured_meta = PtLayeredTextureCpu::from_material(textured_material, 4, false);
    let clearcoat_meta = PtClearcoatTextureCpu::from_material(textured_material, 4, false);
    let sheen_meta = PtSheenTextureCpu::from_material(textured_material, 4, false);
    let iridescence_meta = PtIridescenceTextureCpu::from_material(textured_material, 4, false);
    assert!(sheen.has_sheen() && sheen.has_qualified_transport());
    assert!(clearcoat.has_clearcoat() && clearcoat.has_qualified_transport());
    assert!(specular.has_specular_ior() && specular.has_qualified_transport());
    assert!(anisotropy.has_anisotropy() && anisotropy.has_qualified_transport());
    assert!(textured.active() && !textured.has_qualified_transport());
    assert!(textured_meta.has_specular_ior());
    assert!(clearcoat_meta.active());
    assert!(sheen_meta.active());
    assert!(iridescence_meta.active());
    assert_eq!(
        textured.header[2],
        crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE
            | crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE
            | crate::models::MaterialLayeredPbr::SHEEN_LOBE
            | crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE
            | crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE
    );
}

#[test]
fn specialization_uses_separate_group_without_touching_base_kernel() {
    assert!(PT_LAYERED_BINDINGS_WGSL.contains("@group(2) @binding(0)"));
    assert!(PT_LAYERED_TEXTURE_BINDINGS_WGSL.contains("@group(2) @binding(2)"));
    assert!(PT_CLEARCOAT_TEXTURE_BINDINGS_WGSL.contains("@group(2) @binding(4)"));
    assert!(PT_SHEEN_TEXTURE_BINDINGS_WGSL.contains("@group(2) @binding(5)"));
    assert!(PT_IRIDESCENCE_TEXTURE_BINDINGS_WGSL.contains("@group(2) @binding(6)"));
    assert!(!PT_LAYERED_SHEEN_DISABLED_WGSL.contains("@binding(1)"));
    assert!(PT_LAYERED_SHEEN_WGSL.contains("@group(2) @binding(1)"));
    assert!(PT_LAYERED_TRANSPORT_WGSL.contains("PT_HAS_SCALAR_ANISOTROPY"));
    assert!(PT_LAYERED_TRANSPORT_WGSL.contains("slot * PT_VSTRIDE + 20u"));
    assert!(PT_LAYERED_TRANSPORT_WGSL.contains("hit.object_to_world"));
    assert!(PT_LAYERED_IRIDESCENCE_WGSL.contains("pt_eval_iridescence"));
    assert!(!PT_LAYERED_IRIDESCENCE_DISABLED_WGSL.contains("exp("));
    assert!(!pt_kernel_variant(false).contains("pt_layered_materials"));
}

#[test]
fn layered_specialization_rewrites_every_transport_vertex() {
    for diagnostics in [false, true] {
        let base = pt_kernel_variant(diagnostics);
        let specialized = layered_kernel_variant(base.as_ref());
        assert!(specialized.contains("let layered_primary = pt_layered_primary_surface(p0, n0);"));
        assert_eq!(specialized.matches("pt_sample_layered_brdf(").count(), 1);
        assert_eq!(
            specialized
                .matches("throughput * pt_layered_direct_light(")
                .count(),
            1
        );
        assert!(specialized.contains("layered_cur = layered_hit;"));
        assert!(specialized.contains("layered_tangent_cur = layered_tangent_hit;"));
        assert_eq!(base.matches("pt_layered_").count(), 0);
    }
}
