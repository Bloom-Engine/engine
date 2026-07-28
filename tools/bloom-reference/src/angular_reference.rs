//! Multi-view/multi-light comparison corpus for the complete layered model.

use crate::layered_pbr::{
    evaluate_layered_brdf, BaseMaterial, LayeredBrdfEvaluation, LayeredMaterial,
    IRIDESCENCE_REFERENCE_VERSION,
};
use glam::Vec3;
use serde::Serialize;

const ABSOLUTE_COMPONENT_TOLERANCE: f32 = 3.0e-5;

#[derive(Serialize)]
struct AngularMaterial {
    base_color: [f32; 3],
    metallic: f32,
    perceptual_roughness: f32,
    ior: f32,
    specular_factor: f32,
    specular_color: [f32; 3],
    clearcoat_factor: f32,
    clearcoat_perceptual_roughness: f32,
    sheen_color: [f32; 3],
    sheen_perceptual_roughness: f32,
    anisotropy_strength: f32,
    anisotropy_rotation: f32,
    iridescence_factor: f32,
    iridescence_ior: f32,
    iridescence_thickness_nm: f32,
}

impl From<LayeredMaterial> for AngularMaterial {
    fn from(material: LayeredMaterial) -> Self {
        Self {
            base_color: material.base.base_color.to_array(),
            metallic: material.base.metallic,
            perceptual_roughness: material.base.perceptual_roughness,
            ior: material.ior,
            specular_factor: material.specular_factor,
            specular_color: material.specular_color.to_array(),
            clearcoat_factor: material.clearcoat_factor,
            clearcoat_perceptual_roughness: material.clearcoat_perceptual_roughness,
            sheen_color: material.sheen_color.to_array(),
            sheen_perceptual_roughness: material.sheen_perceptual_roughness,
            anisotropy_strength: material.anisotropy_strength,
            anisotropy_rotation: material.anisotropy_rotation,
            iridescence_factor: material.iridescence_factor,
            iridescence_ior: material.iridescence_ior,
            iridescence_thickness_nm: material.iridescence_thickness_nm,
        }
    }
}

#[derive(Serialize)]
struct AngularSample {
    id: String,
    lobe: &'static str,
    material: AngularMaterial,
    n_dot_v: f32,
    view_azimuth_radians: f32,
    n_dot_l: f32,
    light_azimuth_radians: f32,
    direct_diffuse: [f32; 3],
    direct_base_specular: [f32; 3],
    direct_sheen_specular: [f32; 3],
    direct_clearcoat_specular: [f32; 3],
    direct_brdf_cos: [f32; 3],
    direct_pdf: f32,
    reciprocity_max_abs_error: f32,
}

#[derive(Serialize)]
struct Thresholds {
    serialized_decimal_places: u32,
    checked_in_regeneration: &'static str,
    cpu_brdf_component_absolute: f32,
    reciprocity_max_absolute: f32,
}

#[derive(Serialize)]
pub(crate) struct AngularContractReport {
    schema: &'static str,
    corpus_version: u32,
    material_contract_version: u32,
    comparison_target: &'static str,
    coordinate_convention: &'static str,
    coverage: &'static str,
    thresholds: Thresholds,
    known_model_differences: [&'static str; 5],
    maximum_reciprocity_error: f32,
    samples: Vec<AngularSample>,
}

fn rounded(value: f32) -> f32 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn rounded_vec(value: Vec3) -> [f32; 3] {
    [rounded(value.x), rounded(value.y), rounded(value.z)]
}

fn direction(n_dot: f32, azimuth: f32) -> Vec3 {
    let sin_theta = (1.0 - n_dot * n_dot).sqrt();
    Vec3::new(sin_theta * azimuth.cos(), sin_theta * azimuth.sin(), n_dot)
}

fn maximum_component(value: Vec3) -> f32 {
    value.x.max(value.y).max(value.z)
}

fn brdf(evaluation: LayeredBrdfEvaluation) -> Vec3 {
    evaluation.diffuse
        + evaluation.base_specular
        + evaluation.sheen_specular
        + evaluation.clearcoat_specular
}

fn scenarios() -> [(&'static str, LayeredMaterial); 7] {
    let base = || {
        LayeredMaterial::from_base(BaseMaterial {
            base_color: Vec3::new(0.42, 0.18, 0.055),
            metallic: 0.2,
            perceptual_roughness: 0.38,
        })
    };
    [
        ("base", base()),
        (
            "specular-ior",
            LayeredMaterial {
                ior: 1.82,
                specular_factor: 0.72,
                specular_color: Vec3::new(1.2, 0.62, 0.28),
                ..base()
            },
        ),
        (
            "clearcoat",
            LayeredMaterial {
                clearcoat_factor: 0.85,
                clearcoat_perceptual_roughness: 0.17,
                ..base()
            },
        ),
        (
            "sheen",
            LayeredMaterial {
                sheen_color: Vec3::new(0.25, 0.65, 0.9),
                sheen_perceptual_roughness: 0.43,
                ..base()
            },
        ),
        (
            "anisotropy",
            LayeredMaterial {
                base: BaseMaterial {
                    metallic: 1.0,
                    ..base().base
                },
                anisotropy_strength: 0.82,
                anisotropy_rotation: 0.61,
                ..base()
            },
        ),
        (
            "iridescence",
            LayeredMaterial {
                iridescence_factor: 0.9,
                iridescence_ior: 1.36,
                iridescence_thickness_nm: 475.0,
                ..base()
            },
        ),
        (
            "combined",
            LayeredMaterial {
                ior: 1.7,
                specular_factor: 0.8,
                specular_color: Vec3::new(1.1, 0.7, 0.4),
                clearcoat_factor: 0.65,
                clearcoat_perceptual_roughness: 0.2,
                sheen_color: Vec3::new(0.2, 0.55, 0.9),
                sheen_perceptual_roughness: 0.38,
                anisotropy_strength: 0.75,
                anisotropy_rotation: 0.63,
                iridescence_factor: 0.9,
                iridescence_ior: 1.38,
                iridescence_thickness_nm: 560.0,
                ..base()
            },
        ),
    ]
}

pub(crate) fn build_angular_report() -> AngularContractReport {
    let normal = Vec3::Z;
    let views = [(0.2, 0.0), (0.5, 0.7), (0.85, 1.4)];
    let lights = [(0.25, 2.4), (0.55, 1.2), (0.9, 0.35)];
    let mut maximum_reciprocity_error = 0.0f32;
    let mut samples = Vec::with_capacity(scenarios().len() * views.len() * lights.len());
    for (lobe, material) in scenarios() {
        for (view_index, (n_dot_v, view_azimuth)) in views.into_iter().enumerate() {
            let view = direction(n_dot_v, view_azimuth);
            for (light_index, (n_dot_l, light_azimuth)) in lights.into_iter().enumerate() {
                let light = direction(n_dot_l, light_azimuth);
                let direct = evaluate_layered_brdf(material, normal, view, light);
                let reverse = evaluate_layered_brdf(material, normal, light, view);
                let reciprocity_error = maximum_component((brdf(direct) - brdf(reverse)).abs());
                maximum_reciprocity_error = maximum_reciprocity_error.max(reciprocity_error);
                samples.push(AngularSample {
                    id: format!("{lobe}-v{view_index}-l{light_index}"),
                    lobe,
                    material: material.into(),
                    n_dot_v,
                    view_azimuth_radians: rounded(view_azimuth),
                    n_dot_l,
                    light_azimuth_radians: rounded(light_azimuth),
                    direct_diffuse: rounded_vec(direct.diffuse),
                    direct_base_specular: rounded_vec(direct.base_specular),
                    direct_sheen_specular: rounded_vec(direct.sheen_specular),
                    direct_clearcoat_specular: rounded_vec(direct.clearcoat_specular),
                    direct_brdf_cos: rounded_vec(direct.brdf_cos),
                    direct_pdf: rounded(direct.pdf),
                    reciprocity_max_abs_error: rounded(reciprocity_error),
                });
            }
        }
    }
    assert!(
        maximum_reciprocity_error <= ABSOLUTE_COMPONENT_TOLERANCE,
        "angular corpus reciprocity error {maximum_reciprocity_error} exceeds \
         {ABSOLUTE_COMPONENT_TOLERANCE}"
    );
    AngularContractReport {
        schema: "bloom-layered-pbr-angular-reference",
        corpus_version: 1,
        material_contract_version: IRIDESCENCE_REFERENCE_VERSION,
        comparison_target:
            "tools/bloom-reference::layered_pbr::evaluate_layered_brdf (linear RGB)",
        coordinate_convention:
            "+Z normal, +X tangent, right-handed; azimuth is counter-clockwise from +X",
        coverage:
            "base, specular/IOR, clearcoat, sheen, anisotropy, iridescence, and combined; \
             3 view directions x 3 light directions per scenario",
        thresholds: Thresholds {
            serialized_decimal_places: 6,
            checked_in_regeneration: "byte exact",
            cpu_brdf_component_absolute: ABSOLUTE_COMPONENT_TOLERANCE,
            reciprocity_max_absolute: ABSOLUTE_COMPONENT_TOLERANCE,
        },
        known_model_differences: [
            "direct punctual-light BRDF only; environment IBL, SSR, exposure, and tone mapping are excluded",
            "realtime finite-highlight compression is excluded and must be compared in image space",
            "normal maps and their LEADR/Toksvig or derivative variance are evaluated by the motion corpus",
            "path tracing is stochastic; transport comparison uses converged radiance tolerances, not byte equality",
            "iridescence is the bounded Khronos Belcour/Barla Rec.709 approximation, not spectral conductor Fresnel",
        ],
        maximum_reciprocity_error: rounded(maximum_reciprocity_error),
        samples,
    }
}
