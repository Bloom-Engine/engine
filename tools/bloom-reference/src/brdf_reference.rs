//! Deterministic parameter-matrix output for Bloom's layered-PBR contract.

mod layered_pbr;
mod sheen_lut;

use glam::Vec3;
use layered_pbr::{
    evaluate_base_brdf, evaluate_layered_brdf, integrate_layered_white_furnace,
    integrate_white_furnace, iridescence_fresnel, BaseMaterial, LayeredMaterial, CLEARCOAT_IOR,
    CLEARCOAT_SPECULAR_REFERENCE_VERSION, DIELECTRIC_F0, IRIDESCENCE_REFERENCE_VERSION,
    LAYERED_PBR_REFERENCE_VERSION, MIN_PERCEPTUAL_ROUGHNESS, SHEEN_ANISOTROPY_REFERENCE_VERSION,
};
use serde::Serialize;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Serialize)]
struct MatrixSample {
    id: String,
    base_color: [f32; 3],
    metallic: f32,
    perceptual_roughness: f32,
    n_dot_v: f32,
    direct_diffuse: [f32; 3],
    direct_specular: [f32; 3],
    direct_brdf_cos: [f32; 3],
    direct_pdf: f32,
    white_furnace_reflectance: [f32; 3],
}

#[derive(Serialize)]
struct ContractReport {
    schema: &'static str,
    version: u32,
    model: &'static str,
    dielectric_f0: f32,
    minimum_perceptual_roughness: f32,
    visibility: &'static str,
    diffuse: &'static str,
    furnace_integration: &'static str,
    samples: Vec<MatrixSample>,
}

#[derive(Serialize)]
struct LayeredMatrixSample {
    id: String,
    base_color: [f32; 3],
    metallic: f32,
    perceptual_roughness: f32,
    ior: f32,
    specular_factor: f32,
    specular_color: [f32; 3],
    clearcoat_factor: f32,
    clearcoat_perceptual_roughness: f32,
    n_dot_v: f32,
    direct_diffuse: [f32; 3],
    direct_base_specular: [f32; 3],
    direct_clearcoat_specular: [f32; 3],
    direct_brdf_cos: [f32; 3],
    direct_pdf: f32,
    white_furnace_reflectance: [f32; 3],
}

#[derive(Serialize)]
struct LayeredContractReport {
    schema: &'static str,
    version: u32,
    base_contract_version: u32,
    model: &'static str,
    ior_zero_mode: &'static str,
    specular_fresnel: &'static str,
    diffuse_complement: &'static str,
    clearcoat_ior: f32,
    clearcoat_layering: &'static str,
    minimum_perceptual_roughness: f32,
    furnace_integration: &'static str,
    samples: Vec<LayeredMatrixSample>,
}

#[derive(Serialize)]
struct SheenAnisotropyMatrixSample {
    id: String,
    base_color: [f32; 3],
    metallic: f32,
    perceptual_roughness: f32,
    sheen_color: [f32; 3],
    sheen_perceptual_roughness: f32,
    anisotropy_strength: f32,
    anisotropy_rotation: f32,
    clearcoat_factor: f32,
    n_dot_v: f32,
    direct_diffuse: [f32; 3],
    direct_base_specular: [f32; 3],
    direct_sheen_specular: [f32; 3],
    direct_clearcoat_specular: [f32; 3],
    direct_brdf_cos: [f32; 3],
    direct_pdf: f32,
    white_furnace_reflectance: [f32; 3],
}

#[derive(Serialize)]
struct SheenAnisotropyContractReport {
    schema: &'static str,
    version: u32,
    previous_contract_version: u32,
    model: &'static str,
    sheen_distribution: &'static str,
    sheen_visibility: &'static str,
    sheen_layering: &'static str,
    sheen_albedo_lut: &'static str,
    anisotropic_distribution: &'static str,
    anisotropic_visibility: &'static str,
    anisotropy_frame: &'static str,
    furnace_integration: &'static str,
    samples: Vec<SheenAnisotropyMatrixSample>,
}

#[derive(Serialize)]
struct IridescenceMatrixSample {
    id: String,
    base_color: [f32; 3],
    metallic: f32,
    perceptual_roughness: f32,
    ior: f32,
    specular_factor: f32,
    specular_color: [f32; 3],
    iridescence_factor: f32,
    iridescence_ior: f32,
    iridescence_thickness_nm: f32,
    clearcoat_factor: f32,
    sheen_color: [f32; 3],
    anisotropy_strength: f32,
    n_dot_v: f32,
    raw_dielectric_thin_film_fresnel: [f32; 3],
    raw_conductor_thin_film_fresnel: [f32; 3],
    direct_diffuse: [f32; 3],
    direct_base_specular: [f32; 3],
    direct_sheen_specular: [f32; 3],
    direct_clearcoat_specular: [f32; 3],
    direct_brdf_cos: [f32; 3],
    direct_pdf: f32,
    white_furnace_reflectance: [f32; 3],
}

#[derive(Serialize)]
struct IridescenceContractReport {
    schema: &'static str,
    version: u32,
    previous_contract_version: u32,
    model: &'static str,
    spectral_integration: &'static str,
    interfaces: &'static str,
    diffuse_complement: &'static str,
    inactive_path: &'static str,
    color_space: &'static str,
    furnace_integration: &'static str,
    samples: Vec<IridescenceMatrixSample>,
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

fn build_report() -> ContractReport {
    let normal = Vec3::Z;
    let light = direction(0.5, std::f32::consts::FRAC_PI_4);
    let mut samples = Vec::new();
    for (color_name, base_color) in [
        ("neutral", Vec3::splat(0.18)),
        ("red", Vec3::new(0.8, 0.08, 0.03)),
    ] {
        for metallic in [0.0, 1.0] {
            for roughness in [0.04, 0.25, 0.5, 1.0] {
                for n_dot_v in [0.1, 0.5, 1.0] {
                    let material = BaseMaterial {
                        base_color,
                        metallic,
                        perceptual_roughness: roughness,
                    };
                    let view = direction(n_dot_v, 0.0);
                    let direct = evaluate_base_brdf(material, normal, view, light);
                    let furnace = integrate_white_furnace(material, n_dot_v, 96, 192);
                    samples.push(MatrixSample {
                        id: format!("{color_name}-m{metallic:.0}-r{roughness:.2}-v{n_dot_v:.1}"),
                        base_color: base_color.to_array(),
                        metallic,
                        perceptual_roughness: roughness,
                        n_dot_v,
                        direct_diffuse: rounded_vec(direct.diffuse),
                        direct_specular: rounded_vec(direct.specular),
                        direct_brdf_cos: rounded_vec(direct.brdf_cos),
                        direct_pdf: rounded(direct.pdf),
                        white_furnace_reflectance: rounded_vec(furnace),
                    });
                }
            }
        }
    }
    ContractReport {
        schema: "bloom-layered-pbr-reference",
        version: LAYERED_PBR_REFERENCE_VERSION,
        model: "metallic-roughness / correlated GGX / Schlick / energy-normalized Burley",
        dielectric_f0: DIELECTRIC_F0,
        minimum_perceptual_roughness: MIN_PERCEPTUAL_ROUGHNESS,
        visibility: "height-correlated Smith, including 1/(4 NdotV NdotL)",
        diffuse: "energy-normalized Burley, reciprocal view/light Fresnel transmission, 1/pi",
        furnace_integration: "96x192 deterministic GGX-VNDF and cosine samples",
        samples,
    }
}

fn layered_scenarios() -> Vec<(&'static str, LayeredMaterial)> {
    let material = |base_color, metallic, roughness| {
        LayeredMaterial::from_base(BaseMaterial {
            base_color,
            metallic,
            perceptual_roughness: roughness,
        })
    };
    vec![
        (
            "base-neutral-dielectric",
            material(Vec3::splat(0.18), 0.0, 0.5),
        ),
        (
            "base-red-mixed",
            material(Vec3::new(0.8, 0.08, 0.03), 0.5, 0.25),
        ),
        (
            "base-gold-conductor",
            material(Vec3::new(1.0, 0.71, 0.29), 1.0, 0.4),
        ),
        (
            "ior-water",
            LayeredMaterial {
                ior: 1.33,
                ..material(Vec3::splat(0.7), 0.0, 0.2)
            },
        ),
        (
            "ior-diamond",
            LayeredMaterial {
                ior: 2.42,
                ..material(Vec3::splat(0.7), 0.0, 0.2)
            },
        ),
        (
            "ior-zero-compatibility",
            LayeredMaterial {
                ior: 0.0,
                specular_color: Vec3::new(0.9, 0.35, 0.1),
                ..material(Vec3::new(0.4, 0.1, 0.03), 0.0, 0.35)
            },
        ),
        (
            "specular-disabled",
            LayeredMaterial {
                specular_factor: 0.0,
                ..material(Vec3::new(0.5, 0.15, 0.04), 0.0, 0.55)
            },
        ),
        (
            "specular-half",
            LayeredMaterial {
                specular_factor: 0.5,
                ..material(Vec3::new(0.5, 0.15, 0.04), 0.0, 0.55)
            },
        ),
        (
            "specular-colored",
            LayeredMaterial {
                specular_color: Vec3::new(1.5, 0.4, 0.2),
                ..material(Vec3::new(0.5, 0.15, 0.04), 0.0, 0.55)
            },
        ),
        (
            "clearcoat-smooth",
            LayeredMaterial {
                clearcoat_factor: 1.0,
                clearcoat_perceptual_roughness: 0.04,
                ..material(Vec3::new(0.8, 0.08, 0.03), 0.0, 0.7)
            },
        ),
        (
            "clearcoat-rough",
            LayeredMaterial {
                clearcoat_factor: 1.0,
                clearcoat_perceptual_roughness: 0.75,
                ..material(Vec3::new(1.0, 0.71, 0.29), 1.0, 0.7)
            },
        ),
        (
            "clearcoat-half",
            LayeredMaterial {
                clearcoat_factor: 0.5,
                clearcoat_perceptual_roughness: 0.2,
                ..material(Vec3::new(0.8, 0.08, 0.03), 0.5, 0.35)
            },
        ),
        (
            "clearcoat-specular-ior-combined",
            LayeredMaterial {
                ior: 1.76,
                specular_factor: 0.7,
                specular_color: Vec3::new(1.2, 0.7, 0.3),
                clearcoat_factor: 0.85,
                clearcoat_perceptual_roughness: 0.14,
                ..material(Vec3::new(0.12, 0.35, 0.8), 0.15, 0.42)
            },
        ),
    ]
}

fn build_layered_report() -> LayeredContractReport {
    let normal = Vec3::Z;
    let light = direction(0.5, std::f32::consts::FRAC_PI_4);
    let mut samples = Vec::new();
    for (name, material) in layered_scenarios() {
        for n_dot_v in [0.1, 0.5, 1.0] {
            let view = direction(n_dot_v, 0.0);
            let direct = evaluate_layered_brdf(material, normal, view, light);
            let furnace = integrate_layered_white_furnace(material, n_dot_v, 96, 192);
            samples.push(LayeredMatrixSample {
                id: format!("{name}-v{n_dot_v:.1}"),
                base_color: material.base.base_color.to_array(),
                metallic: material.base.metallic,
                perceptual_roughness: material.base.perceptual_roughness,
                ior: material.ior,
                specular_factor: material.specular_factor,
                specular_color: material.specular_color.to_array(),
                clearcoat_factor: material.clearcoat_factor,
                clearcoat_perceptual_roughness: material.clearcoat_perceptual_roughness,
                n_dot_v,
                direct_diffuse: rounded_vec(direct.diffuse),
                direct_base_specular: rounded_vec(direct.base_specular),
                direct_clearcoat_specular: rounded_vec(direct.clearcoat_specular),
                direct_brdf_cos: rounded_vec(direct.brdf_cos),
                direct_pdf: rounded(direct.pdf),
                white_furnace_reflectance: rounded_vec(furnace),
            });
        }
    }
    LayeredContractReport {
        schema: "bloom-layered-pbr-reference",
        version: CLEARCOAT_SPECULAR_REFERENCE_VERSION,
        base_contract_version: LAYERED_PBR_REFERENCE_VERSION,
        model: "version-1 base + KHR clearcoat/specular/IOR",
        ior_zero_mode: "positive-infinity compatibility mode; dielectric F0=1",
        specular_fresnel: "IOR F0 * color (clamped before factor), explicit factor F90",
        diffuse_complement: "max-channel dielectric Fresnel at reciprocal view/light interfaces",
        clearcoat_ior: CLEARCOAT_IOR,
        clearcoat_layering:
            "fixed-IOR GGX; reciprocal view/light transmission attenuates the full base",
        minimum_perceptual_roughness: MIN_PERCEPTUAL_ROUGHNESS,
        furnace_integration: "96x192 deterministic per-lobe GGX-VNDF and cosine samples",
        samples,
    }
}

fn sheen_anisotropy_scenarios() -> Vec<(&'static str, LayeredMaterial)> {
    let material = |base_color, metallic, roughness| {
        LayeredMaterial::from_base(BaseMaterial {
            base_color,
            metallic,
            perceptual_roughness: roughness,
        })
    };
    vec![
        (
            "v2-default",
            LayeredMaterial {
                clearcoat_factor: 0.5,
                clearcoat_perceptual_roughness: 0.2,
                ..material(Vec3::new(0.8, 0.08, 0.03), 0.2, 0.4)
            },
        ),
        (
            "velvet-smooth",
            LayeredMaterial {
                sheen_color: Vec3::new(0.9, 0.08, 0.03),
                sheen_perceptual_roughness: 0.08,
                ..material(Vec3::new(0.12, 0.02, 0.01), 0.0, 0.7)
            },
        ),
        (
            "velvet-rough",
            LayeredMaterial {
                sheen_color: Vec3::new(0.7, 0.3, 0.08),
                sheen_perceptual_roughness: 0.8,
                ..material(Vec3::new(0.18, 0.04, 0.01), 0.0, 0.85)
            },
        ),
        (
            "brushed-half",
            LayeredMaterial {
                anisotropy_strength: 0.5,
                anisotropy_rotation: 0.0,
                ..material(Vec3::new(0.9, 0.45, 0.12), 1.0, 0.28)
            },
        ),
        (
            "brushed-strong",
            LayeredMaterial {
                anisotropy_strength: 1.0,
                anisotropy_rotation: 0.0,
                ..material(Vec3::new(0.9, 0.45, 0.12), 1.0, 0.28)
            },
        ),
        (
            "brushed-rotated-45",
            LayeredMaterial {
                anisotropy_strength: 0.85,
                anisotropy_rotation: std::f32::consts::FRAC_PI_4,
                ..material(Vec3::new(0.75, 0.78, 0.82), 1.0, 0.32)
            },
        ),
        (
            "brushed-rotated-90",
            LayeredMaterial {
                anisotropy_strength: 0.85,
                anisotropy_rotation: std::f32::consts::FRAC_PI_2,
                ..material(Vec3::new(0.75, 0.78, 0.82), 1.0, 0.32)
            },
        ),
        (
            "fabric-anisotropic",
            LayeredMaterial {
                sheen_color: Vec3::new(0.25, 0.45, 0.9),
                sheen_perceptual_roughness: 0.45,
                anisotropy_strength: 0.7,
                anisotropy_rotation: 1.1,
                ..material(Vec3::new(0.03, 0.08, 0.3), 0.15, 0.4)
            },
        ),
        (
            "coated-fabric",
            LayeredMaterial {
                sheen_color: Vec3::new(0.8, 0.2, 0.08),
                sheen_perceptual_roughness: 0.5,
                clearcoat_factor: 0.7,
                clearcoat_perceptual_roughness: 0.18,
                ..material(Vec3::new(0.2, 0.03, 0.01), 0.0, 0.65)
            },
        ),
        (
            "all-v3-lobes",
            LayeredMaterial {
                ior: 1.72,
                specular_factor: 0.8,
                specular_color: Vec3::new(1.1, 0.7, 0.4),
                clearcoat_factor: 0.65,
                clearcoat_perceptual_roughness: 0.2,
                sheen_color: Vec3::new(0.2, 0.55, 0.9),
                sheen_perceptual_roughness: 0.38,
                anisotropy_strength: 0.75,
                anisotropy_rotation: 0.63,
                ..material(Vec3::new(0.05, 0.2, 0.6), 0.35, 0.36)
            },
        ),
    ]
}

fn build_sheen_anisotropy_report() -> SheenAnisotropyContractReport {
    let normal = Vec3::Z;
    let light = direction(0.5, std::f32::consts::FRAC_PI_4);
    let mut samples = Vec::new();
    for (name, material) in sheen_anisotropy_scenarios() {
        for n_dot_v in [0.1, 0.5, 1.0] {
            let view = direction(n_dot_v, 0.0);
            let direct = evaluate_layered_brdf(material, normal, view, light);
            let furnace = integrate_layered_white_furnace(material, n_dot_v, 96, 192);
            samples.push(SheenAnisotropyMatrixSample {
                id: format!("{name}-v{n_dot_v:.1}"),
                base_color: material.base.base_color.to_array(),
                metallic: material.base.metallic,
                perceptual_roughness: material.base.perceptual_roughness,
                sheen_color: material.sheen_color.to_array(),
                sheen_perceptual_roughness: material.sheen_perceptual_roughness,
                anisotropy_strength: material.anisotropy_strength,
                anisotropy_rotation: material.anisotropy_rotation,
                clearcoat_factor: material.clearcoat_factor,
                n_dot_v,
                direct_diffuse: rounded_vec(direct.diffuse),
                direct_base_specular: rounded_vec(direct.base_specular),
                direct_sheen_specular: rounded_vec(direct.sheen_specular),
                direct_clearcoat_specular: rounded_vec(direct.clearcoat_specular),
                direct_brdf_cos: rounded_vec(direct.brdf_cos),
                direct_pdf: rounded(direct.pdf),
                white_furnace_reflectance: rounded_vec(furnace),
            });
        }
    }
    SheenAnisotropyContractReport {
        schema: "bloom-layered-pbr-reference",
        version: SHEEN_ANISOTROPY_REFERENCE_VERSION,
        previous_contract_version: CLEARCOAT_SPECULAR_REFERENCE_VERSION,
        model: "version-2 layered PBR + KHR sheen/anisotropy",
        sheen_distribution: "Charlie with alpha_g = perceptual roughness squared",
        sheen_visibility: "full fitted Charlie lambda; no Ashikhmin shortcut",
        sheen_layering:
            "max-channel reciprocal view/light directional-albedo scaling below clearcoat",
        sheen_albedo_lut: "128x128 R16F, 4096 Charlie-importance samples per texel",
        anisotropic_distribution: "Burley anisotropic GGX; alpha_t=mix(alpha,1,strength^2)",
        anisotropic_visibility: "height-correlated anisotropic Smith",
        anisotropy_frame:
            "counter-clockwise radians from glTF tangent; mirrored handedness retained",
        furnace_integration:
            "96x192 deterministic anisotropic GGX-VNDF, Charlie, clearcoat, and cosine samples",
        samples,
    }
}

fn iridescence_scenarios() -> Vec<(&'static str, LayeredMaterial)> {
    let material = |base_color, metallic, roughness| {
        LayeredMaterial::from_base(BaseMaterial {
            base_color,
            metallic,
            perceptual_roughness: roughness,
        })
    };
    vec![
        (
            "v3-inactive",
            LayeredMaterial {
                clearcoat_factor: 0.55,
                clearcoat_perceptual_roughness: 0.2,
                sheen_color: Vec3::new(0.3, 0.55, 0.9),
                sheen_perceptual_roughness: 0.4,
                anisotropy_strength: 0.7,
                anisotropy_rotation: 0.63,
                ..material(Vec3::new(0.08, 0.3, 0.75), 0.35, 0.36)
            },
        ),
        (
            "dielectric-thin-100nm",
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_thickness_nm: 100.0,
                ..material(Vec3::splat(0.55), 0.0, 0.3)
            },
        ),
        (
            "dielectric-mid-400nm",
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_thickness_nm: 400.0,
                ..material(Vec3::splat(0.55), 0.0, 0.3)
            },
        ),
        (
            "dielectric-thick-800nm",
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_thickness_nm: 800.0,
                ..material(Vec3::splat(0.55), 0.0, 0.3)
            },
        ),
        (
            "iridescence-half-strength",
            LayeredMaterial {
                iridescence_factor: 0.5,
                iridescence_thickness_nm: 400.0,
                ..material(Vec3::new(0.65, 0.18, 0.05), 0.0, 0.45)
            },
        ),
        (
            "film-ior-air",
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_ior: 1.0,
                iridescence_thickness_nm: 400.0,
                ..material(Vec3::splat(0.55), 0.0, 0.3)
            },
        ),
        (
            "film-ior-high",
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_ior: 2.0,
                iridescence_thickness_nm: 400.0,
                ..material(Vec3::splat(0.55), 0.0, 0.3)
            },
        ),
        (
            "gold-conductor-film",
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_ior: 1.45,
                iridescence_thickness_nm: 400.0,
                ..material(Vec3::new(1.0, 0.71, 0.29), 1.0, 0.28)
            },
        ),
        (
            "mixed-metal-film",
            LayeredMaterial {
                iridescence_factor: 0.85,
                iridescence_ior: 1.35,
                iridescence_thickness_nm: 520.0,
                ..material(Vec3::new(0.8, 0.12, 0.03), 0.5, 0.42)
            },
        ),
        (
            "colored-specular-film",
            LayeredMaterial {
                ior: 1.76,
                specular_factor: 0.7,
                specular_color: Vec3::new(1.2, 0.6, 0.25),
                iridescence_factor: 1.0,
                iridescence_ior: 1.25,
                iridescence_thickness_nm: 310.0,
                ..material(Vec3::new(0.15, 0.4, 0.8), 0.0, 0.38)
            },
        ),
        (
            "clearcoat-over-film",
            LayeredMaterial {
                clearcoat_factor: 0.8,
                clearcoat_perceptual_roughness: 0.16,
                iridescence_factor: 1.0,
                iridescence_ior: 1.3,
                iridescence_thickness_nm: 450.0,
                ..material(Vec3::new(0.7, 0.08, 0.02), 0.1, 0.5)
            },
        ),
        (
            "all-v4-lobes",
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
                ..material(Vec3::new(0.05, 0.2, 0.6), 0.35, 0.36)
            },
        ),
        (
            "zero-thickness-inactive",
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_ior: 2.0,
                iridescence_thickness_nm: 0.0,
                ..material(Vec3::new(0.3, 0.6, 0.9), 0.2, 0.4)
            },
        ),
    ]
}

fn build_iridescence_report() -> IridescenceContractReport {
    let normal = Vec3::Z;
    let light = direction(0.5, std::f32::consts::FRAC_PI_4);
    let mut samples = Vec::new();
    for (name, material) in iridescence_scenarios() {
        for n_dot_v in [0.1, 0.5, 1.0] {
            let view = direction(n_dot_v, 0.0);
            let direct = evaluate_layered_brdf(material, normal, view, light);
            let furnace = integrate_layered_white_furnace(material, n_dot_v, 96, 192);
            let dielectric_thin_film = iridescence_fresnel(
                1.0,
                material.iridescence_ior,
                n_dot_v,
                material.iridescence_thickness_nm,
                material.dielectric_f0(),
            );
            let conductor_thin_film = iridescence_fresnel(
                1.0,
                material.iridescence_ior,
                n_dot_v,
                material.iridescence_thickness_nm,
                material.base.base_color,
            );
            samples.push(IridescenceMatrixSample {
                id: format!("{name}-v{n_dot_v:.1}"),
                base_color: material.base.base_color.to_array(),
                metallic: material.base.metallic,
                perceptual_roughness: material.base.perceptual_roughness,
                ior: material.ior,
                specular_factor: material.specular_factor,
                specular_color: material.specular_color.to_array(),
                iridescence_factor: material.iridescence_factor,
                iridescence_ior: material.iridescence_ior,
                iridescence_thickness_nm: material.iridescence_thickness_nm,
                clearcoat_factor: material.clearcoat_factor,
                sheen_color: material.sheen_color.to_array(),
                anisotropy_strength: material.anisotropy_strength,
                n_dot_v,
                raw_dielectric_thin_film_fresnel: rounded_vec(dielectric_thin_film),
                raw_conductor_thin_film_fresnel: rounded_vec(conductor_thin_film),
                direct_diffuse: rounded_vec(direct.diffuse),
                direct_base_specular: rounded_vec(direct.base_specular),
                direct_sheen_specular: rounded_vec(direct.sheen_specular),
                direct_clearcoat_specular: rounded_vec(direct.clearcoat_specular),
                direct_brdf_cos: rounded_vec(direct.brdf_cos),
                direct_pdf: rounded(direct.pdf),
                white_furnace_reflectance: rounded_vec(furnace),
            });
        }
    }
    IridescenceContractReport {
        schema: "bloom-layered-pbr-reference",
        version: IRIDESCENCE_REFERENCE_VERSION,
        previous_contract_version: SHEEN_ANISOTROPY_REFERENCE_VERSION,
        model: "version-3 layered PBR + KHR_materials_iridescence",
        spectral_integration: "Belcour/Barla Fourier sensitivity, orders 0 through 2",
        interfaces: "air / dielectric thin film / dielectric-or-approximated-conductor base",
        diffuse_complement:
            "reciprocal view/light transmission from max-channel blended dielectric Fresnel",
        inactive_path: "factor zero or thickness zero is exactly the version-3 evaluator",
        color_space: "linear Rec.709, bounded to physical reflectance [0,1]",
        furnace_integration:
            "96x192 deterministic anisotropic GGX-VNDF, Charlie, clearcoat, and cosine samples",
        samples,
    }
}

struct Options {
    output: Option<PathBuf>,
    version: u32,
}

fn options() -> Result<Option<Options>, String> {
    let mut args = std::env::args().skip(1);
    let mut output = None;
    let mut version = LAYERED_PBR_REFERENCE_VERSION;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                let value = args.next().ok_or("--out requires a path")?;
                output = Some(PathBuf::from(value));
            }
            "--version" => {
                let value = args.next().ok_or("--version requires 1, 2, 3, or 4")?;
                version = value
                    .parse::<u32>()
                    .map_err(|_| "--version requires 1, 2, 3, or 4".to_owned())?;
                if !matches!(
                    version,
                    LAYERED_PBR_REFERENCE_VERSION
                        | CLEARCOAT_SPECULAR_REFERENCE_VERSION
                        | SHEEN_ANISOTROPY_REFERENCE_VERSION
                        | IRIDESCENCE_REFERENCE_VERSION
                ) {
                    return Err("--version requires 1, 2, 3, or 4".to_owned());
                }
            }
            "--help" | "-h" => {
                println!("usage: bloom-brdf-reference [--version 1|2|3|4] [--out REPORT.json]");
                return Ok(None);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    Ok(Some(Options { output, version }))
}

fn main() -> ExitCode {
    let options = match options() {
        Ok(Some(options)) => options,
        Ok(None) => return ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let encoded = match options.version {
        LAYERED_PBR_REFERENCE_VERSION => serde_json::to_string_pretty(&build_report()),
        CLEARCOAT_SPECULAR_REFERENCE_VERSION => {
            serde_json::to_string_pretty(&build_layered_report())
        }
        SHEEN_ANISOTROPY_REFERENCE_VERSION => {
            serde_json::to_string_pretty(&build_sheen_anisotropy_report())
        }
        IRIDESCENCE_REFERENCE_VERSION => serde_json::to_string_pretty(&build_iridescence_report()),
        _ => unreachable!("version validated by argument parser"),
    }
    .expect("serialize BRDF report");
    if let Some(path) = options.output {
        if let Err(error) = std::fs::write(&path, format!("{encoded}\n")) {
            eprintln!("error writing {}: {error}", path.display());
            return ExitCode::from(1);
        }
        println!("wrote {}", path.display());
    } else {
        println!("{encoded}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_v1_matrix_matches_the_reference_evaluator() {
        let encoded = serde_json::to_string_pretty(&build_report()).expect("serialize BRDF matrix");
        assert_eq!(
            format!("{encoded}\n"),
            include_str!("../reference/layered-pbr-v1.json")
        );
    }

    #[test]
    fn checked_in_v2_matrix_matches_the_reference_evaluator() {
        let encoded =
            serde_json::to_string_pretty(&build_layered_report()).expect("serialize BRDF matrix");
        assert_eq!(
            format!("{encoded}\n"),
            include_str!("../reference/layered-pbr-v2.json")
        );
    }

    #[test]
    fn checked_in_v3_matrix_matches_the_reference_evaluator() {
        let encoded = serde_json::to_string_pretty(&build_sheen_anisotropy_report())
            .expect("serialize BRDF matrix");
        assert_eq!(
            format!("{encoded}\n"),
            include_str!("../reference/layered-pbr-v3.json")
        );
    }

    #[test]
    fn checked_in_v4_matrix_matches_the_reference_evaluator() {
        let encoded =
            serde_json::to_string_pretty(&build_iridescence_report()).expect("serialize BRDF matrix");
        assert_eq!(
            format!("{encoded}\n"),
            include_str!("../reference/layered-pbr-v4.json")
        );
    }
}
