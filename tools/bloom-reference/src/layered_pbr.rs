//! Versioned, side-effect-free reference evaluation for Bloom's layered PBR.
//!
//! Version 1 intentionally describes only the existing metallic/roughness base
//! layer. Later lobes must compose through this contract instead of creating
//! another unrelated shader formula. The deterministic corpus consumes this
//! module directly; realtime and transport paths are wired only after each
//! version has passed its independent energy and reciprocity gates.

use glam::Vec3;

#[allow(dead_code)]
pub const LAYERED_PBR_REFERENCE_VERSION: u32 = 1;
pub const CLEARCOAT_SPECULAR_REFERENCE_VERSION: u32 = 2;
pub const SHEEN_ANISOTROPY_REFERENCE_VERSION: u32 = 3;
pub const IRIDESCENCE_REFERENCE_VERSION: u32 = 4;
pub const DIELECTRIC_F0: f32 = 0.04;
pub const MIN_PERCEPTUAL_ROUGHNESS: f32 = 0.04;
pub const DEFAULT_DIELECTRIC_IOR: f32 = 1.5;
pub const CLEARCOAT_IOR: f32 = 1.5;
pub const DEFAULT_IRIDESCENCE_IOR: f32 = 1.3;
pub const DEFAULT_IRIDESCENCE_THICKNESS_NM: f32 = 400.0;
const SHEEN_ALBEDO_LUT_SIZE: usize = 128;
const SHEEN_ALBEDO_LUT: &[u8] =
    include_bytes!("../../../native/shared/shaders/sheen_albedo_lut_r16f.bin");

#[derive(Clone, Copy, Debug)]
pub struct BaseMaterial {
    pub base_color: Vec3,
    pub metallic: f32,
    pub perceptual_roughness: f32,
}

impl BaseMaterial {
    #[allow(dead_code)]
    pub fn validated(self) -> Self {
        Self {
            base_color: self.base_color.clamp(Vec3::ZERO, Vec3::ONE),
            metallic: self.metallic.clamp(0.0, 1.0),
            perceptual_roughness: self
                .perceptual_roughness
                .clamp(MIN_PERCEPTUAL_ROUGHNESS, 1.0),
        }
    }

    pub fn f0(self) -> Vec3 {
        Vec3::splat(DIELECTRIC_F0).lerp(self.base_color, self.metallic)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BaseBrdfEvaluation {
    /// BRDF without the incident cosine.
    pub diffuse: Vec3,
    /// BRDF without the incident cosine.
    pub specular: Vec3,
    /// `(diffuse + specular) * NdotL`, ready for light transport.
    pub brdf_cos: Vec3,
    /// The base sampler's current mixture PDF, used by reference MIS.
    pub pdf: f32,
}

/// Version-2 clearcoat and dielectric-specular parameters layered over the
/// version-1 metallic/roughness base.
///
/// `ior == 0` is the glTF specular-glossiness compatibility mode and maps to
/// unit dielectric F0. `specular_color` is intentionally not capped at one:
/// KHR_materials_specular permits larger factors and clamps the resulting F0.
#[derive(Clone, Copy, Debug)]
pub struct LayeredMaterial {
    pub base: BaseMaterial,
    pub ior: f32,
    pub specular_factor: f32,
    pub specular_color: Vec3,
    pub clearcoat_factor: f32,
    pub clearcoat_perceptual_roughness: f32,
    pub sheen_color: Vec3,
    pub sheen_perceptual_roughness: f32,
    pub anisotropy_strength: f32,
    pub anisotropy_rotation: f32,
    /// Effective KHR_materials_iridescence strength after texture sampling.
    pub iridescence_factor: f32,
    /// Index of refraction of the dielectric thin-film layer.
    pub iridescence_ior: f32,
    /// Effective thin-film thickness after texture sampling, in nanometers.
    pub iridescence_thickness_nm: f32,
}

impl LayeredMaterial {
    pub fn from_base(base: BaseMaterial) -> Self {
        Self {
            base,
            ior: DEFAULT_DIELECTRIC_IOR,
            specular_factor: 1.0,
            specular_color: Vec3::ONE,
            clearcoat_factor: 0.0,
            clearcoat_perceptual_roughness: 0.0,
            sheen_color: Vec3::ZERO,
            sheen_perceptual_roughness: 0.0,
            anisotropy_strength: 0.0,
            anisotropy_rotation: 0.0,
            iridescence_factor: 0.0,
            iridescence_ior: DEFAULT_IRIDESCENCE_IOR,
            iridescence_thickness_nm: DEFAULT_IRIDESCENCE_THICKNESS_NM,
        }
    }

    #[allow(dead_code)]
    pub fn validated(self) -> Self {
        let finite_non_negative = |value: f32, fallback: f32| {
            if value.is_finite() {
                value.max(0.0)
            } else {
                fallback
            }
        };
        let ior = if !self.ior.is_finite() {
            DEFAULT_DIELECTRIC_IOR
        } else if self.ior == 0.0 {
            0.0
        } else {
            self.ior.max(1.0)
        };
        let specular_color = Vec3::new(
            finite_non_negative(self.specular_color.x, 1.0),
            finite_non_negative(self.specular_color.y, 1.0),
            finite_non_negative(self.specular_color.z, 1.0),
        );
        Self {
            base: self.base.validated(),
            ior,
            specular_factor: if self.specular_factor.is_finite() {
                self.specular_factor.clamp(0.0, 1.0)
            } else {
                1.0
            },
            specular_color,
            clearcoat_factor: if self.clearcoat_factor.is_finite() {
                self.clearcoat_factor.clamp(0.0, 1.0)
            } else {
                0.0
            },
            clearcoat_perceptual_roughness: if self.clearcoat_perceptual_roughness.is_finite() {
                self.clearcoat_perceptual_roughness.clamp(0.0, 1.0)
            } else {
                0.0
            },
            sheen_color: self.sheen_color.clamp(Vec3::ZERO, Vec3::ONE),
            sheen_perceptual_roughness: if self.sheen_perceptual_roughness.is_finite() {
                self.sheen_perceptual_roughness.clamp(0.0, 1.0)
            } else {
                0.0
            },
            anisotropy_strength: if self.anisotropy_strength.is_finite() {
                self.anisotropy_strength.clamp(0.0, 1.0)
            } else {
                0.0
            },
            anisotropy_rotation: if self.anisotropy_rotation.is_finite() {
                self.anisotropy_rotation
            } else {
                0.0
            },
            iridescence_factor: if self.iridescence_factor.is_finite() {
                self.iridescence_factor.clamp(0.0, 1.0)
            } else {
                0.0
            },
            iridescence_ior: if self.iridescence_ior.is_finite() {
                self.iridescence_ior.max(1.0)
            } else {
                DEFAULT_IRIDESCENCE_IOR
            },
            iridescence_thickness_nm: if self.iridescence_thickness_nm.is_finite() {
                self.iridescence_thickness_nm.max(0.0)
            } else {
                DEFAULT_IRIDESCENCE_THICKNESS_NM
            },
        }
    }

    pub fn dielectric_f0(self) -> Vec3 {
        (self.specular_color * ior_f0(self.ior)).min(Vec3::ONE) * self.specular_factor
    }

    fn base_fresnel_v3(self, cos_theta: f32) -> Vec3 {
        let dielectric = fresnel_schlick_f90(
            cos_theta,
            self.dielectric_f0(),
            Vec3::splat(self.specular_factor),
        );
        let conductor = fresnel_schlick(cos_theta, self.base.base_color);
        dielectric.lerp(conductor, self.base.metallic)
    }

    fn has_iridescence(self) -> bool {
        self.iridescence_factor > 0.0 && self.iridescence_thickness_nm > 0.0
    }

    fn base_fresnel(self, cos_theta: f32) -> Vec3 {
        let base = self.base_fresnel_v3(cos_theta);
        if !self.has_iridescence() {
            return base;
        }
        let dielectric = iridescence_fresnel(
            1.0,
            self.iridescence_ior,
            cos_theta,
            self.iridescence_thickness_nm,
            self.dielectric_f0(),
        );
        let conductor = iridescence_fresnel(
            1.0,
            self.iridescence_ior,
            cos_theta,
            self.iridescence_thickness_nm,
            self.base.base_color,
        );
        base.lerp(
            dielectric.lerp(conductor, self.base.metallic),
            self.iridescence_factor,
        )
    }

    fn has_specular_ior_lobe(self) -> bool {
        self.ior != DEFAULT_DIELECTRIC_IOR
            || self.specular_factor != 1.0
            || self.specular_color != Vec3::ONE
    }

    fn dielectric_transmission(self, cos_theta: f32) -> f32 {
        let base = fresnel_schlick_f90(
            cos_theta,
            self.dielectric_f0(),
            Vec3::splat(self.specular_factor),
        );
        let fresnel = if self.has_iridescence() {
            base.lerp(
                iridescence_fresnel(
                    1.0,
                    self.iridescence_ior,
                    cos_theta,
                    self.iridescence_thickness_nm,
                    self.dielectric_f0(),
                ),
                self.iridescence_factor,
            )
        } else {
            base
        };
        (1.0 - fresnel.max_element()).clamp(0.0, 1.0)
    }

    fn clearcoat_fresnel(self, cos_theta: f32) -> f32 {
        self.clearcoat_factor * fresnel_schlick(cos_theta, Vec3::splat(ior_f0(CLEARCOAT_IOR))).x
    }

    fn clearcoat_transmission(self, cos_theta: f32) -> f32 {
        1.0 - self.clearcoat_fresnel(cos_theta)
    }

    fn clearcoat_roughness(self) -> f32 {
        self.clearcoat_perceptual_roughness
            .max(MIN_PERCEPTUAL_ROUGHNESS)
    }

    fn sheen_roughness(self) -> f32 {
        self.sheen_perceptual_roughness.max(1e-3)
    }

    fn has_sheen(self) -> bool {
        self.sheen_color.max_element() > 0.0
    }

    fn anisotropic_alpha(self) -> [f32; 2] {
        let alpha = self.base.perceptual_roughness * self.base.perceptual_roughness;
        [
            alpha + (1.0 - alpha) * self.anisotropy_strength.powi(2),
            alpha,
        ]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LayeredBrdfEvaluation {
    /// Energy-compensated base diffuse BRDF, without the incident cosine.
    pub diffuse: Vec3,
    /// Energy-compensated base GGX BRDF, without the incident cosine.
    pub base_specular: Vec3,
    /// Clearcoat GGX BRDF, without the incident cosine.
    pub clearcoat_specular: Vec3,
    /// Charlie micro-fiber BRDF below clearcoat, without the incident cosine.
    pub sheen_specular: Vec3,
    /// Sum of every lobe multiplied by `NdotL`.
    pub brdf_cos: Vec3,
    /// Three-lobe mixture PDF used by the future reference transport path.
    pub pdf: f32,
}

/// Schlick's Fresnel approximation. `f0` is reflectance at normal
/// incidence; `cos_theta` is the view/half-vector dot product.
#[inline]
pub fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    f0 + (Vec3::ONE - f0) * (m * m * m * m * m)
}

/// Schlick Fresnel with an explicit grazing reflectance. This is required by
/// KHR_materials_specular: its scalar factor scales both dielectric F0 and
/// F90, while pure conductors remain unaffected.
#[inline]
pub fn fresnel_schlick_f90(cos_theta: f32, f0: Vec3, f90: Vec3) -> Vec3 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    f0 + (f90 - f0) * (m * m * m * m * m)
}

/// Dielectric normal-incidence reflectance for an air/material interface.
///
/// glTF reserves IOR zero for specular-glossiness compatibility, where the
/// effective IOR is positive infinity and F0 is exactly one.
#[inline]
pub fn ior_f0(ior: f32) -> f32 {
    if ior == 0.0 {
        1.0
    } else {
        let ior = ior.max(1.0);
        let ratio = (ior - 1.0) / (ior + 1.0);
        ratio * ratio
    }
}

fn fresnel0_to_ior(f0: Vec3) -> Vec3 {
    let value = f0.clamp(Vec3::ZERO, Vec3::splat(0.9999));
    let root = Vec3::new(value.x.sqrt(), value.y.sqrt(), value.z.sqrt());
    (Vec3::ONE + root) / (Vec3::ONE - root)
}

fn ior_to_fresnel0(transmitted_ior: Vec3, incident_ior: f32) -> Vec3 {
    let incident = Vec3::splat(incident_ior);
    let ratio = (transmitted_ior - incident) / (transmitted_ior + incident);
    ratio * ratio
}

fn eval_iridescence_sensitivity(optical_path_difference_nm: f32, shift: Vec3) -> Vec3 {
    let phase = std::f32::consts::TAU * optical_path_difference_nm * 1.0e-9;
    let phase_squared = phase * phase;
    let value = Vec3::new(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let position = Vec3::new(1.6810e6, 1.7953e6, 2.2084e6);
    let variance = Vec3::new(4.3278e9, 9.3046e9, 6.6121e9);
    let cosine = Vec3::new(
        (position.x * phase + shift.x).cos(),
        (position.y * phase + shift.y).cos(),
        (position.z * phase + shift.z).cos(),
    );
    let gaussian = Vec3::new(
        (-phase_squared * variance.x).exp(),
        (-phase_squared * variance.y).exp(),
        (-phase_squared * variance.z).exp(),
    );
    let variance_scale = Vec3::new(
        (std::f32::consts::TAU * variance.x).sqrt(),
        (std::f32::consts::TAU * variance.y).sqrt(),
        (std::f32::consts::TAU * variance.z).sqrt(),
    );
    let mut xyz = value * variance_scale * cosine * gaussian;
    xyz.x += 9.7470e-14
        * (std::f32::consts::TAU * 4.5282e9).sqrt()
        * (2.2399e6 * phase + shift.x).cos()
        * (-4.5282e9 * phase_squared).exp();
    xyz /= 1.0685e-7;

    Vec3::new(
        3.240_454_2 * xyz.x - 0.969_266 * xyz.y + 0.055_643_4 * xyz.z,
        -1.537_138_5 * xyz.x + 1.876_010_8 * xyz.y - 0.204_025_9 * xyz.z,
        -0.498_531_4 * xyz.x + 0.041_556 * xyz.y + 1.057_225_2 * xyz.z,
    )
}

/// Khronos' bounded Belcour/Barla thin-film Fresnel approximation.
///
/// The result is expressed in linear Rec.709. The ratified glTF formulation
/// clamps negative out-of-gamut components; Bloom also caps the upper bound
/// because this value is a reflectance used by an energy-conserving BRDF.
pub fn iridescence_fresnel(
    outside_ior: f32,
    film_ior: f32,
    cos_theta_1: f32,
    thickness_nm: f32,
    base_f0: Vec3,
) -> Vec3 {
    let outside_ior = outside_ior.max(1.0e-4);
    let cos_theta_1 = cos_theta_1.clamp(0.0, 1.0);
    let thickness_nm = thickness_nm.max(0.0);
    let transition = (thickness_nm / 0.03).clamp(0.0, 1.0);
    let transition = transition * transition * (3.0 - 2.0 * transition);
    let film_ior = outside_ior + (film_ior.max(1.0) - outside_ior) * transition;

    let sin_theta_2_squared = (outside_ior / film_ior).powi(2) * (1.0 - cos_theta_1 * cos_theta_1);
    let cos_theta_2_squared = 1.0 - sin_theta_2_squared;
    if cos_theta_2_squared < 0.0 {
        return Vec3::ONE;
    }
    let cos_theta_2 = cos_theta_2_squared.sqrt();

    let r0 = ior_f0(film_ior / outside_ior);
    let r12 = fresnel_schlick(cos_theta_1, Vec3::splat(r0)).x;
    let t121 = 1.0 - r12;
    let phi12 = if film_ior < outside_ior {
        std::f32::consts::PI
    } else {
        0.0
    };
    let phi21 = std::f32::consts::PI - phi12;

    let base_ior = fresnel0_to_ior(base_f0);
    let r1 = ior_to_fresnel0(base_ior, film_ior);
    let r23 = fresnel_schlick(cos_theta_2, r1);
    let phi23 = Vec3::new(
        if base_ior.x < film_ior {
            std::f32::consts::PI
        } else {
            0.0
        },
        if base_ior.y < film_ior {
            std::f32::consts::PI
        } else {
            0.0
        },
        if base_ior.z < film_ior {
            std::f32::consts::PI
        } else {
            0.0
        },
    );

    let optical_path_difference = 2.0 * film_ior * thickness_nm * cos_theta_2;
    let phase_shift = Vec3::splat(phi21) + phi23;
    let r123 = (r23 * r12).clamp(Vec3::splat(1.0e-5), Vec3::splat(0.9999));
    let mut cm = (r23 * (t121 * t121) / (Vec3::ONE - r123)) - Vec3::splat(t121);
    let mut result = Vec3::splat(r12) + r23 * (t121 * t121) / (Vec3::ONE - r123);
    let r123_amplitude = Vec3::new(r123.x.sqrt(), r123.y.sqrt(), r123.z.sqrt());
    for order in 1..=2 {
        cm *= r123_amplitude;
        result +=
            cm * eval_iridescence_sensitivity(
                order as f32 * optical_path_difference,
                order as f32 * phase_shift,
            ) * 2.0;
    }
    result.clamp(Vec3::ZERO, Vec3::ONE)
}

/// GGX (Trowbridge-Reitz) normal distribution.
#[inline]
pub fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let nh2 = n_dot_h * n_dot_h;
    let denom = nh2 * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * denom * denom)
}

/// Height-correlated Smith visibility, including `1/(4 NdotV NdotL)`.
#[inline]
pub fn v_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let ggx_l = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (ggx_v + ggx_l + 1e-6)
}

#[inline]
pub fn d_ggx_anisotropic(n_dot_h: f32, t_dot_h: f32, b_dot_h: f32, at: f32, ab: f32) -> f32 {
    let at = at.max(0.001);
    let ab = ab.max(0.001);
    let a2 = at * ab;
    let f = Vec3::new(ab * t_dot_h, at * b_dot_h, a2 * n_dot_h);
    let w2 = a2 / f.length_squared().max(1e-12);
    a2 * w2 * w2 / std::f32::consts::PI
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn v_smith_anisotropic(
    n_dot_l: f32,
    n_dot_v: f32,
    t_dot_v: f32,
    b_dot_v: f32,
    t_dot_l: f32,
    b_dot_l: f32,
    at: f32,
    ab: f32,
) -> f32 {
    let ggx_v = n_dot_l * Vec3::new(at * t_dot_v, ab * b_dot_v, n_dot_v).length();
    let ggx_l = n_dot_v * Vec3::new(at * t_dot_l, ab * b_dot_l, n_dot_l).length();
    (0.5 / (ggx_v + ggx_l + 1e-6)).clamp(0.0, 1.0)
}

#[inline]
pub fn sheen_directional_albedo(n_dot: f32, perceptual_roughness: f32) -> f32 {
    crate::sheen_lut::sample_r16f_lut(
        SHEEN_ALBEDO_LUT,
        SHEEN_ALBEDO_LUT_SIZE,
        n_dot,
        perceptual_roughness,
    )
}

fn default_tangent(normal: Vec3) -> Vec3 {
    let candidate = Vec3::X - normal * normal.x;
    if candidate.length_squared() > 1e-8 {
        candidate.normalize()
    } else {
        Vec3::Y
    }
}

/// Energy-normalized Burley diffuse, including its `1/pi` normalization.
///
/// The original Disney form can return more than one unit of directional
/// albedo for a rough surface at grazing view. The Frostbite normalization
/// fades to `1/1.51` at roughness one and keeps the lobe reciprocal.
#[inline]
pub fn burley_diffuse(n_dot_l: f32, n_dot_v: f32, l_dot_h: f32, perceptual_roughness: f32) -> f32 {
    let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * perceptual_roughness;
    let light = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_l).powi(5);
    let view = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_v).powi(5);
    let energy_factor = 1.0 + (1.0 / 1.51 - 1.0) * perceptual_roughness;
    light * view * energy_factor / std::f32::consts::PI
}

/// Evaluate the version-1 base layer for one view/light pair.
///
/// Inputs are expected to be normalized. Invalid/back-facing pairs return
/// zero instead of allowing a non-finite half vector into later lobes.
#[inline]
pub fn evaluate_base_brdf(
    material: BaseMaterial,
    normal: Vec3,
    view: Vec3,
    light: Vec3,
) -> BaseBrdfEvaluation {
    let n_dot_v = normal.dot(view).max(0.0);
    let n_dot_l = normal.dot(light).max(0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return BaseBrdfEvaluation::default();
    }
    let half_raw = view + light;
    if half_raw.length_squared() <= 1e-12 {
        return BaseBrdfEvaluation::default();
    }
    let half = half_raw.normalize();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);

    let f0 = material.f0();
    let alpha = material.perceptual_roughness * material.perceptual_roughness;
    let fresnel = fresnel_schlick(v_dot_h, f0);
    let distribution = d_ggx(n_dot_h, alpha);
    let visibility = v_smith(n_dot_v, n_dot_l, alpha);
    let specular = fresnel * distribution * visibility;

    let diffuse_factor = burley_diffuse(n_dot_l, n_dot_v, v_dot_h, material.perceptual_roughness);
    // Diffuse crosses the dielectric boundary twice. Gating it by both the
    // view-side and light-side interface transmission is reciprocal and
    // prevents grazing specular from stacking on top of full diffuse energy.
    let view_transmission = Vec3::ONE - fresnel_schlick(n_dot_v, f0);
    let light_transmission = Vec3::ONE - fresnel_schlick(n_dot_l, f0);
    let diffuse_albedo =
        material.base_color * (1.0 - material.metallic) * view_transmission * light_transmission;
    let diffuse = diffuse_albedo * diffuse_factor;
    let brdf_cos = (specular + diffuse) * n_dot_l;

    let specular_weight = f0.element_sum() / 3.0;
    let diffuse_weight = (1.0 - specular_weight) * (1.0 - material.metallic);
    let specular_probability = specular_weight / (specular_weight + diffuse_weight + 1e-6);
    let diffuse_probability = 1.0 - specular_probability;
    let specular_pdf = distribution * n_dot_h / (4.0 * v_dot_h + 1e-6);
    let diffuse_pdf = n_dot_l / std::f32::consts::PI;
    let pdf = (specular_probability * specular_pdf + diffuse_probability * diffuse_pdf).max(0.0);

    BaseBrdfEvaluation {
        diffuse,
        specular,
        brdf_cos,
        pdf,
    }
}

/// Evaluate the current layered contract with a deterministic tangent frame.
///
/// Clearcoat uses a fixed IOR of 1.5. For direct lighting its Schlick term is
/// evaluated at the symmetric view/half angle, so attenuating the base by
/// `1 - clearcoatFactor * Fc` is reciprocal. This follows the common
/// Filament/glTF implementation shape while avoiding the view-only layering
/// shortcut that changes when view and light are exchanged.
#[inline]
pub fn evaluate_layered_brdf(
    material: LayeredMaterial,
    normal: Vec3,
    view: Vec3,
    light: Vec3,
) -> LayeredBrdfEvaluation {
    evaluate_layered_brdf_with_tangent(material, normal, default_tangent(normal), view, light)
}

/// Evaluate clearcoat, sheen, anisotropic GGX, and specular/IOR for one
/// view/light pair. `tangent` is re-orthogonalized against `normal`; the
/// material rotation is counter-clockwise in the resulting tangent plane.
#[inline]
pub fn evaluate_layered_brdf_with_tangent(
    material: LayeredMaterial,
    normal: Vec3,
    tangent: Vec3,
    view: Vec3,
    light: Vec3,
) -> LayeredBrdfEvaluation {
    let n_dot_v = normal.dot(view).max(0.0);
    let n_dot_l = normal.dot(light).max(0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return LayeredBrdfEvaluation::default();
    }
    let half_raw = view + light;
    if half_raw.length_squared() <= 1e-12 {
        return LayeredBrdfEvaluation::default();
    }
    let half = half_raw.normalize();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);

    let tangent = (tangent - normal * normal.dot(tangent)).normalize_or_zero();
    let tangent = if tangent.length_squared() > 0.0 {
        tangent
    } else {
        default_tangent(normal)
    };
    let bitangent = normal.cross(tangent).normalize();
    let (sin_rotation, cos_rotation) = material.anisotropy_rotation.sin_cos();
    let anisotropic_tangent = tangent * cos_rotation + bitangent * sin_rotation;
    let anisotropic_bitangent = normal.cross(anisotropic_tangent).normalize();
    let [at, ab] = material.anisotropic_alpha();
    let base_distribution = if material.anisotropy_strength > 0.0 {
        d_ggx_anisotropic(
            n_dot_h,
            anisotropic_tangent.dot(half),
            anisotropic_bitangent.dot(half),
            at,
            ab,
        )
    } else {
        d_ggx(n_dot_h, ab)
    };
    let base_visibility = if material.anisotropy_strength > 0.0 {
        v_smith_anisotropic(
            n_dot_l,
            n_dot_v,
            anisotropic_tangent.dot(view),
            anisotropic_bitangent.dot(view),
            anisotropic_tangent.dot(light),
            anisotropic_bitangent.dot(light),
            at,
            ab,
        )
    } else {
        v_smith(n_dot_v, n_dot_l, ab)
    };
    let (uncoated_diffuse, uncoated_base_specular) = if material.has_specular_ior_lobe()
        || material.anisotropy_strength > 0.0
        || material.has_iridescence()
    {
        let base_fresnel = material.base_fresnel(v_dot_h);
        let base_specular = base_fresnel * base_distribution * base_visibility;
        let diffuse_factor = burley_diffuse(
            n_dot_l,
            n_dot_v,
            v_dot_h,
            material.base.perceptual_roughness,
        );
        // Khronos specifies max(R,G,B) for colored dielectric Fresnel so the
        // diffuse complement cannot create an inverse color. Applying the
        // complement at both interfaces keeps the diffuse term reciprocal.
        let diffuse_albedo = material.base.base_color
            * (1.0 - material.base.metallic)
            * material.dielectric_transmission(n_dot_v)
            * material.dielectric_transmission(n_dot_l);
        (diffuse_albedo * diffuse_factor, base_specular)
    } else {
        let base = evaluate_base_brdf(material.base, normal, view, light);
        (base.diffuse, base.specular)
    };

    let (sheen_scale, uncoated_sheen) = if material.has_sheen() {
        let sheen_roughness = material.sheen_roughness();
        let view_albedo = sheen_directional_albedo(n_dot_v, sheen_roughness);
        let light_albedo = sheen_directional_albedo(n_dot_l, sheen_roughness);
        let scale = (1.0 - material.sheen_color.max_element() * view_albedo.max(light_albedo))
            .clamp(0.0, 1.0);
        let sheen_distribution = crate::sheen_lut::distribution_charlie(n_dot_h, sheen_roughness);
        let sheen_visibility =
            crate::sheen_lut::visibility_sheen(n_dot_l, n_dot_v, sheen_roughness);
        (
            scale,
            material.sheen_color * sheen_distribution * sheen_visibility,
        )
    } else {
        (1.0, Vec3::ZERO)
    };

    let clearcoat_fresnel = material.clearcoat_fresnel(v_dot_h);
    // A real top interface is crossed in both directions. The two symmetric
    // transmission factors prevent rough/grazing base energy from stacking
    // under a second white-furnace lobe. This is also the direct-light
    // counterpart of Filament's squared clearcoat attenuation for IBL.
    let base_attenuation =
        material.clearcoat_transmission(n_dot_v) * material.clearcoat_transmission(n_dot_l);
    let clearcoat_alpha = material.clearcoat_roughness() * material.clearcoat_roughness();
    let clearcoat_distribution = d_ggx(n_dot_h, clearcoat_alpha);
    let clearcoat_visibility = v_smith(n_dot_v, n_dot_l, clearcoat_alpha);
    let clearcoat_specular =
        Vec3::splat(clearcoat_fresnel * clearcoat_distribution * clearcoat_visibility);

    let diffuse = uncoated_diffuse * sheen_scale * base_attenuation;
    let base_specular = uncoated_base_specular * sheen_scale * base_attenuation;
    let sheen_specular = uncoated_sheen * base_attenuation;
    let brdf_cos = (diffuse + base_specular + sheen_specular + clearcoat_specular) * n_dot_l;

    let base_specular_weight = material.base_fresnel(n_dot_v).element_sum() / 3.0;
    let diffuse_weight = material.dielectric_transmission(n_dot_v) * (1.0 - material.base.metallic);
    let clearcoat_weight = material.clearcoat_fresnel(n_dot_v);
    let sheen_weight = material.sheen_color.element_sum() / 3.0;
    let weight_sum = base_specular_weight + diffuse_weight + clearcoat_weight + sheen_weight + 1e-6;
    let base_specular_probability = base_specular_weight / weight_sum;
    let diffuse_probability = diffuse_weight / weight_sum;
    let clearcoat_probability = clearcoat_weight / weight_sum;
    let sheen_probability = sheen_weight / weight_sum;
    let base_specular_pdf = base_distribution * n_dot_h / (4.0 * v_dot_h + 1e-6);
    let diffuse_pdf = n_dot_l / std::f32::consts::PI;
    let clearcoat_pdf = clearcoat_distribution * n_dot_h / (4.0 * v_dot_h + 1e-6);
    let sheen_pdf = crate::sheen_lut::distribution_charlie(n_dot_h, material.sheen_roughness())
        * n_dot_h
        / (4.0 * v_dot_h + 1e-6);
    let pdf = (base_specular_probability * base_specular_pdf
        + diffuse_probability * diffuse_pdf
        + clearcoat_probability * clearcoat_pdf
        + sheen_probability * sheen_pdf)
        .max(0.0);

    LayeredBrdfEvaluation {
        diffuse,
        base_specular,
        clearcoat_specular,
        sheen_specular,
        brdf_cos,
        pdf,
    }
}

fn reflect(incoming: Vec3, normal: Vec3) -> Vec3 {
    incoming - normal * (2.0 * incoming.dot(normal))
}

fn smith_g1(n_dot_x: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let inner = ((1.0 - a2) * n_dot_x * n_dot_x + a2).sqrt();
    2.0 * n_dot_x / (n_dot_x + inner + 1e-6)
}

fn smith_g1_anisotropic(direction: Vec3, at: f32, ab: f32) -> f32 {
    let n_dot = direction.z.max(0.0);
    if n_dot <= 0.0 {
        return 0.0;
    }
    let projected = Vec3::new(at * direction.x, ab * direction.y, n_dot).length();
    2.0 * n_dot / (n_dot + projected + 1e-6)
}

/// Heitz visible-normal GGX sampling in the local z-up frame.
fn sample_ggx_vndf(view: Vec3, alpha: f32, sample: [f32; 2]) -> Vec3 {
    sample_ggx_vndf_anisotropic(view, alpha, alpha, sample)
}

/// Heitz visible-normal GGX sampling with independent tangent/bitangent
/// alpha roughness.
fn sample_ggx_vndf_anisotropic(view: Vec3, at: f32, ab: f32, sample: [f32; 2]) -> Vec3 {
    let stretched_view = Vec3::new(at * view.x, ab * view.y, view.z).normalize();
    let lensq = stretched_view.x * stretched_view.x + stretched_view.y * stretched_view.y;
    let tangent = if lensq > 0.0 {
        Vec3::new(-stretched_view.y, stretched_view.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let bitangent = stretched_view.cross(tangent);
    let radius = sample[0].sqrt();
    let phi = 2.0 * std::f32::consts::PI * sample[1];
    let tangent_x = radius * phi.cos();
    let mut tangent_y = radius * phi.sin();
    let blend = 0.5 * (1.0 + stretched_view.z);
    tangent_y = (1.0 - blend) * (1.0 - tangent_x * tangent_x).max(0.0).sqrt() + blend * tangent_y;
    let stretched_normal = tangent_x * tangent
        + tangent_y * bitangent
        + (1.0 - tangent_x * tangent_x - tangent_y * tangent_y)
            .max(0.0)
            .sqrt()
            * stretched_view;
    Vec3::new(
        at * stretched_normal.x,
        ab * stretched_normal.y,
        stretched_normal.z.max(0.0),
    )
    .normalize()
}

fn sample_charlie_half(perceptual_roughness: f32, sample: [f32; 2]) -> Vec3 {
    let alpha = perceptual_roughness.max(1e-3).powi(2);
    let sin_theta = sample[0].powf(alpha / (2.0 * alpha + 1.0));
    let cos_theta = (1.0 - sin_theta * sin_theta).max(0.0).sqrt();
    let phi = std::f32::consts::TAU * sample[1];
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

fn sample_cosine_hemisphere(sample: [f32; 2]) -> Vec3 {
    let radius = sample[0].sqrt();
    let phi = 2.0 * std::f32::consts::PI * sample[1];
    Vec3::new(
        radius * phi.cos(),
        radius * phi.sin(),
        (1.0 - sample[0]).max(0.0).sqrt(),
    )
}

/// Deterministic importance-sampled integration under a unit white furnace.
///
/// Specular uses the GGX visible-normal distribution, so mirror-like lobes do
/// not alias into false energy gain. Diffuse uses cosine sampling. The two
/// lobe integrals are accumulated separately and then summed; this is a
/// qualification oracle, never a runtime shading path.
pub fn integrate_white_furnace(
    material: BaseMaterial,
    n_dot_v: f32,
    sample_rows: u32,
    sample_columns: u32,
) -> Vec3 {
    assert!(sample_rows > 0 && sample_columns > 0);
    let n_dot_v = n_dot_v.clamp(1e-4, 1.0);
    let view = Vec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let normal = Vec3::Z;
    let alpha = material.perceptual_roughness * material.perceptual_roughness;
    let sample_count = (sample_rows * sample_columns) as f32;
    let mut diffuse_sum = Vec3::ZERO;
    let mut specular_sum = Vec3::ZERO;
    for row in 0..sample_rows {
        for column in 0..sample_columns {
            let sample = [
                (row as f32 + 0.5) / sample_rows as f32,
                (column as f32 + 0.5) / sample_columns as f32,
            ];

            let diffuse_light = sample_cosine_hemisphere(sample);
            let diffuse = evaluate_base_brdf(material, normal, view, diffuse_light).diffuse;
            diffuse_sum += diffuse * std::f32::consts::PI;

            let half = sample_ggx_vndf(view, alpha, sample);
            let specular_light = reflect(-view, half);
            let n_dot_l = specular_light.z;
            if n_dot_l > 0.0 {
                // For a GGX VNDF sample, `BRDF * cos / PDF` cancels D
                // analytically to `F * G2/G1`. Using that stable form is
                // essential for the near-delta roughness floor.
                let g1_view = smith_g1(n_dot_v, alpha);
                if g1_view > 0.0 {
                    let g2 = v_smith(n_dot_v, n_dot_l, alpha) * (4.0 * n_dot_v * n_dot_l);
                    let fresnel = fresnel_schlick(view.dot(half).max(0.0), material.f0());
                    specular_sum += fresnel * g2 / g1_view;
                }
            }
        }
    }
    (diffuse_sum + specular_sum) / sample_count
}

/// Deterministic unit-white-furnace integration for the version-2 model.
///
/// Each GGX lobe uses its own visible-normal sampler. The stable
/// `F * G2/G1` estimator avoids losing or over-counting the near-delta
/// clearcoat peak; diffuse remains cosine sampled.
pub fn integrate_layered_white_furnace(
    material: LayeredMaterial,
    n_dot_v: f32,
    sample_rows: u32,
    sample_columns: u32,
) -> Vec3 {
    assert!(sample_rows > 0 && sample_columns > 0);
    let n_dot_v = n_dot_v.clamp(1e-4, 1.0);
    let view = Vec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let normal = Vec3::Z;
    let [at, ab] = material.anisotropic_alpha();
    let base_alpha = material.base.perceptual_roughness * material.base.perceptual_roughness;
    let (sin_rotation, cos_rotation) = material.anisotropy_rotation.sin_cos();
    let tangent = Vec3::new(cos_rotation, sin_rotation, 0.0);
    let bitangent = Vec3::new(-sin_rotation, cos_rotation, 0.0);
    let to_local = |direction: Vec3| {
        Vec3::new(
            tangent.dot(direction),
            bitangent.dot(direction),
            direction.z,
        )
    };
    let to_world =
        |direction: Vec3| tangent * direction.x + bitangent * direction.y + normal * direction.z;
    let local_view = to_local(view);
    let clearcoat_alpha = material.clearcoat_roughness() * material.clearcoat_roughness();
    let sample_count = (sample_rows * sample_columns) as f32;
    let mut diffuse_sum = Vec3::ZERO;
    let mut base_specular_sum = Vec3::ZERO;
    let mut sheen_sum = Vec3::ZERO;
    let mut clearcoat_sum = Vec3::ZERO;
    for row in 0..sample_rows {
        for column in 0..sample_columns {
            let sample = [
                (row as f32 + 0.5) / sample_rows as f32,
                (column as f32 + 0.5) / sample_columns as f32,
            ];

            let diffuse_light = sample_cosine_hemisphere(sample);
            let diffuse = evaluate_layered_brdf(material, normal, view, diffuse_light).diffuse;
            diffuse_sum += diffuse * std::f32::consts::PI;

            let base_half = if material.anisotropy_strength > 0.0 {
                to_world(sample_ggx_vndf_anisotropic(local_view, at, ab, sample))
            } else {
                sample_ggx_vndf(view, base_alpha, sample)
            };
            let base_light = reflect(-view, base_half);
            let base_n_dot_l = base_light.z;
            if base_n_dot_l > 0.0 {
                let g1_view = if material.anisotropy_strength > 0.0 {
                    smith_g1_anisotropic(local_view, at, ab)
                } else {
                    smith_g1(n_dot_v, base_alpha)
                };
                if g1_view > 0.0 {
                    let local_light = to_local(base_light);
                    let g2 = if material.anisotropy_strength > 0.0 {
                        v_smith_anisotropic(
                            base_n_dot_l,
                            n_dot_v,
                            local_view.x,
                            local_view.y,
                            local_light.x,
                            local_light.y,
                            at,
                            ab,
                        )
                    } else {
                        v_smith(n_dot_v, base_n_dot_l, base_alpha)
                    } * (4.0 * n_dot_v * base_n_dot_l);
                    let v_dot_h = view.dot(base_half).max(0.0);
                    let base_attenuation = material.clearcoat_transmission(n_dot_v)
                        * material.clearcoat_transmission(base_n_dot_l);
                    let sheen_scale = if material.has_sheen() {
                        let view_albedo =
                            sheen_directional_albedo(n_dot_v, material.sheen_roughness());
                        let light_albedo =
                            sheen_directional_albedo(base_n_dot_l, material.sheen_roughness());
                        (1.0 - material.sheen_color.max_element() * view_albedo.max(light_albedo))
                            .clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    base_specular_sum += material.base_fresnel(v_dot_h) * g2 / g1_view
                        * base_attenuation
                        * sheen_scale;
                }
            }

            if material.has_sheen() {
                let sheen_half = sample_charlie_half(material.sheen_roughness(), sample);
                let sheen_light = reflect(-view, sheen_half);
                let sheen_n_dot_l = sheen_light.z;
                let v_dot_h = view.dot(sheen_half).max(0.0);
                if sheen_n_dot_l > 0.0 && v_dot_h > 0.0 && sheen_half.z > 0.0 {
                    let visibility = crate::sheen_lut::visibility_sheen(
                        sheen_n_dot_l,
                        n_dot_v,
                        material.sheen_roughness(),
                    );
                    let estimator = visibility * sheen_n_dot_l * 4.0 * v_dot_h / sheen_half.z;
                    let coat_attenuation = material.clearcoat_transmission(n_dot_v)
                        * material.clearcoat_transmission(sheen_n_dot_l);
                    sheen_sum += material.sheen_color * estimator * coat_attenuation;
                }
            }

            if material.clearcoat_factor > 0.0 {
                let clearcoat_half = sample_ggx_vndf(view, clearcoat_alpha, sample);
                let clearcoat_light = reflect(-view, clearcoat_half);
                let clearcoat_n_dot_l = clearcoat_light.z;
                if clearcoat_n_dot_l > 0.0 {
                    let g1_view = smith_g1(n_dot_v, clearcoat_alpha);
                    if g1_view > 0.0 {
                        let g2 = v_smith(n_dot_v, clearcoat_n_dot_l, clearcoat_alpha)
                            * (4.0 * n_dot_v * clearcoat_n_dot_l);
                        let fresnel = material.clearcoat_fresnel(view.dot(clearcoat_half).max(0.0));
                        clearcoat_sum += Vec3::splat(fresnel * g2 / g1_view);
                    }
                }
            }
        }
    }
    (diffuse_sum + base_specular_sum + sheen_sum + clearcoat_sum) / sample_count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direction(n_dot: f32, azimuth: f32) -> Vec3 {
        let sin_theta = (1.0 - n_dot * n_dot).sqrt();
        Vec3::new(sin_theta * azimuth.cos(), sin_theta * azimuth.sin(), n_dot)
    }

    #[test]
    fn base_brdf_is_reciprocal_and_finite_across_parameter_matrix() {
        let normal = Vec3::Z;
        for base_color in [Vec3::splat(0.18), Vec3::new(0.8, 0.08, 0.03)] {
            for metallic in [0.0, 0.5, 1.0] {
                for roughness in [0.04, 0.25, 0.5, 1.0] {
                    let material = BaseMaterial {
                        base_color,
                        metallic,
                        perceptual_roughness: roughness,
                    };
                    for n_dot_v in [0.1, 0.5, 1.0] {
                        let view = direction(n_dot_v, 0.0);
                        let light = direction(0.35, 1.1);
                        let forward = evaluate_base_brdf(material, normal, view, light);
                        let reverse = evaluate_base_brdf(material, normal, light, view);
                        assert!(forward.brdf_cos.is_finite() && forward.pdf.is_finite());
                        assert!(forward.diffuse.min_element() >= 0.0);
                        assert!(forward.specular.min_element() >= 0.0);
                        let forward_brdf = forward.diffuse + forward.specular;
                        let reverse_brdf = reverse.diffuse + reverse.specular;
                        assert!(
                            forward_brdf.abs_diff_eq(reverse_brdf, 2e-5),
                            "{material:?}: {forward_brdf:?} != {reverse_brdf:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn white_furnace_has_no_unbounded_energy_gain() {
        let mut maximum = Vec3::ZERO;
        let mut maximum_case = None;
        for base_color in [Vec3::ONE, Vec3::new(1.0, 0.1, 0.02)] {
            for metallic in [0.0, 0.5, 1.0] {
                for roughness in [0.04, 0.25, 0.5, 1.0] {
                    let material = BaseMaterial {
                        base_color,
                        metallic,
                        perceptual_roughness: roughness,
                    };
                    for n_dot_v in [0.1, 0.25, 0.5, 1.0] {
                        let reflected = integrate_white_furnace(material, n_dot_v, 96, 192);
                        assert!(reflected.is_finite());
                        assert!(reflected.min_element() >= 0.0);
                        if reflected.max_element() > maximum.max_element() {
                            maximum_case = Some((material, n_dot_v));
                        }
                        maximum = maximum.max(reflected);
                    }
                }
            }
        }
        assert!(
            maximum.max_element() <= 1.02,
            "white-furnace gain: {maximum:?} at {maximum_case:?}"
        );
    }

    #[test]
    fn invalid_parameters_have_one_documented_sanitization() {
        let material = BaseMaterial {
            base_color: Vec3::new(-1.0, 0.5, 2.0),
            metallic: 4.0,
            perceptual_roughness: 0.0,
        }
        .validated();
        assert_eq!(material.base_color, Vec3::new(0.0, 0.5, 1.0));
        assert_eq!(material.metallic, 1.0);
        assert_eq!(material.perceptual_roughness, MIN_PERCEPTUAL_ROUGHNESS);
    }

    #[test]
    fn version_two_defaults_are_exactly_the_version_one_base_model() {
        let normal = Vec3::Z;
        for base_color in [Vec3::splat(0.18), Vec3::new(0.8, 0.08, 0.03)] {
            for metallic in [0.0, 0.5, 1.0] {
                for roughness in [0.04, 0.25, 0.5, 1.0] {
                    let base = BaseMaterial {
                        base_color,
                        metallic,
                        perceptual_roughness: roughness,
                    };
                    let layered = LayeredMaterial::from_base(base);
                    for n_dot_v in [0.1, 0.5, 1.0] {
                        let view = direction(n_dot_v, 0.0);
                        let light = direction(0.35, 1.1);
                        let old = evaluate_base_brdf(base, normal, view, light);
                        let new = evaluate_layered_brdf(layered, normal, view, light);
                        assert!(old.diffuse.abs_diff_eq(new.diffuse, 2e-6));
                        assert!(old.specular.abs_diff_eq(new.base_specular, 2e-6));
                        assert_eq!(new.clearcoat_specular, Vec3::ZERO);
                        assert!(old.brdf_cos.abs_diff_eq(new.brdf_cos, 2e-6));
                    }
                }
            }
        }
    }

    #[test]
    fn layered_brdf_is_reciprocal_finite_and_non_negative() {
        let normal = Vec3::Z;
        for ior in [0.0, 1.0, 1.33, 1.5, 2.42] {
            for specular_factor in [0.0, 0.35, 1.0] {
                for clearcoat_factor in [0.0, 0.5, 1.0] {
                    let material = LayeredMaterial {
                        base: BaseMaterial {
                            base_color: Vec3::new(0.8, 0.08, 0.03),
                            metallic: 0.25,
                            perceptual_roughness: 0.4,
                        },
                        ior,
                        specular_factor,
                        specular_color: Vec3::new(1.4, 0.5, 0.2),
                        clearcoat_factor,
                        clearcoat_perceptual_roughness: 0.18,
                        sheen_color: Vec3::ZERO,
                        sheen_perceptual_roughness: 0.0,
                        anisotropy_strength: 0.0,
                        anisotropy_rotation: 0.0,
                        iridescence_factor: 0.0,
                        iridescence_ior: DEFAULT_IRIDESCENCE_IOR,
                        iridescence_thickness_nm: DEFAULT_IRIDESCENCE_THICKNESS_NM,
                    };
                    let view = direction(0.21, 0.3);
                    let light = direction(0.63, 1.4);
                    let forward = evaluate_layered_brdf(material, normal, view, light);
                    let reverse = evaluate_layered_brdf(material, normal, light, view);
                    let forward_brdf = forward.diffuse
                        + forward.base_specular
                        + forward.sheen_specular
                        + forward.clearcoat_specular;
                    let reverse_brdf = reverse.diffuse
                        + reverse.base_specular
                        + reverse.sheen_specular
                        + reverse.clearcoat_specular;
                    assert!(forward_brdf.is_finite() && forward.pdf.is_finite());
                    assert!(forward_brdf.min_element() >= 0.0);
                    assert!(
                        forward_brdf.abs_diff_eq(reverse_brdf, 3e-5),
                        "{material:?}: {forward_brdf:?} != {reverse_brdf:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn specular_and_ior_do_not_modify_a_pure_conductor() {
        let normal = Vec3::Z;
        let view = direction(0.35, 0.0);
        let light = direction(0.7, 1.0);
        let base = BaseMaterial {
            base_color: Vec3::new(0.9, 0.45, 0.1),
            metallic: 1.0,
            perceptual_roughness: 0.3,
        };
        let first = evaluate_layered_brdf(
            LayeredMaterial {
                ior: 1.0,
                specular_factor: 0.0,
                specular_color: Vec3::ZERO,
                ..LayeredMaterial::from_base(base)
            },
            normal,
            view,
            light,
        );
        let second = evaluate_layered_brdf(
            LayeredMaterial {
                ior: 2.42,
                specular_factor: 1.0,
                specular_color: Vec3::new(4.0, 0.2, 2.0),
                ..LayeredMaterial::from_base(base)
            },
            normal,
            view,
            light,
        );
        assert!(first.brdf_cos.abs_diff_eq(second.brdf_cos, 2e-6));
    }

    #[test]
    fn layered_white_furnace_has_no_unexplained_energy_gain() {
        let mut maximum = Vec3::ZERO;
        let mut maximum_case = None;
        for metallic in [0.0, 0.5, 1.0] {
            for roughness in [0.04, 0.25, 0.75, 1.0] {
                for (ior, specular_factor, specular_color) in [
                    (1.0, 1.0, Vec3::ONE),
                    (1.33, 1.0, Vec3::ONE),
                    (1.5, 0.35, Vec3::ONE),
                    (1.5, 1.0, Vec3::new(1.5, 0.4, 0.2)),
                    (2.42, 1.0, Vec3::ONE),
                ] {
                    for (clearcoat_factor, clearcoat_roughness) in
                        [(0.0, 0.0), (0.5, 0.04), (1.0, 0.2), (1.0, 0.75)]
                    {
                        let material = LayeredMaterial {
                            base: BaseMaterial {
                                base_color: Vec3::ONE,
                                metallic,
                                perceptual_roughness: roughness,
                            },
                            ior,
                            specular_factor,
                            specular_color,
                            clearcoat_factor,
                            clearcoat_perceptual_roughness: clearcoat_roughness,
                            sheen_color: Vec3::ZERO,
                            sheen_perceptual_roughness: 0.0,
                            anisotropy_strength: 0.0,
                            anisotropy_rotation: 0.0,
                            iridescence_factor: 0.0,
                            iridescence_ior: DEFAULT_IRIDESCENCE_IOR,
                            iridescence_thickness_nm: DEFAULT_IRIDESCENCE_THICKNESS_NM,
                        };
                        for n_dot_v in [0.1, 0.25, 0.5, 1.0] {
                            let reflected =
                                integrate_layered_white_furnace(material, n_dot_v, 64, 128);
                            assert!(reflected.is_finite());
                            assert!(reflected.min_element() >= 0.0);
                            if reflected.max_element() > maximum.max_element() {
                                maximum_case = Some((material, n_dot_v, reflected));
                            }
                            maximum = maximum.max(reflected);
                        }
                    }
                }
            }
        }
        assert!(
            maximum.max_element() <= 1.02,
            "layered white-furnace gain: {maximum:?} at {maximum_case:?}"
        );
    }

    #[test]
    fn layered_validation_preserves_ior_zero_and_clamps_only_bounded_factors() {
        let material = LayeredMaterial {
            base: BaseMaterial {
                base_color: Vec3::new(-1.0, 0.5, 2.0),
                metallic: 3.0,
                perceptual_roughness: 0.0,
            },
            ior: 0.0,
            specular_factor: 4.0,
            specular_color: Vec3::new(-1.0, 2.0, f32::NAN),
            clearcoat_factor: -1.0,
            clearcoat_perceptual_roughness: 2.0,
            sheen_color: Vec3::new(-1.0, 0.5, 2.0),
            sheen_perceptual_roughness: 2.0,
            anisotropy_strength: -1.0,
            anisotropy_rotation: f32::NAN,
            iridescence_factor: 2.0,
            iridescence_ior: f32::NAN,
            iridescence_thickness_nm: -1.0,
        }
        .validated();
        assert_eq!(material.ior, 0.0);
        assert_eq!(material.specular_factor, 1.0);
        assert_eq!(material.specular_color, Vec3::new(0.0, 2.0, 1.0));
        assert_eq!(material.clearcoat_factor, 0.0);
        assert_eq!(material.clearcoat_perceptual_roughness, 1.0);
        assert_eq!(material.sheen_color, Vec3::new(0.0, 0.5, 1.0));
        assert_eq!(material.sheen_perceptual_roughness, 1.0);
        assert_eq!(material.anisotropy_strength, 0.0);
        assert_eq!(material.anisotropy_rotation, 0.0);
        assert_eq!(material.iridescence_factor, 1.0);
        assert_eq!(material.iridescence_ior, DEFAULT_IRIDESCENCE_IOR);
        assert_eq!(material.iridescence_thickness_nm, 0.0);
        assert_eq!(material.base.base_color, Vec3::new(0.0, 0.5, 1.0));
    }

    #[test]
    fn sheen_lut_tracks_the_full_charlie_directional_albedo_oracle() {
        for (n_dot_v, roughness) in [
            (0.08, 0.08),
            (0.2, 0.35),
            (0.5, 0.5),
            (0.85, 0.75),
            (1.0, 1.0),
        ] {
            let table = sheen_directional_albedo(n_dot_v, roughness);
            let integrated = crate::sheen_lut::directional_albedo(n_dot_v, roughness, 65_536);
            assert!(
                (table - integrated).abs() <= 0.012,
                "Charlie E mismatch at ({n_dot_v}, {roughness}): {table} vs {integrated}"
            );
        }
    }

    #[test]
    fn sheen_and_anisotropic_ggx_are_reciprocal_finite_and_rotation_periodic() {
        let normal = Vec3::Z;
        let tangent = Vec3::X;
        for sheen_roughness in [0.04, 0.35, 1.0] {
            for anisotropy_strength in [0.0, 0.45, 1.0] {
                for rotation in [0.0, 0.7, std::f32::consts::FRAC_PI_2] {
                    let material = LayeredMaterial {
                        sheen_color: Vec3::new(0.8, 0.25, 0.05),
                        sheen_perceptual_roughness: sheen_roughness,
                        anisotropy_strength,
                        anisotropy_rotation: rotation,
                        ..LayeredMaterial::from_base(BaseMaterial {
                            base_color: Vec3::new(0.3, 0.08, 0.02),
                            metallic: 0.65,
                            perceptual_roughness: 0.32,
                        })
                    };
                    let view = direction(0.23, 0.2);
                    let light = direction(0.61, 1.3);
                    let forward =
                        evaluate_layered_brdf_with_tangent(material, normal, tangent, view, light);
                    let reverse =
                        evaluate_layered_brdf_with_tangent(material, normal, tangent, light, view);
                    let sum = |value: LayeredBrdfEvaluation| {
                        value.diffuse
                            + value.base_specular
                            + value.sheen_specular
                            + value.clearcoat_specular
                    };
                    assert!(sum(forward).is_finite() && forward.pdf.is_finite());
                    assert!(sum(forward).min_element() >= 0.0);
                    assert!(
                        sum(forward).abs_diff_eq(sum(reverse), 4e-5),
                        "{material:?}: {:?} != {:?}",
                        sum(forward),
                        sum(reverse)
                    );
                    let periodic = evaluate_layered_brdf_with_tangent(
                        LayeredMaterial {
                            anisotropy_rotation: rotation + std::f32::consts::TAU,
                            ..material
                        },
                        normal,
                        tangent,
                        view,
                        light,
                    );
                    assert!(sum(forward).abs_diff_eq(sum(periodic), 4e-5));
                }
            }
        }
    }

    #[test]
    fn sheen_and_anisotropy_white_furnace_stay_bounded() {
        let mut maximum = Vec3::ZERO;
        let mut maximum_case = None;
        for metallic in [0.0, 0.5, 1.0] {
            for roughness in [0.12, 0.4, 0.8] {
                for sheen_roughness in [0.08, 0.4, 1.0] {
                    for anisotropy_strength in [0.0, 0.65, 1.0] {
                        let material = LayeredMaterial {
                            sheen_color: Vec3::new(1.0, 0.45, 0.12),
                            sheen_perceptual_roughness: sheen_roughness,
                            anisotropy_strength,
                            anisotropy_rotation: 0.73,
                            ..LayeredMaterial::from_base(BaseMaterial {
                                base_color: Vec3::ONE,
                                metallic,
                                perceptual_roughness: roughness,
                            })
                        };
                        for n_dot_v in [0.1, 0.4, 1.0] {
                            let reflected =
                                integrate_layered_white_furnace(material, n_dot_v, 64, 128);
                            assert!(reflected.is_finite() && reflected.min_element() >= 0.0);
                            if reflected.max_element() > maximum.max_element() {
                                maximum_case = Some((material, n_dot_v, reflected));
                            }
                            maximum = maximum.max(reflected);
                        }
                    }
                }
            }
        }
        assert!(
            maximum.max_element() <= 1.03,
            "sheen/anisotropy white-furnace gain: {maximum:?} at {maximum_case:?}"
        );
    }

    #[test]
    fn inactive_iridescence_is_exactly_the_version_three_model() {
        let normal = Vec3::Z;
        let view = direction(0.27, 0.2);
        let light = direction(0.64, 1.4);
        let v3 = LayeredMaterial {
            ior: 1.72,
            specular_factor: 0.7,
            specular_color: Vec3::new(1.2, 0.6, 0.3),
            clearcoat_factor: 0.6,
            clearcoat_perceptual_roughness: 0.22,
            sheen_color: Vec3::new(0.2, 0.5, 0.9),
            sheen_perceptual_roughness: 0.4,
            anisotropy_strength: 0.75,
            anisotropy_rotation: 0.61,
            ..LayeredMaterial::from_base(BaseMaterial {
                base_color: Vec3::new(0.08, 0.3, 0.75),
                metallic: 0.35,
                perceptual_roughness: 0.36,
            })
        };
        let expected = evaluate_layered_brdf(v3, normal, view, light);
        for inactive in [
            LayeredMaterial {
                iridescence_factor: 0.0,
                iridescence_ior: 2.2,
                iridescence_thickness_nm: 800.0,
                ..v3
            },
            LayeredMaterial {
                iridescence_factor: 1.0,
                iridescence_ior: 2.2,
                iridescence_thickness_nm: 0.0,
                ..v3
            },
        ] {
            let actual = evaluate_layered_brdf(inactive, normal, view, light);
            assert_eq!(actual.diffuse, expected.diffuse);
            assert_eq!(actual.base_specular, expected.base_specular);
            assert_eq!(actual.sheen_specular, expected.sheen_specular);
            assert_eq!(actual.clearcoat_specular, expected.clearcoat_specular);
            assert_eq!(actual.brdf_cos, expected.brdf_cos);
            assert_eq!(actual.pdf, expected.pdf);
        }
    }

    #[test]
    fn thin_film_fresnel_is_finite_bounded_and_spectrally_varying() {
        let base_f0 = Vec3::splat(0.04);
        let mut colors = Vec::new();
        for thickness_nm in [100.0, 250.0, 400.0, 800.0] {
            for cos_theta in [0.12, 0.45, 0.9] {
                let value = iridescence_fresnel(1.0, 1.3, cos_theta, thickness_nm, base_f0);
                assert!(value.is_finite());
                assert!(value.min_element() >= 0.0);
                assert!(value.max_element() <= 1.0);
                colors.push(value);
            }
        }
        let spread = colors
            .iter()
            .flat_map(|left| colors.iter().map(move |right| (*left - *right).length()))
            .fold(0.0_f32, f32::max);
        assert!(spread > 0.1, "thin film did not produce spectral variation");
    }

    #[test]
    fn iridescent_layer_is_reciprocal_and_white_furnace_bounded() {
        let normal = Vec3::Z;
        let view = direction(0.19, 0.4);
        let light = direction(0.68, 1.3);
        let sum = |value: LayeredBrdfEvaluation| {
            value.diffuse + value.base_specular + value.sheen_specular + value.clearcoat_specular
        };
        let mut maximum = Vec3::ZERO;
        for metallic in [0.0, 0.5, 1.0] {
            for thickness_nm in [100.0, 400.0, 800.0] {
                let material = LayeredMaterial {
                    iridescence_factor: 1.0,
                    iridescence_ior: 1.3,
                    iridescence_thickness_nm: thickness_nm,
                    ..LayeredMaterial::from_base(BaseMaterial {
                        base_color: Vec3::new(0.92, 0.55, 0.12),
                        metallic,
                        perceptual_roughness: 0.35,
                    })
                };
                let forward = evaluate_layered_brdf(material, normal, view, light);
                let reverse = evaluate_layered_brdf(material, normal, light, view);
                assert!(sum(forward).is_finite() && sum(forward).min_element() >= 0.0);
                assert!(sum(forward).abs_diff_eq(sum(reverse), 4e-5));
                for n_dot_v in [0.1, 0.5, 1.0] {
                    let reflected = integrate_layered_white_furnace(material, n_dot_v, 64, 128);
                    assert!(reflected.is_finite() && reflected.min_element() >= 0.0);
                    maximum = maximum.max(reflected);
                }
            }
        }
        assert!(
            maximum.max_element() <= 1.03,
            "iridescence white-furnace gain: {maximum:?}"
        );
    }
}
