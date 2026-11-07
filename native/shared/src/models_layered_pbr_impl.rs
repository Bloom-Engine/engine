//! MaterialLayeredPbr: lobe flags and authoring. Split from models.rs.

#[allow(unused_imports)]
use super::*;

impl MaterialLayeredPbr {
    pub const CLEARCOAT_LOBE: u32 = 1 << 0;
    pub const SHEEN_LOBE: u32 = 1 << 1;
    pub const ANISOTROPY_LOBE: u32 = 1 << 2;
    pub const IRIDESCENCE_LOBE: u32 = 1 << 3;
    pub const SPECULAR_IOR_LOBE: u32 = 1 << 4;

    /// Build the texture-free layered material used by the public authoring
    /// API. The lobe mask is authoritative: values belonging to an absent
    /// lobe are ignored so an omitted descriptor always restores glTF
    /// defaults and the allocation-free base-material path.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_authoring_factors(
        lobe_mask: u32,
        clearcoat_factor: f32,
        clearcoat_roughness: f32,
        clearcoat_normal_scale: f32,
        specular_factor: f32,
        specular_color: [f32; 3],
        ior: f32,
        sheen_color: [f32; 3],
        sheen_roughness: f32,
        anisotropy_strength: f32,
        anisotropy_rotation: f32,
        iridescence_factor: f32,
        iridescence_ior: f32,
        iridescence_thickness_minimum: f32,
        iridescence_thickness_maximum: f32,
    ) -> Self {
        fn finite_or(value: f32, fallback: f32) -> f32 {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        }
        fn unit(value: f32, fallback: f32) -> f32 {
            finite_or(value, fallback).clamp(0.0, 1.0)
        }
        fn non_negative(value: f32, fallback: f32) -> f32 {
            finite_or(value, fallback).max(0.0)
        }
        fn material_ior(value: f32) -> f32 {
            let value = finite_or(value, 1.5);
            if value == 0.0 {
                0.0
            } else {
                value.max(1.0)
            }
        }

        let mut material = Self::default();
        if lobe_mask & Self::CLEARCOAT_LOBE != 0 {
            material.clearcoat_authored = true;
            material.clearcoat_factor = unit(clearcoat_factor, 0.0);
            material.clearcoat_roughness_factor = unit(clearcoat_roughness, 0.0);
            material.clearcoat_normal_scale = finite_or(clearcoat_normal_scale, 1.0);
        }
        if lobe_mask & Self::SPECULAR_IOR_LOBE != 0 {
            material.specular_authored = true;
            material.specular_factor = unit(specular_factor, 1.0);
            material.specular_color_factor = specular_color.map(|value| non_negative(value, 1.0));
            material.ior_authored = true;
            material.ior = material_ior(ior);
        }
        if lobe_mask & Self::SHEEN_LOBE != 0 {
            material.sheen_authored = true;
            material.sheen_color_factor = sheen_color.map(|value| unit(value, 0.0));
            material.sheen_roughness_factor = unit(sheen_roughness, 0.0);
        }
        if lobe_mask & Self::ANISOTROPY_LOBE != 0 {
            material.anisotropy_authored = true;
            material.anisotropy_strength = unit(anisotropy_strength, 0.0);
            material.anisotropy_rotation = finite_or(anisotropy_rotation, 0.0);
        }
        if lobe_mask & Self::IRIDESCENCE_LOBE != 0 {
            material.iridescence_authored = true;
            material.iridescence_factor = unit(iridescence_factor, 0.0);
            material.iridescence_ior = finite_or(iridescence_ior, 1.3).max(1.0);
            material.iridescence_thickness_minimum =
                non_negative(iridescence_thickness_minimum, 100.0);
            material.iridescence_thickness_maximum =
                non_negative(iridescence_thickness_maximum, 400.0);
        }
        material
    }

    pub(crate) fn is_active(self) -> bool {
        self.has_clearcoat()
            || self.has_specular_ior()
            || self.has_sheen()
            || self.has_anisotropy()
            || self.has_iridescence()
    }

    pub(crate) fn has_clearcoat(self) -> bool {
        self.clearcoat_authored && self.clearcoat_factor.is_finite() && self.clearcoat_factor > 0.0
    }

    pub(crate) fn has_specular_ior(self) -> bool {
        self.ior != 1.5
            || self.specular_factor != 1.0
            || self.specular_color_factor != [1.0; 3]
            || self.specular_texture.is_some()
            || self.specular_color_texture.is_some()
    }

    pub(crate) fn has_sheen(self) -> bool {
        self.sheen_authored
            && self
                .sheen_color_factor
                .iter()
                .any(|value| value.is_finite() && *value > 0.0)
    }

    pub(crate) fn has_anisotropy(self) -> bool {
        self.anisotropy_authored
            && self.anisotropy_strength.is_finite()
            && self.anisotropy_strength > 0.0
    }

    pub(crate) fn has_iridescence(self) -> bool {
        let thickness_can_contribute = if self.iridescence_thickness_texture.is_some() {
            self.iridescence_thickness_minimum > 0.0 || self.iridescence_thickness_maximum > 0.0
        } else {
            self.iridescence_thickness_maximum > 0.0
        };
        self.iridescence_authored
            && self.iridescence_factor.is_finite()
            && self.iridescence_factor > 0.0
            && self.iridescence_thickness_minimum.is_finite()
            && self.iridescence_thickness_maximum.is_finite()
            && thickness_can_contribute
    }

    pub(crate) fn requests_tex_coord(self, tex_coord: u32) -> bool {
        let clearcoat = if self.has_clearcoat() {
            [
                self.clearcoat_texture,
                self.clearcoat_roughness_texture,
                self.clearcoat_normal_texture,
            ]
            .into_iter()
            .flatten()
            .any(|binding| binding.transform.tex_coord == tex_coord)
        } else {
            false
        };
        let specular = if self.has_specular_ior() {
            [self.specular_texture, self.specular_color_texture]
                .into_iter()
                .flatten()
                .any(|binding| binding.transform.tex_coord == tex_coord)
        } else {
            false
        };
        let sheen = self.has_sheen()
            && [self.sheen_color_texture, self.sheen_roughness_texture]
                .into_iter()
                .flatten()
                .any(|binding| binding.transform.tex_coord == tex_coord);
        let anisotropy = self.has_anisotropy()
            && self
                .anisotropy_texture
                .is_some_and(|binding| binding.transform.tex_coord == tex_coord);
        let iridescence = self.has_iridescence()
            && [self.iridescence_texture, self.iridescence_thickness_texture]
                .into_iter()
                .flatten()
                .any(|binding| binding.transform.tex_coord == tex_coord);
        clearcoat || specular || sheen || anisotropy || iridescence
    }

    pub(crate) fn has_resolved_tex_coord(self, tex_coord: u32) -> bool {
        self.requests_tex_coord(tex_coord)
            && [
                self.clearcoat_texture,
                self.clearcoat_roughness_texture,
                self.clearcoat_normal_texture,
                self.specular_texture,
                self.specular_color_texture,
                self.sheen_color_texture,
                self.sheen_roughness_texture,
                self.anisotropy_texture,
                self.iridescence_texture,
                self.iridescence_thickness_texture,
            ]
            .into_iter()
            .flatten()
            .any(|binding| {
                binding.transform.tex_coord == tex_coord
                    && binding.runtime_texture_idx.is_some_and(|index| index != 0)
            })
    }
}
