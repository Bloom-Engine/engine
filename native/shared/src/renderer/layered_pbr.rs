//! Packed layered-PBR material metadata shared by the legacy/custom uniform
//! record and the global material-indirection storage record.
//!
//! Version 1 is an ABI-only foundation: every existing material has an empty
//! lobe mask and no shader branches on it. Reserved lanes keep both record
//! sizes and bind-group layouts unchanged.

use super::material_system::MaterialFactorsUniforms;

/// Version of the per-material layered-PBR metadata, independent from the
/// broader custom-shader ABI version.
pub(crate) const MATERIAL_RECORD_VERSION: u32 = 1;

/// The global storage record keeps its version in the high eight bits and its
/// lobe mask in the low 24 bits of `header.y`.
pub(crate) const MATERIAL_RECORD_VERSION_SHIFT: u32 = 24;
pub(crate) const MATERIAL_LOBE_MASK_BITS: u32 = MATERIAL_RECORD_VERSION_SHIFT;
pub(crate) const MATERIAL_LOBE_MASK_MASK: u32 = (1 << MATERIAL_LOBE_MASK_BITS) - 1;

/// Reserved lobe assignments. No production material enables these in the ABI
/// foundation package; the names pin future import/runtime interoperability.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct MaterialLobeMask(u32);

// Web does not compile the native quality-telemetry consumer yet. Keep the
// reserved assignments visible to every target without adding wasm-only
// dead-code warnings.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
impl MaterialLobeMask {
    pub(crate) const NONE: Self = Self(0);
    pub(crate) const CLEARCOAT: Self = Self(1 << 0);
    pub(crate) const SHEEN: Self = Self(1 << 1);
    pub(crate) const ANISOTROPY: Self = Self(1 << 2);
    pub(crate) const IRIDESCENCE: Self = Self(1 << 3);
    pub(crate) const SPECULAR_IOR: Self = Self(1 << 4);
    pub(crate) const TRANSMISSION: Self = Self(1 << 5);
    pub(crate) const KNOWN: Self = Self(
        Self::CLEARCOAT.0
            | Self::SHEEN.0
            | Self::ANISOTROPY.0
            | Self::IRIDESCENCE.0
            | Self::SPECULAR_IOR.0
            | Self::TRANSMISSION.0,
    );

    pub(crate) const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits & MATERIAL_LOBE_MASK_MASK)
    }

    pub(crate) const fn bits(self) -> u32 {
        self.0
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

pub(crate) const fn pack_global_material_metadata(mask: MaterialLobeMask) -> u32 {
    (MATERIAL_RECORD_VERSION << MATERIAL_RECORD_VERSION_SHIFT)
        | (mask.bits() & MATERIAL_LOBE_MASK_MASK)
}

pub(crate) const fn global_material_version(metadata: u32) -> u32 {
    metadata >> MATERIAL_RECORD_VERSION_SHIFT
}

pub(crate) const fn global_material_lobe_mask(metadata: u32) -> MaterialLobeMask {
    MaterialLobeMask::from_bits_truncate(metadata)
}

/// Custom `MaterialFactors` keeps the exact u32 version and mask bit patterns
/// in the previously reserved `foliage_params.zw` f32 lanes.
pub(crate) const fn bound_material_version_lane() -> f32 {
    f32::from_bits(MATERIAL_RECORD_VERSION)
}

pub(crate) const fn bound_material_lobe_mask_lane(mask: MaterialLobeMask) -> f32 {
    f32::from_bits(mask.bits())
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) const fn bound_material_version(lane: f32) -> u32 {
    lane.to_bits()
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) const fn bound_material_lobe_mask(lane: f32) -> MaterialLobeMask {
    MaterialLobeMask::from_bits_truncate(lane.to_bits())
}

impl MaterialFactorsUniforms {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn layered_pbr_version(&self) -> u32 {
        bound_material_version(self.foliage_params[2])
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn layered_pbr_lobe_mask(&self) -> MaterialLobeMask {
        bound_material_lobe_mask(self.foliage_params[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_metadata_round_trips_version_and_mask_without_overlap() {
        let metadata = pack_global_material_metadata(MaterialLobeMask::KNOWN);
        assert_eq!(global_material_version(metadata), MATERIAL_RECORD_VERSION);
        assert_eq!(global_material_lobe_mask(metadata), MaterialLobeMask::KNOWN);
        assert_eq!(MaterialLobeMask::KNOWN.bits() & !MATERIAL_LOBE_MASK_MASK, 0);
    }

    #[test]
    fn bound_uniform_lanes_preserve_exact_integer_bits() {
        assert_eq!(
            bound_material_version(bound_material_version_lane()),
            MATERIAL_RECORD_VERSION
        );
        let mask_lane = bound_material_lobe_mask_lane(MaterialLobeMask::KNOWN);
        assert_eq!(bound_material_lobe_mask(mask_lane), MaterialLobeMask::KNOWN);
    }
}
