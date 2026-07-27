//! Runtime qualification and record updates for layered path tracing.

use super::*;

impl Renderer {
    pub(in crate::renderer) fn pt_layered_transport_active(&self) -> bool {
        self.pt_layered
            .records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_qualified_transport)
            || (self.pt_texture_arrays_enabled
                && (self.pt_layered_texture_active()
                    || self.pt_layered_clearcoat_texture_active()
                    || self.pt_layered_sheen_texture_active()
                    || self.pt_layered_iridescence_texture_active()
                    || self.pt_layered_anisotropy_texture_active()))
    }

    pub(in crate::renderer) fn pt_layered_sheen_active(&self) -> bool {
        self.pt_layered
            .records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_sheen)
            || (self.pt_texture_arrays_enabled && self.pt_layered_sheen_texture_active())
    }

    pub(in crate::renderer) fn pt_layered_anisotropy_active(&self) -> bool {
        self.pt_layered
            .records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_anisotropy)
            || (self.pt_texture_arrays_enabled && self.pt_layered_anisotropy_texture_active())
    }

    pub(in crate::renderer) fn pt_layered_iridescence_active(&self) -> bool {
        self.pt_layered
            .records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_iridescence)
            || (self.pt_texture_arrays_enabled && self.pt_layered_iridescence_texture_active())
    }

    pub(in crate::renderer) fn pt_layered_texture_active(&self) -> bool {
        self.pt_layered
            .texture_records
            .iter()
            .copied()
            .any(PtLayeredTextureCpu::has_specular_ior)
    }

    pub(in crate::renderer) fn pt_layered_clearcoat_texture_active(&self) -> bool {
        self.pt_layered
            .clearcoat_texture_records
            .iter()
            .copied()
            .any(PtClearcoatTextureCpu::active)
    }

    pub(in crate::renderer) fn pt_layered_sheen_texture_active(&self) -> bool {
        self.pt_layered
            .sheen_texture_records
            .iter()
            .copied()
            .any(PtSheenTextureCpu::active)
    }

    pub(in crate::renderer) fn pt_layered_iridescence_texture_active(&self) -> bool {
        self.pt_layered
            .iridescence_texture_records
            .iter()
            .copied()
            .any(PtIridescenceTextureCpu::active)
    }

    pub(in crate::renderer) fn pt_layered_anisotropy_texture_active(&self) -> bool {
        self.pt_layered
            .anisotropy_texture_records
            .iter()
            .copied()
            .any(PtAnisotropyTextureCpu::active)
    }

    pub(in crate::renderer) fn pt_layered_uv1_active(&self) -> bool {
        self.pt_layered
            .texture_records
            .iter()
            .copied()
            .any(PtLayeredTextureCpu::has_uv1)
            || self
                .pt_layered
                .clearcoat_texture_records
                .iter()
                .copied()
                .any(PtClearcoatTextureCpu::has_uv1)
            || self
                .pt_layered
                .sheen_texture_records
                .iter()
                .copied()
                .any(PtSheenTextureCpu::has_uv1)
            || self
                .pt_layered
                .iridescence_texture_records
                .iter()
                .copied()
                .any(PtIridescenceTextureCpu::has_uv1)
            || self
                .pt_layered
                .anisotropy_texture_records
                .iter()
                .copied()
                .any(PtAnisotropyTextureCpu::has_uv1)
    }

    pub(in crate::renderer) fn set_pt_layered_records(
        &mut self,
        records: Option<Vec<PtLayeredMaterialCpu>>,
        texture_records: Option<Vec<PtLayeredTextureCpu>>,
        clearcoat_texture_records: Option<Vec<PtClearcoatTextureCpu>>,
        sheen_texture_records: Option<Vec<PtSheenTextureCpu>>,
        iridescence_texture_records: Option<Vec<PtIridescenceTextureCpu>>,
        anisotropy_texture_records: Option<Vec<PtAnisotropyTextureCpu>>,
        instance_count: usize,
    ) {
        let records = records.unwrap_or_default();
        let texture_records = texture_records.unwrap_or_default();
        let clearcoat_texture_records = clearcoat_texture_records.unwrap_or_default();
        let sheen_texture_records = sheen_texture_records.unwrap_or_default();
        let iridescence_texture_records = iridescence_texture_records.unwrap_or_default();
        let anisotropy_texture_records = anisotropy_texture_records.unwrap_or_default();
        debug_assert!(records.is_empty() || records.len() == instance_count);
        debug_assert!(texture_records.is_empty() || texture_records.len() == instance_count);
        debug_assert!(
            clearcoat_texture_records.is_empty()
                || clearcoat_texture_records.len() == instance_count
        );
        debug_assert!(
            sheen_texture_records.is_empty() || sheen_texture_records.len() == instance_count
        );
        debug_assert!(
            iridescence_texture_records.is_empty()
                || iridescence_texture_records.len() == instance_count
        );
        debug_assert!(
            anisotropy_texture_records.is_empty()
                || anisotropy_texture_records.len() == instance_count
        );
        if self.pt_layered.records != records {
            self.pt_layered.records = records;
            self.pt_layered.dirty = !self.pt_layered.records.is_empty();
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
        }
        if self.pt_layered.clearcoat_texture_records != clearcoat_texture_records {
            self.pt_layered.clearcoat_texture_records = clearcoat_texture_records;
            self.pt_layered.clearcoat_texture_dirty =
                !self.pt_layered.clearcoat_texture_records.is_empty();
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
        }
        if self.pt_layered.sheen_texture_records != sheen_texture_records {
            self.pt_layered.sheen_texture_records = sheen_texture_records;
            self.pt_layered.sheen_texture_dirty = !self.pt_layered.sheen_texture_records.is_empty();
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
        }
        if self.pt_layered.iridescence_texture_records != iridescence_texture_records {
            self.pt_layered.iridescence_texture_records = iridescence_texture_records;
            self.pt_layered.iridescence_texture_dirty =
                !self.pt_layered.iridescence_texture_records.is_empty();
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
        }
        if self.pt_layered.anisotropy_texture_records != anisotropy_texture_records {
            self.pt_layered.anisotropy_texture_records = anisotropy_texture_records;
            self.pt_layered.anisotropy_texture_dirty =
                !self.pt_layered.anisotropy_texture_records.is_empty();
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
        }
        if self.pt_layered.texture_records != texture_records {
            self.pt_layered.texture_records = texture_records;
            self.pt_layered.texture_dirty = !self.pt_layered.texture_records.is_empty();
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
        }
    }
}
