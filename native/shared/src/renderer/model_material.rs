//! Shared imported-model material translation for ordinary and virtual caches.

use super::{material_indirection, Renderer};

impl Renderer {
    pub(crate) fn model_gpu_material_record(
        &self,
        mesh: &crate::models::MeshData,
    ) -> material_indirection::GpuMaterialRecord {
        let texture_id = |index: u32| {
            self.global_texture_ids
                .get(index as usize)
                .copied()
                .unwrap_or(material_indirection::TextureId::FALLBACK)
        };
        let mut record = material_indirection::GpuMaterialRecord::default();
        record.metal_rough = [
            mesh.metallic_factor,
            mesh.roughness_factor,
            if mesh.specular_glossiness_factor.is_some() {
                2.0
            } else {
                mesh.metallic_roughness_texture_idx.is_some() as u8 as f32
            },
            mesh.alpha_mode.shader_alpha_value(mesh.alpha_cutoff),
        ];
        record.emissive = [
            mesh.emissive_factor[0],
            mesh.emissive_factor[1],
            mesh.emissive_factor[2],
            if mesh.alpha_coverage_mips { 1.0 } else { 0.0 },
        ];
        record.spec_gloss = mesh.specular_glossiness_factor.unwrap_or([1.0; 4]);
        record.texture_ids_0 = [
            texture_id(mesh.texture_idx.unwrap_or(0)).raw(),
            texture_id(mesh.normal_texture_idx.unwrap_or(0)).raw(),
            texture_id(mesh.metallic_roughness_texture_idx.unwrap_or(0)).raw(),
            texture_id(mesh.emissive_texture_idx.unwrap_or(0)).raw(),
        ];
        record.texture_ids_1[0] = texture_id(mesh.occlusion_texture_idx.unwrap_or(0)).raw();
        record.sampler_ids_0 = [self.global_linear_sampler_id.raw(); 4];
        record.sampler_ids_1[0] = self.global_linear_sampler_id.raw();
        record
    }

    pub(crate) fn allocate_model_gpu_material(
        &mut self,
        mesh: &crate::models::MeshData,
    ) -> material_indirection::MaterialId {
        if (self.imported_refraction_enabled && mesh.transmission.is_active())
            || mesh.layered_pbr.is_active()
        {
            return material_indirection::MaterialId::FALLBACK;
        }
        let record = self.model_gpu_material_record(mesh);
        self.material_system
            .indirection
            .allocate_material(&self.device, record)
    }
}
