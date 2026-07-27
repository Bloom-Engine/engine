//! Lazy path-tracing sidecar for layered PBR material factors.
//!
//! The shared `InstanceGiDataCpu` buffer is also consumed by SSGI and WSRC,
//! so layered path tracing must not grow it. This module keeps a parallel
//! record only for scenes with a contributing layered material and compiles a
//! group-2 PT specialization only when that scene is actually path traced.

use super::*;

pub(super) const PT_LAYERED_RECORD_VERSION: u32 = 1;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PtLayeredMaterialCpu {
    /// x = ABI version, y = MaterialLayeredPbr lobe mask.
    pub(super) header: [u32; 4],
    pub(super) clearcoat_ior: [f32; 4],
    pub(super) specular: [f32; 4],
    pub(super) sheen: [f32; 4],
    /// x = strength, yz = cos/sin authored rotation.
    pub(super) anisotropy: [f32; 4],
    pub(super) iridescence: [f32; 4],
}

const PT_LAYERED_RECORD_BYTES: u64 = std::mem::size_of::<PtLayeredMaterialCpu>() as u64;

impl Default for PtLayeredMaterialCpu {
    fn default() -> Self {
        Self {
            header: [PT_LAYERED_RECORD_VERSION, 0, 0, 0],
            clearcoat_ior: [0.0, 0.0, 1.0, 1.5],
            specular: [1.0, 1.0, 1.0, 1.0],
            sheen: [0.0; 4],
            anisotropy: [0.0, 1.0, 0.0, 0.0],
            iridescence: [0.0, 1.3, 100.0, 400.0],
        }
    }
}

impl PtLayeredMaterialCpu {
    fn from_material(material: crate::models::MaterialLayeredPbr) -> Self {
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

        let mut mask = 0;
        if material.has_clearcoat() {
            mask |= crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE;
        }
        if material.has_sheen() {
            mask |= crate::models::MaterialLayeredPbr::SHEEN_LOBE;
        }
        if material.has_anisotropy() {
            mask |= crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE;
        }
        if material.has_iridescence() {
            mask |= crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE;
        }
        if material.has_specular_ior() {
            mask |= crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE;
        }
        let rotation = finite_or(material.anisotropy_rotation, 0.0);
        let (rotation_sine, rotation_cosine) = rotation.sin_cos();
        Self {
            header: [PT_LAYERED_RECORD_VERSION, mask, 0, 0],
            clearcoat_ior: [
                unit(material.clearcoat_factor, 0.0),
                unit(material.clearcoat_roughness_factor, 0.0),
                finite_or(material.clearcoat_normal_scale, 1.0),
                non_negative(material.ior, 1.5),
            ],
            specular: [
                non_negative(material.specular_color_factor[0], 1.0),
                non_negative(material.specular_color_factor[1], 1.0),
                non_negative(material.specular_color_factor[2], 1.0),
                unit(material.specular_factor, 1.0),
            ],
            sheen: [
                unit(material.sheen_color_factor[0], 0.0),
                unit(material.sheen_color_factor[1], 0.0),
                unit(material.sheen_color_factor[2], 0.0),
                unit(material.sheen_roughness_factor, 0.0),
            ],
            anisotropy: [
                unit(material.anisotropy_strength, 0.0),
                rotation_cosine,
                rotation_sine,
                0.0,
            ],
            iridescence: [
                unit(material.iridescence_factor, 0.0),
                non_negative(material.iridescence_ior, 1.3).max(1.0),
                non_negative(material.iridescence_thickness_minimum, 100.0),
                non_negative(material.iridescence_thickness_maximum, 400.0),
            ],
        }
    }

    pub(super) fn active(self) -> bool {
        self.header[1] != 0
    }
}

/// Append the record parallel to the next TLAS instance. The vector itself is
/// absent until the first active record, then earlier base instances are
/// backfilled. Base-only scenes therefore allocate no per-instance sidecar.
pub(super) fn append_record(
    records: &mut Option<Vec<PtLayeredMaterialCpu>>,
    instance_index: usize,
    material: crate::models::MaterialLayeredPbr,
) {
    let active = material.is_active();
    if records.is_none() && !active {
        return;
    }
    if records.is_none() {
        *records = Some(vec![PtLayeredMaterialCpu::default(); instance_index]);
    }
    if let Some(records) = records {
        debug_assert_eq!(records.len(), instance_index);
        records.push(if active {
            PtLayeredMaterialCpu::from_material(material)
        } else {
            PtLayeredMaterialCpu::default()
        });
    }
}

const PT_LAYERED_BINDINGS_WGSL: &str = r#"
struct PtLayeredMaterial {
    header: vec4<u32>,
    clearcoat_ior: vec4<f32>,
    specular: vec4<f32>,
    sheen: vec4<f32>,
    anisotropy: vec4<f32>,
    iridescence: vec4<f32>,
};
@group(2) @binding(0)
var<storage, read> pt_layered_materials: array<PtLayeredMaterial>;
"#;

fn texture_variant(enabled: bool) -> &'static str {
    if enabled {
        "const PT_HAS_TEXTURES: bool = true;\n\
         @group(1) @binding(0) var pt_textures: binding_array<texture_2d<f32>>;\n\
         fn pt_tex_sample(idx: u32, uv: vec2<f32>) -> vec3<f32> {\n\
             return textureSampleLevel(pt_textures[idx], card_samp, uv, 0.0).rgb;\n\
         }\n"
    } else {
        "const PT_HAS_TEXTURES: bool = false;\n\
         fn pt_tex_sample(idx: u32, uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(1.0); }\n"
    }
}

impl Renderer {
    pub(super) fn set_pt_layered_records(
        &mut self,
        records: Option<Vec<PtLayeredMaterialCpu>>,
        instance_count: usize,
    ) {
        let records = records.unwrap_or_default();
        debug_assert!(records.is_empty() || records.len() == instance_count);
        if self.pt_layered_records != records {
            self.pt_layered_records = records;
            self.pt_layered_dirty = !self.pt_layered_records.is_empty();
            self.pt_accum_count = 0;
            self.pt_wrote_frame = false;
        }
    }

    /// Materialize the specialized pipeline and sidecar on the first frame
    /// where active layered instances actually reach the path tracer.
    pub(super) fn ensure_pt_layered_resources(&mut self) {
        if self.pt_layered_records.is_empty() {
            return;
        }
        if self.pt_layered_layout.is_none() {
            self.pt_layered_layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("pt_layered_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(PT_LAYERED_RECORD_BYTES),
                        },
                        count: None,
                    }],
                },
            ));
        }
        if self.pt_layered_pipeline.is_none() {
            let query_diagnostics = std::env::var("BLOOM_GOLDEN_DIAGNOSTICS")
                .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
                || std::env::var("BLOOM_PT_DEBUG")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .is_some_and(|view| (6..=19).contains(&view));
            let fault = std::env::var("BLOOM_PT_TEST_FAULT").ok();
            let source = format!(
                "enable wgpu_ray_query;\n{}\n{}\n{}\n{}\n{}",
                ray_query_backend_variant(&self.device),
                pt_fault_constants(fault.as_deref()),
                pt_kernel_variant(query_diagnostics),
                texture_variant(self.pt_texture_arrays_enabled),
                PT_LAYERED_BINDINGS_WGSL,
            );
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("pt_layered_shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let groups = [
                self.pt_layout.as_ref(),
                self.pt_tex_layout.as_ref(),
                self.pt_layered_layout.as_ref(),
            ];
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("pt_layered_pipeline_layout"),
                        bind_group_layouts: &groups,
                        immediate_size: 0,
                    });
            self.pt_layered_pipeline = Some(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("pt_layered_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("cs_main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
        }

        let needed = PT_LAYERED_RECORD_BYTES * self.pt_layered_records.len() as u64;
        let recreate = self
            .pt_layered_instance_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < needed);
        if recreate {
            let capacity = self.pt_layered_records.len().next_power_of_two() as u64;
            self.pt_layered_instance_buffer =
                Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pt_layered_instances"),
                    size: PT_LAYERED_RECORD_BYTES * capacity,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.pt_layered_bg = None;
            self.pt_layered_dirty = true;
        }
        if self.pt_layered_dirty {
            self.queue.write_buffer(
                self.pt_layered_instance_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&self.pt_layered_records),
            );
            self.pt_layered_dirty = false;
        }
        if self.pt_layered_bg.is_none() {
            self.pt_layered_bg = Some(
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pt_layered_bg"),
                    layout: self.pt_layered_layout.as_ref().unwrap(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self
                            .pt_layered_instance_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    }],
                }),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_abi_is_six_vec4s_and_default_is_inactive() {
        assert_eq!(std::mem::size_of::<PtLayeredMaterialCpu>(), 96);
        let record = PtLayeredMaterialCpu::default();
        assert_eq!(record.header, [PT_LAYERED_RECORD_VERSION, 0, 0, 0]);
        assert!(!record.active());
    }

    #[test]
    fn first_active_record_backfills_base_instances_lazily() {
        let mut records = None;
        append_record(&mut records, 0, Default::default());
        append_record(&mut records, 1, Default::default());
        assert!(records.is_none());

        let layered = crate::models::MaterialLayeredPbr::from_authoring_factors(
            crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE,
            1.0,
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
        append_record(&mut records, 2, layered);
        let records = records.unwrap();
        assert_eq!(records.len(), 3);
        assert!(!records[0].active());
        assert!(!records[1].active());
        assert_eq!(
            records[2].header[1],
            crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE
        );
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
    }

    #[test]
    fn specialization_uses_separate_group_without_touching_base_kernel() {
        assert!(PT_LAYERED_BINDINGS_WGSL.contains("@group(2) @binding(0)"));
        assert!(!pt_kernel_variant(false).contains("pt_layered_materials"));
    }
}
