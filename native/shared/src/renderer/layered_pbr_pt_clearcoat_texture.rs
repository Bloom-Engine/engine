//! Independently lazy clearcoat factor/roughness metadata for path tracing.
//!
//! Clearcoat normal maps remain unqualified until their complete
//! tangent-space hit-shading path lands. Keeping this record separate from
//! specular textures preserves the established specular-only ABI and cost.

use super::*;

const PT_CLEARCOAT_TEXTURE_RECORD_VERSION: u32 = 1;
const PT_CLEARCOAT_FACTOR_UV1: u32 = 1 << 16;
const PT_CLEARCOAT_ROUGHNESS_UV1: u32 = 1 << 17;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::renderer) struct PtClearcoatTextureCpu {
    /// x = ABI version, y = qualified lobe + UV flags,
    /// z/w = clearcoat-factor/roughness runtime texture indices.
    header: [u32; 4],
    /// Column-major 2x2 UV matrices after authored scale + rotation.
    factor_matrix: [f32; 4],
    roughness_matrix: [f32; 4],
    /// xy = factor offset, zw = roughness offset.
    offsets: [f32; 4],
}

pub(super) const PT_CLEARCOAT_TEXTURE_RECORD_BYTES: u64 =
    std::mem::size_of::<PtClearcoatTextureCpu>() as u64;

impl Default for PtClearcoatTextureCpu {
    fn default() -> Self {
        Self {
            header: [PT_CLEARCOAT_TEXTURE_RECORD_VERSION, 0, 0, 0],
            factor_matrix: [1.0, 0.0, 0.0, 1.0],
            roughness_matrix: [1.0, 0.0, 0.0, 1.0],
            offsets: [0.0; 4],
        }
    }
}

impl PtClearcoatTextureCpu {
    pub(super) fn from_material(
        material: crate::models::MaterialLayeredPbr,
        runtime_texture_count: usize,
        has_secondary_uv: bool,
    ) -> Self {
        fn usable(
            binding: crate::models::MaterialTextureBinding,
            runtime_texture_count: usize,
            has_secondary_uv: bool,
        ) -> bool {
            let transform = binding.transform;
            binding.runtime_texture_idx.is_some_and(|index| {
                index != 0
                    && (index as usize) < PT_MAX_TEXTURES
                    && (index as usize) < runtime_texture_count
            }) && (transform.tex_coord == 0 || (transform.tex_coord == 1 && has_secondary_uv))
                && transform.offset.iter().all(|value| value.is_finite())
                && transform.scale.iter().all(|value| value.is_finite())
                && transform.rotation.is_finite()
        }

        fn transform(
            binding: Option<crate::models::MaterialTextureBinding>,
        ) -> ([f32; 4], [f32; 2]) {
            let transform = binding.map(|binding| binding.transform).unwrap_or_default();
            let (sine, cosine) = transform.rotation.sin_cos();
            (
                [
                    cosine * transform.scale[0],
                    sine * transform.scale[0],
                    -sine * transform.scale[1],
                    cosine * transform.scale[1],
                ],
                transform.offset,
            )
        }

        let factor = material.clearcoat_texture;
        let roughness = material.clearcoat_roughness_texture;
        let has_texture = factor.is_some() || roughness.is_some();
        let all_usable = factor
            .is_none_or(|binding| usable(binding, runtime_texture_count, has_secondary_uv))
            && roughness
                .is_none_or(|binding| usable(binding, runtime_texture_count, has_secondary_uv));
        if !material.has_clearcoat()
            || !has_texture
            || !all_usable
            || material.clearcoat_normal_texture.is_some()
        {
            return Self::default();
        }

        let (factor_matrix, factor_offset) = transform(factor);
        let (roughness_matrix, roughness_offset) = transform(roughness);
        Self {
            header: [
                PT_CLEARCOAT_TEXTURE_RECORD_VERSION,
                crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE
                    | if factor.is_some_and(|binding| binding.transform.tex_coord == 1) {
                        PT_CLEARCOAT_FACTOR_UV1
                    } else {
                        0
                    }
                    | if roughness.is_some_and(|binding| binding.transform.tex_coord == 1) {
                        PT_CLEARCOAT_ROUGHNESS_UV1
                    } else {
                        0
                    },
                factor
                    .and_then(|binding| binding.runtime_texture_idx)
                    .unwrap_or(0),
                roughness
                    .and_then(|binding| binding.runtime_texture_idx)
                    .unwrap_or(0),
            ],
            factor_matrix,
            roughness_matrix,
            offsets: [
                factor_offset[0],
                factor_offset[1],
                roughness_offset[0],
                roughness_offset[1],
            ],
        }
    }

    pub(super) fn active(self) -> bool {
        self.header[0] == PT_CLEARCOAT_TEXTURE_RECORD_VERSION
            && self.header[1] & crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE != 0
    }

    pub(super) fn has_uv1(self) -> bool {
        self.active()
            && self.header[1] & (PT_CLEARCOAT_FACTOR_UV1 | PT_CLEARCOAT_ROUGHNESS_UV1) != 0
    }
}

pub(super) const PT_CLEARCOAT_TEXTURE_BINDINGS_WGSL: &str = r#"
const PT_HAS_CLEARCOAT_TEXTURES: bool = true;
const PT_CLEARCOAT_FACTOR_UV1: u32 = 65536u;
const PT_CLEARCOAT_ROUGHNESS_UV1: u32 = 131072u;

struct PtClearcoatTexture {
    header: vec4<u32>,
    factor_matrix: vec4<f32>,
    roughness_matrix: vec4<f32>,
    offsets: vec4<f32>,
};
@group(2) @binding(4)
var<storage, read> pt_clearcoat_textures: array<PtClearcoatTexture>;

fn pt_clearcoat_transform_uv(
    uv: vec2<f32>,
    matrix: vec4<f32>,
    offset: vec2<f32>,
) -> vec2<f32> {
    return vec2<f32>(
        matrix.x * uv.x + matrix.z * uv.y,
        matrix.y * uv.x + matrix.w * uv.y,
    ) + offset;
}

fn pt_layered_apply_clearcoat_textures(
    material_in: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
) -> PtLayeredMaterial {
    var material = material_in;
    let texture_meta = pt_clearcoat_textures[instance_index];
    if (
        texture_meta.header.x != 1u
            || (texture_meta.header.y & PT_LAYERED_CLEARCOAT_LOBE) == 0u
    ) {
        return material;
    }
    var factor = material.clearcoat_ior.x;
    var roughness = material.clearcoat_ior.y;
    if (texture_meta.header.z != 0u) {
        let factor_uv = pt_clearcoat_transform_uv(
            select(uv0, uv1, (texture_meta.header.y & PT_CLEARCOAT_FACTOR_UV1) != 0u),
            texture_meta.factor_matrix,
            texture_meta.offsets.xy,
        );
        factor *= pt_tex_sample_rgba(texture_meta.header.z, factor_uv).r;
    }
    if (texture_meta.header.w != 0u) {
        let roughness_uv = pt_clearcoat_transform_uv(
            select(uv0, uv1, (texture_meta.header.y & PT_CLEARCOAT_ROUGHNESS_UV1) != 0u),
            texture_meta.roughness_matrix,
            texture_meta.offsets.zw,
        );
        roughness *= pt_tex_sample_rgba(texture_meta.header.w, roughness_uv).g;
    }
    material.clearcoat_ior = vec4<f32>(
        clamp(factor, 0.0, 1.0),
        clamp(roughness, 0.0, 1.0),
        material.clearcoat_ior.zw,
    );
    material.header = vec4<u32>(
        material.header.xy,
        material.header.z & ~PT_LAYERED_CLEARCOAT_LOBE,
        material.header.w,
    );
    return material;
}
"#;

pub(super) const PT_CLEARCOAT_TEXTURE_DISABLED_WGSL: &str = r#"
const PT_HAS_CLEARCOAT_TEXTURES: bool = false;

fn pt_layered_apply_clearcoat_textures(
    material: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
) -> PtLayeredMaterial {
    return material;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(index: Option<u32>, tex_coord: u32) -> crate::models::MaterialTextureBinding {
        crate::models::MaterialTextureBinding {
            source_texture_index: 0,
            source_image_index: 0,
            runtime_texture_idx: index,
            transform: crate::models::MaterialTextureTransform {
                tex_coord,
                ..Default::default()
            },
        }
    }

    #[test]
    fn record_is_independently_compact_and_inactive_by_default() {
        assert_eq!(PT_CLEARCOAT_TEXTURE_RECORD_BYTES, 64);
        assert!(!PtClearcoatTextureCpu::default().active());
    }

    #[test]
    fn qualification_requires_resolved_coordinates_and_no_normal_map() {
        let material = crate::models::MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 0.8,
            clearcoat_texture: Some(binding(Some(2), 0)),
            clearcoat_roughness_texture: Some(binding(Some(3), 1)),
            ..Default::default()
        };
        assert!(!PtClearcoatTextureCpu::from_material(material, 4, false).active());
        let record = PtClearcoatTextureCpu::from_material(material, 4, true);
        assert!(record.active() && record.has_uv1());
        assert_eq!(record.header[2..], [2, 3]);

        let unresolved = crate::models::MaterialLayeredPbr {
            clearcoat_texture: Some(binding(None, 0)),
            ..material
        };
        assert!(!PtClearcoatTextureCpu::from_material(unresolved, 4, true).active());

        let normal_mapped = crate::models::MaterialLayeredPbr {
            clearcoat_normal_texture: Some(binding(Some(1), 0)),
            ..material
        };
        assert!(!PtClearcoatTextureCpu::from_material(normal_mapped, 4, true).active());
    }

    #[test]
    fn first_clearcoat_texture_backfills_parallel_records_and_reports_uv1() {
        let mut records = None;
        let mut specular_records = None;
        let mut clearcoat_records = None;
        assert!(!super::super::texture::append_record(
            &mut records,
            &mut specular_records,
            &mut clearcoat_records,
            0,
            Default::default(),
            4,
            true,
        ));
        let material = crate::models::MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 0.8,
            clearcoat_texture: Some(binding(Some(2), 1)),
            ..Default::default()
        };
        assert!(super::super::texture::append_record(
            &mut records,
            &mut specular_records,
            &mut clearcoat_records,
            1,
            material,
            4,
            true,
        ));
        assert!(specular_records.is_none());
        let records = clearcoat_records.unwrap();
        assert_eq!(records.len(), 2);
        assert!(!records[0].active());
        assert!(records[1].active() && records[1].has_uv1());
    }
}
