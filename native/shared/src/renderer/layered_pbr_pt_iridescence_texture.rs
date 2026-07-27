//! Independently lazy iridescence factor/thickness metadata for path tracing.

use super::*;

const PT_IRIDESCENCE_TEXTURE_RECORD_VERSION: u32 = 1;
const PT_IRIDESCENCE_FACTOR_UV1: u32 = 1 << 16;
const PT_IRIDESCENCE_THICKNESS_UV1: u32 = 1 << 17;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::renderer) struct PtIridescenceTextureCpu {
    /// x = ABI version, y = qualified lobe + UV flags,
    /// z/w = factor/thickness runtime texture indices.
    header: [u32; 4],
    /// Column-major 2x2 UV matrices after authored scale + rotation.
    factor_matrix: [f32; 4],
    thickness_matrix: [f32; 4],
    /// xy = factor offset, zw = thickness offset.
    offsets: [f32; 4],
}

pub(super) const PT_IRIDESCENCE_TEXTURE_RECORD_BYTES: u64 =
    std::mem::size_of::<PtIridescenceTextureCpu>() as u64;

impl Default for PtIridescenceTextureCpu {
    fn default() -> Self {
        Self {
            header: [PT_IRIDESCENCE_TEXTURE_RECORD_VERSION, 0, 0, 0],
            factor_matrix: [1.0, 0.0, 0.0, 1.0],
            thickness_matrix: [1.0, 0.0, 0.0, 1.0],
            offsets: [0.0; 4],
        }
    }
}

impl PtIridescenceTextureCpu {
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

        let factor = material.iridescence_texture;
        let thickness = material.iridescence_thickness_texture;
        let has_texture = factor.is_some() || thickness.is_some();
        let all_usable = factor
            .is_none_or(|binding| usable(binding, runtime_texture_count, has_secondary_uv))
            && thickness
                .is_none_or(|binding| usable(binding, runtime_texture_count, has_secondary_uv));
        if !material.has_iridescence() || !has_texture || !all_usable {
            return Self::default();
        }

        let (factor_matrix, factor_offset) = transform(factor);
        let (thickness_matrix, thickness_offset) = transform(thickness);
        Self {
            header: [
                PT_IRIDESCENCE_TEXTURE_RECORD_VERSION,
                crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE
                    | if factor.is_some_and(|binding| binding.transform.tex_coord == 1) {
                        PT_IRIDESCENCE_FACTOR_UV1
                    } else {
                        0
                    }
                    | if thickness.is_some_and(|binding| binding.transform.tex_coord == 1) {
                        PT_IRIDESCENCE_THICKNESS_UV1
                    } else {
                        0
                    },
                factor
                    .and_then(|binding| binding.runtime_texture_idx)
                    .unwrap_or(0),
                thickness
                    .and_then(|binding| binding.runtime_texture_idx)
                    .unwrap_or(0),
            ],
            factor_matrix,
            thickness_matrix,
            offsets: [
                factor_offset[0],
                factor_offset[1],
                thickness_offset[0],
                thickness_offset[1],
            ],
        }
    }

    pub(super) fn active(self) -> bool {
        self.header[0] == PT_IRIDESCENCE_TEXTURE_RECORD_VERSION
            && self.header[1] & crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE != 0
    }

    pub(super) fn has_uv1(self) -> bool {
        self.active()
            && self.header[1] & (PT_IRIDESCENCE_FACTOR_UV1 | PT_IRIDESCENCE_THICKNESS_UV1) != 0
    }
}

pub(super) const PT_IRIDESCENCE_TEXTURE_BINDINGS_WGSL: &str = r#"
const PT_HAS_IRIDESCENCE_TEXTURES: bool = true;
const PT_IRIDESCENCE_FACTOR_UV1: u32 = 65536u;
const PT_IRIDESCENCE_THICKNESS_UV1: u32 = 131072u;

struct PtIridescenceTexture {
    header: vec4<u32>,
    factor_matrix: vec4<f32>,
    thickness_matrix: vec4<f32>,
    offsets: vec4<f32>,
};
@group(2) @binding(6)
var<storage, read> pt_iridescence_textures: array<PtIridescenceTexture>;

fn pt_iridescence_transform_uv(
    uv: vec2<f32>,
    matrix: vec4<f32>,
    offset: vec2<f32>,
) -> vec2<f32> {
    return vec2<f32>(
        matrix.x * uv.x + matrix.z * uv.y,
        matrix.y * uv.x + matrix.w * uv.y,
    ) + offset;
}

fn pt_layered_apply_iridescence_textures(
    material_in: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
) -> PtLayeredMaterial {
    var material = material_in;
    let texture_meta = pt_iridescence_textures[instance_index];
    if (
        texture_meta.header.x != 1u
            || (texture_meta.header.y & PT_LAYERED_IRIDESCENCE_LOBE) == 0u
    ) {
        return material;
    }
    var factor = material.iridescence.x;
    var thickness = material.iridescence.w;
    if (texture_meta.header.z != 0u) {
        let factor_uv = pt_iridescence_transform_uv(
            select(uv0, uv1, (texture_meta.header.y & PT_IRIDESCENCE_FACTOR_UV1) != 0u),
            texture_meta.factor_matrix,
            texture_meta.offsets.xy,
        );
        factor *= pt_tex_sample_rgba(texture_meta.header.z, factor_uv).r;
    }
    if (texture_meta.header.w != 0u) {
        let thickness_uv = pt_iridescence_transform_uv(
            select(
                uv0,
                uv1,
                (texture_meta.header.y & PT_IRIDESCENCE_THICKNESS_UV1) != 0u,
            ),
            texture_meta.thickness_matrix,
            texture_meta.offsets.zw,
        );
        let sampled = pt_tex_sample_rgba(texture_meta.header.w, thickness_uv).g;
        thickness = mix(
            max(material.iridescence.z, 0.0),
            max(material.iridescence.w, 0.0),
            sampled,
        );
    }
    if (thickness <= 0.0) {
        factor = 0.0;
    }
    material.iridescence = vec4<f32>(
        clamp(factor, 0.0, 1.0),
        material.iridescence.y,
        material.iridescence.z,
        max(thickness, 0.0),
    );
    material.header = vec4<u32>(
        material.header.xy,
        material.header.z & ~PT_LAYERED_IRIDESCENCE_LOBE,
        material.header.w,
    );
    return material;
}
"#;

pub(super) const PT_IRIDESCENCE_TEXTURE_DISABLED_WGSL: &str = r#"
const PT_HAS_IRIDESCENCE_TEXTURES: bool = false;

fn pt_layered_apply_iridescence_textures(
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
        assert_eq!(PT_IRIDESCENCE_TEXTURE_RECORD_BYTES, 64);
        assert!(!PtIridescenceTextureCpu::default().active());
    }

    #[test]
    fn qualification_requires_resolved_coordinates() {
        let material = crate::models::MaterialLayeredPbr {
            iridescence_authored: true,
            iridescence_factor: 1.0,
            iridescence_thickness_minimum: 100.0,
            iridescence_thickness_maximum: 400.0,
            iridescence_texture: Some(binding(Some(2), 0)),
            iridescence_thickness_texture: Some(binding(Some(3), 1)),
            ..Default::default()
        };
        assert!(!PtIridescenceTextureCpu::from_material(material, 4, false).active());
        let record = PtIridescenceTextureCpu::from_material(material, 4, true);
        assert!(record.active() && record.has_uv1());
        assert_eq!(record.header[2..], [2, 3]);

        let unresolved = crate::models::MaterialLayeredPbr {
            iridescence_texture: Some(binding(None, 0)),
            ..material
        };
        assert!(!PtIridescenceTextureCpu::from_material(unresolved, 4, true).active());
    }

    #[test]
    fn first_iridescence_texture_backfills_parallel_records_and_reports_uv1() {
        let mut records = None;
        let mut specular_records = None;
        let mut clearcoat_records = None;
        let mut sheen_records = None;
        let mut iridescence_records = None;
        let mut anisotropy_records = None;
        assert!(!super::super::texture::append_record(
            &mut records,
            &mut specular_records,
            &mut clearcoat_records,
            &mut sheen_records,
            &mut iridescence_records,
            &mut anisotropy_records,
            0,
            Default::default(),
            4,
            true,
        ));
        let material = crate::models::MaterialLayeredPbr {
            iridescence_authored: true,
            iridescence_factor: 1.0,
            iridescence_thickness_minimum: 100.0,
            iridescence_thickness_maximum: 400.0,
            iridescence_texture: Some(binding(Some(2), 1)),
            ..Default::default()
        };
        assert!(super::super::texture::append_record(
            &mut records,
            &mut specular_records,
            &mut clearcoat_records,
            &mut sheen_records,
            &mut iridescence_records,
            &mut anisotropy_records,
            1,
            material,
            4,
            true,
        ));
        assert!(specular_records.is_none());
        assert!(clearcoat_records.is_none());
        assert!(sheen_records.is_none());
        assert!(anisotropy_records.is_none());
        let records = iridescence_records.unwrap();
        assert_eq!(records.len(), 2);
        assert!(!records[0].active());
        assert!(records[1].active() && records[1].has_uv1());
    }
}
