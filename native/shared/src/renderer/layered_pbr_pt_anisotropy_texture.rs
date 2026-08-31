//! Independently lazy anisotropy direction/strength metadata for path tracing.

use super::*;

const PT_ANISOTROPY_TEXTURE_RECORD_VERSION: u32 = 1;
const PT_ANISOTROPY_TEXTURE_UV1: u32 = 1 << 16;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::renderer) struct PtAnisotropyTextureCpu {
    /// x = ABI version, y = qualified lobe + UV flags,
    /// z = runtime texture index.
    header: [u32; 4],
    /// Column-major 2x2 UV matrix after authored scale + rotation.
    matrix: [f32; 4],
    /// xy = authored UV offset; zw are reserved.
    offset: [f32; 4],
    reserved: [f32; 4],
}

pub(super) const PT_ANISOTROPY_TEXTURE_RECORD_BYTES: u64 =
    std::mem::size_of::<PtAnisotropyTextureCpu>() as u64;

impl Default for PtAnisotropyTextureCpu {
    fn default() -> Self {
        Self {
            header: [PT_ANISOTROPY_TEXTURE_RECORD_VERSION, 0, 0, 0],
            matrix: [1.0, 0.0, 0.0, 1.0],
            offset: [0.0; 4],
            reserved: [0.0; 4],
        }
    }
}

impl PtAnisotropyTextureCpu {
    pub(super) fn from_material(
        material: crate::models::MaterialLayeredPbr,
        runtime_texture_count: usize,
        has_secondary_uv: bool,
    ) -> Self {
        let Some(binding) = material.anisotropy_texture else {
            return Self::default();
        };
        let transform = binding.transform;
        let usable = material.has_anisotropy()
            && binding.runtime_texture_idx.is_some_and(|index| {
                index != 0
                    && (index as usize) < PT_MAX_TEXTURES
                    && (index as usize) < runtime_texture_count
            })
            && (transform.tex_coord == 0 || (transform.tex_coord == 1 && has_secondary_uv))
            && transform.offset.iter().all(|value| value.is_finite())
            && transform.scale.iter().all(|value| value.is_finite())
            && transform.rotation.is_finite();
        if !usable {
            return Self::default();
        }

        let (sine, cosine) = transform.rotation.sin_cos();
        Self {
            header: [
                PT_ANISOTROPY_TEXTURE_RECORD_VERSION,
                crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE
                    | if transform.tex_coord == 1 {
                        PT_ANISOTROPY_TEXTURE_UV1
                    } else {
                        0
                    },
                binding.runtime_texture_idx.unwrap(),
                0,
            ],
            matrix: [
                cosine * transform.scale[0],
                sine * transform.scale[0],
                -sine * transform.scale[1],
                cosine * transform.scale[1],
            ],
            offset: [transform.offset[0], transform.offset[1], 0.0, 0.0],
            reserved: [0.0; 4],
        }
    }

    pub(super) fn active(self) -> bool {
        self.header[0] == PT_ANISOTROPY_TEXTURE_RECORD_VERSION
            && self.header[1] & crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE != 0
    }

    pub(super) fn has_uv1(self) -> bool {
        self.active() && self.header[1] & PT_ANISOTROPY_TEXTURE_UV1 != 0
    }
}

pub(super) const PT_ANISOTROPY_TEXTURE_BINDINGS_WGSL: &str = r#"
const PT_HAS_ANISOTROPY_TEXTURES: bool = true;
const PT_ANISOTROPY_TEXTURE_UV1: u32 = 65536u;

struct PtAnisotropyTexture {
    header: vec4<u32>,
    matrix: vec4<f32>,
    offset: vec4<f32>,
    reserved: vec4<f32>,
};
@group(2) @binding(7)
var<storage, read> pt_anisotropy_textures: array<PtAnisotropyTexture>;

fn pt_anisotropy_transform_uv(
    uv: vec2<f32>,
    matrix: vec4<f32>,
    offset: vec2<f32>,
) -> vec2<f32> {
    return vec2<f32>(
        matrix.x * uv.x + matrix.z * uv.y,
        matrix.y * uv.x + matrix.w * uv.y,
    ) + offset;
}

fn pt_layered_apply_anisotropy_texture(
    material_in: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
) -> PtLayeredMaterial {
    var material = material_in;
    let texture_meta = pt_anisotropy_textures[instance_index];
    if (
        texture_meta.header.x != 1u
            || (texture_meta.header.y & PT_LAYERED_ANISOTROPY_LOBE) == 0u
    ) {
        return material;
    }
    let uv = pt_anisotropy_transform_uv(
        select(uv0, uv1, (texture_meta.header.y & PT_ANISOTROPY_TEXTURE_UV1) != 0u),
        texture_meta.matrix,
        texture_meta.offset.xy,
    );
    let sampled = pt_tex_sample_rgba(texture_meta.header.z, uv);
    var direction = sampled.rg * 2.0 - vec2<f32>(1.0);
    if (dot(direction, direction) <= 1e-6) {
        direction = vec2<f32>(1.0, 0.0);
    } else {
        direction = normalize(direction);
    }
    let rotated_direction = vec2<f32>(
        material.anisotropy.y * direction.x - material.anisotropy.z * direction.y,
        material.anisotropy.z * direction.x + material.anisotropy.y * direction.y,
    );
    material.anisotropy = vec4<f32>(
        clamp(material.anisotropy.x * sampled.b, 0.0, 1.0),
        rotated_direction,
        0.0,
    );
    material.header = vec4<u32>(
        material.header.xy,
        material.header.z & ~PT_LAYERED_ANISOTROPY_LOBE,
        material.header.w,
    );
    return material;
}
"#;

pub(super) const PT_ANISOTROPY_TEXTURE_DISABLED_WGSL: &str = r#"
const PT_HAS_ANISOTROPY_TEXTURES: bool = false;

fn pt_layered_apply_anisotropy_texture(
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

    fn material(
        binding: crate::models::MaterialTextureBinding,
    ) -> crate::models::MaterialLayeredPbr {
        crate::models::MaterialLayeredPbr {
            anisotropy_authored: true,
            anisotropy_strength: 0.8,
            anisotropy_texture: Some(binding),
            ..Default::default()
        }
    }

    #[test]
    fn record_is_independently_compact_and_inactive_by_default() {
        assert_eq!(PT_ANISOTROPY_TEXTURE_RECORD_BYTES, 64);
        assert!(!PtAnisotropyTextureCpu::default().active());
    }

    #[test]
    fn qualification_requires_resolved_coordinates() {
        assert!(
            !PtAnisotropyTextureCpu::from_material(material(binding(Some(2), 1)), 3, false,)
                .active()
        );
        let record = PtAnisotropyTextureCpu::from_material(material(binding(Some(2), 1)), 3, true);
        assert!(record.active() && record.has_uv1());
        assert_eq!(record.header[2], 2);
        assert!(
            !PtAnisotropyTextureCpu::from_material(material(binding(None, 0)), 3, true,).active()
        );
    }

    #[test]
    fn first_anisotropy_texture_backfills_parallel_records_and_reports_uv1() {
        let mut records = None;
        let mut specular_records = None;
        let mut clearcoat_records = None;
        let mut clearcoat_normal_records = None;
        let mut sheen_records = None;
        let mut iridescence_records = None;
        let mut anisotropy_records = None;
        assert!(!super::super::texture::append_record(
            &mut records,
            &mut specular_records,
            &mut clearcoat_records,
            &mut clearcoat_normal_records,
            &mut sheen_records,
            &mut iridescence_records,
            &mut anisotropy_records,
            0,
            Default::default(),
            3,
            true,
        ));
        assert!(super::super::texture::append_record(
            &mut records,
            &mut specular_records,
            &mut clearcoat_records,
            &mut clearcoat_normal_records,
            &mut sheen_records,
            &mut iridescence_records,
            &mut anisotropy_records,
            1,
            material(binding(Some(2), 1)),
            3,
            true,
        ));
        assert!(specular_records.is_none());
        assert!(clearcoat_records.is_none());
        assert!(sheen_records.is_none());
        assert!(iridescence_records.is_none());
        let records = anisotropy_records.unwrap();
        assert_eq!(records.len(), 2);
        assert!(!records[0].active());
        assert!(records[1].active() && records[1].has_uv1());
    }
}
