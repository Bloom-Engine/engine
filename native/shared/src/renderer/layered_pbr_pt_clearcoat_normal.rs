//! Independently lazy clearcoat-normal metadata for path tracing.
//!
//! Keeping this record separate preserves the exact 64-byte
//! factor/roughness sidecar for materials that do not author a coat normal.

use super::*;

const PT_CLEARCOAT_NORMAL_RECORD_VERSION: u32 = 1;
const PT_CLEARCOAT_NORMAL_UV1: u32 = 1 << 16;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::renderer) struct PtClearcoatNormalCpu {
    /// x = ABI version, y = qualified lobe + UV flags,
    /// z = normal runtime texture index, w = reserved.
    header: [u32; 4],
    /// Column-major 2x2 UV matrix after authored scale + rotation.
    matrix: [f32; 4],
    /// xy = offset, z = tangent-space normal scale, w = reserved.
    params: [f32; 4],
}

pub(super) const PT_CLEARCOAT_NORMAL_RECORD_BYTES: u64 =
    std::mem::size_of::<PtClearcoatNormalCpu>() as u64;

impl Default for PtClearcoatNormalCpu {
    fn default() -> Self {
        Self {
            header: [PT_CLEARCOAT_NORMAL_RECORD_VERSION, 0, 0, 0],
            matrix: [1.0, 0.0, 0.0, 1.0],
            params: [0.0, 0.0, 1.0, 0.0],
        }
    }
}

impl PtClearcoatNormalCpu {
    pub(super) fn from_material(
        material: crate::models::MaterialLayeredPbr,
        runtime_texture_count: usize,
        has_secondary_uv: bool,
    ) -> Self {
        let Some(binding) = material.clearcoat_normal_texture else {
            return Self::default();
        };
        let transform = binding.transform;
        let usable = binding.runtime_texture_idx.is_some_and(|index| {
            index != 0
                && (index as usize) < PT_MAX_TEXTURES
                && (index as usize) < runtime_texture_count
        }) && (transform.tex_coord == 0
            || (transform.tex_coord == 1 && has_secondary_uv))
            && transform.offset.iter().all(|value| value.is_finite())
            && transform.scale.iter().all(|value| value.is_finite())
            && transform.rotation.is_finite()
            && material.clearcoat_normal_scale.is_finite();
        if !material.has_clearcoat() || !usable {
            return Self::default();
        }

        let (sine, cosine) = transform.rotation.sin_cos();
        Self {
            header: [
                PT_CLEARCOAT_NORMAL_RECORD_VERSION,
                crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE
                    | if transform.tex_coord == 1 {
                        PT_CLEARCOAT_NORMAL_UV1
                    } else {
                        0
                    },
                binding.runtime_texture_idx.unwrap_or(0),
                0,
            ],
            matrix: [
                cosine * transform.scale[0],
                sine * transform.scale[0],
                -sine * transform.scale[1],
                cosine * transform.scale[1],
            ],
            params: [
                transform.offset[0],
                transform.offset[1],
                material.clearcoat_normal_scale,
                0.0,
            ],
        }
    }

    pub(super) fn active(self) -> bool {
        self.header[0] == PT_CLEARCOAT_NORMAL_RECORD_VERSION
            && self.header[1] & crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE != 0
            && self.header[2] != 0
    }

    pub(super) fn has_uv1(self) -> bool {
        self.active() && self.header[1] & PT_CLEARCOAT_NORMAL_UV1 != 0
    }
}

pub(super) const PT_CLEARCOAT_NORMAL_BINDINGS_WGSL: &str = r#"
const PT_HAS_CLEARCOAT_NORMALS: bool = true;
const PT_CLEARCOAT_NORMAL_UV1: u32 = 65536u;

struct PtClearcoatNormalTexture {
    header: vec4<u32>,
    matrix: vec4<f32>,
    params: vec4<f32>,
};
struct PtClearcoatNormalSample {
    material: PtLayeredMaterial,
    normal: vec3<f32>,
};
@group(2) @binding(8)
var<storage, read> pt_clearcoat_normals: array<PtClearcoatNormalTexture>;

fn pt_layered_has_clearcoat_normal(instance_index: u32) -> bool {
    let texture_meta = pt_clearcoat_normals[instance_index];
    return texture_meta.header.x == 1u
        && (texture_meta.header.y & PT_LAYERED_CLEARCOAT_LOBE) != 0u
        && texture_meta.header.z != 0u;
}

fn pt_layered_apply_clearcoat_normal(
    material_in: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    base_normal: vec3<f32>,
    tangent: vec4<f32>,
) -> PtClearcoatNormalSample {
    var result = PtClearcoatNormalSample(material_in, base_normal);
    let texture_meta = pt_clearcoat_normals[instance_index];
    if (
        !pt_layered_has_clearcoat(material_in)
            || !pt_layered_has_clearcoat_normal(instance_index)
    ) {
        return result;
    }
    let source_uv = select(
        uv0,
        uv1,
        (texture_meta.header.y & PT_CLEARCOAT_NORMAL_UV1) != 0u,
    );
    let uv = vec2<f32>(
        texture_meta.matrix.x * source_uv.x
            + texture_meta.matrix.z * source_uv.y,
        texture_meta.matrix.y * source_uv.x
            + texture_meta.matrix.w * source_uv.y,
    ) + texture_meta.params.xy;
    let sampled = pt_tex_sample_rgba(texture_meta.header.z, uv);
    var tangent_normal = sampled.xyz * 2.0 - vec3<f32>(1.0);
    tangent_normal.x *= texture_meta.params.z;
    tangent_normal.y *= texture_meta.params.z;
    let normal_len2 = clamp(dot(tangent_normal, tangent_normal), 0.01, 1.0);
    tangent_normal *= inverseSqrt(normal_len2);

    let tangent_ortho =
        tangent.xyz - base_normal * dot(base_normal, tangent.xyz);
    var basis = onb(base_normal);
    if (dot(tangent_ortho, tangent_ortho) > 1e-4) {
        let mesh_tangent = normalize(tangent_ortho);
        let mesh_bitangent =
            normalize(cross(base_normal, mesh_tangent)) * tangent.w;
        basis = mat3x3<f32>(mesh_tangent, mesh_bitangent, base_normal);
    }
    let mapped_raw = basis * tangent_normal;
    let mapped_len2 = dot(mapped_raw, mapped_raw);
    var mapped = base_normal;
    if (mapped_len2 > 1e-8) {
        mapped = mapped_raw * inverseSqrt(mapped_len2);
    }
    let hemisphere = dot(mapped, base_normal);
    result.normal = normalize(
        mapped + base_normal * max(0.05 - hemisphere, 0.0),
    );

    // PT has no screen-space derivatives, but it consumes the same baked
    // normal-length/variance channels as realtime shading. This keeps the
    // sharp coat stable without imposing derivative work on the compute path.
    let sigma2_toksvig = (1.0 - normal_len2) / normal_len2;
    let baked_variance = clamp(sampled.a, 0.0, 0.999);
    let sigma2_baked = baked_variance / max(1.0 - baked_variance, 0.001);
    let roughness2 = min(
        material_in.clearcoat_ior.y * material_in.clearcoat_ior.y
            + sigma2_toksvig
            + sigma2_baked,
        1.0,
    );
    result.material.clearcoat_ior = vec4<f32>(
        material_in.clearcoat_ior.x,
        sqrt(roughness2),
        material_in.clearcoat_ior.zw,
    );
    return result;
}
"#;

pub(super) const PT_CLEARCOAT_NORMAL_DISABLED_WGSL: &str = r#"
const PT_HAS_CLEARCOAT_NORMALS: bool = false;

struct PtClearcoatNormalSample {
    material: PtLayeredMaterial,
    normal: vec3<f32>,
};

fn pt_layered_has_clearcoat_normal(instance_index: u32) -> bool {
    return false;
}

fn pt_layered_apply_clearcoat_normal(
    material: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    base_normal: vec3<f32>,
    tangent: vec4<f32>,
) -> PtClearcoatNormalSample {
    return PtClearcoatNormalSample(material, base_normal);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(index: Option<u32>, tex_coord: u32) -> crate::models::MaterialTextureBinding {
        crate::models::MaterialTextureBinding {
            source_texture_index: 4,
            source_image_index: 5,
            runtime_texture_idx: index,
            transform: crate::models::MaterialTextureTransform {
                offset: [0.25, -0.5],
                rotation: std::f32::consts::FRAC_PI_2,
                scale: [2.0, 3.0],
                tex_coord,
            },
        }
    }

    #[test]
    fn record_is_three_vec4s_and_inactive_by_default() {
        assert_eq!(PT_CLEARCOAT_NORMAL_RECORD_BYTES, 48);
        assert!(!PtClearcoatNormalCpu::default().active());
    }

    #[test]
    fn qualification_preserves_transform_scale_and_uv_selection() {
        let material = crate::models::MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 0.8,
            clearcoat_normal_scale: 0.6,
            clearcoat_normal_texture: Some(binding(Some(3), 1)),
            ..Default::default()
        };
        assert!(!PtClearcoatNormalCpu::from_material(material, 4, false).active());
        let record = PtClearcoatNormalCpu::from_material(material, 4, true);
        assert!(record.active() && record.has_uv1());
        assert_eq!(record.header[2], 3);
        assert!(record.matrix[0].abs() < 1e-6);
        assert!((record.matrix[1] - 2.0).abs() < 1e-6);
        assert!((record.matrix[2] + 3.0).abs() < 1e-6);
        assert!(record.matrix[3].abs() < 1e-6);
        assert_eq!(record.params[..3], [0.25, -0.5, 0.6]);

        let unresolved = crate::models::MaterialLayeredPbr {
            clearcoat_normal_texture: Some(binding(None, 0)),
            ..material
        };
        assert!(!PtClearcoatNormalCpu::from_material(unresolved, 4, true).active());

        let invalid_scale = crate::models::MaterialLayeredPbr {
            clearcoat_normal_scale: f32::NAN,
            clearcoat_normal_texture: Some(binding(Some(3), 0)),
            ..material
        };
        assert!(!PtClearcoatNormalCpu::from_material(invalid_scale, 4, true).active());
    }

    #[test]
    fn first_normal_map_backfills_only_its_independent_sidecar() {
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
            4,
            true,
        ));
        let material = crate::models::MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 0.8,
            clearcoat_normal_texture: Some(binding(Some(3), 1)),
            ..Default::default()
        };
        assert!(super::super::texture::append_record(
            &mut records,
            &mut specular_records,
            &mut clearcoat_records,
            &mut clearcoat_normal_records,
            &mut sheen_records,
            &mut iridescence_records,
            &mut anisotropy_records,
            1,
            material,
            4,
            true,
        ));
        assert_eq!(records.as_ref().map(Vec::len), Some(2));
        assert!(records.unwrap()[1].has_clearcoat());
        assert!(specular_records.is_none());
        assert!(clearcoat_records.is_none());
        assert_eq!(clearcoat_normal_records.unwrap().len(), 2);
        assert!(sheen_records.is_none());
        assert!(iridescence_records.is_none());
        assert!(anisotropy_records.is_none());
    }
}
