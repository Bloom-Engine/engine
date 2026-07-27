//! Independently lazy texture metadata for layered path tracing.
//!
//! This stays separate from the scalar sidecar so texture transforms never
//! grow or bind the scalar-only ABI.

use super::*;

const PT_LAYERED_TEXTURE_RECORD_VERSION: u32 = 1;
const PT_LAYERED_FACTOR_UV1: u32 = 1 << 16;
const PT_LAYERED_COLOR_UV1: u32 = 1 << 17;

pub(in crate::renderer) struct PtLayeredRuntimeState {
    pub(in crate::renderer) pipelines: [Option<wgpu::ComputePipeline>; 256],
    pub(in crate::renderer) layouts: [Option<wgpu::BindGroupLayout>; 64],
    pub(in crate::renderer) bind_groups: [Option<wgpu::BindGroup>; 64],
    pub(in crate::renderer) instance_buffer: Option<wgpu::Buffer>,
    pub(in crate::renderer) records: Vec<PtLayeredMaterialCpu>,
    pub(in crate::renderer) texture_buffer: Option<wgpu::Buffer>,
    pub(in crate::renderer) uv1_buffer: Option<wgpu::Buffer>,
    pub(in crate::renderer) texture_records: Vec<PtLayeredTextureCpu>,
    pub(in crate::renderer) clearcoat_texture_buffer: Option<wgpu::Buffer>,
    pub(in crate::renderer) clearcoat_texture_records: Vec<PtClearcoatTextureCpu>,
    pub(in crate::renderer) sheen_texture_buffer: Option<wgpu::Buffer>,
    pub(in crate::renderer) sheen_texture_records: Vec<PtSheenTextureCpu>,
    pub(in crate::renderer) iridescence_texture_buffer: Option<wgpu::Buffer>,
    pub(in crate::renderer) iridescence_texture_records: Vec<PtIridescenceTextureCpu>,
    pub(in crate::renderer) dirty: bool,
    pub(in crate::renderer) texture_dirty: bool,
    pub(in crate::renderer) clearcoat_texture_dirty: bool,
    pub(in crate::renderer) sheen_texture_dirty: bool,
    pub(in crate::renderer) iridescence_texture_dirty: bool,
}

impl Default for PtLayeredRuntimeState {
    fn default() -> Self {
        Self {
            pipelines: std::array::from_fn(|_| None),
            layouts: std::array::from_fn(|_| None),
            bind_groups: std::array::from_fn(|_| None),
            instance_buffer: None,
            records: Vec::new(),
            texture_buffer: None,
            uv1_buffer: None,
            texture_records: Vec::new(),
            clearcoat_texture_buffer: None,
            clearcoat_texture_records: Vec::new(),
            sheen_texture_buffer: None,
            sheen_texture_records: Vec::new(),
            iridescence_texture_buffer: None,
            iridescence_texture_records: Vec::new(),
            dirty: false,
            texture_dirty: false,
            clearcoat_texture_dirty: false,
            sheen_texture_dirty: false,
            iridescence_texture_dirty: false,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::renderer) struct PtLayeredTextureCpu {
    /// x = ABI version, y = qualified textured-lobe mask,
    /// z/w = specular-factor/specular-color runtime texture indices.
    header: [u32; 4],
    /// Column-major 2x2 UV matrices after authored scale + rotation.
    specular_factor_matrix: [f32; 4],
    specular_color_matrix: [f32; 4],
    /// xy = factor offset, zw = color offset.
    specular_offsets: [f32; 4],
}

pub(super) const PT_LAYERED_TEXTURE_RECORD_BYTES: u64 =
    std::mem::size_of::<PtLayeredTextureCpu>() as u64;

impl Default for PtLayeredTextureCpu {
    fn default() -> Self {
        Self {
            header: [PT_LAYERED_TEXTURE_RECORD_VERSION, 0, 0, 0],
            specular_factor_matrix: [1.0, 0.0, 0.0, 1.0],
            specular_color_matrix: [1.0, 0.0, 0.0, 1.0],
            specular_offsets: [0.0; 4],
        }
    }
}

impl PtLayeredTextureCpu {
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

        let factor = material.specular_texture;
        let color = material.specular_color_texture;
        let has_texture = factor.is_some() || color.is_some();
        let all_usable = factor
            .is_none_or(|binding| usable(binding, runtime_texture_count, has_secondary_uv))
            && color.is_none_or(|binding| usable(binding, runtime_texture_count, has_secondary_uv));
        if !material.has_specular_ior() || !has_texture || !all_usable {
            return Self::default();
        }

        let (factor_matrix, factor_offset) = transform(factor);
        let (color_matrix, color_offset) = transform(color);
        Self {
            header: [
                PT_LAYERED_TEXTURE_RECORD_VERSION,
                crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE
                    | if factor.is_some_and(|binding| binding.transform.tex_coord == 1) {
                        PT_LAYERED_FACTOR_UV1
                    } else {
                        0
                    }
                    | if color.is_some_and(|binding| binding.transform.tex_coord == 1) {
                        PT_LAYERED_COLOR_UV1
                    } else {
                        0
                    },
                factor
                    .and_then(|binding| binding.runtime_texture_idx)
                    .unwrap_or(0),
                color
                    .and_then(|binding| binding.runtime_texture_idx)
                    .unwrap_or(0),
            ],
            specular_factor_matrix: factor_matrix,
            specular_color_matrix: color_matrix,
            specular_offsets: [
                factor_offset[0],
                factor_offset[1],
                color_offset[0],
                color_offset[1],
            ],
        }
    }

    pub(super) fn has_specular_ior(self) -> bool {
        self.header[0] == PT_LAYERED_TEXTURE_RECORD_VERSION
            && self.header[1] & crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE != 0
    }

    pub(super) fn has_uv1(self) -> bool {
        self.has_specular_ior()
            && self.header[1] & (PT_LAYERED_FACTOR_UV1 | PT_LAYERED_COLOR_UV1) != 0
    }
}

/// Append scalar and texture records parallel to the next TLAS instance.
/// Both vectors remain absent until their first contributing material.
pub(in crate::renderer) fn append_record(
    records: &mut Option<Vec<PtLayeredMaterialCpu>>,
    texture_records: &mut Option<Vec<PtLayeredTextureCpu>>,
    clearcoat_texture_records: &mut Option<Vec<PtClearcoatTextureCpu>>,
    sheen_texture_records: &mut Option<Vec<PtSheenTextureCpu>>,
    iridescence_texture_records: &mut Option<Vec<PtIridescenceTextureCpu>>,
    instance_index: usize,
    material: crate::models::MaterialLayeredPbr,
    runtime_texture_count: usize,
    has_secondary_uv: bool,
) -> bool {
    let active = material.is_active();
    if records.is_none() && !active {
        return false;
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

    let texture_record =
        PtLayeredTextureCpu::from_material(material, runtime_texture_count, has_secondary_uv);
    let uses_uv1 = texture_record.has_uv1();
    if texture_records.is_none() && texture_record.has_specular_ior() {
        *texture_records = Some(vec![PtLayeredTextureCpu::default(); instance_index]);
    }
    if let Some(texture_records) = texture_records {
        debug_assert_eq!(texture_records.len(), instance_index);
        texture_records.push(texture_record);
    }

    let clearcoat_texture_record =
        PtClearcoatTextureCpu::from_material(material, runtime_texture_count, has_secondary_uv);
    let uses_uv1 = uses_uv1 || clearcoat_texture_record.has_uv1();
    if clearcoat_texture_records.is_none() && clearcoat_texture_record.active() {
        *clearcoat_texture_records = Some(vec![PtClearcoatTextureCpu::default(); instance_index]);
    }
    if let Some(clearcoat_texture_records) = clearcoat_texture_records {
        debug_assert_eq!(clearcoat_texture_records.len(), instance_index);
        clearcoat_texture_records.push(clearcoat_texture_record);
    }

    let sheen_texture_record =
        PtSheenTextureCpu::from_material(material, runtime_texture_count, has_secondary_uv);
    let uses_uv1 = uses_uv1 || sheen_texture_record.has_uv1();
    if sheen_texture_records.is_none() && sheen_texture_record.active() {
        *sheen_texture_records = Some(vec![PtSheenTextureCpu::default(); instance_index]);
    }
    if let Some(sheen_texture_records) = sheen_texture_records {
        debug_assert_eq!(sheen_texture_records.len(), instance_index);
        sheen_texture_records.push(sheen_texture_record);
    }

    let iridescence_texture_record =
        PtIridescenceTextureCpu::from_material(material, runtime_texture_count, has_secondary_uv);
    let uses_uv1 = uses_uv1 || iridescence_texture_record.has_uv1();
    if iridescence_texture_records.is_none() && iridescence_texture_record.active() {
        *iridescence_texture_records =
            Some(vec![PtIridescenceTextureCpu::default(); instance_index]);
    }
    if let Some(iridescence_texture_records) = iridescence_texture_records {
        debug_assert_eq!(iridescence_texture_records.len(), instance_index);
        iridescence_texture_records.push(iridescence_texture_record);
    }
    uses_uv1
}

pub(super) fn texture_variant(enabled: bool) -> &'static str {
    if enabled {
        "const PT_HAS_TEXTURES: bool = true;\n\
         @group(1) @binding(0) var pt_textures: binding_array<texture_2d<f32>>;\n\
         fn pt_tex_sample_rgba(idx: u32, uv: vec2<f32>) -> vec4<f32> {\n\
             return textureSampleLevel(pt_textures[idx], card_samp, uv, 0.0);\n\
         }\n\
         fn pt_tex_sample(idx: u32, uv: vec2<f32>) -> vec3<f32> {\n\
             return pt_tex_sample_rgba(idx, uv).rgb;\n\
         }\n"
    } else {
        "const PT_HAS_TEXTURES: bool = false;\n\
         fn pt_tex_sample_rgba(idx: u32, uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(1.0); }\n\
         fn pt_tex_sample(idx: u32, uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(1.0); }\n"
    }
}

pub(super) const PT_LAYERED_TEXTURE_BINDINGS_WGSL: &str = r#"
const PT_HAS_LAYERED_TEXTURES: bool = true;
const PT_LAYERED_FACTOR_UV1: u32 = 65536u;
const PT_LAYERED_COLOR_UV1: u32 = 131072u;

struct PtLayeredTexture {
    header: vec4<u32>,
    specular_factor_matrix: vec4<f32>,
    specular_color_matrix: vec4<f32>,
    specular_offsets: vec4<f32>,
};
@group(2) @binding(2)
var<storage, read> pt_layered_textures: array<PtLayeredTexture>;

fn pt_layered_transform_uv(
    uv: vec2<f32>,
    matrix: vec4<f32>,
    offset: vec2<f32>,
) -> vec2<f32> {
    return vec2<f32>(
        matrix.x * uv.x + matrix.z * uv.y,
        matrix.y * uv.x + matrix.w * uv.y,
    ) + offset;
}

fn pt_layered_srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}

fn pt_layered_apply_textures(
    material_in: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
) -> PtLayeredMaterial {
    var material = material_in;
    let texture_meta = pt_layered_textures[instance_index];
    if (
        texture_meta.header.x != 1u
            || (texture_meta.header.y & PT_LAYERED_SPECULAR_IOR_LOBE) == 0u
    ) {
        return material;
    }
    var specular_color = material.specular.xyz;
    var specular_factor = material.specular.w;
    if (texture_meta.header.z != 0u) {
        let factor_uv = pt_layered_transform_uv(
            select(uv0, uv1, (texture_meta.header.y & PT_LAYERED_FACTOR_UV1) != 0u),
            texture_meta.specular_factor_matrix,
            texture_meta.specular_offsets.xy,
        );
        specular_factor *= pt_tex_sample_rgba(
            texture_meta.header.z, factor_uv,
        ).a;
    }
    if (texture_meta.header.w != 0u) {
        let color_uv = pt_layered_transform_uv(
            select(uv0, uv1, (texture_meta.header.y & PT_LAYERED_COLOR_UV1) != 0u),
            texture_meta.specular_color_matrix,
            texture_meta.specular_offsets.zw,
        );
        specular_color *= pt_layered_srgb_to_linear(
            pt_tex_sample_rgba(texture_meta.header.w, color_uv).rgb,
        );
    }
    material.specular = vec4<f32>(
        max(specular_color, vec3<f32>(0.0)),
        clamp(specular_factor, 0.0, 1.0),
    );
    material.header = vec4<u32>(
        material.header.xy,
        material.header.z & ~PT_LAYERED_SPECULAR_IOR_LOBE,
        material.header.w,
    );
    return material;
}
"#;

pub(super) const PT_LAYERED_TEXTURE_DISABLED_WGSL: &str = r#"
const PT_HAS_LAYERED_TEXTURES: bool = false;

fn pt_layered_apply_textures(
    material: PtLayeredMaterial,
    instance_index: u32,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
) -> PtLayeredMaterial {
    return material;
}
"#;

pub(super) const PT_LAYERED_UV1_BINDINGS_WGSL: &str = r#"
@group(2) @binding(3)
var<storage, read> pt_layered_uv1_vertices: array<vec2<f32>>;

fn pt_layered_hit_uv1(
    geo: vec4<u32>,
    primitive: u32,
    barycentrics: vec2<f32>,
) -> vec2<f32> {
    let base = geo.y + primitive * 3u;
    let slot0 = geo.x + geo_i[base];
    let slot1 = geo.x + geo_i[base + 1u];
    let slot2 = geo.x + geo_i[base + 2u];
    let weight0 = 1.0 - barycentrics.x - barycentrics.y;
    return weight0 * pt_layered_uv1_vertices[slot0]
        + barycentrics.x * pt_layered_uv1_vertices[slot1]
        + barycentrics.y * pt_layered_uv1_vertices[slot2];
}
"#;

pub(super) const PT_LAYERED_UV1_DISABLED_WGSL: &str = r#"
fn pt_layered_hit_uv1(
    geo: vec4<u32>,
    primitive: u32,
    barycentrics: vec2<f32>,
) -> vec2<f32> {
    return vec2<f32>(0.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_is_four_vec4s_and_default_is_inactive() {
        assert_eq!(std::mem::size_of::<PtLayeredTextureCpu>(), 64);
        assert!(!PtLayeredTextureCpu::default().has_specular_ior());
    }

    #[test]
    fn first_active_record_backfills_base_instances_lazily() {
        let mut records = None;
        let mut texture_records = None;
        let mut clearcoat_texture_records = None;
        let mut sheen_texture_records = None;
        let mut iridescence_texture_records = None;
        append_record(
            &mut records,
            &mut texture_records,
            &mut clearcoat_texture_records,
            &mut sheen_texture_records,
            &mut iridescence_texture_records,
            0,
            Default::default(),
            1,
            false,
        );
        append_record(
            &mut records,
            &mut texture_records,
            &mut clearcoat_texture_records,
            &mut sheen_texture_records,
            &mut iridescence_texture_records,
            1,
            Default::default(),
            1,
            false,
        );
        assert!(records.is_none());
        assert!(texture_records.is_none());
        assert!(clearcoat_texture_records.is_none());
        assert!(sheen_texture_records.is_none());
        assert!(iridescence_texture_records.is_none());

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
        append_record(
            &mut records,
            &mut texture_records,
            &mut clearcoat_texture_records,
            &mut sheen_texture_records,
            &mut iridescence_texture_records,
            2,
            layered,
            1,
            false,
        );
        let records = records.unwrap();
        assert_eq!(records.len(), 3);
        assert!(!records[0].active());
        assert!(!records[1].active());
        assert_eq!(
            records[2].header[1],
            crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE
        );
        assert!(texture_records.is_none());
        assert!(clearcoat_texture_records.is_none());
        assert!(sheen_texture_records.is_none());
        assert!(iridescence_texture_records.is_none());
    }

    #[test]
    fn qualifies_only_resolved_specular_bindings_with_available_coordinates() {
        let binding = crate::models::MaterialTextureBinding {
            source_texture_index: 4,
            source_image_index: 5,
            runtime_texture_idx: Some(3),
            transform: crate::models::MaterialTextureTransform {
                offset: [0.25, -0.5],
                rotation: std::f32::consts::FRAC_PI_2,
                scale: [2.0, 3.0],
                tex_coord: 0,
            },
        };
        let material = crate::models::MaterialLayeredPbr {
            specular_authored: true,
            specular_factor: 0.8,
            specular_texture: Some(binding),
            ..Default::default()
        };
        let record = PtLayeredTextureCpu::from_material(material, 4, false);
        assert!(record.has_specular_ior());
        assert_eq!(record.header[2..], [3, 0]);
        assert!(record.specular_factor_matrix[0].abs() < 1e-6);
        assert!((record.specular_factor_matrix[1] - 2.0).abs() < 1e-6);
        assert!((record.specular_factor_matrix[2] + 3.0).abs() < 1e-6);
        assert!(record.specular_factor_matrix[3].abs() < 1e-6);
        assert_eq!(record.specular_offsets[..2], [0.25, -0.5]);

        let unresolved = crate::models::MaterialLayeredPbr {
            specular_texture: Some(crate::models::MaterialTextureBinding {
                runtime_texture_idx: None,
                ..binding
            }),
            ..material
        };
        assert!(!PtLayeredTextureCpu::from_material(unresolved, 4, false).has_specular_ior());

        let uv1 = crate::models::MaterialLayeredPbr {
            specular_texture: Some(crate::models::MaterialTextureBinding {
                transform: crate::models::MaterialTextureTransform {
                    tex_coord: 1,
                    ..binding.transform
                },
                ..binding
            }),
            ..material
        };
        assert!(!PtLayeredTextureCpu::from_material(uv1, 4, false).has_specular_ior());
        let uv1_record = PtLayeredTextureCpu::from_material(uv1, 4, true);
        assert!(uv1_record.has_specular_ior() && uv1_record.has_uv1());
    }

    #[test]
    fn scalar_uv0_and_uv1_specializations_parse() {
        for (textures, clearcoat_textures, sheen_textures, iridescence_textures, uv1) in [
            (false, false, false, false, false),
            (true, false, false, false, false),
            (true, false, false, false, true),
            (false, true, false, false, false),
            (false, true, false, false, true),
            (false, false, true, false, false),
            (false, false, true, false, true),
            (false, false, false, true, false),
            (false, false, false, true, true),
            (true, true, true, true, true),
        ] {
            let source = format!(
                "enable wgpu_ray_query;\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                "const BLOOM_RAY_QUERY_NEEDS_PROCEED: bool = false;",
                pt_fault_constants(None),
                layered_kernel_variant(pt_kernel_variant(false).as_ref()),
                texture_variant(
                    textures || clearcoat_textures || sheen_textures || iridescence_textures
                ),
                PT_LAYERED_BINDINGS_WGSL,
                if textures {
                    PT_LAYERED_TEXTURE_BINDINGS_WGSL
                } else {
                    PT_LAYERED_TEXTURE_DISABLED_WGSL
                },
                if clearcoat_textures {
                    super::super::PT_CLEARCOAT_TEXTURE_BINDINGS_WGSL
                } else {
                    super::super::PT_CLEARCOAT_TEXTURE_DISABLED_WGSL
                },
                if sheen_textures {
                    super::super::PT_SHEEN_TEXTURE_BINDINGS_WGSL
                } else {
                    super::super::PT_SHEEN_TEXTURE_DISABLED_WGSL
                },
                if iridescence_textures {
                    super::super::PT_IRIDESCENCE_TEXTURE_BINDINGS_WGSL
                } else {
                    super::super::PT_IRIDESCENCE_TEXTURE_DISABLED_WGSL
                },
                if uv1 {
                    PT_LAYERED_UV1_BINDINGS_WGSL
                } else {
                    PT_LAYERED_UV1_DISABLED_WGSL
                },
                "const PT_HAS_SCALAR_ANISOTROPY: bool = false;",
                PT_LAYERED_TRANSPORT_WGSL,
                if iridescence_textures {
                    PT_LAYERED_IRIDESCENCE_WGSL
                } else {
                    PT_LAYERED_IRIDESCENCE_DISABLED_WGSL
                },
                if sheen_textures {
                    PT_LAYERED_SHEEN_WGSL
                } else {
                    PT_LAYERED_SHEEN_DISABLED_WGSL
                },
            );
            wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                panic!(
                    "layered PT WGSL (specular={textures}, clearcoat={clearcoat_textures}, \
                     sheen={sheen_textures}, iridescence={iridescence_textures}, \
                     uv1={uv1}) failed: {error}"
                )
            });
        }
    }
}
