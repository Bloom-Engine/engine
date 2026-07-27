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
    /// x = ABI version, y = lobe mask, z = texture-bearing lobe mask.
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
        let mut texture_mask = 0;
        if material.has_clearcoat() {
            mask |= crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE;
        }
        if material.clearcoat_texture.is_some()
            || material.clearcoat_roughness_texture.is_some()
            || material.clearcoat_normal_texture.is_some()
        {
            texture_mask |= crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE;
        }
        if material.has_sheen() {
            mask |= crate::models::MaterialLayeredPbr::SHEEN_LOBE;
        }
        if material.sheen_color_texture.is_some() || material.sheen_roughness_texture.is_some() {
            texture_mask |= crate::models::MaterialLayeredPbr::SHEEN_LOBE;
        }
        if material.has_anisotropy() {
            mask |= crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE;
        }
        if material.anisotropy_texture.is_some() {
            texture_mask |= crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE;
        }
        if material.has_iridescence() {
            mask |= crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE;
        }
        if material.iridescence_texture.is_some()
            || material.iridescence_thickness_texture.is_some()
        {
            texture_mask |= crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE;
        }
        if material.has_specular_ior() {
            mask |= crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE;
        }
        if material.specular_texture.is_some() || material.specular_color_texture.is_some() {
            texture_mask |= crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE;
        }
        let rotation = finite_or(material.anisotropy_rotation, 0.0);
        let (rotation_sine, rotation_cosine) = rotation.sin_cos();
        Self {
            header: [PT_LAYERED_RECORD_VERSION, mask, texture_mask, 0],
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

    fn has_clearcoat(self) -> bool {
        self.header[1] & crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE != 0
            && self.header[2] & crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE == 0
            && self.clearcoat_ior[0] > 0.0
    }

    fn has_specular_ior(self) -> bool {
        self.header[1] & crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE != 0
            && self.header[2] & crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE == 0
    }

    fn has_sheen(self) -> bool {
        self.header[1] & crate::models::MaterialLayeredPbr::SHEEN_LOBE != 0
            && self.header[2] & crate::models::MaterialLayeredPbr::SHEEN_LOBE == 0
            && self.sheen[..3].iter().any(|value| *value > 0.0)
    }

    fn has_anisotropy(self) -> bool {
        self.header[1] & crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE != 0
            && self.header[2] & crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE == 0
            && self.anisotropy[0] > 0.0
    }

    fn has_iridescence(self) -> bool {
        self.header[1] & crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE != 0
            && self.header[2] & crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE == 0
            && self.iridescence[0] > 0.0
            && self.iridescence[3] > 0.0
    }

    fn has_qualified_transport(self) -> bool {
        self.has_clearcoat()
            || self.has_specular_ior()
            || self.has_sheen()
            || self.has_anisotropy()
            || self.has_iridescence()
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

const PT_LAYERED_TRANSPORT_WGSL: &str = r#"
const PT_LAYERED_CLEARCOAT_LOBE: u32 = 1u;
const PT_LAYERED_ANISOTROPY_LOBE: u32 = 4u;
const PT_LAYERED_SPECULAR_IOR_LOBE: u32 = 16u;

struct PtLayeredSurface {
    material: PtLayeredMaterial,
    tangent: vec4<f32>,
};

fn pt_layered_default() -> PtLayeredMaterial {
    return PtLayeredMaterial(
        vec4<u32>(1u, 0u, 0u, 0u),
        vec4<f32>(0.0, 0.0, 1.0, 1.5),
        vec4<f32>(1.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.3, 100.0, 400.0),
    );
}

fn pt_layered_has_clearcoat(material: PtLayeredMaterial) -> bool {
    return material.header.x == 1u
        && (material.header.y & PT_LAYERED_CLEARCOAT_LOBE) != 0u
        && (material.header.z & PT_LAYERED_CLEARCOAT_LOBE) == 0u
        && material.clearcoat_ior.x > 0.0;
}

fn pt_layered_has_specular_ior(material: PtLayeredMaterial) -> bool {
    return material.header.x == 1u
        && (material.header.y & PT_LAYERED_SPECULAR_IOR_LOBE) != 0u
        && (material.header.z & PT_LAYERED_SPECULAR_IOR_LOBE) == 0u;
}

fn pt_layered_has_anisotropy(material: PtLayeredMaterial) -> bool {
    return PT_HAS_SCALAR_ANISOTROPY
        && material.header.x == 1u
        && (material.header.y & PT_LAYERED_ANISOTROPY_LOBE) != 0u
        && (material.header.z & PT_LAYERED_ANISOTROPY_LOBE) == 0u
        && material.anisotropy.x > 0.0;
}

fn pt_layered_has_transport(material: PtLayeredMaterial) -> bool {
    return pt_layered_has_clearcoat(material)
        || pt_layered_has_specular_ior(material)
        || pt_layered_has_sheen(material)
        || pt_layered_has_anisotropy(material)
        || pt_layered_has_iridescence(material);
}

fn pt_ior_f0(ior_value: f32) -> f32 {
    if (ior_value == 0.0) {
        return 1.0;
    }
    let ior = max(ior_value, 1.0);
    let ratio = (ior - 1.0) / (ior + 1.0);
    return ratio * ratio;
}

fn pt_dielectric_f0(material: PtLayeredMaterial) -> vec3<f32> {
    return min(
        max(material.specular.xyz, vec3<f32>(0.0)) * pt_ior_f0(material.clearcoat_ior.w),
        vec3<f32>(1.0),
    ) * clamp(material.specular.w, 0.0, 1.0);
}

fn pt_fresnel_f90(cos_theta: f32, f0: vec3<f32>, f90: vec3<f32>) -> vec3<f32> {
    let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
    return f0 + (f90 - f0) * (m * m * m * m * m);
}

fn pt_layered_base_fresnel(
    cos_theta: f32,
    base_color: vec3<f32>,
    metallic: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    var base: vec3<f32>;
    if (pt_layered_has_specular_ior(material)) {
        let dielectric = pt_fresnel_f90(
            cos_theta,
            pt_dielectric_f0(material),
            vec3<f32>(clamp(material.specular.w, 0.0, 1.0)),
        );
        let conductor = fresnel_schlick3(cos_theta, base_color);
        base = mix(dielectric, conductor, metallic);
    } else {
        base = fresnel_schlick3(
            cos_theta, mix(vec3<f32>(0.04), base_color, metallic),
        );
    }
    return pt_apply_iridescence_base_fresnel(
        base, cos_theta, base_color, metallic, material,
    );
}

fn pt_dielectric_transmission(cos_theta: f32, material: PtLayeredMaterial) -> f32 {
    let base = pt_fresnel_f90(
        cos_theta,
        pt_dielectric_f0(material),
        vec3<f32>(clamp(material.specular.w, 0.0, 1.0)),
    );
    let fresnel = pt_apply_iridescence_dielectric_fresnel(
        base, cos_theta, material,
    );
    return clamp(1.0 - max(fresnel.x, max(fresnel.y, fresnel.z)), 0.0, 1.0);
}

fn pt_clearcoat_fresnel(cos_theta: f32, material: PtLayeredMaterial) -> f32 {
    let schlick = 0.04 + 0.96 * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
    return clamp(material.clearcoat_ior.x, 0.0, 1.0) * schlick;
}

fn pt_clearcoat_transmission(cos_theta: f32, material: PtLayeredMaterial) -> f32 {
    return 1.0 - pt_clearcoat_fresnel(cos_theta, material);
}

fn pt_clearcoat_alpha(material: PtLayeredMaterial) -> f32 {
    let perceptual_roughness = max(clamp(material.clearcoat_ior.y, 0.0, 1.0), 0.04);
    return perceptual_roughness * perceptual_roughness;
}

fn pt_layered_default_tangent(n: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(onb(n)[0], 1.0);
}

fn pt_layered_vertex_tangent(slot: u32) -> vec4<f32> {
    let offset = slot * PT_VSTRIDE + 20u;
    return vec4<f32>(
        geo_v[offset],
        geo_v[offset + 1u],
        geo_v[offset + 2u],
        geo_v[offset + 3u],
    );
}

fn pt_layered_hit_tangent(
    geo: vec4<u32>,
    primitive: u32,
    barycentrics: vec2<f32>,
    object_to_world: mat4x3<f32>,
    n: vec3<f32>,
) -> vec4<f32> {
    if (geo.z == 0u) {
        return pt_layered_default_tangent(n);
    }
    let base = geo.y + primitive * 3u;
    let slot0 = geo.x + geo_i[base];
    let slot1 = geo.x + geo_i[base + 1u];
    let slot2 = geo.x + geo_i[base + 2u];
    let weight0 = 1.0 - barycentrics.x - barycentrics.y;
    let tangent_os = weight0 * pt_layered_vertex_tangent(slot0)
        + barycentrics.x * pt_layered_vertex_tangent(slot1)
        + barycentrics.y * pt_layered_vertex_tangent(slot2);
    let tangent_raw = object_to_world * vec4<f32>(tangent_os.xyz, 0.0);
    let tangent_ortho = tangent_raw - n * dot(n, tangent_raw);
    let tangent_length = length(tangent_ortho);
    if (tangent_length <= 1e-4) {
        return pt_layered_default_tangent(n);
    }
    let model_handedness = select(
        -1.0,
        1.0,
        dot(
            cross(object_to_world[0], object_to_world[1]),
            object_to_world[2],
        ) >= 0.0,
    );
    let authored_handedness = select(-1.0, 1.0, tangent_os.w >= 0.0);
    return vec4<f32>(
        tangent_ortho / tangent_length,
        authored_handedness * model_handedness,
    );
}

fn pt_layered_anisotropy_basis(
    n: vec3<f32>,
    tangent: vec4<f32>,
    material: PtLayeredMaterial,
) -> mat3x3<f32> {
    let tangent_ortho = tangent.xyz - n * dot(n, tangent.xyz);
    let tangent_length = length(tangent_ortho);
    if (tangent_length <= 1e-4) {
        return onb(n);
    }
    let mesh_tangent = tangent_ortho / tangent_length;
    let mesh_bitangent = normalize(cross(n, mesh_tangent)) * tangent.w;
    let rotated_raw = mesh_tangent * material.anisotropy.y
        + mesh_bitangent * material.anisotropy.z;
    let rotated_tangent = normalize(
        rotated_raw - n * dot(n, rotated_raw),
    );
    let rotated_bitangent = normalize(cross(n, rotated_tangent));
    return mat3x3<f32>(rotated_tangent, rotated_bitangent, n);
}

fn pt_layered_anisotropy_alpha(
    roughness: f32,
    material: PtLayeredMaterial,
) -> vec2<f32> {
    let alpha = max(roughness * roughness, 1e-3);
    let strength = clamp(material.anisotropy.x, 0.0, 1.0);
    return vec2<f32>(
        alpha + (1.0 - alpha) * strength * strength,
        alpha,
    );
}

fn pt_d_ggx_anisotropic(
    n_dot_h: f32,
    t_dot_h: f32,
    b_dot_h: f32,
    alpha: vec2<f32>,
) -> f32 {
    let product = alpha.x * alpha.y;
    let projected = vec3<f32>(
        alpha.y * t_dot_h,
        alpha.x * b_dot_h,
        product * n_dot_h,
    );
    let weight = product / max(dot(projected, projected), 1e-12);
    return product * weight * weight / 3.14159265;
}

fn pt_v_smith_anisotropic(
    light_local: vec3<f32>,
    view_local: vec3<f32>,
    alpha: vec2<f32>,
) -> f32 {
    let ggx_view = light_local.z * length(vec3<f32>(
        alpha.x * view_local.x,
        alpha.y * view_local.y,
        view_local.z,
    ));
    let ggx_light = view_local.z * length(vec3<f32>(
        alpha.x * light_local.x,
        alpha.y * light_local.y,
        light_local.z,
    ));
    return clamp(0.5 / (ggx_view + ggx_light + 1e-6), 0.0, 1.0);
}

fn pt_smith_g1_anisotropic(direction: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let n_dot = max(direction.z, 0.0);
    if (n_dot <= 0.0) {
        return 0.0;
    }
    let projected = length(vec3<f32>(
        alpha.x * direction.x,
        alpha.y * direction.y,
        n_dot,
    ));
    return 2.0 * n_dot / (n_dot + projected + 1e-6);
}

fn pt_sample_ggx_vndf_anisotropic(
    view: vec3<f32>,
    alpha: vec2<f32>,
    sample: vec2<f32>,
) -> vec3<f32> {
    let stretched_view = normalize(vec3<f32>(
        alpha.x * view.x,
        alpha.y * view.y,
        view.z,
    ));
    let lensq = stretched_view.x * stretched_view.x
        + stretched_view.y * stretched_view.y;
    var tangent = vec3<f32>(1.0, 0.0, 0.0);
    if (lensq > 0.0) {
        tangent = vec3<f32>(
            -stretched_view.y, stretched_view.x, 0.0,
        ) / sqrt(lensq);
    }
    let bitangent = cross(stretched_view, tangent);
    let radius = sqrt(sample.x);
    let phi = 6.2831853 * sample.y;
    let tangent_x = radius * cos(phi);
    var tangent_y = radius * sin(phi);
    let blend = 0.5 * (1.0 + stretched_view.z);
    tangent_y = (1.0 - blend)
        * sqrt(max(0.0, 1.0 - tangent_x * tangent_x))
        + blend * tangent_y;
    let stretched_normal = tangent_x * tangent
        + tangent_y * bitangent
        + sqrt(max(
            0.0, 1.0 - tangent_x * tangent_x - tangent_y * tangent_y,
        )) * stretched_view;
    return normalize(vec3<f32>(
        alpha.x * stretched_normal.x,
        alpha.y * stretched_normal.y,
        max(stretched_normal.z, 0.0),
    ));
}

fn pt_layered_primary_surface(
    p: vec3<f32>,
    n: vec3<f32>,
) -> PtLayeredSurface {
    let to_surface = p - u.cam_pos.xyz;
    let distance = length(to_surface);
    if (distance <= 1e-4) {
        return PtLayeredSurface(pt_layered_default(), vec4<f32>(0.0));
    }
    var query: ray_query;
    rayQueryInitialize(
        &query,
        accel,
        RayDesc(
            0u, 0xFFu, 0.001, distance * 1.02 + 0.1,
            u.cam_pos.xyz, to_surface / distance,
        ),
    );
    if (BLOOM_RAY_QUERY_NEEDS_PROCEED) {
        loop {
            if (!rayQueryProceed(&query)) { break; }
        }
    }
    let hit = rayQueryGetCommittedIntersection(&query);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return PtLayeredSurface(pt_layered_default(), vec4<f32>(0.0));
    }
    let material = pt_layered_materials[hit.instance_custom_data];
    if (!pt_layered_has_anisotropy(material)) {
        return PtLayeredSurface(material, vec4<f32>(0.0));
    }
    let instance = instance_data[hit.instance_custom_data];
    let tangent = pt_layered_hit_tangent(
        instance.geo,
        hit.primitive_index,
        hit.barycentrics,
        hit.object_to_world,
        n,
    );
    return PtLayeredSurface(material, tangent);
}

fn pt_layered_base_nee(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    ldir: vec3<f32>,
    ndl: f32,
    full_alb: vec3<f32>,
    rough: f32,
    metal: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    if (
        !pt_layered_has_specular_ior(material)
            && !pt_layered_has_anisotropy(material)
            && !pt_layered_has_iridescence(material)
    ) {
        return nee_diffuse(n, view, ldir, ndl, full_alb, rough, metal)
            + nee_spec(n, view, ldir, ndl, full_alb, rough, metal);
    }
    let half_raw = view + ldir;
    if (dot(half_raw, half_raw) <= 1e-8) {
        return vec3<f32>(0.0);
    }
    let half = normalize(half_raw);
    let ndv = max(dot(n, view), 1e-4);
    let ndh = max(dot(n, half), 0.0);
    let vdh = max(dot(view, half), 1e-4);
    let alpha = max(rough * rough, 1e-3);
    var distribution: f32;
    var visibility: f32;
    if (pt_layered_has_anisotropy(material)) {
        let basis = pt_layered_anisotropy_basis(n, tangent, material);
        let view_local = vec3<f32>(
            dot(view, basis[0]), dot(view, basis[1]), ndv,
        );
        let light_local = vec3<f32>(
            dot(ldir, basis[0]), dot(ldir, basis[1]), ndl,
        );
        let anisotropic_alpha = pt_layered_anisotropy_alpha(rough, material);
        distribution = pt_d_ggx_anisotropic(
            ndh,
            dot(half, basis[0]),
            dot(half, basis[1]),
            anisotropic_alpha,
        );
        visibility = pt_v_smith_anisotropic(
            light_local, view_local, anisotropic_alpha,
        );
    } else {
        let a2 = alpha * alpha;
        let denominator = ndh * ndh * (a2 - 1.0) + 1.0;
        distribution = a2 / (3.14159265 * denominator * denominator);
        visibility = v_smith(ndv, ndl, alpha);
    }
    let specular = pt_layered_base_fresnel(vdh, full_alb, metal, material)
        * distribution * visibility * ndl;
    let diffuse_albedo = full_alb * (1.0 - metal)
        * pt_dielectric_transmission(ndv, material)
        * pt_dielectric_transmission(ndl, material);
    let diffuse = diffuse_albedo
        * burley_diffuse(ndl, ndv, max(dot(ldir, half), 0.0), rough) * ndl;
    return diffuse + specular;
}

fn pt_layered_nee(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    ldir: vec3<f32>,
    ndl: f32,
    full_alb: vec3<f32>,
    rough: f32,
    metal: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    let undercoat = pt_layered_undercoat_nee(
        n, tangent, view, ldir, ndl, full_alb, rough, metal, material,
    );
    let half_raw = view + ldir;
    if (!pt_layered_has_clearcoat(material) || dot(half_raw, half_raw) <= 1e-8) {
        return undercoat;
    }
    let half = normalize(half_raw);
    let ndv = max(dot(n, view), 1e-4);
    let ndh = max(dot(n, half), 0.0);
    let vdh = max(dot(view, half), 1e-4);
    let alpha = pt_clearcoat_alpha(material);
    let a2 = alpha * alpha;
    let denominator = ndh * ndh * (a2 - 1.0) + 1.0;
    let distribution = a2 / (3.14159265 * denominator * denominator);
    let clearcoat = pt_clearcoat_fresnel(vdh, material)
        * distribution * v_smith(ndv, ndl, alpha) * ndl;
    let attenuation = pt_clearcoat_transmission(ndv, material)
        * pt_clearcoat_transmission(ndl, material);
    return undercoat * attenuation + vec3<f32>(clearcoat);
}

fn pt_layered_direct_light(
    p: vec3<f32>,
    n: vec3<f32>,
    tangent: vec4<f32>,
    sun_r2: vec2<f32>,
    view: vec3<f32>,
    full_alb: vec3<f32>,
    rough: f32,
    metal: f32,
    with_points: bool,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    if (!pt_layered_has_transport(material)) {
        return direct_light(p, n, sun_r2, view, full_alb, rough, metal, with_points);
    }
    var result = vec3<f32>(0.0);
    let sun_ndl = max(dot(n, u.sun_dir.xyz), 0.0);
    if (sun_ndl > 0.0) {
        let visibility = sun_visibility(p, n, sun_r2);
        if (visibility > 0.0) {
            result += pt_layered_nee(
                n, tangent, view, u.sun_dir.xyz, sun_ndl,
                full_alb, rough, metal, material,
            ) * u.sun_color.rgb * visibility;
        }
    }
    let count = u32(u.cfg.z);
    if (count > 0u && with_points) {
        let pick = min(u32(rand_f() * f32(count)), count - 1u);
        let light = u.lights[pick];
        let to_light = light.pos_range.xyz - p;
        let distance = length(to_light);
        let range = light.pos_range.w;
        if (distance < range && distance > 1e-3) {
            let direction = to_light / distance;
            let ndl = dot(n, direction);
            if (ndl > 0.0 && !occluded(p, direction, distance - 0.02)) {
                let falloff = 1.0 - distance / range;
                let incident = light.color_int.rgb * light.color_int.w
                    * falloff * falloff * f32(count);
                result += pt_layered_nee(
                    n, tangent, view, direction, ndl,
                    full_alb, rough, metal, material,
                ) * incident;
            }
        }
    }
    return result;
}

fn pt_sample_layered_base(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    material: PtLayeredMaterial,
) -> BrdfSample {
    if (
        !pt_layered_has_specular_ior(material)
            && !pt_layered_has_anisotropy(material)
            && !pt_layered_has_iridescence(material)
    ) {
        return sample_brdf(n, view, base_color, roughness, metallic);
    }
    var out: BrdfSample;
    out.valid = false;
    let alpha = max(roughness * roughness, 1e-3);
    let anisotropic = pt_layered_has_anisotropy(material);
    var specular_basis = onb(n);
    if (anisotropic) {
        specular_basis = pt_layered_anisotropy_basis(n, tangent, material);
    }
    let view_specular = vec3<f32>(
        dot(view, specular_basis[0]),
        dot(view, specular_basis[1]),
        dot(view, n),
    );
    if (view_specular.z <= 0.0) {
        return out;
    }
    let n_dot_v = max(view_specular.z, 1e-4);
    let fresnel_view = pt_layered_base_fresnel(
        n_dot_v, base_color, metallic, material,
    );
    let specular_weight = (
        fresnel_view.x + fresnel_view.y + fresnel_view.z
    ) / 3.0;
    let diffuse_weight = pt_dielectric_transmission(n_dot_v, material)
        * (1.0 - metallic);
    var specular_probability = specular_weight
        / (specular_weight + diffuse_weight + 1e-6);
    if (specular_weight > 0.0 && diffuse_weight > 0.0) {
        specular_probability = clamp(specular_probability, 0.05, 0.95);
    }
    let sample = rand_2f();
    if (rand_f() < specular_probability) {
        let anisotropic_alpha = pt_layered_anisotropy_alpha(
            roughness, material,
        );
        var half_specular: vec3<f32>;
        if (anisotropic) {
            half_specular = pt_sample_ggx_vndf_anisotropic(
                view_specular, anisotropic_alpha, sample,
            );
        } else {
            half_specular = sample_ggx_vndf(view_specular, alpha, sample);
        }
        let light_specular = reflect(-view_specular, half_specular);
        if (light_specular.z <= 0.0) {
            return out;
        }
        let n_dot_l = light_specular.z;
        let v_dot_h = max(dot(view_specular, half_specular), 1e-4);
        var visibility: f32;
        var g1_view: f32;
        if (anisotropic) {
            visibility = pt_v_smith_anisotropic(
                light_specular, view_specular, anisotropic_alpha,
            );
            g1_view = pt_smith_g1_anisotropic(
                view_specular, anisotropic_alpha,
            );
        } else {
            visibility = v_smith(n_dot_v, n_dot_l, alpha);
            g1_view = smith_g1(n_dot_v, alpha);
        }
        let g2 = visibility * 4.0 * n_dot_v * n_dot_l;
        out.dir = specular_basis * light_specular;
        out.weight = pt_layered_base_fresnel(
            v_dot_h, base_color, metallic, material,
        ) * g2 / max(g1_view * specular_probability, 1e-6);
        if (u.cfg.x >= 2.0) {
            out.weight = min(out.weight, vec3<f32>(4.0));
        }
        out.valid = true;
        return out;
    }
    let diffuse_basis = onb(n);
    let view_diffuse = vec3<f32>(
        dot(view, diffuse_basis[0]), dot(view, diffuse_basis[1]), dot(view, n),
    );
    let radius = sqrt(sample.x);
    let phi = 6.2831853 * sample.y;
    let light_diffuse = vec3<f32>(
        radius * cos(phi),
        radius * sin(phi),
        sqrt(max(0.0, 1.0 - sample.x)),
    );
    let n_dot_l = max(light_diffuse.z, 1e-4);
    let half_raw = view_diffuse + light_diffuse;
    var l_dot_h = 0.0;
    if (dot(half_raw, half_raw) > 1e-8) {
        l_dot_h = max(dot(light_diffuse, normalize(half_raw)), 0.0);
    }
    let diffuse_albedo = base_color * (1.0 - metallic)
        * pt_dielectric_transmission(n_dot_v, material)
        * pt_dielectric_transmission(n_dot_l, material);
    out.dir = diffuse_basis * light_diffuse;
    out.weight = diffuse_albedo
        * burley_diffuse(n_dot_l, n_dot_v, l_dot_h, roughness)
        * 3.14159265 / max(1.0 - specular_probability, 1e-6);
    if (u.cfg.x >= 2.0) {
        out.weight = min(out.weight, vec3<f32>(4.0));
    }
    out.valid = true;
    return out;
}

fn pt_sample_layered_brdf(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    material: PtLayeredMaterial,
) -> BrdfSample {
    if (!pt_layered_has_transport(material)) {
        return sample_brdf(n, view, base_color, roughness, metallic);
    }
    if (!pt_layered_has_clearcoat(material)) {
        return pt_sample_layered_undercoat(
            n, tangent, view, base_color, roughness, metallic, material,
        );
    }
    var out: BrdfSample;
    out.valid = false;
    let ndv = max(dot(n, view), 0.0);
    if (ndv <= 0.0) {
        return out;
    }
    let base_f = pt_layered_base_fresnel(ndv, base_color, metallic, material);
    let base_specular_weight = (base_f.x + base_f.y + base_f.z) / 3.0;
    var diffuse_weight = (1.0 - base_specular_weight) * (1.0 - metallic);
    if (
        pt_layered_has_specular_ior(material)
            || pt_layered_has_anisotropy(material)
            || pt_layered_has_iridescence(material)
    ) {
        diffuse_weight = pt_dielectric_transmission(ndv, material) * (1.0 - metallic);
    }
    let sheen_weight = pt_layered_sheen_weight(material);
    let clearcoat_weight = pt_clearcoat_fresnel(ndv, material);
    let clearcoat_probability = clearcoat_weight
        / (
            base_specular_weight + diffuse_weight + sheen_weight
                + clearcoat_weight + 1e-6
        );

    if (rand_f() < clearcoat_probability) {
        let basis = onb(n);
        let view_tangent = vec3<f32>(
            dot(view, basis[0]), dot(view, basis[1]), dot(view, n),
        );
        let alpha = pt_clearcoat_alpha(material);
        let half_tangent = sample_ggx_vndf(view_tangent, alpha, rand_2f());
        let light_tangent = reflect(-view_tangent, half_tangent);
        if (light_tangent.z <= 0.0) {
            return out;
        }
        let n_dot_l = light_tangent.z;
        let n_dot_v = max(view_tangent.z, 1e-4);
        let v_dot_h = max(dot(view_tangent, half_tangent), 1e-4);
        let g2 = v_smith(n_dot_v, n_dot_l, alpha)
            * 4.0 * n_dot_v * n_dot_l;
        let g1_view = smith_g1(n_dot_v, alpha);
        out.dir = basis * light_tangent;
        out.weight = vec3<f32>(
            pt_clearcoat_fresnel(v_dot_h, material) * g2
                / max(g1_view * clearcoat_probability, 1e-6),
        );
        if (u.cfg.x >= 2.0) {
            out.weight = min(out.weight, vec3<f32>(4.0));
        }
        out.valid = true;
        return out;
    }

    out = pt_sample_layered_undercoat(
        n, tangent, view, base_color, roughness, metallic, material,
    );
    if (!out.valid) {
        return out;
    }
    let n_dot_l = max(dot(n, out.dir), 0.0);
    let attenuation = pt_clearcoat_transmission(ndv, material)
        * pt_clearcoat_transmission(n_dot_l, material);
    out.weight *= attenuation / max(1.0 - clearcoat_probability, 1e-6);
    if (u.cfg.x >= 2.0) {
        out.weight = min(out.weight, vec3<f32>(4.0));
    }
    return out;
}
"#;

const PT_LAYERED_IRIDESCENCE_DISABLED_WGSL: &str = r#"
fn pt_layered_has_iridescence(material: PtLayeredMaterial) -> bool {
    return false;
}

fn pt_apply_iridescence_base_fresnel(
    base: vec3<f32>,
    cos_theta: f32,
    base_color: vec3<f32>,
    metallic: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    return base;
}

fn pt_apply_iridescence_dielectric_fresnel(
    base: vec3<f32>,
    cos_theta: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    return base;
}
"#;

const PT_LAYERED_IRIDESCENCE_WGSL: &str = r#"
const PT_LAYERED_IRIDESCENCE_LOBE: u32 = 8u;
const PT_IRIDESCENCE_PI: f32 = 3.14159265;

fn pt_layered_has_iridescence(material: PtLayeredMaterial) -> bool {
    return material.header.x == 1u
        && (material.header.y & PT_LAYERED_IRIDESCENCE_LOBE) != 0u
        && (material.header.z & PT_LAYERED_IRIDESCENCE_LOBE) == 0u
        && material.iridescence.x > 0.0
        && material.iridescence.w > 0.0;
}

fn pt_fresnel0_to_ior(f0: vec3<f32>) -> vec3<f32> {
    let root = sqrt(clamp(f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
    return (vec3<f32>(1.0) + root) / (vec3<f32>(1.0) - root);
}

fn pt_ior_to_fresnel0(
    transmitted_ior: vec3<f32>,
    incident_ior: f32,
) -> vec3<f32> {
    let incident = vec3<f32>(incident_ior);
    let ratio = (transmitted_ior - incident) / (transmitted_ior + incident);
    return ratio * ratio;
}

fn pt_iridescence_sensitivity(
    optical_path_difference_nm: f32,
    shift: vec3<f32>,
) -> vec3<f32> {
    let phase = 2.0 * PT_IRIDESCENCE_PI
        * optical_path_difference_nm * 1e-9;
    let phase_squared = phase * phase;
    let value = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let position = vec3<f32>(1.6810e6, 1.7953e6, 2.2084e6);
    let variance = vec3<f32>(4.3278e9, 9.3046e9, 6.6121e9);
    var xyz = value
        * sqrt(vec3<f32>(2.0 * PT_IRIDESCENCE_PI) * variance)
        * cos(position * phase + shift)
        * exp(-vec3<f32>(phase_squared) * variance);
    xyz.x += 9.7470e-14
        * sqrt(2.0 * PT_IRIDESCENCE_PI * 4.5282e9)
        * cos(2.2399e6 * phase + shift.x)
        * exp(-4.5282e9 * phase_squared);
    xyz /= 1.0685e-7;
    return vec3<f32>(
        3.2404542 * xyz.x - 0.9692660 * xyz.y + 0.0556434 * xyz.z,
        -1.5371385 * xyz.x + 1.8760108 * xyz.y - 0.2040259 * xyz.z,
        -0.4985314 * xyz.x + 0.0415560 * xyz.y + 1.0572252 * xyz.z,
    );
}

fn pt_eval_iridescence(
    outside_ior: f32,
    authored_film_ior: f32,
    cos_theta_1: f32,
    authored_thickness_nm: f32,
    base_f0: vec3<f32>,
) -> vec3<f32> {
    let safe_outside_ior = max(outside_ior, 1e-4);
    let thickness_nm = max(authored_thickness_nm, 0.0);
    let film_ior = mix(
        safe_outside_ior,
        max(authored_film_ior, 1.0),
        smoothstep(0.0, 0.03, thickness_nm),
    );
    let cosine_1 = clamp(cos_theta_1, 0.0, 1.0);
    let sin_theta_2_squared = pow(
        safe_outside_ior / film_ior, 2.0,
    ) * (1.0 - cosine_1 * cosine_1);
    let cos_theta_2_squared = 1.0 - sin_theta_2_squared;
    if (cos_theta_2_squared < 0.0) {
        return vec3<f32>(1.0);
    }
    let cosine_2 = sqrt(cos_theta_2_squared);

    let r0 = pt_ior_f0(film_ior / safe_outside_ior);
    let r12 = pt_fresnel_f90(
        cosine_1, vec3<f32>(r0), vec3<f32>(1.0),
    ).x;
    let t121 = 1.0 - r12;
    let phi12 = select(
        0.0, PT_IRIDESCENCE_PI, film_ior < safe_outside_ior,
    );
    let phi21 = PT_IRIDESCENCE_PI - phi12;

    let base_ior = pt_fresnel0_to_ior(base_f0);
    let r1 = pt_ior_to_fresnel0(base_ior, film_ior);
    let r23 = pt_fresnel_f90(cosine_2, r1, vec3<f32>(1.0));
    let phi23 = vec3<f32>(
        select(0.0, PT_IRIDESCENCE_PI, base_ior.x < film_ior),
        select(0.0, PT_IRIDESCENCE_PI, base_ior.y < film_ior),
        select(0.0, PT_IRIDESCENCE_PI, base_ior.z < film_ior),
    );
    let optical_path_difference =
        2.0 * film_ior * thickness_nm * cosine_2;
    let phase_shift = vec3<f32>(phi21) + phi23;
    let r123 = clamp(
        vec3<f32>(r12) * r23,
        vec3<f32>(1e-5),
        vec3<f32>(0.9999),
    );
    let reflected_series = vec3<f32>(t121 * t121)
        * r23 / (vec3<f32>(1.0) - r123);
    var result = vec3<f32>(r12) + reflected_series;
    var coefficient = reflected_series - vec3<f32>(t121);
    let amplitude = sqrt(r123);
    for (var order = 1u; order <= 2u; order += 1u) {
        coefficient *= amplitude;
        result += coefficient * 2.0 * pt_iridescence_sensitivity(
            f32(order) * optical_path_difference,
            f32(order) * phase_shift,
        );
    }
    return clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn pt_apply_iridescence_base_fresnel(
    base: vec3<f32>,
    cos_theta: f32,
    base_color: vec3<f32>,
    metallic: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    if (!pt_layered_has_iridescence(material)) {
        return base;
    }
    let dielectric = pt_eval_iridescence(
        1.0,
        material.iridescence.y,
        cos_theta,
        material.iridescence.w,
        pt_dielectric_f0(material),
    );
    let conductor = pt_eval_iridescence(
        1.0,
        material.iridescence.y,
        cos_theta,
        material.iridescence.w,
        base_color,
    );
    let thin_film = mix(dielectric, conductor, metallic);
    return mix(
        base, thin_film, clamp(material.iridescence.x, 0.0, 1.0),
    );
}

fn pt_apply_iridescence_dielectric_fresnel(
    base: vec3<f32>,
    cos_theta: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    if (!pt_layered_has_iridescence(material)) {
        return base;
    }
    let thin_film = pt_eval_iridescence(
        1.0,
        material.iridescence.y,
        cos_theta,
        material.iridescence.w,
        pt_dielectric_f0(material),
    );
    return mix(
        base, thin_film, clamp(material.iridescence.x, 0.0, 1.0),
    );
}
"#;

const PT_LAYERED_SHEEN_DISABLED_WGSL: &str = r#"
fn pt_layered_has_sheen(material: PtLayeredMaterial) -> bool {
    return false;
}

fn pt_layered_sheen_weight(material: PtLayeredMaterial) -> f32 {
    return 0.0;
}

fn pt_layered_undercoat_nee(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    ldir: vec3<f32>,
    ndl: f32,
    full_alb: vec3<f32>,
    rough: f32,
    metal: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    return pt_layered_base_nee(
        n, tangent, view, ldir, ndl, full_alb, rough, metal, material,
    );
}

fn pt_sample_layered_undercoat(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    material: PtLayeredMaterial,
) -> BrdfSample {
    return pt_sample_layered_base(
        n, tangent, view, base_color, roughness, metallic, material,
    );
}
"#;

const PT_LAYERED_SHEEN_WGSL: &str = r#"
@group(2) @binding(1)
var pt_sheen_albedo_tex: texture_2d<f32>;

fn pt_layered_has_sheen(material: PtLayeredMaterial) -> bool {
    return material.header.x == 1u
        && (material.header.y & 2u) != 0u
        && (material.header.z & 2u) == 0u
        && max(material.sheen.x, max(material.sheen.y, material.sheen.z)) > 0.0;
}

fn pt_layered_sheen_weight(material: PtLayeredMaterial) -> f32 {
    if (!pt_layered_has_sheen(material)) {
        return 0.0;
    }
    return (material.sheen.x + material.sheen.y + material.sheen.z) / 3.0;
}

fn pt_sheen_roughness(material: PtLayeredMaterial) -> f32 {
    return max(clamp(material.sheen.w, 0.0, 1.0), 1e-3);
}

fn pt_sheen_lambda_helper(x: f32, alpha_g: f32) -> f32 {
    let one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);
    let a = mix(21.5473, 25.3245, one_minus_alpha_sq);
    let b = mix(3.82987, 3.32435, one_minus_alpha_sq);
    let c = mix(0.19823, 0.16801, one_minus_alpha_sq);
    let d = mix(-1.97760, -1.27393, one_minus_alpha_sq);
    let e = mix(-4.32054, -4.85967, one_minus_alpha_sq);
    return a / (1.0 + b * pow(max(x, 0.0), c)) + d * x + e;
}

fn pt_sheen_lambda(cos_theta: f32, alpha_g: f32) -> f32 {
    let cosine = clamp(abs(cos_theta), 0.0, 1.0);
    if (cosine < 0.5) {
        return exp(pt_sheen_lambda_helper(cosine, alpha_g));
    }
    return exp(
        2.0 * pt_sheen_lambda_helper(0.5, alpha_g)
            - pt_sheen_lambda_helper(1.0 - cosine, alpha_g),
    );
}

fn pt_sheen_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha_g = max(roughness * roughness, 1e-6);
    let inverse_alpha = 1.0 / alpha_g;
    let sin2_h = max(1.0 - n_dot_h * n_dot_h, 0.0);
    return (2.0 + inverse_alpha) * pow(sin2_h, 0.5 * inverse_alpha)
        / 6.2831853;
}

fn pt_sheen_visibility(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let alpha_g = max(roughness * roughness, 1e-6);
    let denominator = (
        1.0 + pt_sheen_lambda(n_dot_v, alpha_g)
            + pt_sheen_lambda(n_dot_l, alpha_g)
    ) * (4.0 * n_dot_v * n_dot_l);
    return 1.0 / max(denominator, 1e-6);
}

fn pt_sheen_directional_albedo(n_dot: f32, roughness: f32) -> f32 {
    return textureSampleLevel(
        pt_sheen_albedo_tex,
        card_samp,
        vec2<f32>(clamp(n_dot, 0.0, 1.0), clamp(roughness, 0.0, 1.0)),
        0.0,
    ).r;
}

fn pt_sheen_scale(
    material: PtLayeredMaterial,
    n_dot_v: f32,
    n_dot_l: f32,
) -> f32 {
    let maximum_color = max(
        material.sheen.x, max(material.sheen.y, material.sheen.z),
    );
    let view_albedo = pt_sheen_directional_albedo(
        n_dot_v, pt_sheen_roughness(material),
    );
    let light_albedo = pt_sheen_directional_albedo(
        n_dot_l, pt_sheen_roughness(material),
    );
    return clamp(
        1.0 - maximum_color * max(view_albedo, light_albedo), 0.0, 1.0,
    );
}

fn pt_layered_undercoat_nee(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    ldir: vec3<f32>,
    ndl: f32,
    full_alb: vec3<f32>,
    rough: f32,
    metal: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    let base = pt_layered_base_nee(
        n, tangent, view, ldir, ndl, full_alb, rough, metal, material,
    );
    if (!pt_layered_has_sheen(material)) {
        return base;
    }
    let half_raw = view + ldir;
    if (dot(half_raw, half_raw) <= 1e-8) {
        return base;
    }
    let half = normalize(half_raw);
    let n_dot_v = max(dot(n, view), 1e-4);
    let n_dot_h = max(dot(n, half), 0.0);
    let roughness = pt_sheen_roughness(material);
    let sheen = material.sheen.xyz
        * pt_sheen_distribution(n_dot_h, roughness)
        * pt_sheen_visibility(ndl, n_dot_v, roughness) * ndl;
    return base * pt_sheen_scale(material, n_dot_v, ndl) + sheen;
}

fn pt_sample_charlie_half(
    perceptual_roughness: f32,
    sample: vec2<f32>,
) -> vec3<f32> {
    let alpha = max(perceptual_roughness, 1e-3);
    let alpha_g = alpha * alpha;
    let sin_theta = pow(sample.x, alpha_g / (2.0 * alpha_g + 1.0));
    let cos_theta = sqrt(max(0.0, 1.0 - sin_theta * sin_theta));
    let phi = 6.2831853 * sample.y;
    return vec3<f32>(
        sin_theta * cos(phi), sin_theta * sin(phi), cos_theta,
    );
}

fn pt_sample_layered_undercoat(
    n: vec3<f32>,
    tangent: vec4<f32>,
    view: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    material: PtLayeredMaterial,
) -> BrdfSample {
    if (!pt_layered_has_sheen(material)) {
        return pt_sample_layered_base(
            n, tangent, view, base_color, roughness, metallic, material,
        );
    }
    var out: BrdfSample;
    out.valid = false;
    let n_dot_v = max(dot(n, view), 0.0);
    if (n_dot_v <= 0.0) {
        return out;
    }
    let base_f = pt_layered_base_fresnel(
        n_dot_v, base_color, metallic, material,
    );
    let base_specular_weight = (base_f.x + base_f.y + base_f.z) / 3.0;
    var diffuse_weight = (1.0 - base_specular_weight) * (1.0 - metallic);
    if (
        pt_layered_has_specular_ior(material)
            || pt_layered_has_anisotropy(material)
            || pt_layered_has_iridescence(material)
    ) {
        diffuse_weight = pt_dielectric_transmission(n_dot_v, material)
            * (1.0 - metallic);
    }
    let sheen_weight = pt_layered_sheen_weight(material);
    let sheen_probability = sheen_weight
        / (base_specular_weight + diffuse_weight + sheen_weight + 1e-6);
    if (rand_f() < sheen_probability) {
        let basis = onb(n);
        let view_tangent = vec3<f32>(
            dot(view, basis[0]), dot(view, basis[1]), dot(view, n),
        );
        let half_tangent = pt_sample_charlie_half(
            pt_sheen_roughness(material), rand_2f(),
        );
        let v_dot_h = max(dot(view_tangent, half_tangent), 0.0);
        if (v_dot_h <= 0.0 || half_tangent.z <= 0.0) {
            return out;
        }
        let light_tangent = reflect(-view_tangent, half_tangent);
        if (light_tangent.z <= 0.0) {
            return out;
        }
        let visibility = pt_sheen_visibility(
            light_tangent.z, n_dot_v, pt_sheen_roughness(material),
        );
        out.dir = basis * light_tangent;
        out.weight = material.sheen.xyz * visibility * light_tangent.z
            * 4.0 * v_dot_h
            / max(half_tangent.z * sheen_probability, 1e-6);
        if (u.cfg.x >= 2.0) {
            out.weight = min(out.weight, vec3<f32>(4.0));
        }
        out.valid = true;
        return out;
    }
    out = pt_sample_layered_base(
        n, tangent, view, base_color, roughness, metallic, material,
    );
    if (!out.valid) {
        return out;
    }
    let n_dot_l = max(dot(n, out.dir), 0.0);
    out.weight *= pt_sheen_scale(material, n_dot_v, n_dot_l)
        / max(1.0 - sheen_probability, 1e-6);
    if (u.cfg.x >= 2.0) {
        out.weight = min(out.weight, vec3<f32>(4.0));
    }
    return out;
}
"#;

fn replace_once(source: &mut String, needle: &str, replacement: &str) {
    let count = source.matches(needle).count();
    assert_eq!(
        count, 1,
        "layered PT specialization expected one source anchor, found {count}: {needle}"
    );
    *source = source.replacen(needle, replacement, 1);
}

fn layered_kernel_variant(base: &str) -> String {
    let mut source = base.to_owned();
    replace_once(
        &mut source,
        "    var rough_cur = mr0.g;",
        "    var rough_cur = mr0.g;\n\
         \x20   let layered_primary = pt_layered_primary_surface(p0, n0);\n\
         \x20   var layered_cur = layered_primary.material;\n\
         \x20   var layered_tangent_cur = layered_primary.tangent;",
    );
    replace_once(
        &mut source,
        "    let use_restir = u.ext.w == 1u && u.cfg.x >= 2.0;",
        "    let use_restir = u.ext.w == 1u && u.cfg.x >= 2.0\n        \
         && !pt_layered_has_transport(layered_cur);",
    );
    replace_once(
        &mut source,
        "    var radiance = direct_light(\n\
         \x20       p0 + n0 * 0.02, n0, sun_r2, view_cur,\n\
         \x20       albedo0, rough_cur, metal_cur, !use_restir,\n\
         \x20   );",
        "    var radiance = pt_layered_direct_light(\n\
         \x20       p0 + n0 * 0.02, n0, layered_tangent_cur, sun_r2, view_cur,\n\
         \x20       albedo0, rough_cur, metal_cur, !use_restir, layered_cur,\n\
         \x20   );",
    );
    replace_once(
        &mut source,
        "        let s = sample_brdf(n_cur, view_cur, alb_cur, rough_cur, metal_cur);",
        "        let s = pt_sample_layered_brdf(\n\
         \x20           n_cur, layered_tangent_cur, view_cur,\n\
         \x20           alb_cur, rough_cur, metal_cur, layered_cur,\n\
         \x20       );",
    );
    replace_once(
        &mut source,
        "        radiance += throughput * direct_light(\n\
         \x20           hit_p, n_hit, rand_2f(), -dir,\n\
         \x20           alb_hit, inst.mat_params.x, inst.mat_params.y, true,\n\
         \x20       );",
        "        let layered_hit = pt_layered_materials[hit.instance_custom_data];\n\
         \x20       var layered_tangent_hit = vec4<f32>(0.0);\n\
         \x20       if (pt_layered_has_anisotropy(layered_hit)) {\n\
         \x20           layered_tangent_hit = pt_layered_hit_tangent(\n\
         \x20               inst.geo, hit.primitive_index, hit.barycentrics,\n\
         \x20               hit.object_to_world, n_hit,\n\
         \x20           );\n\
         \x20       }\n\
         \x20       radiance += throughput * pt_layered_direct_light(\n\
         \x20           hit_p, n_hit, layered_tangent_hit, rand_2f(), -dir,\n\
         \x20           alb_hit, inst.mat_params.x, inst.mat_params.y, true, layered_hit,\n\
         \x20       );",
    );
    replace_once(
        &mut source,
        "        metal_cur = inst.mat_params.y;\n        view_cur = -dir;",
        "        metal_cur = inst.mat_params.y;\n\
         \x20       layered_cur = layered_hit;\n\
         \x20       layered_tangent_cur = layered_tangent_hit;\n\
         \x20       view_cur = -dir;",
    );
    source
}

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
    pub(super) fn pt_layered_transport_active(&self) -> bool {
        self.pt_layered_records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_qualified_transport)
    }

    pub(super) fn pt_layered_sheen_active(&self) -> bool {
        self.pt_layered_records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_sheen)
    }

    pub(super) fn pt_layered_anisotropy_active(&self) -> bool {
        self.pt_layered_records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_anisotropy)
    }

    pub(super) fn pt_layered_iridescence_active(&self) -> bool {
        self.pt_layered_records
            .iter()
            .copied()
            .any(PtLayeredMaterialCpu::has_iridescence)
    }

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
        let sheen = self.pt_layered_sheen_active();
        let anisotropy = self.pt_layered_anisotropy_active();
        let iridescence = self.pt_layered_iridescence_active();
        let resource_variant = sheen as usize;
        let pipeline_variant =
            resource_variant | ((anisotropy as usize) << 1) | ((iridescence as usize) << 2);
        if sheen {
            self.ensure_scene_sheen_albedo_lut();
        }
        if self.pt_layered_layouts[resource_variant].is_none() {
            let mut entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(PT_LAYERED_RECORD_BYTES),
                },
                count: None,
            }];
            if sheen {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
            }
            self.pt_layered_layouts[resource_variant] = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some(if sheen {
                        "pt_layered_sheen_layout"
                    } else {
                        "pt_layered_layout"
                    }),
                    entries: &entries,
                },
            ));
        }
        if self.pt_layered_pipelines[pipeline_variant].is_none() {
            let query_diagnostics = std::env::var("BLOOM_GOLDEN_DIAGNOSTICS")
                .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
                || std::env::var("BLOOM_PT_DEBUG")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .is_some_and(|view| (6..=19).contains(&view));
            let fault = std::env::var("BLOOM_PT_TEST_FAULT").ok();
            let base_kernel = pt_kernel_variant(query_diagnostics);
            let layered_kernel = layered_kernel_variant(base_kernel.as_ref());
            let source = format!(
                "enable wgpu_ray_query;\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                ray_query_backend_variant(&self.device),
                pt_fault_constants(fault.as_deref()),
                layered_kernel,
                texture_variant(self.pt_texture_arrays_enabled),
                PT_LAYERED_BINDINGS_WGSL,
                if anisotropy {
                    "const PT_HAS_SCALAR_ANISOTROPY: bool = true;"
                } else {
                    "const PT_HAS_SCALAR_ANISOTROPY: bool = false;"
                },
                PT_LAYERED_TRANSPORT_WGSL,
                if iridescence {
                    PT_LAYERED_IRIDESCENCE_WGSL
                } else {
                    PT_LAYERED_IRIDESCENCE_DISABLED_WGSL
                },
                if sheen {
                    PT_LAYERED_SHEEN_WGSL
                } else {
                    PT_LAYERED_SHEEN_DISABLED_WGSL
                },
            );
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(if sheen {
                        "pt_layered_sheen_shader"
                    } else {
                        "pt_layered_shader"
                    }),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let groups = [
                self.pt_layout.as_ref(),
                self.pt_tex_layout.as_ref(),
                self.pt_layered_layouts[resource_variant].as_ref(),
            ];
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(if sheen {
                            "pt_layered_sheen_pipeline_layout"
                        } else {
                            "pt_layered_pipeline_layout"
                        }),
                        bind_group_layouts: &groups,
                        immediate_size: 0,
                    });
            self.pt_layered_pipelines[pipeline_variant] = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(if iridescence {
                            "pt_layered_iridescence_pipeline"
                        } else if anisotropy {
                            "pt_layered_anisotropy_pipeline"
                        } else if sheen {
                            "pt_layered_sheen_pipeline"
                        } else {
                            "pt_layered_pipeline"
                        }),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some("cs_main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );
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
            self.pt_layered_bgs = [None, None];
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
        if self.pt_layered_bgs[resource_variant].is_none() {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: self
                    .pt_layered_instance_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            }];
            if sheen {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &self.scene_sheen_albedo_lut.as_ref().unwrap().view,
                    ),
                });
            }
            self.pt_layered_bgs[resource_variant] =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(if sheen {
                        "pt_layered_sheen_bg"
                    } else {
                        "pt_layered_bg"
                    }),
                    layout: self.pt_layered_layouts[resource_variant].as_ref().unwrap(),
                    entries: &entries,
                }));
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
        assert!(record.has_iridescence() && record.has_qualified_transport());
    }

    #[test]
    fn only_qualified_lobes_select_layered_transport() {
        let sheen = crate::models::MaterialLayeredPbr::from_authoring_factors(
            crate::models::MaterialLayeredPbr::SHEEN_LOBE,
            0.0,
            0.0,
            1.0,
            1.0,
            [1.0; 3],
            1.5,
            [0.3, 0.1, 0.05],
            0.4,
            0.0,
            0.0,
            0.0,
            1.3,
            100.0,
            400.0,
        );
        let clearcoat = crate::models::MaterialLayeredPbr::from_authoring_factors(
            crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE,
            0.8,
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
        let specular = crate::models::MaterialLayeredPbr::from_authoring_factors(
            crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE,
            0.0,
            0.0,
            1.0,
            0.7,
            [0.8, 0.6, 0.4],
            1.8,
            [0.0; 3],
            0.0,
            0.0,
            0.0,
            0.0,
            1.3,
            100.0,
            400.0,
        );
        let anisotropy = crate::models::MaterialLayeredPbr::from_authoring_factors(
            crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE,
            0.0,
            0.0,
            1.0,
            1.0,
            [1.0; 3],
            1.5,
            [0.0; 3],
            0.0,
            0.6,
            0.3,
            0.0,
            1.3,
            100.0,
            400.0,
        );
        let texture = crate::models::MaterialTextureBinding {
            source_texture_index: 1,
            source_image_index: 2,
            runtime_texture_idx: Some(3),
            transform: Default::default(),
        };
        let textured = crate::models::MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 0.8,
            clearcoat_texture: Some(texture),
            specular_authored: true,
            specular_factor: 0.7,
            specular_texture: Some(texture),
            sheen_authored: true,
            sheen_color_factor: [0.3, 0.1, 0.05],
            sheen_color_texture: Some(texture),
            anisotropy_authored: true,
            anisotropy_strength: 0.6,
            anisotropy_texture: Some(texture),
            iridescence_authored: true,
            iridescence_factor: 0.75,
            iridescence_texture: Some(texture),
            ..Default::default()
        };
        let sheen = PtLayeredMaterialCpu::from_material(sheen);
        let clearcoat = PtLayeredMaterialCpu::from_material(clearcoat);
        let specular = PtLayeredMaterialCpu::from_material(specular);
        let anisotropy = PtLayeredMaterialCpu::from_material(anisotropy);
        let textured = PtLayeredMaterialCpu::from_material(textured);
        assert!(sheen.has_sheen() && sheen.has_qualified_transport());
        assert!(clearcoat.has_clearcoat() && clearcoat.has_qualified_transport());
        assert!(specular.has_specular_ior() && specular.has_qualified_transport());
        assert!(anisotropy.has_anisotropy() && anisotropy.has_qualified_transport());
        assert!(textured.active() && !textured.has_qualified_transport());
        assert_eq!(
            textured.header[2],
            crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE
                | crate::models::MaterialLayeredPbr::SPECULAR_IOR_LOBE
                | crate::models::MaterialLayeredPbr::SHEEN_LOBE
                | crate::models::MaterialLayeredPbr::ANISOTROPY_LOBE
                | crate::models::MaterialLayeredPbr::IRIDESCENCE_LOBE
        );
    }

    #[test]
    fn specialization_uses_separate_group_without_touching_base_kernel() {
        assert!(PT_LAYERED_BINDINGS_WGSL.contains("@group(2) @binding(0)"));
        assert!(!PT_LAYERED_SHEEN_DISABLED_WGSL.contains("@binding(1)"));
        assert!(PT_LAYERED_SHEEN_WGSL.contains("@group(2) @binding(1)"));
        assert!(PT_LAYERED_TRANSPORT_WGSL.contains("PT_HAS_SCALAR_ANISOTROPY"));
        assert!(PT_LAYERED_TRANSPORT_WGSL.contains("slot * PT_VSTRIDE + 20u"));
        assert!(PT_LAYERED_TRANSPORT_WGSL.contains("hit.object_to_world"));
        assert!(PT_LAYERED_IRIDESCENCE_WGSL.contains("pt_eval_iridescence"));
        assert!(!PT_LAYERED_IRIDESCENCE_DISABLED_WGSL.contains("exp("));
        assert!(!pt_kernel_variant(false).contains("pt_layered_materials"));
    }

    #[test]
    fn layered_specialization_rewrites_every_transport_vertex() {
        for diagnostics in [false, true] {
            let base = pt_kernel_variant(diagnostics);
            let specialized = layered_kernel_variant(base.as_ref());
            assert!(
                specialized.contains("let layered_primary = pt_layered_primary_surface(p0, n0);")
            );
            assert_eq!(specialized.matches("pt_sample_layered_brdf(").count(), 1);
            assert_eq!(
                specialized
                    .matches("throughput * pt_layered_direct_light(")
                    .count(),
                1
            );
            assert!(specialized.contains("layered_cur = layered_hit;"));
            assert!(specialized.contains("layered_tangent_cur = layered_tangent_hit;"));
            assert_eq!(base.matches("pt_layered_").count(), 0);
        }
    }
}
