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

    fn has_clearcoat(self) -> bool {
        self.header[1] & crate::models::MaterialLayeredPbr::CLEARCOAT_LOBE != 0
            && self.clearcoat_ior[0] > 0.0
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

const PT_LAYERED_CLEARCOAT_WGSL: &str = r#"
const PT_LAYERED_CLEARCOAT_LOBE: u32 = 1u;

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
        && material.clearcoat_ior.x > 0.0;
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

fn pt_layered_primary_material(p: vec3<f32>) -> PtLayeredMaterial {
    let to_surface = p - u.cam_pos.xyz;
    let distance = length(to_surface);
    if (distance <= 1e-4) {
        return pt_layered_default();
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
        return pt_layered_default();
    }
    return pt_layered_materials[hit.instance_custom_data];
}

fn pt_layered_nee(
    n: vec3<f32>,
    view: vec3<f32>,
    ldir: vec3<f32>,
    ndl: f32,
    full_alb: vec3<f32>,
    rough: f32,
    metal: f32,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    let undercoat = nee_diffuse(n, view, ldir, ndl, full_alb, rough, metal)
        + nee_spec(n, view, ldir, ndl, full_alb, rough, metal);
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
    sun_r2: vec2<f32>,
    view: vec3<f32>,
    full_alb: vec3<f32>,
    rough: f32,
    metal: f32,
    with_points: bool,
    material: PtLayeredMaterial,
) -> vec3<f32> {
    if (!pt_layered_has_clearcoat(material)) {
        return direct_light(p, n, sun_r2, view, full_alb, rough, metal, with_points);
    }
    var result = vec3<f32>(0.0);
    let sun_ndl = max(dot(n, u.sun_dir.xyz), 0.0);
    if (sun_ndl > 0.0) {
        let visibility = sun_visibility(p, n, sun_r2);
        if (visibility > 0.0) {
            result += pt_layered_nee(
                n, view, u.sun_dir.xyz, sun_ndl, full_alb, rough, metal, material,
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
                    n, view, direction, ndl, full_alb, rough, metal, material,
                ) * incident;
            }
        }
    }
    return result;
}

fn pt_sample_layered_brdf(
    n: vec3<f32>,
    view: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    material: PtLayeredMaterial,
) -> BrdfSample {
    if (!pt_layered_has_clearcoat(material)) {
        return sample_brdf(n, view, base_color, roughness, metallic);
    }
    var out: BrdfSample;
    out.valid = false;
    let ndv = max(dot(n, view), 0.0);
    if (ndv <= 0.0) {
        return out;
    }
    let base_f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let base_f = fresnel_schlick3(ndv, base_f0);
    let base_specular_weight = (base_f.x + base_f.y + base_f.z) / 3.0;
    let diffuse_weight = (1.0 - base_specular_weight) * (1.0 - metallic);
    let clearcoat_weight = pt_clearcoat_fresnel(ndv, material);
    let clearcoat_probability = clearcoat_weight
        / (base_specular_weight + diffuse_weight + clearcoat_weight + 1e-6);

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

    out = sample_brdf(n, view, base_color, roughness, metallic);
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

fn replace_once(source: &mut String, needle: &str, replacement: &str) {
    let count = source.matches(needle).count();
    assert_eq!(
        count, 1,
        "layered PT specialization expected one source anchor, found {count}: {needle}"
    );
    *source = source.replacen(needle, replacement, 1);
}

fn clearcoat_kernel_variant(base: &str) -> String {
    let mut source = base.to_owned();
    replace_once(
        &mut source,
        "    var rough_cur = mr0.g;",
        "    var rough_cur = mr0.g;\n    var layered_cur = pt_layered_primary_material(p0);",
    );
    replace_once(
        &mut source,
        "    let use_restir = u.ext.w == 1u && u.cfg.x >= 2.0;",
        "    let use_restir = u.ext.w == 1u && u.cfg.x >= 2.0\n        \
         && !pt_layered_has_clearcoat(layered_cur);",
    );
    replace_once(
        &mut source,
        "    var radiance = direct_light(\n\
         \x20       p0 + n0 * 0.02, n0, sun_r2, view_cur,\n\
         \x20       albedo0, rough_cur, metal_cur, !use_restir,\n\
         \x20   );",
        "    var radiance = pt_layered_direct_light(\n\
         \x20       p0 + n0 * 0.02, n0, sun_r2, view_cur,\n\
         \x20       albedo0, rough_cur, metal_cur, !use_restir, layered_cur,\n\
         \x20   );",
    );
    replace_once(
        &mut source,
        "        let s = sample_brdf(n_cur, view_cur, alb_cur, rough_cur, metal_cur);",
        "        let s = pt_sample_layered_brdf(\n\
         \x20           n_cur, view_cur, alb_cur, rough_cur, metal_cur, layered_cur,\n\
         \x20       );",
    );
    replace_once(
        &mut source,
        "        radiance += throughput * direct_light(\n\
         \x20           hit_p, n_hit, rand_2f(), -dir,\n\
         \x20           alb_hit, inst.mat_params.x, inst.mat_params.y, true,\n\
         \x20       );",
        "        let layered_hit = pt_layered_materials[hit.instance_custom_data];\n\
         \x20       radiance += throughput * pt_layered_direct_light(\n\
         \x20           hit_p, n_hit, rand_2f(), -dir,\n\
         \x20           alb_hit, inst.mat_params.x, inst.mat_params.y, true, layered_hit,\n\
         \x20       );",
    );
    replace_once(
        &mut source,
        "        metal_cur = inst.mat_params.y;\n        view_cur = -dir;",
        "        metal_cur = inst.mat_params.y;\n\
         \x20       layered_cur = layered_hit;\n\
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
            .any(PtLayeredMaterialCpu::has_clearcoat)
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
            let base_kernel = pt_kernel_variant(query_diagnostics);
            let layered_kernel = clearcoat_kernel_variant(base_kernel.as_ref());
            let source = format!(
                "enable wgpu_ray_query;\n{}\n{}\n{}\n{}\n{}\n{}",
                ray_query_backend_variant(&self.device),
                pt_fault_constants(fault.as_deref()),
                layered_kernel,
                texture_variant(self.pt_texture_arrays_enabled),
                PT_LAYERED_BINDINGS_WGSL,
                PT_LAYERED_CLEARCOAT_WGSL,
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
    fn only_nonzero_qualified_clearcoat_selects_layered_transport() {
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
        assert!(!PtLayeredMaterialCpu::from_material(sheen).has_clearcoat());
        assert!(PtLayeredMaterialCpu::from_material(clearcoat).has_clearcoat());
    }

    #[test]
    fn specialization_uses_separate_group_without_touching_base_kernel() {
        assert!(PT_LAYERED_BINDINGS_WGSL.contains("@group(2) @binding(0)"));
        assert!(!pt_kernel_variant(false).contains("pt_layered_materials"));
    }

    #[test]
    fn clearcoat_specialization_rewrites_every_transport_vertex() {
        for diagnostics in [false, true] {
            let base = pt_kernel_variant(diagnostics);
            let specialized = clearcoat_kernel_variant(base.as_ref());
            assert!(specialized.contains("var layered_cur = pt_layered_primary_material(p0);"));
            assert_eq!(specialized.matches("pt_sample_layered_brdf(").count(), 1);
            assert_eq!(
                specialized
                    .matches("throughput * pt_layered_direct_light(")
                    .count(),
                1
            );
            assert!(specialized.contains("layered_cur = layered_hit;"));
            assert_eq!(base.matches("pt_layered_").count(), 0);
        }
    }
}
