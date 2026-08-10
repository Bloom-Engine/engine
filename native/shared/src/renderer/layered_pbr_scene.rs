//! Lazy scene-pipeline specialization for clearcoat, dielectric specular/IOR,
//! sheen, and tangent-space anisotropy.
//!
//! Base-only materials never create these resources and continue to use the
//! established scene shader, group-2 layout, GPU-driven record, and draw path.

use super::*;

const LAYERED_PBR_V3_WGSL: &str = include_str!("../../shaders/layered_pbr_v3.wgsl");
const SHEEN_ALBEDO_LUT_R16F: &[u8] = include_bytes!("../../shaders/sheen_albedo_lut_r16f.bin");
const SHEEN_ALBEDO_LUT_SIZE: u32 = 128;
pub(super) const SHEEN_ALBEDO_LUT_BYTES: usize =
    (SHEEN_ALBEDO_LUT_SIZE * SHEEN_ALBEDO_LUT_SIZE * 2) as usize;

const CLEARCOAT_FACTOR_TEXTURE: usize = 0;
const CLEARCOAT_ROUGHNESS_TEXTURE: usize = 1;
const CLEARCOAT_NORMAL_TEXTURE: usize = 2;
const SPECULAR_FACTOR_TEXTURE: usize = 3;
const SPECULAR_COLOR_TEXTURE: usize = 4;
const SHEEN_COLOR_TEXTURE: usize = 5;
const SHEEN_ROUGHNESS_TEXTURE: usize = 6;
const ANISOTROPY_TEXTURE: usize = 7;
const IRIDESCENCE_FACTOR_TEXTURE: usize = 8;
const IRIDESCENCE_THICKNESS_TEXTURE: usize = 9;
pub(super) const LAYERED_TEXTURE_COUNT: usize = 10;

// The ordinary layered layout contributes sixteen sampled textures (the five
// base PBR maps, ten optional lobe maps, and the sheen energy LUT). The shared
// lighting layout contributes six more, or eight when virtual-shadow textures
// are present. wgpu validates this limit across the complete pipeline layout,
// rather than one bind group at a time.
pub(super) const SCENE_LAYERED_PBR_SAMPLED_TEXTURES: u32 = 22;
pub(super) const SCENE_LAYERED_PBR_VSM_SAMPLED_TEXTURES: u32 = 24;

pub(super) const fn scene_layered_pbr_sampled_texture_requirement(virtual_shadows: bool) -> u32 {
    if virtual_shadows {
        SCENE_LAYERED_PBR_VSM_SAMPLED_TEXTURES
    } else {
        SCENE_LAYERED_PBR_SAMPLED_TEXTURES
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SceneLayeredPbrUniforms {
    clearcoat_ior: [f32; 4],
    specular: [f32; 4],
    sheen: [f32; 4],
    anisotropy: [f32; 4],
    iridescence: [f32; 4],
    clearcoat_factor_uv: [f32; 4],
    clearcoat_factor_rotation: [f32; 4],
    clearcoat_roughness_uv: [f32; 4],
    clearcoat_roughness_rotation: [f32; 4],
    clearcoat_normal_uv: [f32; 4],
    clearcoat_normal_rotation: [f32; 4],
    specular_factor_uv: [f32; 4],
    specular_factor_rotation: [f32; 4],
    specular_color_uv: [f32; 4],
    specular_color_rotation: [f32; 4],
    sheen_color_uv: [f32; 4],
    sheen_color_rotation: [f32; 4],
    sheen_roughness_uv: [f32; 4],
    sheen_roughness_rotation: [f32; 4],
    anisotropy_uv: [f32; 4],
    anisotropy_texture_rotation: [f32; 4],
    iridescence_factor_uv: [f32; 4],
    iridescence_factor_rotation: [f32; 4],
    iridescence_thickness_uv: [f32; 4],
    iridescence_thickness_rotation: [f32; 4],
}

impl SceneLayeredPbrUniforms {
    pub(super) fn new(
        material: crate::models::MaterialLayeredPbr,
        texture_usable: [bool; LAYERED_TEXTURE_COUNT],
    ) -> Self {
        let transform = |binding: Option<crate::models::MaterialTextureBinding>,
                         usable: bool|
         -> ([f32; 4], [f32; 4]) {
            let transform = binding.map(|binding| binding.transform).unwrap_or_default();
            let (sin, cos) = transform.rotation.sin_cos();
            (
                [
                    transform.offset[0],
                    transform.offset[1],
                    transform.scale[0],
                    transform.scale[1],
                ],
                [
                    cos,
                    sin,
                    if usable { 1.0 } else { 0.0 },
                    transform.tex_coord as f32,
                ],
            )
        };
        let (clearcoat_factor_uv, clearcoat_factor_rotation) = transform(
            material.clearcoat_texture,
            texture_usable[CLEARCOAT_FACTOR_TEXTURE],
        );
        let (clearcoat_roughness_uv, clearcoat_roughness_rotation) = transform(
            material.clearcoat_roughness_texture,
            texture_usable[CLEARCOAT_ROUGHNESS_TEXTURE],
        );
        let (clearcoat_normal_uv, clearcoat_normal_rotation) = transform(
            material.clearcoat_normal_texture,
            texture_usable[CLEARCOAT_NORMAL_TEXTURE],
        );
        let (specular_factor_uv, specular_factor_rotation) = transform(
            material.specular_texture,
            texture_usable[SPECULAR_FACTOR_TEXTURE],
        );
        let (specular_color_uv, specular_color_rotation) = transform(
            material.specular_color_texture,
            texture_usable[SPECULAR_COLOR_TEXTURE],
        );
        let (sheen_color_uv, sheen_color_rotation) = transform(
            material.sheen_color_texture,
            texture_usable[SHEEN_COLOR_TEXTURE],
        );
        let (sheen_roughness_uv, sheen_roughness_rotation) = transform(
            material.sheen_roughness_texture,
            texture_usable[SHEEN_ROUGHNESS_TEXTURE],
        );
        let (anisotropy_uv, anisotropy_texture_rotation) = transform(
            material.anisotropy_texture,
            texture_usable[ANISOTROPY_TEXTURE],
        );
        let (iridescence_factor_uv, iridescence_factor_rotation) = transform(
            material.iridescence_texture,
            texture_usable[IRIDESCENCE_FACTOR_TEXTURE],
        );
        let (iridescence_thickness_uv, iridescence_thickness_rotation) = transform(
            material.iridescence_thickness_texture,
            texture_usable[IRIDESCENCE_THICKNESS_TEXTURE],
        );
        let (anisotropy_sine, anisotropy_cosine) = material.anisotropy_rotation.sin_cos();
        Self {
            clearcoat_ior: [
                material.clearcoat_factor.clamp(0.0, 1.0),
                material.clearcoat_roughness_factor.clamp(0.0, 1.0),
                material.clearcoat_normal_scale,
                material.ior,
            ],
            specular: [
                material.specular_color_factor[0].max(0.0),
                material.specular_color_factor[1].max(0.0),
                material.specular_color_factor[2].max(0.0),
                material.specular_factor.clamp(0.0, 1.0),
            ],
            sheen: [
                material.sheen_color_factor[0].clamp(0.0, 1.0),
                material.sheen_color_factor[1].clamp(0.0, 1.0),
                material.sheen_color_factor[2].clamp(0.0, 1.0),
                material.sheen_roughness_factor.clamp(0.0, 1.0),
            ],
            anisotropy: [
                material.anisotropy_strength.clamp(0.0, 1.0),
                anisotropy_cosine,
                anisotropy_sine,
                0.0,
            ],
            iridescence: [
                material.iridescence_factor.clamp(0.0, 1.0),
                material.iridescence_ior.max(1.0),
                material.iridescence_thickness_minimum.max(0.0),
                material.iridescence_thickness_maximum.max(0.0),
            ],
            clearcoat_factor_uv,
            clearcoat_factor_rotation,
            clearcoat_roughness_uv,
            clearcoat_roughness_rotation,
            clearcoat_normal_uv,
            clearcoat_normal_rotation,
            specular_factor_uv,
            specular_factor_rotation,
            specular_color_uv,
            specular_color_rotation,
            sheen_color_uv,
            sheen_color_rotation,
            sheen_roughness_uv,
            sheen_roughness_rotation,
            anisotropy_uv,
            anisotropy_texture_rotation,
            iridescence_factor_uv,
            iridescence_factor_rotation,
            iridescence_thickness_uv,
            iridescence_thickness_rotation,
        }
    }
}

fn replace_once(source: String, anchor: &str, replacement: &str, label: &str) -> String {
    let replaced = source.replacen(anchor, replacement, 1);
    assert_ne!(
        replaced, source,
        "scene shader {label} changed; layered-PBR specialization must be updated"
    );
    replaced
}

pub(super) fn scene_layered_shader_source(base_scene_shader: &str, secondary_uv: bool) -> String {
    scene_layered_shader_source_with_bindings(base_scene_shader, secondary_uv, 11)
}

pub(super) fn scene_layered_shader_source_with_bindings(
    base_scene_shader: &str,
    secondary_uv: bool,
    first_material_binding: u32,
) -> String {
    const JOINT_DECLARATION: &str =
        "@group(3) @binding(1) var<uniform> joints_prev: JointMatrices;";
    let secondary_uv_function = if secondary_uv {
        r#"
fn layered_secondary_uv(in: VertexOutputScene) -> vec2<f32> {
    return in.secondary_uv;
}
"#
    } else {
        r#"
fn layered_secondary_uv(in: VertexOutputScene) -> vec2<f32> {
    return in.uv;
}
"#
    };
    let declarations = format!(
        r#"{JOINT_DECLARATION}
@group(2) @binding({clearcoat_factor_texture}) var layered_clearcoat_factor_tex: texture_2d<f32>;
@group(2) @binding({clearcoat_roughness_texture}) var layered_clearcoat_roughness_tex: texture_2d<f32>;
@group(2) @binding({clearcoat_normal_texture}) var layered_clearcoat_normal_tex: texture_2d<f32>;
@group(2) @binding({specular_factor_texture}) var layered_specular_factor_tex: texture_2d<f32>;
@group(2) @binding({specular_color_texture}) var layered_specular_color_tex: texture_2d<f32>;
@group(2) @binding({sheen_color_texture}) var layered_sheen_color_tex: texture_2d<f32>;
@group(2) @binding({sheen_roughness_texture}) var layered_sheen_roughness_tex: texture_2d<f32>;
@group(2) @binding({anisotropy_texture}) var layered_anisotropy_tex: texture_2d<f32>;
@group(2) @binding({iridescence_factor_texture}) var layered_iridescence_factor_tex: texture_2d<f32>;
@group(2) @binding({iridescence_thickness_texture}) var layered_iridescence_thickness_tex: texture_2d<f32>;
@group(2) @binding({sheen_albedo_texture}) var layered_sheen_albedo_tex: texture_2d<f32>;
@group(2) @binding({sampler}) var layered_sampler: sampler;
@group(2) @binding({uniform}) var<uniform> layered_material: LayeredPbrFactors;

{LAYERED_PBR_V3_WGSL}
{secondary_uv_function}"#,
        clearcoat_factor_texture = first_material_binding,
        clearcoat_roughness_texture = first_material_binding + 1,
        clearcoat_normal_texture = first_material_binding + 2,
        specular_factor_texture = first_material_binding + 3,
        specular_color_texture = first_material_binding + 4,
        sheen_color_texture = first_material_binding + 5,
        sheen_roughness_texture = first_material_binding + 6,
        anisotropy_texture = first_material_binding + 7,
        iridescence_factor_texture = first_material_binding + 8,
        iridescence_thickness_texture = first_material_binding + 9,
        sheen_albedo_texture = first_material_binding + 10,
        sampler = first_material_binding + 11,
        uniform = first_material_binding + 12,
    );
    let mut source = replace_once(
        base_scene_shader.to_owned(),
        JOINT_DECLARATION,
        &declarations,
        "joint declaration",
    );

    if secondary_uv {
        source = replace_once(
            source,
            "    @location(6) tangent: vec4<f32>,\n};",
            "    @location(6) tangent: vec4<f32>,\n\
             @location(7) secondary_uv: vec2<f32>,\n\
             };",
            "vertex input",
        );
        source = replace_once(
            source,
            "    @location(6) prev_clip: vec4<f32>,",
            "    @location(6) prev_clip: vec4<f32>,\n\
             @location(7) secondary_uv: vec2<f32>,",
            "vertex output",
        );
        source = replace_once(
            source,
            "        o.uv = in.uv;",
            "        o.uv = in.uv;\n        o.secondary_uv = in.secondary_uv;",
            "skinned UV output",
        );
        source = replace_once(
            source,
            "    out.uv = in.uv;",
            "    out.uv = in.uv;\n    out.secondary_uv = in.secondary_uv;",
            "rigid UV output",
        );
    }

    source = replace_once(
        source,
        "        o.tangent = vec4<f32>(safe_scene_tangent(tan4.xyz), in.tangent.w);",
        "        let layered_model_handedness = select(\n\
         -1.0,\n\
         1.0,\n\
         dot(cross(u.model[0].xyz, u.model[1].xyz), u.model[2].xyz) >= 0.0,\n\
         );\n\
         o.tangent = vec4<f32>(\n\
         safe_scene_tangent(tan4.xyz),\n\
         in.tangent.w * layered_model_handedness,\n\
         );",
        "skinned mirrored-transform tangent handedness",
    );
    source = replace_once(
        source,
        r#"    out.tangent = vec4<f32>(
        safe_scene_tangent((u.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        in.tangent.w,
    );"#,
        r#"    let layered_model_handedness = select(
        -1.0,
        1.0,
        dot(cross(u.model[0].xyz, u.model[1].xyz), u.model[2].xyz) >= 0.0,
    );
    out.tangent = vec4<f32>(
        safe_scene_tangent((u.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        in.tangent.w * layered_model_handedness,
    );"#,
        "rigid mirrored-transform tangent handedness",
    );

    source = replace_once(
        source,
        "fn shade_pbr(\n    n: vec3<f32>,",
        "fn shade_layered_base_pbr(\n    surface: LayeredSurface,\n    n: vec3<f32>,",
        "direct BRDF entry point",
    );
    source = replace_once(
        source,
        r#"        let kd0 = (vec3<f32>(1.0) - mix(vec3<f32>(0.04), base_color, metallic)) * (1.0 - metallic);
        return kd0 * base_color / PI * light_color * intensity * n_dot_l;"#,
        r#"        let interface_transmission =
            layered_dielectric_transmission(surface, n_dot_v)
            * layered_dielectric_transmission(surface, n_dot_l);
        return base_color * (1.0 - metallic) * interface_transmission
            / PI * light_color * intensity * n_dot_l;"#,
        "degenerate-half diffuse",
    );
    source = replace_once(
        source,
        r#"    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let f = f_schlick(v_dot_h, f0);"#,
        r#"    let f0 = mix(surface.dielectric_f0, base_color, metallic);
    let f = layered_base_fresnel(surface, v_dot_h, base_color, metallic);"#,
        "direct Fresnel",
    );
    source = replace_once(
        source,
        r#"    let d = d_ggx(n_dot_h, alpha2);
    let vis = v_smith_ggx_correlated(n_dot_l, n_dot_v, alpha2);"#,
        r#"    let d = layered_base_distribution(surface, n, h, n_dot_h, alpha);
    let vis = layered_base_visibility(
        surface,
        n,
        v,
        l_dir,
        n_dot_l,
        n_dot_v,
        alpha,
    );"#,
        "anisotropic direct GGX",
    );
    source = replace_once(
        source,
        r#"    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;"#,
        r#"    let interface_transmission =
        layered_dielectric_transmission(surface, n_dot_v)
        * layered_dielectric_transmission(surface, n_dot_l);
    let diffuse = base_color * (1.0 - metallic) * interface_transmission / PI;"#,
        "direct diffuse complement",
    );

    source = source.replace(
        "shade_pbr(n, v,",
        "shade_layered_pbr(layered_surface, n, v,",
    );
    assert_eq!(
        source
            .matches("shade_layered_pbr(layered_surface, n, v,")
            .count(),
        3,
        "scene direct-light call count changed; layered specialization must be updated"
    );

    source = replace_once(
        source,
        r#"    // --- PBR direct lighting ---
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);"#,
        r#"    // --- PBR direct lighting ---
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);
    let layered_surface = evaluate_layered_surface(in, n, lod_bias);
    let layered_base_attenuation =
        layered_clearcoat_ibl_attenuation(layered_surface, v);
    let layered_sheen_base_attenuation = layered_sheen_ibl_scale(
        layered_surface,
        max(dot(n, v), 0.0),
    );"#,
        "surface evaluation",
    );
    source = replace_once(
        source,
        "    var lit = lighting.ambient.rgb * lighting.ambient.a * base_color;",
        r#"    var lit = lighting.ambient.rgb * lighting.ambient.a * base_color
        * layered_base_attenuation * layered_sheen_base_attenuation;"#,
        "ambient coat attenuation",
    );

    source = replace_once(
        source,
        "    let f0 = mix(vec3<f32>(0.04), base_color, metallic);",
        r#"    let f0 = layered_ibl_f0(
        layered_surface,
        max(dot(n, v), 0.0),
        base_color,
        metallic,
    );
    let f90 = mix(
        vec3<f32>(layered_surface.dielectric_f90),
        vec3<f32>(1.0),
        metallic,
    );"#,
        "IBL F0/F90",
    );
    source = replace_once(
        source,
        r#"    let fc_n = pow(1.0 - n_dot_v_ibl, 5.0);
    let f_ibl = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * fc_n;
    let kd = (vec3<f32>(1.0) - f_ibl) * (1.0 - metallic);"#,
        r#"    let dielectric_f_ibl = layered_base_fresnel_roughness(
        layered_surface,
        n_dot_v_ibl,
        base_color,
        0.0,
        roughness,
    );
    let kd = vec3<f32>(max(1.0 - max(
        dielectric_f_ibl.r,
        max(dielectric_f_ibl.g, dielectric_f_ibl.b),
    ), 0.0)) * (1.0 - metallic);"#,
        "IBL diffuse complement",
    );
    source = replace_once(
        source,
        "    let r = reflect(-v, n);",
        "    let r = layered_ibl_reflection(layered_surface, n, v, roughness);",
        "anisotropic IBL reflection",
    );
    source = replace_once(
        source,
        "    let single_spec = prefiltered_env * (f0 * brdf.x + vec3<f32>(brdf.y));",
        "    let single_spec = prefiltered_env * (f0 * brdf.x + f90 * brdf.y);",
        "IBL single scatter F90",
    );
    source = replace_once(
        source,
        "    let f_avg = f0 + (vec3<f32>(1.0) - f0) * (1.0 / 21.0);",
        "    let f_avg = f0 + (f90 - f0) * (1.0 / 21.0);",
        "IBL average Fresnel",
    );
    source = replace_once(
        source,
        "        * (f0 * brdf.x + vec3<f32>(brdf.y) + ms_contribution);",
        "        * (f0 * brdf.x + f90 * brdf.y + ms_contribution);",
        "IBL compensated F90",
    );
    source = replace_once(
        source,
        "    //IBL_STRIP_END",
        r#"    let ibl_sheen = layered_sheen_ibl(
        layered_surface,
        n,
        v,
        max_spec_mip,
        occlusion,
    );
    let clearcoat_n_dot_v = max(
        dot(layered_surface.clearcoat_normal, v),
        0.0,
    );
    let clearcoat_reflection = reflect(-v, layered_surface.clearcoat_normal);
    let clearcoat_prefiltered = env_sample_lod(
        clearcoat_reflection,
        layered_surface.clearcoat_roughness * max_spec_mip,
    );
    let clearcoat_brdf = textureSample(
        brdf_lut_tex,
        brdf_lut_samp,
        vec2<f32>(clearcoat_n_dot_v, layered_surface.clearcoat_roughness),
    ).rg;
    let clearcoat_f0 = vec3<f32>(0.04 * layered_surface.clearcoat_factor);
    let clearcoat_f90 = vec3<f32>(layered_surface.clearcoat_factor);
    let clearcoat_spec_raw = clearcoat_prefiltered
        * (clearcoat_f0 * clearcoat_brdf.x + clearcoat_f90 * clearcoat_brdf.y);
    let clearcoat_luma = dot(
        clearcoat_spec_raw,
        vec3<f32>(0.2126, 0.7152, 0.0722),
    );
    let clearcoat_cap = 1.0 / (1.0 + clearcoat_luma / 0.3);
    let clearcoat_spec_occ = clamp(
        pow(
            clearcoat_n_dot_v + occlusion,
            exp2(-16.0 * layered_surface.clearcoat_roughness - 1.0),
        ) - 1.0 + occlusion,
        0.0,
        1.0,
    );
    // The existing SSR material buffer describes the base lobe. It can own
    // coated-metal reflections, but must not suppress dielectric varnish.
    let clearcoat_ssr_own = ssr_own * metallic
        * (1.0 - smoothstep(
            0.5,
            0.85,
            layered_surface.clearcoat_roughness,
        ));
    let ibl_clearcoat = clearcoat_spec_raw
        * clearcoat_cap
        * clearcoat_spec_occ
        * (1.0 - clearcoat_ssr_own);
    //IBL_STRIP_END"#,
        "clearcoat IBL",
    );
    source = replace_once(
        source,
        "    let hdr_raw = lit + (ibl_diffuse + ibl_spec) * indirect_shadow + emissive;",
        r#"    let hdr_raw = lit
        + (((ibl_diffuse + ibl_spec) * layered_sheen_base_attenuation + ibl_sheen)
                * layered_base_attenuation
            + ibl_clearcoat)
            * indirect_shadow
        + emissive * layered_sheen_base_attenuation * layered_base_attenuation;"#,
        "final layered composition",
    );
    source
}

const LAYERED_FINAL_COMPOSITION: &str = r#"    let hdr_raw = lit
        + (((ibl_diffuse + ibl_spec) * layered_sheen_base_attenuation + ibl_sheen)
                * layered_base_attenuation
            + ibl_clearcoat)
            * indirect_shadow
        + emissive * layered_sheen_base_attenuation * layered_base_attenuation;"#;

/// Opt-in, compile-time material diagnostics for capture qualification.
///
/// The selected expression replaces the final layered composition before
/// wgpu compiles the lazy specialization, so production shaders pay no branch
/// or binding cost. This is deliberately internal: the public renderer debug
/// API owns stable user-facing modes.
fn apply_layered_capture_debug(source: String, mode: Option<&str>) -> String {
    let Some(replacement) = mode.and_then(|mode| match mode {
        "material" => Some(
            "    let hdr_raw = vec3<f32>(\n\
             roughness,\n\
             metallic,\n\
             layered_surface.iridescence_factor,\n\
             );",
        ),
        "normal" => Some("    let hdr_raw = n * 0.5 + vec3<f32>(0.5);"),
        "normal-texel" => Some("    let hdr_raw = nm_sample4.rgb;"),
        "normal-tangent-length" => Some(
            "    let layered_debug_tangent_length = min(tlen2, 1.0);\n\
             let hdr_raw = vec3<f32>(layered_debug_tangent_length);",
        ),
        "geometric-normal" => Some(
            "    let layered_debug_geometric_normal = normalize(in.normal);\n\
             let hdr_raw = layered_debug_geometric_normal * 0.5 + vec3<f32>(0.5);",
        ),
        "view-cosine" => Some(
            "    let layered_debug_n_dot_v = max(dot(n, v), 0.0);\n\
             let hdr_raw = vec3<f32>(layered_debug_n_dot_v);",
        ),
        "geometric-view-cosine" => Some(
            "    let layered_debug_n_dot_v = max(dot(normalize(in.normal), v), 0.0);\n\
             let hdr_raw = vec3<f32>(layered_debug_n_dot_v);",
        ),
        "iridescence-fresnel" => Some(
            "    let hdr_raw = layered_raw_iridescence_base_fresnel(\n\
             layered_surface,\n\
             max(dot(n, v), 0.0),\n\
             base_color,\n\
             metallic,\n\
             );",
        ),
        "iridescence-ibl-f0" => Some(
            "    let hdr_raw = layered_ibl_f0(\n\
             layered_surface,\n\
             max(dot(n, v), 0.0),\n\
             base_color,\n\
             metallic,\n\
             );",
        ),
        "iridescence-prefiltered-env" => Some("    let hdr_raw = prefiltered_env;"),
        "iridescence-single-spec" => Some("    let hdr_raw = single_spec;"),
        "iridescence-multiscatter" => Some("    let hdr_raw = ms_contribution;"),
        "iridescence-ibl-spec-raw" => Some("    let hdr_raw = ibl_spec_raw;"),
        "iridescence-ibl-spec" => Some("    let hdr_raw = ibl_spec;"),
        "iridescence-roughness" => Some("    let hdr_raw = vec3<f32>(roughness);"),
        _ => None,
    }) else {
        return source;
    };
    replace_once(
        source,
        LAYERED_FINAL_COMPOSITION,
        replacement,
        "capture debug composition",
    )
}

pub(crate) struct SceneLayeredPbrResources {
    pub(crate) material_layout: wgpu::BindGroupLayout,
    opaque: wgpu::RenderPipeline,
    opaque_prepassed: wgpu::RenderPipeline,
    opaque_uv1: wgpu::RenderPipeline,
    opaque_prepassed_uv1: wgpu::RenderPipeline,
    transparent: wgpu::RenderPipeline,
    transparent_double_sided: wgpu::RenderPipeline,
    transparent_uv1: wgpu::RenderPipeline,
    transparent_uv1_double_sided: wgpu::RenderPipeline,
    reactive: Option<wgpu::RenderPipeline>,
    reactive_double_sided: Option<wgpu::RenderPipeline>,
    reactive_uv1: Option<wgpu::RenderPipeline>,
    reactive_uv1_double_sided: Option<wgpu::RenderPipeline>,
    weighted: Option<wgpu::RenderPipeline>,
    weighted_double_sided: Option<wgpu::RenderPipeline>,
    weighted_uv1: Option<wgpu::RenderPipeline>,
    weighted_uv1_double_sided: Option<wgpu::RenderPipeline>,
}

pub(crate) struct SceneSheenAlbedoLut {
    _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

impl Renderer {
    pub(super) fn ensure_scene_sheen_albedo_lut(&mut self) {
        if self.scene_sheen_albedo_lut.is_some() {
            return;
        }
        assert_eq!(
            SHEEN_ALBEDO_LUT_R16F.len(),
            SHEEN_ALBEDO_LUT_BYTES,
            "checked sheen LUT dimensions changed"
        );
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_sheen_albedo_lut"),
            size: wgpu::Extent3d {
                width: SHEEN_ALBEDO_LUT_SIZE,
                height: SHEEN_ALBEDO_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            SHEEN_ALBEDO_LUT_R16F,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SHEEN_ALBEDO_LUT_SIZE * 2),
                rows_per_image: Some(SHEEN_ALBEDO_LUT_SIZE),
            },
            wgpu::Extent3d {
                width: SHEEN_ALBEDO_LUT_SIZE,
                height: SHEEN_ALBEDO_LUT_SIZE,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.scene_sheen_albedo_lut = Some(SceneSheenAlbedoLut {
            _texture: texture,
            view,
        });
    }
}

impl SceneLayeredPbrResources {
    pub(crate) fn opaque_pipeline(
        &self,
        secondary_uv: bool,
        prepassed: bool,
    ) -> &wgpu::RenderPipeline {
        match (secondary_uv, prepassed) {
            (false, false) => &self.opaque,
            (false, true) => &self.opaque_prepassed,
            (true, false) => &self.opaque_uv1,
            (true, true) => &self.opaque_prepassed_uv1,
        }
    }

    pub(crate) fn transparent_pipeline(
        &self,
        secondary_uv: bool,
        double_sided: bool,
        reactive: bool,
        weighted: bool,
    ) -> &wgpu::RenderPipeline {
        match (weighted, reactive, secondary_uv, double_sided) {
            (false, false, false, false) => &self.transparent,
            (false, false, false, true) => &self.transparent_double_sided,
            (false, false, true, false) => &self.transparent_uv1,
            (false, false, true, true) => &self.transparent_uv1_double_sided,
            (false, true, false, false) => self
                .reactive
                .as_ref()
                .expect("layered reactive resources are initialized"),
            (false, true, false, true) => self
                .reactive_double_sided
                .as_ref()
                .expect("layered reactive resources are initialized"),
            (false, true, true, false) => self
                .reactive_uv1
                .as_ref()
                .expect("layered reactive UV1 resources are initialized"),
            (false, true, true, true) => self
                .reactive_uv1_double_sided
                .as_ref()
                .expect("layered reactive UV1 resources are initialized"),
            (true, _, false, false) => self
                .weighted
                .as_ref()
                .expect("layered weighted resources are initialized"),
            (true, _, false, true) => self
                .weighted_double_sided
                .as_ref()
                .expect("layered weighted resources are initialized"),
            (true, _, true, false) => self
                .weighted_uv1
                .as_ref()
                .expect("layered weighted UV1 resources are initialized"),
            (true, _, true, true) => self
                .weighted_uv1_double_sided
                .as_ref()
                .expect("layered weighted UV1 resources are initialized"),
        }
    }
}

fn create_layered_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let uniform = |binding, visibility| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_layered_pbr_material_layout"),
        entries: &[
            texture(0),
            sampler(1),
            texture(2),
            sampler(3),
            texture(4),
            sampler(5),
            texture(6),
            sampler(7),
            uniform(8, wgpu::ShaderStages::VERTEX_FRAGMENT),
            texture(9),
            sampler(10),
            texture(11),
            texture(12),
            texture(13),
            texture(14),
            texture(15),
            texture(16),
            texture(17),
            texture(18),
            texture(19),
            texture(20),
            texture(21),
            sampler(22),
            uniform(23, wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

fn strip_prepassed_discard(source: &str) -> String {
    let (begin, end) = (
        source
            .find("//PREPASS_STRIP_BEGIN")
            .expect("layered scene shader keeps prepass strip start"),
        source
            .find("//PREPASS_STRIP_END")
            .expect("layered scene shader keeps prepass strip end"),
    );
    assert!(end > begin, "layered prepass strip markers are ordered");
    let end = end + "//PREPASS_STRIP_END".len();
    format!("{}{}", &source[..begin], &source[end..])
}

fn scene_vertex_buffers(secondary_uv: bool) -> Vec<wgpu::VertexBufferLayout<'static>> {
    let mut buffers = vec![Vertex3D::desc()];
    if secondary_uv {
        buffers.push(secondary_uv_desc());
    }
    buffers
}

fn scene_main_targets() -> Vec<Option<wgpu::ColorTargetState>> {
    #[cfg(lean_mrt)]
    {
        vec![
            Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            None,
            Some(wgpu::ColorTargetState {
                format: VELOCITY_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            None,
        ]
    }
    #[cfg(not(lean_mrt))]
    {
        vec![
            Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: MATERIAL_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: VELOCITY_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ]
    }
}

struct LayeredPipelineModules<'a> {
    scalar: &'a wgpu::ShaderModule,
    scalar_prepassed: &'a wgpu::ShaderModule,
    uv1: &'a wgpu::ShaderModule,
    uv1_prepassed: &'a wgpu::ShaderModule,
}

fn create_layered_opaque_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    modules: &LayeredPipelineModules<'_>,
    secondary_uv: bool,
    prepassed: bool,
    label: &'static str,
) -> wgpu::RenderPipeline {
    let shader = match (secondary_uv, prepassed) {
        (false, false) => modules.scalar,
        (false, true) => modules.scalar_prepassed,
        (true, false) => modules.uv1,
        (true, true) => modules.uv1_prepassed,
    };
    let buffers = scene_vertex_buffers(secondary_uv);
    let targets = scene_main_targets();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main_scene"),
            buffers: &buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main_scene"),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(!prepassed),
            depth_compare: Some(if prepassed {
                wgpu::CompareFunction::Equal
            } else {
                wgpu::CompareFunction::Less
            }),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn create_layered_transparent_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &'static str,
    targets: &[Option<wgpu::ColorTargetState>],
    secondary_uv: bool,
    double_sided: bool,
    label: &'static str,
) -> wgpu::RenderPipeline {
    let buffers = scene_vertex_buffers(secondary_uv);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main_scene"),
            buffers: &buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(entry_point),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl Renderer {
    pub(super) fn scene_layered_pbr_sampled_texture_requirement(&self) -> u32 {
        scene_layered_pbr_sampled_texture_requirement(self.shadow_map.virtual_map.requested())
    }

    pub(super) fn scene_layered_pbr_available(&self) -> bool {
        self.device.limits().max_sampled_textures_per_shader_stage
            >= self.scene_layered_pbr_sampled_texture_requirement()
    }

    pub(crate) fn ensure_scene_layered_pbr_resources(&mut self) -> bool {
        if self.scene_layered_pbr_resources.is_some() {
            return true;
        }
        if !self.scene_layered_pbr_available() {
            static WARN_UNAVAILABLE: std::sync::Once = std::sync::Once::new();
            WARN_UNAVAILABLE.call_once(|| {
                log::warn!(
                    "bloom materials: layered-PBR scene specialization requires {} sampled \
                     textures per fragment stage, but the negotiated device grants {}; \
                     retaining the base PBR material path",
                    self.scene_layered_pbr_sampled_texture_requirement(),
                    self.device.limits().max_sampled_textures_per_shader_stage,
                );
            });
            return false;
        }
        let material_layout = create_layered_material_layout(&self.device);
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scene_layered_pbr_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&self.uniform_3d_layout),
                    Some(&self.lighting_layout),
                    Some(&material_layout),
                    Some(&self.joint_layout),
                ],
                immediate_size: 0,
            });
        let source = |secondary_uv| {
            let source = specialized_scene_shader_source_from(
                scene_layered_shader_source(SCENE_SHADER, secondary_uv).into(),
                self.froxel.is_some(),
                self.shadow_map.virtual_map.requested(),
            )
            .into_owned();
            apply_layered_capture_debug(
                source,
                std::env::var("BLOOM_LAYERED_DEBUG").ok().as_deref(),
            )
        };
        let scalar_source = source(false);
        let uv1_source = source(true);
        let scalar_prepassed_source = strip_prepassed_discard(&scalar_source);
        let uv1_prepassed_source = strip_prepassed_discard(&uv1_source);
        let scalar = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_shader"),
                source: wgpu::ShaderSource::Wgsl(scalar_source.into()),
            });
        let scalar_prepassed = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_prepassed_shader"),
                source: wgpu::ShaderSource::Wgsl(scalar_prepassed_source.into()),
            });
        let uv1 = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_uv1_shader"),
                source: wgpu::ShaderSource::Wgsl(uv1_source.into()),
            });
        let uv1_prepassed = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_uv1_prepassed_shader"),
                source: wgpu::ShaderSource::Wgsl(uv1_prepassed_source.into()),
            });
        let modules = LayeredPipelineModules {
            scalar: &scalar,
            scalar_prepassed: &scalar_prepassed,
            uv1: &uv1,
            uv1_prepassed: &uv1_prepassed,
        };
        let opaque = create_layered_opaque_pipeline(
            &self.device,
            &pipeline_layout,
            &modules,
            false,
            false,
            "scene_layered_pbr_pipeline",
        );
        let opaque_prepassed = create_layered_opaque_pipeline(
            &self.device,
            &pipeline_layout,
            &modules,
            false,
            true,
            "scene_layered_pbr_prepassed_pipeline",
        );
        let opaque_uv1 = create_layered_opaque_pipeline(
            &self.device,
            &pipeline_layout,
            &modules,
            true,
            false,
            "scene_layered_pbr_uv1_pipeline",
        );
        let opaque_prepassed_uv1 = create_layered_opaque_pipeline(
            &self.device,
            &pipeline_layout,
            &modules,
            true,
            true,
            "scene_layered_pbr_prepassed_uv1_pipeline",
        );
        let transparent_targets = [Some(wgpu::ColorTargetState {
            format: HDR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let transparent = create_layered_transparent_pipeline(
            &self.device,
            &pipeline_layout,
            &scalar,
            "fs_transparent_scene",
            &transparent_targets,
            false,
            false,
            "scene_layered_pbr_transparent_pipeline",
        );
        let transparent_double_sided = create_layered_transparent_pipeline(
            &self.device,
            &pipeline_layout,
            &scalar,
            "fs_transparent_scene",
            &transparent_targets,
            false,
            true,
            "scene_layered_pbr_transparent_double_sided_pipeline",
        );
        let transparent_uv1 = create_layered_transparent_pipeline(
            &self.device,
            &pipeline_layout,
            &uv1,
            "fs_transparent_scene",
            &transparent_targets,
            true,
            false,
            "scene_layered_pbr_transparent_uv1_pipeline",
        );
        let transparent_uv1_double_sided = create_layered_transparent_pipeline(
            &self.device,
            &pipeline_layout,
            &uv1,
            "fs_transparent_scene",
            &transparent_targets,
            true,
            true,
            "scene_layered_pbr_transparent_uv1_double_sided_pipeline",
        );
        self.scene_layered_pbr_resources = Some(SceneLayeredPbrResources {
            material_layout,
            opaque,
            opaque_prepassed,
            opaque_uv1,
            opaque_prepassed_uv1,
            transparent,
            transparent_double_sided,
            transparent_uv1,
            transparent_uv1_double_sided,
            reactive: None,
            reactive_double_sided: None,
            reactive_uv1: None,
            reactive_uv1_double_sided: None,
            weighted: None,
            weighted_double_sided: None,
            weighted_uv1: None,
            weighted_uv1_double_sided: None,
        });
        self.created_pipelines(8);
        log::info!(
            "bloom materials: lazy layered-PBR v4 scene specialization enabled \
             (base-only scene pipelines remain unchanged)"
        );
        true
    }
}

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_scene_layered_pbr_material_bg(
        &mut self,
        base_color_tex_idx: u32,
        normal_tex_idx: u32,
        metallic_roughness_tex_idx: u32,
        emissive_tex_idx: u32,
        occlusion_tex_idx: u32,
        material_uniform: &wgpu::Buffer,
        material: crate::models::MaterialLayeredPbr,
        has_secondary_tex_coords: bool,
    ) -> Option<(wgpu::Buffer, wgpu::BindGroup, bool)> {
        if !material.is_active() {
            return None;
        }

        let mut indices = [0u32; LAYERED_TEXTURE_COUNT];
        let mut usable = [false; LAYERED_TEXTURE_COUNT];
        let bindings = [
            material.clearcoat_texture,
            material.clearcoat_roughness_texture,
            material.clearcoat_normal_texture,
            material.specular_texture,
            material.specular_color_texture,
            material.sheen_color_texture,
            material.sheen_roughness_texture,
            material.anisotropy_texture,
            material.iridescence_texture,
            material.iridescence_thickness_texture,
        ];
        let contributes = [
            material.has_clearcoat(),
            material.has_clearcoat(),
            material.has_clearcoat(),
            material.has_specular_ior() && material.specular_factor > 0.0,
            material.has_specular_ior() && material.specular_factor > 0.0,
            material.has_sheen(),
            material.has_sheen(),
            material.has_anisotropy(),
            material.has_iridescence(),
            material.has_iridescence(),
        ];
        for (slot, binding) in bindings.into_iter().enumerate() {
            let Some(binding) = binding.filter(|_| contributes[slot]) else {
                continue;
            };
            match binding.transform.tex_coord {
                0 => {}
                1 if has_secondary_tex_coords => {}
                1 => {
                    log::warn!(
                        "bloom materials: layered-PBR texture requests TEXCOORD_1 but \
                         this primitive has no valid secondary UV stream; preserving \
                         the authored data and using its scalar factor"
                    );
                    continue;
                }
                tex_coord => {
                    log::warn!(
                        "bloom materials: layered-PBR texture TEXCOORD_{tex_coord} is \
                         preserved but only TEXCOORD_0/1 are renderable; using its \
                         scalar factor"
                    );
                    continue;
                }
            }
            let Some(index) = binding
                .runtime_texture_idx
                .filter(|index| *index != 0 && (*index as usize) < self.textures.len())
            else {
                log::warn!(
                    "bloom materials: layered-PBR source texture {} is unavailable \
                     at runtime; preserving its source metadata and using the scalar factor",
                    binding.source_texture_index
                );
                continue;
            };
            indices[slot] = index;
            usable[slot] = true;
        }
        let uses_uv1 = bindings.into_iter().enumerate().any(|(slot, binding)| {
            usable[slot] && binding.is_some_and(|binding| binding.transform.tex_coord == 1)
        });

        if !self.ensure_scene_layered_pbr_resources() {
            return None;
        }
        if material.has_sheen() {
            self.ensure_scene_sheen_albedo_lut();
        }
        let uniforms = SceneLayeredPbrUniforms::new(material, usable);
        let layered_uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene_layered_pbr_uniform"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let view_or_white = |index: u32| {
            self.textures
                .get(index as usize)
                .unwrap_or(&self.textures[0])
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let base_view = view_or_white(base_color_tex_idx);
        let mr_view = view_or_white(metallic_roughness_tex_idx);
        let emissive_view = view_or_white(emissive_tex_idx);
        let occlusion_view = view_or_white(occlusion_tex_idx);
        let clearcoat_factor_view = view_or_white(indices[CLEARCOAT_FACTOR_TEXTURE]);
        let clearcoat_roughness_view = view_or_white(indices[CLEARCOAT_ROUGHNESS_TEXTURE]);
        let specular_factor_view = view_or_white(indices[SPECULAR_FACTOR_TEXTURE]);
        let specular_color_view = view_or_white(indices[SPECULAR_COLOR_TEXTURE]);
        let sheen_color_view = view_or_white(indices[SHEEN_COLOR_TEXTURE]);
        let sheen_roughness_view = view_or_white(indices[SHEEN_ROUGHNESS_TEXTURE]);
        let anisotropy_view = view_or_white(indices[ANISOTROPY_TEXTURE]);
        let iridescence_factor_view = view_or_white(indices[IRIDESCENCE_FACTOR_TEXTURE]);
        let iridescence_thickness_view = view_or_white(indices[IRIDESCENCE_THICKNESS_TEXTURE]);
        let sheen_lut_fallback = view_or_white(0);
        let sheen_albedo_view = self
            .scene_sheen_albedo_lut
            .as_ref()
            .map(|lut| &lut.view)
            .unwrap_or(&sheen_lut_fallback);
        let normal_view_owned = self
            .textures
            .get(normal_tex_idx as usize)
            .filter(|_| normal_tex_idx != 0)
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let normal_view = normal_view_owned
            .as_ref()
            .unwrap_or(&self.default_normal_view);
        let clearcoat_normal_view_owned = self
            .textures
            .get(indices[CLEARCOAT_NORMAL_TEXTURE] as usize)
            .filter(|_| usable[CLEARCOAT_NORMAL_TEXTURE])
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let clearcoat_normal_view = clearcoat_normal_view_owned
            .as_ref()
            .unwrap_or(&self.default_normal_view);
        let layout = &self
            .scene_layered_pbr_resources
            .as_ref()
            .expect("layered resources were initialized")
            .material_layout;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_layered_pbr_material_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&emissive_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: material_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&occlusion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&clearcoat_factor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&clearcoat_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(clearcoat_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(&specular_factor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&specular_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&sheen_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(&sheen_roughness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&anisotropy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::TextureView(&iridescence_factor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: wgpu::BindingResource::TextureView(&iridescence_thickness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: wgpu::BindingResource::TextureView(sheen_albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: layered_uniform.as_entire_binding(),
                },
            ],
        });
        Some((layered_uniform, bind_group, uses_uv1))
    }
}

fn layered_pipeline_layout(renderer: &Renderer, label: &'static str) -> wgpu::PipelineLayout {
    let material_layout = &renderer
        .scene_layered_pbr_resources
        .as_ref()
        .expect("layered resources are initialized")
        .material_layout;
    renderer
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[
                Some(&renderer.uniform_3d_layout),
                Some(&renderer.lighting_layout),
                Some(material_layout),
                Some(&renderer.joint_layout),
            ],
            immediate_size: 0,
        })
}

impl Renderer {
    pub(super) fn ensure_scene_layered_pbr_reactive_resources(&mut self) {
        let Some(resources) = self.scene_layered_pbr_resources.as_ref() else {
            return;
        };
        if resources.reactive.is_some() {
            return;
        }
        let layout = layered_pipeline_layout(self, "scene_layered_pbr_reactive_pipeline_layout");
        let source = |secondary_uv| {
            let source = specialized_scene_shader_source_from(
                scene_layered_shader_source(SCENE_SHADER, secondary_uv).into(),
                self.froxel.is_some(),
                self.shadow_map.virtual_map.requested(),
            );
            temporal_reactive::scene_transparent_reactive_shader_source(&source)
        };
        let scalar = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_reactive_shader"),
                source: wgpu::ShaderSource::Wgsl(source(false).into()),
            });
        let uv1 = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_reactive_uv1_shader"),
                source: wgpu::ShaderSource::Wgsl(source(true).into()),
            });
        let targets = [
            Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: temporal_reactive::TEMPORAL_REACTIVE_FORMAT,
                blend: Some(temporal_reactive::reactive_union_blend()),
                write_mask: wgpu::ColorWrites::RED,
            }),
        ];
        let reactive = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &scalar,
            "fs_transparent_scene_reactive",
            &targets,
            false,
            false,
            "scene_layered_pbr_reactive_pipeline",
        );
        let reactive_double_sided = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &scalar,
            "fs_transparent_scene_reactive",
            &targets,
            false,
            true,
            "scene_layered_pbr_reactive_double_sided_pipeline",
        );
        let reactive_uv1 = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &uv1,
            "fs_transparent_scene_reactive",
            &targets,
            true,
            false,
            "scene_layered_pbr_reactive_uv1_pipeline",
        );
        let reactive_uv1_double_sided = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &uv1,
            "fs_transparent_scene_reactive",
            &targets,
            true,
            true,
            "scene_layered_pbr_reactive_uv1_double_sided_pipeline",
        );
        let resources = self
            .scene_layered_pbr_resources
            .as_mut()
            .expect("layered resources are initialized");
        resources.reactive = Some(reactive);
        resources.reactive_double_sided = Some(reactive_double_sided);
        resources.reactive_uv1 = Some(reactive_uv1);
        resources.reactive_uv1_double_sided = Some(reactive_uv1_double_sided);
        self.created_pipelines(4);
    }

    pub(super) fn ensure_scene_layered_pbr_weighted_resources(&mut self) {
        let Some(resources) = self.scene_layered_pbr_resources.as_ref() else {
            return;
        };
        if resources.weighted.is_some() {
            return;
        }
        let layout = layered_pipeline_layout(self, "scene_layered_pbr_weighted_pipeline_layout");
        let source = |secondary_uv| {
            let source = specialized_scene_shader_source_from(
                scene_layered_shader_source(SCENE_SHADER, secondary_uv).into(),
                self.froxel.is_some(),
                self.shadow_map.virtual_map.requested(),
            );
            scene_weighted_transparency_shader_source(&source)
        };
        let scalar = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_weighted_shader"),
                source: wgpu::ShaderSource::Wgsl(source(false).into()),
            });
        let uv1 = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene_layered_pbr_weighted_uv1_shader"),
                source: wgpu::ShaderSource::Wgsl(source(true).into()),
            });
        let accumulation_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let revealage_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let targets = [
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(accumulation_blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R16Float,
                blend: Some(revealage_blend),
                write_mask: wgpu::ColorWrites::RED,
            }),
        ];
        let weighted = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &scalar,
            "fs_weighted_transparent_scene",
            &targets,
            false,
            false,
            "scene_layered_pbr_weighted_pipeline",
        );
        let weighted_double_sided = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &scalar,
            "fs_weighted_transparent_scene",
            &targets,
            false,
            true,
            "scene_layered_pbr_weighted_double_sided_pipeline",
        );
        let weighted_uv1 = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &uv1,
            "fs_weighted_transparent_scene",
            &targets,
            true,
            false,
            "scene_layered_pbr_weighted_uv1_pipeline",
        );
        let weighted_uv1_double_sided = create_layered_transparent_pipeline(
            &self.device,
            &layout,
            &uv1,
            "fs_weighted_transparent_scene",
            &targets,
            true,
            true,
            "scene_layered_pbr_weighted_uv1_double_sided_pipeline",
        );
        let resources = self
            .scene_layered_pbr_resources
            .as_mut()
            .expect("layered resources are initialized");
        resources.weighted = Some(weighted);
        resources.weighted_double_sided = Some(weighted_double_sided);
        resources.weighted_uv1 = Some(weighted_uv1);
        resources.weighted_uv1_double_sided = Some(weighted_uv1_double_sided);
        self.created_pipelines(4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_texture_contract_counts_complete_pipeline_layout() {
        assert_eq!(scene_layered_pbr_sampled_texture_requirement(false), 22);
        assert_eq!(scene_layered_pbr_sampled_texture_requirement(true), 24);
    }

    #[test]
    fn ordinary_shader_remains_free_of_layered_bindings_and_calls() {
        assert!(!SCENE_SHADER.contains("layered_material"));
        assert!(!SCENE_SHADER.contains("shade_layered_pbr"));
        assert!(!SCENE_SHADER.contains("layered_iridescence"));
        assert!(!SCENE_SHADER.contains("@group(2) @binding(23)"));
    }

    #[test]
    fn scalar_and_secondary_uv_layered_variants_parse() {
        for secondary_uv in [false, true] {
            let source = scene_layered_shader_source(SCENE_SHADER, secondary_uv);
            wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                panic!("layered scene WGSL (secondary_uv={secondary_uv}) failed: {error}")
            });
            assert_eq!(source.matches("fn shade_layered_pbr(").count(), 1);
            assert_eq!(source.matches("@group(2) @binding(23)").count(), 1);
            assert_eq!(source.matches("@group(2) @binding(19)").count(), 1);
            assert_eq!(source.matches("@group(2) @binding(20)").count(), 1);
            assert_eq!(source.matches("@group(2) @binding(21)").count(), 1);
            assert_eq!(source.matches("var layered_sampler: sampler;").count(), 1);
            assert!(source.contains("fn layered_eval_iridescence("));
            assert_eq!(source.matches("let layered_model_handedness =").count(), 2);
            assert!(source.contains("in.tangent.w * layered_model_handedness"));
            assert_eq!(source.contains("@location(7) secondary_uv"), secondary_uv);
        }
    }

    #[test]
    fn opt_in_capture_debug_modes_parse_without_production_shader_branches() {
        let production = scene_layered_shader_source(SCENE_SHADER, false);
        assert_eq!(
            apply_layered_capture_debug(production.clone(), None),
            production
        );
        for mode in [
            "material",
            "normal",
            "normal-texel",
            "normal-tangent-length",
            "geometric-normal",
            "view-cosine",
            "geometric-view-cosine",
            "iridescence-fresnel",
            "iridescence-ibl-f0",
            "iridescence-prefiltered-env",
            "iridescence-single-spec",
            "iridescence-multiscatter",
            "iridescence-ibl-spec-raw",
            "iridescence-ibl-spec",
            "iridescence-roughness",
        ] {
            let source = apply_layered_capture_debug(production.clone(), Some(mode));
            wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("layered debug mode {mode} failed: {error}"));
            assert!(!source.contains(LAYERED_FINAL_COMPOSITION));
        }
    }

    #[test]
    fn iridescence_uniforms_preserve_reversed_range_and_both_uv_transforms() {
        let binding = |source_texture_index, tex_coord, rotation| {
            Some(crate::models::MaterialTextureBinding {
                source_texture_index,
                source_image_index: 0,
                runtime_texture_idx: Some(source_texture_index + 1),
                transform: crate::models::MaterialTextureTransform {
                    offset: [0.1, 0.2],
                    rotation,
                    scale: [0.4, 0.5],
                    tex_coord,
                },
            })
        };
        let material = crate::models::MaterialLayeredPbr {
            iridescence_authored: true,
            iridescence_factor: 0.82,
            iridescence_texture: binding(3, 1, 0.3),
            iridescence_ior: 1.42,
            iridescence_thickness_minimum: 620.0,
            iridescence_thickness_maximum: 180.0,
            iridescence_thickness_texture: binding(4, 0, -0.2),
            ..Default::default()
        };
        let mut usable = [false; LAYERED_TEXTURE_COUNT];
        usable[IRIDESCENCE_FACTOR_TEXTURE] = true;
        usable[IRIDESCENCE_THICKNESS_TEXTURE] = true;
        let uniforms = SceneLayeredPbrUniforms::new(material, usable);

        assert_eq!(uniforms.iridescence, [0.82, 1.42, 620.0, 180.0]);
        assert_eq!(uniforms.iridescence_factor_uv, [0.1, 0.2, 0.4, 0.5]);
        assert_eq!(uniforms.iridescence_factor_rotation[2..], [1.0, 1.0]);
        assert_eq!(uniforms.iridescence_thickness_rotation[2..], [1.0, 0.0]);
    }

    #[test]
    fn specialized_opaque_and_transparent_variants_parse() {
        for secondary_uv in [false, true] {
            for clustered in [false, true] {
                for virtual_shadows in [false, true] {
                    let source = specialized_scene_shader_source_from(
                        scene_layered_shader_source(SCENE_SHADER, secondary_uv).into(),
                        clustered,
                        virtual_shadows,
                    );
                    wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                        panic!(
                            "layered scene WGSL (secondary_uv={secondary_uv}, \
                             clustered={clustered}, virtual_shadows={virtual_shadows}) failed: \
                             {error}"
                        )
                    });
                }
            }

            let source = specialized_scene_shader_source_from(
                scene_layered_shader_source(SCENE_SHADER, secondary_uv).into(),
                false,
                false,
            );
            for (label, source) in [
                (
                    "reactive",
                    temporal_reactive::scene_transparent_reactive_shader_source(&source),
                ),
                (
                    "weighted",
                    scene_weighted_transparency_shader_source(&source),
                ),
            ] {
                wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                    panic!("layered {label} WGSL (secondary_uv={secondary_uv}) failed: {error}")
                });
            }
        }
    }
}
