// Shader sources are grouped by pass cluster (2000-line file policy);
// everything re-exports from here so call sites keep `shaders::NAME`.

mod core;
pub(super) use core::{scene_refractive_shader_source, SCENE_SHADER};
#[cfg(test)]
mod core_tests;
mod legacy;
pub(super) use legacy::{SHADER_2D, SHADER_3D};
mod weighted;
pub(super) use weighted::{
    scene_weighted_transparency_shader_source, WEIGHTED_TRANSPARENCY_RESOLVE_SHADER,
};
mod env;
pub(super) use env::{
    AERIAL_PERSPECTIVE_SHADER_WGSL, EQUIRECT_FROM_SKY_VIEW_SHADER_WGSL, PREFILTER_SHADER_WGSL,
    PROCEDURAL_SKY_SHADER_WGSL, REFLECT_SCENE_WGSL, SKY_SHADER_WGSL, SKY_VIEW_LUT_SHADER_WGSL,
};
mod ao;
pub(super) use ao::{
    HIZ_DOWNSAMPLE_SHADER_WGSL, HIZ_LINEARIZE_SHADER_WGSL, SSAO_BLUR_SHADER_WGSL, SSAO_SHADER_WGSL,
};
mod gi;
pub(super) use gi::{
    CARD_CAPTURE_WGSL, CARD_LIGHT_HW_WGSL, CARD_LIGHT_WGSL, SDF_BAKE_WGSL, SDF_CLIPMAP_BAKE_WGSL,
    WSRC_BAKE_HW_WGSL, WSRC_BAKE_WGSL,
};
mod ssgi;
pub(super) use ssgi::{
    PROBE_HELPERS_WGSL, SSGI_PROBE_PLACE_WGSL, SSGI_PROBE_RESOLVE_WGSL, SSGI_PROBE_TEMPORAL_WGSL,
    SSGI_PROBE_TRACE_HW_WGSL, SSGI_PROBE_TRACE_SDF_WGSL, SSGI_PROBE_TRACE_SW_WGSL,
};
mod ssr;
pub(super) use ssr::{SSR_SHADER_WGSL, SSR_TEMPORAL_SHADER_WGSL};
mod pt;
#[cfg(test)]
pub(super) use pt::PT_KERNEL_WGSL;
pub(super) use pt::{pt_fault_constants, pt_kernel_variant, PT_ATROUS_WGSL, PT_SKIN_WGSL};

/// Naga's Metal lowering completes a query in `rayQueryInitialize`; its
/// non-modern `rayQueryProceed` only reads a `ready` flag that never clears.
/// Metal must skip that loop, while DX12/Vulkan retain canonical WGSL flow.
pub(super) fn ray_query_backend_variant(device: &wgpu::Device) -> &'static str {
    ray_query_backend_variant_for(device.adapter_info().backend)
}

fn ray_query_backend_variant_for(backend: wgpu::Backend) -> &'static str {
    if backend == wgpu::Backend::Metal {
        "const BLOOM_RAY_QUERY_NEEDS_PROCEED: bool = false;\n"
    } else {
        "const BLOOM_RAY_QUERY_NEEDS_PROCEED: bool = true;\n"
    }
}

#[cfg(test)]
mod ray_query_variant_tests {
    use super::*;

    #[test]
    fn metal_skips_proceed_and_explicit_query_backends_keep_it() {
        assert!(ray_query_backend_variant_for(wgpu::Backend::Metal).contains("false"));
        assert!(ray_query_backend_variant_for(wgpu::Backend::Dx12).contains("true"));
        assert!(ray_query_backend_variant_for(wgpu::Backend::Vulkan).contains("true"));
    }

    #[test]
    fn every_hardware_query_loop_has_the_backend_guard() {
        let production_pt = pt_kernel_variant(false);
        for source in [
            production_pt.as_ref(),
            SSGI_PROBE_TRACE_HW_WGSL,
            WSRC_BAKE_HW_WGSL,
            CARD_LIGHT_HW_WGSL,
        ] {
            assert_eq!(
                source.matches("rayQueryProceed").count(),
                source.matches("if (BLOOM_RAY_QUERY_NEEDS_PROCEED)").count(),
            );
        }
    }

    #[test]
    fn wsrc_corner_wrap_is_shared_by_software_and_hardware_bakes() {
        assert!(PROBE_HELPERS_WGSL.contains("select(oct_size - 1, 0, padded.x == padded_max)"));
        assert!(PROBE_HELPERS_WGSL.contains("select(oct_size - 1, 0, padded.y == padded_max)"));
        for source in [WSRC_BAKE_WGSL, WSRC_BAKE_HW_WGSL] {
            assert_eq!(
                source.matches("wsrc_real_octel(vec2<i32>(lid.xy))").count(),
                1
            );
            assert!(!source.contains("Corner — nearest-inside"));
        }

        let software = format!("{PROBE_HELPERS_WGSL}{WSRC_BAKE_WGSL}");
        wgpu::naga::front::wgsl::parse_str(&software)
            .unwrap_or_else(|error| panic!("software WSRC WGSL failed: {error:?}"));
        let hardware = format!(
            "enable wgpu_ray_query;\n\
             const BLOOM_RAY_QUERY_NEEDS_PROCEED: bool = false;\n\
             {PROBE_HELPERS_WGSL}{WSRC_BAKE_HW_WGSL}"
        );
        wgpu::naga::front::wgsl::parse_str(&hardware)
            .unwrap_or_else(|error| panic!("hardware WSRC WGSL failed: {error:?}"));
        let card_light_hardware = format!(
            "enable wgpu_ray_query;\n\
             const BLOOM_RAY_QUERY_NEEDS_PROCEED: bool = false;\n\
             {CARD_LIGHT_HW_WGSL}"
        );
        wgpu::naga::front::wgsl::parse_str(&card_light_hardware)
            .unwrap_or_else(|error| panic!("hardware card-light WGSL failed: {error:?}"));
    }

    #[test]
    fn hardware_indirect_sun_visibility_is_baked_in_world_space() {
        assert!(SSGI_PROBE_TRACE_HW_WGSL.contains("card_radiance_atlas"));
        assert!(SSGI_PROBE_TRACE_HW_WGSL.contains(
            "facade-sized light fragments when that sample changes"
        ));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("stable_visibility"));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("textureSampleCompareLevel"));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("probe_sun_visibility"));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("shadow_query"));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("hw_wsrc_sun_visibility"));
        assert!(SSGI_PROBE_TRACE_HW_WGSL.contains("return hw_gi_cap(u.sky_color.xyz * up * up);"));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("hw_wsrc_sample(origin_ws + dir_ws * max_t"));
        assert!(CARD_LIGHT_HW_WGSL.contains("pos_ws + sun_dir * 0.02"));
        assert!(CARD_LIGHT_HW_WGSL.contains("coarse_visibility"));
        assert!(!CARD_LIGHT_HW_WGSL.contains("textureSampleCompareLevel"));
        assert!(CARD_LIGHT_WGSL.contains("never manufacture direct sunlight in the GI feed"));
        assert!(WSRC_BAKE_HW_WGSL
            .contains("cache_probe_sun_visibility = hw_bake_trace_sun_visibility(probe_pos);"));
        assert!(WSRC_BAKE_HW_WGSL.contains("vec4<f32>(radiance, cache_probe_sun_visibility)"));
        assert!(WSRC_BAKE_WGSL.contains("vec4<f32>(radiance, shadow)"));
        assert!(WSRC_BAKE_HW_WGSL.contains("pos_ws + sun_dir * 0.02"));
        assert!(!WSRC_BAKE_HW_WGSL.contains("textureSampleCompareLevel"));
    }

    #[test]
    fn mesh_cards_reuse_material_alpha_for_captured_hit_normals() {
        assert!(CARD_CAPTURE_WGSL.contains("let normal_oct = card_oct_encode(in.normal_os);"));
        assert!(CARD_CAPTURE_WGSL.contains("vec4<f32>(albedo, normal_oct.x)"));
        assert!(CARD_CAPTURE_WGSL.contains("normal_oct.y"));
        assert!(SSGI_PROBE_TRACE_HW_WGSL.contains("vec2<f32>(albedo_sample.a, emissive_sample.a)"));
        assert!(SSGI_PROBE_TRACE_HW_WGSL.contains("if (dot(normal_ws, incoming_dir_ws) > 0.0)"));
    }

    #[test]
    fn ssgi_sampling_is_temporally_stable_and_world_point_owned() {
        let traces = [
            SSGI_PROBE_TRACE_SW_WGSL,
            SSGI_PROBE_TRACE_HW_WGSL,
            SSGI_PROBE_TRACE_SDF_WGSL,
        ];
        for source in traces {
            assert!(source.contains("let dir_ws = octel_direction(lid.xy);"));
            assert!(!source.contains("textureLoad(prev_history"));
            assert!(!source.contains("octel_jitter("));
        }
        assert!(!SSGI_PROBE_PLACE_WGSL.contains("let frame = u.params.x"));
        assert!(!SSGI_PROBE_PLACE_WGSL.contains("let jx ="));
        assert!(SSGI_PROBE_TRACE_HW_WGSL.contains("card_radiance_atlas"));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("textureSampleCompareLevel"));
        assert!(!SSGI_PROBE_TRACE_HW_WGSL.contains("let view_z = -(u.view"));
        assert!(CARD_LIGHT_WGSL
            .contains("for (var cascade: i32 = 0; cascade < 3; cascade = cascade + 1)"));
        assert!(!CARD_LIGHT_WGSL.contains("let view_z = -(u.view_matrix"));
        assert!(SSGI_PROBE_TEMPORAL_WGSL.contains("mix(u.params.x, 0.65, motion_refresh)"));
        assert!(SSGI_PROBE_RESOLVE_WGSL.contains("w_corner * w_depth * w_normal"));
        assert_eq!(
            SSGI_PROBE_TEMPORAL_WGSL
                .matches("textureLoad(history_in")
                .count(),
            1,
            "motion refresh must not add temporal-history texture reads"
        );

        for source in [
            format!("{PROBE_HELPERS_WGSL}{SSGI_PROBE_PLACE_WGSL}"),
            format!("{PROBE_HELPERS_WGSL}{SSGI_PROBE_TRACE_SW_WGSL}"),
            format!("{PROBE_HELPERS_WGSL}{SSGI_PROBE_TRACE_SDF_WGSL}"),
            format!("{PROBE_HELPERS_WGSL}{CARD_LIGHT_WGSL}"),
        ] {
            wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("stable SSGI WGSL failed: {error:?}"));
        }
        let hardware = format!(
            "enable wgpu_ray_query;\n\
             const BLOOM_RAY_QUERY_NEEDS_PROCEED: bool = false;\n\
             {PROBE_HELPERS_WGSL}{SSGI_PROBE_TRACE_HW_WGSL}"
        );
        wgpu::naga::front::wgsl::parse_str(&hardware)
            .unwrap_or_else(|error| panic!("stable hardware SSGI WGSL failed: {error:?}"));
    }

    #[test]
    fn ssr_derivative_normal_faces_the_camera() {
        wgpu::naga::front::wgsl::parse_str(SSR_SHADER_WGSL)
            .unwrap_or_else(|error| panic!("SSR WGSL failed: {error:?}"));
        assert!(SSR_SHADER_WGSL.contains("let n_raw = normalize(cross(dy, dx));"));
        assert!(!SSR_SHADER_WGSL.contains("let n = normalize(cross(dx, dy));"));
        assert!(SSR_SHADER_WGSL.contains("dot(n_raw, v) >= 0.0"));
    }

    #[test]
    fn ssr_uses_stable_roughness_ownership_and_hit_provenance() {
        wgpu::naga::front::wgsl::parse_str(SSR_SHADER_WGSL)
            .unwrap_or_else(|error| panic!("SSR WGSL failed: {error:?}"));
        wgpu::naga::front::wgsl::parse_str(SSR_TEMPORAL_SHADER_WGSL)
            .unwrap_or_else(|error| panic!("SSR temporal WGSL failed: {error:?}"));

        let ownership = "1.0 - smoothstep(0.45, 0.70, roughness)";
        assert!(SSR_SHADER_WGSL.contains(ownership));
        assert!(SCENE_SHADER.contains(ownership));
        assert!(SSR_SHADER_WGSL.contains("luma / 4.0"));
        assert!(SSR_TEMPORAL_SHADER_WGSL.contains("provenance_disocclusion"));
        assert!(SSR_TEMPORAL_SHADER_WGSL.contains("current_hit == history_hit"));
    }
}
mod post;
pub(super) use post::{
    BLOOM_SHADER_WGSL, COMPOSITE_SHADER_WGSL, DOF_SHADER_WGSL, EXPOSURE_SHADER_WGSL,
    MOTION_BLUR_SHADER_WGSL, RCAS_SHADER_WGSL, SCENE_COMPOSE_SHADER_WGSL, SSS_SHADER_WGSL,
    TAA_SHADER_WGSL, UPSCALE_SHADER_WGSL,
};
