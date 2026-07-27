// Shader sources are grouped by pass cluster (2000-line file policy);
// everything re-exports from here so call sites keep `shaders::NAME`.

mod core;
pub(super) use core::{scene_refractive_shader_source, SCENE_SHADER, SHADER_2D, SHADER_3D};
#[cfg(test)]
mod core_tests;
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
    CARD_CAPTURE_WGSL, CARD_LIGHT_WGSL, SDF_BAKE_WGSL, SDF_CLIPMAP_BAKE_WGSL, WSRC_BAKE_HW_WGSL,
    WSRC_BAKE_WGSL,
};
mod ssgi;
pub(super) use ssgi::{
    PROBE_HELPERS_WGSL, SSGI_PROBE_PLACE_WGSL, SSGI_PROBE_RESOLVE_WGSL, SSGI_PROBE_TEMPORAL_WGSL,
    SSGI_PROBE_TRACE_HW_WGSL, SSGI_PROBE_TRACE_SDF_WGSL, SSGI_PROBE_TRACE_SW_WGSL, SSR_SHADER_WGSL,
    SSR_TEMPORAL_SHADER_WGSL,
};
mod pt;
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
        ] {
            assert_eq!(
                source.matches("rayQueryProceed").count(),
                source.matches("if (BLOOM_RAY_QUERY_NEEDS_PROCEED)").count(),
            );
        }
    }
}
mod post;
pub(super) use post::{
    BLOOM_SHADER_WGSL, COMPOSITE_SHADER_WGSL, DOF_SHADER_WGSL, EXPOSURE_SHADER_WGSL,
    MOTION_BLUR_SHADER_WGSL, RCAS_SHADER_WGSL, SCENE_COMPOSE_SHADER_WGSL, SSS_SHADER_WGSL,
    TAA_SHADER_WGSL, UPSCALE_SHADER_WGSL,
};
