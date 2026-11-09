//! The mobile scene shading: on mobile GPUs the scene shader's
//! `shade_main_scene` is replaced by `shaders/scene_mobile.wgsl`, a lighter
//! forward base pass (as Unreal's mobile renderer and Unity URP have). Same
//! inputs, outputs, lights and shadows (the sun's with the 4-tap tent
//! filter only, chosen when the shader is built); the full shader's normal-variance
//! roughness, BRDF LUT (an analytic fit instead), coverage-dithered masks, cloud
//! shadows and firefly caps are left out. Those cost a mobile GPU most of the
//! main pass through register pressure, not arithmetic (Pixel 4a, Adreno
//! 618).
//!
//! Applied to the main scene shader before its froxel and virtual-shadow
//! specializations (the mobile function keeps their markers), so the scene
//! pipelines, their prepassed variant and the GPU-driven variant all shade
//! with it. The layered-material and layered-refraction shaders keep the full
//! shading.

use std::borrow::Cow;

const MOBILE_SHADE_MAIN_SCENE: &str = include_str!("shaders/scene_mobile.wgsl");
const SHADE_MAIN_SCENE: &str =
    "fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {";
const AFTER_SHADE_MAIN_SCENE: &str = "@fragment\nfn fs_main_scene(";
const SAMPLE_CASCADE: &str =
    "fn sample_cascade(cascade: i32, shadow_uv: vec2<f32>, depth_ref: f32) -> f32 {";
const MOBILE_SAMPLE_CASCADE: &str =
    "fn sample_cascade(cascade: i32, shadow_uv: vec2<f32>, depth_ref: f32) -> f32 {
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_tex));
    return pcf_tent4(cascade, shadow_uv, texel, depth_ref) * 0.25;
}
";

/// Mobile shading on Android by default; `BLOOM_MOBILE_SHADING=0|1` overrides
/// it (to compare the two on a device, or to try the mobile shading on a
/// desktop GPU).
pub(super) fn mobile_shading_enabled() -> bool {
    match std::env::var("BLOOM_MOBILE_SHADING").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => cfg!(target_os = "android"),
    }
}

/// `source` with the mobile `shade_main_scene` in place of the full one, when
/// mobile shading is on.
pub(super) fn scene_shader(source: &'static str) -> Cow<'static, str> {
    if !mobile_shading_enabled() {
        return source.into();
    }
    with_mobile_shading(source).into()
}

fn with_mobile_shading(source: &str) -> String {
    let start = source
        .find(SHADE_MAIN_SCENE)
        .expect("scene shader missing shade_main_scene");
    let end = start
        + source[start..]
            .find(AFTER_SHADE_MAIN_SCENE)
            .expect("scene shader missing fs_main_scene after shade_main_scene");
    let source = format!(
        "{}{}{}",
        &source[..start],
        MOBILE_SHADE_MAIN_SCENE,
        &source[end..]
    );
    // The sun shadow takes the 4-tap tent only. The 16-tap Poisson disc
    // behind the runtime filter switch costs a mobile GPU even when it never
    // runs (its code is the pass's register budget): 10 ms of a 36 ms main
    // pass on a Pixel 4a.
    let start = source
        .find(SAMPLE_CASCADE)
        .expect("scene shader missing sample_cascade");
    let end = start
        + source[start..]
            .find("\n}\n")
            .expect("sample_cascade without an end")
        + "\n}\n".len();
    format!(
        "{}{}{}",
        &source[..start],
        MOBILE_SAMPLE_CASCADE,
        &source[end..]
    )
}

#[cfg(test)]
mod tests {
    use super::super::shaders::SCENE_SHADER;
    use super::*;

    #[test]
    fn mobile_scene_shader_validates_and_replaces_the_full_shading() {
        let mobile = with_mobile_shading(SCENE_SHADER);
        let module = wgpu::naga::front::wgsl::parse_str(&mobile)
            .unwrap_or_else(|error| panic!("mobile scene WGSL failed: {error:?}"));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("mobile scene WGSL invalid: {error:?}"));
        assert_eq!(mobile.matches(SHADE_MAIN_SCENE).count(), 1);
        // the full shading's LUT and variance terms are gone
        assert!(!mobile.contains("brdf_lut_tex, brdf_lut_samp, vec2<f32>(n_dot_v_ibl"));
        assert!(!mobile.contains("sigma2_toksvig"));
        // the shadow filter is the tent alone
        assert!(!mobile.contains("pcf_poisson16(cascade"));
        assert_eq!(mobile.matches(SAMPLE_CASCADE).count(), 1);
        // the markers the prepass, froxel and VSM variants rely on
        for marker in [
            "//PREPASS_STRIP_BEGIN",
            "//PREPASS_STRIP_END",
            "//IBL_STRIP_BEGIN",
            "//IBL_STRIP_END",
            "// BEGIN-POINT-LIGHT-LOOP",
            "// END-POINT-LIGHT-LOOP",
            "pl.color.w * atten2",
        ] {
            assert_eq!(mobile.matches(marker).count(), 1, "{marker}");
        }
    }

    #[test]
    fn mobile_scene_shader_takes_the_ibl_strip_diagnostic() {
        // As BLOOM_SKIP=ibl does in specialized_scene_shader_source_from.
        let mobile = with_mobile_shading(SCENE_SHADER);
        let begin = mobile.find("//IBL_STRIP_BEGIN").unwrap();
        let end = mobile.find("//IBL_STRIP_END").unwrap() + "//IBL_STRIP_END".len();
        let stripped = format!(
            "{}\n    let ibl_diffuse = base_color * (1.0 - metallic) * occlusion\n        * lighting.camera_pos.w * 0.35;\n    let ibl_spec = vec3<f32>(0.0);\n    let ibl_clearcoat = vec3<f32>(0.0);\n{}",
            &mobile[..begin],
            &mobile[end..]
        );
        assert!(!stripped.contains("env_sample_lod(reflect(-v, n)"));
        wgpu::naga::front::wgsl::parse_str(&stripped)
            .unwrap_or_else(|error| panic!("IBL-stripped mobile WGSL failed: {error:?}"));
    }

    #[test]
    fn mobile_scene_shader_takes_the_froxel_and_gpu_driven_variants() {
        let mobile = with_mobile_shading(SCENE_SHADER);
        let clustered = super::super::froxel::clustered_scene_shader(&mobile);
        wgpu::naga::front::wgsl::parse_str(&clustered)
            .unwrap_or_else(|error| panic!("clustered mobile WGSL failed: {error:?}"));
        let gpu = super::super::gpu_driven::make_gpu_scene_shader(&mobile);
        wgpu::naga::front::wgsl::parse_str(&gpu)
            .unwrap_or_else(|error| panic!("GPU-driven mobile WGSL failed: {error:?}"));
    }
}
