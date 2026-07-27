use super::core::{scene_refractive_shader_source, SCENE_SHADER};

#[test]
fn scene_vertex_shader_preserves_the_missing_tangent_sentinel() {
    assert!(SCENE_SHADER.contains("fn safe_scene_tangent("));
    assert_eq!(SCENE_SHADER.matches("safe_scene_tangent(").count(), 3);
    assert!(SCENE_SHADER.contains("length_squared > 1e-8"));
    assert!(!SCENE_SHADER.contains("normalize(tan4.xyz)"));
    assert!(!SCENE_SHADER.contains("normalize((u.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz)"));
}

#[test]
fn native_and_folded_refractive_variants_parse_without_touching_ordinary_shader() {
    assert!(!SCENE_SHADER.contains("fs_refractive_scene"));
    for (folded, screen_space_reflections) in [(false, false), (false, true), (true, false)] {
        for secondary_uv in [false, true] {
            let source = scene_refractive_shader_source(
                SCENE_SHADER,
                folded,
                screen_space_reflections,
                secondary_uv,
            );
            wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                panic!(
                    "{}{} refractive WGSL failed: {error}",
                    if folded {
                        "folded"
                    } else if screen_space_reflections {
                        "native reflection hierarchy"
                    } else {
                        "native environment-only"
                    },
                    if secondary_uv { " UV1" } else { "" },
                )
            });
            assert!(source.contains("fn fs_refractive_scene"));
            assert_eq!(
                source.contains("@group(4) @binding(0) var refractive_scene_color_tex"),
                !folded
            );
            assert_eq!(
                source.contains("fn refractive_screen_reflection"),
                screen_space_reflections
            );
            assert_eq!(
                source.contains("@group(4) @binding(3) var<uniform> refractive_reflection"),
                screen_space_reflections
            );
            assert_eq!(
                source.contains("@group(4) @binding(4) var refractive_planar_tex"),
                screen_space_reflections
            );
            assert_eq!(
                source.contains("fn refractive_planar_sample"),
                screen_space_reflections
            );
            assert_eq!(
                source.contains("if (planar_reflected.a < 0.0)"),
                screen_space_reflections,
                "only non-planar glass may pay for the bounded screen march"
            );
            assert_eq!(
                source.contains("@location(8) secondary_uv"),
                secondary_uv,
                "UV1 must exist only in the lazy refractive variant"
            );
        }
    }
}
