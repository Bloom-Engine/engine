//! Lazy weighted-blended transparency shader variants.

/// Add a dedicated weighted-blended transparency entry point to the
/// specialized scene shader. The ordinary scene module/pipelines remain
/// unchanged; this source is compiled lazily only after the weighted route
/// becomes active.
pub(in crate::renderer) fn scene_weighted_transparency_shader_source(
    base_scene_shader: &str,
) -> String {
    let mut source = String::with_capacity(base_scene_shader.len() + 1_400);
    source.push_str(base_scene_shader);
    source.push_str(
        r#"

struct WeightedTransparencyOut {
    @location(0) accumulation: vec4<f32>,
    @location(1) revealage: f32,
};

@fragment
fn fs_weighted_transparent_scene(
    in: VertexOutputScene,
    @builtin(front_facing) front_facing: bool,
) -> WeightedTransparencyOut {
    let shaded = shade_main_scene(in, front_facing).color;
    let alpha = clamp(shaded.a, 0.0, 1.0);

    // Bounded McGuire/Bavoil-style weighted OIT. Nearer fragments receive
    // more influence without the unbounded exponential weights that can
    // overflow rgba16float on dense particle/glass layers. With one layer
    // the resolve is algebraically identical to conventional alpha blend.
    let depth = clamp(in.clip_position.z, 0.0, 1.0);
    let depth_weight = 0.1 + 0.9 * pow(1.0 - depth, 3.0);
    let weighted_alpha = alpha * depth_weight;
    let finite_color = select(vec3<f32>(0.0), shaded.rgb, shaded.rgb == shaded.rgb);
    return WeightedTransparencyOut(
        vec4<f32>(finite_color * weighted_alpha, weighted_alpha),
        alpha,
    );
}
"#,
    );
    source
}

/// Full-screen resolve for weighted-blended transparency. Revealage is the
/// multiplicative product of `(1 - alpha)`; accumulation stores
/// `(radiance * alpha * weight, alpha * weight)`.
pub(in crate::renderer) const WEIGHTED_TRANSPARENCY_RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var accumulation_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

@vertex
fn vs_weighted_transparency_resolve(
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_weighted_transparency_resolve(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let accumulation = textureLoad(accumulation_tex, pixel, 0);
    let revealage = clamp(textureLoad(revealage_tex, pixel, 0).r, 0.0, 1.0);
    let opacity = 1.0 - revealage;
    let color = accumulation.rgb / max(accumulation.a, 0.00001);
    let finite_color = select(vec3<f32>(0.0), color, color == color);
    return vec4<f32>(finite_color, opacity);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::shaders::core::SCENE_SHADER;

    #[test]
    fn weighted_transparency_variants_parse_without_touching_ordinary_shader() {
        assert!(!SCENE_SHADER.contains("fs_weighted_transparent_scene"));
        let source = scene_weighted_transparency_shader_source(SCENE_SHADER);
        wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("weighted transparency WGSL failed: {error}"));
        wgpu::naga::front::wgsl::parse_str(WEIGHTED_TRANSPARENCY_RESOLVE_SHADER)
            .unwrap_or_else(|error| panic!("weighted transparency resolve WGSL failed: {error}"));
        assert!(source.contains("fn fs_weighted_transparent_scene"));
    }
}
