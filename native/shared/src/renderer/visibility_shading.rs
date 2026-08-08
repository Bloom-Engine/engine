//! Full-PBR visibility shading for the opt-in #27 A/B path.
//!
//! The shader is derived from the exact GPU-driven forward source and calls
//! its `shade_main_scene` function after reconstructing the fragment inputs.
//! This keeps lighting and material evolution shared instead of maintaining a
//! second deferred copy of Bloom's PBR implementation.

pub(super) fn make_shader(gpu_scene_source: &str) -> String {
    let source = strip_prepass_discard(gpu_scene_source);
    format!(
        "{source}\n{}\n{}\n{VISIBILITY_SHADE_WGSL}",
        super::visibility_buffer::RECONSTRUCTION_WGSL,
        super::visibility_buffer::GEOMETRY_WGSL,
    )
}

pub(super) fn make_forward_compatibility_shader(gpu_scene_source: &str) -> String {
    const ANCHOR: &str = concat!(
        "fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {\n",
        "    let material = bloom_material_record(in.material_id);"
    );
    const REPLACEMENT: &str = concat!(
        "fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {\n",
        "    let material = bloom_material_record(in.material_id);\n",
        "    if ((in.draw_flags & 2u) != 0u) { discard; }"
    );
    assert_eq!(gpu_scene_source.matches(ANCHOR).count(), 1);
    gpu_scene_source.replace(ANCHOR, REPLACEMENT)
}

pub(super) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("visibility_buffer_pbr_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage(1),
            storage(2),
            storage(3),
        ],
    })
}

pub(super) fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    #[cfg(lean_mrt)]
    let targets = &[
        Some(target(super::HDR_FORMAT, true)),
        None,
        Some(target(super::VELOCITY_FORMAT, false)),
        None,
    ];
    #[cfg(not(lean_mrt))]
    let targets = &[
        Some(target(super::HDR_FORMAT, true)),
        Some(target(super::MATERIAL_FORMAT, false)),
        Some(target(super::VELOCITY_FORMAT, false)),
        Some(target(wgpu::TextureFormat::Rgba8Unorm, false)),
    ];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("visibility_buffer_pbr_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_visibility_shade"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_visibility_shade"),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn target(format: wgpu::TextureFormat, alpha_blend: bool) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: alpha_blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

pub(super) fn load_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    }
}

fn strip_prepass_discard(source: &str) -> String {
    match (
        source.find("//PREPASS_STRIP_BEGIN"),
        source.find("//PREPASS_STRIP_END"),
    ) {
        (Some(begin), Some(end)) if end > begin => {
            let suffix = end + "//PREPASS_STRIP_END".len();
            format!("{}{}", &source[..begin], &source[suffix..])
        }
        _ => source.to_string(),
    }
}

const VISIBILITY_SHADE_WGSL: &str = r#"
struct VisibilityVertexOut {
    @builtin(position) position: vec4<f32>,
};

struct VisibilityVertexTable { records: array<BloomPackedVertex3D>, };
struct VisibilityIndexTable { values: array<u32>, };

@group(4) @binding(0) var visibility_shade_ids: texture_2d<u32>;
@group(4) @binding(1) var<storage, read> visibility_shade_vertices: VisibilityVertexTable;
@group(4) @binding(2) var<storage, read> visibility_shade_indices: VisibilityIndexTable;
@group(4) @binding(3) var<storage, read> visibility_shade_draws: GpuDrawTable;

@vertex
fn vs_visibility_shade(@builtin(vertex_index) vertex_index: u32) -> VisibilityVertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VisibilityVertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

fn visibility_shade_fault() -> SceneOut {
    return SceneOut(
        vec4<f32>(8.0, 0.0, 8.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0),
        vec4<f32>(1.0, 0.0, 1.0, 0.0),
    );
}

@fragment
fn fs_visibility_shade(in: VisibilityVertexOut) -> SceneOut {
    let pixel = vec2<i32>(in.position.xy);
    let raw_visibility = textureLoad(visibility_shade_ids, pixel, 0).xy;
    if (!bloom_visibility_valid(raw_visibility)) { discard; }
    let visibility = bloom_decode_visibility(raw_visibility);
    if (visibility.draw_id >= arrayLength(&visibility_shade_draws.records)) {
        return visibility_shade_fault();
    }
    let draw = visibility_shade_draws.records[visibility.draw_id];
    let primitive_offset = visibility.primitive_id * 3u;
    if (primitive_offset + 2u >= draw.draw.x) {
        return visibility_shade_fault();
    }
    let first_index = draw.draw.y + primitive_offset;
    if (first_index + 2u >= arrayLength(&visibility_shade_indices.values)) {
        return visibility_shade_fault();
    }
    let base_vertex = bitcast<i32>(draw.draw.z);
    let signed0 = i32(visibility_shade_indices.values[first_index]) + base_vertex;
    let signed1 = i32(visibility_shade_indices.values[first_index + 1u]) + base_vertex;
    let signed2 = i32(visibility_shade_indices.values[first_index + 2u]) + base_vertex;
    if (signed0 < 0 || signed1 < 0 || signed2 < 0) {
        return visibility_shade_fault();
    }
    let index0 = u32(signed0);
    let index1 = u32(signed1);
    let index2 = u32(signed2);
    let vertex_count = arrayLength(&visibility_shade_vertices.records);
    if (index0 >= vertex_count || index1 >= vertex_count || index2 >= vertex_count) {
        return visibility_shade_fault();
    }

    let vertex0 = bloom_decode_vertex3d(visibility_shade_vertices.records[index0]);
    let vertex1 = bloom_decode_vertex3d(visibility_shade_vertices.records[index1]);
    let vertex2 = bloom_decode_vertex3d(visibility_shade_vertices.records[index2]);
    let local0 = vec4<f32>(vertex0.position, 1.0);
    let local1 = vec4<f32>(vertex1.position, 1.0);
    let local2 = vec4<f32>(vertex2.position, 1.0);
    let clip0 = draw.uniforms.mvp * local0;
    let clip1 = draw.uniforms.mvp * local1;
    let clip2 = draw.uniforms.mvp * local2;
    let dimensions = textureDimensions(visibility_shade_ids);
    let point_ndc = vec2<f32>(
        in.position.x / f32(dimensions.x) * 2.0 - 1.0,
        1.0 - in.position.y / f32(dimensions.y) * 2.0,
    );
    let bary = bloom_perspective_barycentrics(point_ndc, clip0, clip1, clip2);

    let world0 = draw.uniforms.model * local0;
    let world1 = draw.uniforms.model * local1;
    let world2 = draw.uniforms.model * local2;
    let normal0 = normalize((draw.uniforms.model * vec4<f32>(vertex0.normal, 0.0)).xyz);
    let normal1 = normalize((draw.uniforms.model * vec4<f32>(vertex1.normal, 0.0)).xyz);
    let normal2 = normalize((draw.uniforms.model * vec4<f32>(vertex2.normal, 0.0)).xyz);
    let tangent0 = vec4<f32>(
        safe_scene_tangent((draw.uniforms.model * vec4<f32>(vertex0.tangent.xyz, 0.0)).xyz),
        vertex0.tangent.w,
    );
    let tangent1 = vec4<f32>(
        safe_scene_tangent((draw.uniforms.model * vec4<f32>(vertex1.tangent.xyz, 0.0)).xyz),
        vertex1.tangent.w,
    );
    let tangent2 = vec4<f32>(
        safe_scene_tangent((draw.uniforms.model * vec4<f32>(vertex2.tangent.xyz, 0.0)).xyz),
        vertex2.tangent.w,
    );

    var fragment: VertexOutputScene;
    fragment.clip_position = in.position;
    fragment.normal = bloom_interpolate3(normal0, normal1, normal2, bary);
    fragment.color = bloom_interpolate4(vertex0.color, vertex1.color, vertex2.color, bary)
        * draw.uniforms.model_tint;
    fragment.uv = bloom_interpolate2(vertex0.uv, vertex1.uv, vertex2.uv, bary);
    fragment.world_pos = bloom_interpolate3(world0.xyz, world1.xyz, world2.xyz, bary);
    fragment.tangent = bloom_interpolate4(tangent0, tangent1, tangent2, bary);
    fragment.curr_clip = bloom_interpolate4(clip0, clip1, clip2, bary);
    fragment.prev_clip = bloom_interpolate4(
        draw.uniforms.prev_mvp * local0,
        draw.uniforms.prev_mvp * local1,
        draw.uniforms.prev_mvp * local2,
        bary,
    );
    fragment.material_id = draw.draw.w;
    fragment.draw_flags = bitcast<u32>(draw.bounds_min.w);
    return shade_main_scene(fragment, visibility.front_facing);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_shading_and_forward_compatibility_variants_parse() {
        let gpu =
            super::super::gpu_driven::make_gpu_scene_shader(super::super::shaders::SCENE_SHADER);
        let shade = make_shader(&gpu);
        wgpu::naga::front::wgsl::parse_str(&shade)
            .unwrap_or_else(|error| panic!("visibility PBR WGSL failed to parse: {error:?}"));
        let compatibility = make_forward_compatibility_shader(&gpu);
        wgpu::naga::front::wgsl::parse_str(&compatibility)
            .unwrap_or_else(|error| panic!("visibility compatibility WGSL failed: {error:?}"));
        assert!(shade.contains("return shade_main_scene(fragment, visibility.front_facing);"));
        assert!(compatibility.contains("(in.draw_flags & 2u) != 0u"));
    }
}
