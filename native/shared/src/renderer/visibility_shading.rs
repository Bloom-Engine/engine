//! Full-PBR visibility shading for the opt-in #27 A/B path.
//!
//! The shader is derived from the exact GPU-driven forward source and calls
//! its `shade_main_scene` function after reconstructing the fragment inputs.
//! This keeps lighting and material evolution shared instead of maintaining a
//! second deferred copy of Bloom's PBR implementation.

pub(super) fn make_shader(gpu_scene_source: &str) -> String {
    let source = specialize_visibility_derivatives(&strip_prepass_discard(gpu_scene_source));
    format!(
        "{source}\n{}\n{}\n{VISIBILITY_SHADE_WGSL}",
        super::visibility_buffer::RECONSTRUCTION_WGSL,
        super::visibility_buffer::GEOMETRY_WGSL,
    )
}

#[cfg(feature = "models3d")]
pub(crate) fn make_virtual_shader(gpu_scene_source: &str) -> String {
    let source = specialize_visibility_derivatives(&strip_prepass_discard(gpu_scene_source));
    format!(
        "{source}\n{}\n{}\n{}\n{}\n{}\n{}",
        super::visibility_buffer::RECONSTRUCTION_WGSL,
        super::visibility_buffer::GEOMETRY_WGSL,
        VIRTUAL_RENDER_ABI_WGSL,
        VIRTUAL_DECODE_WGSL,
        VIRTUAL_VISIBILITY_RECONSTRUCT_WGSL,
        VIRTUAL_VISIBILITY_SHADE_WGSL,
    )
}

#[cfg(feature = "models3d")]
const VIRTUAL_RENDER_ABI_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/render_abi.wgsl");
#[cfg(feature = "models3d")]
const VIRTUAL_DECODE_WGSL: &str = include_str!("../../shaders/virtual_geometry/decode.wgsl");
#[cfg(feature = "models3d")]
const VIRTUAL_VISIBILITY_RECONSTRUCT_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/visibility_reconstruct.wgsl");
#[cfg(feature = "models3d")]
const VIRTUAL_VISIBILITY_SHADE_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/visibility_shading.wgsl");

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
    create_pipeline_for_entry(
        device,
        layout,
        shader,
        "visibility_buffer_pbr_pipeline",
        "vs_visibility_shade",
        "fs_visibility_shade",
    )
}

#[cfg(feature = "models3d")]
pub(crate) fn create_virtual_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    create_pipeline_for_entry(
        device,
        layout,
        shader,
        "virtual_geometry_visibility_pbr_pipeline",
        "vs_virtual_visibility_shade",
        "fs_virtual_visibility_shade",
    )
}

fn create_pipeline_for_entry(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &'static str,
    vertex_entry: &'static str,
    fragment_entry: &'static str,
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
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: super::DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
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

/// Replace derivative-dependent operations only inside the visibility copy of
/// `shade_main_scene`. A fullscreen visibility pass cannot use `dpdx`/`dpdy`
/// on reconstructed attributes: the adjacent quad lane can belong to another
/// primitive (or the background), unlike raster helper invocations which
/// extrapolate the current triangle. The fragment entry point supplies the
/// equivalent same-triangle quad gradients below.
fn specialize_visibility_derivatives(source: &str) -> String {
    const START: &str =
        "fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {";
    const END: &str = "\n@fragment\nfn fs_main_scene(";
    let begin = source
        .find(START)
        .expect("GPU scene shader keeps the shared shade_main_scene entry point");
    let end = source[begin..]
        .find(END)
        .map(|offset| begin + offset)
        .expect("GPU scene shader keeps fs_main_scene after shared shading");
    let mut body = source[begin..end].to_string();

    replace_once(
        &mut body,
        START,
        "fn shade_main_scene(\n    in: VertexOutputScene,\n    front_facing: bool,\n    visibility_gradients: BloomVisibilityGradients,\n) -> SceneOut {",
    );
    replace_once(
        &mut body,
        "bloom_sample_normal_raw_bias(material, in.uv, 1.0 + lod_bias)",
        "bloom_visibility_sample_normal_raw_grad_bias(\n        material, in.uv, visibility_gradients.uv_dx, visibility_gradients.uv_dy,\n        1.0 + lod_bias,\n    )",
    );
    replace_once(
        &mut body,
        "let tbn_dp1 = dpdx(in.world_pos);",
        "let tbn_dp1 = visibility_gradients.world_dx;",
    );
    replace_once(
        &mut body,
        "let tbn_dp2 = dpdy(in.world_pos);",
        "let tbn_dp2 = visibility_gradients.world_dy;",
    );
    replace_once(
        &mut body,
        "let tbn_duv1 = dpdx(in.uv);",
        "let tbn_duv1 = visibility_gradients.uv_dx;",
    );
    replace_once(
        &mut body,
        "let tbn_duv2 = dpdy(in.uv);",
        "let tbn_duv2 = visibility_gradients.uv_dy;",
    );
    replace_once(
        &mut body,
        "bloom_sample_raw_bias(material.texture_ids_0.x, material.sampler_ids_0.x, in.uv, lod_bias)",
        "bloom_visibility_sample_raw_grad_bias(\n        material.texture_ids_0.x, material.sampler_ids_0.x, in.uv,\n        visibility_gradients.uv_dx, visibility_gradients.uv_dy, lod_bias,\n    )",
    );
    replace_once(
        &mut body,
        "bloom_sample_raw(material.texture_ids_0.z, material.sampler_ids_0.z, in.uv)",
        "bloom_visibility_sample_raw_grad(\n        material.texture_ids_0.z, material.sampler_ids_0.z, in.uv,\n        visibility_gradients.uv_dx, visibility_gradients.uv_dy,\n    )",
    );
    replace_once(
        &mut body,
        "let nm_dx = dpdx(n);\n    let nm_dy = dpdy(n);",
        "let visibility_normal_x = bloom_visibility_surface_normal(\n        visibility_gradients.normal_x, visibility_gradients.tangent_x,\n        visibility_gradients.uv_x, material, visibility_gradients, 1.0 + lod_bias,\n    );\n    let visibility_normal_y = bloom_visibility_surface_normal(\n        visibility_gradients.normal_y, visibility_gradients.tangent_y,\n        visibility_gradients.uv_y, material, visibility_gradients, 1.0 + lod_bias,\n    );\n    let nm_dx = (visibility_normal_x - n) * visibility_gradients.x_sign;\n    let nm_dy = (visibility_normal_y - n) * visibility_gradients.y_sign;",
    );
    replace_once(
        &mut body,
        "bloom_sample_raw_bias(material.texture_ids_0.w, material.sampler_ids_0.w, in.uv, 0.0)",
        "bloom_visibility_sample_raw_grad(\n        material.texture_ids_0.w, material.sampler_ids_0.w, in.uv,\n        visibility_gradients.uv_dx, visibility_gradients.uv_dy,\n    )",
    );
    replace_once(
        &mut body,
        "bloom_sample_raw(material.texture_ids_1.x, material.sampler_ids_1.x, in.uv)",
        "bloom_visibility_sample_raw_grad(\n        material.texture_ids_1.x, material.sampler_ids_1.x, in.uv,\n        visibility_gradients.uv_dx, visibility_gradients.uv_dy,\n    )",
    );

    // The original raster entry points follow `shade_main_scene`; retaining
    // them would leave invalid two-argument calls to the specialized function.
    // They are not part of the visibility pipeline, whose fullscreen entry
    // points are appended below.
    format!("{}\n{VISIBILITY_GRADIENT_WGSL}\n{}", &source[..begin], body)
}

fn replace_once(source: &mut String, needle: &str, replacement: &str) {
    assert_eq!(
        source.matches(needle).count(),
        1,
        "visibility shader specialization anchor changed: {needle}"
    );
    *source = source.replacen(needle, replacement, 1);
}

const VISIBILITY_GRADIENT_WGSL: &str = r#"
struct BloomVisibilityGradients {
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
    normal_x: vec3<f32>,
    normal_y: vec3<f32>,
    tangent_x: vec4<f32>,
    tangent_y: vec4<f32>,
    uv_x: vec2<f32>,
    uv_y: vec2<f32>,
    x_sign: f32,
    y_sign: f32,
};

fn bloom_visibility_sample_raw_grad(
    texture_id: u32,
    sampler_id: u32,
    uv: vec2<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
) -> vec4<f32> {
    let texture_slot = bloom_texture_slot(texture_id);
    let sampler_slot = bloom_sampler_slot(sampler_id);
    return textureSampleGrad(
        global_textures[texture_slot],
        global_samplers[sampler_slot],
        uv,
        uv_dx,
        uv_dy,
    );
}

fn bloom_visibility_sample_raw_grad_bias(
    texture_id: u32,
    sampler_id: u32,
    uv: vec2<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
    bias: f32,
) -> vec4<f32> {
    let gradient_scale = exp2(bias);
    return bloom_visibility_sample_raw_grad(
        texture_id,
        sampler_id,
        uv,
        uv_dx * gradient_scale,
        uv_dy * gradient_scale,
    );
}

fn bloom_visibility_sample_normal_raw_grad_bias(
    material_record: GlobalMaterialRecord,
    uv: vec2<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
    bias: f32,
) -> vec4<f32> {
    let texture_slot = bloom_texture_slot(material_record.texture_ids_0.y);
    if (texture_slot == 0u) {
        return vec4<f32>(128.0 / 255.0, 128.0 / 255.0, 1.0, 0.0);
    }
    let sampler_slot = bloom_sampler_slot(material_record.sampler_ids_0.y);
    let gradient_scale = exp2(bias);
    return textureSampleGrad(
        global_textures[texture_slot],
        global_samplers[sampler_slot],
        uv,
        uv_dx * gradient_scale,
        uv_dy * gradient_scale,
    );
}

// Evaluate the adjacent helper lane from the same triangle. This reproduces
// raster normal derivatives even when the visible adjacent pixel belongs to a
// different primitive, and includes normal-map variation in the specular-AA
// kernel rather than silently reducing it to geometric-normal variation.
fn bloom_visibility_surface_normal(
    geometric_normal: vec3<f32>,
    tangent: vec4<f32>,
    uv: vec2<f32>,
    material_record: GlobalMaterialRecord,
    gradients: BloomVisibilityGradients,
    normal_lod_bias: f32,
) -> vec3<f32> {
    var normal = normalize(geometric_normal);
    let normal_sample4 = bloom_visibility_sample_normal_raw_grad_bias(
        material_record,
        uv,
        gradients.uv_dx,
        gradients.uv_dy,
        normal_lod_bias,
    );
    let normal_raw = normal_sample4.xyz * 2.0 - 1.0;
    let normal_sample = normal_raw * inverseSqrt(clamp(dot(normal_raw, normal_raw), 0.01, 1.0));
    if (dot(tangent.xyz, tangent.xyz) > 0.0001) {
        let tangent_normalized = normalize(tangent.xyz);
        let tangent_ortho = normalize(
            tangent_normalized - normal * dot(normal, tangent_normalized),
        );
        let bitangent = cross(normal, tangent_ortho) * tangent.w;
        normal = normalize(
            tangent_ortho * normal_sample.x
                + bitangent * normal_sample.y
                + normal * normal_sample.z,
        );
    } else {
        let tbn = compute_tbn(
            gradients.world_dx,
            gradients.world_dy,
            gradients.uv_dx,
            gradients.uv_dy,
            normal,
        );
        normal = normalize(tbn * normal_sample);
    }
    return normal;
}
"#;

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
    // Virtual IDs are owned by the disjoint raw-page shading pass. Never
    // reinterpret their local selected-record index as a compatibility draw.
    if (visibility.virtual_geometry) { discard; }
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
    let current_clip = bloom_interpolate4(clip0, clip1, clip2, bary);
    // Fragment derivatives operate on 2x2 quads. Select the other lane in
    // each axis and extrapolate this same triangle there, exactly as raster
    // helper invocations do even when that lane is outside triangle coverage.
    let x_step = select(-1.0, 1.0, (u32(pixel.x) & 1u) == 0u);
    let y_step = select(-1.0, 1.0, (u32(pixel.y) & 1u) == 0u);
    let point_x_ndc = vec2<f32>(
        (in.position.x + x_step) / f32(dimensions.x) * 2.0 - 1.0,
        point_ndc.y,
    );
    let point_y_ndc = vec2<f32>(
        point_ndc.x,
        1.0 - (in.position.y + y_step) / f32(dimensions.y) * 2.0,
    );
    let bary_x = bloom_perspective_barycentrics(point_x_ndc, clip0, clip1, clip2);
    let bary_y = bloom_perspective_barycentrics(point_y_ndc, clip0, clip1, clip2);

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

    let fragment_normal = bloom_interpolate3(normal0, normal1, normal2, bary);
    let fragment_uv = bloom_interpolate2(vertex0.uv, vertex1.uv, vertex2.uv, bary);
    let fragment_world = bloom_interpolate3(world0.xyz, world1.xyz, world2.xyz, bary);
    let fragment_tangent = bloom_interpolate4(tangent0, tangent1, tangent2, bary);
    let normal_x = bloom_interpolate3(normal0, normal1, normal2, bary_x);
    let normal_y = bloom_interpolate3(normal0, normal1, normal2, bary_y);
    let tangent_x = bloom_interpolate4(tangent0, tangent1, tangent2, bary_x);
    let tangent_y = bloom_interpolate4(tangent0, tangent1, tangent2, bary_y);
    let uv_x = bloom_interpolate2(vertex0.uv, vertex1.uv, vertex2.uv, bary_x);
    let uv_y = bloom_interpolate2(vertex0.uv, vertex1.uv, vertex2.uv, bary_y);
    let world_x = bloom_interpolate3(world0.xyz, world1.xyz, world2.xyz, bary_x);
    let world_y = bloom_interpolate3(world0.xyz, world1.xyz, world2.xyz, bary_y);
    let visibility_gradients = BloomVisibilityGradients(
        (world_x - fragment_world) * x_step,
        (world_y - fragment_world) * y_step,
        (uv_x - fragment_uv) * x_step,
        (uv_y - fragment_uv) * y_step,
        normal_x,
        normal_y,
        tangent_x,
        tangent_y,
        uv_x,
        uv_y,
        x_step,
        y_step,
    );

    var fragment: VertexOutputScene;
    fragment.clip_position = in.position;
    fragment.normal = fragment_normal;
    fragment.color = bloom_interpolate4(vertex0.color, vertex1.color, vertex2.color, bary)
        * draw.uniforms.model_tint;
    fragment.uv = fragment_uv;
    fragment.world_pos = fragment_world;
    fragment.tangent = fragment_tangent;
    fragment.curr_clip = current_clip;
    fragment.prev_clip = bloom_interpolate4(
        draw.uniforms.prev_mvp * local0,
        draw.uniforms.prev_mvp * local1,
        draw.uniforms.prev_mvp * local2,
        bary,
    );
    fragment.material_id = draw.draw.w;
    fragment.draw_flags = bitcast<u32>(draw.bounds_min.w);
    return shade_main_scene(fragment, visibility.front_facing, visibility_gradients);
}
"#;

/// Shade mode's depth pipeline writes packed IDs while it primes depth. This
/// retains the depth prepass alpha test and removes a depth-equal traversal.
pub(super) fn make_visibility_depth_shader(gpu_scene_source: &str) -> String {
    const DEPTH_HEADER: &str = concat!(
        "fn fs_depth_prepass(in: VertexOutputScene, ",
        "@builtin(front_facing) front_facing: bool) {"
    );
    const DEPTH_OUTPUT_HEADER: &str = concat!(
        "fn fs_depth_prepass(\n",
        "    in: VertexOutputScene,\n",
        "    @builtin(primitive_index) primitive_id: u32,\n",
        "    @builtin(front_facing) front_facing: bool,\n",
        ") -> @location(0) vec2<u32> {"
    );
    const DEPTH_END: &str = concat!(
        "        if (!survives) { discard; }\n",
        "    }\n",
        "}\n\n",
        "fn shade_main_scene"
    );
    const DEPTH_OUTPUT_END: &str = concat!(
        "        if (!survives) { discard; }\n",
        "    }\n",
        "    if ((in.draw_flags & 2u) == 0u) {\n",
        "        return vec2<u32>(0xffffffffu, 0xffffffffu);\n",
        "    }\n",
        "    let face = select(0u, 0x80000000u, front_facing);\n",
        "    return vec2<u32>(in.draw_id, primitive_id | face);\n",
        "}\n\n",
        "fn shade_main_scene"
    );
    let source = super::visibility_buffer::add_visibility_draw_id(gpu_scene_source);
    assert_eq!(source.matches(DEPTH_HEADER).count(), 1);
    assert_eq!(source.matches(DEPTH_END).count(), 1);
    format!(
        "enable primitive_index;\n{}",
        source
            .replace(DEPTH_HEADER, DEPTH_OUTPUT_HEADER)
            .replace(DEPTH_END, DEPTH_OUTPUT_END)
    )
}

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
        assert!(shade.contains(
            "return shade_main_scene(fragment, visibility.front_facing, visibility_gradients);"
        ));
        assert!(shade.contains("if (visibility.virtual_geometry) { discard; }"));
        assert!(shade.contains("textureSampleGrad("));
        assert!(!shade[shade
            .find("fn shade_main_scene(")
            .expect("specialized shade function")
            ..shade
                .find("// Bloom packed visibility-buffer ABI")
                .expect("reconstruction header after shared shading")]
            .contains("dpdx("));
        assert!(!shade.contains("fn fs_main_scene("));
        assert!(compatibility.contains("(in.draw_flags & 2u) != 0u"));
    }

    #[cfg(feature = "models3d")]
    #[test]
    fn virtual_shading_reuses_the_authoritative_pbr_and_all_mrts() {
        let gpu =
            super::super::gpu_driven::make_gpu_scene_shader(super::super::shaders::SCENE_SHADER);
        let shade = make_virtual_shader(&gpu);
        wgpu::naga::front::wgsl::parse_str(&shade)
            .unwrap_or_else(|error| panic!("virtual visibility PBR WGSL failed: {error:?}"));
        assert!(shade.contains("if (!visibility.virtual_geometry) { discard; }"));
        assert!(shade.contains("instance.normal_rows[0].xyz"));
        assert!(shade.contains("BLOOM_VIRTUAL_INSTANCE_NEGATIVE_DETERMINANT"));
        assert!(shade.contains("virtual_frame.previous_view_projection"));
        assert!(shade.contains("fragment.material_id = selection.material_id;"));
        assert!(shade.contains("* instance.model_tint;"));
        assert!(shade.contains(
            "return shade_main_scene(\n        fragment,\n        visibility.front_facing,"
        ));
        assert!(shade.contains("struct SceneOut"));
        assert!(shade.contains("@location(3) albedo: vec4<f32>"));
    }
}
