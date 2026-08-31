//! Lazy, bounded colored shadows for imported physical transmission.
//!
//! The established 2048² opaque/MASK CSM remains the authority for opaque
//! visibility.  When (and only when) a physical-transmission material exists,
//! this module adds one lower-resolution nearest-layer transmittance + depth
//! pair per cascade.  A post-opaque fullscreen pass subtracts the portion of
//! primary-sun radiance absorbed by that layer.  Keeping the correction
//! separate means ordinary scene/material pipelines, layouts, and fragments
//! remain byte-identical in opaque-only applications.

use super::*;

pub(super) const TRANSMITTED_SHADOW_MAP_SIZE: u32 = 1024;
pub(super) const TRANSMITTED_SHADOW_COLOR_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba8Unorm;
pub(super) const TRANSMITTED_SHADOW_DEPTH_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Depth16Unorm;
pub(super) const TRANSMITTED_SHADOW_PERSISTENT_BYTES: u64 = TRANSMITTED_SHADOW_MAP_SIZE as u64
    * TRANSMITTED_SHADOW_MAP_SIZE as u64
    * crate::shadows::NUM_CASCADES as u64
    * (4 + 2);

pub(super) fn transmitted_shadows_enabled() -> bool {
    std::env::var("BLOOM_TRANSMITTED_SHADOWS")
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "disabled"
            )
        })
        .unwrap_or(true)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransmittedShadowDrawUniforms {
    light_vp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    tint: [f32; 4],
    /// x = cached-skinned joint offset. Remaining lanes are reserved.
    misc: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransmittedShadowResolveUniforms {
    inv_vp: [[f32; 4]; 4],
    cascade_vps: [[[f32; 4]; 4]; crate::shadows::NUM_CASCADES],
    camera_pos: [f32; 4],
    cascade_splits: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    wind: [f32; 4],
    cloud: [f32; 4],
    /// xy = render extent, zw reserved.
    target_size: [f32; 4],
}

const TRANSMITTED_SHADOW_CASTER_WGSL: &str = r#"
struct DrawUniforms {
    light_vp: mat4x4<f32>,
    model: mat4x4<f32>,
    tint: vec4<f32>,
    misc: vec4<f32>,
};

struct MaterialFactors {
    metal_rough: vec4<f32>,
    emissive: vec4<f32>,
    spec_gloss: vec4<f32>,
};

struct TransmissionFactors {
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    transmission_uv: vec4<f32>,
    transmission_rotation: vec4<f32>,
    thickness_uv: vec4<f32>,
    thickness_rotation: vec4<f32>,
};

struct JointMatrices {
    matrices: array<mat4x4<f32>, 1024>,
};

@group(0) @binding(0) var<uniform> u: DrawUniforms;
@group(1) @binding(0) var base_color_tex: texture_2d<f32>;
@group(1) @binding(1) var base_color_samp: sampler;
@group(1) @binding(4) var mr_tex: texture_2d<f32>;
@group(1) @binding(5) var mr_samp: sampler;
@group(1) @binding(8) var<uniform> material: MaterialFactors;
@group(1) @binding(11) var transmission_tex: texture_2d<f32>;
@group(1) @binding(12) var transmission_samp: sampler;
@group(1) @binding(13) var thickness_tex: texture_2d<f32>;
@group(1) @binding(14) var thickness_samp: sampler;
@group(1) @binding(15) var<uniform> physical: TransmissionFactors;
@group(2) @binding(0) var opaque_depth: texture_depth_2d;
@group(3) @binding(0) var<uniform> joints: JointMatrices;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joint_indices: vec4<f32>,
    @location(5) weights: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) model_scale: f32,
};

fn physical_uv(
    uv: vec2<f32>,
    offset_scale: vec4<f32>,
    rotation: vec2<f32>,
) -> vec2<f32> {
    let scaled = uv * offset_scale.zw;
    return offset_scale.xy + vec2<f32>(
        rotation.x * scaled.x - rotation.y * scaled.y,
        rotation.y * scaled.x + rotation.x * scaled.y,
    );
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var local = vec4<f32>(v.position, 1.0);
    let weight_sum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
    var world = u.model * local;
    if (weight_sum > 0.01) {
        let j0 = u32(v.joint_indices.x + u.misc.x);
        let j1 = u32(v.joint_indices.y + u.misc.x);
        let j2 = u32(v.joint_indices.z + u.misc.x);
        let j3 = u32(v.joint_indices.w + u.misc.x);
        world =
            joints.matrices[j0] * local * v.weights.x
            + joints.matrices[j1] * local * v.weights.y
            + joints.matrices[j2] * local * v.weights.z
            + joints.matrices[j3] * local * v.weights.w;
    }

    var out: VertexOut;
    out.position = u.light_vp * world;
    out.uv = v.uv;
    out.color = v.color * u.tint;
    out.model_scale = (
        length(u.model[0].xyz)
        + length(u.model[1].xyz)
        + length(u.model[2].xyz)
    ) / 3.0;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // The color target is half the opaque CSM extent.  Compare against the
    // matching opaque texel explicitly: color/depth attachments of different
    // sizes cannot share a render pass, and hidden glass must not tint an
    // opaque blocker's shadow.
    let blocker_dims = textureDimensions(opaque_depth);
    let color_dims = vec2<f32>(
        f32(TRANSMITTED_SHADOW_MAP_SIZE),
        f32(TRANSMITTED_SHADOW_MAP_SIZE),
    );
    let blocker_pixel = clamp(
        vec2<i32>(
            floor(in.position.xy * vec2<f32>(blocker_dims) / color_dims)
        ),
        vec2<i32>(0),
        vec2<i32>(blocker_dims) - vec2<i32>(1),
    );
    let blocker = textureLoad(opaque_depth, blocker_pixel, 0);
    if (blocker + 0.00075 < in.position.z) {
        discard;
    }

    let base_texel = textureSample(base_color_tex, base_color_samp, in.uv);
    let base_color = base_texel.rgb * in.color.rgb;
    let base_alpha = clamp(base_texel.a * in.color.a, 0.0, 1.0);
    let alpha_mode = material.metal_rough.w;
    if (alpha_mode > 0.0 && base_alpha < alpha_mode) {
        discard;
    }

    let mr = textureSample(mr_tex, mr_samp, in.uv);
    let metallic = select(
        clamp(material.metal_rough.x, 0.0, 1.0),
        clamp(material.metal_rough.x * mr.b, 0.0, 1.0),
        material.metal_rough.z > 0.5,
    );
    let transmission_uv = physical_uv(
        in.uv,
        physical.transmission_uv,
        physical.transmission_rotation.xy,
    );
    let transmission_texture = select(
        1.0,
        textureSample(
            transmission_tex,
            transmission_samp,
            transmission_uv,
        ).r,
        physical.transmission.w > 0.5,
    );
    let transmission_weight = clamp(
        physical.transmission.x * transmission_texture * (1.0 - metallic),
        0.0,
        1.0,
    );

    let thickness_uv = physical_uv(
        in.uv,
        physical.thickness_uv,
        physical.thickness_rotation.xy,
    );
    let thickness_texture = select(
        1.0,
        textureSample(thickness_tex, thickness_samp, thickness_uv).g,
        physical.transmission_rotation.z > 0.5,
    );
    let thickness_world = max(
        physical.transmission.z * thickness_texture * in.model_scale,
        0.0,
    );
    var absorption = vec3<f32>(1.0);
    if (physical.attenuation.w > 0.0 && thickness_world > 0.0) {
        absorption = pow(
            max(physical.attenuation.rgb, vec3<f32>(0.000001)),
            vec3<f32>(thickness_world / physical.attenuation.w),
        );
    }

    // Bound the directional-light energy exactly like the camera-facing
    // transmission lobe: metallic content does not transmit, normal-incidence
    // dielectric Fresnel reflects instead of passing, and BLEND alpha denotes
    // geometric coverage (uncovered area remains fully lit).
    let ior = max(physical.transmission.y, 1.0);
    let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let physical_transmittance = clamp(
        base_color * absorption * transmission_weight * (1.0 - f0),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let coverage = select(1.0, base_alpha, alpha_mode < 0.0);
    let transmittance = mix(vec3<f32>(1.0), physical_transmittance, coverage);
    return vec4<f32>(transmittance, 1.0);
}
"#;

fn transmitted_shadow_caster_shader_source(secondary_uv: bool) -> String {
    let mut source = TRANSMITTED_SHADOW_CASTER_WGSL.replace(
        "TRANSMITTED_SHADOW_MAP_SIZE",
        &format!("{TRANSMITTED_SHADOW_MAP_SIZE}.0"),
    );
    if !secondary_uv {
        return source;
    }
    source = source.replacen(
        "    @location(5) weights: vec4<f32>,\n};",
        "    @location(5) weights: vec4<f32>,\n\
         @location(7) secondary_uv: vec2<f32>,\n\
         };",
        1,
    );
    source = source.replacen(
        "    @location(2) model_scale: f32,\n};",
        "    @location(2) model_scale: f32,\n\
         @location(3) secondary_uv: vec2<f32>,\n\
         };",
        1,
    );
    source = source.replacen(
        "    out.uv = v.uv;",
        "    out.uv = v.uv;\n    out.secondary_uv = v.secondary_uv;",
        1,
    );
    source = source.replacen(
        "    let transmission_uv = physical_uv(\n        in.uv,",
        "    let transmission_source_uv = select(\n\
             in.uv,\n\
             in.secondary_uv,\n\
             physical.transmission_rotation.w > 0.5,\n\
         );\n\
         let transmission_uv = physical_uv(\n\
             transmission_source_uv,",
        1,
    );
    source = source.replacen(
        "    let thickness_uv = physical_uv(\n        in.uv,",
        "    let thickness_source_uv = select(\n\
             in.uv,\n\
             in.secondary_uv,\n\
             physical.thickness_rotation.z > 0.5,\n\
         );\n\
         let thickness_uv = physical_uv(\n\
             thickness_source_uv,",
        1,
    );
    assert!(
        source.contains("@location(3) secondary_uv"),
        "transmitted-shadow vertex ABI changed; UV1 injection must be updated"
    );
    source
}

const TRANSMITTED_SHADOW_RESOLVE_WGSL: &str = concat!(
    include_str!("../../shaders/common/clouds.wgsl"),
    r#"
struct ResolveUniforms {
    inv_vp: mat4x4<f32>,
    cascade_vps: array<mat4x4<f32>, 3>,
    camera_pos: vec4<f32>,
    cascade_splits: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    wind: vec4<f32>,
    cloud: vec4<f32>,
    target_size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var scene_albedo: texture_2d<f32>;
@group(0) @binding(3) var scene_material: texture_2d<f32>;
@group(0) @binding(4) var trans_color_0: texture_2d<f32>;
@group(0) @binding(5) var trans_color_1: texture_2d<f32>;
@group(0) @binding(6) var trans_color_2: texture_2d<f32>;
@group(0) @binding(7) var trans_depth_0: texture_depth_2d;
@group(0) @binding(8) var trans_depth_1: texture_depth_2d;
@group(0) @binding(9) var trans_depth_2: texture_depth_2d;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var out: VertexOut;
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

fn load_trans_depth(cascade: i32, pixel: vec2<i32>) -> f32 {
    if (cascade == 0) {
        return textureLoad(trans_depth_0, pixel, 0);
    }
    if (cascade == 1) {
        return textureLoad(trans_depth_1, pixel, 0);
    }
    return textureLoad(trans_depth_2, pixel, 0);
}

fn load_trans_color(cascade: i32, pixel: vec2<i32>) -> vec3<f32> {
    if (cascade == 0) {
        return textureLoad(trans_color_0, pixel, 0).rgb;
    }
    if (cascade == 1) {
        return textureLoad(trans_color_1, pixel, 0).rgb;
    }
    return textureLoad(trans_color_2, pixel, 0).rgb;
}

fn transmitted_visibility(
    cascade: i32,
    world_pos: vec3<f32>,
    receiver_bias: f32,
) -> vec3<f32> {
    let clip = u.cascade_vps[cascade] * vec4<f32>(world_pos, 1.0);
    let ndc = clip.xyz / max(abs(clip.w), 0.000001);
    if (
        ndc.x < -1.0 || ndc.x > 1.0
        || ndc.y < -1.0 || ndc.y > 1.0
        || ndc.z < 0.0 || ndc.z > 1.0
    ) {
        return vec3<f32>(1.0);
    }

    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let dims = vec2<i32>(textureDimensions(trans_depth_0));
    let texel = uv * vec2<f32>(dims) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(texel));
    let fraction = fract(texel);
    var taps = array<vec3<f32>, 4>();
    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(0, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, 1),
        vec2<i32>(1, 1),
    );
    for (var i = 0; i < 4; i = i + 1) {
        let pixel = clamp(base + offsets[i], vec2<i32>(0), dims - vec2<i32>(1));
        let caster_depth = load_trans_depth(cascade, pixel);
        taps[i] = select(
            vec3<f32>(1.0),
            load_trans_color(cascade, pixel),
            ndc.z > caster_depth + receiver_bias,
        );
    }
    let row0 = mix(taps[0], taps[1], fraction.x);
    let row1 = mix(taps[2], taps[3], fraction.x);
    return mix(row0, row1, fraction.y);
}

fn sample_transmitted_shadow(world_pos: vec3<f32>) -> vec3<f32> {
    let distance_to_camera = length(world_pos - u.camera_pos.xyz);
    var cascade = 2;
    if (distance_to_camera <= u.cascade_splits.x) {
        cascade = 0;
    } else if (distance_to_camera <= u.cascade_splits.y) {
        cascade = 1;
    }
    let receiver_bias = 0.0015;
    let value = transmitted_visibility(cascade, world_pos, receiver_bias);

    var split_near = 0.0;
    var split_far = u.cascade_splits.x;
    if (cascade == 1) {
        split_near = u.cascade_splits.x;
        split_far = u.cascade_splits.y;
    } else if (cascade == 2) {
        split_near = u.cascade_splits.y;
        split_far = u.cascade_splits.z;
    }
    let blend_zone = max((split_far - split_near) * 0.1, 0.0001);
    let distance_to_edge = split_far - distance_to_camera;
    if (distance_to_edge < blend_zone && cascade < 2) {
        let next_value = transmitted_visibility(
            cascade + 1,
            world_pos,
            receiver_bias,
        );
        return mix(next_value, value, clamp(distance_to_edge / blend_zone, 0.0, 1.0));
    }
    return value;
}

const PI: f32 = 3.14159265;

fn f_schlick(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {
    let fc = pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * fc;
}

fn d_ggx(n_dot_h: f32, alpha2: f32) -> f32 {
    let x = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * x * x);
}

fn v_smith_ggx_correlated(
    n_dot_l: f32,
    n_dot_v: f32,
    alpha2: f32,
) -> f32 {
    let ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    let ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    return 0.5 / max(ggxv + ggxl, 0.00001);
}

fn primary_direct(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + l);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let alpha = max(roughness * roughness, 0.002025);
    let alpha2 = alpha * alpha;
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let f = f_schlick(v_dot_h, f0);
    let d = d_ggx(n_dot_h, alpha2);
    let vis = v_smith_ggx_correlated(n_dot_l, n_dot_v, alpha2);
    let specular_raw = d * vis * f;

    let direct_luma = dot(specular_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let direct_cap = 1.0 / (1.0 + direct_luma / 0.3);
    let universal_damp = smoothstep(0.05, 0.75, roughness);
    let dielectric_factor = 1.0 - metallic;
    let dielectric_direct_amp = mix(0.08, 1.0, smoothstep(0.15, 0.75, roughness));
    let direct_spec_scale = mix(1.0, dielectric_direct_amp, dielectric_factor);
    let specular = specular_raw * direct_spec_scale * universal_damp * direct_cap;
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;
    return (diffuse + specular)
        * u.sun_color.rgb
        * u.sun_dir.w
        * n_dot_l;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let depth = textureLoad(scene_depth, pixel, 0);
    if (depth >= 0.999999) {
        return vec4<f32>(0.0);
    }
    let uv = (in.position.xy + vec2<f32>(0.5)) / u.target_size.xy;
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0,
        depth,
        1.0,
    );
    let world_h = u.inv_vp * ndc;
    let world_pos = world_h.xyz / max(abs(world_h.w), 0.000001);

    var n = normalize(cross(dpdy(world_pos), dpdx(world_pos)));
    let v = normalize(u.camera_pos.xyz - world_pos);
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let albedo_sample = textureLoad(scene_albedo, pixel, 0);
    let material_sample = textureLoad(scene_material, pixel, 0);
    let base_color = clamp(albedo_sample.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let metallic = clamp(material_sample.r, 0.0, 1.0);
    let roughness = clamp(material_sample.g, 0.045, 1.0);
    let transmittance = sample_transmitted_shadow(world_pos);
    if (all(transmittance >= vec3<f32>(0.999))) {
        return vec4<f32>(0.0);
    }

    // albedo.a stores 1 - opaque CSM visibility.  Scale only the sun samples
    // that survived opaque blockers; the 3% artistic bounce floor remains an
    // indirect term and must not be recolored by glass.
    let opaque_visibility = clamp(1.0 - albedo_sample.a, 0.0, 1.0);
    let light_dir = normalize(u.sun_dir.xyz);
    let cloud_visibility = cloud_shadow_at(
        world_pos,
        light_dir,
        u.wind.xy,
        u.wind.w,
        u.cloud,
    );
    let direct = primary_direct(
        n,
        v,
        light_dir,
        base_color,
        metallic,
        roughness,
    ) * opaque_visibility * cloud_visibility;
    let correction = direct * (transmittance - vec3<f32>(1.0));
    return vec4<f32>(correction, 0.0);
}
"#
);

fn create_transmitted_shadow_caster_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    secondary_uv: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(if secondary_uv {
            "transmitted_shadow_caster_uv1_shader"
        } else {
            "transmitted_shadow_caster_shader"
        }),
        source: wgpu::ShaderSource::Wgsl(
            transmitted_shadow_caster_shader_source(secondary_uv).into(),
        ),
    });
    let base_layouts = [Vertex3D::desc()];
    let secondary_layouts;
    let vertex_layouts = if secondary_uv {
        secondary_layouts = [Vertex3D::desc(), secondary_uv_desc()];
        &secondary_layouts[..]
    } else {
        &base_layouts[..]
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if secondary_uv {
            "transmitted_shadow_caster_uv1_pipeline"
        } else {
            "transmitted_shadow_caster_pipeline"
        }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: TRANSMITTED_SHADOW_COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            // Nearest-layer depth removes the back surface of closed volumes
            // while still accepting either orientation of a pane.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: TRANSMITTED_SHADOW_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: 1,
                slope_scale: 1.0,
                clamp: 0.0,
            },
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) struct TransmittedShadowResources {
    _color_textures: [wgpu::Texture; crate::shadows::NUM_CASCADES],
    pub(super) color_views: [wgpu::TextureView; crate::shadows::NUM_CASCADES],
    _depth_textures: [wgpu::Texture; crate::shadows::NUM_CASCADES],
    pub(super) depth_views: [wgpu::TextureView; crate::shadows::NUM_CASCADES],
    draw_uniform_buffer: wgpu::Buffer,
    draw_uniform_bind_group: wgpu::BindGroup,
    blocker_bind_groups: [wgpu::BindGroup; crate::shadows::NUM_CASCADES],
    caster_pipeline_layout: wgpu::PipelineLayout,
    caster_pipeline: wgpu::RenderPipeline,
    caster_uv1_pipeline: Option<wgpu::RenderPipeline>,
    resolve_uniform_buffer: wgpu::Buffer,
    resolve_layout: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_bind_group: Option<wgpu::BindGroup>,
    rendered_signatures: [u64; crate::shadows::NUM_CASCADES],
    rendered_vps: Option<[[[f32; 4]; 4]; crate::shadows::NUM_CASCADES]>,
    blocker_generations: [u64; crate::shadows::NUM_CASCADES],
    pub(super) last_caster_count: u32,
    warned_overflow: bool,
}

impl TransmittedShadowResources {
    pub(super) fn new(
        device: &wgpu::Device,
        shadow_map: &crate::shadows::ShadowMap,
        material_layout: &wgpu::BindGroupLayout,
        joint_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let make_color = |cascade: usize| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("transmitted_shadow_color_{cascade}")),
                size: wgpu::Extent3d {
                    width: TRANSMITTED_SHADOW_MAP_SIZE,
                    height: TRANSMITTED_SHADOW_MAP_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TRANSMITTED_SHADOW_COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let make_depth = |cascade: usize| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("transmitted_shadow_depth_{cascade}")),
                size: wgpu::Extent3d {
                    width: TRANSMITTED_SHADOW_MAP_SIZE,
                    height: TRANSMITTED_SHADOW_MAP_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TRANSMITTED_SHADOW_DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let color_textures = std::array::from_fn(make_color);
        let color_views =
            std::array::from_fn(|i| color_textures[i].create_view(&Default::default()));
        let depth_textures = std::array::from_fn(make_depth);
        let depth_views =
            std::array::from_fn(|i| depth_textures[i].create_view(&Default::default()));

        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transmitted_shadow_draw_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                        TransmittedShadowDrawUniforms,
                    >() as u64),
                },
                count: None,
            }],
        });
        let draw_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transmitted_shadow_draw_uniforms"),
            size: u64::from(
                crate::shadows::SHADOW_UNIFORM_STRIDE
                    * crate::shadows::SHADOW_MAX_NODES
                    * crate::shadows::NUM_CASCADES as u32,
            ),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transmitted_shadow_draw_bg"),
            layout: &draw_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &draw_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<TransmittedShadowDrawUniforms>() as u64,
                    ),
                }),
            }],
        });
        let blocker_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transmitted_shadow_blocker_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let blocker_bind_groups = std::array::from_fn(|cascade| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("transmitted_shadow_blocker_bg"),
                layout: &blocker_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_map.depth_views[cascade]),
                }],
            })
        });
        let caster_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("transmitted_shadow_caster_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&draw_layout),
                    Some(material_layout),
                    Some(&blocker_layout),
                    Some(joint_layout),
                ],
                immediate_size: 0,
            });
        let caster_pipeline =
            create_transmitted_shadow_caster_pipeline(device, &caster_pipeline_layout, false);

        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transmitted_shadow_resolve_layout"),
            entries: &[
                uniform_layout_entry(0, wgpu::ShaderStages::FRAGMENT),
                depth_texture_layout_entry(1),
                float_texture_layout_entry(2),
                float_texture_layout_entry(3),
                float_texture_layout_entry(4),
                float_texture_layout_entry(5),
                float_texture_layout_entry(6),
                depth_texture_layout_entry(7),
                depth_texture_layout_entry(8),
                depth_texture_layout_entry(9),
            ],
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transmitted_shadow_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(TRANSMITTED_SHADOW_RESOLVE_WGSL.into()),
        });
        let resolve_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("transmitted_shadow_resolve_pipeline_layout"),
                bind_group_layouts: &[Some(&resolve_layout)],
                immediate_size: 0,
            });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("transmitted_shadow_resolve_pipeline"),
            layout: Some(&resolve_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::RED
                        | wgpu::ColorWrites::GREEN
                        | wgpu::ColorWrites::BLUE,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let resolve_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("transmitted_shadow_resolve_uniforms"),
            contents: bytemuck::bytes_of(&TransmittedShadowResolveUniforms {
                inv_vp: IDENTITY_MAT4,
                cascade_vps: [IDENTITY_MAT4; crate::shadows::NUM_CASCADES],
                camera_pos: [0.0; 4],
                cascade_splits: [0.0; 4],
                sun_dir: [0.0; 4],
                sun_color: [0.0; 4],
                wind: [0.0; 4],
                cloud: [0.0; 4],
                target_size: [1.0, 1.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            _color_textures: color_textures,
            color_views,
            _depth_textures: depth_textures,
            depth_views,
            draw_uniform_buffer,
            draw_uniform_bind_group,
            blocker_bind_groups,
            caster_pipeline_layout,
            caster_pipeline,
            caster_uv1_pipeline: None,
            resolve_uniform_buffer,
            resolve_layout,
            resolve_pipeline,
            resolve_bind_group: None,
            rendered_signatures: [0; crate::shadows::NUM_CASCADES],
            rendered_vps: None,
            blocker_generations: [0; crate::shadows::NUM_CASCADES],
            last_caster_count: 0,
            warned_overflow: false,
        }
    }

    pub(super) fn invalidate_resolve_bind_group(&mut self) {
        self.resolve_bind_group = None;
    }

    fn ensure_uv1_pipeline(&mut self, device: &wgpu::Device) -> bool {
        if self.caster_uv1_pipeline.is_none() {
            self.caster_uv1_pipeline = Some(create_transmitted_shadow_caster_pipeline(
                device,
                &self.caster_pipeline_layout,
                true,
            ));
            return true;
        }
        false
    }
}

fn uniform_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn float_texture_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_texture_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn transmission_hash(mut hash: u64, transmission: crate::models::MaterialTransmission) -> u64 {
    hash = fnv1a_bytes(hash, &[u8::from(transmission.authored)]);
    for value in [
        transmission.factor,
        transmission.ior,
        transmission.thickness_factor,
        transmission.baked_thickness_scale,
        transmission.attenuation_distance,
        transmission.attenuation_color[0],
        transmission.attenuation_color[1],
        transmission.attenuation_color[2],
    ] {
        hash = fnv1a_bytes(hash, &value.to_bits().to_le_bytes());
    }
    for binding in [transmission.texture, transmission.thickness_texture] {
        match binding {
            Some(binding) => {
                hash = fnv1a_bytes(hash, &[1]);
                hash = fnv1a_bytes(hash, &binding.source_texture_index.to_le_bytes());
                hash = fnv1a_bytes(
                    hash,
                    &binding
                        .runtime_texture_idx
                        .unwrap_or_default()
                        .to_le_bytes(),
                );
                hash = fnv1a_bytes(hash, &binding.transform.tex_coord.to_le_bytes());
                for value in [
                    binding.transform.offset[0],
                    binding.transform.offset[1],
                    binding.transform.rotation,
                    binding.transform.scale[0],
                    binding.transform.scale[1],
                ] {
                    hash = fnv1a_bytes(hash, &value.to_bits().to_le_bytes());
                }
            }
            None => hash = fnv1a_bytes(hash, &[0]),
        }
    }
    hash
}

impl Renderer {
    pub(super) fn ensure_transmitted_shadow_resources(&mut self) {
        if !transmitted_shadows_enabled() || self.transmitted_shadow_resources.is_some() {
            return;
        }
        let Some(material_layout) = self.scene_refractive_material_layout.as_ref() else {
            return;
        };
        self.transmitted_shadow_resources = Some(TransmittedShadowResources::new(
            &self.device,
            &self.shadow_map,
            material_layout,
            &self.joint_layout,
        ));
        self.created_pipelines(2);
        log::info!(
            "bloom materials: transmitted directional shadows enabled \
             (nearest-layer, {}x{}, rgba8+depth16, lazy)",
            TRANSMITTED_SHADOW_MAP_SIZE,
            TRANSMITTED_SHADOW_MAP_SIZE,
        );
    }

    pub(super) fn ensure_transmitted_shadow_uv1_resources(&mut self) {
        if !transmitted_shadows_enabled() || !self.shadow_map.enabled {
            return;
        }
        self.ensure_transmitted_shadow_resources();
        let created = self
            .transmitted_shadow_resources
            .as_mut()
            .is_some_and(|resources| resources.ensure_uv1_pipeline(&self.device));
        if created {
            self.created_pipelines(1);
        }
    }

    pub(super) fn select_transmitted_shadow_route(&mut self, scene: &crate::scene::SceneGraph) {
        if !transmitted_shadows_enabled()
            || !self.imported_refraction_enabled
            || !self.shadow_map.enabled
        {
            self.transmitted_shadows_active = false;
            return;
        }
        let has_caster = self.has_refractive_model_draws
            || (self.has_refractive_scene_nodes && scene.has_transmitted_shadow_casters());
        if has_caster && self.transmitted_shadow_resources.is_none() {
            self.ensure_transmitted_shadow_resources();
        }
        self.transmitted_shadows_active = has_caster && self.transmitted_shadow_resources.is_some();
    }

    pub(super) fn record_transmitted_shadow_maps(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        scene: &crate::scene::SceneGraph,
    ) {
        if !self.transmitted_shadows_active {
            return;
        }
        let Some(mut resources) = self.transmitted_shadow_resources.take() else {
            return;
        };

        struct Draw<'a> {
            vertex: &'a wgpu::Buffer,
            secondary_uv: Option<&'a wgpu::Buffer>,
            index: &'a wgpu::Buffer,
            material: &'a wgpu::BindGroup,
            vertex_byte_offset: u64,
            index_byte_offset: u64,
            first_index: u32,
            index_count: u32,
            base_vertex: i32,
            model: [[f32; 4]; 4],
            tint: [f32; 4],
            bounds_min: [f32; 3],
            bounds_max: [f32; 3],
            joint_offset: f32,
            signature: u64,
        }

        let mut draws = Vec::new();
        for (index, (_handle, node)) in scene.nodes.iter().enumerate() {
            if !node.visible
                || node.gi_only
                || !node.cast_shadow
                || !node.material.transmission.is_active()
                || node.indices().is_empty()
            {
                continue;
            }
            let (Some(vertex), Some(index_buffer), Some(material)) = (
                node.gpu_vb.as_ref(),
                node.gpu_ib.as_ref(),
                node.gpu_refractive_material_bg.as_ref(),
            ) else {
                continue;
            };
            let secondary_uv = if node.gpu_refractive_uses_uv1 {
                let Some(buffer) = node.gpu_secondary_uv_vb.as_ref() else {
                    continue;
                };
                Some(buffer)
            } else {
                None
            };
            let mut signature = fnv1a_bytes(FNV_OFFSET, &[0]);
            signature = fnv1a_bytes(signature, &(index as u64).to_le_bytes());
            let world_transform = node.world_transform();
            signature = fnv1a_bytes(signature, bytemuck::bytes_of(&world_transform));
            signature = transmission_hash(signature, node.material.transmission);
            signature = fnv1a_bytes(signature, &node.material.texture_idx.to_le_bytes());
            for value in [
                node.material.color[0],
                node.material.color[1],
                node.material.color[2],
                node.material.opacity,
            ] {
                signature = fnv1a_bytes(signature, &value.to_bits().to_le_bytes());
            }
            let opacity = if node.material.opacity.is_finite() {
                node.material.opacity
            } else {
                1.0
            };
            draws.push(Draw {
                vertex,
                secondary_uv,
                index: index_buffer,
                material,
                vertex_byte_offset: 0,
                index_byte_offset: 0,
                first_index: 0,
                index_count: node.gpu_index_count,
                base_vertex: 0,
                model: world_transform,
                tint: [
                    node.material.color[0],
                    node.material.color[1],
                    node.material.color[2],
                    opacity,
                ],
                bounds_min: node.world_bounds_min,
                bounds_max: node.world_bounds_max,
                joint_offset: 0.0,
                signature,
            });
        }
        for command in &self.model_draw_commands {
            let Some(Some(meshes)) = self.model_gpu_cache.get(&command.cache_handle) else {
                continue;
            };
            let Some(mesh) = meshes.get(command.mesh_idx) else {
                continue;
            };
            if !mesh.transmission.is_active() {
                continue;
            }
            let Some(material) = mesh.refractive_material_bg.as_ref() else {
                continue;
            };
            let secondary_uv = if mesh.refractive_uses_uv1 {
                let Some(buffer) = mesh.refractive_uv1_buffer.as_ref() else {
                    continue;
                };
                Some(buffer)
            } else {
                None
            };
            let (geometry, vertex_byte_offset, index_byte_offset) = if secondary_uv.is_some() {
                self.gpu_driven
                    .mesh_draw_localized(&mesh.geometry, mesh.index_count)
            } else {
                (
                    self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count),
                    0,
                    0,
                )
            };
            let (bounds_min, bounds_max) = command
                .bounds_override
                .unwrap_or_else(|| transform_aabb(&command.model, mesh.local_min, mesh.local_max));
            let mut signature = fnv1a_bytes(FNV_OFFSET, &[1]);
            signature = fnv1a_bytes(signature, &command.cache_handle.to_le_bytes());
            signature = fnv1a_bytes(signature, &(command.mesh_idx as u64).to_le_bytes());
            signature = fnv1a_bytes(signature, bytemuck::bytes_of(&command.model));
            signature = transmission_hash(signature, mesh.transmission);
            signature = fnv1a_bytes(signature, &mesh.base_color_idx.to_le_bytes());
            for value in command.tint {
                signature = fnv1a_bytes(signature, &value.to_bits().to_le_bytes());
            }
            if command.skinned {
                signature = fnv1a_bytes(signature, &self.shadow_map.frame_nonce.to_le_bytes());
            }
            draws.push(Draw {
                vertex: geometry.vertex,
                secondary_uv,
                index: geometry.index,
                material,
                vertex_byte_offset,
                index_byte_offset,
                first_index: geometry.first_index,
                index_count: geometry.index_count,
                base_vertex: geometry.base_vertex,
                model: command.model,
                tint: command.tint,
                bounds_min,
                bounds_max,
                joint_offset: command.joint_offset,
                signature,
            });
        }

        resources.last_caster_count = draws.len().min(u32::MAX as usize) as u32;
        let cascade_planes: [[[f32; 4]; 6]; crate::shadows::NUM_CASCADES] =
            std::array::from_fn(|cascade| {
                crate::scene::extract_frustum_planes(&self.shadow_map.light_vps[cascade])
            });
        profiler.begin("transmitted_shadow_maps");
        for cascade in 0..crate::shadows::NUM_CASCADES {
            let mut indices = Vec::with_capacity(draws.len());
            let mut signature = FNV_OFFSET;
            for (index, draw) in draws.iter().enumerate() {
                let has_bounds = draw.bounds_min[0] <= draw.bounds_max[0];
                if has_bounds
                    && crate::scene::aabb_outside_frustum(
                        &cascade_planes[cascade],
                        draw.bounds_min,
                        draw.bounds_max,
                    )
                {
                    continue;
                }
                signature = fnv1a_bytes(signature, &draw.signature.to_le_bytes());
                indices.push(index);
            }
            let stale = resources.rendered_vps.map_or(true, |vps| {
                vps[cascade] != self.shadow_map.light_vps[cascade]
            }) || resources.rendered_signatures[cascade] != signature
                || resources.blocker_generations[cascade]
                    != self.shadow_map.live_cascade_generation[cascade];
            if !stale {
                continue;
            }

            let max = crate::shadows::SHADOW_MAX_NODES as usize;
            if indices.len() > max {
                indices.truncate(max);
                if !resources.warned_overflow {
                    log::warn!(
                        "bloom transmitted shadows: caster budget exceeded ({} > {}); \
                         keeping deterministic submission prefix",
                        draws.len(),
                        max,
                    );
                    resources.warned_overflow = true;
                }
            }
            let stride = crate::shadows::SHADOW_UNIFORM_STRIDE as usize;
            let cascade_base = cascade * max * stride;
            let mut payload = vec![0_u8; indices.len().max(1) * stride];
            for (slot, &draw_index) in indices.iter().enumerate() {
                let draw = &draws[draw_index];
                let uniforms = TransmittedShadowDrawUniforms {
                    light_vp: self.shadow_map.light_vps[cascade],
                    model: draw.model,
                    tint: draw.tint,
                    misc: [draw.joint_offset, 0.0, 0.0, 0.0],
                };
                let offset = slot * stride;
                payload[offset..offset + std::mem::size_of_val(&uniforms)]
                    .copy_from_slice(bytemuck::bytes_of(&uniforms));
            }
            if !indices.is_empty() {
                self.queue.write_buffer(
                    &resources.draw_uniform_buffer,
                    cascade_base as u64,
                    &payload[..indices.len() * stride],
                );
            }
            let timestamp_writes = profiler.pass_timestamp_writes("transmitted_shadow_maps");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transmitted_shadow_map"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.color_views[cascade],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &resources.depth_views[cascade],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(2, &resources.blocker_bind_groups[cascade], &[]);
            pass.set_bind_group(3, &self.joint_bind_group, &[]);
            let mut current_uses_uv1 = None;
            for (slot, &draw_index) in indices.iter().enumerate() {
                let draw = &draws[draw_index];
                let uses_uv1 = draw.secondary_uv.is_some();
                if current_uses_uv1 != Some(uses_uv1) {
                    pass.set_pipeline(if uses_uv1 {
                        resources
                            .caster_uv1_pipeline
                            .as_ref()
                            .expect("UV1 transmitted-shadow pipeline initialized on material use")
                    } else {
                        &resources.caster_pipeline
                    });
                    current_uses_uv1 = Some(uses_uv1);
                }
                let offset = (cascade_base + slot * stride) as u32;
                pass.set_bind_group(0, &resources.draw_uniform_bind_group, &[offset]);
                pass.set_bind_group(1, draw.material, &[]);
                pass.set_vertex_buffer(0, draw.vertex.slice(draw.vertex_byte_offset..));
                if let Some(secondary_uv) = draw.secondary_uv {
                    pass.set_vertex_buffer(1, secondary_uv.slice(..));
                }
                pass.set_index_buffer(
                    draw.index.slice(draw.index_byte_offset..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(
                    draw.first_index..draw.first_index + draw.index_count,
                    draw.base_vertex,
                    0..1,
                );
            }
            drop(pass);
            resources.rendered_signatures[cascade] = signature;
            resources.blocker_generations[cascade] =
                self.shadow_map.live_cascade_generation[cascade];
        }
        resources.rendered_vps = Some(self.shadow_map.light_vps);
        profiler.end("transmitted_shadow_maps");
        self.transmitted_shadow_resources = Some(resources);
    }

    pub(super) fn record_transmitted_shadow_resolve(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
    ) {
        if !self.transmitted_shadows_active {
            return;
        }
        let (width, height) = self.render_extent();
        let Some(resources) = self.transmitted_shadow_resources.as_mut() else {
            return;
        };
        let uniforms = TransmittedShadowResolveUniforms {
            inv_vp: self.current_inv_vp_matrix,
            cascade_vps: self.shadow_map.light_vps,
            camera_pos: [
                self.current_camera_pos[0],
                self.current_camera_pos[1],
                self.current_camera_pos[2],
                0.0,
            ],
            cascade_splits: self.lighting_uniforms.shadow_cascade_splits,
            sun_dir: self.lighting_uniforms.light_dir,
            sun_color: self.lighting_uniforms.light_color,
            wind: self.lighting_uniforms.wind,
            cloud: self.lighting_uniforms.cloud,
            target_size: [width as f32, height as f32, 0.0, 0.0],
        };
        self.queue.write_buffer(
            &resources.resolve_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        if resources.resolve_bind_group.is_none() {
            resources.resolve_bind_group =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("transmitted_shadow_resolve_bg"),
                    layout: &resources.resolve_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: resources.resolve_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.albedo_rt_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.material_rt_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&resources.color_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&resources.color_views[1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&resources.color_views[2]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&resources.depth_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&resources.depth_views[1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::TextureView(&resources.depth_views[2]),
                        },
                    ],
                }));
        }

        profiler.begin("transmitted_shadow_resolve");
        let timestamp_writes = profiler.pass_timestamp_writes("transmitted_shadow_resolve");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transmitted_shadow_resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_rt_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&resources.resolve_pipeline);
        pass.set_bind_group(
            0,
            resources
                .resolve_bind_group
                .as_ref()
                .expect("transmitted-shadow resolve bind group was initialized"),
            &[],
        );
        pass.draw(0..3, 0..1);
        drop(pass);
        profiler.end("transmitted_shadow_resolve");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transmitted_shadow_shaders_parse() {
        for secondary_uv in [false, true] {
            let caster = transmitted_shadow_caster_shader_source(secondary_uv);
            wgpu::naga::front::wgsl::parse_str(&caster).unwrap_or_else(|error| {
                panic!(
                    "transmitted-shadow caster{} WGSL failed: {error:?}",
                    if secondary_uv { " UV1" } else { "" }
                )
            });
            assert_eq!(
                caster.contains("@location(3) secondary_uv"),
                secondary_uv,
                "ordinary transmitted shadows must not fetch a second vertex stream"
            );
        }
        wgpu::naga::front::wgsl::parse_str(TRANSMITTED_SHADOW_RESOLVE_WGSL)
            .unwrap_or_else(|error| panic!("transmitted-shadow resolve WGSL failed: {error:?}"));
    }

    #[test]
    fn transmitted_shadow_memory_is_bounded_and_lazy_by_contract() {
        assert_eq!(TRANSMITTED_SHADOW_PERSISTENT_BYTES, 18_874_368);
        assert!(TRANSMITTED_SHADOW_PERSISTENT_BYTES < 20 * 1024 * 1024);
    }
}
