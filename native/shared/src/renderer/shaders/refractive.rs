//! Imported transmission shader specialization.

/// Build the dedicated imported-transmission scene shader without changing
/// the shader compiled for ordinary scene materials.
///
/// Desktop/native backends add group 4 and sample the render-graph-owned
/// pre-translucency color/depth snapshots. Four-bind-group targets
/// (`fold_scene_inputs`: WebGPU/Android) compile the same physical lobe but
/// source transmitted radiance from the environment map instead.
pub(in crate::renderer) fn scene_refractive_shader_source(
    base_scene_shader: &str,
    folded_scene_inputs: bool,
    screen_space_reflections: bool,
    secondary_uv: bool,
) -> String {
    assert!(
        !folded_scene_inputs || !screen_space_reflections,
        "folded four-bind-group targets cannot add native reflection inputs"
    );
    const JOINT_DECLARATION: &str =
        "@group(3) @binding(1) var<uniform> joints_prev: JointMatrices;";
    let scene_inputs = if folded_scene_inputs {
        ""
    } else if screen_space_reflections {
        r#"
struct RefractiveReflectionParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    params: vec4<f32>,
    planar_plane: vec4<f32>,
};

@group(4) @binding(0) var refractive_scene_color_tex: texture_2d<f32>;
@group(4) @binding(1) var refractive_scene_color_samp: sampler;
@group(4) @binding(2) var refractive_scene_depth_tex: texture_depth_2d;
@group(4) @binding(3) var<uniform> refractive_reflection: RefractiveReflectionParams;
@group(4) @binding(4) var refractive_planar_tex: texture_2d<f32>;
"#
    } else {
        r#"
@group(4) @binding(0) var refractive_scene_color_tex: texture_2d<f32>;
@group(4) @binding(1) var refractive_scene_color_samp: sampler;
@group(4) @binding(2) var refractive_scene_depth_tex: texture_depth_2d;
"#
    };
    let physical_declarations = format!(
        r#"{JOINT_DECLARATION}

struct TransmissionFactors {{
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    transmission_uv: vec4<f32>,
    transmission_rotation: vec4<f32>,
    thickness_uv: vec4<f32>,
    thickness_rotation: vec4<f32>,
}};

@group(2) @binding(11) var transmission_tex: texture_2d<f32>;
@group(2) @binding(12) var transmission_samp: sampler;
@group(2) @binding(13) var thickness_tex: texture_2d<f32>;
@group(2) @binding(14) var thickness_samp: sampler;
@group(2) @binding(15) var<uniform> transmission_material: TransmissionFactors;
{scene_inputs}"#
    );
    let mut source = base_scene_shader.replacen(JOINT_DECLARATION, &physical_declarations, 1);
    assert_ne!(
        source, base_scene_shader,
        "scene shader joint declaration changed; refractive ABI injection must be updated"
    );
    let layered_secondary_uv =
        secondary_uv && base_scene_shader.contains("fn layered_secondary_uv(in:");
    let model_scale_location = if layered_secondary_uv { 8 } else { 7 };
    // The ordinary per-draw group is vertex-visible only. Carry model scale as
    // a refractive-variant-only interpolant instead of widening that established
    // layout to the fragment stage for every scene material.
    source = source.replacen(
        "    @location(6) prev_clip: vec4<f32>,",
        &format!(
            "    @location(6) prev_clip: vec4<f32>,\n    @location({model_scale_location}) model_scale: f32,"
        ),
        1,
    );
    source = source.replacen(
        "        o.prev_clip = u.prev_mvp * prev_world4;",
        "        o.prev_clip = u.prev_mvp * prev_world4;\n\
         o.model_scale = (length(u.model[0].xyz) + length(u.model[1].xyz) \
         + length(u.model[2].xyz)) / 3.0;",
        1,
    );
    source = source.replacen(
        "    out.prev_clip = u.prev_mvp * vec4<f32>(prev_local, 1.0);",
        "    out.prev_clip = u.prev_mvp * vec4<f32>(prev_local, 1.0);\n\
         out.model_scale = (length(u.model[0].xyz) + length(u.model[1].xyz) \
         + length(u.model[2].xyz)) / 3.0;",
        1,
    );
    if secondary_uv && !layered_secondary_uv {
        source = source.replacen(
            "    @location(6) tangent: vec4<f32>,\n};",
            "    @location(6) tangent: vec4<f32>,\n\
             @location(7) secondary_uv: vec2<f32>,\n\
             };",
            1,
        );
        source = source.replacen(
            "    @location(7) model_scale: f32,",
            "    @location(7) model_scale: f32,\n\
             @location(8) secondary_uv: vec2<f32>,",
            1,
        );
        source = source.replacen(
            "        o.uv = in.uv;",
            "        o.uv = in.uv;\n        o.secondary_uv = in.secondary_uv;",
            1,
        );
        source = source.replacen(
            "    out.uv = in.uv;",
            "    out.uv = in.uv;\n    out.secondary_uv = in.secondary_uv;",
            1,
        );
        assert!(
            source.contains("@location(8) secondary_uv"),
            "scene vertex ABI changed; refractive UV1 injection must be updated"
        );
    } else if layered_secondary_uv {
        assert!(
            source.contains("@location(7) secondary_uv")
                && source.contains("@location(8) model_scale"),
            "layered refractive vertex ABI changed; specialization must be updated"
        );
    }

    let transmitted_radiance = if folded_scene_inputs {
        r#"
    // Constrained four-bind-group fallback: preserve the refractive material
    // type and its Fresnel/absorption response, but source off-screen
    // transmitted radiance from the prefiltered environment.
    let max_transmission_mip = max(f32(textureNumLevels(env_tex)) - 1.0, 0.0);
    let transmitted_radiance = env_sample_lod(
        refracted_direction,
        roughness * max_transmission_mip,
    );
    let undistorted_radiance = env_sample_lod(-v, roughness * max_transmission_mip);
"#
    } else {
        r#"
    let scene_dimensions_u = textureDimensions(refractive_scene_color_tex, 0);
    let scene_dimensions = vec2<f32>(scene_dimensions_u);
    let current_ndc = in.curr_clip.xy / max(abs(in.curr_clip.w), 0.000001);
    let current_uv = clamp(
        vec2<f32>(current_ndc.x * 0.5 + 0.5, 0.5 - current_ndc.y * 0.5),
        vec2<f32>(0.0001),
        vec2<f32>(0.9999),
    );

    // Convert the refracted world-space direction into a stable screen-space
    // travel distance using the fragment's world-position derivatives. The
    // material factor already carries static glTF node scale baked by the
    // importer; the interpolant adds the later draw/instance scale. A 64-pixel
    // cap prevents pathological assets from sampling unrelated frame regions.
    let world_dx = dpdx(in.world_pos);
    let world_dy = dpdy(in.world_pos);
    let world_dx_len = max(length(world_dx), 0.000001);
    let world_dy_len = max(length(world_dy), 0.000001);
    let screen_tangent_x = world_dx / world_dx_len;
    let screen_tangent_y = world_dy / world_dy_len;
    let ray_distance = thickness_world
        / max(abs(dot(refracted_direction, n)), 0.15);
    var offset_pixels = vec2<f32>(
        dot(refracted_direction, screen_tangent_x) * ray_distance / world_dx_len,
        dot(refracted_direction, screen_tangent_y) * ray_distance / world_dy_len,
    );
    offset_pixels = clamp(offset_pixels, vec2<f32>(-64.0), vec2<f32>(64.0));
    var refracted_uv = clamp(
        current_uv + offset_pixels / scene_dimensions,
        vec2<f32>(0.0001),
        vec2<f32>(0.9999),
    );

    // Reject offsets that cross in front of this glass surface. This keeps a
    // nearby opaque silhouette from being pulled through the refractor.
    let candidate_pixel = clamp(
        vec2<i32>(refracted_uv * scene_dimensions),
        vec2<i32>(0),
        vec2<i32>(scene_dimensions_u) - vec2<i32>(1),
    );
    let candidate_depth = textureLoad(
        refractive_scene_depth_tex,
        candidate_pixel,
        0,
    );
    if (candidate_depth + 0.0005 < in.clip_position.z) {
        refracted_uv = current_uv;
    }

    var transmitted_radiance = textureSampleLevel(
        refractive_scene_color_tex,
        refractive_scene_color_samp,
        refracted_uv,
        0.0,
    ).rgb;
    // Deterministic five-tap rough transmission. Smooth glass stays at one
    // fetch; rough glass integrates a bounded footprint without temporal noise.
    if (roughness > 0.08) {
        let blur_uv = vec2<f32>(8.0 * roughness * roughness) / scene_dimensions;
        transmitted_radiance = (
            transmitted_radiance * 4.0
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv + vec2<f32>(blur_uv.x, 0.0),
                0.0,
            ).rgb
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv - vec2<f32>(blur_uv.x, 0.0),
                0.0,
            ).rgb
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv + vec2<f32>(0.0, blur_uv.y),
                0.0,
            ).rgb
            + textureSampleLevel(
                refractive_scene_color_tex,
                refractive_scene_color_samp,
                refracted_uv - vec2<f32>(0.0, blur_uv.y),
                0.0,
            ).rgb
        ) * 0.125;
    }
    let undistorted_radiance = textureSampleLevel(
        refractive_scene_color_tex,
        refractive_scene_color_samp,
        current_uv,
        0.0,
    ).rgb;
"#
    };

    let reflection_helpers = if screen_space_reflections {
        r#"
fn refractive_screen_reflection(
    in: VertexOutputScene,
    reflected_direction: vec3<f32>,
    roughness: f32,
    environment_fallback: vec3<f32>,
) -> vec3<f32> {
    // The ordinary SSR target cannot be reused here: it was traced from the
    // opaque surface behind this fragment and therefore owns a different
    // normal/reflection ray. Launch one bounded ray from the glass fragment
    // against the immutable opaque snapshots instead.
    if (refractive_reflection.params.x < 0.5
        || roughness >= refractive_reflection.params.w) {
        return environment_fallback;
    }

    let dimensions_u = textureDimensions(refractive_scene_color_tex, 0);
    let dimensions = vec2<f32>(dimensions_u);
    let step_count = u32(refractive_reflection.params.z);
    let max_distance = refractive_reflection.params.y;
    let start_view = (
        refractive_reflection.view * vec4<f32>(in.world_pos, 1.0)
    ).xyz;
    let reflected_view = normalize((
        refractive_reflection.view * vec4<f32>(reflected_direction, 0.0)
    ).xyz);
    let start_clip = refractive_reflection.proj * vec4<f32>(start_view, 1.0);
    var previous_ray_depth = start_clip.z / max(abs(start_clip.w), 0.000001);
    var hit_uv = vec2<f32>(-1.0);
    var hit_confidence = 0.0;

    // Quadratic spacing keeps the nearest samples dense enough for window
    // frames and props while still reaching the same architectural range as
    // the established opaque SSR pass. The loop bound and every texture read
    // are fixed by the lazy uniform (currently eight).
    for (var step = 0u; step < step_count; step = step + 1u) {
        let fraction = f32(step + 1u) / f32(step_count);
        let distance = max_distance * fraction * fraction;
        let ray_view = start_view + reflected_view * distance;
        let ray_clip = refractive_reflection.proj * vec4<f32>(ray_view, 1.0);
        if (ray_clip.w <= 0.000001) {
            break;
        }
        let ray_ndc = ray_clip.xyz / ray_clip.w;
        if (ray_ndc.x <= -1.0 || ray_ndc.x >= 1.0
            || ray_ndc.y <= -1.0 || ray_ndc.y >= 1.0
            || ray_ndc.z <= 0.0 || ray_ndc.z >= 1.0) {
            break;
        }
        let ray_uv = vec2<f32>(
            ray_ndc.x * 0.5 + 0.5,
            0.5 - ray_ndc.y * 0.5,
        );
        let pixel = clamp(
            vec2<i32>(ray_uv * dimensions),
            vec2<i32>(0),
            vec2<i32>(dimensions_u) - vec2<i32>(1),
        );
        let scene_depth = textureLoad(refractive_scene_depth_tex, pixel, 0);
        let depth_delta = ray_ndc.z - scene_depth;
        let depth_stride = abs(ray_ndc.z - previous_ray_depth);
        let thickness = max(depth_stride * 2.0, 0.00075);
        if (scene_depth < 0.9999
            && depth_delta >= 0.0
            && depth_delta <= thickness) {
            hit_uv = ray_uv;
            hit_confidence = 1.0 - smoothstep(
                thickness * 0.25,
                thickness,
                depth_delta,
            );
            break;
        }
        previous_ray_depth = ray_ndc.z;
    }

    if (hit_uv.x < 0.0) {
        return environment_fallback;
    }
    // Suppress the screen boundary before it can pop. Rough glass fades to
    // the prefiltered environment rather than returning an incorrectly sharp
    // scene-color tap (the snapshot deliberately has no mip chain).
    let edge_pixels = min(
        min(hit_uv.x, 1.0 - hit_uv.x) * dimensions.x,
        min(hit_uv.y, 1.0 - hit_uv.y) * dimensions.y,
    );
    let edge_weight = smoothstep(0.0, 8.0, edge_pixels);
    let roughness_weight = 1.0 - smoothstep(
        refractive_reflection.params.w * 0.45,
        refractive_reflection.params.w,
        roughness,
    );
    let raw = textureSampleLevel(
        refractive_scene_color_tex,
        refractive_scene_color_samp,
        hit_uv,
        0.0,
    ).rgb;
    let screen_radiance = select(vec3<f32>(0.0), raw, raw == raw);
    let source_weight = clamp(
        edge_weight * roughness_weight * hit_confidence,
        0.0,
        1.0,
    );
    return mix(environment_fallback, screen_radiance, source_weight);
}

fn refractive_planar_sample(
    in: VertexOutputScene,
    roughness: f32,
) -> vec4<f32> {
    let plane = refractive_reflection.planar_plane;
    if (dot(plane.xyz, plane.xyz) < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    // A plane crossing unrelated vertical glass must not make the global
    // first-probe choice leak onto that surface. Use the unperturbed vertex
    // normal so authored water waves can still perturb the sampled reflection.
    if (abs(dot(normalize(plane.xyz), normalize(in.normal))) < 0.8) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    let plane_distance = abs(dot(plane.xyz, in.world_pos) - plane.w);
    if (plane_distance > 0.075) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    let ndc = in.curr_clip.xy / max(abs(in.curr_clip.w), 0.000001);
    let uv = clamp(
        vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5),
        vec2<f32>(0.0001),
        vec2<f32>(0.9999),
    );
    let planar = textureSampleLevel(
        refractive_planar_tex,
        refractive_scene_color_samp,
        uv,
        0.0,
    );
    let planar_safe = select(vec3<f32>(0.0), planar.rgb, planar.rgb == planar.rgb);
    // The existing planar capture has no mip chain. Fade its exact reflection
    // into the lower tiers as roughness grows instead of returning an
    // incorrectly sharp image. Alpha zero is the probe's explicit miss value.
    let roughness_weight = 1.0 - smoothstep(0.18, 0.45, roughness);
    let source_weight = clamp(planar.a * roughness_weight, 0.0, 1.0);
    return vec4<f32>(planar_safe, source_weight);
}
"#
    } else {
        ""
    };
    let reflected_radiance = if screen_space_reflections {
        r#"
    let reflected_environment = env_sample_lod(
        reflected_direction,
        roughness * max(f32(textureNumLevels(env_tex)) - 1.0, 0.0),
    );
    let planar_reflected = refractive_planar_sample(in, roughness);
    var reflected = mix(
        reflected_environment,
        planar_reflected.rgb,
        max(planar_reflected.a, 0.0),
    );
    // A matching explicit probe is authoritative for its plane: alpha zero is
    // its documented geometry miss and therefore reveals the environment sky.
    // Only glass without an applicable probe pays for the bounded screen march.
    if (planar_reflected.a < 0.0) {
        reflected = refractive_screen_reflection(
            in,
            reflected_direction,
            roughness,
            reflected_environment,
        );
    }
"#
    } else {
        r#"
    let reflected = env_sample_lod(
        reflected_direction,
        roughness * max(f32(textureNumLevels(env_tex)) - 1.0, 0.0),
    );
"#
    };

    let transmission_source_uv = if secondary_uv {
        "select(\n            in.uv,\n            in.secondary_uv,\n            transmission_material.transmission_rotation.w > 0.5,\n        )"
    } else {
        "in.uv"
    };
    let thickness_source_uv = if secondary_uv {
        "select(\n            in.uv,\n            in.secondary_uv,\n            transmission_material.thickness_rotation.z > 0.5,\n        )"
    } else {
        "in.uv"
    };
    source.push_str(&format!(
        r#"

{reflection_helpers}

fn physical_texture_uv(
    uv: vec2<f32>,
    offset_scale: vec4<f32>,
    rotation: vec2<f32>,
) -> vec2<f32> {{
    let scaled = uv * offset_scale.zw;
    let rotated = vec2<f32>(
        rotation.x * scaled.x - rotation.y * scaled.y,
        rotation.y * scaled.x + rotation.x * scaled.y,
    );
    return offset_scale.xy + rotated;
}}

fn refractive_scene_normal(in: VertexOutputScene, front_facing: bool) -> vec3<f32> {{
    var n = normalize(in.normal);
    let normal_uv = material_uv(
        in.uv, material.uv_transforms[2], material.uv_transforms[3],
    );
    let normal_sample = textureSampleBias(
        normal_tex,
        normal_samp,
        normal_uv,
        1.0 + lighting.shadow_cascade_splits.w,
    ).xyz * 2.0 - 1.0;
    let scaled_normal_sample = vec3<f32>(
        normal_sample.xy * vec2<f32>(
            material.uv_transforms[2].w,
            material.uv_transforms[3].w,
        ),
        normal_sample.z,
    );
    let mapped = scaled_normal_sample / max(length(scaled_normal_sample), 0.000001);
    let tangent_len2 = dot(in.tangent.xyz, in.tangent.xyz);
    if (tangent_len2 > 0.0001) {{
        let tangent = normalize(in.tangent.xyz);
        let tangent_ortho = normalize(tangent - n * dot(n, tangent));
        let bitangent = cross(n, tangent_ortho) * in.tangent.w;
        n = normalize(
            tangent_ortho * mapped.x + bitangent * mapped.y + n * mapped.z,
        );
    }} else {{
        let tbn = compute_tbn(
            dpdx(in.world_pos),
            dpdy(in.world_pos),
            dpdx(normal_uv),
            dpdy(normal_uv),
            n,
        );
        n = normalize(tbn * mapped);
    }}
    if (!front_facing) {{
        n = -n;
    }}
    return n;
}}

struct RefractiveSceneOut {{
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
}};

@fragment
fn fs_refractive_scene(
    in: VertexOutputScene,
    @builtin(front_facing) front_facing: bool,
) -> RefractiveSceneOut {{
    // Reuse the established direct/IBL PBR evaluation for the non-transmitted
    // energy, then split the dielectric lobe below. This also preserves MASK
    // discard semantics for the unusual but legal MASK+transmission case.
    let surface = shade_main_scene(in, front_facing);
    let n = refractive_scene_normal(in, front_facing);
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);

    let base_uv = material_uv(
        in.uv, material.uv_transforms[0], material.uv_transforms[1],
    );
    let mr_uv = material_uv(
        in.uv, material.uv_transforms[4], material.uv_transforms[5],
    );
    let base_texel = textureSample(base_color_tex, base_color_samp, base_uv);
    var base_color = srgb_to_linear_v(base_texel.rgb) * in.color.rgb;
    let base_alpha = base_texel.a * in.color.a;
    let mr_texel = textureSample(mr_tex, mr_samp, mr_uv);
    let has_mr = material.metal_rough.z > 0.5 && material.metal_rough.z < 1.5;
    let has_spec_gloss = material.metal_rough.z > 1.5;
    var metallic = select(
        clamp(material.metal_rough.x, 0.0, 1.0),
        clamp(mr_texel.b * material.metal_rough.x, 0.0, 1.0),
        has_mr,
    );
    var roughness = select(
        clamp(material.metal_rough.y, 0.045, 1.0),
        clamp(mr_texel.g * material.metal_rough.y, 0.045, 1.0),
        has_mr,
    );
    if (has_spec_gloss) {{
        let authored_specular = srgb_to_linear_v(mr_texel.rgb) *
            material.spec_gloss.rgb;
        let converted = specgloss_to_metalrough_pixel(base_color, authored_specular);
        base_color = converted.rgb;
        metallic = converted.a;
        roughness = clamp(
            1.0 - mr_texel.a * material.spec_gloss.a,
            0.045,
            1.0,
        );
    }}

    let transmission_uv = physical_texture_uv(
        {transmission_source_uv},
        transmission_material.transmission_uv,
        transmission_material.transmission_rotation.xy,
    );
    let texture_transmission = select(
        1.0,
        textureSample(
            transmission_tex,
            transmission_samp,
            transmission_uv,
        ).r,
        transmission_material.transmission.w > 0.5,
    );
    let dielectric_weight = 1.0 - metallic;
    let transmission_weight = clamp(
        transmission_material.transmission.x
            * texture_transmission
            * dielectric_weight,
        0.0,
        1.0,
    );

    let thickness_uv = physical_texture_uv(
        {thickness_source_uv},
        transmission_material.thickness_uv,
        transmission_material.thickness_rotation.xy,
    );
    let texture_thickness = select(
        1.0,
        textureSample(
            thickness_tex,
            thickness_samp,
            thickness_uv,
        ).g,
        transmission_material.transmission_rotation.z > 0.5,
    );
    let mean_model_scale = max(in.model_scale, 0.0);
    let thickness_world = max(
        transmission_material.transmission.z
            * texture_thickness
            * mean_model_scale,
        0.0,
    );

    let ior = max(transmission_material.transmission.y, 1.0);
    let eta = 1.0 / ior;
    var refracted_direction = refract(-v, n, eta);
    if (dot(refracted_direction, refracted_direction) < 0.000001) {{
        refracted_direction = reflect(-v, n);
    }}
    refracted_direction = normalize(refracted_direction);

{transmitted_radiance}

    var absorption = vec3<f32>(1.0);
    if (transmission_material.attenuation.w > 0.0 && thickness_world > 0.0) {{
        let optical_distance = thickness_world
            / transmission_material.attenuation.w;
        absorption = pow(
            max(transmission_material.attenuation.rgb, vec3<f32>(0.000001)),
            vec3<f32>(optical_distance),
        );
    }}
    let transmitted = transmitted_radiance * base_color * absorption;

    let f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = f0_scalar
        + (1.0 - f0_scalar) * pow(1.0 - n_dot_v, 5.0);
    let reflected_direction = reflect(-v, n);
{reflected_radiance}

    // Energy partition: the ordinary PBR surface owns the opaque fraction;
    // the transmission fraction is split exactly between Fresnel reflection
    // and absorbed transmitted radiance.
    let dielectric_transmission = mix(transmitted, reflected, fresnel);
    var hdr = surface.color.rgb * (1.0 - transmission_weight)
        + dielectric_transmission * transmission_weight;

    // glTF BLEND+transmission additionally applies base-color alpha. Because
    // this fragment already composites against the snapshot, write alpha=1
    // to avoid applying the background a second time in fixed-function blend.
    if (material.metal_rough.w < 0.0) {{
        hdr = mix(undistorted_radiance, hdr, clamp(base_alpha, 0.0, 1.0));
    }}
    hdr = select(vec3<f32>(0.0), hdr, hdr == hdr);
    return RefractiveSceneOut(vec4<f32>(hdr, 1.0), surface.velocity);
}}
"#
    ));
    source
}
