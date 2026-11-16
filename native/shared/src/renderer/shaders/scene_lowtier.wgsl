// The low-tier scene shading (renderer/low_tier.rs, SH-069): replaces shade_main_scene on
// GPUs the engine classifies as low tier (a Pixel 4a's Adreno 618 and its class), in place of
// the mobile shading. One texture read per pixel (base colour, still alpha-tested); vertex
// normals; metallic and roughness from the material constants; sun and lamps as diffuse plus
// the sun's specular; ambient from the irradiance map. Dropped against the mobile shading:
// the normal, metallic-roughness and occlusion maps and the reflection of the sky (the emissive
// map is read only for materials that glow).
// On the reference phone the main pass is where the frame goes (48 of 78 ms in a fight), and
// on that GPU the cost is texture reads and registers, not arithmetic.
//
// The markers and names the prepass, froxel and GPU-driven variants rewrite are kept:
// PREPASS_STRIP, the point-light loop markers and the lamp attenuation term, `v`,
// `light_diffuse`, `light_f0`, `roughness`, and the canonical base-colour sample.
fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {
    var n = normalize(in.normal);
    let base_uv = material_uv(
        in.uv, material.uv_transforms[0], material.uv_transforms[1],
    );
    let lod_bias = lighting.shadow_cascade_splits.w;
    let base_tex = textureSampleBias(base_color_tex, base_color_samp, base_uv, lod_bias);
    let base_color = srgb_to_linear_v(base_tex.rgb) * in.color.rgb;
    let base_alpha = base_tex.a * in.color.a;
    let alpha_cutoff = material.metal_rough.w;
    //PREPASS_STRIP_BEGIN — removed in the prepassed-pipeline variant.
    if (alpha_cutoff > 0.0 && base_alpha < alpha_cutoff) {
        discard;
    }
    //PREPASS_STRIP_END
    if (alpha_cutoff > 0.0 && dot(n, lighting.camera_pos.xyz - in.world_pos) < 0.0) {
        n = -n;
    }
    if (alpha_cutoff < 0.0 && !front_facing) {
        n = -n;
    }

    let metallic = clamp(material.metal_rough.x, 0.0, 1.0)
        * select(1.0, 0.0, material.metal_rough.z > 1.5);
    // The roughness map is not read; textured materials keep their factor, which for a map
    // is a multiplier usually left at 1, so take it as mid-rough.
    let roughness = select(
        clamp(material.metal_rough.y, 0.3, 1.0),
        clamp(material.metal_rough.y * 0.7, 0.3, 1.0),
        material.metal_rough.z > 0.5,
    );
    let light_diffuse = base_color * (1.0 - metallic);
    let light_f0 = mix(vec3<f32>(0.04), base_color, metallic);
    // The emissive map only for materials that glow (a per-material branch): reading the
    // factor alone would light a whole textured surface at its factor.
    var emissive = vec3<f32>(0.0);
    if (dot(material.emissive.rgb, vec3<f32>(1.0)) > 0.0) {
        let emissive_uv = material_uv(
            in.uv, material.uv_transforms[6], material.uv_transforms[7],
        );
        let em_tex_sample = textureSample(em_tex, em_samp, emissive_uv);
        emissive = srgb_to_linear_v(em_tex_sample.rgb) * material.emissive.rgb;
    }

    let v = normalize(lighting.camera_pos.xyz - in.world_pos);
    var lit = lighting.ambient.rgb * lighting.ambient.a * base_color;
    let shadow_factor = sample_shadow(in.world_pos, normalize(in.normal), in.clip_position.xy);
    let direct_shadow = mix(0.03, 1.0, shadow_factor);
    let legacy_dir = normalize(lighting.light_dir.xyz);
    if (alpha_cutoff > 0.0) {
        // Foliage: wrap diffuse plus transmission, no specular (as the other shadings).
        let wrap = 0.45;
        let ndl_wrap = clamp((dot(n, legacy_dir) + wrap) / ((1.0 + wrap) * (1.0 + wrap)), 0.0, 1.0);
        let trans = pow(max(dot(v, -legacy_dir), 0.0), 3.0) * 0.85;
        lit += base_color * lighting.light_color.rgb * lighting.light_dir.w
             * (ndl_wrap / PI + trans) * direct_shadow;
    } else {
        lit += shade_pbr_f0(n, v, legacy_dir, lighting.light_color.rgb,
                            lighting.light_dir.w, light_diffuse, light_f0, roughness)
             * direct_shadow;
    }
    // BEGIN-POINT-LIGHT-LOOP (the froxel-clustered variant replaces this
    // loop, as in the full shader)
    let pt_count = u32(lighting.point_light_count.x);
    for (var i = 0u; i < pt_count; i++) {
        let pl = lighting.point_lights[i];
        let to_light = pl.position.xyz - in.world_pos;
        let dist = length(to_light);
        let range = pl.position.w;
        if (dist < range && dist > 0.0) {
            let l = to_light / dist;
            let atten = 1.0 - (dist / range);
            let atten2 = atten * atten;
            lit += shade_pbr_f0(n, v, l, pl.color.rgb, pl.color.w * atten2,
                                light_diffuse, light_f0, roughness);
        }
    }
    // END-POINT-LIGHT-LOOP

    // Ambient from the irradiance map only: one sample, no reflection of the sky.
    let irradiance = textureSampleLevel(
        env_diffuse_tex, env_samp, seamless_equirect_uv(dir_to_equirect_uv(n)), 0.0,
    ).rgb * lighting.camera_pos.w;
    let indirect_shadow = mix(0.35, 1.0, shadow_factor);
    let hdr_raw = lit + irradiance * light_diffuse * indirect_shadow + emissive;
    let hdr = select(vec3<f32>(0.0), hdr_raw, hdr_raw == hdr_raw);

    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let vel = (curr_ndc - prev_ndc) * 0.5;
    let non_opaque_alpha = select(base_alpha, 1.0, alpha_cutoff > 0.0);
    let out_alpha = select(in.color.a, non_opaque_alpha, alpha_cutoff != 0.0);
    return SceneOut(
        vec4<f32>(hdr, out_alpha),
        vec2<f32>(metallic, roughness),
        vel,
        vec4<f32>(base_color, 1.0 - shadow_factor),
    );
}
