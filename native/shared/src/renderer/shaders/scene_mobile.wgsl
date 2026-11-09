// The mobile scene shading (renderer/mobile_shading.rs): replaces
// shade_main_scene in the scene shader on mobile GPUs, the way Unreal's mobile
// renderer and Unity URP have their own forward base pass. Same inputs,
// outputs and lights; dropped: normal-map variance and curvature roughness
// (Toksvig / LEADR / screen-space kernel), the BRDF LUT (Karis's analytic
// environment BRDF instead, as Unreal mobile),
// coverage-dithered alpha masks (a plain alpha test), cloud shadows and the
// specular firefly caps. The texture reads are the canonical ones, so the
// GPU-driven variant rewrites them like the full shader's.
fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {
    var n = normalize(in.normal);
    let base_uv = material_uv(
        in.uv, material.uv_transforms[0], material.uv_transforms[1],
    );
    let normal_uv = material_uv(
        in.uv, material.uv_transforms[2], material.uv_transforms[3],
    );
    let mr_uv = material_uv(
        in.uv, material.uv_transforms[4], material.uv_transforms[5],
    );
    let emissive_uv = material_uv(
        in.uv, material.uv_transforms[6], material.uv_transforms[7],
    );
    let occlusion_uv = material_uv(
        in.uv, material.uv_transforms[8], material.uv_transforms[9],
    );

    let lod_bias = lighting.shadow_cascade_splits.w;
    let nm_sample4 = textureSampleBias(normal_tex, normal_samp, normal_uv, 0.25 + lod_bias);
    let nm_sample = normalize((nm_sample4.xyz * 2.0 - 1.0) * vec3<f32>(
        material.uv_transforms[2].w,
        material.uv_transforms[3].w,
        1.0,
    ));
    if (dot(in.tangent.xyz, in.tangent.xyz) > 0.0001) {
        let t = normalize(in.tangent.xyz);
        let t_ortho = normalize(t - n * dot(n, t));
        let b = cross(n, t_ortho) * in.tangent.w;
        n = normalize(t_ortho * nm_sample.x + b * nm_sample.y + n * nm_sample.z);
    } else {
        let tbn = compute_tbn(
            dpdx(in.world_pos), dpdy(in.world_pos), dpdx(normal_uv), dpdy(normal_uv), n,
        );
        n = normalize(tbn * nm_sample);
    }

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

    let mr_tex_sample = textureSample(mr_tex, mr_samp, mr_uv);
    let has_mr = material.metal_rough.z > 0.5 && material.metal_rough.z < 1.5;
    let has_spec_gloss = material.metal_rough.z > 1.5;
    let metallic = select(
        select(
            clamp(material.metal_rough.x, 0.0, 1.0),
            clamp(mr_tex_sample.b * material.metal_rough.x, 0.0, 1.0),
            has_mr,
        ),
        0.0,
        has_spec_gloss,
    );
    let roughness_raw = select(
        select(
            clamp(material.metal_rough.y, 0.045, 1.0),
            clamp(mr_tex_sample.g * material.metal_rough.y, 0.045, 1.0),
            has_mr,
        ),
        clamp(1.0 - mr_tex_sample.a * material.spec_gloss.a, 0.045, 1.0),
        has_spec_gloss,
    );
    let roughness = max(roughness_raw, select(0.15, 0.045, has_spec_gloss) * (1.0 - metallic));
    // The lights' view of either workflow (see shade_pbr_f0).
    let authored_specular = srgb_to_linear_v(mr_tex_sample.rgb) * material.spec_gloss.rgb;
    let light_diffuse = select(base_color * (1.0 - metallic), base_color, has_spec_gloss);
    let light_f0 = select(
        mix(vec3<f32>(0.04), base_color, metallic),
        authored_specular,
        has_spec_gloss,
    );

    let em_tex_sample = textureSample(em_tex, em_samp, emissive_uv);
    let emissive = srgb_to_linear_v(em_tex_sample.rgb) * material.emissive.rgb;
    let sampled_occlusion = textureSample(occ_tex, occ_samp, occlusion_uv).r;
    let occlusion = mix(1.0, sampled_occlusion, material.uv_transforms[8].w);

    // --- Direct lighting ---
    let v = normalize(lighting.camera_pos.xyz - in.world_pos);
    var lit = lighting.ambient.rgb * lighting.ambient.a * base_color;
    let shadow_factor = sample_shadow(in.world_pos, normalize(in.normal), in.clip_position.xy);
    let direct_shadow = mix(0.03, 1.0, shadow_factor);
    let legacy_dir = normalize(lighting.light_dir.xyz);
    if (alpha_cutoff > 0.0) {
        // Foliage: wrap diffuse plus transmission, no specular.
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
    let dir_count = u32(lighting.dir_light_count.x);
    for (var i = 0u; i < dir_count; i++) {
        let dl = lighting.dir_lights[i];
        let l = normalize(dl.direction.xyz);
        lit += shade_pbr_f0(n, v, l, dl.color.rgb, dl.direction.w,
                            light_diffuse, light_f0, roughness);
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

    // --- Image-based lighting ---
    // Karis's analytic fit of the split-sum environment BRDF ("Physically
    // Based Shading on Mobile", the one Unreal's mobile renderer uses) in
    // place of the BRDF LUT.
    let n_dot_v = max(dot(n, v), 0.0);
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let rc = roughness * c0 + c1;
    let a004 = min(rc.x * rc.x, exp2(-9.28 * n_dot_v)) * rc.x + rc.y;
    let env_ab = vec2<f32>(-1.04, 1.04) * a004 + rc.zw;
    let env_brdf = light_f0 * env_ab.x + vec3<f32>(env_ab.y);
    // Multi-scatter compensation from the fit's energy, as the full shader
    // does from the LUT's (Fdez-Aguera 2019).
    let ess = env_ab.x + env_ab.y;
    let f_avg = light_f0 + (vec3<f32>(1.0) - light_f0) * (1.0 / 21.0);
    let ms = f_avg * ess / (vec3<f32>(1.0) - f_avg * (1.0 - ess)) * (1.0 - ess);

    let irradiance = textureSampleLevel(
        env_diffuse_tex, env_samp, seamless_equirect_uv(dir_to_equirect_uv(n)), 0.0,
    ).rgb * lighting.camera_pos.w;
    let ibl_diffuse = irradiance * light_diffuse * (vec3<f32>(1.0) - env_brdf) * occlusion;
    let max_spec_mip = max(f32(textureNumLevels(env_tex)) - 1.0, 0.0);
    let prefiltered_env = env_sample_lod(reflect(-v, n), roughness * max_spec_mip);
    // SSR owns the smooth lobes when it runs (dir_light_count.z), as in the
    // full shader.
    let ssr_own = clamp(
        lighting.dir_light_count.z * (1.0 - smoothstep(0.45, 0.70, roughness)),
        0.0, 1.0);
    let spec_occ = clamp(
        pow(n_dot_v + occlusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + occlusion,
        0.0, 1.0,
    );
    let ibl_spec = prefiltered_env * (env_brdf + ms) * spec_occ * (1.0 - ssr_own);
    let indirect_shadow = mix(0.35, 1.0, shadow_factor);

    let hdr_raw = lit + (ibl_diffuse + ibl_spec) * indirect_shadow + emissive;
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

