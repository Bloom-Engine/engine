// Bloom layered-PBR reference contract, realtime version 4.
//
// This source is injected only into the lazy layered scene specialization.
// Ordinary scene shaders do not declare, load, or branch on these values.

struct LayeredPbrFactors {
    // x = clearcoat factor, y = perceptual roughness,
    // z = clearcoat normal scale, w = dielectric IOR (zero is glTF compat).
    clearcoat_ior: vec4<f32>,
    // rgb = specular color factor, w = scalar specular factor.
    specular: vec4<f32>,
    // rgb = linear sheen color factor, w = sheen perceptual roughness.
    sheen: vec4<f32>,
    // x = anisotropy strength, yz = cos/sin rotation, w = reserved.
    anisotropy: vec4<f32>,
    // x = factor, y = film IOR, z/w = thickness min/max in nanometers.
    iridescence: vec4<f32>,
    // For each texture: uv.xy = offset, uv.zw = scale;
    // rotation.xy = cos/sin, rotation.z = usable texture, rotation.w = UV set.
    clearcoat_factor_uv: vec4<f32>,
    clearcoat_factor_rotation: vec4<f32>,
    clearcoat_roughness_uv: vec4<f32>,
    clearcoat_roughness_rotation: vec4<f32>,
    clearcoat_normal_uv: vec4<f32>,
    clearcoat_normal_rotation: vec4<f32>,
    specular_factor_uv: vec4<f32>,
    specular_factor_rotation: vec4<f32>,
    specular_color_uv: vec4<f32>,
    specular_color_rotation: vec4<f32>,
    sheen_color_uv: vec4<f32>,
    sheen_color_rotation: vec4<f32>,
    sheen_roughness_uv: vec4<f32>,
    sheen_roughness_rotation: vec4<f32>,
    anisotropy_uv: vec4<f32>,
    anisotropy_texture_rotation: vec4<f32>,
    iridescence_factor_uv: vec4<f32>,
    iridescence_factor_rotation: vec4<f32>,
    iridescence_thickness_uv: vec4<f32>,
    iridescence_thickness_rotation: vec4<f32>,
};

struct LayeredSurface {
    dielectric_f0: vec3<f32>,
    dielectric_f90: f32,
    clearcoat_normal: vec3<f32>,
    clearcoat_factor: f32,
    clearcoat_roughness: f32,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    anisotropic_tangent: vec3<f32>,
    anisotropy_strength: f32,
    iridescence_factor: f32,
    iridescence_ior: f32,
    iridescence_thickness_nm: f32,
};

fn layered_transform_uv(
    primary_uv: vec2<f32>,
    secondary_uv: vec2<f32>,
    params: vec4<f32>,
    rotation: vec4<f32>,
) -> vec2<f32> {
    let source_uv = select(primary_uv, secondary_uv, rotation.w > 0.5);
    let scaled = source_uv * params.zw;
    let rotated = vec2<f32>(
        rotation.x * scaled.x - rotation.y * scaled.y,
        rotation.y * scaled.x + rotation.x * scaled.y,
    );
    return rotated + params.xy;
}

fn layered_ior_f0(ior: f32) -> f32 {
    if (ior == 0.0) {
        return 1.0;
    }
    let safe_ior = max(ior, 1.0);
    let ratio = (safe_ior - 1.0) / (safe_ior + 1.0);
    return ratio * ratio;
}

fn layered_fresnel_schlick_f90(
    cos_theta: f32,
    f0: vec3<f32>,
    f90: vec3<f32>,
) -> vec3<f32> {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m2 = m * m;
    let m5 = m2 * m2 * m;
    return f0 + (f90 - f0) * m5;
}

fn layered_fresnel0_to_ior(f0: vec3<f32>) -> vec3<f32> {
    let root = sqrt(clamp(f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
    return (vec3<f32>(1.0) + root) / (vec3<f32>(1.0) - root);
}

fn layered_ior_to_fresnel0(
    transmitted_ior: vec3<f32>,
    incident_ior: f32,
) -> vec3<f32> {
    let incident = vec3<f32>(incident_ior);
    let ratio =
        (transmitted_ior - incident) / (transmitted_ior + incident);
    return ratio * ratio;
}

fn layered_iridescence_sensitivity(
    optical_path_difference_nm: f32,
    shift: vec3<f32>,
) -> vec3<f32> {
    let phase = 2.0 * PI * optical_path_difference_nm * 1e-9;
    let phase_squared = phase * phase;
    let value = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let position = vec3<f32>(1.6810e6, 1.7953e6, 2.2084e6);
    let variance = vec3<f32>(4.3278e9, 9.3046e9, 6.6121e9);
    var xyz = value * sqrt(vec3<f32>(2.0 * PI) * variance)
        * cos(position * phase + shift)
        * exp(-vec3<f32>(phase_squared) * variance);
    xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e9)
        * cos(2.2399e6 * phase + shift.x)
        * exp(-4.5282e9 * phase_squared);
    xyz /= 1.0685e-7;
    return vec3<f32>(
        3.2404542 * xyz.x - 0.9692660 * xyz.y + 0.0556434 * xyz.z,
        -1.5371385 * xyz.x + 1.8760108 * xyz.y - 0.2040259 * xyz.z,
        -0.4985314 * xyz.x + 0.0415560 * xyz.y + 1.0572252 * xyz.z,
    );
}

fn layered_eval_iridescence(
    outside_ior: f32,
    authored_film_ior: f32,
    cos_theta_1: f32,
    authored_thickness_nm: f32,
    base_f0: vec3<f32>,
) -> vec3<f32> {
    let safe_outside_ior = max(outside_ior, 1e-4);
    let thickness_nm = max(authored_thickness_nm, 0.0);
    let film_ior = mix(
        safe_outside_ior,
        max(authored_film_ior, 1.0),
        smoothstep(0.0, 0.03, thickness_nm),
    );
    let cosine_1 = clamp(cos_theta_1, 0.0, 1.0);
    let sin_theta_2_squared = pow(safe_outside_ior / film_ior, 2.0)
        * (1.0 - cosine_1 * cosine_1);
    let cos_theta_2_squared = 1.0 - sin_theta_2_squared;
    if (cos_theta_2_squared < 0.0) {
        return vec3<f32>(1.0);
    }
    let cosine_2 = sqrt(cos_theta_2_squared);

    let r0 = layered_ior_f0(film_ior / safe_outside_ior);
    let r12 = layered_fresnel_schlick_f90(
        cosine_1,
        vec3<f32>(r0),
        vec3<f32>(1.0),
    ).x;
    let t121 = 1.0 - r12;
    let phi12 = select(0.0, PI, film_ior < safe_outside_ior);
    let phi21 = PI - phi12;

    let base_ior = layered_fresnel0_to_ior(base_f0);
    let r1 = layered_ior_to_fresnel0(base_ior, film_ior);
    let r23 = layered_fresnel_schlick_f90(
        cosine_2,
        r1,
        vec3<f32>(1.0),
    );
    let phi23 = vec3<f32>(
        select(0.0, PI, base_ior.x < film_ior),
        select(0.0, PI, base_ior.y < film_ior),
        select(0.0, PI, base_ior.z < film_ior),
    );
    let optical_path_difference =
        2.0 * film_ior * thickness_nm * cosine_2;
    let phase_shift = vec3<f32>(phi21) + phi23;
    let r123 = clamp(
        vec3<f32>(r12) * r23,
        vec3<f32>(1e-5),
        vec3<f32>(0.9999),
    );
    let reflected_series =
        vec3<f32>(t121 * t121) * r23 / (vec3<f32>(1.0) - r123);
    var result = vec3<f32>(r12) + reflected_series;
    var coefficient = reflected_series - vec3<f32>(t121);
    let amplitude = sqrt(r123);
    for (var order = 1; order <= 2; order += 1) {
        coefficient *= amplitude;
        result += coefficient * 2.0 * layered_iridescence_sensitivity(
            f32(order) * optical_path_difference,
            f32(order) * phase_shift,
        );
    }
    return clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn layered_raw_iridescence_base_fresnel(
    surface: LayeredSurface,
    cos_theta: f32,
    base_color: vec3<f32>,
    metallic: f32,
) -> vec3<f32> {
    let dielectric = layered_eval_iridescence(
        1.0,
        surface.iridescence_ior,
        cos_theta,
        surface.iridescence_thickness_nm,
        surface.dielectric_f0,
    );
    let conductor = layered_eval_iridescence(
        1.0,
        surface.iridescence_ior,
        cos_theta,
        surface.iridescence_thickness_nm,
        base_color,
    );
    return mix(dielectric, conductor, metallic);
}

fn layered_base_fresnel(
    surface: LayeredSurface,
    cos_theta: f32,
    base_color: vec3<f32>,
    metallic: f32,
) -> vec3<f32> {
    let dielectric = layered_fresnel_schlick_f90(
        cos_theta,
        surface.dielectric_f0,
        vec3<f32>(surface.dielectric_f90),
    );
    let conductor = f_schlick(cos_theta, base_color);
    let base = mix(dielectric, conductor, metallic);
    if (surface.iridescence_factor <= 0.0) {
        return base;
    }
    return mix(
        base,
        layered_raw_iridescence_base_fresnel(
            surface,
            cos_theta,
            base_color,
            metallic,
        ),
        surface.iridescence_factor,
    );
}

fn layered_dielectric_fresnel(
    surface: LayeredSurface,
    cos_theta: f32,
) -> vec3<f32> {
    let base = layered_fresnel_schlick_f90(
        cos_theta,
        surface.dielectric_f0,
        vec3<f32>(surface.dielectric_f90),
    );
    if (surface.iridescence_factor <= 0.0) {
        return base;
    }
    let thin_film = layered_eval_iridescence(
        1.0,
        surface.iridescence_ior,
        cos_theta,
        surface.iridescence_thickness_nm,
        surface.dielectric_f0,
    );
    return mix(base, thin_film, surface.iridescence_factor);
}

fn layered_dielectric_transmission(surface: LayeredSurface, cos_theta: f32) -> f32 {
    let fresnel = layered_dielectric_fresnel(surface, cos_theta);
    return max(1.0 - max(fresnel.r, max(fresnel.g, fresnel.b)), 0.0);
}

fn layered_base_fresnel_roughness(
    surface: LayeredSurface,
    n_dot_v: f32,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let grazing = max(
        vec3<f32>(surface.dielectric_f90 * (1.0 - roughness)),
        surface.dielectric_f0,
    );
    let m = clamp(1.0 - n_dot_v, 0.0, 1.0);
    let m2 = m * m;
    let dielectric = surface.dielectric_f0 + (grazing - surface.dielectric_f0) * m2 * m2 * m;
    let conductor_f0 = base_color;
    let conductor_grazing = max(vec3<f32>(1.0 - roughness), conductor_f0);
    let conductor = conductor_f0
        + (conductor_grazing - conductor_f0) * m2 * m2 * m;
    let base = mix(dielectric, conductor, metallic);
    if (surface.iridescence_factor <= 0.0) {
        return base;
    }
    return mix(
        base,
        layered_raw_iridescence_base_fresnel(
            surface,
            n_dot_v,
            base_color,
            metallic,
        ),
        surface.iridescence_factor,
    );
}

fn layered_ibl_f0(
    surface: LayeredSurface,
    n_dot_v: f32,
    base_color: vec3<f32>,
    metallic: f32,
) -> vec3<f32> {
    let base_f0 = mix(surface.dielectric_f0, base_color, metallic);
    if (surface.iridescence_factor <= 0.0) {
        return base_f0;
    }
    let f90 = mix(
        vec3<f32>(surface.dielectric_f90),
        vec3<f32>(1.0),
        metallic,
    );
    let m = clamp(1.0 - n_dot_v, 0.0, 1.0);
    let m2 = m * m;
    let m5 = min(m2 * m2 * m, 0.9999);
    let thin_film_f0 = clamp(
        (
            layered_raw_iridescence_base_fresnel(
                surface,
                n_dot_v,
                base_color,
                metallic,
            ) - f90 * m5
        ) / (1.0 - m5),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return mix(base_f0, thin_film_f0, surface.iridescence_factor);
}

fn layered_clearcoat_fresnel(surface: LayeredSurface, cos_theta: f32) -> f32 {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m2 = m * m;
    return surface.clearcoat_factor * (0.04 + 0.96 * m2 * m2 * m);
}

fn layered_clearcoat_transmission(surface: LayeredSurface, cos_theta: f32) -> f32 {
    return max(1.0 - layered_clearcoat_fresnel(surface, cos_theta), 0.0);
}

fn layered_clearcoat_ibl_attenuation(
    surface: LayeredSurface,
    view: vec3<f32>,
) -> f32 {
    let n_dot_v = max(dot(surface.clearcoat_normal, view), 0.0);
    let transmission = layered_clearcoat_transmission(surface, n_dot_v);
    return transmission * transmission;
}

fn layered_sheen_lambda_helper(x: f32, alpha_g: f32) -> f32 {
    let one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);
    let a = mix(21.5473, 25.3245, one_minus_alpha_sq);
    let b = mix(3.82987, 3.32435, one_minus_alpha_sq);
    let c = mix(0.19823, 0.16801, one_minus_alpha_sq);
    let d = mix(-1.97760, -1.27393, one_minus_alpha_sq);
    let e = mix(-4.32054, -4.85967, one_minus_alpha_sq);
    return a / (1.0 + b * pow(max(x, 0.0), c)) + d * x + e;
}

fn layered_sheen_lambda(cos_theta: f32, alpha_g: f32) -> f32 {
    let cosine = clamp(abs(cos_theta), 0.0, 1.0);
    if (cosine < 0.5) {
        return exp(layered_sheen_lambda_helper(cosine, alpha_g));
    }
    return exp(
        2.0 * layered_sheen_lambda_helper(0.5, alpha_g)
            - layered_sheen_lambda_helper(1.0 - cosine, alpha_g),
    );
}

fn layered_sheen_distribution(n_dot_h: f32, perceptual_roughness: f32) -> f32 {
    let alpha_g = max(perceptual_roughness * perceptual_roughness, 1e-6);
    let inverse_alpha = 1.0 / alpha_g;
    let sin2_h = max(1.0 - n_dot_h * n_dot_h, 0.0);
    return (2.0 + inverse_alpha) * pow(sin2_h, 0.5 * inverse_alpha)
        / (2.0 * PI);
}

fn layered_sheen_visibility(
    n_dot_l: f32,
    n_dot_v: f32,
    perceptual_roughness: f32,
) -> f32 {
    let alpha_g = max(perceptual_roughness * perceptual_roughness, 1e-6);
    let denominator = (
        1.0
            + layered_sheen_lambda(n_dot_v, alpha_g)
            + layered_sheen_lambda(n_dot_l, alpha_g)
    ) * (4.0 * n_dot_v * n_dot_l);
    return 1.0 / max(denominator, 1e-6);
}

fn layered_sheen_directional_albedo(n_dot: f32, roughness: f32) -> f32 {
    return textureSample(
        layered_sheen_albedo_tex,
        layered_sampler,
        vec2<f32>(clamp(n_dot, 0.0, 1.0), clamp(roughness, 0.0, 1.0)),
    ).r;
}

fn layered_sheen_scale(
    surface: LayeredSurface,
    n_dot_v: f32,
    n_dot_l: f32,
) -> f32 {
    let maximum_color = max(
        surface.sheen_color.r,
        max(surface.sheen_color.g, surface.sheen_color.b),
    );
    if (maximum_color <= 0.0) {
        return 1.0;
    }
    let view_albedo =
        layered_sheen_directional_albedo(n_dot_v, surface.sheen_roughness);
    let light_albedo =
        layered_sheen_directional_albedo(n_dot_l, surface.sheen_roughness);
    return clamp(1.0 - maximum_color * max(view_albedo, light_albedo), 0.0, 1.0);
}

fn layered_sheen_ibl_scale(surface: LayeredSurface, n_dot_v: f32) -> f32 {
    let maximum_color = max(
        surface.sheen_color.r,
        max(surface.sheen_color.g, surface.sheen_color.b),
    );
    if (maximum_color <= 0.0) {
        return 1.0;
    }
    let albedo =
        layered_sheen_directional_albedo(n_dot_v, surface.sheen_roughness);
    return clamp(1.0 - maximum_color * albedo, 0.0, 1.0);
}

fn layered_d_ggx_anisotropic(
    n_dot_h: f32,
    t_dot_h: f32,
    b_dot_h: f32,
    at: f32,
    ab: f32,
) -> f32 {
    let a2 = at * ab;
    let f = vec3<f32>(ab * t_dot_h, at * b_dot_h, a2 * n_dot_h);
    let w2 = a2 / max(dot(f, f), 1e-8);
    return a2 * w2 * w2 / PI;
}

fn layered_v_ggx_anisotropic(
    n_dot_l: f32,
    n_dot_v: f32,
    t_dot_v: f32,
    b_dot_v: f32,
    t_dot_l: f32,
    b_dot_l: f32,
    at: f32,
    ab: f32,
) -> f32 {
    let ggx_v = n_dot_l * length(vec3<f32>(at * t_dot_v, ab * b_dot_v, n_dot_v));
    let ggx_l = n_dot_v * length(vec3<f32>(at * t_dot_l, ab * b_dot_l, n_dot_l));
    return clamp(0.5 / max(ggx_v + ggx_l, 1e-5), 0.0, 1.0);
}

fn layered_base_distribution(
    surface: LayeredSurface,
    n: vec3<f32>,
    h: vec3<f32>,
    n_dot_h: f32,
    alpha: f32,
) -> f32 {
    if (surface.anisotropy_strength <= 0.0) {
        return d_ggx(n_dot_h, alpha * alpha);
    }
    let tangent = surface.anisotropic_tangent;
    let bitangent = normalize(cross(n, tangent));
    let at = mix(alpha, 1.0, surface.anisotropy_strength * surface.anisotropy_strength);
    return layered_d_ggx_anisotropic(
        n_dot_h,
        dot(tangent, h),
        dot(bitangent, h),
        at,
        alpha,
    );
}

fn layered_base_visibility(
    surface: LayeredSurface,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    n_dot_l: f32,
    n_dot_v: f32,
    alpha: f32,
) -> f32 {
    if (surface.anisotropy_strength <= 0.0) {
        return v_smith_ggx_correlated(n_dot_l, n_dot_v, alpha * alpha);
    }
    let tangent = surface.anisotropic_tangent;
    let bitangent = normalize(cross(n, tangent));
    let at = mix(alpha, 1.0, surface.anisotropy_strength * surface.anisotropy_strength);
    return layered_v_ggx_anisotropic(
        n_dot_l,
        n_dot_v,
        dot(tangent, v),
        dot(bitangent, v),
        dot(tangent, l),
        dot(bitangent, l),
        at,
        alpha,
    );
}

fn layered_ibl_reflection(
    surface: LayeredSurface,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    if (surface.anisotropy_strength <= 0.0) {
        return reflect(-v, n);
    }
    let anisotropic_bitangent = normalize(cross(n, surface.anisotropic_tangent));
    let anisotropic_tangent = cross(anisotropic_bitangent, v);
    let anisotropic_normal_raw =
        cross(anisotropic_tangent, anisotropic_bitangent);
    let anisotropic_normal_len2 =
        dot(anisotropic_normal_raw, anisotropic_normal_raw);
    let anisotropic_normal = select(
        n,
        anisotropic_normal_raw
            * inverseSqrt(max(anisotropic_normal_len2, 1e-8)),
        anisotropic_normal_len2 > 1e-8,
    );
    let bend = 1.0 - surface.anisotropy_strength * (1.0 - roughness);
    let bend2 = bend * bend;
    let bent_normal_raw = mix(anisotropic_normal, n, bend2 * bend2);
    let bent_normal = normalize(bent_normal_raw);
    return normalize(reflect(-v, bent_normal));
}

fn layered_sheen_ibl(
    surface: LayeredSurface,
    n: vec3<f32>,
    v: vec3<f32>,
    max_spec_mip: f32,
    occlusion: f32,
) -> vec3<f32> {
    let maximum_color = max(
        surface.sheen_color.r,
        max(surface.sheen_color.g, surface.sheen_color.b),
    );
    if (maximum_color <= 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), 0.0);
    let reflection = reflect(-v, n);
    let prefiltered = env_sample_lod(
        reflection,
        surface.sheen_roughness * max_spec_mip,
    );
    let albedo = layered_sheen_directional_albedo(
        n_dot_v,
        surface.sheen_roughness,
    );
    return prefiltered * surface.sheen_color * albedo * occlusion;
}

fn layered_safe_tangent(normal: vec3<f32>, candidate: vec3<f32>) -> vec3<f32> {
    let projected = candidate - normal * dot(normal, candidate);
    let projected_len2 = dot(projected, projected);
    if (projected_len2 > 1e-8) {
        return projected * inverseSqrt(projected_len2);
    }
    let fallback_axis = select(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        abs(normal.x) > 0.9,
    );
    return normalize(cross(normal, fallback_axis));
}

fn evaluate_layered_surface(
    in: VertexOutputScene,
    base_normal: vec3<f32>,
    lod_bias: f32,
) -> LayeredSurface {
    let secondary_uv = layered_secondary_uv(in);

    var specular_factor = clamp(layered_material.specular.w, 0.0, 1.0);
    if (layered_material.specular_factor_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.specular_factor_uv,
            layered_material.specular_factor_rotation,
        );
        specular_factor *= textureSampleBias(
            layered_specular_factor_tex,
            layered_sampler,
            uv,
            lod_bias,
        ).a;
    }

    var specular_color = max(layered_material.specular.rgb, vec3<f32>(0.0));
    if (layered_material.specular_color_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.specular_color_uv,
            layered_material.specular_color_rotation,
        );
        let sampled = textureSampleBias(
            layered_specular_color_tex,
            layered_sampler,
            uv,
            lod_bias,
        ).rgb;
        specular_color *= srgb_to_linear_v(sampled);
    }
    let dielectric_f0 = min(
        vec3<f32>(layered_ior_f0(layered_material.clearcoat_ior.w)) * specular_color,
        vec3<f32>(1.0),
    ) * specular_factor;

    var clearcoat_factor = clamp(layered_material.clearcoat_ior.x, 0.0, 1.0);
    if (layered_material.clearcoat_factor_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.clearcoat_factor_uv,
            layered_material.clearcoat_factor_rotation,
        );
        clearcoat_factor *= textureSampleBias(
            layered_clearcoat_factor_tex,
            layered_sampler,
            uv,
            lod_bias,
        ).r;
    }

    var clearcoat_roughness = clamp(layered_material.clearcoat_ior.y, 0.04, 1.0);
    if (clearcoat_factor > 0.0
        && layered_material.clearcoat_roughness_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.clearcoat_roughness_uv,
            layered_material.clearcoat_roughness_rotation,
        );
        clearcoat_roughness = clamp(
            clearcoat_roughness * textureSampleBias(
                layered_clearcoat_roughness_tex,
                layered_sampler,
                uv,
                lod_bias,
            ).g,
            0.04,
            1.0,
        );
    }

    var clearcoat_normal = base_normal;
    if (clearcoat_factor > 0.0
        && layered_material.clearcoat_normal_rotation.z > 0.5) {
        let source_uv = select(
            in.uv,
            secondary_uv,
            layered_material.clearcoat_normal_rotation.w > 0.5,
        );
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.clearcoat_normal_uv,
            layered_material.clearcoat_normal_rotation,
        );
        let sampled = textureSampleBias(
            layered_clearcoat_normal_tex,
            layered_sampler,
            uv,
            1.0 + lod_bias,
        );
        var tangent_normal = sampled.xyz * 2.0 - 1.0;
        tangent_normal.x *= layered_material.clearcoat_ior.z;
        tangent_normal.y *= layered_material.clearcoat_ior.z;
        let normal_len2 = clamp(dot(tangent_normal, tangent_normal), 0.01, 1.0);
        tangent_normal *= inverseSqrt(normal_len2);

        var tbn = compute_tbn(
            dpdx(in.world_pos),
            dpdy(in.world_pos),
            dpdx(source_uv),
            dpdy(source_uv),
            base_normal,
        );
        let tangent_ortho =
            in.tangent.xyz - base_normal * dot(base_normal, in.tangent.xyz);
        if (dot(tangent_ortho, tangent_ortho) > 1e-4) {
            let mesh_tangent = normalize(tangent_ortho);
            let mesh_bitangent =
                cross(base_normal, mesh_tangent) * in.tangent.w;
            tbn = mat3x3<f32>(mesh_tangent, mesh_bitangent, base_normal);
        }
        let mapped_raw = tbn * tangent_normal;
        let mapped_len2 = dot(mapped_raw, mapped_raw);
        let mapped = select(
            base_normal,
            mapped_raw * inverseSqrt(max(mapped_len2, 1e-8)),
            mapped_len2 > 1e-8,
        );
        // Keep the independently-authored coat on the geometric hemisphere.
        // This prevents invalid maps or degenerate UV derivatives from turning
        // the top interface through the base surface.
        let hemisphere = dot(mapped, base_normal);
        clearcoat_normal = normalize(
            mapped + base_normal * max(0.05 - hemisphere, 0.0),
        );

        // The same LEADR/Toksvig and screen-space variance treatment as the
        // base lobe keeps the sharper coat from reintroducing temporal sparkle.
        let sigma2_toksvig = (1.0 - normal_len2) / normal_len2;
        let baked_variance = clamp(sampled.a, 0.0, 0.999);
        let sigma2_baked = baked_variance / max(1.0 - baked_variance, 0.001);
        let normal_dx = dpdx(clearcoat_normal);
        let normal_dy = dpdy(clearcoat_normal);
        let curvature_sq = dot(normal_dx, normal_dx) + dot(normal_dy, normal_dy);
        let kernel_alpha = min(2.0 * curvature_sq, 0.9);
        let roughness2 = min(
            clearcoat_roughness * clearcoat_roughness
                + sigma2_toksvig
                + sigma2_baked
                + kernel_alpha,
            1.0,
        );
        clearcoat_roughness = sqrt(roughness2);
    }

    var iridescence_factor = clamp(layered_material.iridescence.x, 0.0, 1.0);
    if (iridescence_factor > 0.0
        && layered_material.iridescence_factor_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.iridescence_factor_uv,
            layered_material.iridescence_factor_rotation,
        );
        iridescence_factor *= textureSampleBias(
            layered_iridescence_factor_tex,
            layered_sampler,
            uv,
            lod_bias,
        ).r;
    }
    var iridescence_thickness_nm =
        max(layered_material.iridescence.w, 0.0);
    if (iridescence_factor > 0.0
        && layered_material.iridescence_thickness_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.iridescence_thickness_uv,
            layered_material.iridescence_thickness_rotation,
        );
        let sampled = textureSampleBias(
            layered_iridescence_thickness_tex,
            layered_sampler,
            uv,
            lod_bias,
        ).g;
        iridescence_thickness_nm = mix(
            max(layered_material.iridescence.z, 0.0),
            max(layered_material.iridescence.w, 0.0),
            sampled,
        );
    }
    if (iridescence_thickness_nm <= 0.0) {
        iridescence_factor = 0.0;
    }

    var sheen_color = max(layered_material.sheen.rgb, vec3<f32>(0.0));
    var anisotropy_strength = clamp(layered_material.anisotropy.x, 0.0, 1.0);
    let sheen_factor_max = max(
        sheen_color.r,
        max(sheen_color.g, sheen_color.b),
    );
    if (sheen_factor_max <= 0.0 && anisotropy_strength <= 0.0) {
        // Version-2 clearcoat/specular/IOR materials keep a short path: no
        // sheen/anisotropy texture reads, UV transforms, derivatives, or LUT
        // work. The tangent is unused while anisotropy strength is zero.
        return LayeredSurface(
            dielectric_f0,
            specular_factor,
            clearcoat_normal,
            clearcoat_factor,
            clearcoat_roughness,
            vec3<f32>(0.0),
            0.04,
            base_normal,
            0.0,
            iridescence_factor,
            max(layered_material.iridescence.y, 1.0),
            iridescence_thickness_nm,
        );
    }
    if (layered_material.sheen_color_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.sheen_color_uv,
            layered_material.sheen_color_rotation,
        );
        sheen_color *= srgb_to_linear_v(textureSampleBias(
            layered_sheen_color_tex,
            layered_sampler,
            uv,
            lod_bias,
        ).rgb);
    }
    var sheen_roughness = clamp(layered_material.sheen.w, 0.04, 1.0);
    if (max(sheen_color.r, max(sheen_color.g, sheen_color.b)) > 0.0
        && layered_material.sheen_roughness_rotation.z > 0.5) {
        let uv = layered_transform_uv(
            in.uv,
            secondary_uv,
            layered_material.sheen_roughness_uv,
            layered_material.sheen_roughness_rotation,
        );
        sheen_roughness = clamp(
            sheen_roughness * textureSampleBias(
                layered_sheen_roughness_tex,
                layered_sampler,
                uv,
                lod_bias,
            ).a,
            0.04,
            1.0,
        );
    }

    var anisotropic_tangent = base_normal;
    if (anisotropy_strength > 0.0) {
        var anisotropy_direction = vec2<f32>(1.0, 0.0);
        let anisotropy_source_uv = select(
            in.uv,
            secondary_uv,
            layered_material.anisotropy_texture_rotation.w > 0.5,
        );
        let anisotropy_tbn = compute_tbn(
            dpdx(in.world_pos),
            dpdy(in.world_pos),
            dpdx(anisotropy_source_uv),
            dpdy(anisotropy_source_uv),
            base_normal,
        );
        if (layered_material.anisotropy_texture_rotation.z > 0.5) {
            let uv = layered_transform_uv(
                in.uv,
                secondary_uv,
                layered_material.anisotropy_uv,
                layered_material.anisotropy_texture_rotation,
            );
            let sampled = textureSampleBias(
                layered_anisotropy_tex,
                layered_sampler,
                uv,
                lod_bias,
            ).rgb;
            anisotropy_direction = sampled.rg * 2.0 - vec2<f32>(1.0);
            if (dot(anisotropy_direction, anisotropy_direction) <= 1e-6) {
                anisotropy_direction = vec2<f32>(1.0, 0.0);
            } else {
                anisotropy_direction = normalize(anisotropy_direction);
            }
            anisotropy_strength *= sampled.b;
        }
        let rotated_direction = vec2<f32>(
            layered_material.anisotropy.y * anisotropy_direction.x
                - layered_material.anisotropy.z * anisotropy_direction.y,
            layered_material.anisotropy.z * anisotropy_direction.x
                + layered_material.anisotropy.y * anisotropy_direction.y,
        );
        anisotropic_tangent = layered_safe_tangent(
            base_normal,
            anisotropy_tbn * vec3<f32>(rotated_direction, 0.0),
        );
        let tangent_ortho =
            in.tangent.xyz - base_normal * dot(base_normal, in.tangent.xyz);
        if (dot(tangent_ortho, tangent_ortho) > 1e-4) {
            let mesh_tangent = normalize(tangent_ortho);
            let mesh_bitangent = cross(base_normal, mesh_tangent) * in.tangent.w;
            anisotropic_tangent = layered_safe_tangent(
                base_normal,
                mesh_tangent * rotated_direction.x + mesh_bitangent * rotated_direction.y,
            );
        }
        // The distribution squares tangent/bitangent projections, but
        // retaining the authored mirrored tangent sign here is still
        // necessary for textured rotations and bent-normal IBL.
        anisotropic_tangent =
            layered_safe_tangent(base_normal, anisotropic_tangent);
    }

    return LayeredSurface(
        dielectric_f0,
        specular_factor,
        clearcoat_normal,
        clearcoat_factor,
        clearcoat_roughness,
        sheen_color,
        sheen_roughness,
        anisotropic_tangent,
        anisotropy_strength,
        iridescence_factor,
        max(layered_material.iridescence.y, 1.0),
        iridescence_thickness_nm,
    );
}

fn shade_layered_pbr(
    surface: LayeredSurface,
    n: vec3<f32>,
    v: vec3<f32>,
    l_dir: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let base = shade_layered_base_pbr(
        surface,
        n,
        v,
        l_dir,
        light_color,
        intensity,
        base_color,
        metallic,
        roughness,
    );
    var undercoat = base;
    let sheen_max = max(
        surface.sheen_color.r,
        max(surface.sheen_color.g, surface.sheen_color.b),
    );
    if (sheen_max > 0.0 && intensity > 0.0) {
        let sheen_n_dot_l = max(dot(n, l_dir), 0.0);
        let sheen_n_dot_v = max(dot(n, v), 1e-4);
        if (sheen_n_dot_l > 0.0) {
            undercoat = base * layered_sheen_scale(
                surface,
                sheen_n_dot_v,
                sheen_n_dot_l,
            );
            let half_raw = l_dir + v;
            let half_len2 = dot(half_raw, half_raw);
            if (half_len2 > 1e-12) {
                let half_vector = half_raw * inverseSqrt(half_len2);
                let n_dot_h = clamp(dot(n, half_vector), 0.0, 1.0);
                let distribution =
                    layered_sheen_distribution(n_dot_h, surface.sheen_roughness);
                let visibility = layered_sheen_visibility(
                    sheen_n_dot_l,
                    sheen_n_dot_v,
                    surface.sheen_roughness,
                );
                let sheen_raw = surface.sheen_color * distribution * visibility;
                let sheen_luma = dot(
                    sheen_raw,
                    vec3<f32>(0.2126, 0.7152, 0.0722),
                );
                let sheen_cap = 1.0 / (1.0 + sheen_luma / 0.3);
                let sheen = sheen_raw * sheen_cap * light_color
                    * intensity * sheen_n_dot_l;
                undercoat += sheen;
            }
        }
    }
    if (surface.clearcoat_factor <= 0.0 || intensity <= 0.0) {
        return undercoat;
    }

    let coat_n = surface.clearcoat_normal;
    let n_dot_l = max(dot(coat_n, l_dir), 0.0);
    let n_dot_v = max(dot(coat_n, v), 1e-4);
    let base_attenuation =
        layered_clearcoat_transmission(surface, n_dot_v)
        * layered_clearcoat_transmission(surface, n_dot_l);
    if (n_dot_l <= 0.0) {
        return undercoat * base_attenuation;
    }
    let half_raw = l_dir + v;
    let half_len2 = dot(half_raw, half_raw);
    if (half_len2 <= 1e-12) {
        return undercoat * base_attenuation;
    }
    let half_vector = half_raw * inverseSqrt(half_len2);
    let n_dot_h = clamp(dot(coat_n, half_vector), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, half_vector), 0.0, 1.0);
    let alpha = max(
        surface.clearcoat_roughness * surface.clearcoat_roughness,
        0.001,
    );
    let alpha2 = alpha * alpha;
    let distribution = d_ggx(n_dot_h, alpha2);
    let visibility = v_smith_ggx_correlated(n_dot_l, n_dot_v, alpha2);
    let fresnel = layered_clearcoat_fresnel(surface, v_dot_h);
    let coat_raw = vec3<f32>(fresnel * distribution * visibility);

    // Smooth finite compression preserves a visible varnish highlight without
    // allowing a point-sun GGX peak to become a temporal firefly.
    let coat_luma = dot(coat_raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let coat_cap = 1.0 / (1.0 + coat_luma / 0.3);
    let coat = coat_raw * coat_cap * light_color * intensity * n_dot_l;
    return undercoat * base_attenuation + coat;
}
