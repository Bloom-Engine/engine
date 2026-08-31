/// SSR temporal denoiser. Same shape as the SSGI temporal pass:
/// reprojects the previous history through the motion vectors,
/// clamps against the 3×3 neighborhood of the noisy current frame,
/// and blends with a low alpha so 4–8 frames of random GGX rays
/// converge to a smooth reflection. Also pre-filters the noisy
/// current frame by the 3×3 mean, which kills single-pixel
/// glossy-ray sparkles in one frame instead of 10.
pub(in crate::renderer) const SSR_TEMPORAL_SHADER_WGSL: &str = "
struct SsrTemporalParams {
    /// x = blend_alpha (0.1), y = 1 for perspective depth.
    params: vec4<f32>,
    inv_vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsrTemporalParams;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var current_samp: sampler;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var history_samp: sampler;
@group(0) @binding(5) var velocity_tex: texture_2d<f32>;
@group(0) @binding(6) var velocity_samp: sampler;
@group(0) @binding(7) var depth_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let current_raw = textureSampleLevel(current_tex, current_samp, in.uv, 0.0);

    // Reconstruct the current surface once and project the same world point
    // through the previous camera. SSR history alpha stores signed geometric
    // depth (positive = traced hit, negative = env miss), so this adds exact
    // disocclusion provenance without another history texture or attachment.
    let depth_dims = vec2<i32>(textureDimensions(depth_tex));
    let depth_coord = clamp(
        vec2<i32>(floor(in.uv * vec2<f32>(depth_dims))),
        vec2<i32>(0),
        depth_dims - vec2<i32>(1),
    );
    let depth = textureLoad(depth_tex, depth_coord, 0);
    let ndc = vec4<f32>(
        in.uv.x * 2.0 - 1.0,
        (1.0 - in.uv.y) * 2.0 - 1.0,
        depth,
        1.0,
    );
    let world_h = u.inv_vp * ndc;
    let world = world_h.xyz / world_h.w;
    let perspective = u.params.y > 0.5;
    let current_depth_key = select(
        depth,
        select(1.0 / max(abs(world_h.w), 0.000001), 10000.0, depth >= 0.9999),
        perspective,
    );
    let prev_world_clip = u.prev_vp * vec4<f32>(world, 1.0);
    let expected_prev_depth = select(
        prev_world_clip.z / max(abs(prev_world_clip.w), 0.000001),
        select(prev_world_clip.w, 10000.0, depth >= 0.9999),
        perspective,
    );
    let signed_current_depth = select(
        -current_depth_key,
        current_depth_key,
        current_raw.a > 0.001,
    );
    // Derivatives must execute in uniform fragment control flow. Browser
    // WebGPU validates this strictly, so compute the depth footprint before
    // the per-pixel off-screen history early return below.
    let depth_base_tolerance = 0.02 + abs(expected_prev_depth) * 0.005;
    let depth_gradient = abs(dpdx(expected_prev_depth)) + abs(dpdy(expected_prev_depth));

    // 3×3 box pre-filter + neighborhood min/max. One texel spread
    // across 9 samples hides single-pixel glossy-ray sparkles in a
    // single frame; the min/max bounds the history so disocclusion
    // and material transitions clamp rather than ghost.
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(current_tex));
    var nmin = current_raw.rgb;
    var nmax = current_raw.rgb;
    var prefilt = vec3<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let s = textureSampleLevel(current_tex, current_samp, in.uv + vec2<f32>(f32(x), f32(y)) * texel, 0.0);
            nmin = min(nmin, s.rgb);
            nmax = max(nmax, s.rgb);
            prefilt = prefilt + s.rgb;
        }
    }
    let current = prefilt * (1.0 / 9.0);

    // Velocity is full-res; UV mapping handles the half-res delta.
    // NDC-space velocity + UV Y-flip → `uv + vel.y` for the Y axis,
    // matching TAA + SSAO + the sibling SSGI temporal pass.
    let vel = textureSampleLevel(velocity_tex, velocity_samp, in.uv, 0.0).xy;
    let vel_len = length(vel);
    let prev_uv = vec2<f32>(in.uv.x - vel.x, in.uv.y + vel.y);
    let off_screen = prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0;
    if (off_screen) { return vec4<f32>(current, signed_current_depth); }

    let history_raw = textureSampleLevel(history_tex, history_samp, prev_uv, 0.0);
    // Scrub NaN/Inf from the history read. Until a clean SSR frame
    // finishes draining the ping-pong pair, any poisoned history
    // pixel would otherwise survive the clamp (clamp(NaN, a, b) is
    // implementation-defined on Metal — frequently NaN) and keep
    // tonemapping to pink. Replace poisoned channels with the
    // current-frame mean, which is the best available estimate.
    let history = select(current, history_raw.rgb, history_raw.rgb == history_raw.rgb);
    let clamped_history = clamp(history, nmin, nmax);
    let history_depth = abs(history_raw.a);
    // Alpha's sign is free hit provenance: positive means that the ray found
    // screen geometry, negative means that it fell back to the environment.
    // These two estimators can differ substantially even when the receiving
    // surface depth remains valid. Never drag one estimator through a camera
    // move after ownership flips; doing so produced a bright reflection patch
    // that visibly lagged across Bistro's cobbles and walls.
    let current_hit = current_raw.a > 0.001;
    let history_hit = history_raw.a > 0.0;
    let provenance_disocclusion = select(1.0, 0.0, current_hit == history_hit);
    let depth_tolerance = max(
        depth_base_tolerance,
        min(depth_gradient * 2.0, depth_base_tolerance * 4.0),
    );
    let depth_error = abs(history_depth - expected_prev_depth);
    let depth_disocclusion = smoothstep(
        depth_tolerance,
        depth_tolerance * 2.0,
        depth_error,
    );
    // A reflection is view-dependent even when its receiving surface remains
    // geometrically valid. Keeping 90% of the old reflection during a camera
    // move makes highlights visibly lag behind the wall/floor that owns them.
    // Match TAA's qualified motion envelope: static pixels retain the full
    // denoising window, while moving pixels refresh quickly from the already
    // 3x3-prefiltered current result. Downstream TAA still removes residual
    // stochastic noise, so motion does not trade the ghost for shimmer.
    let motion_refresh = smoothstep(0.0005, 0.008, vel_len);
    let motion_alpha = mix(u.params.x, 0.85, motion_refresh);
    let alpha = max(max(motion_alpha, depth_disocclusion), provenance_disocclusion);
    let blended = mix(clamped_history, current, alpha);
    let finite_blended = select(current, blended, blended == blended);
    return vec4<f32>(finite_blended, signed_current_depth);
}
";

/// SSR (screen-space reflections) shader. View-space ray march:
///
/// 1. Reconstruct view-space position from the depth buffer.
/// 2. Reconstruct view-space normal from depth derivatives
///    (cross of dpdx/dpdy of view position).
/// 3. Reflect view direction around N → reflection direction R.
/// 4. March along R in view space, project each step to screen
///    coords, sample depth there, hit if our marched z is past the
///    sampled surface.
/// 5. On hit, sample the HDR RT at the hit UV and output it
///    (faded toward edges of screen so off-screen reflections
///    don't pop into existence).
///
/// Output is half-res HDR. The TAA pass adds it on top of the
/// prefiltered IBL specular for the final image.
pub(in crate::renderer) const SSR_SHADER_WGSL: &str = "
struct SsrParams {
    /// Inverse of the projection matrix — depth → view-space pos.
    inv_proj: mat4x4<f32>,
    /// Projection matrix — view-space pos → clip-space.
    proj: mat4x4<f32>,
    /// x = SSR strength (0 = off, 1 = full)
    /// y = max march distance in view-space units
    /// z = number of march steps
    /// w = frame index (Hammersley rotation + march jitter)
    params: vec4<f32>,
    /// EN-021 — view→world ROTATION (inverse of the view matrix's 3×3)
    /// so the env-miss fallback can turn the view-space reflection ray
    /// into a world direction for the equirect lookup.
    inv_view_rot: mat4x4<f32>,
    /// EN-021 — x = env max LOD (matches the material path's
    /// roughness×6 mip ramp), y = env intensity, zw unused.
    params2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsrParams;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var hdr_tex: texture_2d<f32>;
@group(0) @binding(4) var hdr_samp: sampler;
@group(0) @binding(5) var mat_tex: texture_2d<f32>;
@group(0) @binding(6) var mat_samp: sampler;
@group(0) @binding(7) var albedo_tex: texture_2d<f32>;
@group(0) @binding(8) var albedo_samp: sampler;
@group(0) @binding(9) var env_tex: texture_2d<f32>;
@group(0) @binding(10) var env_samp: sampler;

const PI: f32 = 3.14159265;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_h = u.inv_proj * ndc;
    return view_h.xyz / view_h.w;
}

/// EN-021 — env-miss fallback. The scene shader scales its IBL specular
/// down by SSR's ownership share (lighting.dir_light_count.z × the same
/// roughness fade this shader uses), so a miss MUST return the env
/// sample instead of black or off-screen reflections go dark. Same
/// equirect mapping as common/pbr.wgsl's sample_env; explicit-LOD
/// sampling needs no seam handling.
fn env_fallback(r_view: vec3<f32>, roughness: f32) -> vec3<f32> {
    let d = normalize((u.inv_view_rot * vec4<f32>(r_view, 0.0)).xyz);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let phi   = atan2(d.z, d.x);
    let uu    = phi / (2.0 * PI);
    let uv    = vec2<f32>(uu - floor(uu), theta / PI);
    return textureSampleLevel(env_tex, env_samp, uv, roughness * u.params2.x).rgb
         * u.params2.y;
}

/// Match the scene material path's continuous HDR specular compression.
/// Without this, enabling SSR replaces bounded split-sum IBL with an
/// unbounded environment miss, so a stationary smooth dielectric changes
/// brightness merely because reflection ownership moved between passes.
fn compress_environment_specular(radiance: vec3<f32>) -> vec3<f32> {
    let luma = dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Keep this knee identical to core.rs's split-sum IBL. A different knee
    // makes a reflection change brightness when SSR switches between an
    // on-screen hit and its environment fallback.
    return radiance * (1.0 / (1.0 + luma / 4.0));
}

/// Interleaved gradient noise — per-pixel pseudo-random in [0, 1).
/// Varies with frame so the temporal accumulator averages over
/// different march offsets each frame.
fn ign_jitter(frag_coord: vec2<f32>, frame: f32) -> f32 {
    let shifted = frag_coord + vec2<f32>(frame * 5.588238, frame * 3.127137);
    return fract(52.9829189 * fract(0.06711056 * shifted.x + 0.00583715 * shifted.y));
}

/// Cheap 2D hash → two independent low-discrepancy values in [0,1)².
/// Used as GGX microfacet-sample coordinates; the frame index rotates
/// the hash so each pixel draws a different sample every frame and
/// the temporal denoiser averages over the GGX lobe.
fn hash2(frag_coord: vec2<f32>, frame: f32) -> vec2<f32> {
    let p1 = frag_coord + vec2<f32>(frame * 11.13, frame * 7.77);
    let p2 = frag_coord + vec2<f32>(frame * 3.17,  frame * 5.29);
    let a = fract(sin(dot(p1, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let b = fract(sin(dot(p2, vec2<f32>(37.7191, 17.1123))) * 28471.1713);
    return vec2<f32>(a, b);
}

/// GGX importance-sampled microfacet half-vector in tangent space
/// aligned to the surface normal. Isotropic GGX — α = roughness².
fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h_local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * h_local.x + b * h_local.y + n * h_local.z);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSampleLevel(depth_tex, depth_samp, in.uv, 0i);
    // Derivatives must come from uniform control flow (WGSL uniformity
    // analysis; Tint/WebGPU enforces what naga lets slide) — take them
    // BEFORE the early returns. A few ALU on sky pixels, and the quad
    // derivatives are actually well-defined now instead of reading
    // helper-lane garbage at the sky/geometry boundary.
    let view_pos = view_pos_from_depth(in.uv, depth);
    let dx = dpdx(view_pos);
    let dy = dpdy(view_pos);
    if (depth >= 0.9999) { return vec4<f32>(0.0); } // sky

    // SSR for every smooth-enough surface — the metals-only gate is gone.
    // EN-021 EXCLUSIVE OWNERSHIP: within this shader's roughness_fade
    // range, SSR owns specular reflections outright — the scene shader
    // scales its IBL specular by the complement of the same fade
    // (× strength, piped through lighting.dir_light_count.z), and a
    // march MISS falls back to the env sample here instead of black.
    // Metals previously double-counted on hit (IBL spec in hdr + full
    // SSR on top, round-2 audit F10); dielectrics were starved of IBL
    // spec by design and now get the coherent hit-or-env behaviour too.
    // Very rough surfaces still fade to pure IBL where one-ray-per-pixel
    // SSR noise would dominate even after temporal accumulation.
    let mat = textureSampleLevel(mat_tex, mat_samp, in.uv, 0.0).rg;
    let metallic = mat.r;
    let roughness = mat.g;
    let albedo = textureSampleLevel(albedo_tex, albedo_samp, in.uv, 0.0).rgb;
    // Quarter-resolution, one-ray SSR cannot converge a wide GGX lobe without
    // visible low-frequency blobs. Hand rough surfaces back to the stable,
    // prefiltered environment before that point. Smooth glass, paint and metal
    // retain SSR; Bistro's 0.73-rough cobbles and foliage remain pure IBL.
    let roughness_fade = 1.0 - smoothstep(0.45, 0.70, roughness);
    if (roughness_fade <= 0.001) { return vec4<f32>(0.0); }

    let v = normalize(-view_pos);
    // Fragment coordinates increase downward, so `dpdy(view_pos)` points
    // opposite the view-space +Y direction. `cross(dx, dy)` therefore
    // reconstructs the back face of every camera-visible surface and drives
    // NdotV to zero. Schlick Fresnel then becomes 1.0 across the frame,
    // turning SSR into a full-strength pale environment overlay. Reverse the
    // operands and defensively face the derivative normal toward the camera
    // at depth discontinuities.
    let n_raw = normalize(cross(dy, dx));
    let n = select(-n_raw, n_raw, dot(n_raw, v) >= 0.0);

    // Stochastic SSR — cast one GGX-importance-sampled ray per pixel
    // per frame. Different frames draw from different points on the
    // GGX lobe (rotated by frame index) so the downstream temporal
    // denoiser averages a dense roughness cone over 4–8 frames. This
    // replaces the 5-tap Gaussian blur at the hit: we pay one ray +
    // one hdr sample per frame, not 32 march steps + 5 blur taps.
    //
    // xi is clamped away from exact 0 and 1. At roughness → 0 the GGX
    // denominator `1 + (α²-1)·xi.y` collapses to 0 when xi.y = 1, so
    // cos_theta becomes sqrt(0/0) = NaN. That NaN then propagates into
    // ssr_history and each 4× upsampled texel turns into a pink hot
    // pixel after tonemapping. Sponza's mirror-smooth lamp fittings
    // (roughness near 0) are exactly the worst case.
    let xi = clamp(hash2(in.clip_pos.xy, u.params.w), vec2<f32>(1e-4), vec2<f32>(0.9999));
    let h = importance_sample_ggx(xi, n, roughness);
    let r = reflect(-v, h);

    let n_dot_v = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);

    // Camera-facing rays can't be marched — env fallback (EN-021).
    if (r.z > 0.0) {
        let fb = compress_environment_specular(
            env_fallback(r, roughness) * fresnel,
        ) * roughness_fade * u.params.x;
        let fb_safe = select(vec3<f32>(0.0), fb, fb == fb);
        return vec4<f32>(fb_safe, 0.0);
    }

    let max_dist = u.params.y;
    let n_steps_f = u.params.z;
    let n_steps = u32(n_steps_f);
    let step_size = max_dist / n_steps_f;

    let jitter = ign_jitter(in.clip_pos.xy, u.params.w);
    var t = step_size * (0.5 + jitter);

    var hit_uv = vec2<f32>(-1.0);
    var hit_found = false;
    var prev_t = 0.0;
    // FXC (the legacy HLSL compiler used by D3D11 + DX12 fallback in wgpu) refuses
    // to unroll a loop that contains an implicit-gradient texture sample when the
    // iteration count is uniform-driven, and refuses to *not* unroll because the
    // body has the gradient op — the only escape is to take the gradient out of
    // the loop. textureSampleLevel forces explicit LOD and removes the gradient
    // op, which is also what we want here (depth has no mips).
    for (var i = 0u; i < n_steps; i = i + 1u) {
        let ray_view = view_pos + r * t;
        let ray_clip = u.proj * vec4<f32>(ray_view, 1.0);
        let ray_ndc = ray_clip.xyz / ray_clip.w;
        if (ray_ndc.x < -1.0 || ray_ndc.x > 1.0 ||
            ray_ndc.y < -1.0 || ray_ndc.y > 1.0 ||
            ray_ndc.z < 0.0 || ray_ndc.z > 1.0) {
            break;
        }
        let ray_uv = vec2<f32>(ray_ndc.x * 0.5 + 0.5, 1.0 - (ray_ndc.y * 0.5 + 0.5));
        let scene_depth = textureSampleLevel(depth_tex, depth_samp, ray_uv, 0i);

        if (ray_ndc.z >= scene_depth) {
            let hit_view = view_pos_from_depth(ray_uv, scene_depth);
            let thickness = abs(ray_view.z - hit_view.z);
            let step_world = t - prev_t;
            if (thickness < step_world * 2.0 + 0.1) {
                hit_uv = ray_uv;
                hit_found = true;
            }
            break;
        }
        prev_t = t;
        t = t + step_size;
    }
    if (!hit_found) {
        // March left the screen or found nothing — env fallback (EN-021).
        let fb = compress_environment_specular(
            env_fallback(r, roughness) * fresnel,
        ) * roughness_fade * u.params.x;
        let fb_safe = select(vec3<f32>(0.0), fb, fb == fb);
        return vec4<f32>(fb_safe, 0.0);
    }

    let edge_fade = min(
        min(hit_uv.x, 1.0 - hit_uv.x),
        min(hit_uv.y, 1.0 - hit_uv.y),
    ) * 10.0;
    let fade = clamp(edge_fade, 0.0, 1.0);

    // NaN scrubber: WGSL has no isnan(), but NaN == NaN is false for
    // every compliant backend, so a componentwise self-compare gives us
    // a vec3<bool> that is true iff each channel is finite. This nukes
    // the one stray NaN/Inf pixel per few thousand rays that would
    // otherwise ping-pong through ssr_history and tonemap to pink. Same
    // self-compare is applied to the HDR tap in case upstream writes a
    // bad sample (autoexposure ratios, rare shader ops on degenerate
    // triangles, etc).
    let raw = textureSampleLevel(hdr_tex, hdr_samp, hit_uv, 0.0).rgb;
    let reflected = select(vec3<f32>(0.0), raw, raw == raw);
    let out = reflected * fresnel * roughness_fade * u.params.x * fade;
    // EN-061: bound a rare bright hit before it can poison the temporal
    // history and become a quarter-resolution block after TSR. Eight linear
    // luminance units remain far above display white and preserve bloom from
    // valid polished-metal reflections; ordinary hits are byte-for-byte
    // unchanged.
    let out_luma = dot(out, vec3<f32>(0.2126, 0.7152, 0.0722));
    let firefly_cap = 8.0;
    let firefly_scale = select(
        1.0,
        firefly_cap / max(out_luma, 0.0001),
        out_luma > firefly_cap,
    );
    let out_bounded = out * firefly_scale;
    let out_safe = select(vec3<f32>(0.0), out_bounded, out_bounded == out_bounded);
    return vec4<f32>(out_safe, fade);
}
";
