//! Path-tracing megakernel (docs/pt/pt-roadmap.md, ticket PT-1).
//!
//! One compute kernel, one ray budget per pixel per frame. Primary hits come
//! from the G-buffer (depth + albedo + material MRTs — free, and sharper than
//! traced primaries); bounce and shadow rays go through the same TLAS the
//! Lumen HW probe trace uses. Hit shading at bounces reads the mesh-card
//! ALBEDO atlas (not the pre-lit radiance atlas: a path tracer computes its
//! own lighting at every vertex of the path — sampling pre-lit cards would
//! bake Lumen's direct light into ours twice).
//!
//! Radiometric convention: light intensities are treated as π-premultiplied,
//! i.e. diffuse contribution is `albedo * L * NdotL` with no 1/π — matching
//! the raster shader (core.rs point-light loop has no 1/π either), so
//! toggling PT on/off does not jump scene brightness. bloom-reference
//! comparisons account for this in scene config (see the PT-1 ticket).
//!
//! Sky pixels are never written: the raster sky/cloud passes already drew
//! them, and PT replacing a procedural cloud deck with an analytic gradient
//! would be a downgrade. PT owns geometry pixels only. The translucent pass
//! runs AFTER this kernel, so water and glass composite over path-traced
//! opaques exactly as they do over raster ones.
//!
//! Debug modes (uniform cfg.w, set via BLOOM_PT_DEBUG):
//!   1 = raw depth visualised          2 = reconstructed world normals
//!   3 = G-buffer albedo               4 = sun shadow-ray visibility
//!   5 = solid magenta (pipeline probe — proves dispatch + write path)

pub(in crate::renderer) const PT_KERNEL_WGSL: &str = r#"
struct PtLight {
    pos_range: vec4<f32>,   // xyz world position, w = range
    color_int: vec4<f32>,   // rgb color, w = intensity
};

struct PtParams {
    inv_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz camera world pos
    sun_dir: vec4<f32>,     // xyz unit vector toward the sun
    sun_color: vec4<f32>,   // rgb premultiplied by intensity
    sky_color: vec4<f32>,   // rgb ambient-derived sky tint
    size: vec4<u32>,        // x=width, y=height, z=frame_index, w=accum_count
    cfg: vec4<f32>,         // x=mode(1|2), y=max_bounces, z=point_light_count, w=debug
    lights: array<PtLight, 16>,
};

// Layout mirror of the Lumen instance data (ssgi.rs) — same buffer.
struct InstanceGiData {
    albedo: vec3<f32>,
    emissive_luma: f32,
    normal_ws: vec3<f32>,
    _pad0: f32,
    card_slot: vec4<f32>,
    card_aabb_min: vec4<f32>,
    card_aabb_max: vec4<f32>,
    world_aabb_min: vec4<f32>,
    world_aabb_max: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: PtParams;
@group(0) @binding(1) var accel: acceleration_structure;
@group(0) @binding(2) var<storage, read> instance_data: array<InstanceGiData>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var albedo_tex: texture_2d<f32>;
@group(0) @binding(5) var material_tex: texture_2d<f32>;
@group(0) @binding(6) var card_albedo_atlas: texture_2d<f32>;
@group(0) @binding(7) var card_samp: sampler;
@group(0) @binding(8) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(9) var out_hdr: texture_storage_2d<rgba16float, write>;

// ---- RNG: PCG, one stream per (pixel, frame) --------------------------------

var<private> rng_state: u32;

fn rng_seed(px: vec2<u32>, frame: u32) {
    var h = px.x * 374761393u + px.y * 668265263u + frame * 2654435761u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    rng_state = h ^ (h >> 16u);
}

fn rand_f() -> f32 {
    // PCG-XSH-RR step.
    let old = rng_state;
    rng_state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    let out = (word >> 22u) ^ word;
    return f32(out) * 2.3283064e-10;   // / 2^32
}

fn rand_2f() -> vec2<f32> { return vec2<f32>(rand_f(), rand_f()); }

// ---- Geometry reconstruction -------------------------------------------------

fn world_at(px: vec2<i32>, depth: f32) -> vec3<f32> {
    let dims = vec2<f32>(f32(u.size.x), f32(u.size.y));
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / dims;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = u.inv_vp * ndc;
    return w.xyz / w.w;
}

fn depth_at(px: vec2<i32>) -> f32 {
    let clamped = clamp(px, vec2<i32>(0), vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1));
    return textureLoad(depth_tex, clamped, 0);
}

fn is_sky(depth: f32) -> bool {
    // Depth-buffer far plane. If the projection turns out reversed-Z the
    // BLOOM_PT_DEBUG=1 depth view makes it obvious in one screenshot; flip
    // here if geometry reads bright and sky reads dark.
    return depth >= 0.9999999;
}

// Screen-space normal from depth: reconstruct neighbours, take the tighter
// derivative on each axis so depth discontinuities don't smear normals
// across silhouettes.
fn normal_from_depth(px: vec2<i32>, p_center: vec3<f32>) -> vec3<f32> {
    let d_l = depth_at(px + vec2<i32>(-1, 0));
    let d_r = depth_at(px + vec2<i32>(1, 0));
    let d_u = depth_at(px + vec2<i32>(0, -1));
    let d_d = depth_at(px + vec2<i32>(0, 1));
    let d_c = depth_at(px);

    var ddx: vec3<f32>;
    if (abs(d_l - d_c) < abs(d_r - d_c)) {
        ddx = p_center - world_at(px + vec2<i32>(-1, 0), d_l);
    } else {
        ddx = world_at(px + vec2<i32>(1, 0), d_r) - p_center;
    }
    var ddy: vec3<f32>;
    if (abs(d_u - d_c) < abs(d_d - d_c)) {
        ddy = p_center - world_at(px + vec2<i32>(0, -1), d_u);
    } else {
        ddy = world_at(px + vec2<i32>(0, 1), d_d) - p_center;
    }
    var n = cross(ddy, ddx);
    let len = length(n);
    if (len < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
    n = n / len;
    // Face the camera: a G-buffer surface always does.
    if (dot(n, u.cam_pos.xyz - p_center) < 0.0) { n = -n; }
    return n;
}

// ---- Sampling helpers ----------------------------------------------------------

// Branchless ONB (Duff et al. 2017).
fn onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn cosine_sample(n: vec3<f32>, r: vec2<f32>) -> vec3<f32> {
    let phi = 6.2831853 * r.x;
    let sr = sqrt(r.y);
    let local = vec3<f32>(cos(phi) * sr, sin(phi) * sr, sqrt(max(0.0, 1.0 - r.y)));
    return normalize(onb(n) * local);
}

// Uniform direction in the solar cone (half-angle 0.265 deg -> soft shadows).
fn sun_cone_sample(r: vec2<f32>) -> vec3<f32> {
    let cos_max = 0.9999893;
    let cos_t = mix(cos_max, 1.0, r.x);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let phi = 6.2831853 * r.y;
    let local = vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
    return normalize(onb(u.sun_dir.xyz) * local);
}

// Sky radiance for a miss. Analytic horizon-to-zenith gradient off the same
// ambient-derived tint Lumen's traces use; the sun disc is deliberately
// absent (the sun is sampled by NEE only, so it cannot be counted twice).
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return u.sky_color.rgb * mix(0.45, 1.35, t);
}

// ---- Ray casts ------------------------------------------------------------------

fn occluded(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> bool {
    var rq: ray_query;
    rayQueryInitialize(&rq, accel, RayDesc(0u, 0xFFu, 0.001, max_t, origin, dir));
    loop {
        if (!rayQueryProceed(&rq)) { break; }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);
    return hit.kind != RAY_QUERY_INTERSECTION_NONE;
}

// ---- Hit shading: card albedo -----------------------------------------------------

// Same signed-axis card projection as the Lumen HW trace (ssgi.rs), but
// sampling the raw ALBEDO atlas. Falls back to the flat instance albedo when
// the mesh has no captured card.
fn albedo_at_hit(
    inst: InstanceGiData,
    hit_os: vec3<f32>,
    dir_ws: vec3<f32>,
) -> vec3<f32> {
    if (inst.card_slot.w <= 0.5) {
        return inst.albedo;
    }
    let abs_d = abs(dir_ws);
    var axis_idx: u32 = 0u;
    if (abs_d.y >= abs_d.x && abs_d.y >= abs_d.z) {
        axis_idx = 2u;
    } else if (abs_d.z >= abs_d.x) {
        axis_idx = 4u;
    }
    var signed_axis: u32 = axis_idx;
    if (axis_idx == 0u && dir_ws.x > 0.0) { signed_axis = 1u; }
    else if (axis_idx == 2u && dir_ws.y > 0.0) { signed_axis = 3u; }
    else if (axis_idx == 4u && dir_ws.z > 0.0) { signed_axis = 5u; }

    let first_slot = u32(inst.card_slot.x);
    let slot = first_slot + signed_axis;
    let slot_x = slot % 64u;
    let slot_y = slot / 64u;

    let bmin = inst.card_aabb_min.xyz;
    let bmax = inst.card_aabb_max.xyz;
    var u_os: f32; var v_os: f32;
    var u_lo: f32; var u_hi: f32; var v_lo: f32; var v_hi: f32;
    var u_flip: f32 = 1.0;
    if (signed_axis == 0u || signed_axis == 1u) {
        u_os = hit_os.y; v_os = hit_os.z;
        u_lo = bmin.y; u_hi = bmax.y; v_lo = bmin.z; v_hi = bmax.z;
        if (signed_axis == 1u) { u_flip = -1.0; }
    } else if (signed_axis == 2u || signed_axis == 3u) {
        u_os = hit_os.x; v_os = hit_os.z;
        u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.z; v_hi = bmax.z;
        if (signed_axis == 3u) { u_flip = -1.0; }
    } else {
        u_os = hit_os.x; v_os = hit_os.y;
        u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.y; v_hi = bmax.y;
        if (signed_axis == 5u) { u_flip = -1.0; }
    }
    var u_norm = clamp((u_os - u_lo) / max(u_hi - u_lo, 1e-4), 0.0, 1.0);
    let v_norm = clamp((v_os - v_lo) / max(v_hi - v_lo, 1e-4), 0.0, 1.0);
    if (u_flip < 0.0) { u_norm = 1.0 - u_norm; }

    let slot_size_uv = 1.0 / 64.0;
    let texel_in_slot = slot_size_uv / 64.0;
    let slot_u0 = f32(slot_x) * slot_size_uv + texel_in_slot;
    let slot_v0 = f32(slot_y) * slot_size_uv + texel_in_slot;
    let slot_span = slot_size_uv - 2.0 * texel_in_slot;
    let atlas_uv = vec2<f32>(slot_u0 + u_norm * slot_span, slot_v0 + v_norm * slot_span);
    return textureSampleLevel(card_albedo_atlas, card_samp, atlas_uv, 0.0).rgb;
}

// ---- Next-event estimation ---------------------------------------------------------

// Direct light at a surface point: sun through the solar cone + one point
// light chosen uniformly (contribution / pdf). Game-radiometry convention:
// no 1/pi (see file header).
fn direct_light(p: vec3<f32>, n: vec3<f32>, alb: vec3<f32>) -> vec3<f32> {
    var lit = vec3<f32>(0.0);

    let sd = sun_cone_sample(rand_2f());
    let ndl = dot(n, sd);
    if (ndl > 0.0 && !occluded(p, sd, 1000.0)) {
        lit += u.sun_color.rgb * ndl;
    }

    let count = u32(u.cfg.z);
    if (count > 0u) {
        let pick = min(u32(rand_f() * f32(count)), count - 1u);
        let l = u.lights[pick];
        let to_l = l.pos_range.xyz - p;
        let d = length(to_l);
        let range = l.pos_range.w;
        if (d < range && d > 1e-3) {
            let dir = to_l / d;
            let ndl2 = dot(n, dir);
            if (ndl2 > 0.0 && !occluded(p, dir, d - 0.02)) {
                // Raster-parity falloff: (1 - d/range)^2, core.rs.
                let att = 1.0 - d / range;
                lit += l.color_int.rgb * l.color_int.w * ndl2 * att * att * f32(count);
            }
        }
    }
    return alb * lit;
}

// ---- Main -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.size.x || gid.y >= u.size.y) { return; }
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    rng_seed(gid.xy, u.size.z);

    let debug = u.cfg.w;
    if (debug == 5.0) {
        textureStore(out_hdr, px, vec4<f32>(1.0, 0.0, 1.0, 1.0));
        return;
    }

    let depth = depth_at(px);
    if (debug == 1.0) {
        textureStore(out_hdr, px, vec4<f32>(vec3<f32>(depth), 1.0));
        return;
    }

    if (is_sky(depth)) {
        // Leave the raster sky/clouds untouched.
        return;
    }

    let p0 = world_at(px, depth);
    let n0 = normal_from_depth(px, p0);
    let albedo0 = textureLoad(albedo_tex, px, 0).rgb;

    if (debug == 2.0) {
        textureStore(out_hdr, px, vec4<f32>(n0 * 0.5 + 0.5, 1.0));
        return;
    }
    if (debug == 3.0) {
        textureStore(out_hdr, px, vec4<f32>(albedo0, 1.0));
        return;
    }
    if (debug == 4.0) {
        let sd = sun_cone_sample(rand_2f());
        let vis = select(0.0, 1.0, !occluded(p0 + n0 * 0.02, sd, 1000.0));
        textureStore(out_hdr, px, vec4<f32>(vec3<f32>(vis), 1.0));
        return;
    }

    // ---- one path sample --------------------------------------------------

    var radiance = direct_light(p0 + n0 * 0.02, n0, albedo0);
    var throughput = albedo0;
    var origin = p0 + n0 * 0.02;
    var n_cur = n0;

    let max_bounces = u32(u.cfg.y);
    for (var b = 0u; b < max_bounces; b = b + 1u) {
        let dir = cosine_sample(n_cur, rand_2f());

        var rq: ray_query;
        rayQueryInitialize(&rq, accel, RayDesc(0u, 0xFFu, 0.001, 500.0, origin, dir));
        loop {
            if (!rayQueryProceed(&rq)) { break; }
        }
        let hit = rayQueryGetCommittedIntersection(&rq);

        if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
            radiance += throughput * sky_radiance(dir);
            break;
        }

        let inst = instance_data[hit.instance_custom_data];
        let hit_ws = origin + dir * hit.t;
        let hit_os = (hit.world_to_object * vec4<f32>(hit_ws, 1.0)).xyz;
        let alb_hit = albedo_at_hit(inst, hit_os, dir);

        // Hit normal: the flat per-instance normal (Tier-1 honesty note in
        // the roadmap; PT-2 replaces this with interpolated vertex normals).
        var n_hit = inst.normal_ws;
        let n_len = length(n_hit);
        if (n_len < 1e-4) {
            n_hit = -dir;
        } else {
            n_hit = n_hit / n_len;
            if (dot(n_hit, dir) > 0.0) { n_hit = -n_hit; }
        }

        // Emissive surfaces radiate; matches the Lumen fallback semantics
        // (albedo * emissive_luma).
        radiance += throughput * inst.albedo * inst.emissive_luma;

        let hit_p = hit_ws + n_hit * 0.02;
        radiance += throughput * direct_light(hit_p, n_hit, alb_hit);

        throughput *= alb_hit;
        origin = hit_p;
        n_cur = n_hit;

        // Russian roulette from the third bounce.
        if (b >= 2u) {
            let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (rand_f() > q) { break; }
            throughput /= q;
        }
    }

    // NaN/Inf guard + firefly cap so one bad sample cannot poison the
    // accumulator forever.
    if (radiance.r != radiance.r || radiance.g != radiance.g || radiance.b != radiance.b) {
        radiance = vec3<f32>(0.0);
    }
    let luma = dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (luma > 32.0) { radiance *= 32.0 / luma; }

    // ---- accumulate ---------------------------------------------------------

    let idx = gid.y * u.size.x + gid.x;
    let mode = u.cfg.x;
    var prev = accum[idx];
    if (u.size.w == 0u) { prev = vec4<f32>(0.0); }

    var out: vec3<f32>;
    if (mode >= 2.0) {
        // Realtime: rolling exponential window. PT-3 replaces this with
        // motion-reprojected accumulation + a-trous.
        let alpha = select(0.125, 1.0, u.size.w == 0u);
        out = mix(prev.rgb, radiance, alpha);
        accum[idx] = vec4<f32>(out, 1.0);
    } else {
        // Progressive: plain running sum; count lives on the CPU.
        let sum = prev.rgb + radiance;
        let n = f32(u.size.w) + 1.0;
        accum[idx] = vec4<f32>(sum, n);
        out = sum / n;
    }

    textureStore(out_hdr, px, vec4<f32>(out, 1.0));
}
"#;
