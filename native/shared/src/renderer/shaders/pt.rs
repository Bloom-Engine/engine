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
//!   6 = traced-primary interpolated normal (compare against 2;
//!       magenta = TLAS miss where G-buffer had geometry, orange = hit
//!       instance without a geometry window)
//!   7 = traced-primary textured hit albedo (compare against 3;
//!       yellow = adapter lacks texture-array features)
//!   8-15 = binary/quantized probes from the DX12 bring-up (hit-window
//!       flag, normal axes, primitive/instance banding, two-query
//!       aliasing, t-vs-G-buffer sanity, t contours). 13 is the
//!       keeper: green = traced t agrees with the G-buffer, red =
//!       mismatch, blue = miss.
//!   16/17 = NUMERIC dumps via the accum buffer + CPU readback
//!       (pt_trace_dump.txt): 16 = t/instance/prim/kind, 17 = p0 +
//!       raw depth. These found the transposed inv_vp: when every
//!       probe looks "constant", dump numbers before theorizing.

pub(in crate::renderer) const PT_KERNEL_WGSL: &str = r#"
struct PtLight {
    pos_range: vec4<f32>,   // xyz world position, w = range
    color_int: vec4<f32>,   // rgb color, w = intensity
};

struct PtParams {
    inv_vp: mat4x4<f32>,
    // PT-3: previous frame's UNJITTERED view-projection — reprojects
    // this frame's world positions into last frame's screen for
    // temporal history fetch in realtime mode.
    prev_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz camera world pos
    sun_dir: vec4<f32>,     // xyz unit vector toward the sun
    sun_color: vec4<f32>,   // rgb premultiplied by intensity
    sky_color: vec4<f32>,   // rgb ambient-derived sky tint
    size: vec4<u32>,        // x/y = TRACE grid dims, z=frame_index, w=accum_count
    cfg: vec4<f32>,         // x=mode(1|2), y=max_bounces, z=point_light_count, w=debug
    // PT-3 half-res: x/y = full G-buffer dims. z = 1 -> hybrid sun
    // (sample the raster shadow cascades instead of tracing the sun;
    // crisp noise-free direct shadows, rays spent on indirect only).
    // w = 1 -> ReSTIR DI (PT-4, experimental).
    ext: vec4<u32>,
    // Raster shadow cascade view-projections. Uploaded RAW, like
    // prev_vp: mat4_multiply products are already in WGSL M*v layout;
    // only mat4_invert outputs (inv_vp) upload transposed.
    shadow_vps: array<mat4x4<f32>, 3>,
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
    // PT-2: x = vertex_base, y = index_base, z = index_count (0 = no
    // geometry window -> PT-1 fallback), w = albedo texture index.
    geo: vec4<u32>,
    // PT-2: x = roughness, y = metalness.
    mat_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: PtParams;
@group(0) @binding(1) var accel: acceleration_structure;
@group(0) @binding(2) var<storage, read> instance_data: array<InstanceGiData>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var albedo_tex: texture_2d<f32>;
@group(0) @binding(5) var material_tex: texture_2d<f32>;
@group(0) @binding(6) var card_albedo_atlas: texture_2d<f32>;
@group(0) @binding(7) var card_samp: sampler;
// PT-3: ping-pong accumulation. Binding 8 = previous frame's buffer
// (read), binding 13 = this frame's output. Reprojection reads OTHER
// pixels from prev, which a single read_write buffer cannot do safely.
//
// Layout (SVGF): accum = (irradiance rgb, luminance variance);
// moments = (mu1, mu2, history length, raw depth). Progressive mode
// keeps its original (radiance sum, sample count) layout in accum and
// leaves the moments buffers untouched.
@group(0) @binding(8) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(9) var out_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read_write> accum_out: array<vec4<f32>>;
@group(0) @binding(18) var<storage, read_write> moments: array<vec4<f32>>;
@group(0) @binding(19) var<storage, read_write> moments_out: array<vec4<f32>>;
// PT-4 (EXPERIMENTAL, ext.w == 1) — ReSTIR DI reservoirs, ping-pong
// with the accum pair: (light index, W, M, target pdf) per trace texel.
@group(0) @binding(20) var<storage, read_write> resv: array<vec4<f32>>;
@group(0) @binding(21) var<storage, read_write> resv_out: array<vec4<f32>>;
// PT-7 — the raster velocity MRT (uv-space delta, current − previous,
// no Y flip at write; see core.rs). Non-zero where a surface MOVED —
// reprojection follows it instead of the camera-only prev_vp math,
// exactly like TAA, so moving skinned characters keep their history.
@group(0) @binding(22) var velocity_tex: texture_2d<f32>;

// Reprojection of the current surface into the previous frame's trace
// grid — computed once per texel (module privates because both the
// ReSTIR temporal reuse and the SVGF colour accumulation consume it).
var<private> rp_valid: bool;
var<private> rp_base: vec2<i32>;
var<private> rp_fr: vec2<f32>;
var<private> rp_zl_here: f32;
var<private> rp_nearest: u32;

fn compute_reproj(p0: vec3<f32>, px_full: vec2<i32>, depth_cur: f32) {
    rp_valid = false;
    if (u.size.w == 0u) { return; }
    // PT-7 — object motion first: the velocity buffer knows how THIS
    // pixel's surface moved (including skeletal motion, which no
    // camera matrix can express). TAA's convention:
    // prev_uv = (uv.x - vel.x, uv.y + vel.y). Camera-only pixels
    // write ~zero velocity and fall through to the prev_vp math.
    var uv_prev: vec2<f32>;
    var zl_here: f32;
    let vel = textureLoad(velocity_tex, px_full, 0).rg;
    if (abs(vel.x) + abs(vel.y) > 1e-5) {
        let uv_cur = (vec2<f32>(px_full) + 0.5)
            / vec2<f32>(f32(u.ext.x), f32(u.ext.y));
        uv_prev = vec2<f32>(uv_cur.x - vel.x, uv_cur.y + vel.y);
        // Depth along a moving surface changes slowly frame-to-frame;
        // the tap tolerance absorbs it, and a fast approach degrades
        // to a disocclusion reset — the safe direction.
        zl_here = lin_depth(depth_cur);
    } else {
        let clip_prev = u.prev_vp * vec4<f32>(p0, 1.0);
        if (clip_prev.w <= 1e-4) { return; }
        let ndc_prev = clip_prev.xyz / clip_prev.w;
        uv_prev = vec2<f32>(ndc_prev.x * 0.5 + 0.5, 0.5 - ndc_prev.y * 0.5);
        zl_here = lin_depth(ndc_prev.z);
    }
    if (uv_prev.x < 0.0 || uv_prev.x >= 1.0 || uv_prev.y < 0.0 || uv_prev.y >= 1.0) {
        return;
    }
    let pos = uv_prev * vec2<f32>(f32(u.size.x), f32(u.size.y)) - 0.5;
    rp_base = vec2<i32>(floor(pos));
    rp_fr = pos - floor(pos);
    rp_zl_here = zl_here;
    let np = vec2<u32>(
        min(u32(max(rp_base.x + i32(round(rp_fr.x)), 0)), u.size.x - 1u),
        min(u32(max(rp_base.y + i32(round(rp_fr.y)), 0)), u.size.y - 1u),
    );
    rp_nearest = np.y * u.size.x + np.x;
    rp_valid = true;
}

// Approximate linear view distance from the raw depth-buffer value
// (GL-convention matrix, near 0.01: z_view ~= 2n / (1 - d)). Only used
// for RELATIVE history-validation comparisons.
fn lin_depth(d: f32) -> f32 {
    return 0.02 / max(1.0 - d, 1e-6);
}
// PT-2: geometry megabuffers. geo_v holds raw Vertex3D words (stride 24
// f32: position +0, normal +3, color +6, uv +10, ...); geo_i holds the
// concatenated index streams. Windows are per-instance via inst.geo.
// (Binding 12, the texture array + PT_HAS_TEXTURES + pt_tex_sample, is
// appended by the Rust side per adapter support.)
@group(0) @binding(10) var<storage, read> geo_v: array<f32>;
@group(0) @binding(11) var<storage, read> geo_i: array<u32>;
// Hybrid sun (ext.z == 1): the raster shadow cascades.
@group(0) @binding(14) var shadow_atlas_0: texture_depth_2d;
@group(0) @binding(15) var shadow_atlas_1: texture_depth_2d;
@group(0) @binding(16) var shadow_atlas_2: texture_depth_2d;
@group(0) @binding(17) var shadow_samp: sampler_comparison;

// Sun visibility from the shadow cascades (near -> far fallthrough by
// coverage, same scheme as the WSRC bake). Deterministic and smooth —
// the whole reason RT mode's direct light doesn't shimmer or dither.
fn sun_vis_cascade(pos_ws: vec3<f32>) -> f32 {
    for (var c = 0; c < 3; c = c + 1) {
        var clip: vec4<f32>;
        if (c == 0) { clip = u.shadow_vps[0] * vec4<f32>(pos_ws, 1.0); }
        else if (c == 1) { clip = u.shadow_vps[1] * vec4<f32>(pos_ws, 1.0); }
        else { clip = u.shadow_vps[2] * vec4<f32>(pos_ws, 1.0); }
        if (abs(clip.w) < 1e-6) { continue; }
        let ndc = clip.xyz / clip.w;
        if (ndc.x < -0.99 || ndc.x > 0.99 || ndc.y < -0.99 || ndc.y > 0.99 || ndc.z < 0.0 || ndc.z > 1.0) {
            continue;
        }
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let ref_depth = ndc.z - 0.002;
        // Manual load-and-compare with a 2x2 average instead of the
        // comparison sampler: SampleCmp from a COMPUTE stage proved
        // unreliable on this DXC path (constant 0, independent of the
        // matrices — same failure shape as the ray-query saga), and the
        // only other compute-stage user (WSRC bake) was never validated
        // on DX12. textureLoad is proven (the PT depth reads use it).
        var dims: vec2<u32>;
        if (c == 0) { dims = textureDimensions(shadow_atlas_0); }
        else if (c == 1) { dims = textureDimensions(shadow_atlas_1); }
        else { dims = textureDimensions(shadow_atlas_2); }
        let fdims = vec2<f32>(dims);
        var vis = 0.0;
        for (var ty = 0; ty <= 1; ty = ty + 1) {
            for (var tx = 0; tx <= 1; tx = tx + 1) {
                let tc = clamp(
                    vec2<i32>(uv * fdims - vec2<f32>(0.5)) + vec2<i32>(tx, ty),
                    vec2<i32>(0),
                    vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1),
                );
                var stored: f32;
                if (c == 0) { stored = textureLoad(shadow_atlas_0, tc, 0); }
                else if (c == 1) { stored = textureLoad(shadow_atlas_1, tc, 0); }
                else { stored = textureLoad(shadow_atlas_2, tc, 0); }
                if (ref_depth <= stored) { vis += 0.25; }
            }
        }
        return vis;
    }
    return 1.0;
}

const PT_VSTRIDE: u32 = 24u;

struct HitAttrs {
    normal_os: vec3<f32>,
    uv: vec2<f32>,
};

fn vert_normal_os(slot: u32) -> vec3<f32> {
    let o = slot * PT_VSTRIDE + 3u;
    return vec3<f32>(geo_v[o], geo_v[o + 1u], geo_v[o + 2u]);
}

fn vert_uv(slot: u32) -> vec2<f32> {
    let o = slot * PT_VSTRIDE + 10u;
    return vec2<f32>(geo_v[o], geo_v[o + 1u]);
}

// Interpolate the hit triangle's vertex normal + UV. DXR/Vulkan
// barycentric convention: (u, v) weight vertices 1 and 2, w = 1-u-v
// weights vertex 0.
fn fetch_hit_attrs(geo: vec4<u32>, prim: u32, bary: vec2<f32>) -> HitAttrs {
    let base = geo.y + prim * 3u;
    let s0 = geo.x + geo_i[base];
    let s1 = geo.x + geo_i[base + 1u];
    let s2 = geo.x + geo_i[base + 2u];
    let w = 1.0 - bary.x - bary.y;
    var a: HitAttrs;
    a.normal_os = w * vert_normal_os(s0) + bary.x * vert_normal_os(s1) + bary.y * vert_normal_os(s2);
    a.uv = w * vert_uv(s0) + bary.x * vert_uv(s1) + bary.y * vert_uv(s2);
    return a;
}

// Object-space normal -> world space: with M = object_to_world the
// correct transform is (M^-1)^T, and the ray query hands us M^-1 as
// world_to_object. `v * mat3` multiplies by the transpose in WGSL.
fn normal_to_world(n_os: vec3<f32>, w2o: mat4x3<f32>) -> vec3<f32> {
    let lin = mat3x3<f32>(w2o[0], w2o[1], w2o[2]);
    let n = n_os * lin;
    let len = length(n);
    if (len < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
    return n / len;
}

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

// Interleaved gradient noise, scrolled per frame by the golden-ratio
// offset. Spatially STRUCTURED (neighbors get well-distributed values)
// so a 5x5 filter averages it nearly flat — white PCG noise leaves
// mid-frequency blotch at 1-2 spp that shimmers. Used for the primary
// sun test in realtime mode.
fn ign_at(px: vec2<i32>, frame: u32) -> f32 {
    let p = vec2<f32>(px) + f32(frame % 64u) * 5.588238;
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// ---- Geometry reconstruction -------------------------------------------------

// px here is always a FULL-resolution G-buffer pixel (u.ext dims); the
// trace grid may be half of that in realtime mode.
fn world_at(px: vec2<i32>, depth: f32) -> vec3<f32> {
    let dims = vec2<f32>(f32(u.ext.x), f32(u.ext.y));
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / dims;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = u.inv_vp * ndc;
    return w.xyz / w.w;
}

fn depth_at(px: vec2<i32>) -> f32 {
    let clamped = clamp(px, vec2<i32>(0), vec2<i32>(i32(u.ext.x) - 1, i32(u.ext.y) - 1));
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

// ---- GGX BRDF sampling (PT-2; port of bloom-reference sample_brdf) --------

fn fresnel_schlick3(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m2 = m * m;
    return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

fn smith_g1(n_dot_x: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let inner = sqrt((1.0 - a2) * n_dot_x * n_dot_x + a2);
    return 2.0 * n_dot_x / (n_dot_x + inner + 1e-6);
}

fn v_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = n_dot_l * sqrt((n_dot_v * (1.0 - a2) + a2) * n_dot_v);
    let ggx_l = n_dot_v * sqrt((n_dot_l * (1.0 - a2) + a2) * n_dot_l);
    return 0.5 / (ggx_v + ggx_l + 1e-6);
}

fn burley_diffuse(n_dot_l: f32, n_dot_v: f32, l_dot_h: f32, roughness: f32) -> f32 {
    let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * roughness;
    let ml = pow(1.0 - n_dot_l, 5.0);
    let mv = pow(1.0 - n_dot_v, 5.0);
    return (1.0 + (fd90 - 1.0) * ml) * (1.0 + (fd90 - 1.0) * mv) / 3.14159265;
}

// Heitz 2018 VNDF sampler — visible-normal distribution, tangent frame.
fn sample_ggx_vndf(v_t: vec3<f32>, alpha: f32, r2: vec2<f32>) -> vec3<f32> {
    let vh = normalize(vec3<f32>(alpha * v_t.x, alpha * v_t.y, v_t.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (lensq > 0.0) {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq);
    }
    let t2 = cross(vh, t1);
    let r = sqrt(r2.x);
    let phi = 6.2831853 * r2.y;
    let t1v = r * cos(phi);
    var t2v = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    t2v = (1.0 - s) * sqrt(max(0.0, 1.0 - t1v * t1v)) + s * t2v;
    let nh = t1v * t1 + t2v * t2 + sqrt(max(0.0, 1.0 - t1v * t1v - t2v * t2v)) * vh;
    return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(nh.z, 0.0)));
}

struct BrdfSample {
    dir: vec3<f32>,
    // BRDF * cos / pdf, physical convention. For the pure-diffuse case
    // this reduces to plain albedo, so the game's pi-premultiplied
    // light intensities are unaffected.
    weight: vec3<f32>,
    valid: bool,
};

fn sample_brdf(
    n: vec3<f32>,
    view_ws: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
) -> BrdfSample {
    var out: BrdfSample;
    out.valid = false;
    let alpha = max(roughness * roughness, 1e-3);
    let m = onb(n); // columns (t, bt, n): local -> world
    let v_t = vec3<f32>(dot(view_ws, m[0]), dot(view_ws, m[1]), dot(view_ws, n));
    if (v_t.z <= 0.0) {
        return out;
    }
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    // Lobe pick by Fresnel at the ACTUAL view angle, not at normal
    // incidence: at grazing angles specular energy approaches 1, and
    // the estimator divides by the pick probability — picking with the
    // ~0.04 normal-incidence weight amplified rare grazing specular
    // samples ~25x into a field of white fireflies at 1-2 spp (the
    // whole ground plane is grazing at distance). Clamped so neither
    // lobe's 1/p boost can exceed ~20x even in edge cases.
    let n_dot_v_pick = max(dot(n, view_ws), 0.0);
    let f_view = fresnel_schlick3(n_dot_v_pick, f0);
    let spec_weight = (f_view.x + f_view.y + f_view.z) / 3.0;
    let diff_weight = (1.0 - spec_weight) * (1.0 - metallic);
    var p_spec = spec_weight / (spec_weight + diff_weight + 1e-6);
    p_spec = clamp(p_spec, 0.05, 0.95);
    let r2 = rand_2f();
    if (rand_f() < p_spec) {
        let h_t = sample_ggx_vndf(v_t, alpha, r2);
        let l_t = reflect(-v_t, h_t);
        if (l_t.z <= 0.0) {
            return out;
        }
        let n_dot_l = l_t.z;
        let n_dot_v = max(v_t.z, 1e-4);
        let v_dot_h = max(dot(v_t, h_t), 1e-4);
        let f = fresnel_schlick3(v_dot_h, f0);
        // VNDF pdf: throughput collapses to F * G2 / G1(V).
        let g2 = v_smith(n_dot_v, n_dot_l, alpha) * 4.0 * n_dot_v * n_dot_l;
        let g1_v = smith_g1(n_dot_v, alpha);
        out.dir = m * l_t;
        out.weight = f * g2 / (max(g1_v, 1e-6) * p_spec);
        // Realtime mode trades a little energy for stability: a single
        // bounce may not multiply throughput more than 4x (the ~7-frame
        // EMA window cannot average outliers away like progressive
        // accumulation can). Progressive mode stays unclamped.
        if (u.cfg.x >= 2.0) {
            out.weight = min(out.weight, vec3<f32>(4.0));
        }
        out.valid = true;
        return out;
    }
    // Diffuse lobe: cosine hemisphere; weight = albedo * burley * pi
    // (Burley divides by pi internally; pdf = cos/pi cancels the cos).
    let r = sqrt(r2.x);
    let phi = 6.2831853 * r2.y;
    let l_t = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - r2.x)));
    let n_dot_l = max(l_t.z, 1e-4);
    let n_dot_v = max(v_t.z, 1e-4);
    let h_un = v_t + l_t;
    var l_dot_h = 0.0;
    if (dot(h_un, h_un) > 1e-8) {
        l_dot_h = max(dot(l_t, normalize(h_un)), 0.0);
    }
    let diffuse_albedo = base_color * (1.0 - metallic) * (vec3<f32>(1.0) - f0);
    let fd = burley_diffuse(n_dot_l, n_dot_v, l_dot_h, roughness);
    out.dir = m * l_t;
    out.weight = diffuse_albedo * fd * 3.14159265 / (1.0 - p_spec);
    if (u.cfg.x >= 2.0) {
        out.weight = min(out.weight, vec3<f32>(4.0));
    }
    out.valid = true;
    return out;
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
// Sun visibility at a surface point: shadow cascades in hybrid mode
// (deterministic, matches the raster shadows exactly), a traced cone
// ray otherwise (reference quality, soft penumbra).
fn sun_visibility(p: vec3<f32>, n: vec3<f32>, r2: vec2<f32>) -> f32 {
    if (u.ext.z == 1u) {
        return sun_vis_cascade(p);
    }
    let sd = sun_cone_sample(r2);
    if (dot(n, sd) <= 0.0) {
        return 0.0;
    }
    if (occluded(p, sd, 1000.0)) {
        return 0.0;
    }
    return 1.0;
}

// GGX highlight for an NEE light sample — the same D/F/V terms as
// sample_brdf, evaluated for a known light direction. The shadow ray is
// already paid for by the diffuse term, so specular NEE rides along
// free (PT-5: point lights and bounce vertices were diffuse-only, the
// documented PT-2 gap). Analytic lights cannot be hit by BSDF rays and
// sky misses exclude the sun disc, so nothing double-counts.
fn nee_spec(n: vec3<f32>, view: vec3<f32>, ldir: vec3<f32>, ndl: f32,
            full_alb: vec3<f32>, rough: f32, metal: f32) -> vec3<f32> {
    let hv = normalize(view + ldir);
    let ndv = max(dot(n, view), 1e-4);
    let ndh = max(dot(n, hv), 0.0);
    let vdh = max(dot(view, hv), 1e-4);
    let alpha0 = max(rough * rough, 1e-3);
    let a2 = alpha0 * alpha0;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let dterm = a2 / (3.14159265 * dd * dd);
    let f0s = mix(vec3<f32>(0.04), full_alb, metal);
    return fresnel_schlick3(vdh, f0s) * dterm * v_smith(ndv, ndl, alpha0) * ndl;
}

fn direct_light(p: vec3<f32>, n: vec3<f32>, alb: vec3<f32>, sun_r2: vec2<f32>,
                view: vec3<f32>, full_alb: vec3<f32>, rough: f32, metal: f32,
                with_points: bool) -> vec3<f32> {
    var lit = vec3<f32>(0.0);
    var spec = vec3<f32>(0.0);

    let ndl = max(dot(n, u.sun_dir.xyz), 0.0);
    if (ndl > 0.0) {
        let vis = sun_visibility(p, n, sun_r2);
        lit += u.sun_color.rgb * ndl * vis;
        if (vis > 0.0) {
            spec += nee_spec(n, view, u.sun_dir.xyz, ndl, full_alb, rough, metal)
                * u.sun_color.rgb * vis;
        }
    }

    let count = u32(u.cfg.z);
    if (count > 0u && with_points) {
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
                let li = l.color_int.rgb * l.color_int.w * att * att * f32(count);
                lit += li * ndl2;
                spec += nee_spec(n, view, dir, ndl2, full_alb, rough, metal) * li;
            }
        }
    }
    return alb * lit + spec;
}

// ---- PT-4 (EXPERIMENTAL) — ReSTIR DI over the analytic point lights -------
//
// RIS with 8 uniform candidates + temporal reservoir reuse (M-capped at
// 20x), one shadow ray for the winner. The target is re-evaluated at
// the CURRENT shading point when merging history, so the temporal reuse
// carries no geometric bias; visibility reuse bias does not arise
// because visibility is never folded into the reservoir. With this
// game's <=16 analytic lights plain NEE is nearly as good (the roadmap
// said so up front) — this lands the architecture for the day emissive
// particles/muzzle flashes become real light sources.

// Unshadowed contribution of light `li` at the shading point.
fn restir_contrib(li: u32, p: vec3<f32>, n: vec3<f32>, view: vec3<f32>,
                  alb_diff: vec3<f32>, full_alb: vec3<f32>,
                  rough: f32, metal: f32) -> vec3<f32> {
    let l = u.lights[li];
    let to_l = l.pos_range.xyz - p;
    let d = length(to_l);
    let range = l.pos_range.w;
    if (d >= range || d <= 1e-3) { return vec3<f32>(0.0); }
    let dir = to_l / d;
    let ndl = dot(n, dir);
    if (ndl <= 0.0) { return vec3<f32>(0.0); }
    let att = 1.0 - d / range;
    let li_rgb = l.color_int.rgb * l.color_int.w * att * att;
    return alb_diff * li_rgb * ndl
        + nee_spec(n, view, dir, ndl, full_alb, rough, metal) * li_rgb;
}

// Scalar target density: luminance of the unshadowed contribution.
// Correctness never depends on the target — only variance does.
fn restir_target(li: u32, p: vec3<f32>, n: vec3<f32>, view: vec3<f32>,
                 alb_diff: vec3<f32>, full_alb: vec3<f32>,
                 rough: f32, metal: f32) -> f32 {
    return dot(restir_contrib(li, p, n, view, alb_diff, full_alb, rough, metal),
        vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Runs at the primary vertex when ext.w == 1; writes this frame's
// reservoir and returns the winner's shadow-tested contribution.
fn restir_point_light(idx: u32, p: vec3<f32>, n: vec3<f32>, view: vec3<f32>,
                      alb_diff: vec3<f32>, full_alb: vec3<f32>,
                      rough: f32, metal: f32) -> vec3<f32> {
    let count = u32(u.cfg.z);
    if (count == 0u) {
        resv_out[idx] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        return vec3<f32>(0.0);
    }
    var r_y = 0u;
    var r_wsum = 0.0;
    var r_m = 0.0;
    var r_phat = 0.0;
    // RIS: 8 uniform candidates (pdf = 1/count => w = phat * count).
    for (var c = 0u; c < 8u; c = c + 1u) {
        let cand = min(u32(rand_f() * f32(count)), count - 1u);
        let ph = restir_target(cand, p, n, view, alb_diff, full_alb, rough, metal);
        let w = ph * f32(count);
        r_wsum += w;
        r_m += 1.0;
        if (w > 0.0 && rand_f() * r_wsum < w) {
            r_y = cand;
            r_phat = ph;
        }
    }
    // Temporal reuse from the reprojected texel. The stored W already
    // integrates that reservoir's history; its M is capped so stale
    // samples cannot outvote fresh ones forever.
    if (rp_valid) {
        let pr = resv[rp_nearest];
        let pm = min(pr.z, 160.0);
        if (pm > 0.0 && pr.x >= 0.0 && u32(pr.x) < count) {
            let py = u32(pr.x);
            let ph = restir_target(py, p, n, view, alb_diff, full_alb, rough, metal);
            let w = ph * pr.y * pm;
            if (w > 0.0) {
                r_wsum += w;
                if (rand_f() * r_wsum < w) {
                    r_y = py;
                    r_phat = ph;
                }
            }
            r_m += pm;
        }
    }
    var r_w = 0.0;
    if (r_phat > 0.0 && r_m > 0.0) {
        r_w = r_wsum / (r_m * r_phat);
    }
    resv_out[idx] = vec4<f32>(f32(r_y), r_w, r_m, r_phat);
    if (r_w <= 0.0) { return vec3<f32>(0.0); }
    // One shadow ray for the winner.
    let l = u.lights[r_y];
    let to_l = l.pos_range.xyz - p;
    let d = length(to_l);
    if (d <= 1e-3) { return vec3<f32>(0.0); }
    let dir = to_l / d;
    if (occluded(p, dir, d - 0.02)) { return vec3<f32>(0.0); }
    return restir_contrib(r_y, p, n, view, alb_diff, full_alb, rough, metal) * r_w;
}

// ---- Main -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.size.x || gid.y >= u.size.y) { return; }
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    // A rolling per-frame sequence in EVERY mode: SVGF's temporal
    // accumulation needs unbiased fresh samples each frame (a frozen
    // sequence converges the EMA to a wrong-but-stable value that
    // reads as static dirty-lens grain glued to the screen). The
    // variance estimate below is what keeps rolling noise from
    // shimmering: it tells the à-trous exactly where to blur hard.
    rng_seed(gid.xy, u.size.z);

    // PT-3 half-res: realtime mode traces a half grid; map this trace
    // cell to its full-res G-buffer pixel (the 2x2 phase rotates per
    // frame so the EMA integrates all four over time). Full-res modes
    // have ext == size and phase 0, making this the identity.
    var px_full = px;
    if (u.ext.x > u.size.x) {
        // Generalized ratio (the trace grid is budget-capped, so the
        // factor is not necessarily 2): integer scale keeps each trace
        // texel pinned to one owner pixel.
        px_full = min(
            vec2<i32>(
                px.x * i32(u.ext.x) / i32(u.size.x),
                px.y * i32(u.ext.y) / i32(u.size.y),
            ),
            vec2<i32>(i32(u.ext.x) - 1, i32(u.ext.y) - 1),
        );
    }

    let debug = u.cfg.w;
    if (debug == 5.0) {
        textureStore(out_hdr, px_full, vec4<f32>(1.0, 0.0, 1.0, 1.0));
        return;
    }

    let depth = depth_at(px_full);
    if (debug == 1.0) {
        textureStore(out_hdr, px_full, vec4<f32>(vec3<f32>(depth), 1.0));
        return;
    }

    if (is_sky(depth)) {
        // Leave the raster sky/clouds untouched. Realtime mode marks
        // the texel as sky in the MOMENTS buffer (depth channel = far
        // plane) so the a-trous passes and the upsampler skip it.
        if (u.cfg.x >= 2.0 && u.cfg.w == 0.0) {
            let sky_idx = gid.y * u.size.x + gid.x;
            accum_out[sky_idx] = vec4<f32>(0.0);
            moments_out[sky_idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        return;
    }

    let p0 = world_at(px_full, depth);
    let n0 = normal_from_depth(px_full, p0);
    let albedo0 = textureLoad(albedo_tex, px_full, 0).rgb;

    if (debug == 2.0) {
        textureStore(out_hdr, px_full, vec4<f32>(n0 * 0.5 + 0.5, 1.0));
        return;
    }
    if (debug == 3.0) {
        textureStore(out_hdr, px_full, vec4<f32>(albedo0, 1.0));
        return;
    }
    if (debug == 4.0) {
        let sd = sun_cone_sample(rand_2f());
        let vis = select(0.0, 1.0, !occluded(p0 + n0 * 0.02, sd, 1000.0));
        textureStore(out_hdr, px_full, vec4<f32>(vec3<f32>(vis), 1.0));
        return;
    }
    if (debug == 8.0) {
        // Binary probe: white = traced hit has a geometry window,
        // black = geo.z reads 0, red = TLAS miss. HDR-large values so
        // exposure/tonemap can't blur the verdict.
        let dir0 = normalize(p0 - u.cam_pos.xyz);
        var rq8: ray_query;
        rayQueryInitialize(&rq8, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq8)) { break; }
        }
        let h8 = rayQueryGetCommittedIntersection(&rq8);
        var c8 = vec3<f32>(100.0, 0.0, 0.0);
        if (h8.kind != RAY_QUERY_INTERSECTION_NONE) {
            let gi = instance_data[h8.instance_custom_data].geo;
            c8 = select(vec3<f32>(0.0), vec3<f32>(100.0), gi.z > 0u);
        }
        textureStore(out_hdr, px_full, vec4<f32>(c8, 1.0));
        return;
    }
    if (debug == 9.0) {
        // Quantized normal probe: dominant axis of the interpolated
        // world normal as six saturated HDR colours. +X red, -X dark
        // red-ish magenta, +Y green, -Y cyan, +Z blue, -Z yellow.
        // Gray = TLAS miss / no window / zero-length normal.
        let dir0 = normalize(p0 - u.cam_pos.xyz);
        var rq9: ray_query;
        rayQueryInitialize(&rq9, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq9)) { break; }
        }
        let h9 = rayQueryGetCommittedIntersection(&rq9);
        var c9 = vec3<f32>(5.0, 5.0, 5.0);
        if (h9.kind != RAY_QUERY_INTERSECTION_NONE) {
            let inst9 = instance_data[h9.instance_custom_data];
            if (inst9.geo.z > 0u) {
                let a9 = fetch_hit_attrs(inst9.geo, h9.primitive_index, h9.barycentrics);
                let raw = a9.normal_os;
                if (length(raw) > 1e-6) {
                    let n9 = normal_to_world(raw, h9.world_to_object);
                    let an = abs(n9);
                    if (an.y >= an.x && an.y >= an.z) {
                        c9 = select(vec3<f32>(0.0, 50.0, 50.0), vec3<f32>(0.0, 50.0, 0.0), n9.y >= 0.0);
                    } else if (an.x >= an.z) {
                        c9 = select(vec3<f32>(50.0, 0.0, 25.0), vec3<f32>(50.0, 0.0, 0.0), n9.x >= 0.0);
                    } else {
                        c9 = select(vec3<f32>(50.0, 50.0, 0.0), vec3<f32>(0.0, 0.0, 50.0), n9.z >= 0.0);
                    }
                }
            }
        }
        textureStore(out_hdr, px_full, vec4<f32>(c9, 1.0));
        return;
    }
    if (debug == 10.0) {
        // primitive_index sanity probe: banded pseudo-colour of the hit
        // triangle index. Expected: per-triangle colour noise across
        // meshes. A single flat colour everywhere = the field is
        // constant; saturated white = garbage-huge.
        let dir0 = normalize(p0 - u.cam_pos.xyz);
        var rq10: ray_query;
        rayQueryInitialize(&rq10, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq10)) { break; }
        }
        let h10 = rayQueryGetCommittedIntersection(&rq10);
        var c10 = vec3<f32>(0.0);
        if (h10.kind != RAY_QUERY_INTERSECTION_NONE) {
            let prim = f32(h10.primitive_index);
            c10 = vec3<f32>(fract(prim / 64.0), fract(prim / 1024.0), fract(prim / 16384.0)) * 30.0;
        }
        textureStore(out_hdr, px_full, vec4<f32>(c10, 1.0));
        return;
    }
    if (debug == 11.0 || debug == 12.0) {
        // 11: instance_custom_data palette (expect distinct colours per
        //     proxy: terrain vs trees vs building). Constant = broken.
        // 12: raw barycentrics (expect smooth per-triangle gradients).
        let dir0 = normalize(p0 - u.cam_pos.xyz);
        var rq11: ray_query;
        rayQueryInitialize(&rq11, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq11)) { break; }
        }
        let h11 = rayQueryGetCommittedIntersection(&rq11);
        var c11 = vec3<f32>(0.0);
        if (h11.kind != RAY_QUERY_INTERSECTION_NONE) {
            if (debug == 11.0) {
                let id = h11.instance_custom_data;
                c11 = vec3<f32>(
                    f32((id * 37u) % 7u) / 7.0,
                    f32((id * 61u) % 11u) / 11.0,
                    f32((id * 13u) % 5u) / 5.0,
                ) * 30.0;
            } else {
                let b = h11.barycentrics;
                c11 = vec3<f32>(b.x, b.y, max(0.0, 1.0 - b.x - b.y)) * 30.0;
            }
        }
        textureStore(out_hdr, px_full, vec4<f32>(c11, 1.0));
        return;
    }
    if (debug == 13.0) {
        // TLAS sanity: green = traced primary hit distance agrees with
        // the G-buffer depth (within 2% + 0.1m), red = disagreement
        // (wrong geometry committed), blue = TLAS miss on a G-buffer
        // pixel. If this is red/blue everywhere the TLAS itself (not
        // the intersection attributes) is broken on this backend.
        let to_p = p0 - u.cam_pos.xyz;
        let gdist = length(to_p);
        let dir0 = to_p / max(gdist, 1e-4);
        var rq13: ray_query;
        rayQueryInitialize(&rq13, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq13)) { break; }
        }
        let h13 = rayQueryGetCommittedIntersection(&rq13);
        var c13 = vec3<f32>(0.0, 0.0, 50.0);
        if (h13.kind != RAY_QUERY_INTERSECTION_NONE) {
            let err = abs(h13.t - gdist);
            if (err < gdist * 0.02 + 0.1) {
                c13 = vec3<f32>(0.0, 50.0, 0.0);
            } else {
                c13 = vec3<f32>(50.0, 0.0, 0.0);
            }
        }
        textureStore(out_hdr, px_full, vec4<f32>(c13, 1.0));
        return;
    }
    if (debug == 14.0) {
        // Shape probe: contour bands of the traced primary hit distance
        // — shows what world the TLAS actually contains. Blue = miss.
        let dir0 = normalize(p0 - u.cam_pos.xyz);
        var rq14: ray_query;
        rayQueryInitialize(&rq14, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq14)) { break; }
        }
        let h14 = rayQueryGetCommittedIntersection(&rq14);
        var c14 = vec3<f32>(0.0, 0.0, 30.0);
        if (h14.kind != RAY_QUERY_INTERSECTION_NONE) {
            c14 = vec3<f32>(
                fract(h14.t * 0.125),
                fract(h14.t * 0.03125),
                fract(h14.t * 0.0078125),
            ) * 20.0;
        }
        textureStore(out_hdr, px_full, vec4<f32>(c14, 1.0));
        return;
    }
    if (debug == 15.0) {
        // Aliasing probe: two queries, two very different rays.
        // A = primary (per-pixel), B = straight down (t ~= camera
        // height, near-constant). R channel = banded tA, G = banded tB.
        // If R == G everywhere the two queries alias to one object.
        let dirA = normalize(p0 - u.cam_pos.xyz);
        var rqA: ray_query;
        rayQueryInitialize(&rqA, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dirA));
        loop {
            if (!rayQueryProceed(&rqA)) { break; }
        }
        var rqB: ray_query;
        rayQueryInitialize(&rqB, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, vec3<f32>(0.0, -1.0, 0.0)));
        loop {
            if (!rayQueryProceed(&rqB)) { break; }
        }
        let hA = rayQueryGetCommittedIntersection(&rqA);
        let hB = rayQueryGetCommittedIntersection(&rqB);
        var tA = -1.0;
        var tB = -1.0;
        if (hA.kind != RAY_QUERY_INTERSECTION_NONE) { tA = hA.t; }
        if (hB.kind != RAY_QUERY_INTERSECTION_NONE) { tB = hB.t; }
        let c15 = vec3<f32>(fract(tA * 0.125) * 20.0, fract(tB * 0.125) * 20.0, 0.0);
        textureStore(out_hdr, px_full, vec4<f32>(c15, 1.0));
        return;
    }
    if (debug == 16.0) {
        // Raw numeric dump: traced primary intersection into the accum
        // buffer as (t, instance_custom_data, primitive_index, kind).
        // The CPU side reads a window of this buffer back and writes a
        // text file — no tonemap guesswork.
        let dir0 = normalize(p0 - u.cam_pos.xyz);
        var rq16: ray_query;
        rayQueryInitialize(&rq16, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq16)) { break; }
        }
        let h16 = rayQueryGetCommittedIntersection(&rq16);
        let idx16 = gid.y * u.size.x + gid.x;
        accum_out[idx16] = vec4<f32>(
            h16.t,
            f32(h16.instance_custom_data),
            f32(h16.primitive_index),
            f32(h16.kind),
        );
        textureStore(out_hdr, px_full, vec4<f32>(0.2, 0.0, 0.4, 1.0));
        return;
    }
    if (debug == 17.0) {
        // Raw ray-generation dump: reconstructed world position + raw
        // depth, straight into accum for CPU readback.
        let idx17 = gid.y * u.size.x + gid.x;
        accum_out[idx17] = vec4<f32>(p0, depth);
        textureStore(out_hdr, px_full, vec4<f32>(0.4, 0.2, 0.0, 1.0));
        return;
    }
    if (debug == 18.0) {
        // Hybrid-sun validation: cascade shadow visibility at the
        // primary surface. Must match the raster shadow shapes exactly
        // (crisp tree/building shadows). Gray everywhere = the VPs are
        // wrong (transposition or stale).
        let vis18 = sun_vis_cascade(p0 + n0 * 0.02);
        textureStore(out_hdr, px_full, vec4<f32>(vec3<f32>(vis18 * 50.0), 1.0));
        return;
    }
    if (debug == 19.0) {
        // Numeric dump: cascade-0 shadow projection (ndc.xyz) + the
        // stored atlas depth at the landing texel (-1 = out of range,
        // -9 = degenerate w). Read back via pt_trace_dump.txt.
        let pw19 = p0 + n0 * 0.02;
        let clip19 = u.shadow_vps[0] * vec4<f32>(pw19, 1.0);
        var out19 = vec4<f32>(-9.0);
        if (abs(clip19.w) > 1e-6) {
            let ndc19 = clip19.xyz / clip19.w;
            let uv19 = vec2<f32>(ndc19.x * 0.5 + 0.5, 0.5 - ndc19.y * 0.5);
            var stored19 = -1.0;
            if (uv19.x >= 0.0 && uv19.x <= 1.0 && uv19.y >= 0.0 && uv19.y <= 1.0) {
                let dims19 = vec2<f32>(textureDimensions(shadow_atlas_0));
                stored19 = textureLoad(shadow_atlas_0, vec2<i32>(uv19 * dims19), 0);
            }
            out19 = vec4<f32>(ndc19.xyz, stored19);
        }
        accum_out[gid.y * u.size.x + gid.x] = out19;
        textureStore(out_hdr, px_full, vec4<f32>(0.1, 0.0, 0.2, 1.0));
        return;
    }
    if (debug == 6.0 || debug == 7.0) {
        // PT-2 validation: trace the primary ray through the TLAS
        // (ignoring the G-buffer) and show interpolated attributes.
        // Should match debug 2/3 up to smooth-vs-screen normals and
        // card-vs-texture resolution.
        let dir0 = normalize(p0 - u.cam_pos.xyz);
        var rq0: ray_query;
        rayQueryInitialize(&rq0, accel, RayDesc(0u, 0xFFu, 0.001, 1000.0, u.cam_pos.xyz, dir0));
        loop {
            if (!rayQueryProceed(&rq0)) { break; }
        }
        let h = rayQueryGetCommittedIntersection(&rq0);
        var col = vec3<f32>(1.0, 0.0, 1.0);        // magenta: TLAS miss
        if (h.kind != RAY_QUERY_INTERSECTION_NONE) {
            let hinst = instance_data[h.instance_custom_data];
            if (hinst.geo.z > 0u) {
                let attrs = fetch_hit_attrs(hinst.geo, h.primitive_index, h.barycentrics);
                if (debug == 6.0) {
                    col = normal_to_world(attrs.normal_os, h.world_to_object) * 0.5 + vec3<f32>(0.5);
                } else if (PT_HAS_TEXTURES) {
                    col = hinst.albedo * pt_tex_sample(hinst.geo.w, attrs.uv);
                } else {
                    col = vec3<f32>(1.0, 1.0, 0.0);  // yellow: no tex arrays
                }
            } else {
                col = vec3<f32>(1.0, 0.5, 0.0);      // orange: no geo window
            }
        }
        textureStore(out_hdr, px_full, vec4<f32>(col, 1.0));
        return;
    }

    // ---- one path sample --------------------------------------------------

    // Primary surface material from the G-buffer (R = metallic,
    // G = roughness). NEE stays diffuse-only, so scale it by
    // (1 - metallic) — metals have no diffuse lobe. Specular NEE is a
    // known gap (see the PT-2 ticket); specular reflection of sky and
    // scene comes from the GGX bounce below.
    let mr0 = textureLoad(material_tex, px_full, 0).rg;
    var metal_cur = mr0.r;
    var rough_cur = mr0.g;
    // Realtime mode samples the primary sun cone with structured IGN
    // noise, rolling per frame: spatially well-distributed (a 5x5
    // filter averages it nearly flat, unlike white PCG noise) and
    // temporally unbiased so the SVGF accumulation converges to the
    // true mean. Under the hybrid cascade sun this path only matters
    // when shadow maps are disabled. Progressive keeps white noise.
    var sun_r2 = rand_2f();
    if (u.cfg.x >= 2.0) {
        sun_r2 = vec2<f32>(
            ign_at(px_full, u.size.z),
            ign_at(px_full + vec2<i32>(17, 59), u.size.z),
        );
    }
    var view_cur = normalize(u.cam_pos.xyz - p0);
    // Reproject this surface into the previous trace grid ONCE — the
    // ReSTIR temporal reuse and the SVGF colour accumulation below both
    // consume the result (rp_* privates).
    compute_reproj(p0, px_full, depth);
    // PT-4 experimental: ext.w routes the primary point-light NEE
    // through the ReSTIR reservoirs; sun NEE is untouched either way.
    let use_restir = u.ext.w == 1u && u.cfg.x >= 2.0;
    // Sun + point lights, diffuse AND specular (nee_spec inside) — the
    // GGX highlight rides the same visibility as the diffuse term.
    var radiance = direct_light(
        p0 + n0 * 0.02, n0, albedo0 * (1.0 - metal_cur), sun_r2,
        view_cur, albedo0, rough_cur, metal_cur, !use_restir,
    );
    if (use_restir) {
        radiance += restir_point_light(
            gid.y * u.size.x + gid.x, p0 + n0 * 0.02, n0, view_cur,
            albedo0 * (1.0 - metal_cur), albedo0, rough_cur, metal_cur,
        );
    }
    var throughput = vec3<f32>(1.0);
    var origin = p0 + n0 * 0.02;
    var n_cur = n0;
    var alb_cur = albedo0;

    let max_bounces = u32(u.cfg.y);
    for (var b = 0u; b < max_bounces; b = b + 1u) {
        let s = sample_brdf(n_cur, view_cur, alb_cur, rough_cur, metal_cur);
        if (!s.valid) {
            break;
        }
        throughput *= s.weight;
        let dir = s.dir;

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

        // PT-2: interpolated vertex normal + textured albedo when the
        // instance carries a geometry window; PT-1 flat-normal/card
        // fallback otherwise.
        var n_hit: vec3<f32>;
        var alb_hit: vec3<f32>;
        if (inst.geo.z > 0u) {
            let attrs = fetch_hit_attrs(inst.geo, hit.primitive_index, hit.barycentrics);
            n_hit = normal_to_world(attrs.normal_os, hit.world_to_object);
            if (PT_HAS_TEXTURES) {
                alb_hit = inst.albedo * pt_tex_sample(inst.geo.w, attrs.uv);
            } else {
                alb_hit = albedo_at_hit(inst, hit_os, dir);
            }
        } else {
            var nf = inst.normal_ws;
            let n_len = length(nf);
            if (n_len < 1e-4) { nf = -dir; } else { nf = nf / n_len; }
            n_hit = nf;
            alb_hit = albedo_at_hit(inst, hit_os, dir);
        }
        // A backface (or a flat normal pointing away) still bounces
        // outward, matching the OPAQUE two-sided raster convention.
        if (dot(n_hit, dir) > 0.0) { n_hit = -n_hit; }

        // Emissive surfaces radiate; matches the Lumen fallback semantics
        // (albedo * emissive_luma).
        radiance += throughput * inst.albedo * inst.emissive_luma;

        let hit_p = hit_ws + n_hit * 0.02;
        // view at a bounce vertex = back along the incoming ray.
        // Bounce vertices always use plain NEE (reservoirs are per
        // PRIMARY texel; reusing them off-surface would be biased).
        radiance += throughput * direct_light(
            hit_p, n_hit, alb_hit * (1.0 - inst.mat_params.y), rand_2f(),
            -dir, alb_hit, inst.mat_params.x, inst.mat_params.y, true,
        );

        origin = hit_p;
        n_cur = n_hit;
        alb_cur = alb_hit;
        rough_cur = inst.mat_params.x;
        metal_cur = inst.mat_params.y;
        view_cur = -dir;

        // Russian roulette from the third bounce.
        if (b >= 2u) {
            let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (rand_f() > q) { break; }
            throughput /= q;
        }
    }

    // NaN/Inf guard so one bad sample cannot poison the accumulator.
    if (radiance.r != radiance.r || radiance.g != radiance.g || radiance.b != radiance.b) {
        radiance = vec3<f32>(0.0);
    }

    // ---- accumulate ---------------------------------------------------------

    let idx = gid.y * u.size.x + gid.x;
    let mode = u.cfg.x;
    var prev = accum[idx];
    if (u.size.w == 0u) { prev = vec4<f32>(0.0); }

    var out: vec3<f32>;
    if (mode >= 2.0) {
        // SVGF temporal accumulation (Schied et al. 2017). History and
        // output store IRRADIANCE (radiance demodulated by the primary
        // albedo) so the wavelet passes filter lighting only; the
        // final pass re-multiplies by the full-res G-buffer albedo.
        var irr = radiance / max(albedo0, vec3<f32>(0.05));
        // Firefly clamp — the one practical deviation from the paper
        // (reference implementations keep one too): it must bind in
        // IRRADIANCE space, because dividing by a dark albedo
        // amplifies radiance outliers up to 20x. Sunlit irradiance
        // sits around 1-3; 4 leaves real highlights alone.
        let irr_luma = dot(irr, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (irr_luma > 4.0) {
            irr *= 4.0 / irr_luma;
        }
        let l_new = min(irr_luma, 4.0);

        // Reprojection: 2x2 BILINEAR taps around the reprojected
        // position, each tap validated for geometric consistency
        // (relative linearized depth against the moments buffer).
        // Point sampling here quantizes to whole trace texels and
        // forced the old loose-tolerance workaround; weighted taps
        // give sub-texel reprojection and a honest per-tap test.
        var hist_rgb = vec3<f32>(0.0);
        var hist_m1 = 0.0;
        var hist_m2 = 0.0;
        var hist_n = 0.0;
        var wsum = 0.0;
        // Footprint depth window (current frame): one trace texel
        // covers ~3x3 full-res pixels; used below to tell a jitter
        // surface-flip apart from a true disocclusion.
        var fp_lo = 1e30;
        var fp_hi = 0.0;
        {
            let rx = max(i32(u.ext.x) / i32(u.size.x), 1);
            let ry = max(i32(u.ext.y) / i32(u.size.y), 1);
            for (var sy = 0; sy <= 1; sy = sy + 1) {
                for (var sx = 0; sx <= 1; sx = sx + 1) {
                    let sp = min(
                        px_full + vec2<i32>(sx * (rx - 1), sy * (ry - 1)),
                        vec2<i32>(i32(u.ext.x) - 1, i32(u.ext.y) - 1),
                    );
                    let dz = depth_at(sp);
                    if (dz >= 0.9999999) { continue; }
                    let zl = lin_depth(dz);
                    fp_lo = min(fp_lo, zl);
                    fp_hi = max(fp_hi, zl);
                }
            }
        }
        // Reprojection basis was computed once at the top of the frame
        // (rp_* privates, shared with the ReSTIR temporal reuse).
        if (rp_valid && debug == 23.0) {
            // Reprojection dump: where this texel thinks it was
            // last frame (trace-grid units) + the depth pair the
            // acceptance test compares. Static camera => pos
            // must equal the texel's own coordinates.
            accum_out[idx] = vec4<f32>(
                f32(rp_base.x) + rp_fr.x, f32(rp_base.y) + rp_fr.y,
                rp_zl_here, lin_depth(moments[rp_nearest].w));
            moments_out[idx] = vec4<f32>(0.0, 0.0, 0.0, depth);
            return;
        }
        if (rp_valid) {
            // Tap test is TIGHT (surface identity). Cross-surface
            // blending is the expensive error: it leaks bright
            // blade-top lighting onto the ground below (gray-blue
            // mottle). Surface flips are handled after the loop,
            // not by widening this tolerance.
            let tol = 0.1 * rp_zl_here + 0.02;
            for (var ty = 0; ty <= 1; ty = ty + 1) {
                for (var tx = 0; tx <= 1; tx = tx + 1) {
                    let q = rp_base + vec2<i32>(tx, ty);
                    if (q.x < 0 || q.y < 0 || q.x >= i32(u.size.x) || q.y >= i32(u.size.y)) {
                        continue;
                    }
                    let qidx = u32(q.y) * u.size.x + u32(q.x);
                    let m = moments[qidx];
                    // Sky texels and depth-inconsistent taps carry
                    // another surface's lighting — skip them.
                    if (m.w >= 0.9999999) {
                        continue;
                    }
                    let zl_hist = lin_depth(m.w);
                    if (abs(zl_hist - rp_zl_here) > tol) {
                        continue;
                    }
                    let wx = mix(1.0 - rp_fr.x, rp_fr.x, f32(tx));
                    let wy = mix(1.0 - rp_fr.y, rp_fr.y, f32(ty));
                    let wt = wx * wy + 1e-4;
                    hist_rgb += accum[qidx].rgb * wt;
                    hist_m1 += m.x * wt;
                    hist_m2 += m.y * wt;
                    hist_n += m.z * wt;
                    wsum += wt;
                }
            }
        }

        // No tap matched. One trace texel holds ONE surface's history;
        // TAA jitter re-picks which surface the owner pixel sees each
        // frame on sub-texel geometry (grass). If the STORED surface
        // still exists in this texel's current footprint, this frame's
        // sample simply belongs to the other surface: keep the history
        // verbatim and drop the sample (blending would leak lighting
        // across surfaces; resetting would pin the texel at 1 spp and
        // fill the screen with speckle). The upsampler routes trace
        // texels to full-res pixels by depth, so the other surface
        // draws its lighting from neighbouring texels. Only a stored
        // surface that has LEFT the footprint is a true disocclusion.
        if (wsum <= 1e-3 && rp_valid) {
            let mnp = moments[rp_nearest];
            if (mnp.w < 0.9999999 && mnp.z > 0.0 && fp_hi > 0.0 && fp_lo < 1e29) {
                let zl_st = lin_depth(mnp.w);
                let wtol = 0.1 * zl_st + 0.02;
                if (zl_st > fp_lo - wtol && zl_st < fp_hi + wtol) {
                    accum_out[idx] = accum[rp_nearest];
                    moments_out[idx] = mnp;
                    if (debug == 22.0) {
                        // Numeric dump: n / variance / 99 = flip path / depth.
                        accum_out[idx] = vec4<f32>(mnp.z, max(mnp.y - mnp.x * mnp.x, 0.0), 99.0, mnp.w);
                    }
                    if (debug == 20.0) {
                        textureStore(out_hdr, px_full, vec4<f32>(vec3<f32>(mnp.z / 32.0), 1.0));
                    }
                    if (debug == 21.0) {
                        let v_st = max(mnp.y - mnp.x * mnp.x, 0.0);
                        textureStore(out_hdr, px_full, vec4<f32>(vec3<f32>(v_st * 10.0), 1.0));
                    }
                    return;
                }
            }
        }

        var n_hist = 0.0;
        if (wsum > 1e-3) {
            hist_rgb /= wsum;
            hist_m1 /= wsum;
            hist_m2 /= wsum;
            n_hist = hist_n / wsum;
        }
        // Canonical blend: cumulative average while the history is
        // young (alpha = 1/N), settling to EMA. The floor is 0.1
        // rather than the paper's 0.2: our trace is half-res with a
        // 2-bounce sky lottery as the dominant noise source, and the
        // deeper average halves the residual mottle. Direct sun comes
        // from the raster cascades (deterministic), so the slower EMA
        // only delays indirect/ambient changes (~10 frames).
        let n_new = min(n_hist + 1.0, 32.0);
        let alpha_c = max(1.0 / n_new, 0.1);
        let out_irr = mix(hist_rgb, irr, alpha_c);
        let m1 = mix(hist_m1, l_new, alpha_c);
        let m2 = mix(hist_m2, l_new * l_new, alpha_c);
        // Temporal luminance variance — the signal that drives the
        // wavelet filter's luminance sigma. Young history makes this
        // unreliable; the first à-trous iteration substitutes a
        // spatial estimate when n < 4 (accum.w carries n via moments).
        let variance = max(m2 - m1 * m1, 0.0);
        accum_out[idx] = vec4<f32>(out_irr, variance);
        moments_out[idx] = vec4<f32>(m1, m2, n_new, depth);
        if (debug == 22.0) {
            // Numeric dump: n / variance / accepted tap mass / depth.
            accum_out[idx] = vec4<f32>(n_new, variance, wsum, depth);
        }
        if (debug == 20.0) {
            // History length heat: white = full 32-frame history.
            textureStore(out_hdr, px_full, vec4<f32>(vec3<f32>(n_new / 32.0), 1.0));
        }
        if (debug == 21.0) {
            // Variance view (x10 so typical values are visible).
            textureStore(out_hdr, px_full, vec4<f32>(vec3<f32>(variance * 10.0), 1.0));
        }
        return;
    } else {
        // Progressive keeps its firefly cap, relaxing with
        // accumulation depth (a deep average can absorb real energy).
        let luma = dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
        let cap = 4.0 + f32(min(u.size.w, 28u));
        if (luma > cap) { radiance *= cap / luma; }
        // Progressive: plain running sum; count lives on the CPU.
        // Ping-pong read/write at the same index (static camera only).
        let sum = prev.rgb + radiance;
        let n = f32(u.size.w) + 1.0;
        accum_out[idx] = vec4<f32>(sum, n);
        out = sum / n;
        // Interim gameplay behaviour until PT-3's denoiser: a moving
        // camera resets accumulation every frame, and raw 1-spp noise
        // through TSR looks terrible. Keep accumulating but leave the
        // raster frame on screen until a few samples exist — stand
        // still for half a second and PT dissolves in. The CPU side
        // mirrors this threshold (pt_wrote_frame) so SSGI/SSR stay on
        // for the raster frames.
        if (u.size.w < 8u) {
            return;
        }
    }

    textureStore(out_hdr, px_full, vec4<f32>(out, 1.0));
}
"#;

/// PT-3b — SVGF wavelet filter (Schied et al. 2017) for the realtime
/// mode. Four `cs_mid` à-trous iterations (steps 1/2/4/8) run on the
/// trace grid over the temporally-accumulated irradiance; `cs_final`
/// joint-bilaterally upsamples to full resolution and re-modulates the
/// G-buffer albedo. Buffers: src/dst = (irradiance rgb, luminance
/// variance w); `geo` = the kernel's moments buffer (mu1, mu2, history
/// length, raw depth) — static across iterations, it carries the depth
/// for edge-stopping and the sky marker (depth = far plane).
///
/// The luminance edge-stop is VARIANCE-DRIVEN: sigma_l scales with the
/// per-texel noise estimate, so grainy regions blur hard while
/// converged shading detail survives. Variance travels with the signal,
/// filtered by the squared weights, shrinking each iteration exactly as
/// the residual noise does. This replaces the old fixed sigma schedule,
/// the despeckle clamp and the history spike clamp — with a correct
/// variance estimate none of those are needed.
pub(in crate::renderer) const PT_ATROUS_WGSL: &str = r#"
struct AtrousParams {
    // x = step (texels), y = 1.0 on the FIRST iteration (enables the
    // short-history spatial variance fallback), z/w = trace dims
    p: vec4<f32>,
    // x/y = full G-buffer dims (cs_final upsamples trace -> full when
    // they differ), z/w unused.
    p2: vec4<f32>,
};
@group(0) @binding(0) var<uniform> ap: AtrousParams;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var out_hdr_a: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var depth_full: texture_depth_2d;
// Full-res G-buffer albedo: cs_final re-modulates the filtered
// irradiance with it (SVGF demodulation keeps textures crisp).
@group(0) @binding(5) var albedo_full: texture_2d<f32>;
// Kernel moments buffer: (mu1, mu2, history length, raw depth).
@group(0) @binding(6) var<storage, read> geo: array<vec4<f32>>;

fn lin_depth_a(d: f32) -> f32 {
    return 0.02 / max(1.0 - d, 1e-6);
}

fn luma_of(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// B3-spline kernel weight for |offset| 0/1/2.
fn kern(d: i32) -> f32 {
    let a = abs(d);
    if (a == 0) { return 0.375; }
    if (a == 1) { return 0.25; }
    return 0.0625;
}

// SVGF luminance sigma (paper value).
const SIGMA_L: f32 = 4.0;

fn filter_at(px: vec2<i32>, w: i32, h: i32, step: i32, first: bool) -> vec4<f32> {
    let cidx = u32(px.y) * u32(w) + u32(px.x);
    let g_c = geo[cidx];
    let center = src[cidx];
    if (g_c.w >= 0.9999999) {
        return center;
    }
    let zc = lin_depth_a(g_c.w);
    let lc = luma_of(center.rgb);

    // Center variance. Temporal variance from a young history (< 4
    // frames, e.g. right after a disocclusion) is meaningless — the
    // paper substitutes a spatial luminance-variance estimate there.
    var var_c = max(center.w, 0.0);
    if (first && g_c.z < 4.0) {
        var s1 = 0.0;
        var s2 = 0.0;
        var cnt = 0.0;
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let q = px + vec2<i32>(dx, dy);
                if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
                    continue;
                }
                let qi = u32(q.y) * u32(w) + u32(q.x);
                if (geo[qi].w >= 0.9999999) {
                    continue;
                }
                let lq = luma_of(src[qi].rgb);
                s1 += lq;
                s2 += lq * lq;
                cnt += 1.0;
            }
        }
        if (cnt > 1.0) {
            let mu = s1 / cnt;
            var_c = max(var_c, max(s2 / cnt - mu * mu, 0.0));
        }
        // Variance boost: a young history's estimate is unreliable in
        // BOTH directions, and underestimating is the expensive error
        // (a 1-spp outlier with variance 0 survives every iteration as
        // a false edge). Floor it so fresh texels blend spatially until
        // their temporal estimate matures.
        var_c = max(var_c, 0.25);
    }

    // 3x3 gaussian prefilter of the variance (paper 4.2): keeps a
    // single hot texel from stopping its own smoothing.
    var vsum = var_c * 0.25;
    var vwsum = 0.25;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let q = px + vec2<i32>(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
                continue;
            }
            let qi = u32(q.y) * u32(w) + u32(q.x);
            if (geo[qi].w >= 0.9999999) {
                continue;
            }
            let gw = select(0.0625, 0.125, dx == 0 || dy == 0);
            vsum += max(src[qi].w, 0.0) * gw;
            vwsum += gw;
        }
    }
    let sigma_l_denom = SIGMA_L * sqrt(max(vsum / vwsum, 0.0)) + 1e-3;

    // Depth gradient (central differences on linear depth) scales the
    // depth edge-stop so steep-slope surfaces stay connected while
    // depth discontinuities still stop the filter (paper eq. 3).
    var dzdx = 0.0;
    var dzdy = 0.0;
    if (px.x > 0 && px.x < w - 1) {
        dzdx = (lin_depth_a(geo[cidx + 1u].w) - lin_depth_a(geo[cidx - 1u].w)) * 0.5;
    }
    if (px.y > 0 && px.y < h - 1) {
        dzdy = (lin_depth_a(geo[cidx + u32(w)].w) - lin_depth_a(geo[cidx - u32(w)].w)) * 0.5;
    }

    var sum = vec3<f32>(0.0);
    var sum_v = 0.0;
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let q = px + vec2<i32>(dx, dy) * step;
            if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
                continue;
            }
            let qi = u32(q.y) * u32(w) + u32(q.x);
            let g_q = geo[qi];
            if (g_q.w >= 0.9999999) {
                continue;
            }
            let s = src[qi];
            let zq = lin_depth_a(g_q.w);
            let z_denom = abs(dzdx) * f32(abs(dx) * step)
                + abs(dzdy) * f32(abs(dy) * step)
                + 0.01 * zc + 1e-4;
            let wz = exp(-abs(zq - zc) / z_denom);
            let wl = exp(-abs(luma_of(s.rgb) - lc) / sigma_l_denom);
            let wgt = kern(dx) * kern(dy) * wz * wl;
            sum += s.rgb * wgt;
            // Variance contracts with the SQUARED weights — the
            // estimate shrinks exactly as the filtered noise does.
            sum_v += max(s.w, 0.0) * wgt * wgt;
            wsum += wgt;
        }
    }
    if (wsum < 1e-6) {
        return center;
    }
    return vec4<f32>(sum / wsum, sum_v / (wsum * wsum));
}

@compute @workgroup_size(8, 8, 1)
fn cs_mid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(ap.p.z);
    let h = i32(ap.p.w);
    if (i32(gid.x) >= w || i32(gid.y) >= h) {
        return;
    }
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    dst[gid.y * u32(w) + gid.x] = filter_at(px, w, h, i32(ap.p.x), ap.p.y > 0.5);
}

// Final pass runs at FULL resolution: depth-guided joint-bilateral
// upsample of the filtered irradiance, re-modulated by the full-res
// G-buffer albedo. Sky pixels (full-res depth at far plane) are never
// written so the raster sky survives. When the trace grid IS the full
// grid it degenerates to a plain modulate-and-write.
@compute @workgroup_size(8, 8, 1)
fn cs_final(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fw = i32(ap.p2.x);
    let fh = i32(ap.p2.y);
    if (i32(gid.x) >= fw || i32(gid.y) >= fh) {
        return;
    }
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = textureLoad(depth_full, px, 0);
    if (d >= 0.9999999) {
        return;
    }
    let hw = i32(ap.p.z);
    let hh = i32(ap.p.w);
    let alb = max(textureLoad(albedo_full, px, 0).rgb, vec3<f32>(0.05));
    if (hw == fw) {
        let cidx = gid.y * u32(fw) + gid.x;
        if (geo[cidx].w >= 0.9999999) {
            return;
        }
        textureStore(out_hdr_a, px, vec4<f32>(src[cidx].rgb * alb, 1.0));
        return;
    }
    // 3x3 taps around the containing trace texel, weighted by kernel
    // distance and relative linear-depth agreement with THIS full-res
    // pixel. The epsilon keeps thin foreground geometry (whose taps
    // all mismatch) softly averaged rather than black.
    let zc = lin_depth_a(d);
    // Generalized ratio mapping (trace grid is budget-capped, not
    // always exactly half of full res).
    let hx = clamp(px.x * hw / fw, 0, hw - 1);
    let hy = clamp(px.y * hh / fh, 0, hh - 1);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let qx = hx + dx;
            let qy = hy + dy;
            if (qx < 0 || qy < 0 || qx >= hw || qy >= hh) {
                continue;
            }
            let qi = u32(qy) * u32(hw) + u32(qx);
            if (geo[qi].w >= 0.9999999) {
                continue;
            }
            let s = src[qi];
            let wz = exp(-abs(lin_depth_a(geo[qi].w) - zc) / (0.08 * zc + 0.02));
            let wgt = kern(dx) * kern(dy) * wz + 1e-5;
            sum += s.rgb * wgt;
            wsum += wgt;
        }
    }
    if (wsum < 1e-6) {
        return;
    }
    textureStore(out_hdr_a, px, vec4<f32>((sum / wsum) * alb, 1.0));
}
"#;

/// PT-6 — compute pre-skin: poses one skinned mesh into its PT geometry
/// megabuffer window (world space; the joint palette bakes placement).
/// The CPU wrote the bind-pose vertex data into the window this frame;
/// this pass overwrites position + normal, leaving color/uv untouched.
/// The BLAS for the dynamic instance then reads the same window
/// (first_vertex offset), so intersection and hit shading share bytes.
pub(in crate::renderer) const PT_SKIN_WGSL: &str = r#"
struct SkinParams {
    // Places the rare rigid (weightless) verts, same as the raster VS.
    model: mat4x4<f32>,
    // x = megabuffer vertex slot base (Vertex3D units), y = vertex
    // count, z = joint palette base offset, w unused.
    p: vec4<u32>,
};
struct SkinJoints {
    m: array<mat4x4<f32>, 1024>,
};
@group(0) @binding(0) var<uniform> sp: SkinParams;
@group(0) @binding(1) var<storage, read> src_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_v: array<f32>;
@group(0) @binding(3) var<uniform> joints: SkinJoints;

@compute @workgroup_size(64, 1, 1)
fn cs_skin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sp.p.y) { return; }
    // Vertex3D words: pos +0, normal +3, color +6, uv +10, joints +12,
    // weights +16, tangent +20 (stride 24 f32).
    let s = i * 24u;
    let d = (sp.p.x + i) * 24u;
    let pos = vec4<f32>(src_v[s], src_v[s + 1u], src_v[s + 2u], 1.0);
    let nrm = vec4<f32>(src_v[s + 3u], src_v[s + 4u], src_v[s + 5u], 0.0);
    let w = vec4<f32>(
        src_v[s + 16u], src_v[s + 17u], src_v[s + 18u], src_v[s + 19u],
    );
    var wp: vec3<f32>;
    var wn: vec3<f32>;
    if (w.x + w.y + w.z + w.w > 0.01) {
        // Same palette blend as the raster VS (core.rs vs_main_scene).
        let j0 = u32(src_v[s + 12u]) + sp.p.z;
        let j1 = u32(src_v[s + 13u]) + sp.p.z;
        let j2 = u32(src_v[s + 14u]) + sp.p.z;
        let j3 = u32(src_v[s + 15u]) + sp.p.z;
        let m0 = joints.m[j0];
        let m1 = joints.m[j1];
        let m2 = joints.m[j2];
        let m3 = joints.m[j3];
        wp = ((m0 * pos) * w.x + (m1 * pos) * w.y
            + (m2 * pos) * w.z + (m3 * pos) * w.w).xyz;
        wn = ((m0 * nrm) * w.x + (m1 * nrm) * w.y
            + (m2 * nrm) * w.z + (m3 * nrm) * w.w).xyz;
    } else {
        wp = (sp.model * pos).xyz;
        wn = (sp.model * nrm).xyz;
    }
    let ln = length(wn);
    if (ln > 1e-6) { wn = wn / ln; }
    dst_v[d] = wp.x;
    dst_v[d + 1u] = wp.y;
    dst_v[d + 2u] = wp.z;
    dst_v[d + 3u] = wn.x;
    dst_v[d + 4u] = wn.y;
    dst_v[d + 5u] = wn.z;
}
"#;
