//! Screen-space GI probes and SSR (placement, trace SW/HW/SDF,
//! temporal, resolve). Split from renderer/shaders.rs.

// ============================================================================
// Ticket 007a — Lumen-style screen-probe SSGI (software Hi-Z trace)
//
// One probe per 16×16 half-res-pixel tile. Each probe stores 64 radiance
// samples in an 8×8 octahedral atlas. Passes: place → trace → temporal →
// resolve. The resolve pass writes the legacy `ssgi_rt` so downstream
// compositing is untouched.
// ============================================================================

/// Shared helpers prepended to every probe compute/fragment shader.
/// Contains octahedral encode/decode, view-space reconstruction, and
/// the Hi-Z sample helper. Kept as a Rust &str so it can be prepended
/// in the shader-module setup without a WGSL include mechanism.
pub(in crate::renderer) const PROBE_HELPERS_WGSL: &str = "
const PROBE_TILE_SIZE: u32 = 16u;
const PROBE_OCT_SIZE: u32 = 8u;
const PROBE_OCT_TEXELS: u32 = 64u;
const HIZ_SKY_Z: f32 = 10000.0;
const PI: f32 = 3.14159265;

struct ProbeHeader {
    // xyz = world-space probe position; w = valid (1.0 = on surface, 0.0 = sky/invalid)
    world_pos: vec4<f32>,
    // xyz = world-space normal at the probe surface; w = linear |view-z|
    normal: vec4<f32>,
    // xyz = cosine-convolved diffuse radiance, w = reserved.
    diffuse: vec4<f32>,
    // Prior placement at this screen-probe slot. Temporal history is only
    // retained when both placements describe the same surface.
    previous_world_pos: vec4<f32>,
    previous_normal: vec4<f32>,
};

fn probe_history_geometry_valid(
    current: ProbeHeader,
    previous_slot: ProbeHeader,
    maximum_world_shift: f32,
) -> bool {
    if (current.world_pos.w < 0.5 || previous_slot.previous_world_pos.w < 0.5) {
        return false;
    }
    let normal_similarity = dot(current.normal.xyz, previous_slot.previous_normal.xyz);
    return normal_similarity >= 0.85
        && distance(current.world_pos.xyz, previous_slot.previous_world_pos.xyz)
            <= maximum_world_shift;
}

fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    let s = vec2<f32>(
        select(-1.0, 1.0, v.x >= 0.0),
        select(-1.0, 1.0, v.y >= 0.0),
    );
    return (1.0 - abs(vec2<f32>(v.y, v.x))) * s;
}

fn oct_encode(n_in: vec3<f32>) -> vec2<f32> {
    let n = n_in / (abs(n_in.x) + abs(n_in.y) + abs(n_in.z));
    let xy = select(oct_wrap(n.xy), n.xy, n.z >= 0.0);
    return xy * 0.5 + 0.5;
}

fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.0);
    n.x = n.x + select(t, -t, n.x >= 0.0);
    n.y = n.y + select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn octel_direction(octel: vec2<u32>) -> vec3<f32> {
    let uv = (vec2<f32>(octel) + vec2<f32>(0.5)) / f32(PROBE_OCT_SIZE);
    return oct_decode(uv);
}

// Map a 10x10 WSRC slab texel to its wrapped 8x8 octahedral texel. The
// double fold at each padded corner crosses both silhouette edges:
// (0,0)->(7,7), (0,9)->(7,0), (9,0)->(0,7), (9,9)->(0,0).
fn wsrc_real_octel(padded: vec2<i32>) -> vec2<u32> {
    let oct_size = i32(PROBE_OCT_SIZE);
    let padded_max = oct_size + 1;
    let is_edge_x = padded.x == 0 || padded.x == padded_max;
    let is_edge_y = padded.y == 0 || padded.y == padded_max;
    var real = padded - vec2<i32>(1);
    if (is_edge_x && is_edge_y) {
        real = vec2<i32>(
            select(oct_size - 1, 0, padded.x == padded_max),
            select(oct_size - 1, 0, padded.y == padded_max),
        );
    } else if (is_edge_y) {
        real = vec2<i32>(
            oct_size - padded.x,
            clamp(padded.y - 1, 0, oct_size - 1),
        );
    } else if (is_edge_x) {
        real = vec2<i32>(
            clamp(padded.x - 1, 0, oct_size - 1),
            oct_size - padded.y,
        );
    }
    return vec2<u32>(real);
}

fn view_pos_from_linear(uv: vec2<f32>, linear_z: f32,
                        p00: f32, p11: f32, p20: f32, p21: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let view_z = -linear_z;
    let view_x = -(ndc_x + p20) * view_z / p00;
    let view_y = -(ndc_y + p21) * view_z / p11;
    return vec3<f32>(view_x, view_y, view_z);
}

fn bounded_probe_history(value: vec3<f32>) -> vec3<f32> {
    // Rgba16Float can retain undefined Inf/NaN bytes across allocation reuse.
    // Componentwise comparison rejects both without changing finite radiance.
    return select(
        vec3<f32>(0.0),
        value,
        abs(value) <= vec3<f32>(65504.0),
    );
}

fn safe_probe_direction(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let clean = bounded_probe_history(value);
    let len2 = dot(clean, clean);
    if (len2 <= 0.000001) {
        return fallback;
    }
    return clean * inverseSqrt(len2);
}
";

/// Probe placement. One workgroup invocation per probe tile writes a
/// ProbeHeader (world position + world normal + linear view-z). Sky
/// probes are flagged invalid (world_pos.w = 0). Placement stays at the tile
/// centre: camera motion already supplies subpixel coverage, while changing
/// the sampled surface point every frame made static indirect light sparkle.
pub(in crate::renderer) const SSGI_PROBE_PLACE_WGSL: &str = "
struct PlaceParams {
    // Full inverse view matrix — used to lift view-space positions/normals
    // back into world space so the trace can march across the scene.
    inv_view: mat4x4<f32>,
    // x = proj[0][0], y = proj[1][1], z = proj[2][0], w = proj[2][1]
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = reserved, y = tile_size_f (16.0), zw unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: PlaceParams;
@group(0) @binding(1) var hiz0: texture_2d<f32>;
@group(0) @binding(2) var hiz_samp: sampler;
@group(0) @binding(3) var<storage, read_write> probes: array<ProbeHeader>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (gid.x >= grid_w || gid.y >= grid_h) { return; }

    let probe_idx = gid.y * grid_w + gid.x;
    // Preserve the prior placement before publishing this frame's surface.
    // Temporal accumulation compares both records after placement.
    probes[probe_idx].previous_world_pos = probes[probe_idx].world_pos;
    probes[probe_idx].previous_normal = probes[probe_idx].normal;
    let half_w = f32(u.size.x);
    let half_h = f32(u.size.y);
    let tile = u.params.y;
    let px_x = f32(gid.x) * tile + tile * 0.5;
    let px_y = f32(gid.y) * tile + tile * 0.5;
    let uv = vec2<f32>(px_x / half_w, px_y / half_h);

    let linear_z = textureSampleLevel(hiz0, hiz_samp, uv, 0.0).r;

    // Sky probe — mark invalid and bail.
    if (linear_z >= HIZ_SKY_Z * 0.5) {
        probes[probe_idx].world_pos = vec4<f32>(0.0);
        probes[probe_idx].normal = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        return;
    }

    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;
    let P = view_pos_from_linear(uv, linear_z, p00, p11, p20, p21);

    // Finite-difference normal from 3-tap view-pos cross product. One
    // texel to the right and one up. Uses the same Hi-Z mip 0 the
    // center tap read from.
    let texel = vec2<f32>(1.0 / half_w, 1.0 / half_h);
    let uv_r = uv + vec2<f32>(texel.x, 0.0);
    let uv_u = uv + vec2<f32>(0.0, -texel.y);
    let zr = textureSampleLevel(hiz0, hiz_samp, uv_r, 0.0).r;
    let zu = textureSampleLevel(hiz0, hiz_samp, uv_u, 0.0).r;
    let P_r = view_pos_from_linear(uv_r, zr, p00, p11, p20, p21);
    let P_u = view_pos_from_linear(uv_u, zu, p00, p11, p20, p21);
    let N_vs = safe_probe_direction(
        cross(P_r - P, P_u - P),
        vec3<f32>(0.0, 0.0, 1.0),
    );

    let P_world = (u.inv_view * vec4<f32>(P, 1.0)).xyz;
    let N_world = safe_probe_direction(
        (u.inv_view * vec4<f32>(N_vs, 0.0)).xyz,
        vec3<f32>(0.0, 1.0, 0.0),
    );

    probes[probe_idx].world_pos = vec4<f32>(P_world, 1.0);
    probes[probe_idx].normal = vec4<f32>(N_world, linear_z);
}
";

/// Probe trace, software (Hi-Z) path.
///
/// One workgroup per probe; each of the 64 lanes handles one octahedral
/// texel = one ray direction. Hemisphere-cull: rays below the probe's
/// tangent plane contribute zero (not visible from this surface
/// orientation). Surviving rays march the Hi-Z depth pyramid in view
/// space and sample the HDR buffer at hit. Misses contribute zero —
/// sky/off-screen handling is the compose pass's job downstream.
pub(in crate::renderer) const SSGI_PROBE_TRACE_SW_WGSL: &str = "
struct TraceParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = frame_index, y = intensity, z = max_march_t_world, w = firefly_cap
    params: vec4<f32>,
    // Ticket 014 V3/V6/V13 — rest of the shared `ProbeTraceParams`
    // layout. Ignored by Hi-Z; present only so the shader struct
    // size matches the host uniform buffer. V13 replaced the single
    // `wsrc` vec4 with a 3-element cascade array (xyz = origin,
    // w = extent).
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    clipmap: vec4<f32>,
    wsrc_cascades: array<vec4<f32>, 3>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    shadow_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TraceParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var hiz0: texture_2d<f32>;
@group(0) @binding(3) var hiz1: texture_2d<f32>;
@group(0) @binding(4) var hiz2: texture_2d<f32>;
@group(0) @binding(5) var hiz3: texture_2d<f32>;
@group(0) @binding(6) var hiz4: texture_2d<f32>;
@group(0) @binding(7) var hiz_samp: sampler;
@group(0) @binding(8) var hdr_tex: texture_2d<f32>;
@group(0) @binding(9) var hdr_samp: sampler;
@group(0) @binding(10) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(11) var prev_history: texture_3d<f32>;

fn hiz_sample(uv: vec2<f32>, mip: i32) -> f32 {
    switch (clamp(mip, 0, 4)) {
        case 0: { return textureSampleLevel(hiz0, hiz_samp, uv, 0.0).r; }
        case 1: { return textureSampleLevel(hiz1, hiz_samp, uv, 0.0).r; }
        case 2: { return textureSampleLevel(hiz2, hiz_samp, uv, 0.0).r; }
        case 3: { return textureSampleLevel(hiz3, hiz_samp, uv, 0.0).r; }
        default: { return textureSampleLevel(hiz4, hiz_samp, uv, 0.0).r; }
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }
    if (lid.x >= PROBE_OCT_SIZE || lid.y >= PROBE_OCT_SIZE) { return; }

    let probe_idx = wg.y * grid_w + wg.x;
    let header = probes[probe_idx];

    let dst_coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));

    // Invalid probe (sky tile) → contribute zero.
    if (header.world_pos.w < 0.5) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // The 8x8 octahedral centres are a deterministic spherical quadrature.
    // Temporal direction jitter made a static scene produce new radiance
    // every frame and coupled each ray to the previous *screen-grid* slot.
    // Fixed world-space directions retain all 64 samples, eliminate that
    // camera-relative noise, and remove one history texture read per ray.
    let dir_ws = octel_direction(lid.xy);
    let n_ws = header.normal.xyz;

    // Hemisphere cull — rays pointing below the surface carry no diffuse contribution.
    let ndotd = dot(dir_ws, n_ws);
    if (ndotd <= 0.0) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // Trace in view space so the Hi-Z march lines up directly with the
    // rasterized depth pyramid. Start origin at probe_pos + small normal
    // offset to avoid self-intersection at shading point.
    let origin_ws = header.world_pos.xyz + n_ws * 0.02;
    let origin_vs = (u.view * vec4<f32>(origin_ws, 1.0)).xyz;
    let dir_vs = normalize((u.view * vec4<f32>(dir_ws, 0.0)).xyz);

    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;

    let max_t = u.params.z;
    var t = 0.05;
    let n_steps: i32 = 14;
    let growth = pow(max_t / t, 1.0 / f32(n_steps));

    var hit_color = vec3<f32>(0.0);
    var prev_t = 0.0;

    for (var s = 0; s < n_steps; s = s + 1) {
        let pt_vs = origin_vs + dir_vs * t;
        let clip = u.proj * vec4<f32>(pt_vs, 1.0);
        let ndc = clip.xyz / clip.w;

        // Off-screen — no hit possible, stop.
        if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
            break;
        }

        let ray_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

        // Pick the mip such that step footprint ≈ one mip texel. Longer
        // steps sample coarser mips so the early-out fires at coarse
        // resolution; only the last few steps hit mip 0.
        let step_size = t - prev_t;
        let mip = clamp(i32(floor(log2(max(step_size / 0.05, 1.0)))), 0, 4);

        let scene_z = hiz_sample(ray_uv, mip);
        // Hi-Z stores positive |view-z|. ray view-z is negative.
        let ray_abs_z = -pt_vs.z;

        // Have we marched behind a surface? The step tolerance scales
        // with step size: the final tolerance (step_size * 2 + 0.1)
        // lets coarse steps accept wider thickness, which matches the
        // existing SSGI behaviour closely enough for V1.
        if (ray_abs_z >= scene_z && scene_z < HIZ_SKY_Z * 0.5) {
            // Refine against mip 0 to reject coarse-footprint false
            // positives. Exponential stepping can cross a real surface by
            // much more than the old thickness window, so a confirmed
            // front-depth crossing is the hit instead of being dropped.
            let refined_z = hiz_sample(ray_uv, 0);
            if (ray_abs_z >= refined_z && refined_z < HIZ_SKY_Z * 0.5) {
                let tn = t / max_t;
                let falloff = 1.0 - tn * tn;
                var raw = bounded_probe_history(
                    textureSampleLevel(hdr_tex, hdr_samp, ray_uv, 0.0).rgb,
                ) * max(falloff, 0.0);
                // Firefly clamp (cap per-sample luma).
                let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
                let cap = u.params.w;
                if (luma > cap) { raw = raw * (cap / luma); }
                hit_color = raw;
                break;
            }
        }

        prev_t = t;
        t = t * growth;
    }

    let intensity = u.params.y;
    let output = bounded_probe_history(hit_color * intensity * ndotd);
    textureStore(radiance_out, dst_coord, vec4<f32>(output, 1.0));
}
";

/// Probe trace, hardware (ray-query) path (ticket 007b).
///
/// Same workgroup shape as the SW shader — one workgroup per probe, 64
/// lanes per probe, each handling one octahedral texel. The per-ray
/// inner loop replaces Hi-Z screen-space marching with `rayQuery`
/// against the TLAS, which pulls off-screen geometry into the bounce
/// (the whole point of HW-RT here).
///
/// Hit shading is "hit-lighting-lite":
///   - flat per-instance albedo + world-space normal from
///     `instance_data[hit.instance_custom_data]`;
///   - sun direct: NdotL × sun_color × cascaded-shadow visibility;
///     uncovered Mesh-Card hits must not invent camera-moving sunlight;
///   - sky: max(dot(N, up), 0) × sky_color for the upward hemisphere;
///   - emissive: per-instance scalar × albedo;
///   - distance falloff and firefly clamp match the SW path so the
///     two trace variants are visually interchangeable where they
///     both see on-screen geometry.
pub(in crate::renderer) const SSGI_PROBE_TRACE_HW_WGSL: &str = "
struct TraceParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    size: vec4<u32>,
    params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    // Ticket 014 V3/V6/V13 — clipmap + WSRC cascade array. HW path
    // consumes `wsrc_cascades` on its miss branch; the clipmap field
    // is padding here (HW ray-query has its own world-space trace).
    clipmap: vec4<f32>,
    wsrc_cascades: array<vec4<f32>, 3>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    shadow_params: vec4<f32>,
};

struct InstanceGiData {
    albedo: vec3<f32>,
    emissive_luma: f32,
    normal_ws: vec3<f32>,
    _pad0: f32,
    // Ticket 013 V2: x = first_slot_index (first of 6 consecutive
    // signed-axis slots), yz unused, w = has_card flag.
    card_slot: vec4<f32>,
    // Object-space AABB min (xyz) / max (xyz).
    card_aabb_min: vec4<f32>,
    card_aabb_max: vec4<f32>,
    // EN-023 — world-space AABB (SDF path only; layout mirror).
    world_aabb_min: vec4<f32>,
    world_aabb_max: vec4<f32>,
    // PT-2 — layout mirror only (path-tracer geometry window +
    // material params); the GI traces ignore both fields.
    geo: vec4<u32>,
    mat_params: vec4<f32>,
};

const CARD_SLOTS_PER_ROW: f32 = 64.0;
const HW_WSRC_GRID_RES: i32 = 16;
const BLOOM_TRANSPARENT_GI: bool = false;

@group(0) @binding(0) var<uniform> u: TraceParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var accel: acceleration_structure;
@group(0) @binding(3) var<storage, read> instance_data: array<InstanceGiData>;
@group(0) @binding(4) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var card_atlas: texture_2d<f32>;
@group(0) @binding(6) var card_samp: sampler;
@group(0) @binding(7) var wsrc_atlas: texture_3d<f32>;
@group(0) @binding(8) var wsrc_samp: sampler;
@group(0) @binding(9) var prev_history: texture_3d<f32>;
@group(0) @binding(10) var shadow_atlas_0: texture_depth_2d;
@group(0) @binding(11) var shadow_atlas_1: texture_depth_2d;
@group(0) @binding(12) var shadow_atlas_2: texture_depth_2d;
@group(0) @binding(13) var shadow_samp: sampler_comparison;
@group(0) @binding(14) var card_emissive_atlas: texture_2d<f32>;
@group(0) @binding(15) var card_radiance_atlas: texture_2d<f32>;

// Ticket 014 V7/V8 — WSRC lookup shared with the SDF path. V8
// trilinear across the 8 neighbouring probes, nearest octel for
// direction. extent=0 is the cache-not-ready sentinel that the
// host writes before the first bake completes, so the HW miss
// falls back to the pre-V7 return-black behaviour.
// Ticket 014 V10/V13 — HW mirror of the SDF sampler-based WSRC
// lookup. Same 48-slice cascade packing + smallest-containing-
// cascade selection.
fn hw_wsrc_sample_probe(cascade: i32, gx: i32, gy: i32, gz_f: f32, ru: vec2<f32>) -> vec3<f32> {
    let gxc = clamp(gx, 0, 15);
    let gyc = clamp(gy, 0, 15);
    let ax = (f32(gxc) + 0.1 + ru.x * 0.8) / 16.0;
    let ay = (f32(gyc) + 0.1 + ru.y * 0.8) / 16.0;
    let az = (f32(cascade) * 16.0 + gz_f) / 48.0;
    return textureSampleLevel(wsrc_atlas, wsrc_samp,
        vec3<f32>(ax, ay, az), 0.0).rgb;
}

fn hw_wsrc_pick_cascade(pos_ws: vec3<f32>) -> i32 {
    for (var c: i32 = 0; c < 3; c = c + 1) {
        let origin = u.wsrc_cascades[c].xyz;
        let extent = u.wsrc_cascades[c].w;
        if (extent <= 0.0) { continue; }
        let rel = pos_ws - origin;
        let half = extent * 0.5;
        if (abs(rel.x) < half && abs(rel.y) < half && abs(rel.z) < half) {
            return c;
        }
    }
    return -1;
}

fn hw_wsrc_sample(pos_ws: vec3<f32>, dir_ws: vec3<f32>) -> vec3<f32> {
    let cascade = hw_wsrc_pick_cascade(pos_ws);
    if (cascade < 0) {
        return vec3<f32>(0.0);
    }
    let origin = u.wsrc_cascades[cascade].xyz;
    let extent = u.wsrc_cascades[cascade].w;
    let cell = extent / 16.0;
    let rel = pos_ws - origin + vec3<f32>(extent * 0.5);
    let pf = rel / cell - vec3<f32>(0.5);
    let pfx = floor(pf.x);
    let pfy = floor(pf.y);
    let gix = i32(pfx);
    let giy = i32(pfy);
    let fx = pf.x - pfx;
    let fy = pf.y - pfy;
    let gz_f = clamp(pf.z + 0.5, 0.5, 15.5);

    let ru = oct_encode(dir_ws);

    let c00 = hw_wsrc_sample_probe(cascade, gix,     giy,     gz_f, ru);
    let c10 = hw_wsrc_sample_probe(cascade, gix + 1, giy,     gz_f, ru);
    let c01 = hw_wsrc_sample_probe(cascade, gix,     giy + 1, gz_f, ru);
    let c11 = hw_wsrc_sample_probe(cascade, gix + 1, giy + 1, gz_f, ru);

    let ix = 1.0 - fx;
    let iy = 1.0 - fy;
    return c00 * (ix * iy) + c10 * (fx * iy)
         + c01 * (ix * fy) + c11 * (fx * fy);
}

fn hw_gi_cap(raw_in: vec3<f32>) -> vec3<f32> {
    var raw = raw_in;
    let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
    let cap = u.params.w;
    if (luma > cap) { raw = raw * (cap / luma); }
    return raw;
}

fn hw_gi_miss(_origin_ws: vec3<f32>, dir_ws: vec3<f32>, _max_t: f32) -> vec3<f32> {
    // The ray query has already proved this direction contains no geometry
    // inside the SSGI radius. Sampling an unrelated camera-following cache at
    // its terminal point injected rebasing light/dark blocks into open rays.
    // A hardware miss represents the distant environment, which is spatially
    // invariant; local bounced radiance remains owned by actual geometry hits.
    let up = clamp(dir_ws.y * 0.5 + 0.5, 0.0, 1.0);
    return hw_gi_cap(u.sky_color.xyz * up * up);
}

fn hw_gi_card_axis(dir_os: vec3<f32>) -> u32 {
    let abs_d = abs(dir_os);
    var axis_idx: u32 = 0u;
    if (abs_d.y >= abs_d.x && abs_d.y >= abs_d.z) {
        axis_idx = 2u;
    } else if (abs_d.z >= abs_d.x) {
        axis_idx = 4u;
    }
    var signed_axis: u32 = axis_idx;
    if (axis_idx == 0u && dir_os.x > 0.0) { signed_axis = 1u; }
    else if (axis_idx == 2u && dir_os.y > 0.0) { signed_axis = 3u; }
    else if (axis_idx == 4u && dir_os.z > 0.0) { signed_axis = 5u; }
    return signed_axis;
}

fn hw_gi_card_normal_ws(
    normal_oct: vec2<f32>,
    world_to_object: mat4x3<f32>,
    incoming_dir_ws: vec3<f32>,
) -> vec3<f32> {
    let normal_os = oct_decode(normal_oct);

    // Normals transform by inverse-transpose. The ray query supplies the
    // inverse (`world_to_object`), and row-vector multiplication applies its
    // transpose in WGSL. This remains correct under non-uniform scale.
    let inverse_linear = mat3x3<f32>(
        world_to_object[0],
        world_to_object[1],
        world_to_object[2],
    );
    var normal_ws = safe_probe_direction(normal_os * inverse_linear, -incoming_dir_ws);
    // The raster scene is two-sided. Orient an authored back-face normal
    // toward the incident hemisphere so its diffuse bounce matches that
    // convention instead of disappearing or lighting through the surface.
    if (dot(normal_ws, incoming_dir_ws) > 0.0) {
        normal_ws = -normal_ws;
    }
    return normal_ws;
}

fn hw_gi_card_uv(
    inst: InstanceGiData,
    hit_os: vec3<f32>,
    signed_axis: u32,
) -> vec2<f32> {

    let slot = u32(inst.card_slot.x) + signed_axis;
    let slot_x = slot % 64u;
    let slot_y = slot / 64u;
    let bmin = inst.card_aabb_min.xyz;
    let bmax = inst.card_aabb_max.xyz;
    var u_os: f32;
    var v_os: f32;
    var u_lo: f32;
    var u_hi: f32;
    var v_lo: f32;
    var v_hi: f32;
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

    let slot_size_uv = 1.0 / CARD_SLOTS_PER_ROW;
    let texel_in_slot = slot_size_uv / f32(64);
    let slot_u0 = f32(slot_x) * slot_size_uv + texel_in_slot;
    let slot_v0 = f32(slot_y) * slot_size_uv + texel_in_slot;
    let slot_span = slot_size_uv - 2.0 * texel_in_slot;
    return vec2<f32>(
        slot_u0 + u_norm * slot_span,
        slot_v0 + v_norm * slot_span,
    );
}

fn hw_gi_shade_hit(
    inst: InstanceGiData,
    hit_world: vec3<f32>,
    hit_os: vec3<f32>,
    dir_ws: vec3<f32>,
    dir_os: vec3<f32>,
    world_to_object: mat4x3<f32>,
    hit_t: f32,
    max_t: f32,
) -> vec3<f32> {
    let tn = hit_t / max_t;
    let falloff = max(1.0 - tn * tn, 0.0);
    if (inst.card_slot.w > 0.5) {
        let signed_axis = hw_gi_card_axis(dir_os);
        let atlas_uv = hw_gi_card_uv(inst, hit_os, signed_axis);
        // Once the coherent scene finishes streaming, the card-light pass has
        // baked exact world-space sun visibility into this atlas. It is stable
        // under camera motion and turns the normal hit path into one fetch.
        if (u.shadow_params.z > 0.5) {
            let pre_lit = textureSampleLevel(
                card_radiance_atlas,
                card_samp,
                atlas_uv,
                0.0,
            ).rgb;
            return hw_gi_cap(pre_lit * falloff);
        }
        let albedo_sample = textureSampleLevel(card_atlas, card_samp, atlas_uv, 0.0);
        let emissive_sample = textureSampleLevel(
            card_emissive_atlas,
            card_samp,
            atlas_uv,
            0.0,
        );
        let albedo = albedo_sample.rgb;
        let emissive = emissive_sample.rgb;
        // Card capture stores the rasterized triangle normal beside the two
        // already-fetched material samples. This avoids the former card-face
        // proxy, which could turn wall detail into an up-facing sun receiver.
        let hit_n = hw_gi_card_normal_ws(
            vec2<f32>(albedo_sample.a, emissive_sample.a),
            world_to_object,
            dir_ws,
        );
        // While BLAS/card admission is still in flight, use only signals that
        // cannot change with camera-fitted shadow cascades. The exact direct
        // term becomes available atomically with the baked radiance atlas.
        let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
        let sky = u.sky_color.xyz * ndotup;
        return hw_gi_cap((albedo * sky + emissive) * falloff);
    }

    let hit_n = inst.normal_ws;
    // The bounded Mesh-Card atlas cannot represent every placement in very
    // large scenes. A single visibility sample for an entire cardless mesh
    // creates facade-sized light fragments when that sample changes. Keep the
    // fallback conservative and invariant: sky + emissive only. Card-backed
    // hits retain the coherent world-space direct-light bake above.
    let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    let sky = u.sky_color.xyz * ndotup;
    return hw_gi_cap(
        inst.albedo * sky * falloff
        + inst.albedo * inst.emissive_luma
    );
}

fn hw_gi_transmittance(inst: InstanceGiData) -> vec3<f32> {
    let absorption = vec3<f32>(
        inst.card_aabb_min.w,
        inst.card_aabb_max.w,
        inst.world_aabb_min.w,
    );
    let physical = clamp(
        inst.albedo * absorption * inst.mat_params.z * inst.mat_params.w,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return mix(vec3<f32>(1.0), physical, clamp(inst.world_aabb_max.w, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }
    if (lid.x >= PROBE_OCT_SIZE || lid.y >= PROBE_OCT_SIZE) { return; }

    let probe_idx = wg.y * grid_w + wg.x;
    let header = probes[probe_idx];

    let dst_coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));

    if (header.world_pos.w < 0.5) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // Stable 64-direction diffuse quadrature. See the Hi-Z path above.
    let dir_ws = octel_direction(lid.xy);
    let n_ws = header.normal.xyz;
    let origin_ws = header.world_pos.xyz + n_ws * 0.02;
    let ndotd = dot(dir_ws, n_ws);
    if (ndotd <= 0.0) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // 2 cm normal offset — matches the SW start_t and keeps primary
    // hits from self-intersecting the surface the probe sits on.
    let max_t = u.params.z;

    var rq: ray_query;
    rayQueryInitialize(&rq, accel, RayDesc(
        0u,
        0xFFu,
        0.001,
        max_t,
        origin_ws,
        dir_ws,
    ));
    if (BLOOM_RAY_QUERY_NEEDS_PROCEED) {
        loop {
            if (!rayQueryProceed(&rq)) { break; }
        }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);

    var radiance = vec3<f32>(0.0);
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        let inst = instance_data[hit.instance_custom_data];
        let hit_world = origin_ws + dir_ws * hit.t;
        let hit_os = (hit.world_to_object * vec4<f32>(hit_world, 1.0)).xyz;
        let dir_os = safe_probe_direction(
            (hit.world_to_object * vec4<f32>(dir_ws, 0.0)).xyz,
            dir_ws,
        );
        let front = hw_gi_shade_hit(
            inst,
            hit_world,
            hit_os,
            dir_ws,
            dir_os,
            hit.world_to_object,
            hit.t,
            max_t,
        );
        radiance = front;

        if (BLOOM_TRANSPARENT_GI && inst.mat_params.z > 0.0) {
            // Transmission instances use TLAS mask bit 1 only. A second
            // query against bit 0 therefore skips the entire glass volume,
            // including its back face, and returns the nearest opaque
            // receiver. This is one bounded continuation, never a layer loop.
            var opaque_rq: ray_query;
            rayQueryInitialize(&opaque_rq, accel, RayDesc(
                0u,
                0x01u,
                0.001,
                max_t,
                origin_ws,
                dir_ws,
            ));
            if (BLOOM_RAY_QUERY_NEEDS_PROCEED) {
                loop {
                    if (!rayQueryProceed(&opaque_rq)) { break; }
                }
            }
            let opaque_hit = rayQueryGetCommittedIntersection(&opaque_rq);
            var behind = hw_gi_miss(origin_ws, dir_ws, max_t);
            if (opaque_hit.kind != RAY_QUERY_INTERSECTION_NONE) {
                let opaque_inst = instance_data[opaque_hit.instance_custom_data];
                let opaque_world = origin_ws + dir_ws * opaque_hit.t;
                let opaque_os = (
                    opaque_hit.world_to_object * vec4<f32>(opaque_world, 1.0)
                ).xyz;
                let opaque_dir_os = safe_probe_direction(
                    (opaque_hit.world_to_object * vec4<f32>(dir_ws, 0.0)).xyz,
                    dir_ws,
                );
                behind = hw_gi_shade_hit(
                    opaque_inst,
                    opaque_world,
                    opaque_os,
                    dir_ws,
                    opaque_dir_os,
                    opaque_hit.world_to_object,
                    opaque_hit.t,
                    max_t,
                );
            }
            let surface_weight = clamp(inst.world_aabb_max.w, 0.0, 1.0)
                * (1.0 - clamp(inst.mat_params.z, 0.0, 1.0));
            radiance = front * surface_weight + behind * hw_gi_transmittance(inst);
        }
    } else {
        // Ticket 014 V7 — miss path samples the WSRC envelope so HW
        // traces that escape scene geometry still contribute sky /
        // sun-visibility signal. Terminal position is the ray's full
        // march distance; direction picks the octel on the nearest
        // probe.
        radiance = hw_gi_miss(origin_ws, dir_ws, max_t);
    }

    let intensity = u.params.y;
    let output = bounded_probe_history(radiance * intensity * ndotd);
    textureStore(radiance_out, dst_coord, vec4<f32>(output, 1.0));
}
";

/// Ticket 014 V3 — probe trace, software SDF sphere-march path.
///
/// Third trace variant alongside the 007a Hi-Z screen-space and 007b
/// HW ray-query paths. Same workgroup shape (8×8 = one workgroup per
/// probe, each lane handles one octahedral texel). Only the per-ray
/// inner loop differs: instead of marching screen-space depth or
/// firing a `rayQuery`, each lane sphere-marches the scene-wide SDF
/// clipmap baked by ticket 014 V2.
///
/// Hit shading is intentionally minimal for V3 — no mesh-card lookup
/// (the clipmap is a merged SDF with no per-instance identity). At
/// hit we estimate the surface normal by finite-differencing the SDF
/// clipmap around the hit point, then apply analytic sun × NdotL +
/// sky × NdotUp against a constant gray albedo. That gives SW-only
/// adapters a working one-bounce indirect — lower quality than the
/// 013 Mesh-Cards HW path but self-contained.
///
/// The clipmap is a single R32Float 3D texture covering a fixed world-
/// space AABB defined at capture (see `SCENE_SDF_CLIPMAP_EXTENT` /
/// `SCENE_SDF_CLIPMAP_ORIGIN` on the Rust side). Rays whose marched
/// position leaves the AABB treat the miss as "open sky".
pub(in crate::renderer) const SSGI_PROBE_TRACE_SDF_WGSL: &str = "
struct TraceParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    size: vec4<u32>,
    // x = frame_index, y = intensity, z = max_march_t, w = firefly_cap
    params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    // xyz = clipmap origin, w = extent (full width, not half)
    clipmap: vec4<f32>,
    // Ticket 014 V6/V13 — WSRC cascade cubes. Each element is
    // (origin xyz, extent w). Cascades are ordered near→far; the
    // miss path picks the smallest cascade whose cube contains the
    // ray-terminal position. extent <= 0 marks an unbaked cascade
    // (per-cascade); the shader falls back to black if none match.
    wsrc_cascades: array<vec4<f32>, 3>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    shadow_params: vec4<f32>,
};

struct SdfInstanceGiData {
    albedo: vec3<f32>,
    emissive_luma: f32,
    normal_ws: vec3<f32>,
    _pad0: f32,
    card_slot: vec4<f32>,
    card_aabb_min: vec4<f32>,
    card_aabb_max: vec4<f32>,
    // EN-023 — world-space AABB. This trace marches a WORLD-space
    // clipmap and has no world_to_object; comparing world hits against
    // the object-space box above only ever worked for identity-
    // transform assets (Sponza).
    world_aabb_min: vec4<f32>,
    world_aabb_max: vec4<f32>,
    // Layout mirror with the CPU/HW/PT record. The SDF path ignores
    // geometry windows but consumes mat_params.zw in the lazy
    // physical-transmission specialization.
    geo: vec4<u32>,
    mat_params: vec4<f32>,
};

const SDF_CARD_SLOTS_PER_ROW: f32 = 64.0;
const SDF_CARD_SLOT_PX: u32 = 64u;
const WSRC_GRID_RES: i32 = 16;
const BLOOM_TRANSPARENT_GI: bool = false;

@group(0) @binding(0) var<uniform> u: TraceParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var clipmap_tex: texture_3d<f32>;
@group(0) @binding(3) var clipmap_samp: sampler;
@group(0) @binding(4) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> instance_data: array<SdfInstanceGiData>;
@group(0) @binding(6) var card_atlas: texture_2d<f32>;
@group(0) @binding(7) var card_samp: sampler;
@group(0) @binding(8) var wsrc_atlas: texture_3d<f32>;
@group(0) @binding(9) var wsrc_samp: sampler;
@group(0) @binding(10) var prev_history: texture_3d<f32>;

fn clipmap_uv(pos_ws: vec3<f32>) -> vec3<f32> {
    let half_extent = u.clipmap.w * 0.5;
    let origin = u.clipmap.xyz;
    return (pos_ws - origin + vec3<f32>(half_extent)) / u.clipmap.w;
}

fn clipmap_sample(pos_ws: vec3<f32>) -> f32 {
    let uv = clipmap_uv(pos_ws);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || uv.z < 0.0 || uv.z > 1.0) {
        // Outside the clipmap — assume wide open (no hit possible).
        return 1e4;
    }
    return textureSampleLevel(clipmap_tex, clipmap_samp, uv, 0.0).r;
}

fn sdf_ray_aabb(
    origin: vec3<f32>,
    dir: vec3<f32>,
    bmin: vec3<f32>,
    bmax: vec3<f32>,
) -> vec2<f32> {
    let tiny_dir = select(
        vec3<f32>(-0.000001),
        vec3<f32>(0.000001),
        dir >= vec3<f32>(0.0),
    );
    let safe_dir = select(
        tiny_dir,
        dir,
        abs(dir) > vec3<f32>(0.000001),
    );
    let t0 = (bmin - origin) / safe_dir;
    let t1 = (bmax - origin) / safe_dir;
    let near3 = min(t0, t1);
    let far3 = max(t0, t1);
    return vec2<f32>(
        max(near3.x, max(near3.y, near3.z)),
        min(far3.x, min(far3.y, far3.z)),
    );
}

fn sdf_gi_transmittance(inst: SdfInstanceGiData) -> vec3<f32> {
    let absorption = vec3<f32>(
        inst.card_aabb_min.w,
        inst.card_aabb_max.w,
        inst.world_aabb_min.w,
    );
    let physical = clamp(
        inst.albedo * absorption * inst.mat_params.z * inst.mat_params.w,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return mix(vec3<f32>(1.0), physical, clamp(inst.world_aabb_max.w, 0.0, 1.0));
}

// Ticket 014 V10/V13 — WSRC lookup via the hardware linear-filtering
// sampler, now multi-cascade. Each cascade occupies 16 z-slices of
// the atlas at depth offset `cascade_idx * 16`. The miss path picks
// the smallest cascade whose cube contains `pos_ws` and does the
// V10 4-sample trilinear inside that cascade.
//
// Atlas packing (per cascade, same within each 16-slice block):
//   probe (gx, gy, gz) at padded octel (ox_p, oy_p in [0, 9]) lives
//   at texel `(gx*10 + ox_p, gy*10 + oy_p, cascade * 16 + gz)`.
//   Real octel sits at padded (ox+1, oy+1). Borders are
//   octahedrally-wrapped at bake (V11).
//
// Sampler uv formula (atlas x-axis): `atlas_uv_x = (gx + 0.1 +
// ru_x * 0.8) / 16`. Z picks the cascade: `atlas_uv_z = (c * 16 +
// gz + 0.5 + fz) / 48` for 3 cascades.
fn wsrc_sample_probe(cascade: i32, gx: i32, gy: i32, gz_f: f32, ru: vec2<f32>) -> vec3<f32> {
    let gxc = clamp(gx, 0, 15);
    let gyc = clamp(gy, 0, 15);
    let ax = (f32(gxc) + 0.1 + ru.x * 0.8) / 16.0;
    let ay = (f32(gyc) + 0.1 + ru.y * 0.8) / 16.0;
    // 48-slice atlas = 3 cascades × 16 probes in Z. Sample at the
    // cascade's slice block; the `gz_f` carries the per-cascade
    // sub-slice fraction (already centred for the sampler).
    let az = (f32(cascade) * 16.0 + gz_f) / 48.0;
    return textureSampleLevel(wsrc_atlas, wsrc_samp,
        vec3<f32>(ax, ay, az), 0.0).rgb;
}

// V13 — pick the first cascade whose cube contains `pos_ws` and is
// built (extent > 0). Returns -1 if none match.
fn wsrc_pick_cascade(pos_ws: vec3<f32>) -> i32 {
    for (var c: i32 = 0; c < 3; c = c + 1) {
        let origin = u.wsrc_cascades[c].xyz;
        let extent = u.wsrc_cascades[c].w;
        if (extent <= 0.0) { continue; }
        let rel = pos_ws - origin;
        let half = extent * 0.5;
        if (abs(rel.x) < half && abs(rel.y) < half && abs(rel.z) < half) {
            return c;
        }
    }
    return -1;
}

fn wsrc_sample(pos_ws: vec3<f32>, dir_ws: vec3<f32>) -> vec3<f32> {
    let cascade = wsrc_pick_cascade(pos_ws);
    if (cascade < 0) {
        return vec3<f32>(0.0);
    }
    let origin = u.wsrc_cascades[cascade].xyz;
    let extent = u.wsrc_cascades[cascade].w;
    let cell = extent / 16.0;
    let rel = pos_ws - origin + vec3<f32>(extent * 0.5);
    let pf = rel / cell - vec3<f32>(0.5);
    let pfx = floor(pf.x);
    let pfy = floor(pf.y);
    let gix = i32(pfx);
    let giy = i32(pfy);
    let fx = pf.x - pfx;
    let fy = pf.y - pfy;
    let gz_f = clamp(pf.z + 0.5, 0.5, 15.5);

    let ru = oct_encode(dir_ws);

    let c00 = wsrc_sample_probe(cascade, gix,     giy,     gz_f, ru);
    let c10 = wsrc_sample_probe(cascade, gix + 1, giy,     gz_f, ru);
    let c01 = wsrc_sample_probe(cascade, gix,     giy + 1, gz_f, ru);
    let c11 = wsrc_sample_probe(cascade, gix + 1, giy + 1, gz_f, ru);

    let ix = 1.0 - fx;
    let iy = 1.0 - fy;
    return c00 * (ix * iy) + c10 * (fx * iy)
         + c01 * (ix * fy) + c11 * (fx * fy);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u.size.z;
    let grid_h = u.size.w;
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }
    if (lid.x >= PROBE_OCT_SIZE || lid.y >= PROBE_OCT_SIZE) { return; }

    let probe_idx = wg.y * grid_w + wg.x;
    let header = probes[probe_idx];
    let dst_coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));

    if (header.world_pos.w < 0.5) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // Stable 64-direction diffuse quadrature. See the Hi-Z path above.
    let dir_ws = octel_direction(lid.xy);
    let n_ws = header.normal.xyz;
    let ndotd = dot(dir_ws, n_ws);
    if (ndotd <= 0.0) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // 2 cm normal offset matches the SW Hi-Z + HW ray-query paths —
    // keeps primary hits from self-intersecting the probe surface.
    let origin_ws = header.world_pos.xyz + n_ws * 0.02;
    let max_t = u.params.z;

    // The transparent specialization bakes these instances out of the scene
    // SDF. Find one nearest conservative world-AABB crossing up front, then
    // combine it with the unchanged opaque SDF result below. The loop is
    // compile-time absent from the ordinary shader variant.
    var transparent_idx: i32 = -1;
    var transparent_t: f32 = max_t + 1.0;
    if (BLOOM_TRANSPARENT_GI) {
        let instance_count = arrayLength(&instance_data);
        for (var i: u32 = 0u; i < instance_count; i = i + 1u) {
            let candidate = instance_data[i];
            if (candidate.mat_params.z <= 0.0) { continue; }
            let interval = sdf_ray_aabb(
                origin_ws,
                dir_ws,
                candidate.world_aabb_min.xyz,
                candidate.world_aabb_max.xyz,
            );
            let entry = max(interval.x, 0.001);
            if (interval.y >= entry && entry < transparent_t && entry < max_t) {
                transparent_idx = i32(i);
                transparent_t = entry;
            }
        }
    }

    // Sphere-trace. Step is the UDF value; convergence when within a
    // voxel's worth of the surface or when we exhaust the budget.
    let voxel_size = u.clipmap.w / 64.0;  // 64³ clipmap resolution
    let hit_threshold = voxel_size * 1.5;
    var t: f32 = 0.0;
    var hit: bool = false;
    for (var s: i32 = 0; s < 48; s = s + 1) {
        let pos = origin_ws + dir_ws * t;
        let d = clipmap_sample(pos);
        if (d < hit_threshold) {
            hit = true;
            break;
        }
        t = t + max(d, voxel_size * 0.5);
        if (t >= max_t) { break; }
    }

    var radiance = vec3<f32>(0.0);
    if (hit) {
        let hit_pos = origin_ws + dir_ws * t;

        // UDF gradient → outward surface normal (flip since gradient
        // points AWAY from the surface in an unsigned field).
        let h = voxel_size;
        let dx = clipmap_sample(hit_pos + vec3<f32>(h, 0.0, 0.0))
               - clipmap_sample(hit_pos - vec3<f32>(h, 0.0, 0.0));
        let dy = clipmap_sample(hit_pos + vec3<f32>(0.0, h, 0.0))
               - clipmap_sample(hit_pos - vec3<f32>(0.0, h, 0.0));
        let dz = clipmap_sample(hit_pos + vec3<f32>(0.0, 0.0, h))
               - clipmap_sample(hit_pos - vec3<f32>(0.0, 0.0, h));
        var grad = vec3<f32>(dx, dy, dz);
        let glen = length(grad);
        if (glen > 1e-4) { grad = grad / glen; }
        let hit_n = -grad;

        // Ticket 014 V4 — broad-phase lookup: walk `instance_data`,
        // find the first AABB (slightly dilated) containing hit_pos.
        // Pick the axis most aligned with the outward normal; project
        // hit onto its card; sample the pre-lit radiance atlas. Falls
        // back to analytic sun/sky × gray when no instance matches
        // (clipmap sentinel voxels, hits inside unaccounted-for
        // geometry, etc.).
        let count = arrayLength(&instance_data);
        var picked: i32 = -1;
        var picked_vol: f32 = 1e30;
        for (var i: u32 = 0u; i < count; i = i + 1u) {
            let ad = instance_data[i];
            if (ad.card_slot.w < 0.5) { continue; }
            if (BLOOM_TRANSPARENT_GI && ad.mat_params.z > 0.0) { continue; }
            // EN-023 — compare the WORLD hit against the WORLD AABB.
            // The old object-space comparison only matched assets whose
            // vertices were already in world space; every transformed
            // instance fell through to the gray analytic fallback.
            // Pick the SMALLEST containing box, not the first: a scene-
            // spanning instance (the shooter's ±140 m terrain proxy)
            // otherwise swallows every hit — walls and trees included —
            // and its mostly-empty side cards darken the bounce.
            let bmin = ad.world_aabb_min.xyz - vec3<f32>(0.05);
            let bmax = ad.world_aabb_max.xyz + vec3<f32>(0.05);
            if (hit_pos.x >= bmin.x && hit_pos.x <= bmax.x &&
                hit_pos.y >= bmin.y && hit_pos.y <= bmax.y &&
                hit_pos.z >= bmin.z && hit_pos.z <= bmax.z) {
                let ext = bmax - bmin;
                let vol = ext.x * ext.y * ext.z;
                if (vol < picked_vol) {
                    picked = i32(i);
                    picked_vol = vol;
                }
            }
        }

        if (picked >= 0) {
            let ad = instance_data[u32(picked)];
            // Pick signed axis from outward normal. Dominant component
            // picks the axis; sign picks + or - face.
            let abs_n = abs(hit_n);
            var axis_idx: u32 = 0u;
            if (abs_n.y >= abs_n.x && abs_n.y >= abs_n.z) {
                axis_idx = 2u;
            } else if (abs_n.z >= abs_n.x) {
                axis_idx = 4u;
            }
            var signed_axis: u32 = axis_idx;
            if (axis_idx == 0u && hit_n.x < 0.0) { signed_axis = 1u; }
            else if (axis_idx == 2u && hit_n.y < 0.0) { signed_axis = 3u; }
            else if (axis_idx == 4u && hit_n.z < 0.0) { signed_axis = 5u; }

            let first_slot = u32(ad.card_slot.x);
            let slot = first_slot + signed_axis;
            let slot_x = slot % 64u;
            let slot_y = slot / 64u;

            // EN-023 — project against the WORLD AABB, consistent with
            // the world-space hit. Exact for the translate+scale (yaw-0)
            // instances the GI proxies use; a yaw-rotated instance would
            // sample its card with rotated UVs — hue still right, which
            // is what the probe integral actually consumes at 64² cards.
            let bmin = ad.world_aabb_min.xyz;
            let bmax = ad.world_aabb_max.xyz;
            var u_os: f32;
            var v_os: f32;
            var u_lo: f32; var u_hi: f32;
            var v_lo: f32; var v_hi: f32;
            var u_flip: f32 = 1.0;
            if (signed_axis == 0u || signed_axis == 1u) {
                u_os = hit_pos.y; v_os = hit_pos.z;
                u_lo = bmin.y; u_hi = bmax.y; v_lo = bmin.z; v_hi = bmax.z;
                if (signed_axis == 1u) { u_flip = -1.0; }
            } else if (signed_axis == 2u || signed_axis == 3u) {
                u_os = hit_pos.x; v_os = hit_pos.z;
                u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.z; v_hi = bmax.z;
                if (signed_axis == 3u) { u_flip = -1.0; }
            } else {
                u_os = hit_pos.x; v_os = hit_pos.y;
                u_lo = bmin.x; u_hi = bmax.x; v_lo = bmin.y; v_hi = bmax.y;
                if (signed_axis == 5u) { u_flip = -1.0; }
            }
            var u_norm = clamp((u_os - u_lo) / max(u_hi - u_lo, 1e-4), 0.0, 1.0);
            let v_norm = clamp((v_os - v_lo) / max(v_hi - v_lo, 1e-4), 0.0, 1.0);
            if (u_flip < 0.0) { u_norm = 1.0 - u_norm; }
            let slot_size_uv = 1.0 / SDF_CARD_SLOTS_PER_ROW;
            let texel_in_slot = slot_size_uv / f32(SDF_CARD_SLOT_PX);
            let slot_u0 = f32(slot_x) * slot_size_uv + texel_in_slot;
            let slot_v0 = f32(slot_y) * slot_size_uv + texel_in_slot;
            let slot_span = slot_size_uv - 2.0 * texel_in_slot;
            let atlas_uv = vec2<f32>(
                slot_u0 + u_norm * slot_span,
                slot_v0 + v_norm * slot_span,
            );
            let pre_lit = textureSampleLevel(card_atlas, card_samp, atlas_uv, 0.0).rgb;

            let tn = t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = pre_lit * falloff;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        } else {
            // Fallback — analytic sun/sky × gray albedo when no
            // instance matches. Same shading as V3.
            let ndotl = max(dot(hit_n, u.sun_dir.xyz), 0.0);
            let direct = u.sun_color.xyz * ndotl;
            let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
            let sky = u.sky_color.xyz * ndotup;
            let albedo = vec3<f32>(0.55, 0.55, 0.55);
            let tn = t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = albedo * (direct + sky) * falloff;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        }
    } else {
        // Ticket 014 V6 — miss path samples the WSRC envelope instead
        // of returning black. Ray terminal position (origin + dir * t)
        // is where we project into the cache; direction picks the
        // probe's octel. Firefly-clamp to match the hit path.
        let terminal = origin_ws + dir_ws * t;
        var raw = wsrc_sample(terminal, dir_ws);
        let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
        let cap = u.params.w;
        if (luma > cap) { raw = raw * (cap / luma); }
        radiance = raw;
    }

    if (BLOOM_TRANSPARENT_GI && transparent_idx >= 0) {
        let opaque_t = select(max_t, t, hit);
        if (transparent_t < opaque_t) {
            let glass = instance_data[u32(transparent_idx)];
            // The SW representation knows the conservative AABB crossing but
            // not an exact triangle normal. Reuse the instance's established
            // flat hit-lighting fallback for the non-transmitted fraction.
            let hit_n = glass.normal_ws;
            let ndotl = max(dot(hit_n, u.sun_dir.xyz), 0.0);
            let direct = u.sun_color.xyz * ndotl;
            let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
            let sky = u.sky_color.xyz * ndotup;
            let tn = transparent_t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var front = glass.albedo * (direct + sky) * falloff
                      + glass.albedo * glass.emissive_luma;
            let front_luma = dot(front, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (front_luma > cap) { front = front * (cap / front_luma); }
            let surface_weight = clamp(glass.world_aabb_max.w, 0.0, 1.0)
                * (1.0 - clamp(glass.mat_params.z, 0.0, 1.0));
            radiance = front * surface_weight
                     + radiance * sdf_gi_transmittance(glass);
        }
    }

    let intensity = u.params.y;
    let output = bounded_probe_history(radiance * intensity * ndotd);
    textureStore(radiance_out, dst_coord, vec4<f32>(output, 1.0));
}
";

/// Probe temporal accumulator. EMA remains in probe-octel space, but history
/// is retained only while current and previous placements represent the same
/// world-space surface. This bounds camera-motion/disocclusion ghosts without
/// another texture or a screen-space velocity dependency.
pub(in crate::renderer) const SSGI_PROBE_TEMPORAL_WGSL: &str = "
struct TemporalParams {
    // x = alpha (0.25 = 4-frame EMA at steady state),
    // y = force_refresh (1 → alpha 1.0),
    // z = grid_w, w = grid_h
    params: vec4<f32>,
    // x = half_w, y = half_h, z = tile_size, w = projection p00
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TemporalParams;
@group(0) @binding(1) var radiance_in: texture_3d<f32>;
@group(0) @binding(2) var history_in: texture_3d<f32>;
@group(0) @binding(3) var history_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> probes: array<ProbeHeader>;
@group(0) @binding(5) var velocity_tex: texture_2d<f32>;

// The trace stores cosine-weighted incident radiance in 64 directional
// octels. Reserve octel zero in the filtered history for the diffuse
// convolution that resolve actually needs. Keeping the reduction in this
// existing workgroup avoids another texture, pass, or per-pixel 64-tap loop.
var<workgroup> diffuse_radiance: array<vec3<f32>, 64>;
var<workgroup> diffuse_luminance: array<f32, 64>;
var<workgroup> reprojected_history_probe: u32;
var<workgroup> reprojected_history_valid: u32;
var<workgroup> reprojected_history_alpha: f32;

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u32(u.params.z);
    let grid_h = u32(u.params.w);
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }

    let probe_index = wg.y * grid_w + wg.x;
    let coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));
    let lane = lid.y * PROBE_OCT_SIZE + lid.x;
    let curr = bounded_probe_history(textureLoad(radiance_in, coord, 0).rgb);

    if (lane == 0u) {
        reprojected_history_probe = probe_index;
        reprojected_history_valid = 0u;
        reprojected_history_alpha = u.params.x;
        let current_probe = probes[probe_index];
        if (current_probe.world_pos.w >= 0.5) {
            let current_uv = (
                vec2<f32>(wg.xy) * u.size.z + vec2<f32>(u.size.z * 0.5)
            ) / u.size.xy;
            let velocity_size = vec2<i32>(textureDimensions(velocity_tex));
            let velocity_coord = clamp(
                vec2<i32>(current_uv * vec2<f32>(velocity_size)),
                vec2<i32>(0),
                velocity_size - vec2<i32>(1),
            );
            let velocity = textureLoad(velocity_tex, velocity_coord, 0).xy;
            let motion_refresh = smoothstep(0.00025, 0.003, length(velocity));
            reprojected_history_alpha = mix(u.params.x, 0.65, motion_refresh);
            // Velocity stores current-minus-previous NDC. UV's Y axis is
            // flipped, matching the established TAA and SSR reprojection.
            let previous_uv = vec2<f32>(
                current_uv.x - velocity.x,
                current_uv.y + velocity.y,
            );
            if (all(previous_uv >= vec2<f32>(0.0)) &&
                all(previous_uv <= vec2<f32>(1.0))) {
                let previous_grid_position =
                    previous_uv * u.size.xy / u.size.z - vec2<f32>(0.5);
                let previous_grid_center = vec2<i32>(
                    floor(previous_grid_position + vec2<f32>(0.5)),
                );

                // The nearest prior grid sample can sit half a tile from
                // this surface point. Derive a world-space acceptance
                // radius from that footprint instead of using a fixed
                // tolerance that collapses as resolution/depth changes.
                let probe_world_spacing =
                    2.0 * max(current_probe.normal.w, 0.1) * u.size.z /
                    max(abs(u.size.w) * u.size.x, 0.0001);
                let maximum_world_shift = 0.05 + probe_world_spacing * 0.9;
                var best_score = 1e30;
                for (var dy = -1; dy <= 1; dy = dy + 1) {
                    for (var dx = -1; dx <= 1; dx = dx + 1) {
                        let candidate_xy = previous_grid_center + vec2<i32>(dx, dy);
                        if (candidate_xy.x < 0 || candidate_xy.y < 0 ||
                            candidate_xy.x >= i32(grid_w) || candidate_xy.y >= i32(grid_h)) {
                            continue;
                        }
                        let candidate_index =
                            u32(candidate_xy.y) * grid_w + u32(candidate_xy.x);
                        let candidate = probes[candidate_index];
                        if (!probe_history_geometry_valid(
                            current_probe,
                            candidate,
                            maximum_world_shift,
                        )) {
                            continue;
                        }
                        let world_shift = distance(
                            current_probe.world_pos.xyz,
                            candidate.previous_world_pos.xyz,
                        );
                        let normal_penalty = 1.0 - clamp(dot(
                            current_probe.normal.xyz,
                            candidate.previous_normal.xyz,
                        ), 0.0, 1.0);
                        let score = world_shift + normal_penalty * maximum_world_shift;
                        if (score < best_score) {
                            best_score = score;
                            reprojected_history_probe = candidate_index;
                            reprojected_history_valid = 1u;
                        }
                    }
                }
            }
        }
    }
    workgroupBarrier();

    let history_x = reprojected_history_probe % grid_w;
    let history_y = reprojected_history_probe / grid_w;
    let history_coord = vec3<i32>(
        i32(history_x),
        i32(history_y),
        i32(lane),
    );
    let geometry_valid = reprojected_history_valid != 0u;
    var hist = bounded_probe_history(textureLoad(history_in, history_coord, 0).rgb);
    // Octel zero held the previous frame's integrated irradiance rather than
    // directional history. Seed that one directional sample from current.
    if (lane == 0u) {
        hist = curr;
    }

    // Fixed trace directions need less temporal smoothing during camera
    // motion. Refresh over roughly 0.1..1.5 output pixels so high-contrast
    // world-cache radiance does not trail behind its receiver, while a
    // stationary camera retains the established four-frame EMA.
    var alpha = reprojected_history_alpha;
    let force_refresh = u.params.y > 0.5;
    if (force_refresh || !geometry_valid) {
        alpha = 1.0;
    } else {
        // Ticket 016 V4 — variance-adaptive alpha. Scale the base
        // EMA by `|luma(curr) - luma(hist)|` so moving lights /
        // disocclusions / scene cuts converge quickly while stable
        // octels keep strong temporal smoothing. This captures the
        // hierarchical-refinement intent (high-variance regions get
        // more per-frame weight, low-variance regions average more
        // history) without needing a separate refinement probe
        // layer + indirect dispatch.
        //
        // `luma_delta_scale = 0.6` means a 1.0-luma delta pushes
        // alpha up by 0.6 on top of the 0.25 base — up to 0.85
        // before the `min(1.0)` clamp.
        let curr_luma = dot(curr, vec3<f32>(0.2126, 0.7152, 0.0722));
        let hist_luma = dot(hist, vec3<f32>(0.2126, 0.7152, 0.0722));
        let delta = abs(curr_luma - hist_luma);
        alpha = min(1.0, alpha + delta * 0.6);
    }
    var blended = mix(hist, curr, alpha);
    if (force_refresh || !geometry_valid) {
        // `mix(undefined, current, 1)` may still evaluate undefined * zero.
        // Direct assignment guarantees invalid history is never observed.
        blended = curr;
    }

    diffuse_radiance[lane] = blended;
    diffuse_luminance[lane] = dot(
        blended,
        vec3<f32>(0.2126, 0.7152, 0.0722),
    );
    workgroupBarrier();
    if (lane < 8u) {
        var row_sum = 0.0;
        let row_start = lane * 8u;
        for (var column = 0u; column < 8u; column = column + 1u) {
            row_sum = row_sum + diffuse_luminance[row_start + column];
        }
        diffuse_luminance[lane] = row_sum;
    }
    workgroupBarrier();
    if (lane == 0u) {
        var probe_sum = 0.0;
        for (var row = 0u; row < 8u; row = row + 1u) {
            probe_sum = probe_sum + diffuse_luminance[row];
        }
        diffuse_luminance[0] = probe_sum;
    }
    workgroupBarrier();

    // One fixed ray that happens to intersect a tiny bright texture or lamp
    // represents 1/64 of the sphere even when the source covers far less
    // solid angle. Prevent that quadrature outlier from becoming a whole
    // 16-pixel probe streak. The 8x8 octahedral grid's worst smooth cosine
    // field is 4.21x its mean, so 5x preserves every smooth sky/sun field and
    // only winsorizes energy too concentrated for this sampling density.
    let ray_luminance = dot(
        diffuse_radiance[lane],
        vec3<f32>(0.2126, 0.7152, 0.0722),
    );
    let mean_luminance = diffuse_luminance[0] / 64.0;
    let solid_angle_cap = mean_luminance * 5.0;
    if (ray_luminance > solid_angle_cap && ray_luminance > 0.0) {
        diffuse_radiance[lane] =
            diffuse_radiance[lane] * (solid_angle_cap / ray_luminance);
    }
    workgroupBarrier();
    if (lane < 32u) {
        diffuse_radiance[lane] = diffuse_radiance[lane] + diffuse_radiance[lane + 32u];
    }
    workgroupBarrier();
    if (lane < 16u) {
        diffuse_radiance[lane] = diffuse_radiance[lane] + diffuse_radiance[lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        diffuse_radiance[lane] = diffuse_radiance[lane] + diffuse_radiance[lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        diffuse_radiance[lane] = diffuse_radiance[lane] + diffuse_radiance[lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        diffuse_radiance[lane] = diffuse_radiance[lane] + diffuse_radiance[lane + 2u];
    }
    workgroupBarrier();
    if (lane == 0u) {
        diffuse_radiance[0] = diffuse_radiance[0] + diffuse_radiance[1];
        // Uniform sphere samples need 4x their mean cosine-weighted
        // radiance to reproduce constant diffuse incident radiance.
        blended = bounded_probe_history(diffuse_radiance[0] * (4.0 / 64.0));
        probes[probe_index].diffuse = vec4<f32>(blended, 1.0);
    }

    textureStore(history_out, coord, vec4<f32>(blended, 1.0));
}
";

/// Per-pixel probe-cache reconstruction. Writes the half-res ssgi_rt
/// that the downstream compose / TAA passes already read.
///
/// Samples the 2×2 probes whose tiles enclose the pixel's tile. For
/// each probe, evaluates the octahedral atlas along the pixel's
/// world-space normal, then bilateral-weights the contribution by
/// depth-match + normal-match with the pixel itself. Invalid probes
/// (sky) are skipped. When all 4 probes reject (pixel depth/normal
/// wildly off), fall back to a zero contribution — better than leaking
/// a stale distant probe's radiance into a foreground surface.
pub(in crate::renderer) const SSGI_PROBE_RESOLVE_WGSL: &str = "
struct ResolveParams {
    inv_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = tile_size (16.0), y = intensity, zw unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ResolveParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var radiance_tex: texture_3d<f32>;
@group(0) @binding(3) var radiance_samp: sampler;
@group(0) @binding(4) var hiz0: texture_2d<f32>;
@group(0) @binding(5) var hiz_samp: sampler;

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

// Temporal stores the cosine-convolved diffuse result alongside the probe
// header. Resolve already loads that header for its bilateral weights, so
// this adds no texture lookup.
fn sample_probe(probe: ProbeHeader) -> vec3<f32> {
    return probe.diffuse.rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let linear_z = textureSampleLevel(hiz0, hiz_samp, in.uv, 0.0).r;
    if (linear_z >= HIZ_SKY_Z * 0.5) {
        return vec4<f32>(0.0);
    }

    let half_w = f32(u.size.x);
    let half_h = f32(u.size.y);
    let tile = u.params.x;
    let grid_w = i32(u.size.z);
    let grid_h = i32(u.size.w);

    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;
    let P_vs = view_pos_from_linear(in.uv, linear_z, p00, p11, p20, p21);

    // Reconstruct pixel normal (same 3-tap trick as the placement pass).
    let texel = vec2<f32>(1.0 / half_w, 1.0 / half_h);
    let zr = textureSampleLevel(hiz0, hiz_samp, in.uv + vec2<f32>(texel.x, 0.0), 0.0).r;
    let zu = textureSampleLevel(hiz0, hiz_samp, in.uv + vec2<f32>(0.0, -texel.y), 0.0).r;
    let Pr = view_pos_from_linear(in.uv + vec2<f32>(texel.x, 0.0), zr, p00, p11, p20, p21);
    let Pu = view_pos_from_linear(in.uv + vec2<f32>(0.0, -texel.y), zu, p00, p11, p20, p21);
    let N_vs = safe_probe_direction(
        cross(Pr - P_vs, Pu - P_vs),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    let N_ws = safe_probe_direction(
        (u.inv_view * vec4<f32>(N_vs, 0.0)).xyz,
        vec3<f32>(0.0, 1.0, 0.0),
    );

    // Pixel's grid-space fractional position (which probes surround it?).
    let px_x = in.uv.x * half_w;
    let px_y = in.uv.y * half_h;
    let fx = px_x / tile - 0.5;  // -0.5 aligns grid cells centred on tile centres
    let fy = px_y / tile - 0.5;
    let gx0 = i32(floor(fx));
    let gy0 = i32(floor(fy));
    let tx = fract(fx);
    let ty = fract(fy);

    var accum = vec3<f32>(0.0);
    var wsum = 0.0;

    for (var dy = 0; dy <= 1; dy = dy + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let gx = clamp(gx0 + dx, 0, grid_w - 1);
            let gy = clamp(gy0 + dy, 0, grid_h - 1);
            let probe = probes[u32(gy * grid_w + gx)];
            if (probe.world_pos.w < 0.5) { continue; }

            // Bilinear corner weight
            var w_corner = 1.0;
            w_corner = w_corner * select(1.0 - tx, tx, dx == 1);
            w_corner = w_corner * select(1.0 - ty, ty, dy == 1);

            // Depth + normal bilateral weights — reject probes on very
            // different surfaces from the pixel (foreground pixel vs
            // probe on a far wall, or on an orthogonal facet).
            let dz = abs(probe.normal.w - linear_z);
            let w_depth = exp(-dz * dz * 8.0);
            let ndotn = clamp(dot(probe.normal.xyz, N_ws), 0.0, 1.0);
            let w_normal = pow(ndotn, 4.0);
            let w = w_corner * w_depth * w_normal;
            if (w <= 0.0001) { continue; }

            let radiance = sample_probe(probe);
            accum = accum + radiance * w;
            wsum = wsum + w;
        }
    }

    if (wsum > 0.0001) {
        accum = (accum / wsum) * u.params.y;
    }
    return vec4<f32>(accum, 1.0);
}
";
