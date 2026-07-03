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
};

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

// Ticket 016 V1/V2 — temporal octahedral direction jitter with
// per-probe decorrelation (V2). V1 indexed a 2D R2 low-discrepancy
// sequence by frame, giving every probe the same per-frame sample
// offset. That means the 3×3 neighbourhood the resolve pass reads
// sees 9 probes sampling identical sub-texel positions — the
// spatial filter averages correlated noise, which is slower to
// converge than independent samples would be.
//
// V2 folds `probe_idx` into the sequence via a third low-
// discrepancy axis (1/g³ ≈ 0.4301597). Adjacent probes now land
// at different sub-texel positions each frame, so the 3×3 read
// effectively samples 9 × 4 = 36 distinct directions per octel
// over the EMA horizon rather than 4. Same zero-cost structure
// as V1 — two `fract` calls with an extra multiply.
const OCT_JITTER_A1: f32 = 0.7548776662;
const OCT_JITTER_A2: f32 = 0.5698402910;
const OCT_JITTER_A3: f32 = 0.4301597090;
fn octel_jitter(frame: f32, probe_idx: u32) -> vec2<f32> {
    // R2 sequence in `frame` + orthogonal axis in `probe_idx`.
    // The two components use different probe-axis scales (a3 vs
    // a3 × R2's irrational) to stay 2D-decorrelated across probes.
    let pf = f32(probe_idx);
    return vec2<f32>(
        fract(0.5 + OCT_JITTER_A1 * frame + OCT_JITTER_A3 * pf) - 0.5,
        fract(0.5 + OCT_JITTER_A2 * frame + OCT_JITTER_A3 * pf * 1.324718) - 0.5,
    );
}
fn octel_direction_jittered(octel: vec2<u32>, jitter: vec2<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(octel) + vec2<f32>(0.5) + jitter) / f32(PROBE_OCT_SIZE);
    return oct_decode(uv);
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

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
";

/// Probe placement. One workgroup invocation per probe tile writes a
/// ProbeHeader (world position + world normal + linear view-z). Sky
/// probes are flagged invalid (world_pos.w = 0). Per-frame Halton-style
/// jitter moves the probe within its 16×16 tile so adjacent frames
/// cover slightly different surface points, widening effective
/// coverage when combined with temporal accumulation.
pub(in crate::renderer) const SSGI_PROBE_PLACE_WGSL: &str = "
struct PlaceParams {
    // Full inverse view matrix — used to lift view-space positions/normals
    // back into world space so the trace can march across the scene.
    inv_view: mat4x4<f32>,
    // x = proj[0][0], y = proj[1][1], z = proj[2][0], w = proj[2][1]
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = frame_index (temporal jitter), y = tile_size_f (16.0), zw unused
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
    let half_w = f32(u.size.x);
    let half_h = f32(u.size.y);
    let tile = u.params.y;
    let frame = u.params.x;

    // Jitter UV inside the tile — 50% of tile radius so probe stays
    // comfortably away from tile borders. Golden-ratio offsets across
    // frames decorrelate the jitter from TAA/SSAO patterns.
    let jx = ign(vec2<f32>(f32(gid.x) + frame * 1.618, f32(gid.y)));
    let jy = ign(vec2<f32>(f32(gid.x), f32(gid.y) + frame * 2.236));
    let px_x = f32(gid.x) * tile + tile * 0.5 + (jx - 0.5) * tile * 0.5;
    let px_y = f32(gid.y) * tile + tile * 0.5 + (jy - 0.5) * tile * 0.5;
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
    let N_vs = normalize(cross(P_r - P, P_u - P));

    let P_world = (u.inv_view * vec4<f32>(P, 1.0)).xyz;
    let N_world = normalize((u.inv_view * vec4<f32>(N_vs, 0.0)).xyz);

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

    // V1 — temporal jitter within each octel; 4-frame EMA turns
    // this into free super-sampling.
    // V2 — probe_idx folded into the jitter so neighbouring probes
    // sample decorrelated sub-texel positions.
    // V3 — scale jitter inversely with prev-frame luma at this octel:
    // already-bright octels narrow their jitter (exploit / lock in
    // the peak); dark octels keep full jitter (explore for new
    // light). Luma is read from the prev-frame temporal-filtered
    // history texture; `dst_coord` indexes the probe × octel slab
    // identically between trace output and history.
    let prev_slice = textureLoad(prev_history, dst_coord, 0).rgb;
    let prev_luma = dot(prev_slice, vec3<f32>(0.2126, 0.7152, 0.0722));
    let jitter_scale = mix(1.0, 0.3, clamp(prev_luma, 0.0, 1.0));
    let jitter = octel_jitter(u.params.x, probe_idx) * jitter_scale;
    let dir_ws = octel_direction_jittered(lid.xy, jitter);
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
            // Refine against mip 0 to reject far-off misses.
            let refined_z = hiz_sample(ray_uv, 0);
            let thickness = abs(ray_abs_z - refined_z);
            if (thickness < step_size * 2.0 + 0.1) {
                let tn = t / max_t;
                let falloff = 1.0 - tn * tn;
                var raw = textureSampleLevel(hdr_tex, hdr_samp, ray_uv, 0.0).rgb * max(falloff, 0.0);
                // Firefly clamp (cap per-sample luma).
                let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
                let cap = u.params.w;
                if (luma > cap) { raw = raw * (cap / luma); }
                hit_color = raw;
            }
            break;
        }

        prev_t = t;
        t = t * growth;
    }

    let intensity = u.params.y;
    textureStore(radiance_out, dst_coord, vec4<f32>(hit_color * intensity * ndotd, 1.0));
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
///   - sun direct: NdotL × sun_color (no cascade shadow lookup in V1 —
///     the bias is one bounce away and hidden by temporal averaging;
///     re-add when shadow-aware hit shading becomes worth the cost);
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
};

const CARD_SLOTS_PER_ROW: f32 = 64.0;
const HW_WSRC_GRID_RES: i32 = 16;

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

    // V1 — temporal jitter within each octel; 4-frame EMA turns
    // this into free super-sampling.
    // V2 — probe_idx folded into the jitter so neighbouring probes
    // sample decorrelated sub-texel positions.
    // V3 — scale jitter inversely with prev-frame luma at this octel:
    // already-bright octels narrow their jitter (exploit / lock in
    // the peak); dark octels keep full jitter (explore for new
    // light). Luma is read from the prev-frame temporal-filtered
    // history texture; `dst_coord` indexes the probe × octel slab
    // identically between trace output and history.
    let prev_slice = textureLoad(prev_history, dst_coord, 0).rgb;
    let prev_luma = dot(prev_slice, vec3<f32>(0.2126, 0.7152, 0.0722));
    let jitter_scale = mix(1.0, 0.3, clamp(prev_luma, 0.0, 1.0));
    let jitter = octel_jitter(u.params.x, probe_idx) * jitter_scale;
    let dir_ws = octel_direction_jittered(lid.xy, jitter);
    let n_ws = header.normal.xyz;
    let ndotd = dot(dir_ws, n_ws);
    if (ndotd <= 0.0) {
        textureStore(radiance_out, dst_coord, vec4<f32>(0.0));
        return;
    }

    // 2 cm normal offset — matches the SW start_t and keeps primary
    // hits from self-intersecting the surface the probe sits on.
    let origin_ws = header.world_pos.xyz + n_ws * 0.02;
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
    loop {
        if (!rayQueryProceed(&rq)) { break; }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);

    var radiance = vec3<f32>(0.0);
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        let inst = instance_data[hit.instance_custom_data];

        // Ticket 013 V2 — pick the axis facing the incoming ray and
        // sample the pre-lit radiance atlas. Each mesh has 6
        // consecutive slots laid out by signed axis (see host-side
        // capture loop). The card's world-space normal was baked at
        // capture into `card_slot_meta`; lighting was applied once
        // per frame by `card_light_pass`, so the sample IS the
        // bounce contribution — no hit-time shading math.
        if (inst.card_slot.w > 0.5) {
            let hit_world = origin_ws + dir_ws * hit.t;
            let hit_os = (hit.world_to_object * vec4<f32>(hit_world, 1.0)).xyz;
            // Dominant component of the world-space ray direction
            // picks the major axis; its sign selects the front/back
            // face of that axis.
            let abs_d = abs(dir_ws);
            var axis_idx: u32 = 0u;
            if (abs_d.y >= abs_d.x && abs_d.y >= abs_d.z) {
                axis_idx = 2u;
            } else if (abs_d.z >= abs_d.x) {
                axis_idx = 4u;
            }
            // Sign: ray going -X (dir.x < 0) lands on +X face → axis + 0.
            // ray going +X lands on -X face → axis + 1.
            var signed_axis: u32 = axis_idx;
            if (axis_idx == 0u && dir_ws.x > 0.0) { signed_axis = 1u; }
            else if (axis_idx == 2u && dir_ws.y > 0.0) { signed_axis = 3u; }
            else if (axis_idx == 4u && dir_ws.z > 0.0) { signed_axis = 5u; }

            let first_slot = u32(inst.card_slot.x);
            let slot = first_slot + signed_axis;
            let slot_x = slot % 64u;
            let slot_y = slot / 64u;

            // Project hit_os onto the card plane — same math as V1 but
            // with signed-axis-aware u sign flips so the ±pair cards
            // pick up opposite views of the mesh.
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
            let texel_in_slot = slot_size_uv / f32(64);  // 64×64 card
            let slot_u0 = f32(slot_x) * slot_size_uv + texel_in_slot;
            let slot_v0 = f32(slot_y) * slot_size_uv + texel_in_slot;
            let slot_span = slot_size_uv - 2.0 * texel_in_slot;
            let atlas_uv = vec2<f32>(
                slot_u0 + u_norm * slot_span,
                slot_v0 + v_norm * slot_span,
            );
            let pre_lit = textureSampleLevel(card_atlas, card_samp, atlas_uv, 0.0).rgb;

            // V3 — emissive is pre-added into the radiance atlas by the
            // card-lighting pass, so the hit simply picks up the full
            // pre-lit texel and applies distance falloff + firefly cap.
            let tn = hit.t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = pre_lit * falloff;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        } else {
            // Fallback path — no card captured, use the flat
            // instance albedo × hit-time lighting same as 007b.
            let hit_n = inst.normal_ws;
            let ndotl = max(dot(hit_n, u.sun_dir.xyz), 0.0);
            let direct = u.sun_color.xyz * ndotl;
            let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
            let sky = u.sky_color.xyz * ndotup;
            let tn = hit.t / max_t;
            let falloff = max(1.0 - tn * tn, 0.0);
            var raw = inst.albedo * (direct + sky) * falloff
                    + inst.albedo * inst.emissive_luma;
            let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
            let cap = u.params.w;
            if (luma > cap) { raw = raw * (cap / luma); }
            radiance = raw;
        }
    } else {
        // Ticket 014 V7 — miss path samples the WSRC envelope so HW
        // traces that escape scene geometry still contribute sky /
        // sun-visibility signal. Terminal position is the ray's full
        // march distance; direction picks the octel on the nearest
        // probe.
        let terminal = origin_ws + dir_ws * max_t;
        var raw = hw_wsrc_sample(terminal, dir_ws);
        let luma = dot(raw, vec3<f32>(0.2126, 0.7152, 0.0722));
        let cap = u.params.w;
        if (luma > cap) { raw = raw * (cap / luma); }
        radiance = raw;
    }

    let intensity = u.params.y;
    textureStore(radiance_out, dst_coord, vec4<f32>(radiance * intensity * ndotd, 1.0));
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
};

struct SdfInstanceGiData {
    albedo: vec3<f32>,
    emissive_luma: f32,
    normal_ws: vec3<f32>,
    _pad0: f32,
    card_slot: vec4<f32>,
    card_aabb_min: vec4<f32>,
    card_aabb_max: vec4<f32>,
};

const SDF_CARD_SLOTS_PER_ROW: f32 = 64.0;
const SDF_CARD_SLOT_PX: u32 = 64u;
const WSRC_GRID_RES: i32 = 16;

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

    // V1 — temporal jitter within each octel; 4-frame EMA turns
    // this into free super-sampling.
    // V2 — probe_idx folded into the jitter so neighbouring probes
    // sample decorrelated sub-texel positions.
    // V3 — scale jitter inversely with prev-frame luma at this octel:
    // already-bright octels narrow their jitter (exploit / lock in
    // the peak); dark octels keep full jitter (explore for new
    // light). Luma is read from the prev-frame temporal-filtered
    // history texture; `dst_coord` indexes the probe × octel slab
    // identically between trace output and history.
    let prev_slice = textureLoad(prev_history, dst_coord, 0).rgb;
    let prev_luma = dot(prev_slice, vec3<f32>(0.2126, 0.7152, 0.0722));
    let jitter_scale = mix(1.0, 0.3, clamp(prev_luma, 0.0, 1.0));
    let jitter = octel_jitter(u.params.x, probe_idx) * jitter_scale;
    let dir_ws = octel_direction_jittered(lid.xy, jitter);
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
        for (var i: u32 = 0u; i < count; i = i + 1u) {
            let ad = instance_data[i];
            if (ad.card_slot.w < 0.5) { continue; }
            let bmin = ad.card_aabb_min.xyz - vec3<f32>(0.05);
            let bmax = ad.card_aabb_max.xyz + vec3<f32>(0.05);
            if (hit_pos.x >= bmin.x && hit_pos.x <= bmax.x &&
                hit_pos.y >= bmin.y && hit_pos.y <= bmax.y &&
                hit_pos.z >= bmin.z && hit_pos.z <= bmax.z) {
                picked = i32(i);
                break;
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

            let bmin = ad.card_aabb_min.xyz;
            let bmax = ad.card_aabb_max.xyz;
            // Hit is in world space and Sponza meshes are in world
            // space too (no per-instance transform beyond identity on
            // the Sponza asset). For now use world-space hit directly;
            // instances with a non-identity transform would need the
            // transform stored on `instance_data` to round-trip.
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

    let intensity = u.params.y;
    textureStore(radiance_out, dst_coord, vec4<f32>(radiance * intensity * ndotd, 1.0));
}
";

/// Probe temporal accumulator. EMA in probe-octel space. No reprojection
/// in V1 — since every frame traces all 64 octels, history only smooths
/// firefly noise; per-probe world positions jitter by tile-fraction so
/// camera motion eventually converges to a stable signal without explicit
/// velocity. Disocclusion is handled implicitly: moving the camera
/// changes which tile a surface falls into, replacing that probe's
/// header, and the new probe's history blends from zero since the old
/// probe at that grid coord pointed elsewhere.
pub(in crate::renderer) const SSGI_PROBE_TEMPORAL_WGSL: &str = "
struct TemporalParams {
    // x = alpha (0.25 = 4-frame EMA at steady state),
    // y = force_refresh (1 → alpha 1.0),
    // z = grid_w, w = grid_h
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TemporalParams;
@group(0) @binding(1) var radiance_in: texture_3d<f32>;
@group(0) @binding(2) var history_in: texture_3d<f32>;
@group(0) @binding(3) var history_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u32(u.params.z);
    let grid_h = u32(u.params.w);
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }

    let coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));
    let curr = textureLoad(radiance_in, coord, 0).rgb;
    let hist = textureLoad(history_in, coord, 0).rgb;

    var alpha = u.params.x;
    if (u.params.y > 0.5) {
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
    let blended = mix(hist, curr, alpha);

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

// Sample a probe's octahedral atlas in a given world-space direction.
// Uses trilinear sampling on the 3D texture so neighbouring octels
// softly blend — some visible smear on the 8×8 atlas, at the cost of
// a cheap reconstruction.
fn sample_probe(probe_xy: vec2<i32>, dir_ws: vec3<f32>) -> vec3<f32> {
    let oct_uv = oct_encode(dir_ws);
    let u_tex = (f32(probe_xy.x) + 0.5) / f32(u.size.z);
    let v_tex = (f32(probe_xy.y) + 0.5) / f32(u.size.w);
    // z coordinate: octahedral texel index in [0, 64) normalized to [0, 1)
    let oct_x = clamp(oct_uv.x * f32(PROBE_OCT_SIZE), 0.0, f32(PROBE_OCT_SIZE) - 1.0);
    let oct_y = clamp(oct_uv.y * f32(PROBE_OCT_SIZE), 0.0, f32(PROBE_OCT_SIZE) - 1.0);
    let z_idx = floor(oct_y) * f32(PROBE_OCT_SIZE) + floor(oct_x);
    let z = (z_idx + 0.5) / f32(PROBE_OCT_TEXELS);
    return textureSampleLevel(radiance_tex, radiance_samp, vec3<f32>(u_tex, v_tex, z), 0.0).rgb;
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
    let N_vs = normalize(cross(Pr - P_vs, Pu - P_vs));
    let N_ws = normalize((u.inv_view * vec4<f32>(N_vs, 0.0)).xyz);

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

            let radiance = sample_probe(vec2<i32>(gx, gy), N_ws);
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

/// SSR temporal denoiser. Same shape as the SSGI temporal pass:
/// reprojects the previous history through the motion vectors,
/// clamps against the 3×3 neighborhood of the noisy current frame,
/// and blends with a low alpha so 4–8 frames of random GGX rays
/// converge to a smooth reflection. Also pre-filters the noisy
/// current frame by the 3×3 mean, which kills single-pixel
/// glossy-ray sparkles in one frame instead of 10.
pub(in crate::renderer) const SSR_TEMPORAL_SHADER_WGSL: &str = "
struct SsrTemporalParams {
    /// x = blend_alpha (0.1), yzw unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SsrTemporalParams;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var current_samp: sampler;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var history_samp: sampler;
@group(0) @binding(5) var velocity_tex: texture_2d<f32>;
@group(0) @binding(6) var velocity_samp: sampler;

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
    let current_raw = textureSample(current_tex, current_samp, in.uv);

    // 3×3 box pre-filter + neighborhood min/max. One texel spread
    // across 9 samples hides single-pixel glossy-ray sparkles in a
    // single frame; the min/max bounds the history so disocclusion
    // and material transitions clamp rather than ghost.
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(current_tex));
    var nmin = current_raw;
    var nmax = current_raw;
    var prefilt = vec4<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let s = textureSample(current_tex, current_samp, in.uv + vec2<f32>(f32(x), f32(y)) * texel);
            nmin = min(nmin, s);
            nmax = max(nmax, s);
            prefilt = prefilt + s;
        }
    }
    let current = prefilt * (1.0 / 9.0);

    // Velocity is full-res; UV mapping handles the half-res delta.
    // NDC-space velocity + UV Y-flip → `uv + vel.y` for the Y axis,
    // matching TAA + SSAO + the sibling SSGI temporal pass.
    let vel = textureSample(velocity_tex, velocity_samp, in.uv).xy;
    let prev_uv = vec2<f32>(in.uv.x - vel.x, in.uv.y + vel.y);
    let off_screen = prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0;
    if (off_screen) { return current; }

    let history_raw = textureSample(history_tex, history_samp, prev_uv);
    // Scrub NaN/Inf from the history read. Until a clean SSR frame
    // finishes draining the ping-pong pair, any poisoned history
    // pixel would otherwise survive the clamp (clamp(NaN, a, b) is
    // implementation-defined on Metal — frequently NaN) and keep
    // tonemapping to pink. Replace poisoned channels with the
    // current-frame mean, which is the best available estimate.
    let history = select(current, history_raw, history_raw == history_raw);
    let clamped_history = clamp(history, nmin, nmax);
    let alpha = u.params.x;
    let blended = mix(clamped_history, current, alpha);
    return select(current, blended, blended == blended);
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
    let depth = textureSample(depth_tex, depth_samp, in.uv);
    if (depth >= 0.9999) { return vec4<f32>(0.0); } // sky

    // SSR for every smooth-enough surface — the metals-only gate is gone.
    // Rationale: the scene shader deliberately starves polished DIELECTRICS
    // of IBL specular (dielectric_spec_amp rolls to zero on smooth surfaces
    // because the visibility-less prefiltered env produced bright stripes),
    // so screen-space hits are the only grounded reflections a wet floor or
    // polished stone can receive — no double-counting by construction. The
    // F0 = 0.04 Fresnel below keeps dielectric contribution physically
    // small except at grazing angles. Very rough surfaces still fade out
    // to IBL where one-ray-per-pixel SSR noise would dominate even after
    // temporal accumulation.
    let mat = textureSample(mat_tex, mat_samp, in.uv).rg;
    let metallic = mat.r;
    let roughness = mat.g;
    let albedo = textureSample(albedo_tex, albedo_samp, in.uv).rgb;
    let roughness_fade = 1.0 - smoothstep(0.5, 0.85, roughness);
    if (roughness_fade <= 0.001) { return vec4<f32>(0.0); }

    let view_pos = view_pos_from_depth(in.uv, depth);
    let dx = dpdx(view_pos);
    let dy = dpdy(view_pos);
    let n = normalize(cross(dx, dy));
    let v = normalize(-view_pos);

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

    if (r.z > 0.0) { return vec4<f32>(0.0); }

    let n_dot_v = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - n_dot_v, 5.0);

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
    if (!hit_found) { return vec4<f32>(0.0); }

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
    let raw = textureSample(hdr_tex, hdr_samp, hit_uv).rgb;
    let reflected = select(vec3<f32>(0.0), raw, raw == raw);
    let out = reflected * fresnel * roughness_fade * u.params.x * fade;
    let out_safe = select(vec3<f32>(0.0), out, out == out);
    return vec4<f32>(out_safe, fade);
}
";


