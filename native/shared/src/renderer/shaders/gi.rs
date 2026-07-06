//! Surface-cache GI: mesh cards, SDF bakes, WSRC.
//! Split from renderer/shaders.rs.

/// Ticket 013 V3 — Mesh-Cards capture shader with dual render targets.
///
/// Location 0 (albedo) + Location 1 (emissive). Rasterises a mesh
/// orthographically along its assigned signed axis into the card slot,
/// writing baked albedo and emissive into their respective atlases in
/// one draw. Emissive reads the material's emissive texture (if any)
/// and multiplies by the `emissive_factor`. Flat `base_color_factor` +
/// a 1×1 white fallback texture cover the case where the mesh has
/// only a scalar material.
pub(in crate::renderer) const CARD_CAPTURE_WGSL: &str = "
struct CaptureParams {
    ortho_vp: mat4x4<f32>,
    // Mesh's base_color_factor (xyz) + `has_base_texture` flag (w).
    base_color: vec4<f32>,
    // Mesh's emissive_factor (xyz) + `has_emissive_texture` flag (w).
    emissive: vec4<f32>,
};

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FsOut {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: CaptureParams;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_samp: sampler;
@group(1) @binding(2) var emissive_tex: texture_2d<f32>;

@vertex
fn vs_main(v: VertexIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.ortho_vp * vec4<f32>(v.position, 1.0);
    out.uv = v.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var albedo = u.base_color.rgb;
    if (u.base_color.w > 0.5) {
        albedo = albedo * textureSample(albedo_tex, albedo_samp, in.uv).rgb;
    }

    var emissive = u.emissive.rgb;
    if (u.emissive.w > 0.5) {
        emissive = emissive * textureSample(emissive_tex, albedo_samp, in.uv).rgb;
    }

    var out: FsOut;
    out.albedo = vec4<f32>(albedo, 1.0);
    // Rgba8UnormSrgb clamps emissive at 1.0 per channel; that's fine
    // for Sponza-scale lanterns. Pre-divide by EMISSIVE_SCALE at write
    // and multiply back at sample if HDR emissive becomes necessary.
    out.emissive = vec4<f32>(clamp(emissive, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    return out;
}
";

/// Ticket 014 V1 — per-mesh unsigned distance field bake.
///
/// One compute invocation per voxel (32×32×32 = 32768 per mesh). Each
/// lane reads the mesh's vertex + index buffers as storage, iterates
/// all triangles, and computes `min(point-triangle distance)` from the
/// voxel centre. Output is R16Float distance (unsigned — we don't
/// attempt inside/outside classification, which is brittle on open
/// Sponza meshes; sphere-trace works with UDF either way because each
/// march step advances by `d` regardless of sign).
///
/// Vertex stride matches `Vertex3D` (12 f32 = 48 bytes). Only the
/// first 3 floats (position) are read.
pub(in crate::renderer) const SDF_BAKE_WGSL: &str = "
struct SdfBakeParams {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    // x = triangle_count, y = sdf_resolution, zw unused
    counts: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: SdfBakeParams;
@group(0) @binding(1) var<storage, read> vertex_buf: array<f32>;
@group(0) @binding(2) var<storage, read> index_buf: array<u32>;
@group(0) @binding(3) var sdf_out: texture_storage_3d<r32float, write>;

const VERTEX_STRIDE_F32: u32 = 12u;  // Vertex3D: pos(3) + normal(3) + color(4) + uv(2) = 12 f32

fn vtx_pos(idx: u32) -> vec3<f32> {
    let base = idx * VERTEX_STRIDE_F32;
    return vec3<f32>(vertex_buf[base], vertex_buf[base + 1u], vertex_buf[base + 2u]);
}

// Point-triangle distance, clamped-edge form. From Ericson, Real-Time
// Collision Detection. Returns unsigned distance to the closest point
// on triangle abc from point p.
fn point_triangle_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return length(ap); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return length(bp); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return length(p - (a + v * ab));
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return length(cp); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return length(p - (a + w * ac));
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return length(p - (b + w * (c - b)));
    }
    // Closest point in face interior.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return length(p - (a + ab * v + ac * w));
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = u.counts.y;
    if (gid.x >= res || gid.y >= res || gid.z >= res) { return; }

    // Voxel centre in object space.
    let uvw = (vec3<f32>(gid) + vec3<f32>(0.5)) / f32(res);
    let voxel_os = mix(u.aabb_min.xyz, u.aabb_max.xyz, uvw);

    var min_dist = 1e9;
    let tri_count = u.counts.x;
    for (var t: u32 = 0u; t < tri_count; t = t + 1u) {
        let i0 = index_buf[t * 3u + 0u];
        let i1 = index_buf[t * 3u + 1u];
        let i2 = index_buf[t * 3u + 2u];
        let a = vtx_pos(i0);
        let b = vtx_pos(i1);
        let c = vtx_pos(i2);
        let d = point_triangle_distance(voxel_os, a, b, c);
        min_dist = min(min_dist, d);
    }

    textureStore(sdf_out, vec3<i32>(gid), vec4<f32>(min_dist, 0.0, 0.0, 0.0));
}
";

/// Fullscreen-lag fix — scene-clipmap variant of the SDF bake.
///
/// The original clipmap bake reused `SDF_BAKE_WGSL`: every one of the
/// 64³ voxels looped over EVERY scene triangle, and the whole volume
/// baked in a single dispatch. On integrated GPUs that one dispatch
/// stalled the end-of-frame submit for seconds whenever the camera
/// drifted 10 m. This variant fixes both axes of that cost:
///
/// - Triangles are binned on the CPU into `counts.x` cells per axis,
///   each cell's list pre-expanded by one cell in every direction. A
///   voxel only tests its own cell's list. Distances are clamped at
///   one cell width (`band`); because of the one-cell expansion the
///   clamp never OVERestimates the distance to the nearest surface,
///   which keeps sphere-trace steps conservative. Empty-air cells skip
///   the triangle loop entirely.
/// - `counts.z` carries a voxel Z offset so the caller can bake a few
///   Z-layers per frame into a staging texture instead of the whole
///   volume at once.
pub(in crate::renderer) const SDF_CLIPMAP_BAKE_WGSL: &str = "
struct SdfClipmapBakeParams {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    // x = bin cells per axis, y = sdf resolution,
    // z = voxel Z offset of this slice batch, w unused
    counts: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: SdfClipmapBakeParams;
@group(0) @binding(1) var<storage, read> vertex_buf: array<f32>;
@group(0) @binding(2) var<storage, read> index_buf: array<u32>;
@group(0) @binding(3) var sdf_out: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> cell_tris: array<u32>;

const VERTEX_STRIDE_F32: u32 = 12u;  // Vertex3D: pos(3) + normal(3) + color(4) + uv(2)

fn vtx_pos(idx: u32) -> vec3<f32> {
    let base = idx * VERTEX_STRIDE_F32;
    return vec3<f32>(vertex_buf[base], vertex_buf[base + 1u], vertex_buf[base + 2u]);
}

// Point-triangle distance, clamped-edge form (Ericson).
fn point_triangle_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return length(ap); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return length(bp); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return length(p - (a + v * ab));
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return length(cp); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return length(p - (a + w * ac));
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return length(p - (b + w * (c - b)));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return length(p - (a + ab * v + ac * w));
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = u.counts.y;
    let vox = vec3<u32>(gid.x, gid.y, gid.z + u.counts.z);
    if (vox.x >= res || vox.y >= res || vox.z >= res) { return; }

    let uvw = (vec3<f32>(vox) + vec3<f32>(0.5)) / f32(res);
    let voxel_ws = mix(u.aabb_min.xyz, u.aabb_max.xyz, uvw);

    let cells = u.counts.x;
    let vpc = res / cells;
    let cell = vox / vpc;
    let ci = (cell.z * cells + cell.y) * cells + cell.x;
    let start = cell_offsets[ci];
    let end = cell_offsets[ci + 1u];

    let band = (u.aabb_max.x - u.aabb_min.x) / f32(cells);
    var min_dist = band;
    for (var k: u32 = start; k < end; k = k + 1u) {
        let t = cell_tris[k];
        let a = vtx_pos(index_buf[t * 3u + 0u]);
        let b = vtx_pos(index_buf[t * 3u + 1u]);
        let c = vtx_pos(index_buf[t * 3u + 2u]);
        min_dist = min(min_dist, point_triangle_distance(voxel_ws, a, b, c));
    }

    textureStore(sdf_out, vec3<i32>(vox), vec4<f32>(min_dist, 0.0, 0.0, 0.0));
}
";

/// Ticket 013 V3 — per-frame card-lighting compute pass with
/// shadow-cascade sampling + emissive contribution.
///
/// Per texel: reconstruct the world-space position of the card point
/// (slot metadata's aabb + axis + mesh transform), sample the shadow
/// cascade at that point, compute direct + sky + emissive. Writes to
/// the radiance atlas. Shadow lookup makes indirect bounce meaningfully
/// darker on sun-occluded card faces (arches, column undersides).
pub(in crate::renderer) const CARD_LIGHT_WGSL: &str = "
struct SlotMeta {
    // xyz = world-space card-face normal, w = signed axis (0..6 as f32)
    normal_ws: vec4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    transform: mat4x4<f32>,
};

struct CardLightParams {
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    atlas_info: vec4<u32>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    view_matrix: mat4x4<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: CardLightParams;
@group(0) @binding(1) var albedo_atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;
@group(0) @binding(3) var<storage, read> slot_meta: array<SlotMeta>;
@group(0) @binding(4) var radiance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var emissive_atlas: texture_2d<f32>;
@group(0) @binding(6) var shadow_atlas_0: texture_depth_2d;
@group(0) @binding(7) var shadow_atlas_1: texture_depth_2d;
@group(0) @binding(8) var shadow_atlas_2: texture_depth_2d;
@group(0) @binding(9) var shadow_samp: sampler_comparison;

fn sample_cascade(cascade: i32, pos_ws: vec3<f32>, bias: f32) -> f32 {
    var clip: vec4<f32>;
    if (cascade == 0) {
        clip = u.shadow_vps[0] * vec4<f32>(pos_ws, 1.0);
    } else if (cascade == 1) {
        clip = u.shadow_vps[1] * vec4<f32>(pos_ws, 1.0);
    } else {
        clip = u.shadow_vps[2] * vec4<f32>(pos_ws, 1.0);
    }
    let ndc = clip.xyz / clip.w;
    // Outside the cascade frustum → treat as lit (no shadow).
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let ref_depth = ndc.z - bias;
    if (cascade == 0) {
        return textureSampleCompareLevel(shadow_atlas_0, shadow_samp, shadow_uv, ref_depth);
    } else if (cascade == 1) {
        return textureSampleCompareLevel(shadow_atlas_1, shadow_samp, shadow_uv, ref_depth);
    } else {
        return textureSampleCompareLevel(shadow_atlas_2, shadow_samp, shadow_uv, ref_depth);
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.xy;
    let atlas_sz = u.atlas_info.x;
    if (px.x >= atlas_sz || px.y >= atlas_sz) { return; }

    let slot_sz = u.atlas_info.y;
    let slots_per_row = u.atlas_info.z;
    let active_count = u.atlas_info.w;

    let slot_x = px.x / slot_sz;
    let slot_y = px.y / slot_sz;
    let slot_idx = slot_y * slots_per_row + slot_x;

    if (slot_idx >= active_count) {
        textureStore(radiance_out, vec2<i32>(px), vec4<f32>(0.0));
        return;
    }

    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / f32(atlas_sz);
    let albedo = textureSampleLevel(albedo_atlas, atlas_samp, uv, 0.0).rgb;
    let emissive = textureSampleLevel(emissive_atlas, atlas_samp, uv, 0.0).rgb;

    let slot_m = slot_meta[slot_idx];
    let n_ws = slot_m.normal_ws.xyz;
    let axis = u32(slot_m.normal_ws.w);

    // Local slot UV (0..1 inside this slot) → object-space card-plane
    // position. Signed-axis-aware u-flip matches the capture pass's
    // projection so the texel we light is the one the capture baked.
    let sx = f32((px.x) % slot_sz);
    let sy = f32((px.y) % slot_sz);
    let sd = f32(slot_sz);
    var u_norm = (sx + 0.5) / sd;
    let v_norm = (sy + 0.5) / sd;
    var pos_os = vec3<f32>(0.0);
    if (axis == 0u || axis == 1u) {
        // Card plane at x = bmax.x (+X) or bmin.x (-X); u=y, v=z.
        if (axis == 1u) { u_norm = 1.0 - u_norm; }
        pos_os.y = mix(slot_m.aabb_min.y, slot_m.aabb_max.y, u_norm);
        pos_os.z = mix(slot_m.aabb_min.z, slot_m.aabb_max.z, v_norm);
        if (axis == 0u) { pos_os.x = slot_m.aabb_max.x; } else { pos_os.x = slot_m.aabb_min.x; }
    } else if (axis == 2u || axis == 3u) {
        if (axis == 3u) { u_norm = 1.0 - u_norm; }
        pos_os.x = mix(slot_m.aabb_min.x, slot_m.aabb_max.x, u_norm);
        pos_os.z = mix(slot_m.aabb_min.z, slot_m.aabb_max.z, v_norm);
        if (axis == 2u) { pos_os.y = slot_m.aabb_max.y; } else { pos_os.y = slot_m.aabb_min.y; }
    } else {
        if (axis == 5u) { u_norm = 1.0 - u_norm; }
        pos_os.x = mix(slot_m.aabb_min.x, slot_m.aabb_max.x, u_norm);
        pos_os.y = mix(slot_m.aabb_min.y, slot_m.aabb_max.y, v_norm);
        if (axis == 4u) { pos_os.z = slot_m.aabb_max.z; } else { pos_os.z = slot_m.aabb_min.z; }
    }
    let pos_ws = (slot_m.transform * vec4<f32>(pos_os, 1.0)).xyz;

    // Shadow. Cascade selection uses view-space Z against splits.
    var shadow: f32 = 1.0;
    if (u.flags.y > 0.5) {
        let view_z = -(u.view_matrix * vec4<f32>(pos_ws, 1.0)).z;
        var cascade: i32 = 2;
        if (view_z <= u.shadow_splits.x) {
            cascade = 0;
        } else if (view_z <= u.shadow_splits.y) {
            cascade = 1;
        }
        shadow = sample_cascade(cascade, pos_ws, u.flags.x);
    }

    let ndotl = max(dot(n_ws, u.sun_dir.xyz), 0.0);
    let direct = u.sun_color.xyz * ndotl * shadow;
    let ndotup = max(dot(n_ws, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    let sky = u.sky_color.xyz * ndotup;

    let lit = albedo * (direct + sky) + emissive;
    textureStore(radiance_out, vec2<i32>(px), vec4<f32>(lit, 1.0));
}
";

/// Ticket 014 V6 — World-Space Radiance Cache bake.
///
/// One workgroup per probe (16×16×16 probes on a 120 m camera-following
/// cube), 8×8 threads per workgroup = one octel texel each. Output is
/// the flat rgba16float atlas read by the SDF miss path.
///
/// Per texel: analytic sun term for the octel direction, gated by a
/// shadow-cascade lookup at the probe's world position (position-
/// varying occlusion — the crucial signal that makes the cache worth
/// having; flat analytic would mean every probe holds the same
/// content). Analytic sky hemisphere for up-biased directions. No
/// per-direction geometry trace — this is the "distant envelope",
/// not another SDF march.
pub(in crate::renderer) const WSRC_BAKE_WGSL: &str = "
struct WsrcBakeParams {
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    // xyz = cascade origin (world-space cube centre), w = full extent
    grid: vec4<f32>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    // x = shadow bias, y = shadows-enabled flag, z = cascade index
    // (0..WSRC_CASCADE_COUNT), w unused. Cascade index offsets the
    // output z-slice so this pipeline can be dispatched once per
    // cascade with the same layout.
    flags: vec4<f32>,
    // EN-023 — xyz = scene-average albedo for the ground-bounce term.
    ground_albedo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: WsrcBakeParams;
@group(0) @binding(1) var shadow_atlas_0: texture_depth_2d;
@group(0) @binding(2) var shadow_atlas_1: texture_depth_2d;
@group(0) @binding(3) var shadow_atlas_2: texture_depth_2d;
@group(0) @binding(4) var shadow_samp: sampler_comparison;
@group(0) @binding(5) var wsrc_out: texture_storage_3d<rgba16float, write>;

fn wsrc_sample_cascade(cascade: i32, pos_ws: vec3<f32>, bias: f32) -> f32 {
    var clip: vec4<f32>;
    if (cascade == 0) {
        clip = u.shadow_vps[0] * vec4<f32>(pos_ws, 1.0);
    } else if (cascade == 1) {
        clip = u.shadow_vps[1] * vec4<f32>(pos_ws, 1.0);
    } else {
        clip = u.shadow_vps[2] * vec4<f32>(pos_ws, 1.0);
    }
    let ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let ref_depth = ndc.z - bias;
    if (cascade == 0) {
        return textureSampleCompareLevel(shadow_atlas_0, shadow_samp, shadow_uv, ref_depth);
    } else if (cascade == 1) {
        return textureSampleCompareLevel(shadow_atlas_1, shadow_samp, shadow_uv, ref_depth);
    } else {
        return textureSampleCompareLevel(shadow_atlas_2, shadow_samp, shadow_uv, ref_depth);
    }
}

// V10 — workgroup writes the 10×10 padded slab per probe. Thread
// (lid.x, lid.y) writes texel (wg.*10 + lid) in the atlas; border
// threads (lid on 0 or 9) shade for the nearest INSIDE octel
// direction so the sampler's edge-extend behaviour is baked into
// the data. The 1-texel border is what lets the hardware bilinear
// sampler do octel smoothing without leaking into adjacent probes.
const WSRC_OCT_PADDED_SIZE: u32 = 10u;

@compute @workgroup_size(10, 10, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_res: u32 = 16u;
    if (wg.x >= grid_res || wg.y >= grid_res || wg.z >= grid_res) { return; }
    if (lid.x >= WSRC_OCT_PADDED_SIZE || lid.y >= WSRC_OCT_PADDED_SIZE) { return; }

    // Probe world-space centre — cell-centre within the grid cube.
    let extent = u.grid.w;
    let cell = extent / f32(grid_res);
    let probe_pos = u.grid.xyz
        - vec3<f32>(extent * 0.5)
        + (vec3<f32>(f32(wg.x), f32(wg.y), f32(wg.z)) + vec3<f32>(0.5)) * cell;

    // V11 — map padded octel → real octel with true octahedral
    // silhouette wrap on the 4 edges. Beyond v<0 or v>1 in octel uv
    // space the octahedron folds onto itself with u ↔ 1-u; likewise
    // u<0 or u>1 folds with v ↔ 1-v. Corners (both axes out) keep
    // the V10 edge-extend fill since the double-fold has two valid
    // representations and the exact corner only matters when the
    // sampler bilinear-weights it near zero anyway.
    let px = i32(lid.x);
    let py = i32(lid.y);
    let is_edge_x = px == 0 || px == 9;
    let is_edge_y = py == 0 || py == 9;
    var real_ox: i32;
    var real_oy: i32;
    if (is_edge_x && is_edge_y) {
        // Corner — nearest-inside (edge-extend).
        real_ox = clamp(px - 1, 0, 7);
        real_oy = clamp(py - 1, 0, 7);
    } else if (is_edge_y) {
        // Top/bottom border: mirror x across the edge, same row.
        real_ox = 8 - px;
        real_oy = clamp(py - 1, 0, 7);
    } else if (is_edge_x) {
        // Left/right border: same column, mirror y across the edge.
        real_ox = clamp(px - 1, 0, 7);
        real_oy = 8 - py;
    } else {
        // Interior — direct mapping.
        real_ox = px - 1;
        real_oy = py - 1;
    }
    let dir = octel_direction(vec2<u32>(u32(real_ox), u32(real_oy)));

    // Shadow at the probe position (cascade 2 — widest, covers the
    // full 120 m cube without per-probe cascade selection).
    var shadow: f32 = 1.0;
    if (u.flags.y > 0.5) {
        shadow = wsrc_sample_cascade(2, probe_pos, u.flags.x);
    }

    let ndotl = max(dot(dir, u.sun_dir.xyz), 0.0);
    let sun = u.sun_color.xyz * ndotl * shadow;
    let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sky = u.sky_color.xyz * up * up;

    // EN-023 — ground bounce for below-horizon octels. This envelope
    // was light-source-only: any miss ray pointing downward returned
    // ~black, so shaded receivers lost the strongest real bounce
    // source (sunlit ground). Approximate it as the scene-average
    // albedo lit by the sun (shadowed at the probe) + half the sky.
    let down = clamp(-dir.y, 0.0, 1.0);
    let ground_irr = u.sun_color.xyz * max(u.sun_dir.y, 0.0) * shadow
                   + u.sky_color.xyz * 0.5;
    let ground = u.ground_albedo.xyz * ground_irr * down * down;

    let radiance = sun + sky + ground;

    // V13 — cascade index in flags.z offsets the output z-slice.
    let cascade_idx = u32(u.flags.z);
    let tex_coord = vec3<i32>(
        i32(wg.x * WSRC_OCT_PADDED_SIZE + lid.x),
        i32(wg.y * WSRC_OCT_PADDED_SIZE + lid.y),
        i32(cascade_idx * grid_res + wg.z),
    );
    textureStore(wsrc_out, tex_coord, vec4<f32>(radiance, 1.0));
}
";

/// Ticket 014 V14 — HW-ray-traced WSRC bake for RT-capable adapters.
///
/// Same probe grid + padded octel layout as the SW bake; the
/// difference is the radiance computation. For each probe-octel
/// texel:
///   1. Fire a ray from `probe_pos` in `dir` (octel direction) with
///      `t_max = extent * 0.5` — long enough for a probe to reach
///      the cascade boundary, short enough that each cascade stays
///      in its spatial regime.
///   2. On hit: sample the Mesh Cards radiance atlas at the hit
///      point (already pre-lit each frame with sun × shadow +
///      emissive by `card_light_pass`, so the bake propagates
///      one-bounce shaded radiance into WSRC — effectively 2-bounce
///      when the SSGI probe then samples WSRC on miss).
///   3. On miss: analytic sun (shadow-sampled at the probe, not
///      per-direction) + hemisphere sky — same fallback as the V13
///      SW path, so escaped rays still carry a useful envelope.
///
/// Border texels still use the V11 octahedral wrap — same rule as
/// the SW bake. The cascade index comes from `flags.z` and offsets
/// the output z slice so one pipeline covers all 3 cascades via
/// per-dispatch uniform.
pub(in crate::renderer) const WSRC_BAKE_HW_WGSL: &str = "
struct WsrcBakeParams {
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    grid: vec4<f32>,
    shadow_vps: array<mat4x4<f32>, 3>,
    shadow_splits: vec4<f32>,
    flags: vec4<f32>,
    // EN-023 — layout mirror; the HW bake traces real geometry and
    // ignores the scene-average ground albedo.
    ground_albedo: vec4<f32>,
};

struct HwBakeInstanceGiData {
    albedo: vec3<f32>,
    emissive_luma: f32,
    normal_ws: vec3<f32>,
    _pad0: f32,
    card_slot: vec4<f32>,
    card_aabb_min: vec4<f32>,
    card_aabb_max: vec4<f32>,
    // EN-023 — world-space AABB (SDF path only; layout mirror).
    world_aabb_min: vec4<f32>,
    world_aabb_max: vec4<f32>,
};

const HW_BAKE_CARD_SLOTS_PER_ROW: f32 = 64.0;
const HW_BAKE_OCT_PADDED: u32 = 10u;

@group(0) @binding(0) var<uniform> u: WsrcBakeParams;
@group(0) @binding(1) var shadow_atlas_0: texture_depth_2d;
@group(0) @binding(2) var shadow_atlas_1: texture_depth_2d;
@group(0) @binding(3) var shadow_atlas_2: texture_depth_2d;
@group(0) @binding(4) var shadow_samp: sampler_comparison;
@group(0) @binding(5) var wsrc_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var accel: acceleration_structure;
@group(0) @binding(7) var<storage, read> instance_data: array<HwBakeInstanceGiData>;
@group(0) @binding(8) var card_atlas: texture_2d<f32>;
@group(0) @binding(9) var card_samp: sampler;

fn hw_bake_sample_cascade(cascade: i32, pos_ws: vec3<f32>, bias: f32) -> f32 {
    var clip: vec4<f32>;
    if (cascade == 0) { clip = u.shadow_vps[0] * vec4<f32>(pos_ws, 1.0); }
    else if (cascade == 1) { clip = u.shadow_vps[1] * vec4<f32>(pos_ws, 1.0); }
    else { clip = u.shadow_vps[2] * vec4<f32>(pos_ws, 1.0); }
    let ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let ref_depth = ndc.z - bias;
    if (cascade == 0) { return textureSampleCompareLevel(shadow_atlas_0, shadow_samp, shadow_uv, ref_depth); }
    else if (cascade == 1) { return textureSampleCompareLevel(shadow_atlas_1, shadow_samp, shadow_uv, ref_depth); }
    else { return textureSampleCompareLevel(shadow_atlas_2, shadow_samp, shadow_uv, ref_depth); }
}

@compute @workgroup_size(10, 10, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_res: u32 = 16u;
    if (wg.x >= grid_res || wg.y >= grid_res || wg.z >= grid_res) { return; }
    if (lid.x >= HW_BAKE_OCT_PADDED || lid.y >= HW_BAKE_OCT_PADDED) { return; }

    let extent = u.grid.w;
    let cell = extent / f32(grid_res);
    let probe_pos = u.grid.xyz
        - vec3<f32>(extent * 0.5)
        + (vec3<f32>(f32(wg.x), f32(wg.y), f32(wg.z)) + vec3<f32>(0.5)) * cell;

    // V11 octahedral wrap for the padded borders.
    let px = i32(lid.x);
    let py = i32(lid.y);
    let is_edge_x = px == 0 || px == 9;
    let is_edge_y = py == 0 || py == 9;
    var real_ox: i32;
    var real_oy: i32;
    if (is_edge_x && is_edge_y) {
        real_ox = clamp(px - 1, 0, 7);
        real_oy = clamp(py - 1, 0, 7);
    } else if (is_edge_y) {
        real_ox = 8 - px;
        real_oy = clamp(py - 1, 0, 7);
    } else if (is_edge_x) {
        real_ox = clamp(px - 1, 0, 7);
        real_oy = 8 - py;
    } else {
        real_ox = px - 1;
        real_oy = py - 1;
    }
    let dir = octel_direction(vec2<u32>(u32(real_ox), u32(real_oy)));

    // V14 — fire a short ray from the probe centre. Ray length
    // scales with the cascade extent so each cascade's rays stay
    // in its resolution regime (near: ~15 m, mid: ~60 m, far:
    // ~250 m).
    let ray_length = extent * 0.5;
    var rq: ray_query;
    rayQueryInitialize(&rq, accel, RayDesc(
        0u,
        0xFFu,
        0.01,
        ray_length,
        probe_pos,
        dir,
    ));
    loop {
        if (!rayQueryProceed(&rq)) { break; }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);

    var radiance = vec3<f32>(0.0);
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        let inst = instance_data[hit.instance_custom_data];
        if (inst.card_slot.w > 0.5) {
            // Sample Mesh Cards pre-lit radiance at hit. Same
            // projection math as SSGI_PROBE_TRACE_HW_WGSL's hit
            // branch.
            let hit_world = probe_pos + dir * hit.t;
            let hit_os = (hit.world_to_object * vec4<f32>(hit_world, 1.0)).xyz;
            let abs_d = abs(dir);
            var axis_idx: u32 = 0u;
            if (abs_d.y >= abs_d.x && abs_d.y >= abs_d.z) {
                axis_idx = 2u;
            } else if (abs_d.z >= abs_d.x) {
                axis_idx = 4u;
            }
            var signed_axis: u32 = axis_idx;
            if (axis_idx == 0u && dir.x > 0.0) { signed_axis = 1u; }
            else if (axis_idx == 2u && dir.y > 0.0) { signed_axis = 3u; }
            else if (axis_idx == 4u && dir.z > 0.0) { signed_axis = 5u; }

            let first_slot = u32(inst.card_slot.x);
            let slot = first_slot + signed_axis;
            let slot_x = slot % 64u;
            let slot_y = slot / 64u;

            let bmin = inst.card_aabb_min.xyz;
            let bmax = inst.card_aabb_max.xyz;
            var u_os: f32;
            var v_os: f32;
            var u_lo: f32; var u_hi: f32;
            var v_lo: f32; var v_hi: f32;
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

            let slot_size_uv = 1.0 / HW_BAKE_CARD_SLOTS_PER_ROW;
            let texel_in_slot = slot_size_uv / f32(64);
            let slot_u0 = f32(slot_x) * slot_size_uv + texel_in_slot;
            let slot_v0 = f32(slot_y) * slot_size_uv + texel_in_slot;
            let slot_span = slot_size_uv - 2.0 * texel_in_slot;
            let atlas_uv = vec2<f32>(
                slot_u0 + u_norm * slot_span,
                slot_v0 + v_norm * slot_span,
            );
            radiance = textureSampleLevel(card_atlas, card_samp, atlas_uv, 0.0).rgb;
        } else {
            // Instance without a card — shade analytically using
            // its flat normal + albedo.
            let hit_n = inst.normal_ws;
            let ndotl = max(dot(hit_n, u.sun_dir.xyz), 0.0);
            let direct = u.sun_color.xyz * ndotl;
            let ndotup = max(dot(hit_n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
            let sky = u.sky_color.xyz * ndotup;
            radiance = inst.albedo * (direct + sky)
                     + inst.albedo * inst.emissive_luma;
        }
    } else {
        // Miss — V13 analytic fallback (shadow-sampled sun +
        // hemisphere sky). The ray went past `extent * 0.5` without
        // hitting anything, so this is the open-sky direction at
        // probe scale.
        var shadow: f32 = 1.0;
        if (u.flags.y > 0.5) {
            shadow = hw_bake_sample_cascade(2, probe_pos, u.flags.x);
        }
        let ndotl = max(dot(dir, u.sun_dir.xyz), 0.0);
        let sun = u.sun_color.xyz * ndotl * shadow;
        let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
        let sky = u.sky_color.xyz * up * up;
        radiance = sun + sky;
    }

    let cascade_idx = u32(u.flags.z);
    let tex_coord = vec3<i32>(
        i32(wg.x * HW_BAKE_OCT_PADDED + lid.x),
        i32(wg.y * HW_BAKE_OCT_PADDED + lid.y),
        i32(cascade_idx * grid_res + wg.z),
    );
    textureStore(wsrc_out, tex_coord, vec4<f32>(radiance, 1.0));
}
";
