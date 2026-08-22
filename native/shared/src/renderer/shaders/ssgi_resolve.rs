//! Per-pixel screen-probe GI reconstruction.

/// Writes the half-resolution `ssgi_rt` consumed by downstream composition.
///
/// Samples the 2×2 probes whose tiles enclose the pixel's tile and
/// bilateral-weights the contribution by a symmetric world-space same-plane
/// test plus normal match. Invalid probes (sky) are skipped. When all four
/// probes reject, fall back to zero rather than leak a different surface.
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

// The tiny probe-space pass writes a geometry-aware reconstruction into layer
// zero after temporal completes. The other layers retain current samples for
// capture-only diagnostics.
fn sample_probe(probe_coord: vec2<i32>) -> vec3<f32> {
    return textureLoad(radiance_tex, vec3<i32>(probe_coord, 0), 0).rgb;
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
    let P_ws = (u.inv_view * vec4<f32>(P_vs, 1.0)).xyz;

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

            // A view-depth difference rejects valid samples on an oblique
            // wall and changes as the camera moves. Compare the receiver and
            // probe in world space instead, accepting only the same plane.
            let ndotn = clamp(dot(probe.normal.xyz, N_ws), 0.0, 1.0);
            if (ndotn < 0.80) { continue; }
            let world_delta = probe.world_pos.xyz - P_ws;
            let plane_error = max(
                abs(dot(world_delta, N_ws)),
                abs(dot(world_delta, probe.normal.xyz)),
            );
            let probe_world_spacing =
                2.0 * max(linear_z, 0.1) * tile /
                max(abs(p00) * half_w, 0.0001);
            let plane_sigma = 0.015 + probe_world_spacing * 0.12;
            if (plane_error > plane_sigma * 2.5) { continue; }
            let w_plane = exp(
                -0.5 * plane_error * plane_error /
                max(plane_sigma * plane_sigma, 0.000001),
            );
            let w_normal = pow(ndotn, 8.0);
            let w = w_corner * w_plane * w_normal;
            if (w <= 0.0001) { continue; }

            let radiance = sample_probe(vec2<i32>(gx, gy));
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
