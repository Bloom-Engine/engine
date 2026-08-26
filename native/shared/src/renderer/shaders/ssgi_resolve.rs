//! Per-pixel screen-probe GI reconstruction.

/// Writes the half-resolution `ssgi_rt` consumed by downstream composition.
///
/// Samples the probes surrounding the pixel and bilateral-weights the
/// contribution by a symmetric world-space same-plane test plus normal match.
/// A compact 2×2 footprint uses cubic-Hermite interpolation so both its value
/// and first derivative stay continuous when probe membership changes at a
/// screen-cell boundary. Invalid probes (sky) are skipped. An empty strict
/// footprint is reconstructed only from a geometrically compatible pair with
/// meaningful kernel coverage; unsupported edges stay black.
pub(in crate::renderer) const SSGI_PROBE_RESOLVE_WGSL: &str = "
struct ResolveParams {
    inv_view: mat4x4<f32>,
    prev_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    // x = half_w, y = half_h, z = grid_w, w = grid_h
    size: vec4<u32>,
    // x = tile_size (8.0), y = intensity, zw unused
    params: vec4<f32>,
    // x = low-discrepancy placement phase [0, 15], y = jitter active, zw unused
    temporal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ResolveParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var radiance_tex: texture_3d<f32>;
@group(0) @binding(3) var radiance_samp: sampler;
@group(0) @binding(4) var hiz0: texture_2d<f32>;
@group(0) @binding(5) var hiz_samp: sampler;
@group(0) @binding(6) var resolve_history_tex: texture_2d<f32>;
@group(0) @binding(7) var velocity_tex: texture_2d<f32>;

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

fn fallback_probe(probe: ProbeHeader) -> vec3<f32> {
    return probe.previous_diffuse.rgb * max(probe.previous_diffuse.w, 0.0);
}

// Interpolating cubic Hermite coordinate. It reaches both endpoints with a
// zero derivative, so replacing the left probe at a cell boundary does not
// reveal the otherwise screen-fixed slope seam of ordinary bilinear weights.
// This preserves the compact four-probe footprint and adds only a few ALU.
fn continuous_probe_coordinate(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
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
    // Central-view facing is constant across a planar receiver and avoids a
    // per-pixel normalize. It is sufficient to distinguish the shallow road
    // from frontal walls for the broad resolve policy below.
    let view_facing = abs(N_vs.z);
    // The broad estimate owns shallow and moderately facing surfaces. The
    // former 0.45 cut put a typical downward-looking floor in the transition,
    // where narrow same-plane support alternated on exact probe rows. Start a
    // continuous transition only once the receiver is substantially frontal;
    // at both endpoints the two branches evaluate to the same result.
    let broad_only = view_facing <= 0.65;
    let blend_broad = view_facing < 0.80;

    // Pixel's grid-space fractional position (which probes surround it?).
    let px_x = in.uv.x * half_w;
    let px_y = in.uv.y * half_h;
    // Align reconstruction to this frame's actual jittered probe positions.
    // The prior resolve-only shift moved weights over the same fixed samples;
    // it could not remove their screen-space radiance lattice.
    let placement_jitter = probe_lattice_jitter(u32(round(u.temporal.x))) *
        select(0.0, 1.0, u.temporal.y > 0.5);
    let fx = px_x / tile - 0.5 - placement_jitter.x;
    let fy = px_y / tile - 0.5 - placement_jitter.y;
    let gx0 = i32(floor(fx));
    let gy0 = i32(floor(fy));
    let tx = fract(fx);
    let ty = fract(fy);
    let smooth_tx = continuous_probe_coordinate(tx);
    let smooth_ty = continuous_probe_coordinate(ty);

    var accum = vec3<f32>(0.0);
    var wsum = 0.0;
    var fallback_radiance = vec3<f32>(0.0);
    var fallback_weight = 0.0;
    var fallback_count = 0u;
    let probe_world_spacing =
        2.0 * max(linear_z, 0.1) * tile /
        max(abs(p00) * half_w, 0.0001);
    let plane_sigma = 0.015 + probe_world_spacing * 0.12;
    // The broad fallback is used only when the strict bilateral kernel has no
    // support. At grazing angles, one screen tile can span several times its
    // horizontal world footprint along a gently changing road surface. Cover
    // that footprint while still requiring two independently placed,
    // normal-compatible probes; unsupported silhouettes remain black.
    let fallback_plane_limit = (0.08 + probe_world_spacing * 0.85) * 3.0;

    // Four probes remain sufficient when their interpolation coordinate is C1
    // continuous. This avoids the 3x3 grazing cost while removing the screen-
    // row derivative seam that the prior linear weights exposed in motion.
    for (var dy = 0; dy <= 1; dy = dy + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let raw_gx = gx0 + dx;
            let raw_gy = gy0 + dy;
            let gx = clamp(raw_gx, 0, grid_w - 1);
            let gy = clamp(raw_gy, 0, grid_h - 1);
            let probe = probes[u32(gy * grid_w + gx)];
            if (probe.world_pos.w < 0.5) { continue; }

            let w_corner =
                select(1.0 - smooth_tx, smooth_tx, dx == 1) *
                select(1.0 - smooth_ty, smooth_ty, dy == 1);

            // A view-depth difference rejects valid samples on an oblique
            // wall and changes as the camera moves. Compare the receiver and
            // probe in world space instead, accepting only the same plane.
            let ndotn = clamp(dot(probe.normal.xyz, N_ws), 0.0, 1.0);
            let world_delta = probe.world_pos.xyz - P_ws;
            let plane_error = max(
                abs(dot(world_delta, N_ws)),
                abs(dot(world_delta, probe.normal.xyz)),
            );
            let coherent_compatible =
                ndotn >= 0.65 && plane_error <= fallback_plane_limit;
            if (!broad_only) {
                // Fade strict membership continuously. The former hard normal
                // and plane cut admitted an entire probe row at once, which is
                // the exact horizontal support pattern visible in captures.
                let normal_support = smoothstep(0.72, 0.82, ndotn);
                let plane_ratio = plane_error / max(plane_sigma, 0.000001);
                let plane_support = 1.0 - smoothstep(2.0, 3.0, plane_ratio);
                let w_plane = exp(
                    -0.5 * plane_error * plane_error /
                    max(plane_sigma * plane_sigma, 0.000001),
                );
                let w_normal = pow(ndotn, 8.0);
                let w = w_corner * w_plane * w_normal *
                    normal_support * plane_support;
                if (w > 0.0001) {
                    let strict_radiance = sample_probe(vec2<i32>(gx, gy));
                    accum = accum + strict_radiance * w;
                    wsum = wsum + w;
                    // Front-facing detail keeps the accepted fast path. A
                    // grazing receiver also contributes to the already
                    // existing broad accumulator so all four compatible
                    // probes can own the tile without another texture read.
                    if (!blend_broad) { continue; }
                }
            }

            // A strict reject may contribute to a broader, still geometric
            // fallback. Two compatible probes establish coherent receiver
            // support; the count and aggregate-weight gates prevent a lone
            // distant corner from being normalized into a silhouette leak.
            if (coherent_compatible) {
                // The continuous coordinate already reaches zero with a zero
                // derivative. Broad and transition receivers therefore need
                // no artificial weight floor; frontal emergency fallback
                // retains it so a partial edge footprint cannot normalize one
                // tiny corner into a large leak.
                var fallback_corner_weight = max(w_corner, 0.125);
                var fallback_source = fallback_probe(probe);
                if (blend_broad) {
                    // The temporal header is intentionally unfiltered. Read
                    // the geometry-filtered spatial atlas for a floor/oblique
                    // footprint without adding work to the frontal path.
                    fallback_corner_weight = w_corner;
                    fallback_source = sample_probe(vec2<i32>(gx, gy));
                }
                fallback_radiance =
                    fallback_radiance + fallback_source * fallback_corner_weight;
                fallback_weight = fallback_weight + fallback_corner_weight;
                fallback_count = fallback_count + 1u;
            }
        }
    }

    if (broad_only && fallback_count >= 2u && fallback_weight >= 0.25) {
        // At a shallow angle the strict plane test is the screen-row signal
        // we are removing. Reconstruct directly from a coherent pair; this
        // also avoids the strict kernel's exp/pow and radiance texture load.
        accum = (fallback_radiance / fallback_weight) * u.params.y;
    } else if (blend_broad && fallback_count >= 4u && fallback_weight >= 0.75) {
        // A complete same-surface footprint owns grazing reconstruction. The
        // strict kernel remains authoritative on front-facing detail, where
        // it is stable and better preserves geometric edges. Its preference
        // begins at zero exactly where broad-only ends and reaches one exactly
        // where the transition ends, so changing branches cannot pop.
        let broad = fallback_radiance / fallback_weight;
        if (wsum > 0.0001) {
            let strict = accum / wsum;
            let strict_preference = smoothstep(0.65, 0.80, view_facing);
            accum = mix(broad, strict, strict_preference) * u.params.y;
        } else {
            accum = broad * u.params.y;
        }
    } else if (wsum > 0.0001) {
        accum = (accum / wsum) * u.params.y;
    } else if (fallback_count >= 2u && fallback_weight >= 0.25) {
        // Normalize partial support so screen-grid phase cannot modulate
        // energy. Geometry compatibility controls admission; valid indirect
        // light retains the same energy scale as the strict kernel.
        accum = (fallback_radiance / fallback_weight) * u.params.y;
    }
    // Probe history is world-reprojected, but the 2x2 reconstruction lattice
    // above is deliberately screen tiled. Static output is smooth; during a
    // camera move, however, TAA must favor the current frame to avoid scene
    // ghosting and therefore exposes that lattice as a pattern glued to the
    // display. Reproject the already-resolved indirect field here, while its
    // receiver depth is still available, so TAA receives world-anchored GI.
    // This is folded into resolve: no extra pass and no extra current target.
    var resolved = accum;
    var velocity = vec2<f32>(0.0);
    if (u.params.w > 0.5) {
        velocity = textureSampleLevel(velocity_tex, radiance_samp, in.uv, 0.0).xy;
    }
    let velocity_length = length(velocity);
    let motion_amount = smoothstep(0.000001, 0.00005, velocity_length);
    let previous_uv = vec2<f32>(
        in.uv.x - velocity.x,
        in.uv.y + velocity.y,
    );
    let history_in_bounds =
        all(previous_uv >= vec2<f32>(0.0)) &&
        all(previous_uv <= vec2<f32>(1.0));
    if (u.params.z < 0.999 && u.params.w > 0.5 &&
        motion_amount > 0.0 && history_in_bounds) {
        let history_size = vec2<i32>(textureDimensions(resolve_history_tex));
        let history_coord = clamp(
            vec2<i32>(floor(previous_uv * vec2<f32>(history_size))),
            vec2<i32>(0),
            history_size - vec2<i32>(1),
        );
        let history_depth = textureLoad(resolve_history_tex, history_coord, 0).a;
        let previous_view_position = u.prev_view * vec4<f32>(P_ws, 1.0);
        let expected_previous_depth = max(-previous_view_position.z, 0.0);
        // Half-resolution point depth represents a finite pixel footprint.
        // Scale tolerance with distance, but keep it far narrower than the
        // spacing between distinct Bistro facade/foreground surfaces.
        let depth_tolerance = 0.04 + expected_previous_depth * 0.015;
        if (history_depth > 0.0 &&
            abs(history_depth - expected_previous_depth) <= depth_tolerance) {
            let history = max(
                textureSampleLevel(
                    resolve_history_tex,
                    radiance_samp,
                    previous_uv,
                    0.0,
                ).rgb,
                vec3<f32>(0.0),
            );
            // A 64-frame EMA keeps the world-reprojected field responsive to
            // lighting changes while suppressing the screen-probe lattice.
            let moving_current_weight = 0.015625;
            let current_weight = max(
                mix(1.0, moving_current_weight, motion_amount),
                u.params.z,
            );
            resolved = mix(history, accum, current_weight);
        }
    }
    // Alpha is private to SSGI resolve history. Scene composition consumes RGB
    // only, so preserve receiver linear depth here for next-frame validation.
    return vec4<f32>(resolved, linear_z);
}
";
