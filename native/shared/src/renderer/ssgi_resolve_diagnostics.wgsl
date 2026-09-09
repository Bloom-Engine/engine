// Capture-only screen-space classification of SSGI resolve support. Keep the
// geometry tests in lockstep with `shaders/ssgi_resolve.rs`; this pass does not
// write or otherwise affect production history.

struct ResolveParams {
    inv_view: mat4x4<f32>,
    prev_view: mat4x4<f32>,
    proj_row01: vec4<f32>,
    size: vec4<u32>,
    params: vec4<f32>,
    temporal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: ResolveParams;
@group(0) @binding(1) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(2) var hiz0: texture_2d<f32>;
@group(0) @binding(3) var support_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var geometry_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var plane_ratios_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var plane_ratio_w_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var resolve_history_tex: texture_2d<f32>;
@group(0) @binding(8) var velocity_tex: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn cs_resolve_support(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dimensions = textureDimensions(support_out);
    if (gid.x >= dimensions.x || gid.y >= dimensions.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let half_w = f32(u.size.x);
    let half_h = f32(u.size.y);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(half_w, half_h);
    let hiz_dimensions = vec2<i32>(textureDimensions(hiz0));
    let hiz_coord = clamp(
        vec2<i32>(uv * vec2<f32>(hiz_dimensions)),
        vec2<i32>(0),
        hiz_dimensions - vec2<i32>(1),
    );
    // Mirror production's normalized nearest-texel selection instead of
    // assuming the capture target and Hi-Z texture always share dimensions.
    // They currently do at native scale, while this mapping also keeps the
    // capture-only oracle correct for diagnostic extent changes.
    let linear_z = textureLoad(hiz0, hiz_coord, 0).r;
    if (linear_z >= HIZ_SKY_Z * 0.5) {
        textureStore(support_out, coord, vec4<f32>(0.0));
        textureStore(geometry_out, coord, vec4<f32>(0.0));
        textureStore(plane_ratios_out, coord, vec4<f32>(0.0));
        textureStore(plane_ratio_w_out, coord, vec4<f32>(0.0));
        return;
    }

    let tile = u.params.x;
    let grid_w = i32(u.size.z);
    let grid_h = i32(u.size.w);
    let p00 = u.proj_row01.x;
    let p11 = u.proj_row01.y;
    let p20 = u.proj_row01.z;
    let p21 = u.proj_row01.w;
    let P_vs = view_pos_from_linear(uv, linear_z, p00, p11, p20, p21);
    let P_ws = (u.inv_view * vec4<f32>(P_vs, 1.0)).xyz;

    let texel = vec2<f32>(1.0 / half_w, 1.0 / half_h);
    let right_uv = uv + vec2<f32>(texel.x, 0.0);
    let up_uv = uv + vec2<f32>(0.0, -texel.y);
    let right_coord = clamp(
        vec2<i32>(right_uv * vec2<f32>(hiz_dimensions)),
        vec2<i32>(0),
        hiz_dimensions - vec2<i32>(1),
    );
    let up_coord = clamp(
        vec2<i32>(up_uv * vec2<f32>(hiz_dimensions)),
        vec2<i32>(0),
        hiz_dimensions - vec2<i32>(1),
    );
    let zr = textureLoad(hiz0, right_coord, 0).r;
    let zu = textureLoad(hiz0, up_coord, 0).r;
    let Pr = view_pos_from_linear(uv + vec2<f32>(texel.x, 0.0), zr, p00, p11, p20, p21);
    let Pu = view_pos_from_linear(uv + vec2<f32>(0.0, -texel.y), zu, p00, p11, p20, p21);
    let N_vs = safe_probe_direction(
        cross(Pr - P_vs, Pu - P_vs),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    let N_ws = safe_probe_direction(
        (u.inv_view * vec4<f32>(N_vs, 0.0)).xyz,
        vec3<f32>(0.0, 1.0, 0.0),
    );
    let view_facing = abs(N_vs.z);
    let blend_broad = view_facing < 0.80;

    let px_x = uv.x * half_w;
    let px_y = uv.y * half_h;
    let placement_jitter = probe_lattice_jitter(u32(round(u.temporal.x))) *
        select(0.0, 1.0, u.temporal.y > 0.5);
    let fx = px_x / tile - 0.5 - placement_jitter.x;
    let fy = px_y / tile - 0.5 - placement_jitter.y;
    let gx0 = i32(floor(fx));
    let gy0 = i32(floor(fy));
    let tx = fract(fx);
    let ty = fract(fy);
    let smooth_tx = tx * tx * (3.0 - 2.0 * tx);
    let smooth_ty = ty * ty * (3.0 - 2.0 * ty);
    var strict_weight = 0.0;
    var coherent_count = 0u;
    var coherent_weight = 0.0;
    var fallback_count = 0u;
    var fallback_weight = 0.0;
    var valid_count = 0u;
    var normal_compatible_count = 0u;
    var plane_compatible_count = 0u;
    var best_normal_plane_ratio = 4.0;
    var plane_ratios = array<f32, 4>(4.0, 4.0, 4.0, 4.0);
    let probe_world_spacing =
        2.0 * max(linear_z, 0.1) * tile /
        max(abs(p00) * half_w, 0.0001);
    let plane_sigma = 0.015 + probe_world_spacing * 0.12;
    let fallback_plane_limit = (0.08 + probe_world_spacing * 0.85) * 3.0;

    for (var dy = 0; dy <= 1; dy = dy + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let gx = clamp(gx0 + dx, 0, grid_w - 1);
            let gy = clamp(gy0 + dy, 0, grid_h - 1);
            let probe = probes[u32(gy * grid_w + gx)];
            if (probe.world_pos.w < 0.5) { continue; }
            valid_count = valid_count + 1u;

            let w_corner =
                select(1.0 - smooth_tx, smooth_tx, dx == 1) *
                select(1.0 - smooth_ty, smooth_ty, dy == 1);
            let ndotn = clamp(dot(probe.normal.xyz, N_ws), 0.0, 1.0);
            let world_delta = probe.world_pos.xyz - P_ws;
            let plane_error = max(
                abs(dot(world_delta, N_ws)),
                abs(dot(world_delta, probe.normal.xyz)),
            );
            let normal_compatible = ndotn >= 0.65;
            let plane_ratio = plane_error / max(fallback_plane_limit, 0.000001);
            let coherent_compatible = normal_compatible && plane_ratio <= 1.0;
            if (coherent_compatible) {
                coherent_count = coherent_count + 1u;
                coherent_weight = coherent_weight + w_corner;
            }
            var strict_supported = false;
            let normal_support = smoothstep(0.72, 0.82, ndotn);
            let strict_plane_ratio = plane_error / max(plane_sigma, 0.000001);
            let plane_support = 1.0 - smoothstep(2.0, 3.0, strict_plane_ratio);
            let w_plane = exp(
                -0.5 * plane_error * plane_error /
                max(plane_sigma * plane_sigma, 0.000001),
            );
            let w_normal = pow(ndotn, 8.0);
            let strict_w = w_corner * w_plane * w_normal *
                normal_support * plane_support;
            if (strict_w > 0.0001) {
                strict_weight = strict_weight + strict_w;
                strict_supported = true;
            }

            if (normal_compatible) {
                normal_compatible_count = normal_compatible_count + 1u;
                best_normal_plane_ratio = min(best_normal_plane_ratio, plane_ratio);
                let corner_index = u32(dy * 2 + dx);
                plane_ratios[corner_index] = min(plane_ratio, 4.0);
            }
            if (plane_ratio <= 1.0) {
                plane_compatible_count = plane_compatible_count + 1u;
            }
            if (coherent_compatible && !strict_supported) {
                let fallback_corner_weight = select(
                    max(w_corner, 0.125),
                    w_corner,
                    blend_broad,
                );
                fallback_count = fallback_count + 1u;
                fallback_weight = fallback_weight + fallback_corner_weight;
            }
        }
    }

    let coherent_supported = coherent_count == 4u && coherent_weight > 0.0001;
    let fallback_supported = fallback_count >= 2u && fallback_weight >= 0.25;
    let unsupported = !coherent_supported && strict_weight <= 0.0001 && !fallback_supported;
    textureStore(
        support_out,
        coord,
        vec4<f32>(
            select(0.0, 1.0, unsupported),
            f32(coherent_count) * 0.25,
            clamp(strict_weight, 0.0, 1.0),
            1.0,
        ),
    );
    // R = probes with normal similarity >= 0.65, G = probes within the
    // broader plane limit, B = best plane-error/limit ratio among the
    // normal-compatible probes mapped from [0, 4] to [0, 1], A = valid probes.
    textureStore(
        geometry_out,
        coord,
        vec4<f32>(
            f32(normal_compatible_count) * 0.25,
            f32(plane_compatible_count) * 0.25,
            clamp(best_normal_plane_ratio * 0.25, 0.0, 1.0),
            f32(valid_count) * 0.25,
        ),
    );
    // One normal-compatible plane-error/limit ratio per bilinear corner,
    // mapped from [0, 4] to [0, 1]. Quality PNGs deliberately omit alpha, so
    // the fourth value is written to a companion texture's red channel.
    textureStore(
        plane_ratios_out,
        coord,
        vec4<f32>(
            plane_ratios[0] * 0.25,
            plane_ratios[1] * 0.25,
            plane_ratios[2] * 0.25,
            1.0,
        ),
    );
    let velocity_dimensions = vec2<i32>(textureDimensions(velocity_tex));
    let velocity_coord = clamp(
        vec2<i32>(uv * vec2<f32>(velocity_dimensions)),
        vec2<i32>(0),
        velocity_dimensions - vec2<i32>(1),
    );
    var velocity = vec2<f32>(0.0);
    if (u.params.w > 0.5) {
        velocity = textureLoad(velocity_tex, velocity_coord, 0).xy;
    }
    let motion_amount = smoothstep(0.000001, 0.00005, length(velocity));
    let previous_uv = vec2<f32>(uv.x - velocity.x, uv.y + velocity.y);
    let history_in_bounds =
        all(previous_uv >= vec2<f32>(0.0)) &&
        all(previous_uv <= vec2<f32>(1.0));
    var history_accepted = false;
    var normalized_history_depth_error = 4.0;
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
        let depth_tolerance = 0.04 + expected_previous_depth * 0.015;
        normalized_history_depth_error =
            abs(history_depth - expected_previous_depth) / depth_tolerance;
        history_accepted = history_depth > 0.0 &&
            abs(history_depth - expected_previous_depth) <= depth_tolerance;
    }
    // R keeps the fourth plane ratio. G marks accepted resolve history; B
    // stores normalized receiver-depth error (0..4 mapped to 0..1).
    textureStore(
        plane_ratio_w_out,
        coord,
        vec4<f32>(
            plane_ratios[3] * 0.25,
            select(0.0, 1.0, history_accepted),
            clamp(normalized_history_depth_error * 0.25, 0.0, 1.0),
            1.0,
        ),
    );
}
