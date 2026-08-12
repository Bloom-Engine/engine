struct TemporalParams {
    params: vec4<f32>,
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TemporalParams;
@group(0) @binding(1) var radiance_in: texture_3d<f32>;
@group(0) @binding(2) var history_in: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(4) var reason_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var confidence_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var velocity_tex: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dimensions = textureDimensions(reason_out);
    if (gid.x >= dimensions.x || gid.y >= dimensions.y) {
        return;
    }
    let probe = vec2<u32>(gid.xy / vec2<u32>(8u));
    let probe_idx = probe.y * (dimensions.x / 8u) + probe.x;
    let current_probe = probes[probe_idx];
    let valid_probe = current_probe.world_pos.w >= 0.5;
    var history_probe_idx = probe_idx;
    var geometry_valid = false;
    // 0 invalid current, 1 valid current, 2 previous UV outside,
    // 3 searched but found no matching prior surface, 4 matched.
    var reprojection_stage = 0u;
    var motion_length = 0.0;
    if (valid_probe) {
        reprojection_stage = 1u;
        let current_uv = (
            vec2<f32>(probe) * u.size.z + vec2<f32>(u.size.z * 0.5)
        ) / u.size.xy;
        let velocity_size = vec2<i32>(textureDimensions(velocity_tex));
        let velocity_coord = clamp(
            vec2<i32>(current_uv * vec2<f32>(velocity_size)),
            vec2<i32>(0),
            velocity_size - vec2<i32>(1),
        );
        let velocity = textureLoad(velocity_tex, velocity_coord, 0).xy;
        motion_length = length(velocity);
        let previous_uv = vec2<f32>(
            current_uv.x - velocity.x,
            current_uv.y + velocity.y,
        );
        if (all(previous_uv >= vec2<f32>(0.0)) &&
            all(previous_uv <= vec2<f32>(1.0))) {
            reprojection_stage = 3u;
            let previous_grid_position =
                previous_uv * u.size.xy / u.size.z - vec2<f32>(0.5);
            let previous_grid_center = vec2<i32>(
                floor(previous_grid_position + vec2<f32>(0.5)),
            );
            let grid_w = u32(u.params.z);
            let grid_h = u32(u.params.w);
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
                    let candidate_idx =
                        u32(candidate_xy.y) * grid_w + u32(candidate_xy.x);
                    let candidate = probes[candidate_idx];
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
                        history_probe_idx = candidate_idx;
                        geometry_valid = true;
                        reprojection_stage = 4u;
                    }
                }
            }
        } else {
            reprojection_stage = 2u;
        }
    }
    let octel = gid.xy % vec2<u32>(8u);
    let coord = vec3<i32>(
        vec2<i32>(probe),
        i32(octel.y * 8u + octel.x),
    );
    let history_coord = vec3<i32>(
        i32(history_probe_idx % u32(u.params.z)),
        i32(history_probe_idx / u32(u.params.z)),
        i32(octel.y * 8u + octel.x),
    );
    let curr = textureLoad(radiance_in, coord, 0).rgb;
    var hist = textureLoad(history_in, history_coord, 0).rgb;
    // Production reserves octel zero for the probe's cosine-convolved
    // diffuse result and seeds that lane's directional history from current.
    if (octel.x == 0u && octel.y == 0u) {
        hist = curr;
    }
    let curr_finite = all(abs(curr) <= vec3<f32>(65504.0));
    let hist_finite = all(abs(hist) <= vec3<f32>(65504.0));
    let finite = curr_finite && hist_finite;
    let curr_luma = dot(curr, vec3<f32>(0.2126, 0.7152, 0.0722));
    let hist_luma = dot(hist, vec3<f32>(0.2126, 0.7152, 0.0722));
    let delta = abs(curr_luma - hist_luma);
    let motion_refresh = smoothstep(0.00025, 0.003, motion_length);
    var alpha = min(1.0, mix(u.params.x, 0.65, motion_refresh) + delta * 0.6);
    if (u.params.y > 0.5 || !geometry_valid) {
        alpha = 1.0;
    }

    // Shared palette in probe space: gray seed, magenta adaptive radiance
    // refresh/invalid data, blue motion refresh, green retained history.
    var reason = vec3<f32>(0.05, 0.65, 0.10);
    if (!valid_probe || u.params.y > 0.5) {
        reason = vec3<f32>(0.25);
    } else if (!geometry_valid) {
        // Red identifies off-screen motion; cyan means the reprojected
        // footprint was searched but no matching prior surface was found.
        reason = select(
            vec3<f32>(1.0, 0.1, 0.0),
            vec3<f32>(0.0, 0.8, 1.0),
            reprojection_stage == 3u,
        );
    } else if (!finite || delta * 0.6 >= u.params.x) {
        reason = vec3<f32>(1.0, 0.0, 0.8);
    } else if (motion_refresh > 0.01) {
        reason = vec3<f32>(0.05, 0.25, 1.0);
    }
    var variation_heat = 1.0;
    var current_heat = 0.0;
    var retained_history = 0.0;
    if (finite && valid_probe && geometry_valid) {
        variation_heat = 1.0 - exp(-delta * 4.0);
        current_heat = 1.0 - exp(-max(curr_luma, 0.0) * 4.0);
        retained_history = clamp(1.0 - alpha, 0.0, 1.0);
    }
    textureStore(reason_out, vec2<i32>(gid.xy), vec4<f32>(reason, 1.0));
    textureStore(
        confidence_out,
        vec2<i32>(gid.xy),
        vec4<f32>(variation_heat, current_heat, retained_history, 1.0),
    );
}
