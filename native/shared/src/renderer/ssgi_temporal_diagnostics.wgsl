struct TemporalParams {
    params: vec4<f32>,
};

struct ProbeHeader {
    world_pos: vec4<f32>,
    normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TemporalParams;
@group(0) @binding(1) var radiance_in: texture_3d<f32>;
@group(0) @binding(2) var history_in: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> probes: array<ProbeHeader>;
@group(0) @binding(4) var reason_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var confidence_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dimensions = textureDimensions(reason_out);
    if (gid.x >= dimensions.x || gid.y >= dimensions.y) {
        return;
    }
    let probe = vec2<u32>(gid.xy / vec2<u32>(8u));
    let probe_idx = probe.y * (dimensions.x / 8u) + probe.x;
    let valid_probe = probes[probe_idx].world_pos.w >= 0.5;
    let octel = gid.xy % vec2<u32>(8u);
    let coord = vec3<i32>(
        vec2<i32>(probe),
        i32(octel.y * 8u + octel.x),
    );
    let curr = textureLoad(radiance_in, coord, 0).rgb;
    let hist = textureLoad(history_in, coord, 0).rgb;
    let curr_finite = all(abs(curr) <= vec3<f32>(65504.0));
    let hist_finite = all(abs(hist) <= vec3<f32>(65504.0));
    let finite = curr_finite && hist_finite;
    let curr_luma = dot(curr, vec3<f32>(0.2126, 0.7152, 0.0722));
    let hist_luma = dot(hist, vec3<f32>(0.2126, 0.7152, 0.0722));
    let delta = abs(curr_luma - hist_luma);
    var alpha = min(1.0, u.params.x + delta * 0.6);
    if (u.params.y > 0.5) {
        alpha = 1.0;
    }

    // Shared palette in probe space: gray seed, magenta adaptive radiance
    // refresh/invalid data, green retained history. Screen-space categories
    // such as off-screen UV and motion do not exist for this representation.
    var reason = vec3<f32>(0.05, 0.65, 0.10);
    if (!valid_probe || u.params.y > 0.5) {
        reason = vec3<f32>(0.25);
    } else if (!finite || delta * 0.6 >= u.params.x) {
        reason = vec3<f32>(1.0, 0.0, 0.8);
    }
    var variation_heat = 1.0;
    var current_heat = 0.0;
    var retained_history = 0.0;
    if (finite && valid_probe) {
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
