struct PtDiagnosticParams {
    inv_vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    size: vec4<u32>,
    cfg: vec4<f32>,
    ext: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: PtDiagnosticParams;
@group(0) @binding(1) var<storage, read> accum_current: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> moments_current: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> moments_previous: array<vec4<f32>>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var velocity_tex: texture_2d<f32>;
@group(0) @binding(6) var reason_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var motion_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var reprojection_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(9) var confidence_out: texture_storage_2d<rgba8unorm, write>;

fn finite4(value: vec4<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec4<f32>(1e20));
}

fn linear_depth(depth: f32) -> f32 {
    return 0.02 / max(1.0 - depth, 1e-6);
}

fn full_pixel(trace_pixel: vec2<i32>) -> vec2<i32> {
    if (u.ext.x <= u.size.x) {
        return trace_pixel;
    }
    return min(
        vec2<i32>(
            trace_pixel.x * i32(u.ext.x) / i32(u.size.x),
            trace_pixel.y * i32(u.ext.y) / i32(u.size.y),
        ),
        vec2<i32>(i32(u.ext.x) - 1, i32(u.ext.y) - 1),
    );
}

fn depth_at(pixel: vec2<i32>) -> f32 {
    let limit = vec2<i32>(i32(u.ext.x) - 1, i32(u.ext.y) - 1);
    return textureLoad(depth_tex, clamp(pixel, vec2<i32>(0), limit), 0);
}

fn world_at(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    let dims = vec2<f32>(f32(u.ext.x), f32(u.ext.y));
    let uv = (vec2<f32>(pixel) + 0.5) / dims;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = u.inv_vp * ndc;
    return world.xyz / world.w;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.size.x || gid.y >= u.size.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let pixel_full = full_pixel(pixel);
    let index = gid.y * u.size.x + gid.x;
    let current_accum = accum_current[index];
    let current_moments = moments_current[index];
    let depth = current_moments.w;
    let sky = depth >= 0.9999999;
    let seeded = u.size.w == 0u;
    let current_finite = finite4(current_accum) && finite4(current_moments);
    let velocity = textureLoad(velocity_tex, pixel_full, 0).rg;
    let moving = abs(velocity.x) + abs(velocity.y) > 1e-5;

    var history_in_bounds = false;
    var history_finite = true;
    var accepted_mass = 0.0;
    var footprint_retained = false;
    var previous_uv = vec2<f32>(0.0);
    if (!seeded && !sky && current_finite) {
        let position = world_at(pixel_full, depth);
        var linear_here = 0.0;
        if (moving) {
            let current_uv = (vec2<f32>(pixel_full) + 0.5)
                / vec2<f32>(f32(u.ext.x), f32(u.ext.y));
            previous_uv = vec2<f32>(
                current_uv.x - velocity.x,
                current_uv.y + velocity.y,
            );
            linear_here = linear_depth(depth);
        } else {
            let previous_clip = u.prev_vp * vec4<f32>(position, 1.0);
            if (previous_clip.w > 1e-4) {
                let previous_ndc = previous_clip.xyz / previous_clip.w;
                previous_uv = vec2<f32>(
                    previous_ndc.x * 0.5 + 0.5,
                    0.5 - previous_ndc.y * 0.5,
                );
                linear_here = linear_depth(previous_ndc.z);
            }
        }
        history_in_bounds =
            previous_uv.x >= 0.0 && previous_uv.x < 1.0 &&
            previous_uv.y >= 0.0 && previous_uv.y < 1.0 &&
            linear_here > 0.0;
        if (history_in_bounds) {
            let position_previous =
                previous_uv * vec2<f32>(f32(u.size.x), f32(u.size.y)) - 0.5;
            let base = vec2<i32>(floor(position_previous));
            let fraction = position_previous - floor(position_previous);
            let tolerance = 0.1 * linear_here + 0.02;
            for (var ty = 0; ty <= 1; ty = ty + 1) {
                for (var tx = 0; tx <= 1; tx = tx + 1) {
                    let sample_pixel = base + vec2<i32>(tx, ty);
                    if (sample_pixel.x < 0 || sample_pixel.y < 0 ||
                        sample_pixel.x >= i32(u.size.x) ||
                        sample_pixel.y >= i32(u.size.y)) {
                        continue;
                    }
                    let sample_index =
                        u32(sample_pixel.y) * u.size.x + u32(sample_pixel.x);
                    let sample_moments = moments_previous[sample_index];
                    if (!finite4(sample_moments)) {
                        history_finite = false;
                        continue;
                    }
                    if (sample_moments.w >= 0.9999999 ||
                        abs(linear_depth(sample_moments.w) - linear_here) >
                            tolerance) {
                        continue;
                    }
                    let wx = mix(1.0 - fraction.x, fraction.x, f32(tx));
                    let wy = mix(1.0 - fraction.y, fraction.y, f32(ty));
                    accepted_mass += wx * wy + 1e-4;
                }
            }

            if (accepted_mass <= 1e-3) {
                let nearest_pixel = vec2<u32>(
                    min(
                        u32(max(base.x + i32(round(fraction.x)), 0)),
                        u.size.x - 1u,
                    ),
                    min(
                        u32(max(base.y + i32(round(fraction.y)), 0)),
                        u.size.y - 1u,
                    ),
                );
                let nearest_index = nearest_pixel.y * u.size.x + nearest_pixel.x;
                let nearest_moments = moments_previous[nearest_index];
                history_finite = history_finite && finite4(nearest_moments);
                var footprint_low = 1e30;
                var footprint_high = 0.0;
                let ratio_x = max(i32(u.ext.x) / i32(u.size.x), 1);
                let ratio_y = max(i32(u.ext.y) / i32(u.size.y), 1);
                for (var sy = 0; sy <= 1; sy = sy + 1) {
                    for (var sx = 0; sx <= 1; sx = sx + 1) {
                        let sample_full = min(
                            pixel_full + vec2<i32>(
                                sx * (ratio_x - 1),
                                sy * (ratio_y - 1),
                            ),
                            vec2<i32>(i32(u.ext.x) - 1, i32(u.ext.y) - 1),
                        );
                        let sample_depth = depth_at(sample_full);
                        if (sample_depth < 0.9999999) {
                            let sample_linear = linear_depth(sample_depth);
                            footprint_low = min(footprint_low, sample_linear);
                            footprint_high = max(footprint_high, sample_linear);
                        }
                    }
                }
                if (history_finite && nearest_moments.w < 0.9999999 &&
                    nearest_moments.z > 0.0 && footprint_high > 0.0 &&
                    footprint_low < 1e29) {
                    let stored_linear = linear_depth(nearest_moments.w);
                    let window = 0.1 * stored_linear + 0.02;
                    footprint_retained =
                        stored_linear > footprint_low - window &&
                        stored_linear < footprint_high + window;
                }
            }
        }
    }

    // Shared palette: gray seed/sky, red off-screen, magenta invalid or
    // disoccluded, cyan retained footprint flip, blue accepted motion-vector
    // reprojection, green accepted matrix/static history.
    var reason = vec3<f32>(0.05, 0.65, 0.10);
    if (seeded || sky) {
        reason = vec3<f32>(0.25);
    } else if (!history_in_bounds) {
        reason = vec3<f32>(1.0, 0.05, 0.02);
    } else if (!current_finite || !history_finite || accepted_mass <= 1e-3) {
        reason = select(
            vec3<f32>(1.0, 0.0, 0.8),
            vec3<f32>(0.0, 0.9, 1.0),
            footprint_retained && current_finite && history_finite,
        );
    } else if (moving) {
        reason = vec3<f32>(0.05, 0.25, 1.0);
    }

    let motion = vec3<f32>(
        clamp(0.5 + velocity.x * 32.0, 0.0, 1.0),
        clamp(0.5 - velocity.y * 32.0, 0.0, 1.0),
        clamp(length(velocity) * 64.0, 0.0, 1.0),
    );
    let valid_reprojection = history_in_bounds && history_finite;
    let variance_heat = select(
        1.0,
        1.0 - exp(-max(current_accum.w, 0.0) * 10.0),
        current_finite,
    );
    let history_length = select(
        0.0,
        clamp(current_moments.z / 32.0, 0.0, 1.0),
        current_finite && !sky,
    );
    var retained_history = 0.0;
    if (footprint_retained) {
        retained_history = 1.0;
    } else if (current_finite && current_moments.z > 1.0 &&
               history_in_bounds) {
        retained_history =
            1.0 - max(1.0 / current_moments.z, 0.1);
    }

    textureStore(reason_out, pixel, vec4<f32>(reason, 1.0));
    textureStore(motion_out, pixel, vec4<f32>(motion, 1.0));
    textureStore(
        reprojection_out,
        pixel,
        vec4<f32>(
            clamp(previous_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
            select(0.0, 1.0, valid_reprojection),
            1.0,
        ),
    );
    textureStore(
        confidence_out,
        pixel,
        vec4<f32>(variance_heat, history_length, retained_history, 1.0),
    );
}
