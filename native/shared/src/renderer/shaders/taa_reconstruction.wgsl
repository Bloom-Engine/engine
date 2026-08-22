// Exact separable Catmull-Rom upsample, footprint-clipped. The common five-tap
// approximation drops the four diagonal products of the cubic's outer lobes;
// restore them so bootstrap/native reconstruction is energy preserving.
fn sample_catmull_rom(
    uv: vec2<f32>,
    tex_size: vec2<f32>,
    inv_size: vec2<f32>,
) -> vec4<f32> {
    let sample_pos = uv * tex_size;
    let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let offset12 = w2 / w12;
    let tp0 = (tex_pos1 - 1.0) * inv_size;
    let tp3 = (tex_pos1 + 2.0) * inv_size;
    let tp12 = (tex_pos1 + offset12) * inv_size;
    let tap0 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp0.y), 0.0);
    let tap1 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x, tp12.y), 0.0);
    let tap2 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp12.y), 0.0);
    let tap3 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x, tp12.y), 0.0);
    let tap4 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp3.y), 0.0);
    let corner0 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x, tp0.y), 0.0);
    let corner1 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x, tp0.y), 0.0);
    let corner2 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x, tp3.y), 0.0);
    let corner3 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x, tp3.y), 0.0);
    let result =
        tap0 * w12.x * w0.y + tap1 * w0.x * w12.y + tap2 * w12.x * w12.y +
        tap3 * w3.x * w12.y + tap4 * w12.x * w3.y + corner0 * w0.x * w0.y +
        corner1 * w3.x * w0.y + corner2 * w0.x * w3.y + corner3 * w3.x * w3.y;
    let footprint_min = min(
        min(min(tap0, tap1), min(tap2, tap3)),
        min(min(tap4, corner0), min(min(corner1, corner2), corner3)),
    );
    let footprint_max = max(
        max(max(tap0, tap1), max(tap2, tap3)),
        max(max(tap4, corner0), max(max(corner1, corner2), corner3)),
    );
    return clamp(result, max(footprint_min, vec4<f32>(0.0)), footprint_max);
}

// Fractional-resolution reconstruction uses a Lanczos-2 footprint. At 0.75
// and above its four one-dimensional weights form the exact nine-read
// separable filter. Half scale uses the same weights as a five-read radial
// footprint, reserving the remaining reads for phase-stable history
// statistics. The compact polynomial is FidelityFX's sin/sqrt-free Lanczos
// approximation.
fn lanczos2_approx_sq4(distance_sq: vec4<f32>) -> vec4<f32> {
    let x2 = min(distance_sq, vec4<f32>(4.0));
    let a = vec4<f32>(0.4) * x2 - vec4<f32>(1.0);
    let b = vec4<f32>(0.25) * x2 - vec4<f32>(1.0);
    return (1.5625 * a * a - 0.5625) * b * b;
}

struct FractionalSample {
    value: vec4<f32>,
    center: vec4<f32>,
    weight: f32,
    mean: vec3<f32>,
    stddev: vec3<f32>,
};

// Rg16Float provenance has one half-float beside depth. Non-reactive history
// keeps the established normalized 0..1 confidence value byte-for-byte.
// Reactive history stores -(confidence + 1), preserving the same 17 exact
// confidence states while retaining coverage even when confidence is zero.
struct TemporalHistory {
    confidence: f32,
    reactive: f32,
};

fn unpack_temporal_history(payload: f32) -> TemporalHistory {
    let reactive = select(0.0, 1.0, payload < 0.0);
    let confidence = select(payload, -payload - 1.0, payload < 0.0);
    return TemporalHistory(clamp(confidence, 0.0, 1.0), reactive);
}

fn pack_temporal_history(confidence: f32, reactive: f32) -> f32 {
    return select(confidence, -confidence - 1.0, reactive > 0.01);
}

fn sample_fractional_lanczos2(
    uv: vec2<f32>,
    tex_size: vec2<f32>,
    inv_size: vec2<f32>,
    reconstruction_scale: f32,
) -> FractionalSample {
    let sample_pos = uv * tex_size;
    let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
    let phase = sample_pos - tex_pos1;
    let kernel_bias = min(4.0 / 3.0, 1.0 / max(reconstruction_scale, 0.5));

    let offset_x =
        (vec4<f32>(-1.0, 0.0, 1.0, 2.0) - vec4<f32>(phase.x)) * kernel_bias;
    let offset_y =
        (vec4<f32>(-1.0, 0.0, 1.0, 2.0) - vec4<f32>(phase.y)) * kernel_bias;
    let weights_x = lanczos2_approx_sq4(offset_x * offset_x);
    let weights_y = lanczos2_approx_sq4(offset_y * offset_y);
    let weight12 = vec2<f32>(weights_x.y + weights_x.z, weights_y.y + weights_y.z);
    let offset12 = vec2<f32>(weights_x.z, weights_y.z) / weight12;

    let tp0 = (tex_pos1 - 1.0) * inv_size;
    let tp3 = (tex_pos1 + 2.0) * inv_size;
    let tp12 = (tex_pos1 + offset12) * inv_size;

    let tap0 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp0.y), 0.0);
    let tap1 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x, tp12.y), 0.0);
    let tap2 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp12.y), 0.0);
    let tap3 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x, tp12.y), 0.0);
    let tap4 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp3.y), 0.0);
    var accumulated =
        tap0 * weight12.x * weights_y.x +
        tap1 * weights_x.x * weight12.y +
        tap2 * weight12.x * weight12.y +
        tap3 * weights_x.w * weight12.y +
        tap4 * weight12.x * weights_y.w;
    var weight =
        weight12.x * weights_y.x +
        weights_x.x * weight12.y +
        weight12.x * weight12.y +
        weights_x.w * weight12.y +
        weight12.x * weights_y.w;
    var footprint_min = min(min(tap0, tap1), min(tap2, min(tap3, tap4)));
    var footprint_max = max(max(tap0, tap1), max(tap2, max(tap3, tap4)));
    var mean = vec3<f32>(0.0);
    var stddev = vec3<f32>(0.0);
    if (reconstruction_scale >= 0.75) {
        // The 0.75 path retains the exact separable footprint.
        // At half scale the four diagonal outer-lobe products add little
        // useful coverage but cost four full-resolution texture reads; its
        // stable five-tap radial footprint pays for the wider reconstruction
        // without increasing the established pass bandwidth.
        let corner0 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x, tp0.y), 0.0);
        let corner1 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x, tp0.y), 0.0);
        let corner2 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x, tp3.y), 0.0);
        let corner3 = textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x, tp3.y), 0.0);
        accumulated = accumulated +
            corner0 * weights_x.x * weights_y.x +
            corner1 * weights_x.w * weights_y.x +
            corner2 * weights_x.x * weights_y.w +
            corner3 * weights_x.w * weights_y.w;
        weight = dot(weights_x, vec4<f32>(1.0)) * dot(weights_y, vec4<f32>(1.0));
        footprint_min = min(
            footprint_min,
            min(min(corner0, corner1), min(corner2, corner3)),
        );
        footprint_max = max(
            footprint_max,
            max(max(corner0, corner1), max(corner2, corner3)),
        );
        let stat0 = rgb_to_ycocg(tap2.rgb);
        let stat1 = rgb_to_ycocg(tap0.rgb);
        let stat2 = rgb_to_ycocg(tap1.rgb);
        let stat3 = rgb_to_ycocg(tap3.rgb);
        let stat4 = rgb_to_ycocg(tap4.rgb);
        let moment1 = stat0 + stat1 + stat2 + stat3 + stat4;
        let moment2 =
            stat0 * stat0 + stat1 * stat1 + stat2 * stat2 +
            stat3 * stat3 + stat4 * stat4;
        mean = moment1 * 0.2;
        let variance = max(moment2 * 0.2 - mean * mean, vec3<f32>(0.0)) * 2.0;
        stddev = sqrt(variance);
    }
    let result = accumulated / max(weight, 0.00001);
    let bounded = clamp(result, max(footprint_min, vec4<f32>(0.0)), footprint_max);
    return FractionalSample(bounded, tap2, max(weight, 0.00001), mean, stddev);
}
