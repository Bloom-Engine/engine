// Bloom's display tonemap curves: compact ACES (kind 0), AgX (kind 1) and the
// full ACES fit (kind 2). The ACES curves are cheap and evaluated per pixel;
// AgX is baked into a table (renderer/tonemap_lut.rs, tonemap_lut.wgsl).

fn aces_tone(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let cc = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((c * (c * a + b)) / (c * (c * cc + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ACES RRT+ODT fit with an explicit 6.0 white point. This is the MJP
// BakingLab fit used by Godot 4's ACES mode, including its 1.8 exposure bias
// and gamut transforms. Keep the compact Narkowicz fit above as mode 0 for
// output compatibility; mode 2 is the reference-quality/highlight-preserving
// path used by scenes that request the full transform.
fn aces_full_tone(c_in: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 1.8;
    let a = 0.0245786;
    let b = 0.000090537;
    let cc = 0.983729;
    let d = 0.432951;
    let e = 0.238081;
    let rgb_to_rrt = mat3x3<f32>(
        vec3<f32>(0.59719, 0.35458, 0.04823) * exposure_bias,
        vec3<f32>(0.07600, 0.90834, 0.01566) * exposure_bias,
        vec3<f32>(0.02840, 0.13383, 0.83777) * exposure_bias,
    );
    let odt_to_rgb = mat3x3<f32>(
        vec3<f32>( 1.60475, -0.53108, -0.07367),
        vec3<f32>(-0.10208,  1.10813, -0.00605),
        vec3<f32>(-0.00327, -0.07276,  1.07602),
    );
    // Godot's source uses row-vector multiplication (`color *= matrix`).
    // Preserve that convention explicitly; swapping this order transposes the
    // gamut transform and gives neutral surfaces a strong green cast.
    let c = max(c_in, vec3<f32>(0.0)) * rgb_to_rrt;
    let mapped = (c * (c + a) - b) / (c * (cc * c + d) + e);
    let white = 6.0 * exposure_bias;
    let white_mapped = (white * (white + a) - b)
        / (white * (cc * white + d) + e);
    return clamp(mapped * odt_to_rgb / white_mapped,
        vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- AgX tonemap (Blender/Filament reference) ---
// Better hue preservation than ACES in saturated regions — reds
// stay red instead of shifting toward orange, blues stay blue
// instead of shifting toward cyan. Same sigmoid shape overall,
// so the overall contrast is similar.

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return vec3<f32>(15.5) * x4 * x2
         - vec3<f32>(40.14) * x4 * x
         + vec3<f32>(31.96) * x4
         - vec3<f32>(6.868) * x2 * x
         + vec3<f32>(0.4298) * x2
         + vec3<f32>(0.1191) * x
         - vec3<f32>(0.00232);
}

fn agx_tone(val_in: vec3<f32>) -> vec3<f32> {
    // AgX input transform — compresses the input color gamut.
    let agx_mat = mat3x3<f32>(
        vec3<f32>(0.842479062253094,  0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772,  0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    // Log2-space normalization range. Anything outside gets clamped
    // — the sigmoid maps this window to [0, 1].
    let min_ev = -12.47393;
    let max_ev = 4.026069;

    var val = agx_mat * val_in;
    // Log2 encode, clamp to range, normalize to [0, 1].
    val = max(val, vec3<f32>(1e-10));
    val = clamp(log2(val), vec3<f32>(min_ev), vec3<f32>(max_ev));
    val = (val - vec3<f32>(min_ev)) / (max_ev - min_ev);

    // Sigmoid contrast curve.
    val = agx_default_contrast_approx(val);
    return val;
}

fn agx_eotf(val_in: vec3<f32>) -> vec3<f32> {
    // AgX inverse input transform — re-expands back to target
    // display gamut. The surface is sRGB-format so hardware
    // applies the sRGB EOTF on write; we output linear here.
    let agx_mat_inv = mat3x3<f32>(
        vec3<f32>( 1.19687900512017,   -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368,  1.15190312990417,   -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433,  1.15107367264116),
    );
    return agx_mat_inv * val_in;
}

// The AgX display curve. The composite pass reads it from the table baked
// from this function (renderer/tonemap_lut.rs) rather than evaluating it.
fn agx_curve(hdr: vec3<f32>) -> vec3<f32> {
    // Full Filament AgX pipeline:
    //   agx_tone  (inset + log2 + sigmoid)
    //   agx_eotf  (outset = inverse inset)
    //   pow(2.2)  (display-encoded → linear, required for sRGB surface)
    //
    // Filament's polynomial-approximation AgX (what we use) systematically
    // under-saturates vs Blender/Cycles' LUT-based AgX. Apply the 'Punchy'
    // look — saturation 1.4, slight contrast — as a post step to bring
    // colours closer to the Cycles ground-truth reference. Matches
    // Filament::ToneMapper::AGX_PUNCHY.
    let hdr_safe = max(hdr, vec3<f32>(0.0));
    let agx_display = clamp(agx_eotf(agx_tone(hdr_safe)), vec3<f32>(0.0), vec3<f32>(1.0));
    let linear = pow(agx_display, vec3<f32>(2.2));
    // Punchy look: post-tonemap saturation + contrast in display space.
    // slope/offset/power are per-channel lift/gamma/gain; saturation
    // multiplies chroma around the pixel's luma.
    let agx_punchy = agx_look_punchy(linear);
    return agx_punchy;
}

fn agx_look_punchy(val: vec3<f32>) -> vec3<f32> {
    // Per-channel slope (gain): subtle warm white-balance toward the
    // 'golden hour' feel of Cycles-AgX renders — +3% red, -2% blue.
    // The Bistro outdoor.hdr leans cool; without this, the shadows
    // and IBL fill pull the overall image toward blue-grey vs.
    // Cycles's warmer midtone neutral.
    let slope = vec3<f32>(1.03, 1.00, 0.98);
    let offset = vec3<f32>(0.0);
    // A whisker above neutral — Cycles-AgX with its LUT-based DRT
    // preserves enough chroma on its own that our polynomial fit only
    // needs a gentle post-boost to catch up.
    let power = vec3<f32>(1.1);
    let saturation = 1.1;
    // ASC-CDL-ish: (val * slope + offset) ^ power
    let toned = pow(max(val * slope + offset, vec3<f32>(0.0)), power);
    let luma = dot(toned, vec3<f32>(0.2126, 0.7152, 0.0722));
    return vec3<f32>(luma) + saturation * (toned - vec3<f32>(luma));
}
