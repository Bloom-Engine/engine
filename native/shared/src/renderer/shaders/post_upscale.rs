//! Upscale and sharpen (RCAS) passes. Split from post.rs.

// -----------------------------------------------------------------
// Upscale pass — render-res → full-surface. Engages only when
// `render_scale < 1.0 && !taa_enabled` (when TAA is on the TAA pass
// does its own Catmull-Rom reconstruction). Mode 0 = bilinear (cheap,
// soft), mode 1 = Catmull-Rom 5-tap (sharper edge reconstruction,
// same kernel as the TAA pass).
// -----------------------------------------------------------------
pub(in crate::renderer) const UPSCALE_SHADER_WGSL: &str = "
struct UpscaleParams {
    // x = mode (0 = bilinear, 1 = catmull-rom), yzw padding.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: UpscaleParams;
@group(0) @binding(1) var composed_tex: texture_2d<f32>;
@group(0) @binding(2) var composed_samp: sampler;

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

// 5-tap Catmull-Rom (Karis formulation) — same kernel as the TAA
// shader's upsample. Costs 5 bilinear fetches vs 1; reconstructs a
// cubic-Hermite curve through 4 source taps which preserves edges
// where naive bilinear goes mushy.
fn sample_catmull_rom(uv: vec2<f32>) -> vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(composed_tex));
    let inv_size = 1.0 / tex_size;
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
    var result = vec4<f32>(0.0);
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp0.y),  0.0) * w12.x * w0.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp0.x,  tp12.y), 0.0) * w0.x  * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp12.y), 0.0) * w12.x * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp3.x,  tp12.y), 0.0) * w3.x  * w12.y;
    result += textureSampleLevel(composed_tex, composed_samp, vec2<f32>(tp12.x, tp3.y),  0.0) * w12.x * w3.y;
    return max(result, vec4<f32>(0.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let mode = u32(u.params.x);
    if (mode == 1u) {
        return sample_catmull_rom(in.uv);
    }
    return textureSample(composed_tex, composed_samp, in.uv);
}
";

// -----------------------------------------------------------------
// Contrast-adaptive sharpen — a simplified RCAS (FidelityFX). 5-tap
// cross kernel, negative-lobe FIR with lobe amplitude adapted to
// per-pixel luma headroom so flat areas don't amplify noise. Runs
// before composite on whichever texture feeds composite today
// (sss/mb/dof/taa/upscale/composed). Gated on `strength > 0`;
// default 0 so the pass is a no-op unless the user opts in.
// -----------------------------------------------------------------
pub(in crate::renderer) const RCAS_SHADER_WGSL: &str = "
struct RcasParams {
    // x = strength (0 = off, 0.3 = subtle, 0.6 = punchy, 1.0 = max).
    // yzw padding.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: RcasParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var input_samp: sampler;

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
    let strength = u.params.x;
    let center = textureSample(input_tex, input_samp, in.uv);
    if (strength <= 0.0) {
        return center;
    }
    let tex_size = vec2<f32>(textureDimensions(input_tex));
    let px = 1.0 / tex_size;

    let c = center.rgb;
    let n = textureSample(input_tex, input_samp, in.uv + vec2<f32>( 0.0, -px.y)).rgb;
    let s = textureSample(input_tex, input_samp, in.uv + vec2<f32>( 0.0,  px.y)).rgb;
    let w = textureSample(input_tex, input_samp, in.uv + vec2<f32>(-px.x, 0.0)).rgb;
    let e = textureSample(input_tex, input_samp, in.uv + vec2<f32>( px.x, 0.0)).rgb;

    // Luma-based local min/max for contrast adaptation. Rec. 709.
    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let lc = dot(c, lw);
    let ln = dot(n, lw);
    let ls = dot(s, lw);
    let lwl = dot(w, lw);
    let le = dot(e, lw);
    let lmin = min(min(min(ln, ls), min(lwl, le)), lc);
    let lmax = max(max(max(ln, ls), max(lwl, le)), lc);

    // Headroom — how much room is there before we clip at 0 or the
    // local max? Small in flat areas, large at edges. This is the
    // 'Robust' part of RCAS: sharpen only where it helps.
    let headroom = clamp(lmin / max(lmax, 1e-4), 0.0, 1.0);

    // Lobe amplitude — bigger at edges (low headroom), smaller in
    // flat areas (high headroom). 0.125 cap keeps the kernel stable.
    let lobe = 0.125 * strength * (1.0 - headroom);

    // Negative-lobe FIR: center*(1+4*lobe) - lobe*(n+s+w+e).
    // Coefficients sum to 1 → DC preserved.
    let sharpened = c * (1.0 + 4.0 * lobe) - lobe * (n + s + w + e);

    return vec4<f32>(max(sharpened, vec3<f32>(0.0)), center.a);
}
";
