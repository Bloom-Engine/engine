// bloom_postfx.metal — Combined post-process pass for watchOS via SCNTechnique.
//
// Compiled into the .app's default.metallib by Perry's metal_sources
// pipeline. Loaded at runtime by BloomPostFXTechnique.swift, which builds
// an SCNTechnique that runs this pass after SceneKit's main color pass.
//
// Three effects (chromatic aberration, film grain, sun shafts) are chained
// in one fragment to avoid ping-pong target plumbing. Each effect reads its
// strength from the uniforms — strength 0 short-circuits to no-op so games
// only pay for what's enabled. Vignette + exposure stay on the SwiftUI side
// (cheaper, no shader needed).
//
// SCNTechnique on watchOS requires standard vertex+fragment entry points;
// `[[ stitchable ]]` (the SwiftUI Shader annotation) doesn't apply here.

#include <metal_stdlib>
using namespace metal;

struct PostFxVertOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle. SceneKit's DRAW_QUAD draw mode runs this with
// vertex_count = 3; the math expands to a 2:2 triangle whose rasterized
// region is exactly the viewport.
vertex PostFxVertOut bloom_postfx_vertex(uint vid [[vertex_id]]) {
    PostFxVertOut out;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    out.uv = uv;
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Uniforms — three SCNTechnique vec4 symbols bound at buffer 0/1/2.
// params0 = (chromatic aberration strength, film grain strength,
//            film grain time, sun shafts strength)
// params1 = (sun X UV, sun Y UV, sun decay, screen width px)
// params2 = (sun tint R, sun tint G, sun tint B, screen height px)
//
// SCNTechnique symbol-to-buffer binding maps each `vec4` symbol to a
// sequential `[[buffer(N)]]` slot in declaration order.
fragment half4 bloom_postfx_combined(
    PostFxVertOut in [[stage_in]],
    texture2d<half> color [[texture(0)]],
    constant float4 &params0 [[buffer(0)]],
    constant float4 &params1 [[buffer(1)]],
    constant float4 &params2 [[buffer(2)]]
) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);

    // Chromatic aberration — sample R/G/B at radial offset positions.
    half3 rgb;
    half a;
    float ca_strength = params0.x;
    if (ca_strength > 0.001) {
        float2 center = float2(0.5);
        float2 radial = in.uv - center;
        float dist = dot(radial, radial);  // 0 center → 0.5 corner
        float2 px_to_uv = 1.0 / float2(max(params1.w, 1.0), max(params2.w, 1.0));
        float2 offset = radial * (ca_strength * dist * 4.0) * px_to_uv;
        rgb.r = color.sample(s, in.uv + offset).r;
        half4 g = color.sample(s, in.uv);
        rgb.g = g.g;
        a = g.a;
        rgb.b = color.sample(s, in.uv - offset).b;
    } else {
        half4 c = color.sample(s, in.uv);
        rgb = c.rgb;
        a = c.a;
    }

    // Sun shafts — radial accumulation from a screen-space sun position.
    float sun_strength = params0.w;
    if (sun_strength > 0.001) {
        const int STEPS = 12;
        float2 sun = float2(params1.x, params1.y);
        float decay = params1.z;
        float2 delta = (in.uv - sun) / float(STEPS);
        half3 acc = half3(0.0);
        float w = 1.0;
        for (int i = 0; i < STEPS; i++) {
            acc += color.sample(s, in.uv - delta * float(i)).rgb * half(w);
            w *= decay;
        }
        half3 tint = half3(params2.x, params2.y, params2.z);
        rgb += (acc / half(STEPS)) * tint * half(sun_strength);
    }

    // Film grain — hash noise added per-pixel.
    float grain_strength = params0.y;
    if (grain_strength > 0.001) {
        float2 p = in.uv * 1024.0 + float2(params0.z * 13.1, params0.z * 7.3);
        float n = fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
        rgb += half((n - 0.5) * grain_strength);
    }

    return half4(rgb, a);
}
