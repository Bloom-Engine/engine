// bloom_postfx.metal — Post-processing shaders for watchOS.
//
// Compiled into the .app as default.metallib by Perry's metal_sources
// pipeline. SwiftUI's .colorEffect(Shader(...)) / .layerEffect(Shader(...))
// look them up by name via ShaderLibrary.default.
//
// Each entry is [[ stitchable ]] so SwiftUI's shader stitcher can inline
// the function into its pipeline.

#include <metal_stdlib>
#include <SwiftUI/SwiftUI_Metal.h>

using namespace metal;

// ---------- Film grain ----------
//
// Adds time-animated noise to each pixel. Noise is a cheap hash of
// (position, time) — not cryptographic, but visually indistinguishable from
// proper noise at grain strengths < 0.3.
//
// Args: strength (0..1), time (seconds)
[[ stitchable ]]
half4 bloom_film_grain(float2 position, half4 color, float strength, float time) {
    float2 p = position + float2(time * 13.1, time * 7.3);
    float n = fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    half g = half((n - 0.5) * strength);
    return half4(color.rgb + g, color.a);
}

// ---------- Chromatic aberration ----------
//
// Samples the layer at slightly offset positions per RGB channel, with the
// offset growing quadratically with distance from screen center (so it's
// subtle in the middle and pronounced in the corners — matches real lens
// behavior). Alpha carried through from the green channel.
//
// Args: strength (px at corner), size (screen width, height)
[[ stitchable ]]
half4 bloom_chromatic_aberration(float2 position, SwiftUI::Layer layer,
                                 float strength, float2 size) {
    float2 center = size * 0.5;
    float2 radial = (position - center) / center;
    float dist = dot(radial, radial);  // 0 at center, 1 near corner
    float2 offset = radial * (strength * dist);
    half r = layer.sample(position + offset).r;
    half4 g = layer.sample(position);
    half b = layer.sample(position - offset).b;
    return half4(r, g.g, b, g.a);
}

// ---------- Sun shafts ----------
//
// Radial blur from a "sun" screen position — cheap god-ray fake. Samples
// along the ray from `sun` toward `position`, accumulating with a decay
// factor so closer samples contribute more. Tinted and scaled by strength.
//
// Args: sun (screen-space position), strength (0..1), decay (0..1, typical 0.85),
//       tint (half3 RGB, 0..1)
[[ stitchable ]]
half4 bloom_sun_shafts(float2 position, SwiftUI::Layer layer,
                      float2 sun, float strength, float decay, half3 tint) {
    const int STEPS = 12;
    float2 delta = (position - sun) / float(STEPS);
    float w = 1.0;
    half4 acc = half4(0.0);
    for (int i = 0; i < STEPS; i++) {
        acc += layer.sample(position - delta * float(i)) * half(w);
        w *= decay;
    }
    half4 base = layer.sample(position);
    half3 shaft = (acc.rgb / half(STEPS)) * tint * half(strength);
    return half4(base.rgb + shaft, base.a);
}
