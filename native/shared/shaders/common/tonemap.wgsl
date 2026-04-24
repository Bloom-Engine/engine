// ACES filmic tonemapping — the Narkowicz fit, same one the composite
// pass uses today. Kept here as a shared helper so any future custom
// shader that wants its output already tonemapped (rare, usually
// post-FX territory) can reuse the same curve.

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
               vec3<f32>(0.0), vec3<f32>(1.0));
}

// Linear-to-sRGB for the rare case a shader needs to sample a texture
// stored as linear and output sRGB directly. The engine's composite
// pass already does this at display time, so most materials never call
// it.
fn linear_to_srgb(x: vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), x);
  let low    = x * 12.92;
  let high   = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
  return mix(low, high, cutoff);
}
