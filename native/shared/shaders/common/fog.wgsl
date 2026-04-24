// Distance fog — standard exponential-squared falloff.
//
// Depends on material_abi.wgsl for view.fog (rgb + density).

// Apply fog to a lit fragment. `view_distance` is the world-space
// distance from the camera to the fragment. `color` is the lit HDR
// colour before fog.
fn apply_fog(color: vec3<f32>, view_distance: f32) -> vec3<f32> {
  let density = view.fog.w;
  if (density <= 0.0) { return color; }
  // exp² is tighter near the camera and softer at distance than a plain
  // exp — matches what most atmospherics use.
  let f = exp(-density * view_distance * density * view_distance);
  let factor = clamp(f, 0.0, 1.0);
  return mix(view.fog.rgb, color, factor);
}
