// Minimal ABI-compliant test material.
//
// Reads PerFrame.time and PerDraw.mvp to render the bound vertex
// geometry as a solid-colour translucent quad. Proves the compile
// path in material_pipeline.rs works end-to-end.

#include "material_abi.wgsl"

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
  var out: VsOut;
  let world = world_position_from_skinned(in.position, in, true);
  out.clip_position = draw.mvp * vec4<f32>(in.position, 1.0);
  // Animate a colour pulse by PerFrame.time so the test shader
  // demonstrably reads from group 0.
  let pulse = 0.5 + 0.5 * sin(frame.time);
  out.color = vec3<f32>(pulse, 0.4, 1.0 - pulse) * draw.model_tint.rgb;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 0.7);
}
