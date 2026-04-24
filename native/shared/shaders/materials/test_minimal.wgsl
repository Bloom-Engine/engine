// Minimal ABI-compliant test material (opaque profile).
//
// Reads PerFrame.time and PerDraw.mvp to render the bound geometry
// with a solid time-pulsed colour. Writes the full 4-MRT G-buffer so
// the main HDR pass accepts it without a translucent sub-pass (that's
// phase 2's job). Proves the material pipeline + submission plumbing
// works end-to-end.

#include "material_abi.wgsl"

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) world_normal: vec3<f32>,
  @location(1) local_color:  vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
  var out: VsOut;
  out.clip_position = draw.mvp * vec4<f32>(in.position, 1.0);
  out.world_normal = normalize((draw.model * vec4<f32>(in.normal, 0.0)).xyz);

  // Pulse colour with PerFrame.time to demonstrate the binding.
  let pulse = 0.5 + 0.5 * sin(frame.time * 1.5);
  out.local_color = vec3<f32>(pulse, 0.4, 1.0 - pulse) * draw.model_tint.rgb;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> OpaqueOut {
  var out: OpaqueOut;
  // Simple directional-ish shading so the cube reads as 3D.
  let n = normalize(in.world_normal);
  let light_dir = normalize(vec3<f32>(0.3, 0.7, 0.4));
  let diffuse = max(dot(n, light_dir), 0.0);
  let lit = in.local_color * (0.3 + 0.7 * diffuse);
  out.hdr      = vec4<f32>(lit, 1.0);
  out.material = vec2<f32>(0.0, 0.9);            // non-metal, rough
  out.velocity = vec2<f32>(0.0, 0.0);
  out.albedo   = vec4<f32>(in.local_color, 1.0);
  return out;
}
