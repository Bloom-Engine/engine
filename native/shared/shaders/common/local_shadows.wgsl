// Shadows from local (point) lights for material shaders.
//
// Include this after material_abi.wgsl and multiply a point light's
// contribution by `local_light_shadow(i, world_pos, normal)`, where `i` is the
// light's index in `view.point_lights`:
//
//   for (var i = 0u; i < u32(view.point_light_count.x); i = i + 1u) {
//     ...
//     lit += colour * falloff * ndl * local_light_shadow(i, world_pos, n);
//   }
//
// This file is the version for renderers that do not draw local-light shadow
// maps (mobile, web, lower tiers, or the feature left off): every light reads
// as unshadowed and no binding is added. Where the renderer does draw them it
// substitutes shaders/local_shadows/material_helper.wgsl under this same
// include name, so a material is written once.
fn local_light_shadow(light_index: u32, world_pos: vec3<f32>, geo_n: vec3<f32>) -> f32 {
  return 1.0;
}
