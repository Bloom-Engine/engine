// Shadows from local (point) lights for material shaders: the sampling
// version. Substituted for common/local_shadows.wgsl where the renderer draws
// local-light shadow maps (renderer/local_light_shadows.rs holds the other
// half of every number here).
//
// A shadowed light owns six layers of `local_shadow_maps`, one per cube face,
// each a perspective depth map seen from the light. `local_shadows.light_slot`
// says, for every index of `view.point_lights`, which slot (if any) holds that
// light's maps this frame.

struct LocalLightShadowSlot {
  face_vps: array<mat4x4<f32>, 6>,
  // x = world size of one map texel per metre of distance from the light,
  // y = strength (0 = unshadowed, 1 = full; a new slot fades in),
  // z = first layer of this slot, w = 1 / map resolution.
  params: vec4<f32>,
};

struct LocalLightShadowParams {
  // x = number of lights with a shadow this frame (0 = nothing to sample).
  words: vec4<u32>,
  // x = slot + 1 for a shadowed light, 0 for an ordinary one.
  light_slot: array<vec4<u32>, 256>,
  slots: array<LocalLightShadowSlot, 4>,
};

@group(1) @binding(13) var local_shadow_maps: texture_depth_2d_array;
@group(1) @binding(14) var<uniform> local_shadows: LocalLightShadowParams;

// Same face order as the maps are drawn in: +X, -X, +Y, -Y, +Z, -Z.
fn local_light_shadow_face(direction: vec3<f32>) -> u32 {
  let a = abs(direction);
  if (a.x >= a.y && a.x >= a.z) {
    return select(1u, 0u, direction.x >= 0.0);
  }
  if (a.y >= a.z) {
    return select(3u, 2u, direction.y >= 0.0);
  }
  return select(5u, 4u, direction.z >= 0.0);
}

// 1 = lit, 0 = in shadow. `geo_n` is the surface normal (it need not be unit
// length); the receiver is moved along it and toward the light by about a map
// texel before the lookup, which is what keeps a wall from shadowing itself
// without detaching a caster's shadow from its feet.
fn local_light_shadow(light_index: u32, world_pos: vec3<f32>, geo_n: vec3<f32>) -> f32 {
  if (local_shadows.words.x == 0u) {
    return 1.0;
  }
  let entry = local_shadows.light_slot[light_index].x;
  if (entry == 0u) {
    return 1.0;
  }
  let slot = entry - 1u;
  let params = local_shadows.slots[slot].params;
  let light_pos = view.point_lights[light_index].position.xyz;
  let from_light = world_pos - light_pos;
  let dist = length(from_light);
  let to_light = -from_light / max(dist, 0.0001);
  let n = normalize(geo_n);
  let ndl = clamp(dot(n, to_light), 0.0, 1.0);
  let texel_ws = dist * params.x;
  let receiver = world_pos
               + n * texel_ws * (1.0 + 2.0 * (1.0 - ndl))
               + to_light * texel_ws * 1.5;

  let face = local_light_shadow_face(receiver - light_pos);
  let clip = local_shadows.slots[slot].face_vps[face] * vec4<f32>(receiver, 1.0);
  let ndc = clip.xyz / clip.w;
  // Past the light's range (the far plane) the light adds nothing anyway.
  if (clip.w <= 0.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
  let layer = i32(params.z) + i32(face);
  let texel = params.w;
  var offs = array<vec2<f32>, 4>(
    vec2<f32>(-0.75, -0.25), vec2<f32>( 0.25, -0.75),
    vec2<f32>(-0.25,  0.75), vec2<f32>( 0.75,  0.25),
  );
  var lit = 0.0;
  for (var i = 0; i < 4; i = i + 1) {
    lit += textureSampleCompareLevel(
      local_shadow_maps, shadow_samp, uv + offs[i] * texel, layer, ndc.z,
    );
  }
  return mix(1.0, lit * 0.25, params.y);
}
