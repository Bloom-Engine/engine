// Phase 7 — impulse field update compute shader.
//
// Input:  `src` — last frame's R32Float field (world-space top-down).
//         `info` — world bounds, decay factor, queued splats.
// Output: `dst` — this frame's field.
//
// Per texel: value = min(previous * decay + sum(splat strength × (1-d/r)²), 1).
// The clamp keeps repeated splats at the same spot (a player wading
// through water submits one every few frames) from accumulating far
// past 1.0 — an unclamped field takes seconds of decay to drop back
// under 1.0, which reads as a stuck full-strength splat.

struct Splat {
  pos:      vec2<f32>,  // world xz
  radius:   f32,
  strength: f32,
};

struct Info {
  world_min:   vec2<f32>,
  world_size:  vec2<f32>,
  decay:       f32,
  _pad0:       f32,
  splat_count: u32,
  _pad1:       u32,
  splats:      array<Splat, 16>,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> info: Info;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(src);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  // Decay the previous field.
  let prev = textureLoad(src, vec2<i32>(gid.xy), 0).r;
  var value = prev * info.decay;

  // Convert the texel's world position.
  let uv    = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
  let world = info.world_min + uv * info.world_size;

  // Additively accumulate every queued splat.
  for (var i: u32 = 0u; i < info.splat_count; i = i + 1u) {
    let s = info.splats[i];
    let d = distance(world, s.pos);
    if (d < s.radius) {
      let falloff = 1.0 - d / s.radius;
      value = value + s.strength * falloff * falloff;
    }
  }

  textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(min(value, 1.0), 0.0, 0.0, 0.0));
}
