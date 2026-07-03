// Cascaded shadow maps — 3-cascade PCF sampling for the material path.
//
// Depends on material_abi.wgsl for PerView.shadow_cascades, shadow_splits,
// camera_pos, and the three shadow_tex_N depth samplers.
//
// Cascade selection uses world-space DISTANCE from the camera (not view-space
// depth) so that merely rotating the camera doesn't reshuffle which cascade a
// surface falls in — that view-dependent reshuffle made terrain/foliage
// shadows visibly swim as the camera turned. Cross-cascade blending at the
// split boundaries removes the hard seam that otherwise slides across the
// ground with the camera. This mirrors the deferred core path
// (renderer/shaders/core.rs::sample_shadow), so the material-shaded surfaces
// (terrain, grass, trees, water) match the rest of the scene.

// Sample a single cascade with a 4-tap rotated-grid PCF kernel. The kernel
// softens the edge a touch and hides per-texel crawl without the cost of a
// full Poisson disk.
fn sample_shadow_cascade(
  cascade_idx: u32, world_pos: vec3<f32>,
) -> f32 {
  let light_clip = view.shadow_cascades[cascade_idx]
                 * vec4<f32>(world_pos, 1.0);
  let light_ndc = light_clip.xyz / light_clip.w;

  // Outside the cascade's frustum → treat as lit.
  if (abs(light_ndc.x) > 1.0 || abs(light_ndc.y) > 1.0
      || light_ndc.z < 0.0  || light_ndc.z > 1.0) {
    return 1.0;
  }

  let uv = vec2<f32>(
    light_ndc.x * 0.5 + 0.5,
    1.0 - (light_ndc.y * 0.5 + 0.5),
  );
  // Slight depth bias to avoid surface acne.
  let ref_depth = light_ndc.z - 0.001;

  var dims: vec2<u32>;
  switch (cascade_idx) {
    case 0u: { dims = textureDimensions(shadow_tex_0); }
    case 1u: { dims = textureDimensions(shadow_tex_1); }
    default: { dims = textureDimensions(shadow_tex_2); }
  }
  let texel = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));

  var offs = array<vec2<f32>, 4>(
    vec2<f32>(-0.5, -0.5), vec2<f32>( 0.5, -0.5),
    vec2<f32>(-0.5,  0.5), vec2<f32>( 0.5,  0.5),
  );
  var result: f32 = 0.0;
  for (var i: i32 = 0; i < 4; i = i + 1) {
    let suv = uv + offs[i] * texel;
    // wgpu doesn't let us index depth-texture arrays yet — branch.
    switch (cascade_idx) {
      case 0u: { result += textureSampleCompareLevel(shadow_tex_0, shadow_samp, suv, ref_depth); }
      case 1u: { result += textureSampleCompareLevel(shadow_tex_1, shadow_samp, suv, ref_depth); }
      default: { result += textureSampleCompareLevel(shadow_tex_2, shadow_samp, suv, ref_depth); }
    }
  }
  return result * 0.25;
}

// Game-shader entry point. Picks a cascade by rotation-independent world-space
// distance, blends across the boundary, and returns a shadow factor in [0, 1]
// (1 = fully lit, 0 = fully shadowed). Use this from custom materials (grass,
// tree, terrain, water) that want to receive the directional sun shadow with
// one line. Requires `view` (PerView) to be in scope — any shader that
// includes material_abi.wgsl already has it.
fn sample_sun_shadow(world_pos: vec3<f32>) -> f32 {
  let cam = view.camera_pos.xyz;
  let dist = length(world_pos - cam);

  var cascade = 2u;
  if (dist <= view.shadow_splits.x) {
    cascade = 0u;
  } else if (dist <= view.shadow_splits.y) {
    cascade = 1u;
  }

  let shadow_val = sample_shadow_cascade(cascade, world_pos);

  // Blend into the next cascade over the last 10% of this cascade's range so
  // the transition is a soft gradient rather than a hard line that the camera
  // drags across the scene.
  var split_near = 0.0;
  var split_far = view.shadow_splits.x;
  if (cascade == 1u) {
    split_near = view.shadow_splits.x;
    split_far  = view.shadow_splits.y;
  } else if (cascade == 2u) {
    split_near = view.shadow_splits.y;
    split_far  = view.shadow_splits.z;
  }
  let blend_zone = (split_far - split_near) * 0.1;
  let dist_to_edge = split_far - dist;
  if (cascade < 2u && dist_to_edge < blend_zone) {
    let next_val = sample_shadow_cascade(cascade + 1u, world_pos);
    let t = clamp(dist_to_edge / blend_zone, 0.0, 1.0);
    return mix(next_val, shadow_val, t);
  }
  return shadow_val;
}

// ---- Back-compat shims -----------------------------------------------------
// Older callers selected a cascade from view-space depth then sampled it.
// Cascade selection now lives inside sample_sun_shadow (world-space distance),
// so these just forward — kept so any shader still including the old names
// keeps compiling.
fn select_cascade(view_space_depth: f32) -> u32 {
  let d = abs(view_space_depth);
  if (d < view.shadow_splits.x) { return 0u; }
  if (d < view.shadow_splits.y) { return 1u; }
  return 2u;
}

fn shadow(world_pos: vec3<f32>, view_space_depth: f32) -> f32 {
  return sample_sun_shadow(world_pos);
}
