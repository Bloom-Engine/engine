// Cascaded shadow maps — 3-cascade PCF sampling for the material path.
//
// Depends on material_abi.wgsl for PerView.shadow_cascades, shadow_splits,
// camera_pos, and the three shadow_tex_N depth samplers.
//
// Cascade selection uses the same positive view-space depth as the camera
// frustum slices fitted by ShadowMap::compute_cascade_vps. A spherical-distance
// selector can choose a tight cascade whose fitted XY footprint does not cover
// a receiver near the side of the camera, truncating otherwise valid shadows
// when the camera turns. Cross-cascade blending hides the remaining resolution
// transition at split boundaries. This mirrors the deferred core path
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

fn shadow_view_depth(world_pos: vec3<f32>) -> f32 {
  let view_pos = view.shadow_view * vec4<f32>(world_pos, 1.0);
  return max(-view_pos.z, 0.0);
}

// Game-shader entry point. Picks the fitted view-depth cascade, blends across
// the boundary, and returns a shadow factor in [0, 1]
// (1 = fully lit, 0 = fully shadowed). Use this from custom materials (grass,
// tree, terrain, water) that want to receive the directional sun shadow with
// one line. Requires `view` (PerView) to be in scope — any shader that
// includes material_abi.wgsl already has it.
fn sample_sun_shadow(world_pos: vec3<f32>) -> f32 {
  // dir_light_count.y carries the shadows-enabled flag (mirrors the core
  // path's sample_shadow gate; shadow_splits.w is the TSR mip-LOD bias, so
  // it can't double as a flag): disabled shadows must read fully lit, not
  // project through stale cascade VPs whose garbage NDC reads as occluded.
  if (view.dir_light_count.y < 0.5) {
    return 1.0;
  }
  let view_depth = shadow_view_depth(world_pos);

  var cascade = 2u;
  if (view_depth <= view.shadow_splits.x) {
    cascade = 0u;
  } else if (view_depth <= view.shadow_splits.y) {
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
  let dist_to_edge = split_far - view_depth;
  if (cascade < 2u && dist_to_edge < blend_zone) {
    let next_val = sample_shadow_cascade(cascade + 1u, world_pos);
    let t = clamp(dist_to_edge / blend_zone, 0.0, 1.0);
    return mix(next_val, shadow_val, t);
  }
  return shadow_val;
}

// Normal-offset variant — use this when the surface normal is available
// (it always is in a lit material shader). Offsets the receiver position
// along the geometric normal by ~1.5 cascade texels before projecting,
// which kills the full-face acne a constant depth bias can't cover on
// steep slopes and vertical walls (a wall at 40° sun incidence spans
// several texels of depth across one shadow texel — the deferred core
// path learned this the hard way; see core.rs::sample_shadow). Now that
// material-path geometry (building, terrain, trees) casts into the
// cascades, every material surface is a potential self-receiver and
// needs the same treatment.
fn sample_sun_shadow_n(world_pos: vec3<f32>, geo_n: vec3<f32>) -> f32 {
  if (view.dir_light_count.y < 0.5) {
    return 1.0;
  }
  let view_depth = shadow_view_depth(world_pos);
  var cascade = 2u;
  if (view_depth <= view.shadow_splits.x) {
    cascade = 0u;
  } else if (view_depth <= view.shadow_splits.y) {
    cascade = 1u;
  }
  // World-space size of one shadow texel in this cascade: the ortho span
  // is ~2×split distance, mapped across the cascade map's width.
  // textureDimensions keeps this honest if CASCADE_MAP_SIZE changes.
  //
  // Near a split boundary sample_sun_shadow blends in the NEXT cascade,
  // whose texels are larger — an offset sized for this cascade
  // under-biases that sample and paints acne bands exactly in the blend
  // zones (the deferred core path hit the same thing). Size the offset
  // by the next cascade's texel when inside its blend zone.
  var split_far = view.shadow_splits.z;
  if (cascade == 0u) { split_far = view.shadow_splits.x; }
  else if (cascade == 1u) { split_far = view.shadow_splits.y; }
  var split_near = 0.0;
  if (cascade == 1u) { split_near = view.shadow_splits.x; }
  else if (cascade == 2u) { split_near = view.shadow_splits.y; }
  var eff_cascade = cascade;
  if (cascade < 2u && (split_far - view_depth) < (split_far - split_near) * 0.1) {
    eff_cascade = cascade + 1u;
    if (eff_cascade == 1u) { split_far = view.shadow_splits.y; }
    else { split_far = view.shadow_splits.z; }
  }
  var dims: vec2<u32>;
  switch (eff_cascade) {
    case 0u: { dims = textureDimensions(shadow_tex_0); }
    case 1u: { dims = textureDimensions(shadow_tex_1); }
    default: { dims = textureDimensions(shadow_tex_2); }
  }
  let texel_ws = (2.0 * split_far) / f32(dims.x);
  // Slope-adaptive: at grazing sun incidence one shadow texel spans many
  // times its footprint in receiver depth, so the fixed 1.5-texel offset
  // still stripes walls that run nearly parallel to the light and distant
  // hillsides in the coarse cascade. Grow the offset as n·l falls off
  // (up to ~4.5 texels when fully grazing).
  let ndl = clamp(dot(normalize(geo_n), normalize(view.sun_dir.xyz)), 0.0, 1.0);
  let slope_boost = 1.0 + 2.0 * (1.0 - ndl);
  let offset_pos = world_pos + normalize(geo_n) * texel_ws * 1.5 * slope_boost;
  return sample_sun_shadow(offset_pos);
}

// ---- Back-compat shims -----------------------------------------------------
// Older callers selected a cascade from view-space depth then sampled it.
// Cascade selection now lives inside sample_sun_shadow (using the same depth),
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
