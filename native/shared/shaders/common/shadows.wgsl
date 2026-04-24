// Cascaded shadow maps — 3-cascade PCF sampling.
//
// Depends on material_abi.wgsl for PerView.shadow_cascades,
// shadow_splits, and the three shadow_tex_N depth samplers.

// Pick a cascade by comparing world-space depth to the configured splits.
// Returns 0, 1, or 2.
fn select_cascade(view_space_depth: f32) -> u32 {
  let d = abs(view_space_depth);
  if (d < view.shadow_splits.x) { return 0u; }
  if (d < view.shadow_splits.y) { return 1u; }
  return 2u;
}

// Sample a single cascade with 2×2 PCF.
fn sample_shadow_cascade(
  cascade_idx: u32, world_pos: vec3<f32>,
) -> f32 {
  let light_clip = view.shadow_cascades[cascade_idx]
                 * vec4<f32>(world_pos, 1.0);
  let light_ndc = light_clip.xyz / light_clip.w;

  // Outside the cascade's frustum → not shadowed.
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

  // wgpu doesn't let us index texture_depth_2d arrays yet — branch.
  var result: f32 = 1.0;
  switch (cascade_idx) {
    case 0u: {
      result = textureSampleCompareLevel(shadow_tex_0, shadow_samp,
                                         uv, ref_depth);
      break;
    }
    case 1u: {
      result = textureSampleCompareLevel(shadow_tex_1, shadow_samp,
                                         uv, ref_depth);
      break;
    }
    default: {
      result = textureSampleCompareLevel(shadow_tex_2, shadow_samp,
                                         uv, ref_depth);
      break;
    }
  }
  return result;
}

// Full cascaded-shadow query. Pass in world position and the world-space
// depth from the camera's view (positive is "in front of camera").
// Returns a shadow factor in [0, 1]; 1 = lit, 0 = fully shadowed.
fn shadow(world_pos: vec3<f32>, view_space_depth: f32) -> f32 {
  let cascade = select_cascade(view_space_depth);
  return sample_shadow_cascade(cascade, world_pos);
}
