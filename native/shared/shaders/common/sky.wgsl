// Sky sampling helpers — thin wrappers over pbr.wgsl's env sampler so
// the dedicated sky pipeline can stay short.

// Sample the HDR equirect env map along a world-space direction with
// no mip bias (sky pass doesn't want the roughness-based blur).
fn sample_sky(dir: vec3<f32>) -> vec3<f32> {
  return sample_env(dir, 0.0);
}
