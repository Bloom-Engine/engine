// PBR common — GGX specular, Lambert diffuse, split-sum IBL.
//
// Depends on material_abi.wgsl (requires group 1 env + brdf_lut, group 2
// base/normal/mr/em/occ textures + material factors).

const PI: f32 = 3.14159265358979;

// =====================================================================
// Microfacet BRDF components
// =====================================================================

// Trowbridge-Reitz / GGX normal distribution.
fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
  let a  = roughness * roughness;
  let a2 = a * a;
  let nh2 = n_dot_h * n_dot_h;
  let denom = nh2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry term with combined view + light masking.
fn g_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
  let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
  return gv * gl;
}

// Schlick's Fresnel approximation.
fn f_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

// Fresnel with a roughness term for IBL — fades the specular boost at
// grazing angles on rough surfaces, avoiding the "everything reflects"
// artifact.
fn f_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
  let inv_rough = vec3<f32>(1.0 - roughness);
  return f0 + (max(inv_rough, f0) - f0) * pow(1.0 - cos_theta, 5.0);
}

// =====================================================================
// Direct lighting — one directional or point light contribution
// =====================================================================

fn brdf_direct(
  n: vec3<f32>, v: vec3<f32>, l: vec3<f32>,
  albedo: vec3<f32>, metallic: f32, roughness: f32,
) -> vec3<f32> {
  let h = normalize(v + l);
  let n_dot_v = max(dot(n, v), 0.0001);
  let n_dot_l = max(dot(n, l), 0.0);
  let n_dot_h = max(dot(n, h), 0.0);
  let v_dot_h = max(dot(v, h), 0.0);

  // Dielectrics get F0 = 0.04; metals tint the reflection by the albedo.
  let f0 = mix(vec3<f32>(0.04), albedo, metallic);

  let d = d_ggx(n_dot_h, roughness);
  let g = g_smith(n_dot_v, n_dot_l, roughness);
  let f = f_schlick(v_dot_h, f0);

  let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
  let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
  let diffuse = kd * albedo / PI;

  return (diffuse + specular) * n_dot_l;
}

// =====================================================================
// IBL — split-sum approximation
// =====================================================================
//
// env_diffuse_tex is a pre-convolved irradiance map sampled with the
// surface normal. env_tex is the original equirect HDR, sampled with a
// reflection vector and a roughness-derived mip bias. brdf_lut_tex is
// the pre-computed 2D LUT keyed on (n·v, roughness).
//
// Helper sampling functions assume equirectangular layout, matching the
// existing engine convention.

fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
  let d = normalize(dir);
  let theta = acos(clamp(d.y, -1.0, 1.0));
  let phi   = atan2(d.z, d.x);
  let u     = phi / (2.0 * PI);
  return vec2<f32>(u - floor(u), theta / PI);
}

fn seamless_equirect_uv(uv: vec2<f32>) -> vec2<f32> {
  let tex_w = f32(textureDimensions(env_tex, 0).x);
  let half_texel = 0.5 / tex_w;
  return vec2<f32>(clamp(uv.x, half_texel, 1.0 - half_texel), uv.y);
}

fn sample_env(dir: vec3<f32>, lod: f32) -> vec3<f32> {
  return textureSampleLevel(env_tex, env_samp,
                            seamless_equirect_uv(dir_to_equirect_uv(dir)),
                            lod).rgb * view.camera_pos.w;
}

fn sample_env_diffuse(normal: vec3<f32>) -> vec3<f32> {
  return textureSample(env_diffuse_tex, env_samp,
                       dir_to_equirect_uv(normal)).rgb * view.camera_pos.w;
}

fn ibl(
  n: vec3<f32>, v: vec3<f32>,
  albedo: vec3<f32>, metallic: f32, roughness: f32, occlusion: f32,
) -> vec3<f32> {
  let r = reflect(-v, n);
  let n_dot_v = max(dot(n, v), 0.0001);
  let f0 = mix(vec3<f32>(0.04), albedo, metallic);
  let f  = f_schlick_roughness(n_dot_v, f0, roughness);

  // Diffuse IBL — convolved irradiance.
  let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
  let irr = sample_env_diffuse(n);
  let diffuse = kd * albedo * irr;

  // Specular IBL — roughness-weighted mip (approximation).
  // Real split-sum pre-filters per mip; we use a simple lod bias for
  // now. The engine can swap in a proper pre-filtered chain later.
  let lod = roughness * 6.0;
  let env = sample_env(r, lod);
  let brdf = textureSample(brdf_lut_tex, brdf_lut_samp,
                           vec2<f32>(n_dot_v, roughness)).rg;
  let specular = env * (f * brdf.x + brdf.y);

  return (diffuse + specular) * occlusion;
}
