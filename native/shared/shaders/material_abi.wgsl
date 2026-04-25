// Bloom shader ABI header — see docs/rfc/0001-material-render-graph.md.
//
// ABI-VERSION: 1
//
// Included by every 3D shader (built-in PBR, sky, custom user materials).
// Defines the five bind groups and the vertex attribute layout that any
// Bloom-compatible 3D pipeline targets. Groups 0, 1, 3 are required;
// group 2 is required for PBR materials and optional otherwise; group 4
// is opt-in via the material descriptor's `reads_scene = true`.
//
// Bumping the version means every consumer has to be updated in the same
// PR. Version-check happens in engine/renderer/shaders.rs at pipeline
// creation time: mismatched headers are a hard error, not a warning.

// =====================================================================
// Group 0 — PerFrame (written once per frame, read by every 3D draw)
// =====================================================================

struct PerFrame {
  // Clock / counters
  time:              f32,   // seconds since process start, wraps every ~2²³ s
  delta_time:        f32,   // seconds since previous frame
  frame_index:       u32,   // monotonic, wraps at u32::MAX
  _pad0:             u32,

  // Resolutions
  screen_resolution: vec2<f32>, // physical pixels (matches swapchain)
  render_resolution: vec2<f32>, // may be smaller than screen when TSR < 1

  // TAA / TSR jitter applied to the projection this frame
  taa_jitter:        vec2<f32>,
  _pad1:             vec2<f32>,

  // Global wind field. xy = direction in the XZ plane (need not be
  // normalised; magnitude scales effective amplitude); z = amplitude
  // (~0.1 m typical for grass); w = frequency in Hz (~1.0 typical).
  // Foliage / cloth materials sample this in their vertex stage.
  wind:              vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: PerFrame;

// =====================================================================
// Group 1 — PerView (camera + lighting + env maps + shadows)
// =====================================================================

struct DirLight {
  direction: vec4<f32>,  // xyz + intensity
  color:     vec4<f32>,  // rgb + _pad
};

struct PointLight {
  position:  vec4<f32>,  // xyz + range
  color:     vec4<f32>,  // rgb + intensity
};

struct PerView {
  // View matrices
  view:           mat4x4<f32>,
  proj:           mat4x4<f32>,
  view_proj:      mat4x4<f32>,
  prev_view_proj: mat4x4<f32>,  // for motion vectors / TAA reprojection
  inv_proj:       mat4x4<f32>,

  // Camera state
  camera_pos:     vec4<f32>,    // xyz=world pos, w=env_intensity
  camera_dir:     vec4<f32>,    // xyz=forward, w=fovy_rad

  // Basic scene lighting
  ambient:        vec4<f32>,    // rgb + intensity
  fog:            vec4<f32>,    // rgb + density

  // Legacy single-sun shortcut (also in dir_lights[0])
  sun_dir:        vec4<f32>,    // xyz + intensity
  sun_color:      vec4<f32>,    // rgb + 0

  // Arrayed lights
  dir_light_count:   vec4<f32>, // x = count, yzw = 0
  dir_lights:        array<DirLight,  4>,
  point_light_count: vec4<f32>,
  point_lights:     array<PointLight, 16>,

  // Shadow data — three cascades, split by shadow_splits.xyz
  shadow_splits:   vec4<f32>,
  shadow_view:     mat4x4<f32>,                   // orientation for cascade compute
  shadow_cascades: array<mat4x4<f32>, 3>,         // light-space view-proj per cascade
};

@group(1) @binding(0) var<uniform> view: PerView;

// Environment maps — pre-filtered at load time.
@group(1) @binding(1) var env_tex:          texture_2d<f32>;
@group(1) @binding(2) var env_samp:         sampler;
@group(1) @binding(3) var env_diffuse_tex:  texture_2d<f32>;

// BRDF LUT for IBL specular (split-sum approximation).
@group(1) @binding(4) var brdf_lut_tex:     texture_2d<f32>;
@group(1) @binding(5) var brdf_lut_samp:    sampler;

// Shadow depth textures, one per cascade.
@group(1) @binding(6) var shadow_tex_0:     texture_depth_2d;
@group(1) @binding(7) var shadow_tex_1:     texture_depth_2d;
@group(1) @binding(8) var shadow_tex_2:     texture_depth_2d;
@group(1) @binding(9) var shadow_samp:      sampler_comparison;

// =====================================================================
// Group 2 — PerMaterial (PBR textures + factors + user params)
// =====================================================================
//
// All texture bindings have a 1×1 white default bound when a material
// doesn't provide its own — shaders can unconditionally sample.

struct MaterialFactors {
  metal_rough: vec4<f32>,  // x=metallic, y=roughness, z=has_mr_tex, w=alpha_cutoff
  emissive:    vec4<f32>,  // rgb + 0
  base_color:  vec4<f32>,  // rgba tint multiplier
  _reserved:   vec4<f32>,
};

@group(2) @binding(0)  var base_color_tex:   texture_2d<f32>;
@group(2) @binding(1)  var base_color_samp:  sampler;
@group(2) @binding(2)  var normal_tex:       texture_2d<f32>;
@group(2) @binding(3)  var normal_samp:      sampler;
@group(2) @binding(4)  var mr_tex:           texture_2d<f32>;
@group(2) @binding(5)  var mr_samp:          sampler;
@group(2) @binding(6)  var em_tex:           texture_2d<f32>;
@group(2) @binding(7)  var em_samp:          sampler;
@group(2) @binding(8)  var occ_tex:          texture_2d<f32>;
@group(2) @binding(9)  var occ_samp:         sampler;
@group(2) @binding(10) var<uniform> material: MaterialFactors;

// Binding 11 is reserved for per-material user params. Shaders that
// declare user params do so via:
//
//   struct UserParams { … up to 256 bytes … };
//   @group(2) @binding(11) var<uniform> user: UserParams;
//
// Materials that don't use it leave binding 11 unclaimed; the engine's
// pipeline layout allocates a 16-byte stub UBO there so the bind group
// always matches the layout.

// =====================================================================
// Group 3 — PerDraw (transform + skinning)
// =====================================================================

struct PerDraw {
  mvp:        mat4x4<f32>,
  model:      mat4x4<f32>,
  prev_mvp:   mat4x4<f32>,  // for motion vectors
  model_tint: vec4<f32>,
  skin_info:  vec4<u32>,    // x=joint_offset into global buffer, y=joint_count, zw=0
};

struct JointMatrices {
  matrices: array<mat4x4<f32>, 1024>,  // global joint buffer; draws read slices by offset
};

@group(3) @binding(0) var<uniform> draw:   PerDraw;
@group(3) @binding(1) var<uniform> joints: JointMatrices;

// =====================================================================
// Group 4 — SceneInputs (optional; `reads_scene = true` materials only)
// =====================================================================
//
// The render graph synthesises a SceneColor snapshot pass before the
// first consumer runs. Materials that don't declare reads_scene never
// receive this group and pay no cost for the copy.

@group(4) @binding(0) var scene_color_tex:  texture_2d<f32>;
@group(4) @binding(1) var scene_color_samp: sampler;
@group(4) @binding(2) var scene_depth_tex:  texture_depth_2d;
@group(4) @binding(3) var scene_depth_samp: sampler;
@group(4) @binding(4) var impulse_tex:      texture_2d<f32>;  // world-space decals / splashes
@group(4) @binding(5) var impulse_samp:     sampler;
@group(4) @binding(6) var motion_vectors:   texture_2d<f32>;

// =====================================================================
// Vertex attribute layout (matches renderer::types::Vertex3D)
// =====================================================================

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
  @location(2) color:    vec4<f32>,   // per-vertex tint multiplier
  @location(3) uv:       vec2<f32>,
  @location(4) joints:   vec4<f32>,   // bone indices as floats (engine convention)
  @location(5) weights:  vec4<f32>,   // sum to 1.0 for skinned, 0 for unskinned
  @location(6) tangent:  vec4<f32>,   // xyz=tangent, w=bitangent sign (+1 / -1)
};

// =====================================================================
// Fragment output profiles
// =====================================================================

// Opaque profile — writes the full forward G-buffer. Materials that use
// this must live in `Bucket::Opaque` and render in the main_hdr pass.
struct OpaqueOut {
  @location(0) hdr:      vec4<f32>,  // HDR colour, pre-bloom
  @location(1) material: vec2<f32>,  // metallic, roughness for SSR / SSGI
  @location(2) velocity: vec2<f32>,  // NDC-space motion, for TAA reprojection
  @location(3) albedo:   vec4<f32>,  // for SSGI bounce colour
};

// Translucent profile — writes HDR only, alpha blended. Materials in
// `Bucket::Transparent`, `Refractive`, or `Additive` use this. The main
// HDR pass' G-buffer targets are load-op'd in the translucent sub-pass;
// material/velocity/albedo remain whatever the opaque pass left them.
struct TranslucentOut {
  @location(0) hdr: vec4<f32>,
};

// =====================================================================
// Standard helpers
// =====================================================================

// Skinning — blends a world-space position through the joint matrices,
// using the per-draw joint_offset so multiple skinned models share the
// global joint buffer.
fn world_position_from_skinned(
  local_pos: vec3<f32>, vertex: VertexInput, include_non_skinned: bool,
) -> vec4<f32> {
  let total_weight = vertex.weights.x + vertex.weights.y
                   + vertex.weights.z + vertex.weights.w;
  var pos = vec4<f32>(local_pos, 1.0);
  if (total_weight > 0.01) {
    let off = draw.skin_info.x;
    let j0 = off + u32(vertex.joints.x);
    let j1 = off + u32(vertex.joints.y);
    let j2 = off + u32(vertex.joints.z);
    let j3 = off + u32(vertex.joints.w);
    pos = joints.matrices[j0] * pos * vertex.weights.x
        + joints.matrices[j1] * pos * vertex.weights.y
        + joints.matrices[j2] * pos * vertex.weights.z
        + joints.matrices[j3] * pos * vertex.weights.w;
  } else if (include_non_skinned) {
    pos = draw.model * pos;
  }
  return pos;
}

// Standard NDC-space motion vector computation. Pass the current and
// previous clip positions, get back a vec2 suitable for the
// velocity G-buffer target.
fn compute_motion_vector(curr_clip: vec4<f32>, prev_clip: vec4<f32>) -> vec2<f32> {
  let curr_ndc = curr_clip.xy / curr_clip.w;
  let prev_ndc = prev_clip.xy / prev_clip.w;
  return (curr_ndc - prev_ndc) * 0.5;
}

// Linearise a hardware depth value given the bound projection. Useful
// for materials that read `scene_depth_tex` (group 4 binding 2).
fn linearize_depth(z_ndc: f32) -> f32 {
  // Standard reverse-depth linearisation: the engine uses a standard
  // depth range (0..1) with proj.z mapping near→0, far→1.
  // Derived from the inv_proj mat4; this cheap form works for our
  // perspective projections. Override in shaders that don't match.
  let z_clip = z_ndc * 2.0 - 1.0;
  let a = view.proj[2][2];
  let b = view.proj[3][2];
  // z_eye = b / (z_clip + a). Result is negative (camera looks down -Z).
  return -b / (z_clip + a);
}
