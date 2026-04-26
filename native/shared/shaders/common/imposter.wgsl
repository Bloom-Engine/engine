// Octahedral imposter helpers.
//
// The standard "octahedral imposter" technique encodes a hemisphere of
// view directions into a 2D atlas. A model is pre-rendered from N²
// directions on an octahedron (V1 expects N = 8 → 64 views in an 8×8
// grid). At runtime, the fragment shader picks the right grid cell
// based on the camera's direction relative to the imposter's position.
//
// Atlas convention:
//   - 8×8 grid (V1 fixed). 256 to 512 px per cell.
//   - RGBA8 (color + alpha). Future: split into albedo / normal / depth.
//   - Each cell renders the model facing in the encoded direction.
//
// Bake tooling: V1 expects games to use Blender's ScreenSpace
// imposter add-on, Unity's Tree Creator imposter mode, or any
// equivalent external tool. The engine bake tool ships in a follow-up.

const IMPOSTER_GRID: u32 = 8u;
const IMPOSTER_GRID_F: f32 = 8.0;

// Spherical → octahedron: unfold a unit sphere onto the [-1, 1] square.
fn octahedral_encode(dir: vec3<f32>) -> vec2<f32> {
  let n = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));
  if (n.y >= 0.0) {
    return vec2<f32>(n.x, n.z);
  }
  // Lower hemisphere: fold around.
  return vec2<f32>(
    (1.0 - abs(n.z)) * select(-1.0, 1.0, n.x >= 0.0),
    (1.0 - abs(n.x)) * select(-1.0, 1.0, n.z >= 0.0),
  );
}

// Octahedron → spherical (inverse of octahedral_encode).
fn octahedral_decode(uv: vec2<f32>) -> vec3<f32> {
  let abs_uv = abs(uv);
  let y = 1.0 - abs_uv.x - abs_uv.y;
  if (y >= 0.0) {
    return normalize(vec3<f32>(uv.x, y, uv.y));
  }
  let signed_uv = sign(uv);
  return normalize(vec3<f32>(
    signed_uv.x * (1.0 - abs_uv.y),
    y,
    signed_uv.y * (1.0 - abs_uv.x),
  ));
}

// Pick the grid cell + sub-cell UV for sampling an octahedral imposter
// atlas. `view_dir` is the direction from the imposter to the camera
// (world space, normalized). `local_uv` is the per-cell UV
// (0..1, the billboard quad's UV).
//
// Returns the atlas UV in [0, 1] mapping to the right cell + sub-cell.
fn imposter_atlas_uv(view_dir: vec3<f32>, local_uv: vec2<f32>) -> vec2<f32> {
  // Encode direction → octahedral UV in [-1, 1], remap to [0, 1].
  let oct = octahedral_encode(view_dir) * 0.5 + 0.5;
  // Snap to the nearest cell in the 8×8 grid.
  let cell = floor(oct * IMPOSTER_GRID_F);
  let cell_uv = (cell + clamp(local_uv, vec2<f32>(0.0), vec2<f32>(1.0))) / IMPOSTER_GRID_F;
  return cell_uv;
}

// Convenience: pick the UV cell directly from a view direction (ignoring
// per-cell sub-UV). Useful when sampling a pre-baked atlas where each
// cell is a flat texture and the billboard quad's per-vertex UV is
// already in [0, 1].
fn imposter_uv_for_view(view_dir: vec3<f32>) -> vec2<f32> {
  return imposter_atlas_uv(view_dir, vec2<f32>(0.5, 0.5));
}

// Standard billboard: build a screen-aligned quad facing the camera.
// `world_center` is the imposter's world-space anchor.
// `vertex_idx` is the @builtin(vertex_index) (0..3 for two triangles
// in a TriangleStrip, or 0..6 for a TriangleList).
// Returns the world-space vertex position + the per-vertex UV.
struct BillboardVertex {
  world_pos: vec3<f32>,
  uv:        vec2<f32>,
};

fn billboard_quad(
  world_center: vec3<f32>, scale: f32,
  camera_pos: vec3<f32>, camera_up: vec3<f32>,
  vertex_idx: u32,
) -> BillboardVertex {
  // 4 corners of a unit quad: (-,-), (+,-), (-,+), (+,+).
  // TriangleStrip indexing: 0,1,2,3 → 0=BL, 1=BR, 2=TL, 3=TR.
  let bl = vec2<f32>(-0.5, -0.5);
  let br = vec2<f32>( 0.5, -0.5);
  let tl = vec2<f32>(-0.5,  0.5);
  let tr = vec2<f32>( 0.5,  0.5);
  var corner: vec2<f32>;
  switch (vertex_idx) {
    case 0u: { corner = bl; }
    case 1u: { corner = br; }
    case 2u: { corner = tl; }
    default: { corner = tr; }
  }

  // View basis: forward = imposter → camera, right = up × forward, up = forward × right.
  let forward = normalize(camera_pos - world_center);
  let right   = normalize(cross(camera_up, forward));
  let up      = normalize(cross(forward, right));

  let world_pos = world_center + scale * (right * corner.x + up * corner.y);
  let uv        = corner + vec2<f32>(0.5, 0.5);   // 0..1
  var out: BillboardVertex;
  out.world_pos = world_pos;
  out.uv        = uv;
  return out;
}
