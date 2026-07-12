import { spawn, parallelMap } from 'perry/thread';
import { Color, Model, Vec3, Mat4, BoundingBox } from '../core/types';

// FFI declarations
declare function bloom_load_model(path: number): number;
declare function bloom_draw_model(handle: number, x: number, y: number, z: number, scale: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_model_rotated(handle: number, x: number, y: number, z: number, scale: number, rotY: number, colorPackedArgb: number): void;
declare function bloom_unload_model(handle: number): void;
declare function bloom_draw_cube(x: number, y: number, z: number, w: number, h: number, d: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_cube_wires(x: number, y: number, z: number, w: number, h: number, d: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_sphere(x: number, y: number, z: number, radius: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_sphere_wires(x: number, y: number, z: number, radius: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_cylinder(x: number, y: number, z: number, rt: number, rb: number, h: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_cylinder_ex(x: number, y: number, z: number, rt: number, rb: number, h: number, slices: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_plane(x: number, y: number, z: number, w: number, d: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_grid(slices: number, spacing: number): void;
declare function bloom_draw_ray(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, r: number, g: number, b: number, a: number): void;
declare function bloom_gen_mesh_cube(w: number, h: number, d: number): number;
declare function bloom_gen_mesh_heightmap(imageHandle: number, sizeX: number, sizeY: number, sizeZ: number): number;
declare function bloom_load_shader(source: number): number;
declare function bloom_compile_material(source: number): number;
declare function bloom_compile_material_refractive(source: number): number;
declare function bloom_compile_material_transparent(source: number): number;
declare function bloom_compile_material_additive(source: number): number;
declare function bloom_compile_material_cutout(source: number): number;
declare function bloom_compile_material_instanced(source: number): number;
declare function bloom_create_instance_buffer(dataPtr: any, instanceCount: number): number;
declare function bloom_create_instance_buffer_scratch(instanceCount: number): number;
declare function bloom_submit_material_draw_instanced(material: number, meshHandle: number, meshIdx: number, instanceBuffer: number, instanceCount: number): void;
declare function bloom_destroy_instance_buffer(handle: number): void;
declare function bloom_create_planar_reflection(planeY: number, normalX: number, normalY: number, normalZ: number, resolution: number): number;
declare function bloom_set_material_reflection_probe(material: number, probe: number): void;
declare function bloom_create_texture_array(dataPtr: any, dataLen: number, width: number, height: number, layerCount: number): number;
declare function bloom_create_texture_array_ex(dataPtr: any, dataLen: number, width: number, height: number, layerCount: number, format: number, mipLevels: number): number;
declare function bloom_set_material_texture_array(material: number, slot: number, array: number): void;
declare function bloom_set_material_shading_model(material: number, model: number): void;
declare function bloom_set_material_probe_visible(material: number, visible: number): void;
declare function bloom_set_material_foliage(material: number, transR: number, transG: number, transB: number, transAmount: number, wrapFactor: number): void;
declare function bloom_compile_material_from_file(path: number, bucketKind: number): number;
declare function bloom_set_material_params(handle: number, paramsPtr: any, paramCount: number): void;
declare function bloom_draw_material(material: number, meshHandle: number, meshIdx: number, x: number, y: number, z: number, scale: number, r: number, g: number, b: number, a: number): void;
declare function bloom_load_model_animation(path: number): number;
declare function bloom_update_model_animation(handle: number, animIndex: number, time: number, scale: number, px: number, py: number, pz: number, rotY: number): void;
declare function bloom_create_mesh(vertexPtr: number, vertexCount: number, indexPtr: number, indexCount: number): number;
declare function bloom_mesh_scratch_reset(): void;
declare function bloom_mesh_scratch_push_f32(v: number): void;
declare function bloom_mesh_scratch_push_u32(v: number): void;
declare function bloom_create_mesh_scratch(vertexCount: number, indexCount: number): number;
declare function bloom_set_ambient_light(r: number, g: number, b: number, intensity: number): void;
declare function bloom_set_directional_light(dx: number, dy: number, dz: number, r: number, g: number, b: number, intensity: number): void;
declare function bloom_set_procedural_sky(enabled: number, rayleighDensity: number, mieDensity: number, groundAlbedo: number): void;
declare function bloom_set_sun_direction(dx: number, dy: number, dz: number, intensity: number): void;
declare function bloom_gen_mesh_spline_ribbon(pointsPtr: number, pointCount: number, widthsPtr: number, widthCount: number): number;
declare function bloom_gen_mesh_spline_ribbon_scratch(pointCount: number, widthCount: number): number;
declare function bloom_get_model_mesh_count(handle: number): number;
declare function bloom_get_model_material_count(handle: number): number;
declare function bloom_get_model_bounds_min_x(handle: number): number;
declare function bloom_get_model_bounds_min_y(handle: number): number;
declare function bloom_get_model_bounds_min_z(handle: number): number;
declare function bloom_get_model_bounds_max_x(handle: number): number;
declare function bloom_get_model_bounds_max_y(handle: number): number;
declare function bloom_get_model_bounds_max_z(handle: number): number;

function makeModel(handle: number): Model {
  const mc = bloom_get_model_mesh_count(handle);
  const matc = bloom_get_model_material_count(handle);
  return { handle, meshCount: mc, materialCount: matc, transform: [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0] };
}

// OBJ parser (pure TypeScript)
function parseOBJ(text: string): { vertices: number[]; indices: number[] } | null {
  const positions: number[][] = [];
  const normals: number[][] = [];
  const texcoords: number[][] = [];
  const vertexMap = new Map<string, number>();
  const vertices: number[] = [];
  const indices: number[] = [];
  let vertexCount = 0;

  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length === 0 || line[0] === '#') continue;

    const parts = line.split(/\s+/);
    const cmd = parts[0];

    if (cmd === 'v' && parts.length >= 4) {
      positions.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
    } else if (cmd === 'vn' && parts.length >= 4) {
      normals.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
    } else if (cmd === 'vt' && parts.length >= 3) {
      texcoords.push([parseFloat(parts[1]), parseFloat(parts[2])]);
    } else if (cmd === 'f') {
      // Triangulate face (fan from first vertex)
      const faceIndices: number[] = [];
      for (let j = 1; j < parts.length; j++) {
        const key = parts[j];
        if (vertexMap.has(key)) {
          faceIndices.push(vertexMap.get(key)!);
        } else {
          const segs = key.split('/');
          const pi = parseInt(segs[0]) - 1;
          const ti = segs.length > 1 && segs[1] !== '' ? parseInt(segs[1]) - 1 : -1;
          const ni = segs.length > 2 ? parseInt(segs[2]) - 1 : -1;

          const pos = pi >= 0 && pi < positions.length ? positions[pi] : [0, 0, 0];
          const norm = ni >= 0 && ni < normals.length ? normals[ni] : [0, 1, 0];
          const uv = ti >= 0 && ti < texcoords.length ? texcoords[ti] : [0, 0];

          // Format: x,y,z, nx,ny,nz, r,g,b,a, u,v (12 floats per vertex)
          vertices.push(pos[0], pos[1], pos[2]);
          vertices.push(norm[0], norm[1], norm[2]);
          vertices.push(1, 1, 1, 1); // white color
          vertices.push(uv[0], uv[1]);

          const idx = vertexCount;
          vertexCount++;
          vertexMap.set(key, idx);
          faceIndices.push(idx);
        }
      }

      // Fan triangulation
      for (let j = 2; j < faceIndices.length; j++) {
        indices.push(faceIndices[0], faceIndices[j - 1], faceIndices[j]);
      }
    }
  }

  if (vertexCount === 0) return null;
  return { vertices, indices };
}

declare function bloom_read_file(path: number): number;

export function loadModel(path: string): Model {
  // Check for OBJ format
  const pathLower = (path as string).toLowerCase();
  if (pathLower.endsWith('.obj')) {
    const text: string = bloom_read_file(path as any) as any;
    if (text) {
      const parsed = parseOBJ(text);
      if (parsed) {
        return uploadMeshScratch(
          parsed.vertices, parsed.vertices.length / 12,
          parsed.indices, parsed.indices.length,
        );
      }
    }
    return makeModel(0);
  }

  const handle = bloom_load_model(path as any);
  return makeModel(handle);
}

export function drawModel(model: Model, position: Vec3, scale: number, tint: Color): void {
  bloom_draw_model(model.handle, position.x, position.y, position.z, scale, tint.r, tint.g, tint.b, tint.a);
}

/// Draw a model with a Y-axis rotation (radians). RGBA is packed into
/// a single f64 (ARGB byte order) to keep the FFI to 7 args, dodging
/// the Perry-ARM64 9th-arg quirk.
/**
 * Draw a model with a Y-axis rotation in DEGREES (engine-wide angle
 * convention, matching Camera2D.rotation and raylib; was radians before
 * v0.5). Tint components are 0-255.
 */
export function drawModelRotated(
  model: Model, position: Vec3, scale: number, rotY: number, tint: Color,
): void {
  // Color components are 0..255 ints (matching drawModel above).
  const a = (tint.a & 0xff) << 24;
  const r = (tint.r & 0xff) << 16;
  const g = (tint.g & 0xff) <<  8;
  const b =  tint.b & 0xff;
  // Use unsigned-shift-zero to keep the value positive when stored as f64.
  const packed = (a | r | g | b) >>> 0;
  bloom_draw_model_rotated(model.handle, position.x, position.y, position.z, scale, rotY * Math.PI / 180, packed);
}

export function unloadModel(model: Model): void {
  bloom_unload_model(model.handle);
}

/**
 * Return the axis-aligned bounding box of a loaded model in its local
 * coordinate space. Computed once at load time from mesh vertex positions.
 *
 * Used by editors to:
 *   - size move/rotate/scale gizmos to the selected entity
 *   - auto-frame the camera on the current selection
 *   - snap placed entities onto terrain by the lowest vertex
 *   - build precise ray-pick colliders
 *
 * Returns a zero-sized box at the origin if the model handle is invalid.
 */
/**
 * Q9: Generate a ribbon mesh along a Catmull-Rom spline.
 * `points` is a flat array [x0,y0,z0, x1,y1,z1, ...] of control points.
 * `widths` has one width per control point.
 * Returns a Model whose mesh is a smooth triangle-strip ribbon.
 */
/**
 * Build a ribbon mesh that follows a Catmull-Rom spline through `points`
 * (flat x,y,z triples), with a per-point half-width from `widths`.
 *
 * Goes through the mesh scratch buffers — positions first, then widths — for
 * the same reason as `createMesh`: Perry 0.5.x will not pass a `number[]` into
 * an `i64` pointer param, which made the pointer form unreachable from TS.
 */
export function genMeshSplineRibbon(points: number[], widths: number[]): Model {
  const pointCount = Math.floor(points.length / 3);
  const widthCount = widths.length;
  if (pointCount < 2 || widthCount === 0) return makeModel(0);

  bloom_mesh_scratch_reset();
  for (let i = 0; i < pointCount * 3; i++) bloom_mesh_scratch_push_f32(points[i]);
  for (let i = 0; i < widthCount; i++) bloom_mesh_scratch_push_f32(widths[i]);

  const handle = bloom_gen_mesh_spline_ribbon_scratch(pointCount, widthCount);
  return makeModel(handle);
}

export function getModelBounds(model: Model): BoundingBox {
  return {
    min: {
      x: bloom_get_model_bounds_min_x(model.handle),
      y: bloom_get_model_bounds_min_y(model.handle),
      z: bloom_get_model_bounds_min_z(model.handle),
    },
    max: {
      x: bloom_get_model_bounds_max_x(model.handle),
      y: bloom_get_model_bounds_max_y(model.handle),
      z: bloom_get_model_bounds_max_z(model.handle),
    },
  };
}

export interface DrawCubeOpts {
  rotationY?: number;
}

export function drawCube(position: Vec3, width: number, height: number, depth: number, color: Color, opts?: DrawCubeOpts): void {
  // Note: rotationY is accepted for API compatibility but applied only when native support exists
  bloom_draw_cube(position.x, position.y, position.z, width, height, depth, color.r, color.g, color.b, color.a);
}

export function drawCubeWires(position: Vec3, width: number, height: number, depth: number, color: Color): void {
  bloom_draw_cube_wires(position.x, position.y, position.z, width, height, depth, color.r, color.g, color.b, color.a);
}

export function drawSphere(position: Vec3, radius: number, color: Color): void {
  bloom_draw_sphere(position.x, position.y, position.z, radius, color.r, color.g, color.b, color.a);
}

export function drawSphereWires(position: Vec3, radius: number, color: Color): void {
  bloom_draw_sphere_wires(position.x, position.y, position.z, radius, color.r, color.g, color.b, color.a);
}

export function drawCylinder(position: Vec3, radiusTop: number, radiusBottom: number, height: number, color: Color, slices?: number): void {
  bloom_draw_cylinder(position.x, position.y, position.z, radiusTop, radiusBottom, height, color.r, color.g, color.b, color.a);
}

export function drawPlane(position: Vec3, width: number, depth: number, color: Color): void {
  bloom_draw_plane(position.x, position.y, position.z, width, depth, color.r, color.g, color.b, color.a);
}

export function drawGrid(slices: number, spacing: number): void {
  bloom_draw_grid(slices, spacing);
}

export function drawRay(origin: Vec3, direction: Vec3, color: Color): void {
  bloom_draw_ray(origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, color.r, color.g, color.b, color.a);
}

export function genMeshCube(width: number, height: number, depth: number): Model {
  const handle = bloom_gen_mesh_cube(width, height, depth);
  return makeModel(handle, 1, 1);
}

export function genMeshHeightmap(imageHandle: number, sizeX: number, sizeY: number, sizeZ: number): Model {
  const handle = bloom_gen_mesh_heightmap(imageHandle, sizeX, sizeY, sizeZ);
  return makeModel(handle, 1, 1);
}

export function loadShader(wgslSource: string): number {
  return bloom_load_shader(wgslSource as any);
}

/// Phase 1c — compile a material against the shader ABI in
/// `native/shared/shaders/material_abi.wgsl`. Source may `#include`
/// the header and common helpers. Returns a handle (>0 on success, 0
/// on compile failure — errors log to stderr).
export function compileMaterial(wgslSource: string): number {
  return bloom_compile_material(wgslSource as any);
}

/// Material profile — matches `FragmentProfile` in the engine's
/// material_pipeline. Opaque writes the full 4-MRT G-buffer;
/// Translucent writes a single HDR attachment with alpha blending.
export const PROFILE_OPAQUE = 0;
export const PROFILE_TRANSLUCENT = 1;

/// Material bucket — matches `Bucket` in the engine. Decides which
/// pass the draw lands in and what sort order applies. Refractive
/// also triggers a scene-colour snapshot before the pass runs.
export const BUCKET_OPAQUE = 0;
export const BUCKET_TRANSPARENT = 1;
export const BUCKET_REFRACTIVE = 2;
export const BUCKET_ADDITIVE = 3;
export const BUCKET_CUTOUT = 4;

/// Phase 4b — full-control material compile. Pass PROFILE_* and
/// BUCKET_* constants. `readsScene` enables the group-4 SceneInputs
/// binding; required for refraction / shoreline / depth-fade effects.
/// Phase 4b — compile a refractive material (profile=Translucent,
/// bucket=Refractive, reads_scene=true). The material's shader gets
/// the SceneInputs bind group at group 4 and can sample the
/// pre-translucent scene colour via `scene_color_tex`.
export function compileRefractiveMaterial(wgslSource: string): number {
  return bloom_compile_material_refractive(wgslSource as any);
}

/// Compile a transparent material (profile=Translucent, bucket=
/// Transparent, reads_scene=false). Alpha-blended, sorted back-to-
/// front. No scene-colour sampling.
export function compileTransparentMaterial(wgslSource: string): number {
  return bloom_compile_material_transparent(wgslSource as any);
}

/// Compile an additive material (profile=Translucent, bucket=
/// Additive, reads_scene=false). Order-independent — use for
/// particle flares, weapon glows, spell effects.
export function compileAdditiveMaterial(wgslSource: string): number {
  return bloom_compile_material_additive(wgslSource as any);
}

/// Compile a material into the Cutout bucket: writes the full G-buffer
/// (so it casts/receives sun shadow + SSAO), but the fragment shader
/// is expected to call `discard` against `material.metal_rough.w`
/// (`MaterialFactors.alpha_cutoff`) to drop transparent texels. Use
/// for foliage cards, chain-link fences, leaf silhouettes. Rendered
/// double-sided (cull_mode=None) so foliage is visible from both
/// faces.
export function compileMaterialCutout(wgslSource: string): number {
  return bloom_compile_material_cutout(wgslSource as any);
}

/// EN-001 — compile a material that draws via the instanced pipeline.
/// Per-instance data is uploaded once via `createInstanceBuffer` and
/// drawn via `drawMeshWithMaterialInstanced`. The game shader's
/// VertexInput must declare these locations IN ADDITION to the
/// standard 0..6 (see `material_abi.wgsl` for the canonical layout):
///
///   @location(7)  instance_pos:    vec3<f32>
///   @location(8)  instance_rot_y:  f32
///   @location(9)  instance_scale:  f32
///   @location(10) instance_tint:   vec4<f32>
///
/// The pipeline lives in the Opaque bucket (cuts opaque profile +
/// no scene-color reads). For the shooter's grass/foliage workload
/// this is the right default; future work can extend to other buckets.
export function compileMaterialInstanced(wgslSource: string): number {
  return bloom_compile_material_instanced(wgslSource as any);
}

/// EN-001 — upload a flat per-instance buffer to the GPU. `data` is
/// laid out as 9 floats per instance:
///   [pos.x, pos.y, pos.z, rot_y, scale, tint.r, tint.g, tint.b, tint.a]
/// × instanceCount. Returns a handle to use with
/// `drawMeshWithMaterialInstanced`. The buffer is persistent across
/// frames; call `destroyInstanceBuffer` when the data is no longer
/// needed.
///
/// IMPORTANT: pass `instanceCount` derived from a literal-init array
/// length OR from a manually-tracked counter. Don't compute
/// `data.length / 9` if `data` was built via `.push()` — Perry's
/// `.length` reports the literal-init size, not the post-push count
/// (a Perry codegen bug — see `feedback_perry_array_push.md`).
export function createInstanceBuffer(data: number[], instanceCount: number): number {
  // Perry 0.5.x rejects JS arrays passed into i64 pointer params, so the
  // instance data goes through the all-f64 mesh scratch instead (same fix
  // as createMesh). 9 floats per instance; instance buffers are built once
  // at startup, so the per-float FFI calls are a one-time cost.
  bloom_mesh_scratch_reset();
  const n = instanceCount * 9;
  for (let i = 0; i < n; i++) bloom_mesh_scratch_push_f32(data[i]);
  return bloom_create_instance_buffer_scratch(instanceCount);
}

/// EN-001 — draw `mesh` (a single primitive at `meshIdx`)
/// `instanceCount` times using `material`'s instanced pipeline. The
/// instance buffer is bound at vertex slot 1 and the engine emits a
/// single draw_indexed call. Per-draw model/MVP are identity / current
/// camera VP — per-instance pos/rot_y/scale dominate from the buffer.
export function drawMeshWithMaterialInstanced(
  material: number, mesh: Model, meshIdx: number,
  instanceBuffer: number, instanceCount: number,
): void {
  bloom_submit_material_draw_instanced(
    material, mesh.handle, meshIdx, instanceBuffer, instanceCount,
  );
}

/// EN-001 — release the GPU memory backing an instance buffer
/// returned by `createInstanceBuffer`. Safe to call with handle 0
/// (no-op).
export function destroyInstanceBuffer(handle: number): void {
  bloom_destroy_instance_buffer(handle);
}

/// EN-011 — create a planar reflection probe. Each frame, the engine
/// renders the world (minus excluded materials) from a mirror camera
/// across the given plane into an HDR RT, bound at `@group(2)
/// @binding(12)` on materials that opt in via
/// `setMaterialReflectionProbe`.
///
/// Use for water surfaces where the static `env_tex` sky reflection
/// isn't enough — the planar reflection captures the actual scene
/// geometry (trees, bridges) reflected on the water plane, with
/// wave-normal wobble applied by the material's WGSL.
///
/// Arguments:
///   - `planeY` — world-space Y offset of the reflective plane
///   - `normalX/Y/Z` — plane normal (typically (0,1,0) for water)
///   - `resolution` — square texture side in pixels; pass 0 to
///     default to half the swapchain width
///
/// Returns a 1-based probe handle (0 on failure).
///
/// V1 limits: one probe per plane, repainted every frame, hardcoded
/// exclude list (materials linked to a probe never appear in their
/// own reflection — the water plane doesn't show up in itself).
/// Future: cull-mode flip for correct front-face winding (V1 leaves
/// pipelines' compiled cull mode in place; the artifact is mostly
/// hidden for typical horizontal-water-from-above viewpoints).
///
/// WGSL pattern in a water material:
///   ```wgsl
///   let screen_uv = clip_position.xy / vec2<f32>(frame.screen_resolution);
///   let perturb = wave_normal.xz * 0.05;
///   let refl = textureSample(planar_reflection_tex,
///                            planar_reflection_samp,
///                            screen_uv + perturb).rgb;
///   ```
export function createPlanarReflection(
  planeY: number,
  normalX: number, normalY: number, normalZ: number,
  resolution: number,
): number {
  return bloom_create_planar_reflection(planeY, normalX, normalY, normalZ, resolution);
}

/// EN-011 — link `material` to a planar reflection probe (handle
/// returned by `createPlanarReflection`). The probe's RT is bound at
/// `@group(2) @binding(12)` on subsequent draws. Pass `probe = 0` to
/// unlink and revert to the default 1×1 black texture.
///
/// Materials linked to any probe are automatically excluded from
/// every probe's render — the water surface doesn't reflect itself.
export function setMaterialReflectionProbe(material: number, probe: number): void {
  bloom_set_material_reflection_probe(material, probe);
}

/// Whether this material's draws render into planar-reflection probes
/// (default true). Turn off for content that is sub-pixel at probe
/// resolution — e.g. an instanced grass field in a 512-px water probe —
/// where it costs full vertex + raster work and contributes nothing
/// resolvable to the reflection.
export function setMaterialProbeVisible(material: number, visible: boolean): void {
  bloom_set_material_probe_visible(material, visible ? 1 : 0);
}

/// EN-014 — slot indices for `setMaterialTextureArray`.
export const TEXTURE_ARRAY_ALBEDO = 0;
export const TEXTURE_ARRAY_NORMAL = 1;
export const TEXTURE_ARRAY_MR     = 2;

/// EN-014 — create a 2D texture array from a flat RGBA8 byte buffer.
/// All `layerCount` layers must share the same `width × height` (wgpu
/// requires uniform extent for D2Array). The buffer holds each layer
/// back-to-back: byte[0..(W·H·4)] = layer 0, byte[W·H·4..2·W·H·4] =
/// layer 1, and so on. Layer count is capped at 16 in V1.
///
/// Returns a 1-based handle (0 on failure: zero count, zero extent,
/// or short buffer). Pair with `setMaterialTextureArray` to bind to
/// one of three slots (albedo / normal / MR) on a terrain material;
/// the WGSL fragment samples the array via:
///   ```wgsl
///   let albedo = textureSample(albedo_array, albedo_array_samp,
///                              uv_world_xz, layer_idx);
///   ```
///
/// IMPORTANT: pass `dataLen` and `layerCount` derived from manually-
/// tracked counters — NOT `bytes.length` if `bytes` was built via
/// `.push()` — Perry's `.length` reports the literal-init size, not
/// the post-push count. (See `feedback_perry_array_push.md`.)
export function createTextureArray(
  bytes: number[], dataLen: number,
  width: number, height: number, layerCount: number,
): number {
  return bloom_create_texture_array(bytes as any, dataLen, width, height, layerCount);
}

/// EN-014 V2 — texture-array pixel format codes for `createTextureArrayEx`.
///   `TEX_ARRAY_FORMAT_SRGB`   (0) → Rgba8UnormSrgb (albedo / colour textures)
///   `TEX_ARRAY_FORMAT_LINEAR` (1) → Rgba8Unorm (normal / MR / data textures —
///     mandatory for normal maps so the GPU doesn't sRGB-decode the encoded
///     data and silently corrupt the channels).
export const TEX_ARRAY_FORMAT_SRGB:   number = 0;
export const TEX_ARRAY_FORMAT_LINEAR: number = 1;

/// EN-014 V2 — create a texture array with explicit format + mip control.
///
/// `format`:
///   `TEX_ARRAY_FORMAT_SRGB`   (0) → Rgba8UnormSrgb (albedo / colour)
///   `TEX_ARRAY_FORMAT_LINEAR` (1) → Rgba8Unorm (normal / MR / data)
///
/// `mipLevels`:
///   `1` → no mips (matches V1 `createTextureArray`; data is just mip 0)
///   `0` → auto-generate `floor(log2(max(w,h))) + 1` levels, filled by
///         point-downsample copies. V2.5 follow-up will upgrade to a
///         render-pass box filter for higher-quality minification.
///   `N` (N > 1) → not yet supported in V2; treated as auto-generate.
///
/// Backwards compatible: `createTextureArray` (no Ex) stays available
/// and is equivalent to `createTextureArrayEx(.., TEX_ARRAY_FORMAT_SRGB, 1)`.
export function createTextureArrayEx(
  bytes: number[], dataLen: number,
  width: number, height: number, layerCount: number,
  format: number, mipLevels: number,
): number {
  return bloom_create_texture_array_ex(bytes as any, dataLen, width, height, layerCount, format, mipLevels);
}

/// EN-014 — link a texture-array handle to a material at one of three
/// slots: `TEXTURE_ARRAY_ALBEDO` (binding 14), `TEXTURE_ARRAY_NORMAL`
/// (binding 15), `TEXTURE_ARRAY_MR` (binding 16). Pass `array = 0` to
/// revert the slot to the engine's 1×1×1 stub.
///
/// Materials don't need to bind every slot — the stub is safe to
/// sample. A common pattern is to bind only `TEXTURE_ARRAY_ALBEDO`
/// for a non-PBR splat-mapped terrain.
export function setMaterialTextureArray(material: number, slot: number, array: number): void {
  bloom_set_material_texture_array(material, slot, array);
}

/// EN-012 — shading-model selectors. Pass to `setMaterialShadingModel`.
export const SHADING_MODEL_DEFAULT_LIT = 0;
export const SHADING_MODEL_FOLIAGE     = 1;
export const SHADING_MODEL_SUBSURFACE  = 2;   // V2 stub — currently behaves as default lit

/// EN-012 — switch a material's shading model. The game shader is
/// responsible for branching on `material.shading_model.x` and calling
/// either standard PBR or `shade_foliage` (wrap-lambert + transmission)
/// from `common/pbr.wgsl`. The engine just exposes the slot.
///
/// V1 limitation: SSAO doesn't half-strength on backfaces — the
/// G-buffer doesn't carry an isFrontFace channel today. Documented as a
/// follow-up requirement on the EN-012 ticket.
export function setMaterialShadingModel(material: number, model: number): void {
  bloom_set_material_shading_model(material, model);
}

/// EN-012 — set the foliage shading parameters for a material. Only
/// takes effect when `shading_model == SHADING_MODEL_FOLIAGE`.
///
/// `transmissionR/G/B`: rgb tint for back-lit foliage (1,1,1 = neutral).
/// `transmissionAmount`: 0..1 — how much sun bleeds through the leaf.
/// `wrapFactor`: 0..1 — wrap-lambert intensity. 0 = standard lambert
///   (back face goes pure black), 1 = light wraps fully around to the
///   back face.
export function setMaterialFoliage(
  material: number,
  transmissionR: number, transmissionG: number, transmissionB: number,
  transmissionAmount: number, wrapFactor: number,
): void {
  bloom_set_material_foliage(material, transmissionR, transmissionG, transmissionB, transmissionAmount, wrapFactor);
}

/**
 * Phase 6 — file-backed material compile with hot reload. Reads the
 * WGSL from disk, compiles it, and registers the path with the
 * engine's hot-reload watcher. Editing the file while the game is
 * running re-compiles the pipeline and replaces it in place — the
 * material handle stays valid; existing draws automatically pick up
 * the new shader on the next frame.
 *
 * Failures during reload (parse error, validation) keep the previous
 * pipeline running; the error is logged but doesn't crash the game.
 *
 * `bucket` selects the same presets as the dedicated compile* APIs:
 *   'opaque' | 'cutout' | 'transparent' | 'refractive' | 'additive'
 */
export function compileMaterialFromFile(
  path: string,
  bucket: 'opaque' | 'cutout' | 'transparent' | 'refractive' | 'additive',
): number {
  const kind = bucket === 'opaque'      ? 0
             : bucket === 'transparent' ? 1
             : bucket === 'refractive'  ? 2
             : bucket === 'additive'    ? 3
             :                            4; // cutout
  return bloom_compile_material_from_file(path as any, kind);
}

/**
 * Phase 5 — material descriptor loader. Compiles a material from
 * a typed descriptor in one call:
 *   - resolves `shader` via `compileMaterialFromFile` (gets hot
 *     reload)
 *   - applies `params` via `setMaterialParams` if provided
 *
 * Returns the material handle (0 on failure).
 *
 * Why a typed object instead of a JSON string: Perry's runtime
 * `JSON.parse` mishandles array `.length`, so a JSON-string variant
 * would force every game to roll its own parser. Games that DO
 * want JSON-on-disk should preprocess at build time (see
 * `shooter/tools/build-world.ts` for the pattern) and emit a TS
 * module that calls `loadMaterial` with literal descriptors.
 */
export interface MaterialDesc {
  shader: string;
  bucket: 'opaque' | 'cutout' | 'transparent' | 'refractive' | 'additive';
  params?: number[];
}

export function loadMaterial(desc: MaterialDesc): number {
  const handle = compileMaterialFromFile(desc.shader, desc.bucket);
  if (handle > 0 && desc.params) {
    // Bind to a local first — Perry currently mishandles `.length`
    // on an object field passed directly into an FFI call (the FFI
    // sees count = 0). Re-binding to a local lets `.length` evaluate
    // before the FFI call is laid out.
    const p = desc.params;
    if (p.length > 0) {
      bloom_set_material_params(handle, p as any, p.length);
    }
  }
  return handle;
}

/// Draw a mesh with a material. `mesh` must be a Model created via
/// `createMesh` or `loadModel`. Transform is a position + uniform
/// scale; tint is an RGBA color multiplied into PerDraw.model_tint.
/// `meshIdx` selects which primitive of a multi-mesh GLB to draw —
/// default 0 for single-mesh models. Loop 0..mesh.meshCount when
/// rendering a multi-primitive GLB through a custom material.
export function drawMeshWithMaterial(
  material: number, mesh: Model,
  position: Vec3, scale: number, tint: Color,
  meshIdx: number = 0,
): void {
  bloom_draw_material(
    material, mesh.handle, meshIdx,
    position.x, position.y, position.z, scale,
    tint.r, tint.g, tint.b, tint.a,
  );
}

/// Draw every primitive of a multi-mesh GLB with the same material.
/// Convenience wrapper around `drawMeshWithMaterial` that loops
/// 0..mesh.meshCount internally.
export function drawModelWithMaterial(
  material: number, mesh: Model,
  position: Vec3, scale: number, tint: Color,
): void {
  for (let i = 0; i < mesh.meshCount; i = i + 1) {
    bloom_draw_material(
      material, mesh.handle, i,
      position.x, position.y, position.z, scale,
      tint.r, tint.g, tint.b, tint.a,
    );
  }
}

export function loadModelAnimation(path: string): number {
  return bloom_load_model_animation(path as any);
}

export function updateModelAnimation(handle: number, animIndex: number, time: number, scale: number, px: number, py: number, pz: number, rotY: number): void {
  bloom_update_model_animation(handle, animIndex, time, scale, px, py, pz, rotY);
}

// ---- EN-028: animation mixer -----------------------------------------------
// The single-clip `updateModelAnimation` above stays for callers that drive
// their own clip clock. The mixer below owns the clock instead, which is what
// makes crossfades possible at all: a fade needs the *outgoing* clip to keep
// advancing, and a caller that only passes one time value cannot express that.
//
// Typical use, per model per frame:
//   animPlay(h, moving ? CLIP_WALK : CLIP_IDLE, 0.15);   // idempotent
//   animSetLayer(h, attacking ? CLIP_ATTACK : -1, 1, spineJoint);
//   animUpdate(h, dt, scale, x, y, z, yaw);

declare function bloom_anim_play(handle: number, clip: number, fade: number, speed: number, looping: number): void;
declare function bloom_anim_set_layer(handle: number, clip: number, weight: number, maskRoot: number, speed: number, looping: number): void;
declare function bloom_anim_set_root_motion(handle: number, on: number): void;
declare function bloom_anim_update(handle: number, dt: number, scale: number, px: number, py: number, pz: number, rotY: number): void;
declare function bloom_anim_finished(handle: number): number;
declare function bloom_anim_clip_duration(handle: number, clip: number): number;
declare function bloom_anim_root_delta(handle: number, axis: number): number;
declare function bloom_model_find_joint(handle: number, name: number): number;
declare function bloom_model_joint_world(handle: number, joint: number, comp: number): number;

/// Transition the base track to `clip` over `fade` seconds. Safe to call every
/// frame with the clip you *want* — re-requesting the clip already playing is
/// a no-op, so callers don't have to track edges.
export function animPlay(handle: number, clip: number, fade: number = 0.15, speed: number = 1.0, looping: boolean = true): void {
  bloom_anim_play(handle, clip, fade, speed, looping ? 1 : 0);
}

/// Drive the subtree below `maskRoot` (a joint index — see `findJoint`) from a
/// second clip at `weight`. Pass clip = -1 to switch the layer off. This is how
/// a character attacks while still walking.
export function animSetLayer(handle: number, clip: number, weight: number, maskRoot: number, speed: number = 1.0, looping: boolean = false): void {
  bloom_anim_set_layer(handle, clip, weight, maskRoot, speed, looping ? 1 : 0);
}

/// Opt in to authored root motion. Off by default: with it on, the pose stops
/// carrying the root translation and you must feed `animRootDelta` to your
/// character controller, or the model animates in place.
export function animSetRootMotion(handle: number, on: boolean): void {
  bloom_anim_set_root_motion(handle, on ? 1 : 0);
}

/// Advance all clocks on this model and upload the blended pose. One call per
/// model per frame, in place of `updateModelAnimation`.
export function animUpdate(handle: number, dt: number, scale: number, px: number, py: number, pz: number, rotY: number): void {
  bloom_anim_update(handle, dt, scale, px, py, pz, rotY);
}

/// True once a non-looping clip has run past its end — the death/attack
/// one-shot query.
export function animFinished(handle: number): boolean {
  return bloom_anim_finished(handle) !== 0;
}

export function animClipDuration(handle: number, clip: number): number {
  return bloom_anim_clip_duration(handle, clip);
}

/// Root-motion translation applied by the last `animUpdate`, in model space.
export function animRootDelta(handle: number, axis: number): number {
  return bloom_anim_root_delta(handle, axis);
}

// ---- EN-033: bone sockets ---------------------------------------------------

/// Joint index by name (exact, else case-insensitive substring). Call once at
/// load and cache — it parses a string, which must never happen per-frame
/// (perry-quirks #5). Returns -1 if not found.
export function findJoint(handle: number, name: string): number {
  return bloom_model_find_joint(handle, name as any);
}

/// One component of a joint's model-space 4x4 (column-major, 0..15).
/// Translation is 12/13/14. Model-space, not world: apply the same scale /
/// position / yaw you passed to `animUpdate` to place it in the world.
export function jointWorld(handle: number, joint: number, comp: number): number {
  return bloom_model_joint_world(handle, joint, comp);
}

// Upload a mesh via the scratch buffer (array-free). Perry 0.5.1171 rejects
// passing a `number[]` to a native `i64` pointer param (strict safe-integer
// check), so we push each vertex float + index scalar through the all-f64
// scratch FFI and then build. One-time init cost; fine for static meshes.
function uploadMeshScratch(
  vertices: number[], vertexCount: number,
  indices: number[], indexCount: number,
): Model {
  bloom_mesh_scratch_reset();
  const vfloats = vertexCount * 12;
  for (let i = 0; i < vfloats; i++) bloom_mesh_scratch_push_f32(vertices[i]);
  for (let i = 0; i < indexCount; i++) bloom_mesh_scratch_push_u32(indices[i]);
  const handle = bloom_create_mesh_scratch(vertexCount, indexCount);
  return makeModel(handle, 1, 1);
}

export function createMesh(vertices: number[], indices: number[]): Model {
  // vertices: flat array of [x,y,z, nx,ny,nz, r,g,b,a, u,v] per vertex (12 floats each)
  // NOTE: `vertices.length` / `indices.length` are correct only for arrays
  // built via literals or `new Array(N)` + index assignment. Arrays built via
  // `.push()` report the literal-init size (a Perry codegen bug) — use
  // createMeshExplicit and pass the counts manually for those.
  return uploadMeshScratch(vertices, vertices.length / 12, indices, indices.length);
}

/// Explicit-count variant of createMesh — pass `vertexCount` (number
/// of complete 12-float vertex records) and `indexCount` directly.
/// Use this when the underlying `number[]` arrays were built with
/// `.push()`, since Perry's `.length` doesn't reflect post-push size.
export function createMeshExplicit(
  vertices: number[], vertexCount: number,
  indices: number[], indexCount: number,
): Model {
  return uploadMeshScratch(vertices, vertexCount, indices, indexCount);
}

export function setAmbientLight(color: Color, intensity: number): void {
  bloom_set_ambient_light(color.r, color.g, color.b, intensity);
}

export function setDirectionalLight(direction: Vec3, color: Color, intensity: number): void {
  bloom_set_directional_light(direction.x, direction.y, direction.z, color.r, color.g, color.b, intensity);
}

// EN-005 — Hillaire 2020 procedural sky.
//
// `setProceduralSky(true)` swaps the static HDR-panorama background
// for a physics-based atmosphere driven by Rayleigh + Mie scattering.
// `setSunDirection` then steers the sun: the sky-view LUT re-bakes
// on the next frame and the sun disk + transmittance update together.
//
// `setProceduralSky(false)` returns to the panorama path; the
// existing `loadEnvironment` / `setEnvironmentIntensity` flow is
// untouched.

export interface ProceduralSkyOptions {
  /** Rayleigh density multiplier. 1.0 = Earth standard. Higher = bluer
   * thicker air; lower = thinner. */
  rayleighDensity?: number;
  /** Mie density multiplier. Drives haze + sun-glow size. 1.0 = Earth
   * standard, raise for hazy/dusty scenes. */
  mieDensity?: number;
  /** Ground albedo (0..1). Affects how much light bounces back up
   * into the lower atmosphere. 0.1 = soil/grass, 0.9 = snow. */
  groundAlbedo?: number;
}

export function setProceduralSky(enabled: boolean, opts?: ProceduralSkyOptions): void {
  const rd = opts?.rayleighDensity ?? 1.0;
  const md = opts?.mieDensity ?? 1.0;
  const ga = opts?.groundAlbedo ?? 0.1;
  bloom_set_procedural_sky(enabled ? 1 : 0, rd, md, ga);
}

export function setSunDirection(direction: Vec3, intensity: number = 1.0): void {
  bloom_set_sun_direction(direction.x, direction.y, direction.z, intensity);
}

declare function bloom_set_joint_test(joint: number, angle: number): void;
export function setJointTest(joint: number, angle: number): void {
  bloom_set_joint_test(joint, angle);
}

// Async / threaded loading

declare function bloom_stage_model(path: number): number;
declare function bloom_commit_model(handle: number): number;

export async function loadModelAsync(path: string): Promise<Model> {
  const pathLower = (path as string).toLowerCase();
  if (pathLower.endsWith('.obj')) {
    const parsed = await spawn(() => {
      const text: string = bloom_read_file(path as any) as any;
      return text ? parseOBJ(text) : null;
    });
    if (parsed) {
      const handle = bloom_create_mesh(
        parsed.vertices as any,
        parsed.vertices.length / 12,
        parsed.indices as any,
        parsed.indices.length,
      );
      return makeModel(handle, 1, 1);
    }
    return makeModel(0);
  }
  const stagingHandle = await spawn(() => bloom_stage_model(path as any));
  const handle = bloom_commit_model(stagingHandle);
  return makeModel(handle);
}

export function stageModels(paths: string[]): number[] {
  return parallelMap(paths, (path: string) => bloom_stage_model(path as any));
}

export function commitModel(stagingHandle: number): Model {
  const handle = bloom_commit_model(stagingHandle);
  return makeModel(handle);
}

// ---------------------------------------------------------------------
// EN-015 V1 — Octahedral imposter / billboard helpers.
//
// V1 ships the runtime piece only: the WGSL helper library
// (`common/imposter.wgsl`) plus this TS-side LOD selector +
// `drawImposterAtlas` wrapper. Bake tooling is a follow-up — V1 expects
// games to bake atlases externally (Blender's ScreenSpace add-on,
// Unity Tree Creator, etc.) until the engine bake tool ships.
//
// The atlas convention is fixed at 8×8 octahedral views (64 cells
// total) packed into a single RGBA8 texture.
//
// Typical game-side imposter material WGSL (drop in your own .wgsl
// file and pass to compileMaterial):
//
//   #include "material_abi.wgsl"
//   #include "common/imposter.wgsl"
//
//   struct VsOut {
//     @builtin(position) clip_pos: vec4<f32>,
//     @location(0)       view_dir: vec3<f32>,
//     @location(1)       uv:       vec2<f32>,
//   };
//
//   @vertex
//   fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
//     // node.transform[3].xyz holds the per-instance world position;
//     // node.scale_x is the imposter scale.
//     let center = node.transform[3].xyz;
//     let bb = billboard_quad(
//       center, node.scale_x,
//       view.camera_pos.xyz, vec3<f32>(0.0, 1.0, 0.0),
//       vid,
//     );
//     var out: VsOut;
//     out.clip_pos = view.view_proj * vec4<f32>(bb.world_pos, 1.0);
//     out.view_dir = normalize(view.camera_pos.xyz - center);
//     out.uv       = bb.uv;
//     return out;
//   }
//
//   @fragment
//   fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
//     let atlas_uv = imposter_atlas_uv(in.view_dir, in.uv);
//     return textureSample(material_albedo, material_sampler, atlas_uv);
//   }
//
// Then at runtime:
//
//   const useImposter = pickImposterLOD(
//     camera.x, camera.y, camera.z,
//     tree.x, tree.y, tree.z,
//     50.0,
//   );
//   if (useImposter) {
//     drawImposterAtlas(impMaterial, impAtlasMesh, treePos, treeScale);
//   } else {
//     drawModel(treeModel, treePos, treeScale, WHITE);
//   }
// ---------------------------------------------------------------------

/// EN-015 — pick LOD by distance. Returns true if the camera is far
/// enough that the imposter should be drawn in place of the full mesh.
///
/// Typical `switchDistance` is 40–60 m for trees. To mitigate pop at
/// the LOD boundary, layer a small hysteresis band on top from the
/// caller (e.g. switch to imposter at 50 m, switch back to mesh at
/// 47 m); when TAA is enabled the residual flicker is mostly resolved
/// across frames.
export function pickImposterLOD(
  cameraX: number, cameraY: number, cameraZ: number,
  worldX: number, worldY: number, worldZ: number,
  switchDistance: number,
): boolean {
  const dx = cameraX - worldX;
  const dy = cameraY - worldY;
  const dz = cameraZ - worldZ;
  const distSq = dx * dx + dy * dy + dz * dz;
  const threshSq = switchDistance * switchDistance;
  return distSq > threshSq;
}

/// EN-015 — draw a billboard quad sampling an imposter atlas.
///
/// V1 is a thin wrapper around `drawMeshWithMaterial`: the game
/// supplies its own quad mesh (any 4-vertex / 2-triangle plane the
/// imposter material's vertex shader will reposition via
/// `billboard_quad`). The atlas texture is bound through the standard
/// material albedo slot before this call (via the engine's material
/// param / texture binding APIs).
///
/// Future revs may add a dedicated FFI that submits a hard-coded unit
/// quad so games don't need to author a quad mesh; for V1 we keep the
/// API surface tight and reuse the existing draw path.
export function drawImposterAtlas(
  material: number, quadMesh: Model,
  position: Vec3, scale: number,
): void {
  bloom_draw_material(
    material, quadMesh.handle, 0,
    position.x, position.y, position.z, scale,
    1.0, 1.0, 1.0, 1.0,
  );
}
