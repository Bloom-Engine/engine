import { spawn, parallelMap } from 'perry/thread';
import { Color, Model, Vec3, Mat4, BoundingBox } from '../core/types';

// FFI declarations
declare function bloom_load_model(path: number): number;
declare function bloom_draw_model(handle: number, x: number, y: number, z: number, scale: number, r: number, g: number, b: number, a: number): void;
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
declare function bloom_compile_material_from_file(path: number, bucketKind: number): number;
declare function bloom_set_material_params(handle: number, paramsPtr: any, paramCount: number): void;
declare function bloom_draw_material(material: number, meshHandle: number, meshIdx: number, x: number, y: number, z: number, scale: number, r: number, g: number, b: number, a: number): void;
declare function bloom_load_model_animation(path: number): number;
declare function bloom_update_model_animation(handle: number, animIndex: number, time: number, scale: number, px: number, py: number, pz: number, rotY: number): void;
declare function bloom_create_mesh(vertexPtr: number, vertexCount: number, indexPtr: number, indexCount: number): number;
declare function bloom_set_ambient_light(r: number, g: number, b: number, intensity: number): void;
declare function bloom_set_directional_light(dx: number, dy: number, dz: number, r: number, g: number, b: number, intensity: number): void;
declare function bloom_gen_mesh_spline_ribbon(pointsPtr: number, pointCount: number, widthsPtr: number, widthCount: number): number;
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
        const handle = bloom_create_mesh(
          parsed.vertices as any,
          parsed.vertices.length / 12,
          parsed.indices as any,
          parsed.indices.length,
        );
        return makeModel(handle, 1, 1);
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
export function genMeshSplineRibbon(points: number[], widths: number[]): Model {
  const pointCount = points.length / 3;
  const handle = bloom_gen_mesh_spline_ribbon(points as any, pointCount, widths as any, widths.length);
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
 *   'opaque' | 'transparent' | 'refractive' | 'additive'
 */
export function compileMaterialFromFile(
  path: string,
  bucket: 'opaque' | 'transparent' | 'refractive' | 'additive',
): number {
  const kind = bucket === 'opaque'      ? 0
             : bucket === 'transparent' ? 1
             : bucket === 'refractive'  ? 2
             :                            3;
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
  bucket: 'opaque' | 'transparent' | 'refractive' | 'additive';
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
export function drawMeshWithMaterial(
  material: number, mesh: Model,
  position: Vec3, scale: number, tint: Color,
): void {
  bloom_draw_material(
    material, mesh.handle, 0,
    position.x, position.y, position.z, scale,
    tint.r, tint.g, tint.b, tint.a,
  );
}

export function loadModelAnimation(path: string): number {
  return bloom_load_model_animation(path as any);
}

export function updateModelAnimation(handle: number, animIndex: number, time: number, scale: number, px: number, py: number, pz: number, rotY: number): void {
  bloom_update_model_animation(handle, animIndex, time, scale, px, py, pz, rotY);
}

export function createMesh(vertices: number[], indices: number[]): Model {
  // vertices: flat array of [x,y,z, nx,ny,nz, r,g,b,a, u,v] per vertex (12 floats each)
  // NOTE: `vertices.length` and `indices.length` here only return the
  // correct value for arrays built via literals or `new Array(N)` +
  // index assignment. Arrays built via `.push()` reflect the LITERAL
  // initialization size as `.length`, not the post-push size (a
  // Perry codegen bug). For `.push`-built data, use createMeshExplicit
  // and pass the counts manually.
  const handle = bloom_create_mesh(vertices as any, vertices.length / 12, indices as any, indices.length);
  return makeModel(handle, 1, 1);
}

/// Explicit-count variant of createMesh — pass `vertexCount` (number
/// of complete 12-float vertex records) and `indexCount` directly.
/// Use this when the underlying `number[]` arrays were built with
/// `.push()`, since Perry's `.length` doesn't reflect post-push size.
export function createMeshExplicit(
  vertices: number[], vertexCount: number,
  indices: number[], indexCount: number,
): Model {
  const handle = bloom_create_mesh(vertices as any, vertexCount, indices as any, indexCount);
  return makeModel(handle, 1, 1);
}

export function setAmbientLight(color: Color, intensity: number): void {
  bloom_set_ambient_light(color.r, color.g, color.b, intensity);
}

export function setDirectionalLight(direction: Vec3, color: Color, intensity: number): void {
  bloom_set_directional_light(direction.x, direction.y, direction.z, color.r, color.g, color.b, intensity);
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
