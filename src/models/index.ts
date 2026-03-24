import { spawn, parallelMap } from 'perry/thread';
import { Color, Model, Vec3, Mat4 } from '../core/types';

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
declare function bloom_load_model_animation(path: number): number;
declare function bloom_update_model_animation(handle: number, animIndex: number, time: number, scale: number, px: number, py: number, pz: number, rotSin: number, rotCos: number): void;
declare function bloom_create_mesh(vertexPtr: number, vertexCount: number, indexPtr: number, indexCount: number): number;
declare function bloom_set_ambient_light(r: number, g: number, b: number, intensity: number): void;
declare function bloom_set_directional_light(dx: number, dy: number, dz: number, r: number, g: number, b: number, intensity: number): void;
declare function bloom_get_model_mesh_count(handle: number): number;
declare function bloom_get_model_material_count(handle: number): number;

const IDENTITY: Mat4 = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];

function makeModel(handle: number, meshCount?: number, materialCount?: number): Model {
  let mc = meshCount ?? 1;
  let matc = materialCount ?? 1;
  try { mc = bloom_get_model_mesh_count(handle); } catch (_) {}
  try { matc = bloom_get_model_material_count(handle); } catch (_) {}
  return { handle, meshCount: mc, materialCount: matc, transform: [...IDENTITY] };
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
    return makeModel(0, 0, 0);
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

export function loadModelAnimation(path: string): number {
  return bloom_load_model_animation(path as any);
}

export function updateModelAnimation(handle: number, animIndex: number, time: number, scale: number, px: number, py: number, pz: number, rotSin?: number, rotCos?: number): void {
  bloom_update_model_animation(handle, animIndex, time, scale, px, py, pz, rotSin ?? 0.0, rotCos ?? 1.0);
}

export function createMesh(vertices: number[], indices: number[]): Model {
  // vertices: flat array of [x,y,z, nx,ny,nz, r,g,b,a, u,v] per vertex (12 floats each)
  const handle = bloom_create_mesh(vertices as any, vertices.length / 12, indices as any, indices.length);
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
    return makeModel(0, 0, 0);
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
