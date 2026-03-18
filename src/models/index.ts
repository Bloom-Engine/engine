import { Color, Model, Vec3 } from '../core/types';

// FFI declarations
declare function bloom_load_model(path: number): number;
declare function bloom_draw_model(handle: number, x: number, y: number, z: number, scale: number, r: number, g: number, b: number, a: number): void;
declare function bloom_unload_model(handle: number): void;
declare function bloom_draw_cube(x: number, y: number, z: number, w: number, h: number, d: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_cube_wires(x: number, y: number, z: number, w: number, h: number, d: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_sphere(x: number, y: number, z: number, radius: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_sphere_wires(x: number, y: number, z: number, radius: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_cylinder(x: number, y: number, z: number, rt: number, rb: number, h: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_plane(x: number, y: number, z: number, w: number, d: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_grid(slices: number, spacing: number): void;
declare function bloom_draw_ray(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, r: number, g: number, b: number, a: number): void;
declare function bloom_gen_mesh_cube(w: number, h: number, d: number): number;
declare function bloom_gen_mesh_heightmap(imageHandle: number, sizeX: number, sizeY: number, sizeZ: number): number;

export function loadModel(path: string): Model {
  const handle = bloom_load_model(path as any);
  return { handle };
}

export function drawModel(model: Model, position: Vec3, scale: number, tint: Color): void {
  bloom_draw_model(model.handle, position.x, position.y, position.z, scale, tint.r, tint.g, tint.b, tint.a);
}

export function unloadModel(model: Model): void {
  bloom_unload_model(model.handle);
}

export function drawCube(position: Vec3, width: number, height: number, depth: number, color: Color): void {
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

export function drawCylinder(position: Vec3, radiusTop: number, radiusBottom: number, height: number, color: Color): void {
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
  return { handle };
}

export function genMeshHeightmap(imageHandle: number, sizeX: number, sizeY: number, sizeZ: number): Model {
  const handle = bloom_gen_mesh_heightmap(imageHandle, sizeX, sizeY, sizeZ);
  return { handle };
}
