// Terrain helpers for the shared world module.
//
// Terrain in a `WorldData` is a uniform grid heightmap: `width * depth` float
// heights, row-major with `z*width + x`. This file provides the operations
// that both the editor (to sculpt / display) and games (to render / sample)
// need:
//
//   buildHeightmapMesh  — grid floats -> flat vertex/index arrays suitable
//                         for `updateSceneNodeGeometry`. Vertex layout matches
//                         the scene graph's expected 12-float stride:
//                         [x, y, z, nx, ny, nz, r, g, b, a, u, v].
//   sampleHeight         — bilinear interpolation at arbitrary world-space xz.
//                          Used by character controllers and the place tool
//                          to drop entities onto the terrain.
//   raycastTerrain       — mouse ray -> terrain hit point + cell indices. Used
//                          by the brush tool to find where the user is painting.
//   defaultTerrain       — a flat 128x128 terrain centered at origin.

import { TerrainData, Vec3Lit } from './types';

// Vertex stride in floats (matches scene graph expectation).
// See `bloom/engine/src/scene/index.ts` updateSceneNodeGeometry docs.
const STRIDE = 12;

// Default grid size for new worlds. Large enough for meaningful terrain,
// small enough to rebuild in <1ms on any modern CPU.
const DEFAULT_WIDTH = 128;
const DEFAULT_DEPTH = 128;

// Build a flat-shaded triangle mesh from a heightmap. Returns two number
// arrays that can be passed directly to `updateSceneNodeGeometry`. Computes
// per-vertex normals from finite differences of the 4 neighbouring heights.
//
// Vertex count: width * depth
// Triangle count: (width - 1) * (depth - 1) * 2
//
// For a 128x128 grid that's 16,384 vertices and 32,258 triangles — well
// within what one scene node can hold.
export function buildHeightmapMesh(t: TerrainData): { vertices: number[]; indices: number[] } {
  const width = t.width;
  const depth = t.depth;
  const cellSize = t.cellSize;
  const originX = t.origin[0];
  const originY = t.origin[1];
  const originZ = t.origin[2];
  const heights = t.heights;

  const vertexCount = width * depth;
  const vertices: number[] = new Array<number>(vertexCount * STRIDE);
  const indices: number[] = new Array<number>((width - 1) * (depth - 1) * 6);

  // Default vertex color (grayscale stone). Editor / game can tint per-layer
  // via the splat weights once that pipeline lands.
  const cr = 0.55;
  const cg = 0.60;
  const cb = 0.50;
  const ca = 1.0;

  // Vertex pass.
  for (let z = 0; z < depth; z++) {
    for (let x = 0; x < width; x++) {
      const idx = z * width + x;
      const h = heights[idx];

      // World-space position.
      const wx = originX + x * cellSize;
      const wy = originY + h;
      const wz = originZ + z * cellSize;

      // Finite-difference normal (central differences, clamped at edges).
      const xm = x > 0 ? heights[z * width + (x - 1)] : h;
      const xp = x < width - 1 ? heights[z * width + (x + 1)] : h;
      const zm = z > 0 ? heights[(z - 1) * width + x] : h;
      const zp = z < depth - 1 ? heights[(z + 1) * width + x] : h;

      const dx = (xp - xm) / (2 * cellSize);
      const dz = (zp - zm) / (2 * cellSize);
      // Normal of y = f(x, z) is (-df/dx, 1, -df/dz), normalized.
      const nx = -dx;
      const ny = 1.0;
      const nz = -dz;
      const invLen = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz);

      const base = idx * STRIDE;
      vertices[base + 0] = wx;
      vertices[base + 1] = wy;
      vertices[base + 2] = wz;
      vertices[base + 3] = nx * invLen;
      vertices[base + 4] = ny * invLen;
      vertices[base + 5] = nz * invLen;
      vertices[base + 6] = cr;
      vertices[base + 7] = cg;
      vertices[base + 8] = cb;
      vertices[base + 9] = ca;
      vertices[base + 10] = x / (width - 1);
      vertices[base + 11] = z / (depth - 1);
    }
  }

  // Index pass — two triangles per cell, wound counter-clockwise when looking
  // down the -Y axis.
  let idxWrite = 0;
  for (let z = 0; z < depth - 1; z++) {
    for (let x = 0; x < width - 1; x++) {
      const i00 = z * width + x;
      const i10 = i00 + 1;
      const i01 = i00 + width;
      const i11 = i01 + 1;

      indices[idxWrite++] = i00;
      indices[idxWrite++] = i01;
      indices[idxWrite++] = i11;

      indices[idxWrite++] = i00;
      indices[idxWrite++] = i11;
      indices[idxWrite++] = i10;
    }
  }

  return { vertices: vertices, indices: indices };
}

// Bilinear sample of the terrain at world-space (wx, wz). Returns the world
// Y of the surface at that point, including the terrain's origin offset.
// Points outside the grid clamp to the nearest edge cell.
//
// Used by:
//   - character controllers ("what height is the player standing on?")
//   - the place tool (drop an entity onto the terrain surface)
//   - the brush tool (show a cursor ring at the correct height)
export function sampleHeight(t: TerrainData, wx: number, wz: number): number {
  const fx = (wx - t.origin[0]) / t.cellSize;
  const fz = (wz - t.origin[2]) / t.cellSize;

  const x0 = clamp(Math.floor(fx), 0, t.width - 1);
  const z0 = clamp(Math.floor(fz), 0, t.depth - 1);
  const x1 = clamp(x0 + 1, 0, t.width - 1);
  const z1 = clamp(z0 + 1, 0, t.depth - 1);

  const tx = clamp(fx - x0, 0, 1);
  const tz = clamp(fz - z0, 0, 1);

  const h00 = t.heights[z0 * t.width + x0];
  const h10 = t.heights[z0 * t.width + x1];
  const h01 = t.heights[z1 * t.width + x0];
  const h11 = t.heights[z1 * t.width + x1];

  const h0 = h00 * (1 - tx) + h10 * tx;
  const h1 = h01 * (1 - tx) + h11 * tx;
  return t.origin[1] + h0 * (1 - tz) + h1 * tz;
}

export interface TerrainRaycastHit {
  hit: boolean;
  point: Vec3Lit;
  // Grid cell the hit falls into. -1 when there is no hit.
  cellX: number;
  cellZ: number;
  distance: number;
}

// March a ray through the terrain looking for the first intersection with the
// surface. This is not a closed-form solution — it's a simple iterative
// marcher that steps along the ray in small increments and samples the height
// at each step. Good enough for the brush tool, where the player is always
// looking at the ground from above.
//
// @param origin  Ray origin in world space.
// @param dir     Ray direction (should be normalized for accurate distances).
// @param maxDist Maximum distance to march before giving up.
// @param step    Step size. Smaller = more accurate, larger = faster.
export function raycastTerrain(
  t: TerrainData,
  origin: Vec3Lit,
  dir: Vec3Lit,
  maxDist: number,
  step: number,
): TerrainRaycastHit {
  // Early-out: ray must be pointing roughly downward (negative Y component)
  // for the heightmap to be visible from above.
  if (dir[1] >= 0) {
    // Still allow it in case origin is below terrain, but clamp iterations.
  }

  let d = 0;
  let prevAbove = origin[1] > sampleHeight(t, origin[0], origin[2]);

  while (d < maxDist) {
    const px = origin[0] + dir[0] * d;
    const py = origin[1] + dir[1] * d;
    const pz = origin[2] + dir[2] * d;

    const h = sampleHeight(t, px, pz);
    const above = py > h;

    if (above !== prevAbove) {
      // Crossed the surface. Refine with one binary-search step for accuracy.
      let lo = d - step;
      let hi = d;
      for (let i = 0; i < 6; i++) {
        const mid = (lo + hi) * 0.5;
        const mx = origin[0] + dir[0] * mid;
        const my = origin[1] + dir[1] * mid;
        const mz = origin[2] + dir[2] * mid;
        const mh = sampleHeight(t, mx, mz);
        if ((my > mh) === prevAbove) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const finalD = (lo + hi) * 0.5;
      const hitX = origin[0] + dir[0] * finalD;
      const hitY = origin[1] + dir[1] * finalD;
      const hitZ = origin[2] + dir[2] * finalD;
      const cellX = clamp(Math.floor((hitX - t.origin[0]) / t.cellSize), 0, t.width - 1);
      const cellZ = clamp(Math.floor((hitZ - t.origin[2]) / t.cellSize), 0, t.depth - 1);
      return {
        hit: true,
        point: [hitX, hitY, hitZ],
        cellX: cellX,
        cellZ: cellZ,
        distance: finalD,
      };
    }

    d += step;
  }

  return { hit: false, point: [0, 0, 0], cellX: -1, cellZ: -1, distance: 0 };
}

// Build a flat 128x128 terrain centered at the origin. Used by the editor's
// "New World" command when the user chooses a terrain-backed template.
export function defaultTerrain(): TerrainData {
  const width = DEFAULT_WIDTH;
  const depth = DEFAULT_DEPTH;
  const cellSize = 1.0;
  const heights: number[] = new Array<number>(width * depth);
  for (let i = 0; i < heights.length; i++) heights[i] = 0;

  return {
    width: width,
    depth: depth,
    cellSize: cellSize,
    origin: [-(width * cellSize) / 2, 0, -(depth * cellSize) / 2],
    heights: heights,
    layers: [],
  };
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : (v > hi ? hi : v);
}
