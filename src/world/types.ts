// Shared world schema — consumed by the Bloom world editor and by every game
// that loads `*.world.json` files. All fields are chosen to round-trip cleanly
// through `JSON.stringify` / `JSON.parse`, so positions, rotations, colors, and
// matrices are stored as plain number arrays rather than `{x,y,z}` objects.
// The engine's runtime `Vec3` / `Mat4` / `BoundingBox` types (in `core/types.ts`)
// use a different shape; the loader in `./loader.ts` does the conversion.

// v2 (2026-07-11) — lights became a first-class top-level array. Before that,
// games carried them as entities tagged `userData.kind === "point_light"`, so
// only the game knew they were lights: the editor could not light its preview,
// and every game re-invented the same convention. Sun/ambient/fog were already
// first-class in `environment`; point lights now sit beside them.
// `migrateWorldData` converts v1 point_light entities automatically.
export const WORLD_SCHEMA_VERSION = 2;

// Literal types for JSON-friendly serialization.
// Use `Vec3Lit` in serialized data; convert to engine `Vec3` at load time.
export type Vec3Lit = [number, number, number];
export type Vec4Lit = [number, number, number, number];

// Column-major 4x4 matrix, same convention as `engine/src/core/types.ts` Mat4.
// Stored as a flat length-16 number array so it serializes as JSON.
export type Mat4Lit = number[];

// Top-level world document. One `*.world.json` file holds exactly one of these.
export interface WorldData {
  schemaVersion: number;            // Must equal WORLD_SCHEMA_VERSION on save.
  name: string;                     // Human-readable display name.
  id: string;                       // Stable slug, e.g. "garden_main".
  bounds: Bounds;                   // Axis-aligned bounding box in world space.
  environment: EnvironmentData;
  terrain: TerrainData | null;      // null for games that don't use terrain.
  entities: EntityData[];
  lights: LightData[];              // Point lights. Sun/ambient live in `environment`.
  water: WaterVolume[];
  rivers: RiverSpline[];
  metadata: Record<string, string>; // Game-specific extensibility (e.g. "gameId").
}

// A placed light. Only point lights for now — the sun and ambient are
// environment-wide and live in `EnvironmentData`, and spot lights are not yet
// in the renderer.
//
// Lights are engine-universal (every renderer knows what a point light is),
// which is why they are schema rather than `userData`: a spawner or a wave plan
// means nothing without the game, but a light means the same thing everywhere.
export interface LightData {
  id: string;                 // Stable within the world file, e.g. "light_0001".
  name: string;               // Display name.
  kind: "point";
  position: Vec3Lit;
  color: Vec3Lit;             // 0..1 RGB (the runtime API takes 0-255; loader converts).
  intensity: number;
  range: number;              // World units; beyond this the light contributes nothing.
}

export interface Bounds {
  min: Vec3Lit;
  max: Vec3Lit;
}

// Sky, lighting, and atmospheric settings applied when the world is loaded.
export interface EnvironmentData {
  skyColor: Vec3Lit;          // 0..1 RGB, used as clear color.
  ambientColor: Vec3Lit;      // 0..1 RGB.
  ambientIntensity: number;   // 0..1.
  sunDirection: Vec3Lit;      // Unit vector pointing from the sun.
  sunColor: Vec3Lit;          // 0..1 RGB.
  sunIntensity: number;
  fogStart: number;           // World-space distance where fog begins.
  fogEnd: number;             // World-space distance of full fog.
  fogColor: Vec3Lit;
  shadowsEnabled: boolean;
}

// Heightmap terrain. Row-major grid of float heights, indexed as z*width + x.
// Runtime consumers build a mesh via `buildHeightmapMesh` in `./terrain.ts`
// and sample via `sampleHeight` (bilinear).
export interface TerrainData {
  width: number;              // Grid cells along X (e.g. 128).
  depth: number;              // Grid cells along Z.
  cellSize: number;           // World units per cell, e.g. 1.0.
  origin: Vec3Lit;            // World-space position of the (0,0) corner.
  heights: number[];          // Length == width*depth, row-major, z*width + x.
  layers: TerrainLayer[];     // Splat texture layers; empty array if unused.
}

export interface TerrainLayer {
  id: string;                 // "grass", "dirt", "rock".
  textureRef: string;         // Relative asset path.
  weights: number[];          // Length == width*depth, 0..1 per cell.
  tileScale: number;          // UV tiling factor.
}

// A placed instance in the world. Exactly one of `modelRef` / `prefabRef` is
// non-null; the other is null. The editor enforces this invariant.
export interface EntityData {
  id: string;                 // Stable within the world file, e.g. "ent_0001".
  name: string;               // Display name; defaults to model basename.
  modelRef: string | null;    // Relative path, e.g. "models/tree_oak.glb".
  prefabRef: string | null;   // Prefab id, e.g. "small_house".
  transform: TransformData;
  tint: Vec4Lit | null;       // Optional per-instance RGBA color override.
  tags: string[];             // Game-defined, e.g. "climbable", "zone_marker".
  userData: Record<string, string>;  // Arbitrary game-specific key/value data.
}

// Transform expressed as TRS with Euler rotation for diffable JSON.
// The loader converts Euler -> quaternion / matrix as needed.
export interface TransformData {
  position: Vec3Lit;
  rotation: Vec3Lit;          // Euler radians, XYZ order.
  scale: Vec3Lit;             // Uniform scale is [s, s, s].
}

// Axis-aligned water volume with a wave-animated surface.
// M1 supports only `kind: "box"`; future: "mesh" for arbitrary shapes.
export interface WaterVolume {
  id: string;
  kind: "box";
  center: Vec3Lit;
  size: Vec3Lit;              // Full extents (not half-extents).
  surfaceHeight: number;      // World Y of the water surface.
  color: Vec4Lit;             // RGBA tint.
  waveAmplitude: number;
  waveSpeed: number;
}

// Catmull-Rom spline river with per-point width.
export interface RiverSpline {
  id: string;
  controlPoints: Vec3Lit[];   // At least 2 points.
  widths: number[];           // Same length as controlPoints.
  depth: number;              // Below the surface.
  flowSpeed: number;
  color: Vec4Lit;
}

// ---- Prefabs ----------------------------------------------------------------

// A prefab is a reusable composite object saved as its own `*.prefab.json`.
// Each child references either a raw .glb model or another prefab (nested).
// Cycles are detected at load time and rejected.
export interface PrefabData {
  schemaVersion: number;      // Must equal WORLD_SCHEMA_VERSION.
  id: string;                 // Stable slug, e.g. "small_house".
  name: string;               // Display name.
  children: PrefabChild[];
  bounds: Bounds;             // Cached AABB of the expanded prefab for previews.
}

// One child of a prefab. Exactly one of `modelRef` / `prefabRef` is non-null.
export interface PrefabChild {
  id: string;                 // Local id within the prefab, e.g. "wall_0".
  modelRef: string | null;
  prefabRef: string | null;   // Reference to another prefab (nested).
  transform: TransformData;
  tint: Vec4Lit | null;
  tags: string[];
}
