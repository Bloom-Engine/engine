// World loader — reads a `*.world.json` file from disk, validates it, and
// spawns scene nodes. Consumed by both the editor (to display the world) and
// by every Bloom game (to load its level at runtime).
//
// The loader has two entry points:
//
//   loadWorld(path)          — pure: reads + parses + migrates + validates.
//                              Returns a WorldData object. No scene side-effects.
//
//   instantiateWorld(w, ctx) — creates scene nodes, applies lighting, and
//                              returns a map of entity id -> scene node handle.
//                              The caller provides a context that resolves
//                              model references to loaded model handles.
//
// Splitting parse from instantiate lets the editor hold a `WorldData` in memory
// and re-sync scene nodes on edits, without re-reading the file every frame.

import { readFile } from '../core/index';
import {
  createSceneNode,
  attachModelToNode,
  setSceneNodeTransform,
  setSceneNodeColor,
  setSceneNodeVisible,
  setSceneNodeParent,
  updateSceneNodeGeometry,
  SceneNodeHandle,
  enableShadows,
  disableShadows,
  addDirectionalLight,
} from '../scene/index';
import {
  mat4Identity,
  mat4Translate,
  mat4RotateX,
  mat4RotateY,
  mat4RotateZ,
  mat4Scale,
} from '../math/index';
import { Mat4, Vec3 } from '../core/types';
import { spawnWaterVolume, spawnRiver } from './render';
import {
  WORLD_SCHEMA_VERSION,
  WorldData,
  EntityData,
  TransformData,
  Vec3Lit,
  Vec4Lit,
} from './types';
import { migrateWorldData } from './version';
import { validateWorld, formatValidationErrors, listUnknownWorldFields } from './validate';
import {
  PrefabRegistry,
  PrefabLeaf,
  expandPrefab,
} from './prefab';
import { buildHeightmapMesh } from './terrain';

// Context passed to `instantiateWorld`. The caller is responsible for loading
// GLB models and supplying handles via `getModelHandle`. This inverts the
// dependency so games and the editor can use their own asset-caching policy.
export interface InstantiateContext {
  // Resolve a `modelRef` (e.g. "models/tree_oak.glb") to a loaded model handle.
  // Return 0 if the model is unavailable — the entity is then skipped and a
  // warning is appended to `result.warnings`.
  getModelHandle: (modelRef: string) => number;

  // Prefab registry for expanding prefab entities. Passing null causes prefab
  // entities to be skipped (with a warning).
  prefabRegistry: PrefabRegistry | null;

  // Called once per successfully spawned entity. The editor uses this hook to
  // populate its `HandleMap` (EntityId <-> SceneNodeHandle bi-map).
  onEntitySpawned?: (entityId: string, handle: SceneNodeHandle) => void;

  // Called once for the terrain scene node, if the world has a terrain.
  // The editor uses this to track the terrain handle for brush updates.
  onTerrainSpawned?: (handle: SceneNodeHandle) => void;
}

export interface InstantiateResult {
  // One handle per spawned world entity. Prefab entities yield a root group
  // node whose children are the expanded prefab leaves.
  entityHandles: Map<string, SceneNodeHandle>;

  // Scene node handle for the terrain mesh, or 0 if the world has no terrain.
  terrainHandle: SceneNodeHandle;

  // Handles for water volumes and rivers, index-aligned with `world.water` and
  // `world.rivers`. Arrays rather than Maps on purpose: Perry 0.5.x miscompiles
  // interfaces that declare more than one Map field (see the editor's
  // docs/perry-map-size-av.md), and this interface already carries one.
  waterHandles: SceneNodeHandle[];
  riverHandles: SceneNodeHandle[];

  // Non-fatal problems encountered during instantiation: missing models,
  // unresolved prefab references, cycles. The world still instantiates,
  // with the offending entities skipped.
  warnings: string[];
}

// Read, parse, migrate, and validate a world file. Throws on parse error or
// validation failure — the caller can catch and present the error message.
export function loadWorld(path: string): WorldData {
  const text = readFile(path);
  if (!text || text.length === 0) {
    throw new Error('loadWorld: file is empty or missing: ' + path);
  }

  let raw: WorldData;
  try {
    raw = JSON.parse(text) as WorldData;
  } catch (e) {
    throw new Error('loadWorld: invalid JSON in ' + path + ': ' + (e as Error).message);
  }

  const migrated = migrateWorldData(raw);
  const check = validateWorld(migrated);
  if (!check.ok) {
    throw new Error('loadWorld: ' + path + '\n' + formatValidationErrors(check.errors));
  }

  // Unknown fields survive load (JSON.parse keeps them) but NOT save — the
  // schema-explicit saver drops them. Warn loudly rather than lose data
  // silently; game-specific data belongs in metadata/userData/tags.
  // console.error, not console.log: stdout is block-buffered under Perry and
  // lost if the process later crashes.
  const unknown = listUnknownWorldFields(migrated);
  for (let i = 0; i < unknown.length; i++) {
    console.error(
      'loadWorld: WARNING: ' + path + ' contains unknown field "' + unknown[i] +
      '" — not part of schema v' + WORLD_SCHEMA_VERSION +
      '; it will be DROPPED if this file is saved. Extensions belong in metadata/userData/tags.',
    );
  }

  return migrated;
}

// Spawn scene nodes for every entity in the world and apply environment
// settings (lighting, shadows). Terrain, water volumes, and rivers are spawned
// when present — water and rivers via the shared helpers in ./render.ts, which
// the editor uses too so the two never diverge.
export function instantiateWorld(world: WorldData, ctx: InstantiateContext): InstantiateResult {
  const result: InstantiateResult = {
    entityHandles: new Map<string, SceneNodeHandle>(),
    terrainHandle: 0,
    waterHandles: [],
    riverHandles: [],
    warnings: [],
  };

  // Apply environment first so the first frame renders with correct lighting.
  applyEnvironment(world);

  // Terrain before entities so entities can drop onto it visually.
  if (world.terrain) {
    result.terrainHandle = spawnTerrain(world);
    if (result.terrainHandle !== 0 && ctx.onTerrainSpawned) {
      ctx.onTerrainSpawned(result.terrainHandle);
    }
  }

  // Spawn each entity. Prefab entities are expanded into a root group node
  // with one child per leaf glb.
  for (let i = 0; i < world.entities.length; i++) {
    const entity = world.entities[i];
    const handle = spawnEntity(entity, ctx, result.warnings);
    if (handle !== 0) {
      result.entityHandles.set(entity.id, handle);
      if (ctx.onEntitySpawned) {
        ctx.onEntitySpawned(entity.id, handle);
      }
    }
  }

  // Water and rivers render through the shared helpers in ./render.ts, so a
  // game and the editor produce identical geometry and materials.
  for (let i = 0; i < world.water.length; i++) {
    const handle = spawnWaterVolume(world.water[i]);
    result.waterHandles.push(handle);
    if (handle === 0) {
      result.warnings.push('water volume "' + world.water[i].id + '" failed to spawn');
    }
  }

  for (let i = 0; i < world.rivers.length; i++) {
    const river = world.rivers[i];
    const handle = spawnRiver(river);
    result.riverHandles.push(handle);
    if (handle === 0) {
      result.warnings.push(
        'river "' + river.id + '" failed to spawn (needs at least 2 control points)',
      );
    }
  }

  return result;
}

// Build the terrain mesh from the heightmap grid and upload it to a dedicated
// scene node. Called once per world load; the editor's brush tool re-uploads
// the mesh on each stroke via `updateSceneNodeGeometry` directly.
function spawnTerrain(world: WorldData): SceneNodeHandle {
  if (!world.terrain) return 0;
  const mesh = buildHeightmapMesh(world.terrain);
  const node = createSceneNode();
  updateSceneNodeGeometry(node, mesh.vertices, mesh.indices);
  setSceneNodeTransform(node, mat4Identity());
  setSceneNodeVisible(node, true);
  return node;
}

// ---- entity spawning -------------------------------------------------------

function spawnEntity(
  entity: EntityData,
  ctx: InstantiateContext,
  warnings: string[],
): SceneNodeHandle {
  if (entity.modelRef !== null && entity.modelRef.length > 0) {
    return spawnModelEntity(entity, ctx, warnings);
  }

  if (entity.prefabRef !== null && entity.prefabRef.length > 0) {
    return spawnPrefabEntity(entity, ctx, warnings);
  }

  warnings.push('entity ' + entity.id + ' has neither modelRef nor prefabRef — skipped');
  return 0;
}

function spawnModelEntity(
  entity: EntityData,
  ctx: InstantiateContext,
  warnings: string[],
): SceneNodeHandle {
  const modelRef = entity.modelRef as string;
  const modelHandle = ctx.getModelHandle(modelRef);
  if (modelHandle === 0) {
    warnings.push('entity ' + entity.id + ' references unknown model "' + modelRef + '" — skipped');
    return 0;
  }

  const node = createSceneNode();
  attachModelToNode(node, modelHandle, 0);
  setSceneNodeTransform(node, trsToMat4(entity.transform));

  if (entity.tint !== null) {
    applyTint(node, entity.tint);
  }

  setSceneNodeVisible(node, true);
  return node;
}

function spawnPrefabEntity(
  entity: EntityData,
  ctx: InstantiateContext,
  warnings: string[],
): SceneNodeHandle {
  if (!ctx.prefabRegistry) {
    warnings.push(
      'entity ' + entity.id + ' references prefab "' + entity.prefabRef + '" but no PrefabRegistry was provided — skipped',
    );
    return 0;
  }

  // Create a root group node at the entity's transform. Expanded leaves are
  // parented to this root so the editor gizmo can move the whole prefab as
  // one unit.
  const root = createSceneNode();
  const rootMatrix = trsToMat4(entity.transform);
  setSceneNodeTransform(root, rootMatrix);
  setSceneNodeVisible(root, true);

  // Expand the prefab into its leaf glbs and spawn one scene node per leaf.
  // `expandPrefab` accumulates leaves into `leaves` and any errors into a
  // local list which we fold into the top-level warnings.
  const leaves: PrefabLeaf[] = [];
  const expandErrors: string[] = [];
  const visited = new Set<string>();
  expandPrefab(
    ctx.prefabRegistry,
    entity.prefabRef as string,
    mat4Identity(),
    entity.tint,
    entity.tags,
    leaves,
    expandErrors,
    visited,
    'world.entities[' + entity.id + ']',
  );

  for (let i = 0; i < expandErrors.length; i++) warnings.push(expandErrors[i]);

  for (let i = 0; i < leaves.length; i++) {
    const leaf = leaves[i];
    const modelHandle = ctx.getModelHandle(leaf.modelRef);
    if (modelHandle === 0) {
      warnings.push(
        'prefab leaf ' + leaf.sourcePath + ' references unknown model "' + leaf.modelRef + '" — skipped',
      );
      continue;
    }
    const leafNode = createSceneNode();
    attachModelToNode(leafNode, modelHandle, 0);
    setSceneNodeTransform(leafNode, leaf.worldMatrix);
    if (leaf.tint !== null) applyTint(leafNode, leaf.tint);
    setSceneNodeParent(leafNode, root);
    setSceneNodeVisible(leafNode, true);
  }

  return root;
}

// ---- environment -----------------------------------------------------------

function applyEnvironment(world: WorldData): void {
  const env = world.environment;
  if (!env) return;

  // Primary directional light (sun). Additional lights can be added by the
  // game/editor through `addDirectionalLight` / `addPointLight` directly.
  addDirectionalLight(
    env.sunDirection[0], env.sunDirection[1], env.sunDirection[2],
    env.sunColor[0] * env.sunIntensity,
    env.sunColor[1] * env.sunIntensity,
    env.sunColor[2] * env.sunIntensity,
    env.sunIntensity,
  );

  if (env.shadowsEnabled) {
    enableShadows();
  } else {
    disableShadows();
  }
}

// ---- TRS -> column-major 4x4 -----------------------------------------------

// Build a column-major 4x4 matrix from TRS: M = T * Rz * Ry * Rx * S.
// The rotation order matches glTF / Blender / most game engines (XYZ Euler,
// applied right-to-left so X spin happens first in object space).
export function trsToMat4(t: TransformData): Mat4 {
  const pos: Vec3 = { x: t.position[0], y: t.position[1], z: t.position[2] };
  const scl: Vec3 = { x: t.scale[0], y: t.scale[1], z: t.scale[2] };

  let m = mat4Identity();
  m = mat4Translate(m, pos);
  m = mat4RotateZ(m, t.rotation[2]);
  m = mat4RotateY(m, t.rotation[1]);
  m = mat4RotateX(m, t.rotation[0]);
  m = mat4Scale(m, scl);
  return m;
}

// Apply an RGBA tint to a scene node. Pulled into a helper so it can be reused
// by prefab expansion (where children inherit the parent entity's tint unless
// they override it).
function applyTint(node: SceneNodeHandle, tint: Vec4Lit): void {
  // World-format tints stay 0-1 floats (serialized data, unchanged);
  // the runtime API takes 0-255.
  setSceneNodeColor(node, tint[0] * 255, tint[1] * 255, tint[2] * 255, tint[3] * 255);
}

// ---- factory helpers used by the editor to create new data ----------------

// Build an empty world with sensible defaults. The editor calls this from
// File -> New. Games should prefer `loadWorld` from a file on disk.
export function createEmptyWorld(id: string, name: string): WorldData {
  return {
    schemaVersion: WORLD_SCHEMA_VERSION,
    name: name,
    id: id,
    bounds: { min: [-50, -10, -50], max: [50, 50, 50] },
    environment: {
      skyColor: [0.53, 0.81, 0.92],
      ambientColor: [1.0, 1.0, 1.0],
      ambientIntensity: 0.35,
      sunDirection: [-0.5, -1.0, -0.3],
      sunColor: [1.0, 0.95, 0.85],
      sunIntensity: 1.0,
      fogStart: 40,
      fogEnd: 120,
      fogColor: [0.7, 0.8, 0.9],
      shadowsEnabled: true,
    },
    terrain: null,
    entities: [],
    lights: [],
    water: [],
    rivers: [],
    metadata: {},
  };
}

// Build an empty entity with default transform. The editor calls this when
// the place tool drops a model at a hit point.
export function createEntity(id: string, modelRef: string, position: Vec3Lit): EntityData {
  return {
    id: id,
    name: basename(modelRef),
    modelRef: modelRef,
    prefabRef: null,
    transform: {
      position: position,
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
    },
    tint: null,
    tags: [],
    userData: {},
  };
}

function basename(path: string): string {
  const slash = path.lastIndexOf('/');
  const dot = path.lastIndexOf('.');
  const start = slash < 0 ? 0 : slash + 1;
  const end = dot < start ? path.length : dot;
  return path.substring(start, end);
}
