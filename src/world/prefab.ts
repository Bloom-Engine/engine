// Prefab expansion and registry.
//
// A prefab is a reusable composite object — e.g. a house made of 4 walls + a
// roof + a door. Prefabs are authored in the editor and stored as individual
// `*.prefab.json` files on disk. When a world places a prefab (via an entity
// with `prefabRef` set), the runtime expands the prefab tree into a flat list
// of leaf glb placements with world-space transforms, and the loader spawns
// one scene node per leaf.
//
// Prefabs can nest: one prefab's child can reference another prefab. Cycles
// are detected at expansion time (A -> B -> A) and reported as errors; the
// offending child is treated as empty so the rest of the world still loads.

import { readFile } from '../core/index';
import {
  mat4Identity,
  mat4Multiply,
  mat4Translate,
  mat4RotateX,
  mat4RotateY,
  mat4RotateZ,
  mat4Scale,
} from '../math/index';
import { Mat4, Vec3 } from '../core/types';
import {
  PrefabData,
  PrefabChild,
  TransformData,
  Vec4Lit,
} from './types';
import { migratePrefabData } from './version';
import { validatePrefab, formatValidationErrors } from './validate';

// One leaf of an expanded prefab — a single glb to spawn at a world-space
// matrix. The loader turns each of these into one scene node.
export interface PrefabLeaf {
  modelRef: string;        // Always a glb path; prefabs are fully flattened.
  worldMatrix: Mat4;       // Column-major 4x4 in world space.
  tint: Vec4Lit | null;    // Inherited from ancestor unless a child overrides.
  tags: string[];          // Union of the leaf's own tags and ancestor tags.
  sourcePath: string;      // Dotted path for debugging: "small_house.wall_0".
}

// Registry of all prefabs known to a project. The editor populates it at
// startup by scanning `project.prefabsDir`; games populate it from their
// asset bundle. Keyed by `PrefabData.id`.
export interface PrefabRegistry {
  byId: Map<string, PrefabData>;
  getPrefab: (id: string) => PrefabData | null;
}

export function createPrefabRegistry(): PrefabRegistry {
  const byId = new Map<string, PrefabData>();
  return {
    byId: byId,
    getPrefab: function(id: string): PrefabData | null {
      const found = byId.get(id);
      return found ? found : null;
    },
  };
}

// Register a prefab in the registry. Overwrites any existing entry with the
// same id (hot-reload friendly).
export function registerPrefab(registry: PrefabRegistry, prefab: PrefabData): void {
  registry.byId.set(prefab.id, prefab);
}

// Load a single prefab file from disk, validate, migrate, and return it.
// Throws on parse or validation error.
export function loadPrefab(path: string): PrefabData {
  const text = readFile(path);
  if (!text || text.length === 0) {
    throw new Error('loadPrefab: file is empty or missing: ' + path);
  }

  let raw: PrefabData;
  try {
    raw = JSON.parse(text) as PrefabData;
  } catch (e) {
    throw new Error('loadPrefab: invalid JSON in ' + path + ': ' + (e as Error).message);
  }

  const migrated = migratePrefabData(raw);
  const check = validatePrefab(migrated);
  if (!check.ok) {
    throw new Error('loadPrefab: ' + path + '\n' + formatValidationErrors(check.errors));
  }
  return migrated;
}

// Recursively flatten a prefab into its leaf glb placements.
//
// @param registry       Registry to look up child prefab references.
// @param prefabId       Id of the prefab to expand.
// @param parentMatrix   World-space matrix to apply to all children.
// @param parentTint     Tint inherited from the placing entity (null if none).
// @param parentTags     Tags to propagate to each leaf.
// @param out            Output array of flattened leaves. Caller supplies.
// @param errors         Output error list (cycles, missing references).
// @param visited        Set of ancestor prefab ids, used for cycle detection.
// @param pathPrefix     Debug path for error messages, e.g. "world_root".
export function expandPrefab(
  registry: PrefabRegistry,
  prefabId: string,
  parentMatrix: Mat4,
  parentTint: Vec4Lit | null,
  parentTags: ReadonlyArray<string>,
  out: PrefabLeaf[],
  errors: string[],
  visited: Set<string>,
  pathPrefix: string,
): void {
  if (visited.has(prefabId)) {
    errors.push(
      'prefab cycle detected: ' + pathPrefix + ' -> ' + prefabId +
        ' (already in chain)',
    );
    return;
  }

  const prefab = registry.getPrefab(prefabId);
  if (!prefab) {
    errors.push('prefab not found: "' + prefabId + '" (referenced by ' + pathPrefix + ')');
    return;
  }

  visited.add(prefabId);

  for (let i = 0; i < prefab.children.length; i++) {
    const child = prefab.children[i];
    const childMatrix = mat4Multiply(parentMatrix, trsToMat4(child.transform));
    const childTint = child.tint !== null ? child.tint : parentTint;
    const childTags = mergeTags(parentTags, child.tags);
    const childPath = pathPrefix + '.' + child.id;

    if (child.modelRef !== null && child.modelRef.length > 0) {
      out.push({
        modelRef: child.modelRef,
        worldMatrix: childMatrix,
        tint: childTint,
        tags: childTags,
        sourcePath: childPath,
      });
    } else if (child.prefabRef !== null && child.prefabRef.length > 0) {
      expandPrefab(
        registry,
        child.prefabRef,
        childMatrix,
        childTint,
        childTags,
        out,
        errors,
        visited,
        childPath,
      );
    } else {
      errors.push('prefab child ' + childPath + ' has neither modelRef nor prefabRef');
    }
  }

  visited.delete(prefabId);
}

// Build a column-major 4x4 from TRS components. Duplicated here (rather than
// imported from `./loader.ts`) to keep the dependency graph flat — prefab is
// used *by* the loader, not the other way around.
function trsToMat4(t: TransformData): Mat4 {
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

function mergeTags(parent: ReadonlyArray<string>, child: ReadonlyArray<string>): string[] {
  if (parent.length === 0) return [...child];
  if (child.length === 0) return [...parent];
  const seen = new Set<string>();
  const out: string[] = [];
  for (let i = 0; i < parent.length; i++) {
    if (!seen.has(parent[i])) { seen.add(parent[i]); out.push(parent[i]); }
  }
  for (let i = 0; i < child.length; i++) {
    if (!seen.has(child[i])) { seen.add(child[i]); out.push(child[i]); }
  }
  return out;
}

// Build an empty prefab with sensible defaults. Editor calls this from
// "New Prefab".
export function createEmptyPrefab(id: string, name: string): PrefabData {
  return {
    schemaVersion: 1,
    id: id,
    name: name,
    children: [],
    bounds: { min: [0, 0, 0], max: [0, 0, 0] },
  };
}
