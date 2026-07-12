// World saver — serializes a `WorldData` to disk as pretty-printed JSON.
// Called by the editor's File -> Save / Save As commands and by any tooling
// that programmatically generates worlds.
//
// The saver always validates before writing so the editor can't produce
// corrupt files. On validation failure, returns false and populates the
// `errors` array; on success, writes the file and returns true.

import { writeFile } from '../core/index';
import { WORLD_SCHEMA_VERSION, WorldData, PrefabData } from './types';
import { validateWorld, validatePrefab, formatValidationErrors } from './validate';
import { serializeWorld, serializePrefab } from './serialize';

export interface SaveResult {
  ok: boolean;
  errors: string[];
}

// Write a world file. On success, returns `{ ok: true, errors: [] }`.
// On validation failure, returns the errors without touching the filesystem.
// On write failure (disk full, permissions), returns `{ ok: false, errors: [...] }`.
export function saveWorld(path: string, world: WorldData): SaveResult {
  // Always stamp the current schema version on save so stale copies don't
  // drift across editor sessions.
  world.schemaVersion = WORLD_SCHEMA_VERSION;

  const check = validateWorld(world);
  if (!check.ok) {
    return { ok: false, errors: check.errors };
  }

  // NOT JSON.stringify — see serialize.ts. On Perry 0.5.x it corrupts a large object
  // graph that came from JSON.parse, the corrupt string fails the FFI's UTF-8 check,
  // and writeFile then wrote a ZERO-BYTE FILE AND RETURNED SUCCESS. Saving a world
  // destroyed it, silently.
  const json = serializeWorld(world);
  const ok = writeFile(path, json);
  if (!ok) {
    return { ok: false, errors: ['writeFile failed for path: ' + path] };
  }
  return { ok: true, errors: [] };
}

// Write a prefab file. Same semantics as `saveWorld` but for prefabs.
export function savePrefab(path: string, prefab: PrefabData): SaveResult {
  prefab.schemaVersion = WORLD_SCHEMA_VERSION;

  const check = validatePrefab(prefab);
  if (!check.ok) {
    return { ok: false, errors: check.errors };
  }

  const json = serializePrefab(prefab);
  const ok = writeFile(path, json);
  if (!ok) {
    return { ok: false, errors: ['writeFile failed for path: ' + path] };
  }
  return { ok: true, errors: [] };
}

// Convenience: format a SaveResult's errors as a single human-readable string.
// The editor uses this for status bar / dialog messages.
export function formatSaveError(result: SaveResult): string {
  if (result.ok) return '';
  return formatValidationErrors(result.errors);
}
