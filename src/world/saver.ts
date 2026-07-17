// World saver — serializes a `WorldData` to disk as pretty-printed JSON.
// Called by the editor's File -> Save / Save As commands and by any tooling
// that programmatically generates worlds.
//
// The saver always validates before writing so the editor can't produce
// corrupt files. On validation failure, returns false and populates the
// `errors` array; on success, writes the file and returns true.
//
// CRASH SAFETY (2026-07-17): a save is a user's level — the one file that
// must never be half-written. There is no rename FFI, so true atomic replace
// isn't available; instead every save goes through `safeWrite`:
//
//   1. write `<path>.tmp` and READ IT BACK — a byte mismatch fails the save
//      before the real file is touched (this is the check that would have
//      caught the era when writeFile wrote 0 bytes and reported success);
//   2. snapshot the current `<path>` to `<path>.bak`;
//   3. write `<path>`.
//
// A crash at any point leaves a good copy in `.bak` (the previous save) or
// `.tmp` (the new one, already verified) — losing data requires the disk to
// fail three times in a row. The `.tmp`/`.bak` siblings are cheap litter;
// gitignore them in game repos.

import { readFile, writeFile, fileExists } from '../core/index';
import { WORLD_SCHEMA_VERSION, WorldData, PrefabData } from './types';
import { validateWorld, validatePrefab, formatValidationErrors } from './validate';
import { serializeWorld, serializePrefab } from './serialize';

export interface SaveResult {
  ok: boolean;
  errors: string[];
}

// Verified, backup-keeping write. Returns an error string, or null on success.
function safeWrite(path: string, json: string): string | null {
  const tmpPath = path + '.tmp';
  if (!writeFile(tmpPath, json)) {
    return 'could not write ' + tmpPath;
  }
  const readBack = readFile(tmpPath);
  if (readBack !== json) {
    return 'readback mismatch on ' + tmpPath + ' (wrote ' + json.length +
      ' chars, read ' + (readBack ? readBack.length : 0) + ') — disk or FFI fault, original untouched';
  }

  if (fileExists(path)) {
    const prev = readFile(path);
    if (prev && prev.length > 0) {
      // Best-effort: a failed backup shouldn't block the save itself.
      writeFile(path + '.bak', prev);
    }
  }

  if (!writeFile(path, json)) {
    return 'could not write ' + path + ' — previous version is in ' + path +
      '.bak, the new one in ' + tmpPath;
  }
  return null;
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
  const err = safeWrite(path, json);
  if (err !== null) {
    return { ok: false, errors: ['saveWorld: ' + err] };
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
  const err = safeWrite(path, json);
  if (err !== null) {
    return { ok: false, errors: ['savePrefab: ' + err] };
  }
  return { ok: true, errors: [] };
}

// Convenience: format a SaveResult's errors as a single human-readable string.
// The editor uses this for status bar / dialog messages.
export function formatSaveError(result: SaveResult): string {
  if (result.ok) return '';
  return formatValidationErrors(result.errors);
}
