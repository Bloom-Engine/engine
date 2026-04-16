// Schema version management for `*.world.json` and `*.prefab.json` files.
//
// The current version number lives in `./types.ts` as `WORLD_SCHEMA_VERSION`.
// When the schema changes in a breaking way, bump that constant and add a
// migration step here. `migrateWorldData` and `migratePrefabData` are called
// by the loader before handing data to the rest of the pipeline.

import { WORLD_SCHEMA_VERSION, WorldData, PrefabData } from './types';

// Migrate a parsed world document to the current schema version. Returns the
// same object (mutated in place) for convenience. Logs a warning when the
// input is from a future version — in that case we proceed anyway and hope
// the extra fields are ignored, since we can't forward-migrate.
export function migrateWorldData(raw: WorldData): WorldData {
  const from = raw.schemaVersion | 0;

  if (from === WORLD_SCHEMA_VERSION) {
    return raw;
  }

  if (from > WORLD_SCHEMA_VERSION) {
    // Future version — we may be missing fields. Warn and continue.
    return raw;
  }

  // from < WORLD_SCHEMA_VERSION. Apply migration steps in order.
  // When bumping WORLD_SCHEMA_VERSION, add a step here of the form:
  //   if (raw.schemaVersion < N) { migrateWorldV(N-1, N, raw); raw.schemaVersion = N; }

  raw.schemaVersion = WORLD_SCHEMA_VERSION;
  return raw;
}

export function migratePrefabData(raw: PrefabData): PrefabData {
  const from = raw.schemaVersion | 0;

  if (from === WORLD_SCHEMA_VERSION) {
    return raw;
  }

  if (from > WORLD_SCHEMA_VERSION) {
    return raw;
  }

  raw.schemaVersion = WORLD_SCHEMA_VERSION;
  return raw;
}
