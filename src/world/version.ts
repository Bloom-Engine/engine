// Schema version management for `*.world.json` and `*.prefab.json` files.
//
// The current version number lives in `./types.ts` as `WORLD_SCHEMA_VERSION`.
// When the schema changes in a breaking way, bump that constant and add a
// migration step here. `migrateWorldData` and `migratePrefabData` are called
// by the loader before handing data to the rest of the pipeline.

import { WORLD_SCHEMA_VERSION, WorldData, PrefabData, LightData, Vec3Lit } from './types';

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

  if (from < 2) {
    migrateV1ToV2(raw);
  }

  raw.schemaVersion = WORLD_SCHEMA_VERSION;
  return raw;
}

// v1 → v2: lights become a top-level array.
//
// In v1 a light was an ordinary entity carrying `userData.kind = "point_light"`
// plus `range`, `color` ("r, g, b"), and `intensity` strings — a convention each
// game had to know about. Lift those entities into `world.lights` and drop them
// from `entities`, so the editor (and any other consumer) sees a light as a
// light. Worlds with no such entities just gain an empty array.
function migrateV1ToV2(raw: WorldData): void {
  if (!raw.lights) raw.lights = [];

  const kept: typeof raw.entities = [];

  for (let i = 0; i < raw.entities.length; i++) {
    const e = raw.entities[i];
    const kind = e.userData ? e.userData['kind'] : undefined;
    if (kind !== 'point_light') {
      kept.push(e);
      continue;
    }

    const light: LightData = {
      id: e.id,
      name: e.name,
      kind: 'point',
      position: [
        e.transform.position[0],
        e.transform.position[1],
        e.transform.position[2],
      ],
      color: parseColor(e.userData['color']),
      intensity: parseNumber(e.userData['intensity'], 1),
      range: parseNumber(e.userData['range'], 12),
    };
    raw.lights.push(light);
  }

  raw.entities = kept;
}

// userData values are strings; "1.0, 0.85, 0.55" is the colour convention.
function parseColor(s: string | undefined): Vec3Lit {
  if (s === undefined || s.length === 0) return [1, 1, 1];
  const parts = s.split(',');
  if (parts.length !== 3) return [1, 1, 1];
  const r = parseFloat(parts[0]);
  const g = parseFloat(parts[1]);
  const b = parseFloat(parts[2]);
  if (r !== r || g !== g || b !== b) return [1, 1, 1];
  return [r, g, b];
}

function parseNumber(s: string | undefined, fallback: number): number {
  if (s === undefined || s.length === 0) return fallback;
  const v = parseFloat(s);
  return v === v ? v : fallback;
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
