// Runtime schema validation for `WorldData` and `PrefabData`. Called by the
// loader (to guard against corrupted or hand-edited files) and by the saver
// (to catch editor bugs before they reach disk). Errors are returned as a
// list of human-readable strings rather than thrown, so the caller can log
// all problems at once instead of stopping at the first one.

import {
  WORLD_SCHEMA_VERSION,
  WorldData,
  PrefabData,
  EntityData,
  PrefabChild,
  TerrainData,
  TransformData,
  Vec3Lit,
  Vec4Lit,
} from './types';

export interface ValidationResult {
  ok: boolean;
  errors: string[];
}

export function validateWorld(w: WorldData): ValidationResult {
  const errors: string[] = [];

  if (typeof w.schemaVersion !== 'number') {
    errors.push('world.schemaVersion is missing or not a number');
  } else if (w.schemaVersion > WORLD_SCHEMA_VERSION) {
    errors.push(
      'world.schemaVersion ' + w.schemaVersion + ' is newer than engine version ' + WORLD_SCHEMA_VERSION,
    );
  }

  if (typeof w.name !== 'string' || w.name.length === 0) {
    errors.push('world.name must be a non-empty string');
  }
  if (typeof w.id !== 'string' || w.id.length === 0) {
    errors.push('world.id must be a non-empty string');
  }

  checkVec3(errors, 'world.bounds.min', w.bounds ? w.bounds.min : null);
  checkVec3(errors, 'world.bounds.max', w.bounds ? w.bounds.max : null);

  if (!w.environment) {
    errors.push('world.environment is missing');
  }

  if (w.terrain !== null && w.terrain !== undefined) {
    validateTerrain(errors, w.terrain);
  }

  if (!Array.isArray(w.entities)) {
    errors.push('world.entities must be an array');
  } else {
    const seenIds = new Set<string>();
    for (let i = 0; i < w.entities.length; i++) {
      validateEntity(errors, 'world.entities[' + i + ']', w.entities[i], seenIds);
    }
  }

  if (!Array.isArray(w.lights)) {
    errors.push('world.lights must be an array');
  } else {
    const seenLightIds = new Set<string>();
    for (let i = 0; i < w.lights.length; i++) {
      const l = w.lights[i];
      const path = 'world.lights[' + i + ']';
      if (typeof l.id !== 'string' || l.id.length === 0) {
        errors.push(path + '.id is missing');
      } else if (seenLightIds.has(l.id)) {
        errors.push(path + '.id "' + l.id + '" is a duplicate');
      } else {
        seenLightIds.add(l.id);
      }
      if (l.kind !== 'point') {
        errors.push(path + '.kind must be "point"');
      }
      checkVec3(errors, path + '.position', l.position);
      checkVec3(errors, path + '.color', l.color);
      if (typeof l.intensity !== 'number') errors.push(path + '.intensity must be a number');
      if (typeof l.range !== 'number') errors.push(path + '.range must be a number');
    }
  }

  if (!Array.isArray(w.water)) {
    errors.push('world.water must be an array');
  }
  if (!Array.isArray(w.rivers)) {
    errors.push('world.rivers must be an array');
  }

  return { ok: errors.length === 0, errors };
}

export function validatePrefab(p: PrefabData): ValidationResult {
  const errors: string[] = [];

  if (typeof p.schemaVersion !== 'number') {
    errors.push('prefab.schemaVersion is missing or not a number');
  } else if (p.schemaVersion > WORLD_SCHEMA_VERSION) {
    errors.push('prefab.schemaVersion ' + p.schemaVersion + ' is newer than engine');
  }

  if (typeof p.id !== 'string' || p.id.length === 0) {
    errors.push('prefab.id must be a non-empty string');
  }
  if (typeof p.name !== 'string') {
    errors.push('prefab.name must be a string');
  }

  if (!Array.isArray(p.children)) {
    errors.push('prefab.children must be an array');
  } else {
    const seenIds = new Set<string>();
    for (let i = 0; i < p.children.length; i++) {
      validatePrefabChild(errors, 'prefab.children[' + i + ']', p.children[i], seenIds);
    }
  }

  return { ok: errors.length === 0, errors };
}

// ---- helpers ---------------------------------------------------------------

function validateEntity(errors: string[], path: string, e: EntityData, seenIds: Set<string>): void {
  if (!e) {
    errors.push(path + ' is null');
    return;
  }
  if (typeof e.id !== 'string' || e.id.length === 0) {
    errors.push(path + '.id must be a non-empty string');
  } else if (seenIds.has(e.id)) {
    errors.push(path + '.id "' + e.id + '" is duplicated');
  } else {
    seenIds.add(e.id);
  }

  const hasModel = typeof e.modelRef === 'string' && e.modelRef.length > 0;
  const hasPrefab = typeof e.prefabRef === 'string' && e.prefabRef.length > 0;
  if (hasModel === hasPrefab) {
    errors.push(path + ' must have exactly one of modelRef or prefabRef');
  }

  validateTransform(errors, path + '.transform', e.transform);
}

function validatePrefabChild(errors: string[], path: string, c: PrefabChild, seenIds: Set<string>): void {
  if (!c) {
    errors.push(path + ' is null');
    return;
  }
  if (typeof c.id !== 'string' || c.id.length === 0) {
    errors.push(path + '.id must be a non-empty string');
  } else if (seenIds.has(c.id)) {
    errors.push(path + '.id "' + c.id + '" is duplicated');
  } else {
    seenIds.add(c.id);
  }

  const hasModel = typeof c.modelRef === 'string' && c.modelRef.length > 0;
  const hasPrefab = typeof c.prefabRef === 'string' && c.prefabRef.length > 0;
  if (hasModel === hasPrefab) {
    errors.push(path + ' must have exactly one of modelRef or prefabRef');
  }

  validateTransform(errors, path + '.transform', c.transform);
}

function validateTransform(errors: string[], path: string, t: TransformData): void {
  if (!t) {
    errors.push(path + ' is missing');
    return;
  }
  checkVec3(errors, path + '.position', t.position);
  checkVec3(errors, path + '.rotation', t.rotation);
  checkVec3(errors, path + '.scale', t.scale);
}

function validateTerrain(errors: string[], t: TerrainData): void {
  if (!(t.width > 0) || !(t.depth > 0)) {
    errors.push('terrain.width and terrain.depth must be positive');
    return;
  }
  if (!(t.cellSize > 0)) {
    errors.push('terrain.cellSize must be positive');
  }
  if (!Array.isArray(t.heights)) {
    errors.push('terrain.heights must be an array');
  } else if (t.heights.length !== t.width * t.depth) {
    errors.push(
      'terrain.heights length ' + t.heights.length +
        ' does not match width*depth = ' + (t.width * t.depth),
    );
  }
  checkVec3(errors, 'terrain.origin', t.origin);

  if (Array.isArray(t.layers)) {
    for (let i = 0; i < t.layers.length; i++) {
      const layer = t.layers[i];
      if (Array.isArray(layer.weights) && layer.weights.length !== t.width * t.depth) {
        errors.push('terrain.layers[' + i + '].weights length mismatch');
      }
    }
  }
}

function checkVec3(errors: string[], path: string, v: Vec3Lit | null): void {
  if (!v || !Array.isArray(v) || v.length !== 3) {
    errors.push(path + ' must be a length-3 number array');
    return;
  }
  if (typeof v[0] !== 'number' || typeof v[1] !== 'number' || typeof v[2] !== 'number') {
    errors.push(path + ' must contain numbers');
  }
}

// ---- unknown-field detection -------------------------------------------------
//
// The saver (`serialize.ts`) walks the schema by LITERAL KEY, so any field it
// does not know about is silently dropped on the first save. Validation
// deliberately tolerates unknown fields on load (a v3 file should still open in
// a v2 tool) — which means the only honest behavior is to tell the user LOUDLY
// at load time that those fields will not survive an edit-save cycle.
//
// Games that need to attach their own data must use the sanctioned extension
// points, which DO round-trip: `world.metadata`, `entity.userData`,
// `entity.tags`. Anything else is schema, and schema changes land in types.ts,
// serialize.ts, validate.ts, and version.ts together.

const WORLD_KEYS = ['schemaVersion', 'name', 'id', 'bounds', 'environment', 'terrain', 'entities', 'lights', 'water', 'rivers', 'metadata'];
const BOUNDS_KEYS = ['min', 'max'];
const ENVIRONMENT_KEYS = ['skyColor', 'ambientColor', 'ambientIntensity', 'sunDirection', 'sunColor', 'sunIntensity', 'fogStart', 'fogEnd', 'fogColor', 'shadowsEnabled'];
const TERRAIN_KEYS = ['width', 'depth', 'cellSize', 'origin', 'heights', 'layers'];
const TERRAIN_LAYER_KEYS = ['id', 'textureRef', 'weights', 'tileScale'];
const ENTITY_KEYS = ['id', 'name', 'modelRef', 'prefabRef', 'transform', 'tint', 'tags', 'userData'];
const TRANSFORM_KEYS = ['position', 'rotation', 'scale'];
const LIGHT_KEYS = ['id', 'name', 'kind', 'position', 'color', 'intensity', 'range'];
const WATER_KEYS = ['id', 'kind', 'center', 'size', 'surfaceHeight', 'color', 'waveAmplitude', 'waveSpeed'];
const RIVER_KEYS = ['id', 'controlPoints', 'widths', 'depth', 'flowSpeed', 'color'];
const PREFAB_KEYS = ['schemaVersion', 'id', 'name', 'children', 'bounds'];
const PREFAB_CHILD_KEYS = ['id', 'modelRef', 'prefabRef', 'transform', 'tint', 'tags'];

function pushUnknownKeys(obj: unknown, allowed: string[], path: string, out: string[]): void {
  if (obj === null || obj === undefined || typeof obj !== 'object') return;
  const keys = Object.keys(obj as Record<string, unknown>);
  for (let i = 0; i < keys.length; i++) {
    if (allowed.indexOf(keys[i]) < 0) {
      out.push(path + '.' + keys[i]);
    }
  }
}

// List every field in a parsed world document that is not part of the current
// schema, as dotted paths ("world.navmesh", "world.entities[3].loot"). Call on
// the object returned by `loadWorld` — JSON.parse keeps unknown keys, the
// static types just hide them. Empty result means a save is lossless.
export function listUnknownWorldFields(w: WorldData): string[] {
  const out: string[] = [];
  pushUnknownKeys(w, WORLD_KEYS, 'world', out);
  pushUnknownKeys(w.bounds, BOUNDS_KEYS, 'world.bounds', out);
  pushUnknownKeys(w.environment, ENVIRONMENT_KEYS, 'world.environment', out);
  if (w.terrain !== null && w.terrain !== undefined) {
    pushUnknownKeys(w.terrain, TERRAIN_KEYS, 'world.terrain', out);
    if (Array.isArray(w.terrain.layers)) {
      for (let i = 0; i < w.terrain.layers.length; i++) {
        pushUnknownKeys(w.terrain.layers[i], TERRAIN_LAYER_KEYS, 'world.terrain.layers[' + i + ']', out);
      }
    }
  }
  if (Array.isArray(w.entities)) {
    for (let i = 0; i < w.entities.length; i++) {
      const e = w.entities[i];
      pushUnknownKeys(e, ENTITY_KEYS, 'world.entities[' + i + ']', out);
      if (e) pushUnknownKeys(e.transform, TRANSFORM_KEYS, 'world.entities[' + i + '].transform', out);
    }
  }
  if (Array.isArray(w.lights)) {
    for (let i = 0; i < w.lights.length; i++) {
      pushUnknownKeys(w.lights[i], LIGHT_KEYS, 'world.lights[' + i + ']', out);
    }
  }
  if (Array.isArray(w.water)) {
    for (let i = 0; i < w.water.length; i++) {
      pushUnknownKeys(w.water[i], WATER_KEYS, 'world.water[' + i + ']', out);
    }
  }
  if (Array.isArray(w.rivers)) {
    for (let i = 0; i < w.rivers.length; i++) {
      pushUnknownKeys(w.rivers[i], RIVER_KEYS, 'world.rivers[' + i + ']', out);
    }
  }
  return out;
}

// Prefab counterpart of `listUnknownWorldFields`.
export function listUnknownPrefabFields(p: PrefabData): string[] {
  const out: string[] = [];
  pushUnknownKeys(p, PREFAB_KEYS, 'prefab', out);
  pushUnknownKeys(p.bounds, BOUNDS_KEYS, 'prefab.bounds', out);
  if (Array.isArray(p.children)) {
    for (let i = 0; i < p.children.length; i++) {
      const c = p.children[i];
      pushUnknownKeys(c, PREFAB_CHILD_KEYS, 'prefab.children[' + i + ']', out);
      if (c) pushUnknownKeys(c.transform, TRANSFORM_KEYS, 'prefab.children[' + i + '].transform', out);
    }
  }
  return out;
}

// Re-exported for use in loader/saver error formatting.
export function formatValidationErrors(errors: string[]): string {
  if (errors.length === 0) return '';
  let out = 'Validation failed with ' + errors.length + ' error(s):\n';
  for (let i = 0; i < errors.length; i++) {
    out = out + '  - ' + errors[i] + '\n';
  }
  return out;
}
