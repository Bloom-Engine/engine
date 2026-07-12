// Hand-written JSON emitter for WorldData / PrefabData.
//
// WHY THIS EXISTS, AND WHY IT IS NOT `JSON.stringify`.
//
// `JSON.stringify` CORRUPTS a large object graph that came from `JSON.parse` on
// Perry 0.5.x. Minimal repro, no engine code involved:
//
//   const text = readFile('assets/worlds/arena_02.world.json');  // 324 KB, fine
//   const o    = JSON.parse(text);                                // fine
//   const re   = JSON.stringify(o, null, 2);
//   // -> `re` contains 5,296 characters above U+00FF, in a JSON document whose
//   //    source is almost pure ASCII. It is garbage.
//
// The corruption is invisible until the string crosses the FFI: the engine's
// `str_from_header` fails its UTF-8 check, returns "", and `writeFile` then writes
// a ZERO-BYTE FILE AND REPORTS SUCCESS.
//
// So `saveWorld` did not save. It emptied the file and said "ok". Every world the
// editor has ever "saved" was destroyed by saving it — which is why the editor has
// only ever been used to look at worlds that some other tool produced.
//
// It is not size (a fresh 1 MB world-shaped object stringifies fine), not floats,
// not nulls, not `Record` keys, not non-ASCII — every one of those was ruled out by
// probe. It is specifically the parsed graph. A manual deep clone does not escape it.
//
// So the save path does not get to use `JSON.stringify`. This emitter walks the
// schema by LITERAL KEY and builds the document by concatenation, which is the same
// discipline the shooter's `settings.ts` already adopted ("Perry's object handling
// is only trustworthy with literal keys") and which the probes show is reliable.
//
// Being schema-explicit is not only a workaround, it is a feature: an unknown field
// cannot be silently dropped by a serializer that never knew about it — the schema
// and the writer change together, or validation fails.

import {
  WorldData, PrefabData, PrefabChild, EntityData, LightData,
  WaterVolume, RiverSpline, TerrainData, TerrainLayer, EnvironmentData,
  TransformData, Vec3Lit, Vec4Lit,
} from './types';

// --- primitives --------------------------------------------------------------

/// JSON string escaping. Control characters must be escaped or the document is
/// invalid; everything else is emitted as-is (the file is UTF-8).
function str(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 34) out = out + '\\"';
    else if (c === 92) out = out + '\\\\';
    else if (c === 10) out = out + '\\n';
    else if (c === 13) out = out + '\\r';
    else if (c === 9) out = out + '\\t';
    else if (c === 8) out = out + '\\b';
    else if (c === 12) out = out + '\\f';
    else if (c < 32) {
      // \u00XX — the only escapes JSON *requires*.
      let h = c.toString(16);
      while (h.length < 4) h = '0' + h;
      out = out + '\\u' + h;
    } else {
      out = out + s.charAt(i);
    }
  }
  return out + '"';
}

/// Numbers. NaN and Infinity are not JSON; emitting them produces a file that will
/// not parse, so they become 0 rather than a document nobody can open.
function num(n: number): string {
  if (n !== n) return '0';                       // NaN
  if (n === Infinity || n === -Infinity) return '0';
  // Integers stay integers — `1` not `1.0` — so a hand-written world file and a
  // round-tripped one look the same in a diff.
  if (n === Math.floor(n) && Math.abs(n) < 1e15) return '' + n;
  return '' + n;
}

function bool(b: boolean): string { return b ? 'true' : 'false'; }

function vec3(v: Vec3Lit): string {
  return '[' + num(v[0]) + ', ' + num(v[1]) + ', ' + num(v[2]) + ']';
}
function vec4(v: Vec4Lit): string {
  return '[' + num(v[0]) + ', ' + num(v[1]) + ', ' + num(v[2]) + ', ' + num(v[3]) + ']';
}
function nums(a: number[]): string {
  let out = '[';
  for (let i = 0; i < a.length; i++) {
    if (i > 0) out = out + ', ';
    out = out + num(a[i]);
  }
  return out + ']';
}
function strs(a: string[]): string {
  let out = '[';
  for (let i = 0; i < a.length; i++) {
    if (i > 0) out = out + ', ';
    out = out + str(a[i]);
  }
  return out + ']';
}

function ind(n: number): string {
  let s = '';
  for (let i = 0; i < n; i++) s = s + '  ';
  return s;
}

/// A `Record<string, string>`. Keys come from the data, so this is the one place we
/// cannot use literal keys — but `Object.keys` on a *fresh* object is sound, and
/// callers hand us the parsed map directly.
function record(r: Record<string, string>, depth: number): string {
  const keys = Object.keys(r);
  if (keys.length === 0) return '{}';
  let out = '{\n';
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    out = out + ind(depth + 1) + str(k) + ': ' + str(r[k]);
    if (i < keys.length - 1) out = out + ',';
    out = out + '\n';
  }
  return out + ind(depth) + '}';
}

// --- schema ------------------------------------------------------------------

function transform(t: TransformData, d: number): string {
  return '{\n'
    + ind(d + 1) + '"position": ' + vec3(t.position) + ',\n'
    + ind(d + 1) + '"rotation": ' + vec3(t.rotation) + ',\n'
    + ind(d + 1) + '"scale": ' + vec3(t.scale) + '\n'
    + ind(d) + '}';
}

function entity(e: EntityData, d: number): string {
  let s = '{\n';
  s = s + ind(d + 1) + '"id": ' + str(e.id) + ',\n';
  s = s + ind(d + 1) + '"name": ' + str(e.name) + ',\n';
  s = s + ind(d + 1) + '"modelRef": ' + (e.modelRef === null ? 'null' : str(e.modelRef)) + ',\n';
  s = s + ind(d + 1) + '"prefabRef": ' + (e.prefabRef === null ? 'null' : str(e.prefabRef)) + ',\n';
  s = s + ind(d + 1) + '"transform": ' + transform(e.transform, d + 1) + ',\n';
  s = s + ind(d + 1) + '"tint": ' + (e.tint === null ? 'null' : vec4(e.tint)) + ',\n';
  s = s + ind(d + 1) + '"tags": ' + strs(e.tags) + ',\n';
  s = s + ind(d + 1) + '"userData": ' + record(e.userData, d + 1) + '\n';
  return s + ind(d) + '}';
}

function light(l: LightData, d: number): string {
  return '{\n'
    + ind(d + 1) + '"id": ' + str(l.id) + ',\n'
    + ind(d + 1) + '"name": ' + str(l.name) + ',\n'
    + ind(d + 1) + '"kind": ' + str(l.kind) + ',\n'
    + ind(d + 1) + '"position": ' + vec3(l.position) + ',\n'
    + ind(d + 1) + '"color": ' + vec3(l.color) + ',\n'
    + ind(d + 1) + '"intensity": ' + num(l.intensity) + ',\n'
    + ind(d + 1) + '"range": ' + num(l.range) + '\n'
    + ind(d) + '}';
}

function water(w: WaterVolume, d: number): string {
  return '{\n'
    + ind(d + 1) + '"id": ' + str(w.id) + ',\n'
    + ind(d + 1) + '"kind": ' + str(w.kind) + ',\n'
    + ind(d + 1) + '"center": ' + vec3(w.center) + ',\n'
    + ind(d + 1) + '"size": ' + vec3(w.size) + ',\n'
    + ind(d + 1) + '"surfaceHeight": ' + num(w.surfaceHeight) + ',\n'
    + ind(d + 1) + '"color": ' + vec4(w.color) + ',\n'
    + ind(d + 1) + '"waveAmplitude": ' + num(w.waveAmplitude) + ',\n'
    + ind(d + 1) + '"waveSpeed": ' + num(w.waveSpeed) + '\n'
    + ind(d) + '}';
}

function river(r: RiverSpline, d: number): string {
  let pts = '[';
  for (let i = 0; i < r.controlPoints.length; i++) {
    if (i > 0) pts = pts + ', ';
    pts = pts + vec3(r.controlPoints[i]);
  }
  pts = pts + ']';
  return '{\n'
    + ind(d + 1) + '"id": ' + str(r.id) + ',\n'
    + ind(d + 1) + '"controlPoints": ' + pts + ',\n'
    + ind(d + 1) + '"widths": ' + nums(r.widths) + ',\n'
    + ind(d + 1) + '"depth": ' + num(r.depth) + ',\n'
    + ind(d + 1) + '"flowSpeed": ' + num(r.flowSpeed) + ',\n'
    + ind(d + 1) + '"color": ' + vec4(r.color) + '\n'
    + ind(d) + '}';
}

function terrainLayer(l: TerrainLayer, d: number): string {
  return '{\n'
    + ind(d + 1) + '"id": ' + str(l.id) + ',\n'
    + ind(d + 1) + '"textureRef": ' + str(l.textureRef) + ',\n'
    + ind(d + 1) + '"weights": ' + nums(l.weights) + ',\n'
    + ind(d + 1) + '"tileScale": ' + num(l.tileScale) + '\n'
    + ind(d) + '}';
}

function terrain(t: TerrainData, d: number): string {
  let layers = '[]';
  if (t.layers.length > 0) {
    layers = '[\n';
    for (let i = 0; i < t.layers.length; i++) {
      layers = layers + ind(d + 2) + terrainLayer(t.layers[i], d + 2);
      if (i < t.layers.length - 1) layers = layers + ',';
      layers = layers + '\n';
    }
    layers = layers + ind(d + 1) + ']';
  }
  return '{\n'
    + ind(d + 1) + '"width": ' + num(t.width) + ',\n'
    + ind(d + 1) + '"depth": ' + num(t.depth) + ',\n'
    + ind(d + 1) + '"cellSize": ' + num(t.cellSize) + ',\n'
    + ind(d + 1) + '"origin": ' + vec3(t.origin) + ',\n'
    + ind(d + 1) + '"heights": ' + nums(t.heights) + ',\n'
    + ind(d + 1) + '"layers": ' + layers + '\n'
    + ind(d) + '}';
}

function environment(e: EnvironmentData, d: number): string {
  return '{\n'
    + ind(d + 1) + '"skyColor": ' + vec3(e.skyColor) + ',\n'
    + ind(d + 1) + '"ambientColor": ' + vec3(e.ambientColor) + ',\n'
    + ind(d + 1) + '"ambientIntensity": ' + num(e.ambientIntensity) + ',\n'
    + ind(d + 1) + '"sunDirection": ' + vec3(e.sunDirection) + ',\n'
    + ind(d + 1) + '"sunColor": ' + vec3(e.sunColor) + ',\n'
    + ind(d + 1) + '"sunIntensity": ' + num(e.sunIntensity) + ',\n'
    + ind(d + 1) + '"fogStart": ' + num(e.fogStart) + ',\n'
    + ind(d + 1) + '"fogEnd": ' + num(e.fogEnd) + ',\n'
    + ind(d + 1) + '"fogColor": ' + vec3(e.fogColor) + ',\n'
    + ind(d + 1) + '"shadowsEnabled": ' + bool(e.shadowsEnabled) + '\n'
    + ind(d) + '}';
}

function arr<T>(items: T[], d: number, fn: (x: T, d: number) => string): string {
  if (items.length === 0) return '[]';
  let out = '[\n';
  for (let i = 0; i < items.length; i++) {
    out = out + ind(d + 1) + fn(items[i], d + 1);
    if (i < items.length - 1) out = out + ',';
    out = out + '\n';
  }
  return out + ind(d) + ']';
}

// --- entry points ------------------------------------------------------------

export function serializeWorld(w: WorldData): string {
  let s = '{\n';
  s = s + ind(1) + '"schemaVersion": ' + num(w.schemaVersion) + ',\n';
  s = s + ind(1) + '"name": ' + str(w.name) + ',\n';
  s = s + ind(1) + '"id": ' + str(w.id) + ',\n';
  s = s + ind(1) + '"bounds": {\n'
        + ind(2) + '"min": ' + vec3(w.bounds.min) + ',\n'
        + ind(2) + '"max": ' + vec3(w.bounds.max) + '\n'
        + ind(1) + '},\n';
  s = s + ind(1) + '"environment": ' + environment(w.environment, 1) + ',\n';
  s = s + ind(1) + '"terrain": ' + (w.terrain === null ? 'null' : terrain(w.terrain, 1)) + ',\n';
  s = s + ind(1) + '"entities": ' + arr(w.entities, 1, entity) + ',\n';
  s = s + ind(1) + '"lights": ' + arr(w.lights, 1, light) + ',\n';
  s = s + ind(1) + '"water": ' + arr(w.water, 1, water) + ',\n';
  s = s + ind(1) + '"rivers": ' + arr(w.rivers, 1, river) + ',\n';
  s = s + ind(1) + '"metadata": ' + record(w.metadata, 1) + '\n';
  return s + '}\n';
}

function prefabChild(c: PrefabChild, d: number): string {
  let s = '{\n';
  s = s + ind(d + 1) + '"id": ' + str(c.id) + ',\n';
  s = s + ind(d + 1) + '"modelRef": ' + (c.modelRef === null ? 'null' : str(c.modelRef)) + ',\n';
  s = s + ind(d + 1) + '"prefabRef": ' + (c.prefabRef === null ? 'null' : str(c.prefabRef)) + ',\n';
  s = s + ind(d + 1) + '"transform": ' + transform(c.transform, d + 1) + ',\n';
  s = s + ind(d + 1) + '"tint": ' + (c.tint === null ? 'null' : vec4(c.tint)) + ',\n';
  s = s + ind(d + 1) + '"tags": ' + strs(c.tags) + '\n';
  return s + ind(d) + '}';
}

export function serializePrefab(p: PrefabData): string {
  let s = '{\n';
  s = s + ind(1) + '"id": ' + str(p.id) + ',\n';
  s = s + ind(1) + '"name": ' + str(p.name) + ',\n';
  s = s + ind(1) + '"children": ' + arr(p.children, 1, prefabChild) + '\n';
  return s + '}\n';
}
