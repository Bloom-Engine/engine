// Builds a Perry-safe TypeScript module from a directory of material JSON
// descriptors.
//
// Why: Perry's runtime `JSON.parse` produces arrays whose `.length` reads as
// the literal-init size, not the post-push count (see docs/perry-quirks.md).
// So at runtime we can't read `assets/materials/*.json` directly. This tool
// is a build step — run under `bun` — that reads the editor-authored JSON
// files and emits a sibling `.ts` module with literal-initialized
// `MaterialDesc` records that get fed to the engine's `loadMaterial` at
// startup.
//
// Mirrors the same pattern as `bloom/shooter/tools/build-world.ts`. Lives
// in the engine repo because the JSON → MaterialDesc conversion is generic
// across games — every Bloom game using JSON-on-disk material descriptors
// will want this exact pipeline.
//
// Usage:
//   bun engine/tools/build-materials-from-json.ts <input-dir> <output-ts>
//
// Example:
//   bun engine/tools/build-materials-from-json.ts \
//       assets/materials \
//       src/generated/materials.ts
//
// Each .json file in the input dir must match the MaterialDesc shape:
//   {
//     "shader": "assets/shaders/grass.wgsl",
//     "bucket": "opaque" | "cutout" | "transparent" | "refractive" | "additive",
//     "params": [1.0, 0.5, 0.2, ...]   // optional, ≤ 64 floats
//   }
//
// The generated module exports:
//   - one `MAT_<name>_DESC` constant per JSON file (literal-init MaterialDesc)
//   - one `loadMat<Name>(): number` loader per file
//   - a batched `loadAllMaterials()` that returns a typed bundle

import * as fs from 'fs';
import * as path from 'path';

type Bucket = 'opaque' | 'cutout' | 'transparent' | 'refractive' | 'additive';

interface MaterialJson {
  shader: string;
  bucket: Bucket;
  params?: number[];
}

const VALID_BUCKETS: ReadonlyArray<Bucket> = [
  'opaque',
  'cutout',
  'transparent',
  'refractive',
  'additive',
];

function pascal(s: string): string {
  return s
    .split(/[^a-zA-Z0-9]+/)
    .filter(Boolean)
    .map((p) => p[0].toUpperCase() + p.slice(1))
    .join('');
}

function snakeUpper(s: string): string {
  return s.replace(/[^a-zA-Z0-9]+/g, '_').toUpperCase();
}

function validate(name: string, desc: unknown): MaterialJson {
  if (typeof desc !== 'object' || desc === null) {
    throw new Error(`${name}: expected an object, got ${typeof desc}`);
  }
  const d = desc as Record<string, unknown>;
  if (typeof d.shader !== 'string' || d.shader.length === 0) {
    throw new Error(`${name}: missing or empty "shader" field`);
  }
  if (typeof d.bucket !== 'string' || !VALID_BUCKETS.includes(d.bucket as Bucket)) {
    throw new Error(
      `${name}: "bucket" must be one of ${VALID_BUCKETS.join(', ')} (got ${JSON.stringify(d.bucket)})`,
    );
  }
  if (d.params !== undefined) {
    if (!Array.isArray(d.params)) {
      throw new Error(`${name}: "params" must be an array of numbers if present`);
    }
    for (const v of d.params) {
      if (typeof v !== 'number' || Number.isNaN(v)) {
        throw new Error(`${name}: "params" must contain only finite numbers`);
      }
    }
    if (d.params.length > 64) {
      throw new Error(
        `${name}: "params" length ${d.params.length} exceeds the 64-float (256-byte) ABI cap`,
      );
    }
  }
  return {
    shader: d.shader,
    bucket: d.bucket as Bucket,
    params: d.params as number[] | undefined,
  };
}

function emitParams(params: number[] | undefined): string {
  if (!params || params.length === 0) return '';
  // Literal-init array — Perry-safe (`.length` reads correctly because the
  // array isn't built via .push).
  const formatted = params.map((v) => Number.isInteger(v) ? `${v}.0` : `${v}`).join(', ');
  return `,\n  params: [${formatted}]`;
}

function main() {
  const argv = process.argv.slice(2);
  if (argv.length !== 2) {
    console.error('usage: build-materials-from-json.ts <input-dir> <output-ts>');
    process.exit(2);
  }
  const inputDir = path.resolve(argv[0]);
  const outputFile = path.resolve(argv[1]);

  if (!fs.existsSync(inputDir) || !fs.statSync(inputDir).isDirectory()) {
    console.error(`input dir not found or not a directory: ${inputDir}`);
    process.exit(2);
  }

  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith('.json'))
    .sort();

  if (files.length === 0) {
    console.error(`no .json files in ${inputDir}`);
    process.exit(2);
  }

  type Entry = { name: string; pascalName: string; constName: string; desc: MaterialJson };
  const entries: Entry[] = files.map((f) => {
    const stem = path.basename(f, '.json');
    const raw = fs.readFileSync(path.join(inputDir, f), 'utf8');
    let parsed: unknown;
    try { parsed = JSON.parse(raw); }
    catch (e) { console.error(`${f}: invalid JSON: ${(e as Error).message}`); process.exit(1); }
    try {
      return {
        name: stem,
        pascalName: pascal(stem),
        constName: `MAT_${snakeUpper(stem)}_DESC`,
        desc: validate(f, parsed),
      };
    } catch (e) {
      console.error((e as Error).message);
      process.exit(1);
    }
  });

  const lines: string[] = [];
  lines.push('// AUTO-GENERATED by engine/tools/build-materials-from-json.ts.');
  lines.push('// Do not edit. Regenerate from the source JSON via the build script.');
  lines.push('//');
  lines.push(`// Source: ${path.relative(process.cwd(), inputDir)}/*.json (${entries.length} file${entries.length === 1 ? '' : 's'})`);
  lines.push('');
  lines.push("import { loadMaterial, MaterialDesc } from 'bloom/models';");
  lines.push('');

  for (const e of entries) {
    lines.push(`export const ${e.constName}: MaterialDesc = {`);
    lines.push(`  shader: ${JSON.stringify(e.desc.shader)},`);
    lines.push(`  bucket: ${JSON.stringify(e.desc.bucket)}${emitParams(e.desc.params)}`);
    lines.push('};');
    lines.push('');
    lines.push(`export function loadMat${e.pascalName}(): number {`);
    lines.push(`  return loadMaterial(${e.constName});`);
    lines.push('}');
    lines.push('');
  }

  // Batched loader — the bundle interface lets games destructure once at
  // startup. Field names match the input JSON stems verbatim.
  lines.push('export interface MaterialBundle {');
  for (const e of entries) lines.push(`  ${e.name}: number;`);
  lines.push('}');
  lines.push('');
  lines.push('export function loadAllMaterials(): MaterialBundle {');
  lines.push('  return {');
  for (const e of entries) lines.push(`    ${e.name}: loadMat${e.pascalName}(),`);
  lines.push('  };');
  lines.push('}');
  lines.push('');

  fs.mkdirSync(path.dirname(outputFile), { recursive: true });
  fs.writeFileSync(outputFile, lines.join('\n'));
  console.log(`wrote ${outputFile} (${entries.length} material${entries.length === 1 ? '' : 's'})`);
}

main();
