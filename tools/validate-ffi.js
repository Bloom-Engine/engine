#!/usr/bin/env node
// validate-ffi.js — cross-checks the FFI surface three ways:
//
//   package.json (perry.nativeLibrary.functions)   ← what Perry codegen emits
//   native/shared/src/ffi_core.rs + physics_jolt.rs ← what the macros generate
//   native/<platform>/src/*.rs                      ← what each crate hand-writes
//
// Fails (exit 1) when:
//   - a manifest function is not exported by some platform
//   - an exported function's arity disagrees with the manifest
//   - a platform hand-writes a function the shared macros already generate
//     (duplicate symbol at link time)
//
// Warns when a platform exports a bloom_* function the manifest doesn't
// declare (unreachable from TypeScript) unless it is allowlisted below.
//
// Run: node tools/validate-ffi.js        (CI runs this on every push)
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const PLATFORMS = ['macos', 'linux', 'windows', 'android', 'ios', 'tvos', 'watchos'];

// Exported on purpose but not (yet) declared in the manifest — native-side
// tooling/profiling surface. Adding one of these to the manifest is a
// deliberate TS API decision, not a parity bug.
const NOT_IN_MANIFEST_ALLOWLIST = new Set([
  'bloom_compile_material_from_file',
  'bloom_set_material_params',
  'bloom_set_early_z_enabled',
  // Android host-glue entry points called from the NativeActivity shim,
  // not from Perry:
  'bloom_android_on_touch',
  'bloom_android_set_asset_path',
  'bloom_android_set_native_window',
  // watchOS Swift-host glue (called from BloomWatchApp.swift):
  'bloom_watchos_postfx_state',
  'bloom_watchos_scene_drain_destroyed',
]);

// watchOS intentionally implements a reduced surface; everything else is
// generated as no-op stubs by gen_stubs.js from this same manifest, so
// name presence is checked but a thinner hand-written set is expected.
const STUB_PLATFORMS = new Set(['watchos']);

// Renderer controls are status-bearing by contract (#138): 1.0 means the
// active backend applied the request, 0.0 means unsupported/rejected. Keeping
// the policy here prevents a future platform mirror from quietly reverting a
// setter to a void no-op.
const RENDERER_STATUS_FUNCTIONS = new Set([
  'bloom_set_env_clear_from_hdr',
  'bloom_set_material_params_scratch',
  'bloom_set_material_reflection_probe',
  'bloom_set_material_texture_array',
  'bloom_set_material_shading_model',
  'bloom_set_material_probe_visible',
  'bloom_set_material_foliage',
  'bloom_clear_post_pass',
  'bloom_clear_all_post_passes',
  'bloom_set_joint_test',
  'bloom_set_ambient_light',
  'bloom_set_directional_light',
  'bloom_set_procedural_sky',
  'bloom_set_sun_direction',
  'bloom_set_fog',
  'bloom_set_chromatic_aberration',
  'bloom_set_vignette',
  'bloom_set_film_grain',
  'bloom_set_sharpen_strength',
  'bloom_set_present_mode',
  'bloom_set_transparency_composition_mode',
  'bloom_set_sun_shafts',
  'bloom_set_auto_exposure',
  'bloom_set_taa_enabled',
  'bloom_set_occlusion_culling',
  'bloom_set_render_scale',
  'bloom_set_upscale_mode',
  'bloom_set_cas_strength',
  'bloom_set_auto_resolution',
  'bloom_set_manual_exposure',
  'bloom_set_env_intensity',
  'bloom_set_ssgi_enabled',
  'bloom_set_path_tracing',
  'bloom_reset_temporal_history',
  'bloom_set_ssgi_intensity',
  'bloom_set_ssgi_radius',
  'bloom_set_dof',
  'bloom_set_quality_preset',
  'bloom_set_shadows_enabled',
  'bloom_set_shadows_always_fresh',
  'bloom_set_bloom_enabled',
  'bloom_set_bloom_intensity',
  'bloom_set_tonemap',
  'bloom_set_auto_exposure_key',
  'bloom_set_auto_exposure_rate',
  'bloom_set_ssao_enabled',
  'bloom_set_ssao_intensity',
  'bloom_set_ssao_radius',
  'bloom_set_wind',
  'bloom_set_output_scale',
  'bloom_set_model_foliage_wind',
  'bloom_set_foliage_shadow_motion',
  'bloom_set_cloud_shadows',
  'bloom_add_shadowed_point_light',
  'bloom_set_ssr_enabled',
  'bloom_set_motion_blur_enabled',
  'bloom_set_sss_enabled',
  'bloom_scene_set_visible',
  'bloom_scene_set_cast_shadow',
  'bloom_scene_set_gi_only',
  'bloom_scene_set_receive_shadow',
  'bloom_scene_set_parent',
  'bloom_scene_set_transform',
  'bloom_scene_set_trs',
  'bloom_scene_set_transform16',
  'bloom_scene_set_lod',
  'bloom_scene_attach_model_lod',
  'bloom_scene_set_material_color',
  'bloom_scene_set_material_pbr',
  'bloom_scene_set_material_emissive',
  'bloom_scene_set_material_layered_pbr',
  'bloom_scene_set_material_texture',
  'bloom_scene_set_material_water',
  'bloom_scene_set_user_data',
  'bloom_enable_shadows',
  'bloom_disable_shadows',
  'bloom_postfx_set_selected',
  'bloom_postfx_set_hovered',
  'bloom_postfx_set_outline_color',
  'bloom_postfx_set_outline_thickness',
]);

// ---------------------------------------------------------------------------
// parsing helpers

/** Split a Rust/JSON-ish param list on top-level commas (fn-pointer params
 *  contain nested parens). */
function splitParams(s) {
  const out = [];
  let depth = 0, cur = '';
  for (const c of s) {
    if (c === '(' || c === '<' || c === '[') depth++;
    else if (c === ')' || c === '>' || c === ']') depth--;
    if (c === ',' && depth === 0) { out.push(cur.trim()); cur = ''; }
    else cur += c;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

/** Extract `pub extern "C" fn bloom_*` names + arities from Rust source. */
function extractRustFns(src) {
  const fns = new Map(); // name -> arity
  const re = /pub extern "C" fn (bloom_[a-z0-9_]+)\s*\(/g;
  let m;
  while ((m = re.exec(src)) !== null) {
    // capture to matching close paren
    let depth = 1, i = re.lastIndex, start = i;
    while (i < src.length && depth > 0) {
      if (src[i] === '(') depth++;
      else if (src[i] === ')') depth--;
      i++;
    }
    const params = splitParams(src.slice(start, i - 1)).filter(Boolean);
    // gate-paired definitions (cfg + cfg(not)) share name and arity; keep first
    if (!fns.has(m[1])) fns.set(m[1], params.length);
  }
  return fns;
}

/** Return every declared Rust return type for a named exported function. */
function extractRustReturnTypes(src, name) {
  const returns = [];
  const re = new RegExp(`pub (?:async )?(?:extern "C" )?fn ${name}\\s*\\(`, 'g');
  let match;
  while ((match = re.exec(src)) !== null) {
    let depth = 1, i = re.lastIndex;
    while (i < src.length && depth > 0) {
      if (src[i] === '(') depth++;
      else if (src[i] === ')') depth--;
      i++;
    }
    const bodyStart = src.indexOf('{', i);
    const headerTail = bodyStart < 0 ? src.slice(i) : src.slice(i, bodyStart);
    const returnMatch = headerTail.match(/->\s*([A-Za-z0-9_*:<>]+)/);
    returns.push(returnMatch ? returnMatch[1] : '()');
  }
  return returns;
}

function readDirRust(dir) {
  let all = '';
  for (const f of fs.readdirSync(dir)) {
    if (f.endsWith('.rs')) all += fs.readFileSync(path.join(dir, f), 'utf8') + '\n';
  }
  return all;
}

function readTreeTypeScript(dir) {
  let all = '';
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const item = path.join(dir, entry.name);
    if (entry.isDirectory()) all += readTreeTypeScript(item);
    else if (entry.name.endsWith('.ts')) all += fs.readFileSync(item, 'utf8') + '\n';
  }
  return all;
}

// ---------------------------------------------------------------------------
// 1. manifest
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const manifest = new Map(); // name -> param count
const manifestReturns = new Map();
for (const f of pkg.perry.nativeLibrary.functions) {
  manifest.set(f.name, f.params.length);
  manifestReturns.set(f.name, f.returns);
}

// 2. shared macros
const coreSrc = readDirRust(path.join(ROOT, 'native/shared/src/ffi_core'));
const coreFns = extractRustFns(coreSrc);
const physSrc = fs.readFileSync(path.join(ROOT, 'native/shared/src/physics_jolt.rs'), 'utf8');
const physFns = extractRustFns(physSrc);

let failures = 0, warnings = 0;
const fail = (msg) => { console.error(`FAIL  ${msg}`); failures++; };
const warn = (msg) => { console.warn(`warn  ${msg}`); warnings++; };

const watchSrc = readDirRust(path.join(ROOT, 'native/watchos/src'));
const webSrcForStatus = readDirRust(path.join(ROOT, 'native/web/src'));
const typeScriptApiSrc = readTreeTypeScript(path.join(ROOT, 'src'));
for (const name of RENDERER_STATUS_FUNCTIONS) {
  if (manifestReturns.get(name) !== 'f64') {
    fail(`${name}: renderer-control manifest return must be f64 status`);
  }
  for (const [surface, src] of [
    ['shared', coreSrc],
    ['watchos', watchSrc],
    ['web', webSrcForStatus],
  ]) {
    const returns = extractRustReturnTypes(src, name);
    if (returns.length === 0) {
      fail(`${surface}: status-bearing renderer control ${name} is not exported`);
    } else if (returns.some((value) => value !== 'f64')) {
      fail(`${surface}: ${name} must return f64 status (found ${returns.join(', ')})`);
    }
  }
  let declarationCount = 0;
  const marker = `declare function ${name}(`;
  for (let start = typeScriptApiSrc.indexOf(marker);
       start >= 0;
       start = typeScriptApiSrc.indexOf(marker, start + marker.length)) {
    declarationCount++;
    const end = typeScriptApiSrc.indexOf(';', start);
    const declaration = typeScriptApiSrc.slice(start, end + 1);
    if (!declaration.endsWith(': number;')) {
      fail(`TypeScript declaration for ${name} must expose numeric status`);
    }
  }
  if (declarationCount === 0) {
    fail(`TypeScript API does not declare status-bearing renderer control ${name}`);
  }
}

// 3. platforms
for (const platform of PLATFORMS) {
  const dir = path.join(ROOT, 'native', platform, 'src');
  if (!fs.existsSync(dir)) continue;
  const src = readDirRust(dir);
  const own = extractRustFns(src);
  const usesCore = /define_core_ffi!\s*\(\s*\)/.test(src);
  const usesPhys = /define_physics_ffi!\s*\(\s*\)/.test(src);

  // effective export set
  const effective = new Map(own);
  if (usesCore) {
    for (const [n, a] of coreFns) {
      if (own.has(n)) fail(`${platform}: ${n} hand-written but also generated by define_core_ffi! (duplicate symbol)`);
      effective.set(n, a);
    }
  }
  if (usesPhys) {
    for (const [n, a] of physFns) {
      if (own.has(n)) fail(`${platform}: ${n} hand-written but also generated by define_physics_ffi! (duplicate symbol)`);
      effective.set(n, a);
    }
  }

  // coverage + arity vs manifest
  for (const [name, arity] of manifest) {
    if (!effective.has(name)) {
      fail(`${platform}: manifest function ${name} not exported`);
      continue;
    }
    if (effective.get(name) !== arity) {
      fail(`${platform}: ${name} arity ${effective.get(name)} != manifest ${arity}`);
    }
  }

  // exports not in manifest
  for (const name of effective.keys()) {
    if (!manifest.has(name) && !NOT_IN_MANIFEST_ALLOWLIST.has(name)
        && !name.startsWith('bloom_watchos_')) { // Swift-host glue surface
      warn(`${platform}: exports ${name} which is not in the manifest (unreachable from TS)`);
    }
  }

  const note = STUB_PLATFORMS.has(platform) ? ' (stub platform)' : '';
  console.log(`ok    ${platform}: ${effective.size} exports cover ${manifest.size} manifest functions${note}`);
}

// 4. web (wasm_bindgen + jolt_bridge.js) — name coverage PLUS arity for the
// all-f64 mirror functions (EN-063). The web crate hand-mirrors the shared
// `define_core_ffi!` macro, and "name coverage only" let real drift through
// for months: bloom_get_gamepad_axis grew a leading `gamepad` param the
// manifest never declared, so the game's one argument landed in the wrong
// slot and every axis read axis 0; bloom_get_touch_x dropped its index and
// pinned every finger to slot 0. Both are exactly the argument-shift class
// ffi_core/mod.rs says the macro exists to prevent — invisible on the one
// platform with no debugger. Functions whose web ABI is deliberately
// different (String / &[u8] / &[f32] / JsValue params — the _str/_bytes/
// _floats designs) are skipped: only signatures that are pure f64 mirrors
// are compared, which is precisely where drift is a bug.
{
  const webSrc = readDirRust(path.join(ROOT, 'native/web/src'));
  const names = new Set();
  // name -> {arity, allF64} for the Rust exports (the jolt bridge's JS
  // functions are name-only; physics arity is covered by its own manifest
  // generation).
  const webSigs = new Map();
  for (const m of webSrc.matchAll(/pub (?:async )?fn (bloom_[a-z0-9_]+)\s*\(([^)]*)\)/g)) {
    const [, fname, params] = m;
    names.add(fname);
    const parts = params.split(',').map((s) => s.trim()).filter(Boolean);
    const allF64 = parts.every((p) => /:\s*f64$/.test(p));
    webSigs.set(fname, { arity: parts.length, allF64 });
  }
  for (const m of webSrc.matchAll(/pub (?:async )?fn (bloom_[a-z0-9_]+)/g)) names.add(m[1]);
  const bridge = fs.readFileSync(path.join(ROOT, 'native/web/jolt_bridge.js'), 'utf8');
  for (const m of bridge.matchAll(/(bloom_physics_[a-z0-9_]+)\s*[:(=]/g)) names.add(m[1]);
  // Functions whose web story is structurally different and tracked:
  // pointer-taking geometry (cross-module WASM memory TODO) and
  // filesystem captures (no fs on wasm; need _bytes/_str designs).
  const WEB_GAP_ALLOWLIST = new Set([
    // EN-063 — decodes image files from disk; wasm has no fs. The web route
    // is the glue: it fetches each path in the list and feeds the bytes to
    // bloom_texture_array_files_{reset,push,commit}, which decode with the
    // same codecs. Name-mapped in bloom_glue.js, so it is not a direct export.
    'bloom_create_texture_array_from_files',
    // EN-014 V3 — decodes image files from disk; wasm has no fs. The
    // byte-array path (bloom_create_texture_array_ex) is the web route.
    // EN-025 — ragdolls are built on the native Jolt Rust wrapper
    // (physics_jolt.rs). Web routes bloom_physics_* through JoltPhysics.js
    // instead, so there is no Rust-side world to create bodies in.
    'bloom_scene_set_lod',          // Perry-WASM linear-memory bridge TODO
    'bloom_dump_shadow_map',        // debug capture, no fs on wasm
    // Native host-window embed (#70) — N/A on web (no HWND; web builds its
    // surface from the canvas id). bloom_attach_native has a web no-op stub.
    'bloom_attach_hwnd',
    // Pointer-taking mesh scratch buffers (#69) — same cross-module WASM
    // linear-memory bridge TODO as bloom_scene_set_lod.
    // Profiler ABI (round-2 EN-011 / EN-020) — GPU-timestamp profiling
    // needs TIMESTAMP_QUERY, which WebGPU does not expose, so neither the
    // numeric row/history accessors nor the text overlay have a web port.
    // Present mode (round-2 #80) — the browser owns swap/vsync; the
    // Fifo/Mailbox/Immediate selector is a no-op on web.
    // Pointer-taking scratch buffers (round-2) — same cross-module WASM
    // linear-memory bridge TODO as the mesh scratch group above.
    // Water-ripple impulse (round-2 splat compute) — not yet wired on web.
    // Scratch-buffer consumers (EN-049) — same cross-module WASM linear-memory
    // bridge TODO as the mesh scratch group above. Their payload arrives via
    // bloom_mesh_scratch_*, which does not bridge between Perry's WASM module
    // and bloom's, so the callee would read an empty buffer.
    'bloom_scene_update_geometry_scratch',
    // Scene-node transform setter — same group as bloom_scene_set_trs above;
    // web's scene-node setters are only partially ported.
  ]);
  const missing = [];
  for (const name of manifest.keys()) {
    // web exposes some functions only as _str/_bytes variants
    if (!names.has(name) && !names.has(name + '_str') && !names.has(name + '_bytes')
        && !WEB_GAP_ALLOWLIST.has(name)) missing.push(name);
  }
  if (missing.length) {
    for (const name of missing) fail(`web: manifest function ${name} not exported (and not in the documented gap allowlist)`);
  } else {
    console.log(`ok    web: full coverage (${WEB_GAP_ALLOWLIST.size} documented gaps)`);
  }

  // Arity, for the pure-f64 mirrors only (see the note above).
  let checked = 0;
  for (const [name, arity] of manifest) {
    const sig = webSigs.get(name);
    if (!sig || !sig.allF64) continue;   // absent, or a deliberate _str/_bytes/slice ABI
    checked++;
    if (sig.arity !== arity) {
      fail(`web: ${name} arity ${sig.arity} != manifest ${arity} `
         + `(argument shift — the game's args land in the wrong slots)`);
    }
  }
  console.log(`ok    web: arity checked on ${checked} all-f64 mirror functions`);
}

console.log(`\n${failures} failures, ${warnings} warnings`);
process.exit(failures ? 1 : 0);
