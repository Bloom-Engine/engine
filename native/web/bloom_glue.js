/**
 * Bloom Engine — Web Bootstrap & FFI Bridge
 *
 * Loaded as a deferred ES module by both the standalone template (`index.html`,
 * engine-only / no game) and the build-assembled page that `build.sh` produces
 * by splicing this bootstrap into Perry's self-contained WASM HTML.
 *
 * Responsibilities:
 *   1. Load the Bloom rendering engine WASM (`pkg/bloom_web.js`, wasm-pack).
 *   2. Initialise JoltPhysics.js (best-effort; local vendored copy, then CDN).
 *   3. Prefetch the game's asset tree (assets_manifest.json, if the build
 *      emitted one) with a progress readout, so the game's synchronous loads
 *      hit a warm cache instead of issuing one blocking XHR per file.
 *   4. Publish `globalThis.__ffiImports` — the `bloom_*` functions the Perry
 *      game WASM imports under its `"ffi"` namespace.
 *   5. Pre-initialise the wgpu window and wait for the async device setup,
 *      so the game's synchronous `main()` never races "Engine not
 *      initialized" (Perry main is sync; wgpu init is async).
 *   6. Wire DOM input (keyboard/mouse/touch/pointer-lock), HiDPI canvas
 *      sizing, Web Audio, and the rAF game loop.
 *   7. Resolve `window.__bloomReady` so the gated `bootPerryWasm(...)` call
 *      in the page can instantiate the game WASM *after* all of the above.
 *
 * --- The FFI value contract (important) ---
 * Perry's WASM runtime (`wasm_runtime.js`, embedded in the page) wraps the
 * entire `ffi` namespace with `wrapFfiForI64`: it DECODES each NaN-boxed i64
 * argument to a plain JS value (number / string / handle) before calling us,
 * and RE-ENCODES our plain return value. So every function below receives and
 * returns ordinary JS values — there is no manual NaN-boxing to do here. A
 * `bloom_draw_text(text, ...)` import arrives with `text` already a JS string.
 */

import init, * as bloom from './pkg/bloom_web.js';

let bloomModule = null;
let booted = false;

// --- Game loop state ---
let gameCallback = null;   // Perry closure handle captured from bloom_run_game
let gameRunning = false;
let rafId = null;

// --- Asset prefetch cache ---
// path → Uint8Array. Filled from assets_manifest.json before the game boots;
// syncFetchBytes serves from it and only falls back to a blocking XHR for
// paths the manifest didn't know about.
const assetCache = new Map();
let manifestPaths = null;   // Set<string> when a manifest was loaded

// --- Pointer lock state ---
// The game expresses intent through bloom_disable_cursor/enable_cursor; the
// browser only grants pointer lock inside a user gesture, and revokes it on
// ESC without delivering the keydown. Both mismatches are reconciled here.
let wantPointerLock = false;

/**
 * Idempotent engine bootstrap. Safe to call multiple times; only the first
 * call performs work. Returns the resolved Bloom wasm-bindgen module.
 */
export async function bootBloomGame() {
  if (booted) return bloomModule;
  booted = true;

  const loading = document.getElementById('loading');
  const setStatus = (text) => { if (loading) loading.textContent = text; };

  // 1. Load Bloom engine WASM.
  setStatus('Loading engine…');
  await init();
  bloomModule = bloom;

  // 2. Initialise Jolt physics (WASM build of Jolt). The jolt_bridge.js module
  //    wasm-bindgen bundled into pkg/snippets/ owns all physics state; we reach
  //    it via the Rust-exported helper so bloom_physics_* talk to one instance.
  //    Resolution order: explicit override, a vendored local copy (offline
  //    builds), then the CDN.
  try {
    let factory = globalThis.__joltFactory;
    if (!factory) {
      try {
        factory = (await import(new URL('./jolt-physics.mjs', import.meta.url))).default;
      } catch {
        factory = (await import('https://cdn.jsdelivr.net/npm/jolt-physics@1.0.0/+esm')).default;
      }
    }
    await bloom.bloom_physics_init_jolt(factory);
    console.log('[bloom] Jolt physics ready');
  } catch (e) {
    console.warn('[bloom] Jolt init failed:', e, '- bloom_physics_* calls will be no-ops');
  }

  // 3. Prefetch the asset tree, if the build emitted a manifest. Progress is
  //    the one thing a frozen synchronous boot cannot show, so it is shown
  //    HERE, before the game WASM gets control.
  await prefetchAssets(setStatus);

  // 4. Publish the FFI surface for the Perry game WASM.
  const ffi = buildFfiImports();
  if (typeof globalThis.__ffiImports === 'undefined') {
    globalThis.__ffiImports = ffi;
  } else {
    Object.assign(globalThis.__ffiImports, ffi);
  }

  // 5. DOM wiring (input + HiDPI canvas sizing + audio unlock).
  setupDomBridge();

  // 6. Pre-initialise the window and WAIT for the async wgpu setup. Perry's
  //    main() is synchronous and calls engine functions immediately after its
  //    own initWindow (which is an idempotent no-op once this ran) — booting
  //    the game before the device exists panics with "Engine not initialized".
  setStatus('Starting renderer…');
  // Probe for an actual WebGPU adapter first: having the API surface
  // (navigator.gpu) is not having an adapter — Chrome ships the API even
  // when acceleration is off or the GPU is blocklisted, and wgpu would then
  // panic inside the wasm. Fail here with an actionable message instead.
  if (!navigator.gpu) {
    throw new Error('This browser has no WebGPU support. Use a current '
      + 'Chrome/Edge (113+) or Firefox (141+).');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    .catch(() => null);
  if (!adapter) {
    throw new Error('No WebGPU adapter available. Check chrome://gpu — '
      + 'hardware acceleration must be enabled and the GPU not blocklisted.');
  }
  const canvas = document.getElementById('bloom-canvas');
  const w = (canvas && canvas.clientWidth) || window.innerWidth || 1280;
  const h = (canvas && canvas.clientHeight) || window.innerHeight || 720;
  bloom.bloom_init_window(w, h, 0, 0);
  const deadline = performance.now() + 15000;
  while (bloom.bloom_is_initialized() < 0.5) {
    if (performance.now() > deadline) {
      throw new Error(
        'WebGPU device setup timed out. Check chrome://gpu — a WebGPU adapter '
        + 'is required (hardware acceleration on, GPU not blocklisted).');
    }
    await new Promise((r) => setTimeout(r, 16));
  }

  // 7. Hand over to the game. Keep the loading indicator up — the game's own
  //    synchronous boot (asset decode, pool init) runs next; remove it on the
  //    first real frame instead.
  setStatus('Starting game…');
  console.log('[bloom] engine + FFI ready');
  return bloomModule;
}

/** Fetch assets_manifest.json and warm the cache, 8 files at a time. */
async function prefetchAssets(setStatus) {
  let manifest = null;
  try {
    const res = await fetch('assets_manifest.json');
    if (res.ok) manifest = await res.json();
  } catch { /* no manifest = per-file sync XHR, same as before */ }
  if (!manifest || !Array.isArray(manifest.files)) return;

  manifestPaths = new Set(manifest.files);
  const queue = manifest.files.slice();
  const total = queue.length;
  let done = 0;
  setStatus(`Loading assets… 0/${total}`);

  async function worker() {
    for (;;) {
      const path = queue.shift();
      if (path === undefined) return;
      try {
        const res = await fetch(path);
        if (res.ok) assetCache.set(path, new Uint8Array(await res.arrayBuffer()));
      } catch { /* miss -> the sync XHR fallback will report it */ }
      done++;
      if (done % 5 === 0 || done === total) setStatus(`Loading assets… ${done}/${total}`);
    }
  }
  const workers = [];
  for (let i = 0; i < 8; i++) workers.push(worker());
  await Promise.all(workers);
}

/**
 * Build the `bloom_*` FFI import object. Defaults pass straight through to the
 * matching wasm-bindgen export (correct for all numeric-in/out and string-out
 * functions); the overrides below handle string-in, file loading, the host
 * surfaces (window/title/audio/fullscreen/cursor/storage), and the game loop.
 */
function buildFfiImports() {
  const imports = {};

  // Default: every bloom_* export, passed plain values by Perry's wrapFfiForI64.
  for (const [name, fn] of Object.entries(bloom)) {
    if (typeof fn === 'function' && name.startsWith('bloom_')) {
      imports[name] = fn;
    }
  }

  // --- Text: string params route to the _str variants ---
  imports.bloom_draw_text = (text, x, y, size, r, g, b, a) =>
    bloom.bloom_draw_text_str(String(text), x, y, size, r, g, b, a);
  imports.bloom_draw_text_ex = (font, text, x, y, size, spacing, r, g, b, a) =>
    bloom.bloom_draw_text_ex_str(font, String(text), x, y, size, spacing, r, g, b, a);
  imports.bloom_measure_text = (text, size) =>
    bloom.bloom_measure_text_str(String(text), size);
  imports.bloom_measure_text_ex = (font, text, size, spacing) =>
    bloom.bloom_measure_text_ex_str(font, String(text), size, spacing);

  // --- Materials & post-FX: shader source strings route to _str variants ---
  const materialVariants = [
    'bloom_compile_material', 'bloom_compile_material_refractive',
    'bloom_compile_material_transparent', 'bloom_compile_material_additive',
    'bloom_compile_material_cutout', 'bloom_compile_material_instanced',
    'bloom_add_post_pass', 'bloom_set_post_pass',
  ];
  for (const name of materialVariants) {
    const strFn = bloom[name + '_str'];
    if (strFn) imports[name] = (source) => strFn(String(source));
  }
  imports.bloom_compile_material_instanced_bucket = (source, bucket, readsScene) =>
    bloom.bloom_compile_material_instanced_bucket(String(source), bucket, readsScene);

  // compile_material_from_file: no web filesystem — fetch the source and
  // compile it into the requested bucket (0 opaque | 1 transparent |
  // 2 refractive | 3 additive | 4 cutout — the TS wrapper's encoding).
  imports.bloom_compile_material_from_file = (path, bucketKind) => {
    const src = syncFetchText(String(path));
    if (src == null) return 0;
    switch (bucketKind | 0) {
      case 1: return bloom.bloom_compile_material_transparent_str(src);
      case 2: return bloom.bloom_compile_material_refractive_str(src);
      case 3: return bloom.bloom_compile_material_additive_str(src);
      case 4: return bloom.bloom_compile_material_cutout_str(src);
      default: return bloom.bloom_compile_material_str(src);
    }
  };

  // Material params via the legacy pointer signature: Perry's wrapFfiForI64
  // hands us the array as a plain JS array, so route it to the floats entry.
  imports.bloom_set_material_params = (handle, params, _count) => {
    if (params && typeof params.length === 'number') {
      bloom.bloom_set_material_params_floats(handle, Float32Array.from(params));
    }
  };

  // --- Asset loading: fetch the file synchronously, hand bytes to _bytes ---
  const byteLoaders = {
    bloom_load_texture: 'bloom_load_texture_bytes',
    bloom_load_font: 'bloom_load_font_bytes',
    bloom_load_sound: 'bloom_load_sound_bytes',
    bloom_load_music: 'bloom_load_music_bytes',
    bloom_load_model: 'bloom_load_model_bytes',
    bloom_load_model_animation: 'bloom_load_model_animation_bytes',
    bloom_load_image: 'bloom_load_image_bytes',
    bloom_stage_model: 'bloom_stage_model_bytes',
    bloom_set_env_clear_from_hdr: 'bloom_set_env_clear_from_hdr_bytes',
  };
  for (const [name, bytesName] of Object.entries(byteLoaders)) {
    const bytesFn = bloom[bytesName];
    if (!bytesFn) continue;
    imports[name] = (path) => {
      const data = syncFetchBytes(String(path));
      return data ? bytesFn(data) : 0;
    };
  }

  // Texture array from a comma-separated path list: fetch each file's bytes,
  // decode + assemble engine-side (same codecs as the native from-files FFI).
  imports.bloom_create_texture_array_from_files = (paths, format, mipLevels) => {
    if (!bloom.bloom_texture_array_files_push) return 0;
    bloom.bloom_texture_array_files_reset();
    for (const raw of String(paths).split(',')) {
      const p = raw.trim();
      if (!p) continue;
      const data = syncFetchBytes(p);
      if (data) bloom.bloom_texture_array_files_push(data);
      else console.warn('[texarray] missing layer file:', p);
    }
    return bloom.bloom_texture_array_files_commit(format, mipLevels);
  };

  // --- Window / title ---
  imports.bloom_init_window = (w, h, title, fullscreen) => {
    document.title = String(title) || 'Bloom Engine';
    // Web export ignores the title slot (_title: f64); pass 0 for it.
    bloom.bloom_init_window(w, h, 0, fullscreen);
  };
  imports.bloom_set_window_title = (title) => { document.title = String(title); };

  // --- Web Audio ---
  imports.bloom_init_audio = () => initAudioBridge();
  imports.bloom_close_audio = () => closeAudioBridge();

  // --- Fullscreen / cursor ---
  imports.bloom_toggle_fullscreen = () => {
    const canvas = document.getElementById('bloom-canvas');
    if (!document.fullscreenElement) canvas?.requestFullscreen?.().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  };
  imports.bloom_disable_cursor = () => {
    wantPointerLock = true;
    document.getElementById('bloom-canvas')?.requestPointerLock?.();
    bloom.bloom_disable_cursor();
  };
  imports.bloom_enable_cursor = () => {
    wantPointerLock = false;
    document.exitPointerLock?.();
    bloom.bloom_enable_cursor();
  };

  // --- File I/O: localStorage first (saves/settings), then the served
  // asset tree (read-only data files like world JSON and manifests) ---
  const LS_PREFIX = 'bloom_fs:';
  imports.bloom_write_file = (path, data) => {
    try { localStorage.setItem(LS_PREFIX + String(path), String(data)); return 1; }
    catch { return 0; }
  };
  imports.bloom_file_exists = (path) => {
    const p = String(path);
    if (localStorage.getItem(LS_PREFIX + p) !== null) return 1;
    if (manifestPaths) return manifestPaths.has(p) ? 1 : 0;
    return syncFetchBytes(p) !== null ? 1 : 0;
  };
  imports.bloom_read_file = (path) => {
    const p = String(path);
    const v = localStorage.getItem(LS_PREFIX + p);
    if (v !== null) return v;
    const text = syncFetchText(p);
    return text === null ? '' : text; // plain string; Perry re-encodes via wrapFfiForI64
  };

  // --- Game loop ---
  // runGame() on web (bloom_get_platform() === 7) just hands its update closure
  // to bloom_run_game and returns; the blocking native loop is never entered.
  // We capture the closure and drive it from requestAnimationFrame.
  imports.bloom_run_game = (callback) => {
    gameCallback = callback;
    if (!gameRunning) {
      gameRunning = true;
      startRafLoop();
    }
  };
  // Safety net for a game that still spins `while (!windowShouldClose())`:
  // report "should close" once the rAF loop owns frame pacing, so the stray
  // loop exits after one iteration instead of hanging the tab.
  imports.bloom_window_should_close = () => (gameRunning ? 1 : 0);
  imports.bloom_close_window = () => stopGame();

  // Last: wrap every entry so (a) a throw names the FFI call that produced
  // it — Perry's boot catch reports only the message, which for an ABI
  // mismatch reads like a riddle — and (b) BigInt args (manifest-declared
  // i64 params, e.g. pointer slots) are coerced to Numbers, since every
  // implementation here is a plain JS/wasm-bindgen function expecting f64.
  for (const [name, fn] of Object.entries(imports)) {
    imports[name] = (...args) => {
      for (let i = 0; i < args.length; i++) {
        if (typeof args[i] === 'bigint') args[i] = Number(args[i]);
      }
      try {
        return fn(...args);
      } catch (e) {
        console.error('[bloom ffi]', name, 'threw:', e, e?.stack ?? '');
        throw e;
      }
    };
  }

  return imports;
}

/**
 * Drive the captured Perry game closure once per animation frame:
 *   flush queued input → begin_drawing → callback(dt) → end_drawing.
 *
 * The closure is invoked through Perry's `callWasmClosure`, a global helper its
 * runtime exposes that resolves the closure's function-table index + captures
 * against the live game WASM instance. By the time the first frame runs, the
 * runtime classic <script> has executed, so the global is defined.
 */
function startRafLoop() {
  let firstFrame = true;
  function frame() {
    if (!gameRunning) return;
    if (firstFrame) {
      firstFrame = false;
      document.getElementById('loading')?.remove();
    }
    flushInput();
    bloom.bloom_begin_drawing();
    if (gameCallback !== null && typeof globalThis.callWasmClosure === 'function') {
      const dt = bloom.bloom_get_delta_time();
      try {
        globalThis.callWasmClosure(gameCallback, dt);
      } catch (e) {
        console.error('[bloom] game frame threw; stopping loop:', e);
        gameRunning = false;
      }
    }
    bloom.bloom_end_drawing();
    rafId = requestAnimationFrame(frame);
  }
  rafId = requestAnimationFrame(frame);
}

/** Stop the rAF loop. */
export function stopGame() {
  gameRunning = false;
  if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
}

// --- Synchronous asset fetching ---
// Perry calls load functions synchronously (like native). The prefetch cache
// (assets_manifest.json) makes the common case a Map lookup; anything else
// blocks on a sync XHR, which is fine for startup-time same-origin loads.
const textCache = new Map();
function syncFetchBytes(url) {
  const cached = assetCache.get(url);
  if (cached !== undefined) return cached;
  // A manifest is authoritative for the asset tree: paths it doesn't list
  // are treated as absent without a blocking round-trip (level probes and
  // optional-sound checks would otherwise stall boot by one XHR each).
  if (manifestPaths && url.indexOf('://') < 0 && !manifestPaths.has(url)) return null;
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.responseType = 'arraybuffer';
    xhr.send();
    if (xhr.status === 200 || xhr.status === 0) {
      const bytes = new Uint8Array(xhr.response);
      assetCache.set(url, bytes);
      return bytes;
    }
  } catch (e) {
    console.warn('[bloom] syncFetchBytes failed for', url, e);
  }
  return null;
}
function syncFetchText(url) {
  const hit = textCache.get(url);
  if (hit !== undefined) return hit;
  const bytes = syncFetchBytes(url);
  const text = bytes ? new TextDecoder().decode(bytes) : null;
  textCache.set(url, text);
  return text;
}

// --- Web Audio bridge ---
// The Rust mixer is pulled by a ScriptProcessorNode; creation is attempted at
// bloom_init_audio, but browsers gate (or suspend) an AudioContext until a
// user gesture, so the first pointerdown/keydown resumes it.
let audioContext = null;
let audioProcessor = null;
function initAudioBridge() {
  if (audioContext) return;
  try {
    audioContext = new AudioContext({ sampleRate: 44100 });
    const bufSize = 4096;
    audioProcessor = audioContext.createScriptProcessor(bufSize, 0, 2);
    audioProcessor.onaudioprocess = (e) => {
      const left = e.outputBuffer.getChannelData(0);
      const right = e.outputBuffer.getChannelData(1);
      const interleaved = new Float32Array(left.length * 2);
      bloom.bloom_audio_mix(interleaved);
      for (let i = 0; i < left.length; i++) {
        left[i] = interleaved[i * 2];
        right[i] = interleaved[i * 2 + 1];
      }
    };
    audioProcessor.connect(audioContext.destination);
  } catch (e) {
    console.warn('[bloom] Web Audio init failed:', e);
  }
}
function closeAudioBridge() {
  if (audioProcessor) { audioProcessor.disconnect(); audioProcessor = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
}
function resumeAudio() {
  if (audioContext && audioContext.state === 'suspended') {
    audioContext.resume().catch(() => {});
  }
}

// --- Input event queue ---
// Browser input events fire in their own tasks — always BETWEEN frames on a
// single-threaded wasm host — but a down+up pair that both land in one
// inter-frame gap would cancel out before the engine's edge detector ever
// sees the key. So events are queued and drained at a fixed point (top of
// frame), and an -up whose -down arrived in the same batch is deferred one
// frame so the tap stays visible. (Same model Bloom Jump ships with.)
const inputQueue = [];
let deferredEvents = [];
function flushInput() {
  const seenDown = new Set();
  const defer = [];
  for (const ev of deferredEvents) applyInputEvent(ev);
  deferredEvents = defer;
  for (const ev of inputQueue) {
    if (ev.up) {
      if (seenDown.has(ev.k)) { defer.push(ev); continue; }
      applyInputEvent(ev);
    } else {
      seenDown.add(ev.k);
      applyInputEvent(ev);
    }
  }
  deferredEvents = defer;
  inputQueue.length = 0;
}
function applyInputEvent(ev) {
  if (ev.kind === 'key') {
    if (ev.up) bloom.bloom_inject_key_up(ev.code);
    else bloom.bloom_inject_key_down(ev.code);
  } else if (ev.kind === 'mouse') {
    if (ev.up) bloom.bloom_inject_mouse_button_up(ev.code);
    else bloom.bloom_inject_mouse_button_down(ev.code);
  }
}

/** DOM input + HiDPI canvas sizing. */
function setupDomBridge() {
  const canvas = document.getElementById('bloom-canvas');
  if (!canvas) return;

  // HiDPI: keep the canvas backing store + renderer surface in sync with the
  // CSS box and devicePixelRatio. Clamp dpr to 3 to bound GPU cost.
  function syncCanvasSize() {
    const rect = canvas.getBoundingClientRect();
    const logW = Math.max(1, Math.round(rect.width));
    const logH = Math.max(1, Math.round(rect.height));
    const dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
    const physW = Math.round(logW * dpr);
    const physH = Math.round(logH * dpr);
    if (canvas.width !== physW) canvas.width = physW;
    if (canvas.height !== physH) canvas.height = physH;
    if (typeof bloom.bloom_resize === 'function') bloom.bloom_resize(physW, physH, logW, logH);
  }
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(syncCanvasSize).observe(canvas);
  function watchDpr() {
    const q = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    q.addEventListener('change', () => { syncCanvasSize(); watchDpr(); }, { once: true });
  }
  watchDpr();

  // Keyboard. Codes are Bloom's Key enum (engine/src/core/keys.ts), which is
  // NOT GLFW: ENTER=265, ESCAPE=27, UP=256, F1=112, LEFT_SHIFT=280. The old
  // map here used GLFW values, so Enter/Escape/arrows/modifiers/F-keys never
  // matched what games test with isKeyPressed(Key.X) — menus went dead while
  // WASD kept working (letters happen to agree). Keep in lockstep with
  // keys.ts; jump's web/bloom_ffi.js carries the same table.
  const keyMap = {
    'KeyA': 65, 'KeyB': 66, 'KeyC': 67, 'KeyD': 68, 'KeyE': 69, 'KeyF': 70,
    'KeyG': 71, 'KeyH': 72, 'KeyI': 73, 'KeyJ': 74, 'KeyK': 75, 'KeyL': 76,
    'KeyM': 77, 'KeyN': 78, 'KeyO': 79, 'KeyP': 80, 'KeyQ': 81, 'KeyR': 82,
    'KeyS': 83, 'KeyT': 84, 'KeyU': 85, 'KeyV': 86, 'KeyW': 87, 'KeyX': 88,
    'KeyY': 89, 'KeyZ': 90,
    'Digit0': 48, 'Digit1': 49, 'Digit2': 50, 'Digit3': 51, 'Digit4': 52,
    'Digit5': 53, 'Digit6': 54, 'Digit7': 55, 'Digit8': 56, 'Digit9': 57,
    'Space': 32, 'Enter': 265, 'Escape': 27, 'Backspace': 8, 'Tab': 9,
    'ArrowUp': 256, 'ArrowDown': 257, 'ArrowLeft': 258, 'ArrowRight': 259,
    'ShiftLeft': 280, 'ShiftRight': 281, 'ControlLeft': 282, 'ControlRight': 283,
    'AltLeft': 284, 'AltRight': 285,
    'F1': 112, 'F2': 113, 'F3': 114, 'F4': 115, 'F5': 116, 'F6': 117,
    'F7': 118, 'F8': 119, 'F9': 120, 'F10': 121, 'F11': 122, 'F12': 123,
    'Comma': 44, 'Period': 46, 'Slash': 47, 'Semicolon': 59, 'Quote': 39,
    'BracketLeft': 91, 'BracketRight': 93, 'Backslash': 92, 'Minus': 45, 'Equal': 61,
    'Backquote': 96, 'Delete': 127, 'Insert': 260, 'Home': 261, 'End': 262,
    'PageUp': 263, 'PageDown': 264,
  };
  document.addEventListener('keydown', (e) => {
    resumeAudio();
    const code = keyMap[e.code];
    if (code !== undefined) {
      inputQueue.push({ kind: 'key', k: 'k' + code, code, up: false });
      if (e.code !== 'F12' && e.code !== 'F5') e.preventDefault();
    }
    if (typeof bloom.bloom_inject_char === 'function'
        && e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      bloom.bloom_inject_char(e.key.codePointAt(0));
    }
  });
  document.addEventListener('keyup', (e) => {
    const code = keyMap[e.code];
    if (code !== undefined) {
      inputQueue.push({ kind: 'key', k: 'k' + code, code, up: true });
      e.preventDefault();
    }
  });

  // Mouse. DOM buttons are 0=left / 1=middle / 2=right; Bloom's MouseButton
  // is LEFT=0 / RIGHT=1 / MIDDLE=2.
  const domButtonToBloom = [0, 2, 1, 3, 4];
  canvas.addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas) {
      bloom.bloom_inject_mouse_delta(e.movementX, e.movementY);
    } else {
      const rect = canvas.getBoundingClientRect();
      bloom.bloom_inject_mouse_move(e.clientX - rect.left, e.clientY - rect.top);
    }
  });
  canvas.addEventListener('mousedown', (e) => {
    resumeAudio();
    // Re-acquire pointer lock on click when the game wants it (the browser
    // released it on ESC, or never granted it because the request happened
    // outside a user gesture).
    if (wantPointerLock && document.pointerLockElement !== canvas) {
      canvas.requestPointerLock?.();
    }
    const b = domButtonToBloom[e.button];
    if (b !== undefined) inputQueue.push({ kind: 'mouse', k: 'm' + b, code: b, up: false });
    e.preventDefault();
  });
  document.addEventListener('mouseup', (e) => {
    const b = domButtonToBloom[e.button];
    if (b !== undefined) inputQueue.push({ kind: 'mouse', k: 'm' + b, code: b, up: true });
  });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('wheel', (e) => {
    // deltaMode 0 = pixels (~100/notch); 1 = lines (~3/notch). Normalise to
    // roughly ±1 per notch, positive = wheel up (native convention).
    const notches = e.deltaMode === 1 ? -e.deltaY / 3 : -e.deltaY / 100;
    bloom.bloom_inject_mouse_wheel(notches);
    e.preventDefault();
  }, { passive: false });

  // Pointer lock: the browser exits on ESC without delivering the keydown.
  // If the game still wanted the lock, surface that exit as an ESC press so
  // its pause-menu binding fires.
  document.addEventListener('pointerlockchange', () => {
    if (document.pointerLockElement !== canvas && wantPointerLock) {
      inputQueue.push({ kind: 'key', k: 'k27', code: 27, up: false }); // Key.ESCAPE
      inputQueue.push({ kind: 'key', k: 'k27', code: 27, up: true });
    }
  });

  // Touch. Browser touch identifiers are arbitrary; map them onto stable
  // slots 0..N-1 the way the engine's sparse touch API expects. Release goes
  // through the deferred path so a same-frame tap stays visible.
  const touchSlots = new Map(); // identifier → slot
  function slotFor(id) {
    let s = touchSlots.get(id);
    if (s !== undefined) return s;
    const used = new Set(touchSlots.values());
    for (s = 0; used.has(s); s++);
    touchSlots.set(id, s);
    return s;
  }
  function touchXY(t) {
    const rect = canvas.getBoundingClientRect();
    return [t.clientX - rect.left, t.clientY - rect.top];
  }
  canvas.addEventListener('touchstart', (e) => {
    resumeAudio();
    for (const t of e.changedTouches) {
      const [x, y] = touchXY(t);
      bloom.bloom_inject_touch(slotFor(t.identifier), x, y, 1);
    }
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchmove', (e) => {
    for (const t of e.changedTouches) {
      const [x, y] = touchXY(t);
      bloom.bloom_inject_touch(slotFor(t.identifier), x, y, 1);
    }
    e.preventDefault();
  }, { passive: false });
  const touchEnd = (e) => {
    for (const t of e.changedTouches) {
      const s = touchSlots.get(t.identifier);
      if (s === undefined) continue;
      const [x, y] = touchXY(t);
      bloom.bloom_inject_touch_release(s, x, y);
      touchSlots.delete(t.identifier);
    }
    e.preventDefault();
  };
  canvas.addEventListener('touchend', touchEnd, { passive: false });
  canvas.addEventListener('touchcancel', touchEnd, { passive: false });

  document.addEventListener('pointerdown', resumeAudio);
}

// --- Auto-boot on import ---
// The page creates `window.__bloomReady` (a pending promise) in a classic
// inline <script> during parse; the gated `bootPerryWasm(...)` call awaits it.
// We resolve it once the engine + FFI are live so the game WASM instantiates
// against a ready host. If no resolver was installed (engine-only standalone
// use), this is harmless.
bootBloomGame().then(
  () => { globalThis.__bloomReadyResolve?.(); },
  (err) => {
    console.error('[bloom] bootstrap failed:', err);
    const loading = document.getElementById('loading');
    if (loading) loading.textContent = 'Failed to start: ' + (err?.message ?? err);
    globalThis.__bloomReadyReject?.(err);
  },
);
