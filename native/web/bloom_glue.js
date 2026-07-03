/**
 * Bloom Engine — Web Bootstrap & FFI Bridge
 *
 * Loaded as a deferred ES module by both the standalone template (`index.html`,
 * engine-only / no game) and the build-assembled page that `build.sh` produces
 * by splicing this bootstrap into Perry's self-contained WASM HTML.
 *
 * Responsibilities:
 *   1. Load the Bloom rendering engine WASM (`pkg/bloom_web.js`, wasm-pack).
 *   2. Initialise JoltPhysics.js (best-effort).
 *   3. Publish `globalThis.__ffiImports` — the `bloom_*` functions the Perry
 *      game WASM imports under its `"ffi"` namespace.
 *   4. Wire DOM input, HiDPI canvas sizing, Web Audio, and the rAF game loop.
 *   5. Resolve `window.__bloomReady` so the gated `bootPerryWasm(...)` call in
 *      the page can instantiate the game WASM *after* the engine + FFI are live.
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

/**
 * Idempotent engine bootstrap. Safe to call multiple times; only the first
 * call performs work. Returns the resolved Bloom wasm-bindgen module.
 */
export async function bootBloomGame() {
  if (booted) return bloomModule;
  booted = true;

  // 1. Load Bloom engine WASM.
  await init();
  bloomModule = bloom;

  // 2. Initialise Jolt physics (WASM build of Jolt). The jolt_bridge.js module
  //    wasm-bindgen bundled into pkg/snippets/ owns all physics state; we reach
  //    it via the Rust-exported helper so bloom_physics_* talk to one instance.
  //    Override globalThis.__joltFactory to self-host instead of the CDN.
  try {
    const factory = globalThis.__joltFactory
      ?? (await import('https://cdn.jsdelivr.net/npm/jolt-physics@1.0.0/+esm')).default;
    await bloom.bloom_physics_init_jolt(factory);
    console.log('[bloom] Jolt physics ready');
  } catch (e) {
    console.warn('[bloom] Jolt init failed:', e, '- bloom_physics_* calls will be no-ops');
  }

  // 3. Publish the FFI surface for the Perry game WASM.
  const ffi = buildFfiImports();
  if (typeof globalThis.__ffiImports === 'undefined') {
    globalThis.__ffiImports = ffi;
  } else {
    Object.assign(globalThis.__ffiImports, ffi);
  }

  // 4. DOM wiring (input + HiDPI canvas sizing).
  setupDomBridge();

  // 5. Hide the loading indicator, if present.
  const loading = document.getElementById('loading');
  if (loading) loading.style.display = 'none';

  console.log('[bloom] engine + FFI ready');
  return bloomModule;
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
  // compile_material_from_file: no web filesystem — fetch the source, compile it.
  imports.bloom_compile_material_from_file = (path, _bucketKind) => {
    const src = syncFetchText(String(path));
    return src != null ? bloom.bloom_compile_material_str(src) : 0;
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
  };
  for (const [name, bytesName] of Object.entries(byteLoaders)) {
    const bytesFn = bloom[bytesName];
    if (!bytesFn) continue;
    imports[name] = (path) => {
      const data = syncFetchBytes(String(path));
      return data ? bytesFn(data) : 0;
    };
  }

  // --- Window / title ---
  imports.bloom_init_window = (w, h, title, fullscreen) => {
    document.title = String(title) || 'Bloom Engine';
    // Web export ignores the title slot (_title: f64); pass 0 for it.
    bloom.bloom_init_window(w, h, 0, fullscreen);
  };
  imports.bloom_set_window_title = (title) => { document.title = String(title); };

  // --- Web Audio ---
  let audioContext = null;
  let audioProcessor = null;
  imports.bloom_init_audio = () => {
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
  };
  imports.bloom_close_audio = () => {
    if (audioProcessor) { audioProcessor.disconnect(); audioProcessor = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
  };

  // --- Fullscreen / cursor ---
  imports.bloom_toggle_fullscreen = () => {
    const canvas = document.getElementById('bloom-canvas');
    if (!document.fullscreenElement) canvas?.requestFullscreen?.().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  };
  imports.bloom_disable_cursor = () => {
    document.getElementById('bloom-canvas')?.requestPointerLock?.();
    bloom.bloom_disable_cursor();
  };
  imports.bloom_enable_cursor = () => {
    document.exitPointerLock?.();
    bloom.bloom_enable_cursor();
  };

  // --- File I/O via localStorage (no real filesystem on web) ---
  const LS_PREFIX = 'bloom_fs:';
  imports.bloom_write_file = (path, data) => {
    try { localStorage.setItem(LS_PREFIX + String(path), String(data)); return 1; }
    catch { return 0; }
  };
  imports.bloom_file_exists = (path) =>
    localStorage.getItem(LS_PREFIX + String(path)) !== null ? 1 : 0;
  imports.bloom_read_file = (path) => {
    const v = localStorage.getItem(LS_PREFIX + String(path));
    return v === null ? '' : v; // plain string; Perry re-encodes via wrapFfiForI64
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

  return imports;
}

/**
 * Drive the captured Perry game closure once per animation frame:
 *   begin_drawing → callback(dt) → end_drawing.
 *
 * The closure is invoked through Perry's `callWasmClosure`, a global helper its
 * runtime exposes that resolves the closure's function-table index + captures
 * against the live game WASM instance. By the time the first frame runs, the
 * runtime classic <script> has executed, so the global is defined.
 */
function startRafLoop() {
  function frame() {
    if (!gameRunning) return;
    bloom.bloom_begin_drawing();
    if (gameCallback !== null && typeof globalThis.callWasmClosure === 'function') {
      const dt = bloom.bloom_get_delta_time();
      globalThis.callWasmClosure(gameCallback, dt);
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
// Perry calls load functions synchronously (like native), so we block on a sync
// XHR. Fine for startup-time asset loads served same-origin.
function syncFetchBytes(url) {
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.responseType = 'arraybuffer';
    xhr.send();
    if (xhr.status === 200 || xhr.status === 0) return new Uint8Array(xhr.response);
  } catch (e) {
    console.warn('[bloom] syncFetchBytes failed for', url, e);
  }
  return null;
}
function syncFetchText(url) {
  const bytes = syncFetchBytes(url);
  return bytes ? new TextDecoder().decode(bytes) : null;
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

  // Keyboard. (Mouse/touch injection FFI is not yet exported by the web
  // engine — see native/web/src/input_ffi.rs; keyboard only for now.)
  const keyMap = {
    'KeyA': 65, 'KeyB': 66, 'KeyC': 67, 'KeyD': 68, 'KeyE': 69, 'KeyF': 70,
    'KeyG': 71, 'KeyH': 72, 'KeyI': 73, 'KeyJ': 74, 'KeyK': 75, 'KeyL': 76,
    'KeyM': 77, 'KeyN': 78, 'KeyO': 79, 'KeyP': 80, 'KeyQ': 81, 'KeyR': 82,
    'KeyS': 83, 'KeyT': 84, 'KeyU': 85, 'KeyV': 86, 'KeyW': 87, 'KeyX': 88,
    'KeyY': 89, 'KeyZ': 90,
    'Digit0': 48, 'Digit1': 49, 'Digit2': 50, 'Digit3': 51, 'Digit4': 52,
    'Digit5': 53, 'Digit6': 54, 'Digit7': 55, 'Digit8': 56, 'Digit9': 57,
    'Space': 32, 'Enter': 257, 'Escape': 256, 'Backspace': 259, 'Tab': 258,
    'ArrowUp': 265, 'ArrowDown': 264, 'ArrowLeft': 263, 'ArrowRight': 262,
    'ShiftLeft': 340, 'ShiftRight': 344, 'ControlLeft': 341, 'ControlRight': 345,
    'AltLeft': 342, 'AltRight': 346,
    'F1': 290, 'F2': 291, 'F3': 292, 'F4': 293, 'F5': 294, 'F6': 295,
    'F7': 296, 'F8': 297, 'F9': 298, 'F10': 299, 'F11': 300, 'F12': 301,
    'Comma': 44, 'Period': 46, 'Slash': 47, 'Semicolon': 59, 'Quote': 39,
    'BracketLeft': 91, 'BracketRight': 93, 'Backslash': 92, 'Minus': 45, 'Equal': 61,
    'Backquote': 96, 'Delete': 261, 'Insert': 260, 'Home': 268, 'End': 269,
    'PageUp': 266, 'PageDown': 267,
  };
  document.addEventListener('keydown', (e) => {
    const code = keyMap[e.code];
    if (code !== undefined) { bloom.bloom_inject_key_down(code); e.preventDefault(); }
  });
  document.addEventListener('keyup', (e) => {
    const code = keyMap[e.code];
    if (code !== undefined) { bloom.bloom_inject_key_up(code); e.preventDefault(); }
  });
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
    globalThis.__bloomReadyReject?.(err);
    globalThis.__bloomReadyResolve?.(); // unblock boot; FFI proxy auto-stubs
  },
);
