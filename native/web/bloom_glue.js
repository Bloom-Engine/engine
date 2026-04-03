/**
 * Bloom Engine — Web Glue Layer
 *
 * Orchestrates loading of both WASM modules:
 * 1. bloom_web.wasm (Bloom rendering engine, compiled from Rust via wasm-pack)
 * 2. game.wasm (Perry-compiled game logic)
 *
 * Bridges Perry's FFI calls to Bloom's wasm-bindgen exports, handles the
 * requestAnimationFrame loop, and manages cross-module callback invocation.
 *
 * Usage:
 *   <script type="module">
 *     import { bootBloomGame } from './bloom_glue.js';
 *     await bootBloomGame('./pkg/bloom_web.js', './game.wasm');
 *   </script>
 *
 * Or for Perry's self-contained HTML output, set window.__bloomWasmUrl
 * before bootPerryWasm is called.
 */

// State
let bloomModule = null;   // wasm-bindgen exports from bloom_web
let perryInstance = null;  // Perry WASM instance
let perryMemory = null;    // Perry WASM memory
let rafId = null;          // requestAnimationFrame handle
let gameRunning = false;

// Game loop state
let gameCallbackHandle = null; // Perry closure handle for runGame callback
let perryClosureCall1 = null;  // Reference to Perry's closure_call_1 function

/**
 * Boot a Bloom game from two WASM modules.
 *
 * @param {string} bloomPkgUrl - URL to bloom_web.js (wasm-pack output)
 * @param {string} gameWasmUrl - URL to the Perry-compiled game.wasm
 */
export async function bootBloomGame(bloomPkgUrl, gameWasmUrl) {
  // 1. Load Bloom engine WASM
  bloomModule = await import(bloomPkgUrl);
  await bloomModule.default(); // Initialize wasm-bindgen

  // 2. Build FFI import object mapping bloom_* names to wasm-bindgen exports.
  //    Perry WASM imports these under the "ffi" namespace.
  const ffiImports = buildFfiImports();

  // 3. Fetch and instantiate Perry game WASM.
  //    Perry's runtime JS (wasm_runtime.js) is embedded in the HTML.
  //    We provide our FFI imports to it.
  if (typeof globalThis.__ffiImports === 'undefined') {
    globalThis.__ffiImports = ffiImports;
  } else {
    Object.assign(globalThis.__ffiImports, ffiImports);
  }

  // If Perry's bootPerryWasm is available (self-contained HTML mode),
  // it will use __ffiImports. Otherwise, load game WASM directly.
  if (gameWasmUrl && typeof globalThis.bootPerryWasm === 'undefined') {
    console.warn('bloom_glue: Perry runtime not found. Load game via Perry HTML output.');
  }
}

/**
 * Build the FFI imports object from Bloom wasm-bindgen exports.
 * All bloom_* functions are mapped directly, plus bloom_run_game
 * is intercepted to set up the rAF loop.
 */
function buildFfiImports() {
  const imports = {};

  // Map all bloom_* exports
  for (const [name, fn] of Object.entries(bloomModule)) {
    if (typeof fn === 'function' && name.startsWith('bloom_')) {
      imports[name] = fn;
    }
  }

  // Intercept bloom_run_game to set up the rAF loop
  imports['bloom_run_game'] = (callbackHandle) => {
    gameCallbackHandle = callbackHandle;
    gameRunning = true;
    startRafLoop();
  };

  // Intercept bloom_window_should_close to return 1.0 on web
  // so that while(!windowShouldClose()) loops exit immediately
  // when runGame is being used (the rAF loop takes over)
  const origShouldClose = imports['bloom_window_should_close'];
  imports['bloom_window_should_close'] = () => {
    // Once runGame is called, signal the while loop to exit
    if (gameRunning) return f64ToI64(1.0);
    return origShouldClose ? origShouldClose() : f64ToI64(0.0);
  };

  return imports;
}

/**
 * Start the requestAnimationFrame game loop.
 * Each frame: begin_drawing → invoke game callback → end_drawing.
 */
function startRafLoop() {
  function frame() {
    if (!gameRunning) return;

    bloomModule.bloom_begin_drawing();

    // Invoke the game's callback with delta time
    if (gameCallbackHandle !== null) {
      const dt = bloomModule.bloom_get_delta_time();
      invokePerryCallback(gameCallbackHandle, dt);
    }

    bloomModule.bloom_end_drawing();

    rafId = requestAnimationFrame(frame);
  }

  rafId = requestAnimationFrame(frame);
}

/**
 * Invoke a Perry closure/function handle.
 *
 * Perry closures are stored in the JS handle store and invoked via
 * the indirect function table. We use the mem_dispatch closure_call_1
 * function from Perry's runtime.
 */
function invokePerryCallback(handle, arg) {
  // Perry's runtime exposes closure_call_1 via the __memDispatch table
  // which is available in Perry's wasm_runtime.js scope.
  // If we have a direct reference, use it.
  if (perryClosureCall1) {
    perryClosureCall1(handle, arg);
    return;
  }

  // Fallback: try the globally available __memDispatch table
  if (typeof globalThis.__memDispatch !== 'undefined' &&
      globalThis.__memDispatch.closure_call_1) {
    perryClosureCall1 = globalThis.__memDispatch.closure_call_1;
    perryClosureCall1(handle, arg);
    return;
  }

  // Fallback: try to call through Perry's WASM indirect function table directly
  if (perryInstance) {
    const table = perryInstance.exports.__indirect_function_table;
    if (table) {
      // For simple function references (not closures), handle IS the table index
      try {
        const fn = table.get(handle | 0);
        if (fn) {
          fn(f64ToI64(arg));
          return;
        }
      } catch (e) {
        console.warn('bloom_glue: failed to invoke callback via table:', e);
      }
    }
  }

  console.warn('bloom_glue: could not invoke game callback');
}

/**
 * Stop the game loop.
 */
export function stopGame() {
  gameRunning = false;
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

/**
 * Set the Perry WASM instance (called by bloom_glue or Perry's boot code).
 * Needed for cross-module callback invocation.
 */
export function setPerryInstance(instance) {
  perryInstance = instance;
  perryMemory = instance?.exports?.memory;
}

// --- NaN-boxing helpers ---

function f64ToI64(v) {
  const buf = new ArrayBuffer(8);
  new Float64Array(buf)[0] = v;
  return new BigUint64Array(buf)[0];
}

function i64ToF64(b) {
  const buf = new ArrayBuffer(8);
  new BigUint64Array(buf)[0] = BigInt.asUintN(64, b);
  return new Float64Array(buf)[0];
}

// --- Auto-setup for Perry self-contained HTML mode ---

// When Perry generates a self-contained HTML, it calls bootPerryWasm(base64).
// We hook into this by providing __ffiImports before boot.
// The bloom engine must be loaded first (via a <script> that imports bloom_web.js).

if (typeof globalThis.__bloomAutoInit !== 'undefined') {
  // Auto-init mode: bloom_web was already loaded
  (async () => {
    try {
      bloomModule = globalThis.__bloomModule;
      const ffi = buildFfiImports();
      globalThis.__ffiImports = ffi;
      console.log('bloom_glue: FFI imports ready for Perry WASM');
    } catch (e) {
      console.error('bloom_glue: auto-init failed:', e);
    }
  })();
}
