// Observe the unmodified starter through its real FFI and callback boundaries.
// The sole control action is a normal engine stop after eight successful frames.
(() => {
  const state = globalThis.__starterProbe = {
    frames: 0, cleanups: 0, registrations: 0, reads: [], texts: [], rects: [], clears: [],
    errors: [], logs: [], fetches: [], stopped: false,
  };
  const describe = value => String(value?.stack || value).slice(0, 16384);
  const error = value => { if (state.errors.length < 32) state.errors.push(describe(value)); };
  for (const level of ['error', 'warn', 'log']) {
    const original = console[level];
    console[level] = (...args) => {
      if (level === 'error') error(args.map(describe).join(' '));
      else if (state.logs.length < 100) state.logs.push(level + ': ' + args.map(describe).join(' '));
      original.apply(console, args);
    };
  }
  addEventListener('error', event => error(event.error || event.message));
  addEventListener('unhandledrejection', event => error(event.reason));
  const fetchOriginal = globalThis.fetch;
  globalThis.fetch = async (...args) => {
    const response = await fetchOriginal(...args);
    const url = new URL(args[0]?.url || String(args[0]), location.href);
    if (url.origin === location.origin && ['assets_manifest.json', 'assets/welcome.txt'].some(p => url.pathname.endsWith('/' + p))) {
      state.fetches.push({ path: url.pathname, status: response.status });
    }
    return response;
  };
  if (globalThis.GPU) {
    const request = GPU.prototype.requestAdapter;
    GPU.prototype.requestAdapter = async function(...args) {
      const adapter = await request.apply(this, args);
      globalThis.__starterAdapter = adapter;
      return adapter;
    };
  }
  let ffi;
  Object.defineProperty(globalThis, '__ffiImports', {
    configurable: true,
    get: () => ffi,
    set: value => {
      const originalRead = value.bloom_read_file;
      value.bloom_read_file = (...args) => {
        const result = originalRead(...args);
        if (state.reads.length < 32) state.reads.push({ path: String(args[0]), value: String(result).slice(0, 4096) });
        return result;
      };
      for (const [name, destination] of [['bloom_draw_text', 'texts'], ['bloom_draw_rect', 'rects'], ['bloom_clear_background', 'clears']]) {
        const original = value[name];
        value[name] = (...args) => {
          const result = original(...args);
          if (state[destination].length < 120) state[destination].push(args.map(v => typeof v === 'bigint' ? Number(v) : v));
          return result;
        };
      }
      let updateHandle, cleanupHandle;
      const register = value.bloom_run_game_with_cleanup;
      value.bloom_run_game_with_cleanup = (update, cleanup) => {
        state.registrations++;
        updateHandle = update;
        cleanupHandle = cleanup;
        return register(update, cleanup);
      };
      const call = globalThis.callWasmClosure;
      if (typeof call !== 'function') throw new Error('Starter monitor requires the real Perry closure dispatcher');
      globalThis.callWasmClosure = (callback, ...args) => {
        if (callback === cleanupHandle) state.cleanups++;
        const result = call(callback, ...args);
        if (callback === updateHandle) {
          state.frames++;
          if (state.frames === 8) {
            state.stopped = true;
            value.bloom_close_window();
          }
        }
        return result;
      };
      ffi = value;
    },
  });
})();
