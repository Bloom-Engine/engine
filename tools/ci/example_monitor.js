// Observe the original game's real FFI/closure boundaries. Stop normally after
// eight callbacks; production rendering, input, audio and physics stay active.
(() => {
  const state = globalThis.__exampleProbe = {
    frames: 0, cleanups: 0, registrations: 0, stopped: false,
    errors: [], logs: [], badArguments: [], drawCalls: 0, frameDraws: [], calls: {}, windows: [],
  };
  const describe = value => String(value?.stack || value).slice(0, 16384);
  const record = value => { if (state.errors.length < 32) state.errors.push(describe(value)); };
  for (const level of ['error', 'warn', 'log']) {
    const original = console[level];
    console[level] = (...args) => {
      if (level === 'error') record(args.map(describe).join(' '));
      else if (state.logs.length < 100) state.logs.push(level + ': ' + args.map(describe).join(' '));
      original.apply(console, args);
    };
  }
  addEventListener('error', event => record(event.error || event.message));
  addEventListener('unhandledrejection', event => record(event.reason));
  if (globalThis.GPU) {
    const request = GPU.prototype.requestAdapter;
    GPU.prototype.requestAdapter = async function(...args) {
      const adapter = await request.apply(this, args);
      globalThis.__exampleAdapter = adapter;
      return adapter;
    };
  }
  let ffi;
  Object.defineProperty(globalThis, '__ffiImports', {
    configurable: true,
    get: () => ffi,
    set: value => {
      for (const name of Object.keys(value)) {
        const draw = /^bloom_(draw_|clear_background|begin_mode_[23]d)/.test(name);
        const input = /^bloom_is_(key|mouse_button)_/.test(name);
        const resource = ['bloom_init_window', 'bloom_close_window', 'bloom_init_audio',
          'bloom_close_audio', 'bloom_disable_cursor', 'bloom_enable_cursor'].includes(name);
        if (!draw && !resource && !input) continue;
        const original = value[name];
        value[name] = (...args) => {
          state.calls[name] = (state.calls[name] || 0) + 1;
          if (name === 'bloom_init_window') state.windows.push(args.slice(0, 2));
          if (draw || input) {
            if (name.startsWith('bloom_draw_')) state.drawCalls++;
            if (args.some(v => v === undefined || v === null || (typeof v === 'number' && !Number.isFinite(v)))) {
              if (state.badArguments.length < 8) state.badArguments.push({ name, args: args.map(describe) });
            }
          }
          return original(...args);
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
      if (typeof call !== 'function') throw new Error('Example monitor requires the real Perry dispatcher');
      globalThis.callWasmClosure = (callback, ...args) => {
        const before = state.drawCalls;
        const result = call(callback, ...args);
        if (callback === cleanupHandle) state.cleanups++;
        if (callback === updateHandle) {
          state.frameDraws.push(state.drawCalls - before);
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
