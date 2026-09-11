"use strict";
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const script = fs.readFileSync(path.join(__dirname, 'example_monitor.js'), 'utf8');

function context() {
  const listeners = {};
  const runtime = vm.createContext({
    console: { log() {}, warn() {}, error() {} },
    addEventListener: (name, callback) => { listeners[name] = callback; },
    callWasmClosure: (callback, ...args) => callback(...args),
  });
  vm.runInContext(script, runtime);
  return { runtime, listeners };
}

test('monitor forwards real boundaries and requests one stop after eight production-loop frames', async () => {
  const { createGameLoop } = await import('../../native/web/game_loop.mjs');
  const { runtime } = context();
  const queue = [];
  const seen = { begin: 0, end: 0, draw: 0, cleanup: 0, close: 0 };
  const loop = createGameLoop({
    requestFrame: callback => { queue.push(callback); return queue.length; }, cancelFrame() {},
    beginFrame: () => seen.begin++, endFrame: () => seen.end++,
    update: callback => runtime.callWasmClosure(callback, 1/60),
    cleanup: callback => runtime.callWasmClosure(callback), onError: error => { throw error; },
  });
  const ffi = {
    bloom_init_window: (...args) => args.join(','),
    bloom_draw_rect: (...args) => { seen.draw++; return args.length; },
    bloom_close_window: () => { seen.close++; loop.stop(); },
    bloom_run_game_with_cleanup: (...args) => loop.start(...args),
  };
  runtime.__ffiImports = ffi;
  assert.equal(ffi.bloom_init_window(800, 600, 'game'), '800,600,game');
  ffi.bloom_run_game_with_cleanup(dt => {
    assert.equal(dt, 1/60);
    assert.equal(ffi.bloom_draw_rect(10, 20, 30, 40, 255, 255, 255, 255), 8);
  }, () => { seen.cleanup++; ffi.bloom_close_window(); });
  while (queue.length) queue.shift()();
  assert.deepEqual(seen, { begin: 8, end: 8, draw: 8, cleanup: 1, close: 2 });
  const state = JSON.parse(JSON.stringify(runtime.__exampleProbe));
  assert.equal(state.frames, 8);
  assert.equal(state.cleanups, 1);
  assert.equal(state.registrations, 1);
  assert.equal(state.stopped, true);
  assert.deepEqual(state.badArguments, []);
  assert.deepEqual(state.frameDraws, Array(8).fill(1));
  assert.deepEqual(state.windows, [[800, 600]]);
});

test('monitor retains invalid draw values and preserves underlying errors', () => {
  const { runtime, listeners } = context();
  const failure = new Error('actual renderer failure');
  const ffi = { bloom_draw_rect: () => { throw failure; }, bloom_run_game_with_cleanup() {} };
  runtime.__ffiImports = ffi;
  assert.throws(() => ffi.bloom_draw_rect(NaN, undefined, null), error => error === failure);
  assert.equal(runtime.__exampleProbe.badArguments.length, 1);
  assert.deepEqual(Array.from(runtime.__exampleProbe.badArguments[0].args), ['NaN', 'undefined', 'null']);
  listeners.unhandledrejection({ reason: failure });
  assert.match(runtime.__exampleProbe.errors[0], /actual renderer failure/);
});

test('cleanup counts only after the original callback completes', () => {
  const { runtime } = context();
  const ffi = { bloom_run_game_with_cleanup() {} };
  runtime.__ffiImports = ffi;
  const cleanup = () => { throw new Error('cleanup failed'); };
  ffi.bloom_run_game_with_cleanup(() => {}, cleanup);
  assert.throws(() => runtime.callWasmClosure(cleanup), /cleanup failed/);
  assert.equal(runtime.__exampleProbe.cleanups, 0);
});

test('monitor records actual audio boundaries and rejects missing input codes', () => {
  const { runtime } = context();
  let audio = 0;
  const ffi = { bloom_run_game_with_cleanup() {}, bloom_init_audio: () => ++audio,
    bloom_close_audio: () => --audio, bloom_is_key_down: () => false };
  runtime.__ffiImports = ffi;
  assert.equal(ffi.bloom_init_audio(), 1);
  assert.equal(ffi.bloom_close_audio(), 0);
  assert.equal(ffi.bloom_is_key_down(undefined), false);
  assert.equal(runtime.__exampleProbe.calls.bloom_init_audio, 1);
  assert.equal(runtime.__exampleProbe.calls.bloom_close_audio, 1);
  assert.equal(runtime.__exampleProbe.badArguments[0].name, 'bloom_is_key_down');
});
