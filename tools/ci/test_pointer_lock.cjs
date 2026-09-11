const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function harness(request) {
  const source = fs.readFileSync(path.resolve(__dirname, '../../native/web/bloom_glue.js'), 'utf8');
  assert.ok(source.includes('// --- Auto-boot on import ---'));
  // Evaluate the production bridge without importing/booting its GPU module.
  // Only the DOM pointer-lock surface is exercised by these unit controls.
  const body = source.split('// --- Auto-boot on import ---')[0]
    .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '')
    .replaceAll('import.meta.url', '"file:///pointer-lock-unit-test/bloom_glue.js"');
  const canvas = { requestPointerLock: request };
  let exits = 0;
  const document = {
    pointerLockElement: null,
    getElementById: () => canvas,
    exitPointerLock() { exits++; this.pointerLockElement = null; },
  };
  const context = vm.createContext({ document, Promise, console, createGameLoop: () => ({}) });
  vm.runInContext(body + '\nglobalThis.control = { request: requestDesiredPointerLock, want(value) { wantPointerLock = value; } };', context);
  context.control.want(true);
  return { control: context.control, document, canvas, exits: () => exits };
}

const settle = () => new Promise(setImmediate);

test('rejected requests remain retryable and never reject the game loop', async () => {
  let attempts = 0;
  const h = harness(() => { attempts++; return Promise.reject(new Error('WrongDocumentError')); });
  assert.doesNotThrow(() => h.control.request());
  await settle();
  h.control.request();
  await settle();
  assert.equal(attempts, 2);
  assert.equal(h.document.pointerLockElement, null);
  h.control.want(false);
  h.control.request();
  assert.equal(attempts, 2);
});

test('legacy synchronous denial or void return does not throw', async () => {
  let attempts = 0;
  const h = harness(() => { if (++attempts === 1) throw new Error('NotAllowedError'); });
  assert.doesNotThrow(() => h.control.request());
  assert.doesNotThrow(() => h.control.request());
  await settle();
  assert.equal(attempts, 2);
});

test('a grant after cleanup releases the lock; an existing lock is not requested again', async () => {
  let grant;
  let attempts = 0;
  const h = harness(() => { attempts++; return new Promise(resolve => { grant = resolve; }); });
  h.control.request();
  h.control.want(false);
  h.document.pointerLockElement = h.canvas;
  grant();
  await settle();
  assert.equal(h.exits(), 1);
  assert.equal(h.document.pointerLockElement, null);
  h.control.want(true);
  h.document.pointerLockElement = h.canvas;
  h.control.request();
  assert.equal(attempts, 1);
});
