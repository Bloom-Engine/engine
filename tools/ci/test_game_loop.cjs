const assert = require('node:assert/strict');
const { test } = require('node:test');

async function harness() {
  const { createGameLoop } = await import('../../native/web/game_loop.mjs');
  const requests = new Map();
  const events = [];
  const errors = [];
  let nextId = 0;
  const hooks = {
    beginFrame: () => events.push('begin'),
    endFrame: () => events.push('end'),
    update: (callback) => callback(),
    cleanup: (callback) => callback(),
  };
  const loop = createGameLoop({
    requestFrame(callback) { requests.set(++nextId, callback); return nextId; },
    cancelFrame(id) { requests.delete(id); },
    beginFrame: () => hooks.beginFrame(),
    endFrame: () => hooks.endFrame(),
    update: (callback) => hooks.update(callback),
    cleanup: (callback) => hooks.cleanup(callback),
    onError: (error) => errors.push(error.message),
  });
  function tick() {
    assert.equal(requests.size, 1);
    const [id, callback] = requests.entries().next().value;
    requests.delete(id);
    callback();
  }
  return { loop, requests, events, errors, hooks, tick };
}

test('start returns before a frame; stop cancels and cleans up once', async () => {
  const h = await harness();
  h.loop.start(() => h.events.push('update'), () => h.events.push('cleanup'));
  assert.deepEqual(h.events, []);
  assert.equal(h.loop.running, true);
  h.tick();
  assert.deepEqual(h.events, ['begin', 'update', 'end']);
  h.loop.stop();
  h.loop.stop();
  assert.deepEqual(h.events, ['begin', 'update', 'end', 'cleanup']);
  assert.equal(h.loop.running, false);
  assert.equal(h.requests.size, 0);
});

test('stop inside a frame defers disposal until drawing finishes', async () => {
  const h = await harness();
  h.loop.start(() => { h.events.push('update'); h.loop.stop(); }, () => {
    h.events.push('cleanup');
    h.loop.stop(); // Cleanup may itself call closeWindow.
  });
  h.tick();
  assert.deepEqual(h.events, ['begin', 'update', 'end', 'cleanup']);
  assert.equal(h.requests.size, 0);
});

test('legacy callback-only loop still runs and can stop', async () => {
  const h = await harness();
  h.loop.start(() => h.events.push('update'));
  h.tick(); h.tick(); h.loop.stop();
  assert.deepEqual(h.events, ['begin', 'update', 'end', 'begin', 'update', 'end']);
  assert.deepEqual(h.errors, []);
});

for (const stage of ['begin', 'update', 'end']) {
  test(`${stage} failure stops scheduling and performs cleanup`, async () => {
    const h = await harness();
    const fail = () => { h.events.push(stage); throw new Error(stage + ' failed'); };
    if (stage === 'begin') h.hooks.beginFrame = fail;
    if (stage === 'end') h.hooks.endFrame = fail;
    h.loop.start(stage === 'update' ? fail : () => h.events.push('update'),
      () => h.events.push('cleanup'));
    h.tick();
    assert.deepEqual(h.events, stage === 'begin' ? ['begin', 'cleanup'] : ['begin', 'update', 'end', 'cleanup']);
    assert.deepEqual(h.errors, [stage + ' failed']);
    assert.equal(h.loop.running, false);
    assert.equal(h.requests.size, 0);
  });
}

test('cleanup failure is reported once and the next game can start', async () => {
  const h = await harness();
  h.loop.start(() => {}, () => { throw new Error('cleanup failed'); });
  h.loop.stop(); h.loop.stop();
  assert.deepEqual(h.errors, ['cleanup failed']);
  h.loop.start(() => h.loop.stop());
  h.tick();
  assert.equal(h.requests.size, 0);
});

test('a cancelled old callback cannot update a newly started game', async () => {
  const h = await harness();
  h.loop.start(() => h.events.push('old'));
  const stale = h.requests.values().next().value;
  h.loop.stop();
  h.loop.start(() => { h.events.push('new'); h.loop.stop(); });
  stale();
  assert.deepEqual(h.events, []);
  h.tick();
  assert.deepEqual(h.events, ['begin', 'new', 'end']);
});

test('duplicate starts fail without losing the existing cleanup', async () => {
  const h = await harness();
  h.loop.start(() => {}, () => h.events.push('cleanup'));
  assert.throws(() => h.loop.start(() => {}), /already active/);
  h.loop.stop();
  assert.deepEqual(h.events, ['cleanup']);
});
