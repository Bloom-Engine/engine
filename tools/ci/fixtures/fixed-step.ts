import { FixedStepClock } from '../../../src/core/fixed_step';
import { GameLifecycleDriver } from '../../../src/core/game_lifecycle';
import { isFiniteNumber } from '../../../src/core/numbers';

// Same pure TypeScript timing and lifecycle code executes in native Perry and
// generated game WASM. The host verifies observations against independent
// expected values; a compiler's zero exit does not constitute a passing test.
const uniform = new FixedStepClock(0.02, 8, 0.25);
const varied = new FixedStepClock(0.02, 8, 0.25);
for (let i = 0; i < 100; i++) uniform.advance(0.01);
for (let i = 0; i < 10; i++) {
  varied.advance(0.007);
  varied.advance(0.023);
  varied.advance(0.04);
  varied.advance(0.03);
}
const longRun = new FixedStepClock(1 / 60, 8, 0.25);
for (let i = 0; i < 6000; i++) longRun.advance(1 / 144);

const capped = new FixedStepClock(0.02, 3, 0.25);
capped.advance(0.275);
const cappedSteps = capped.steps;
const cappedAlpha = capped.alpha;
const cappedDropped = capped.droppedSeconds;
capped.advance(0.01);

const invalidConfig = new FixedStepClock(0, 8, 0.25);
const invalidCap = new FixedStepClock(0.02, 1.5, 0.25);
const invalidDelta = new FixedStepClock(0.02, 8, 0.25);
invalidDelta.advance(0.01);
const rejectedNegative = !invalidDelta.advance(-1);
const rejectedNaN = !invalidDelta.advance(0 / 0);
const rejectedInfinity = !invalidDelta.advance(1 / 0);

let events = '';
let total = 0;
let previous = 0;
let drawn = 0;
let lastTick = 0;
const driver = new GameLifecycleDriver({
  init: () => { events = events + 'I'; },
  fixedUpdate: (dt, tick) => { events = events + 'F'; previous = total; total = total + dt; lastTick = tick; },
  update: (_dt) => { events = events + 'U'; },
  draw: (alpha) => { events = events + 'D'; drawn = previous + (total - previous) * alpha; },
  cleanup: () => { events = events + 'C'; },
}, { fixedStepSeconds: 0.02, maxFixedSteps: 8, maxFrameSeconds: 0.25 });
const beforeInitRejected = !driver.frame(0.01, () => false);
driver.initialize();
const duplicateInitRejected = !driver.initialize();
driver.frame(0.01, () => false);
driver.frame(0.04, () => false);
driver.dispose();
driver.dispose();
const afterDisposeRejected = !driver.frame(0.01, () => false);

let stopped = false;
let stopEvents = '';
const stopping = new GameLifecycleDriver({
  fixedUpdate: (_dt, _tick) => { stopEvents = stopEvents + 'F'; stopped = true; },
  update: (_dt) => { stopEvents = stopEvents + 'U'; },
  draw: (_alpha) => { stopEvents = stopEvents + 'D'; },
  cleanup: () => { stopEvents = stopEvents + 'C'; },
}, { fixedStepSeconds: 0.02 });
stopping.initialize();
const stopReturned = !stopping.frame(0.08, () => stopped);
stopping.dispose();

// Reject an overflowing accumulated delta without corrupting interpolation.
const huge = new FixedStepClock(1e308, 2, 1.7e308);
huge.advance(9e307);
const overflowRejected = !huge.advance(9e307);
const hugeAlpha = huge.alpha;
const tiny = new FixedStepClock(1e-300, 4, 0.25);
tiny.advance(0.1);
const tinyBounded = tiny.steps === 4 && tiny.alpha >= 0 && tiny.alpha < 1 && isFiniteNumber(tiny.droppedSeconds);
const finiteContract = isFiniteNumber(0) && isFiniteNumber(-0) && isFiniteNumber(1.7976931348623157e308) &&
  !isFiniteNumber(0 / 0) && !isFiniteNumber(1 / 0) && !isFiniteNumber(-1 / 0) &&
  !isFiniteNumber('1' as any);

let sparseDraws = 0;
const sparse = new GameLifecycleDriver({ draw: (_alpha) => { sparseDraws = sparseDraws + 1; } });
const sparseStarted = sparse.initialize();
sparse.frame(0.01, () => false);
sparse.dispose();

let updateStopped = false;
let updateStopEvents = '';
const stopInUpdate = new GameLifecycleDriver({
  update: (_dt) => { updateStopEvents = updateStopEvents + 'U'; updateStopped = true; },
  draw: (_alpha) => { updateStopEvents = updateStopEvents + 'D'; },
  cleanup: () => { updateStopEvents = updateStopEvents + 'C'; },
});
stopInUpdate.initialize();
stopInUpdate.frame(0.01, () => updateStopped);
stopInUpdate.dispose();

let invalidCalls = 0;
const invalidDriver = new GameLifecycleDriver({
  init: () => { invalidCalls = invalidCalls + 1; },
  draw: (_alpha) => { invalidCalls = invalidCalls + 1; },
  cleanup: () => { invalidCalls = invalidCalls + 1; },
}, { maxFixedSteps: 1001 });
const invalidDriverRejected = !invalidDriver.initialize();
invalidDriver.frame(0.01, () => false);
invalidDriver.dispose();

// Read each observation directly. The original object-literal JSON.stringify
// report returned undefined in Perry WASM; preserve that separately from the
// timing/lifecycle contract. Events contain only this fixture's fixed ASCII tags.
let result = "{";
result = result + "\"uniformTicks\":" + (uniform.ticks);
result = result + ",\"variedTicks\":" + (varied.ticks);
result = result + ",\"uniformAlpha\":" + (uniform.alpha);
result = result + ",\"variedAlpha\":" + (varied.alpha);
result = result + ",\"longTicks\":" + (longRun.ticks);
result = result + ",\"longAlpha\":" + (longRun.alpha);
result = result + ",\"cappedSteps\":" + (cappedSteps);
result = result + ",\"cappedAlpha\":" + (cappedAlpha);
result = result + ",\"cappedDropped\":" + (cappedDropped);
result = result + ",\"resumedTicks\":" + (capped.ticks);
result = result + ",\"invalidConfig\":" + (!invalidConfig.valid);
result = result + ",\"invalidCap\":" + (!invalidCap.valid);
result = result + ",\"rejectedNegative\":" + (rejectedNegative);
result = result + ",\"rejectedNaN\":" + (rejectedNaN);
result = result + ",\"rejectedInfinity\":" + (rejectedInfinity);
result = result + ",\"preservedAlpha\":" + (invalidDelta.alpha);
result = result + ",\"events\":\"" + (events) + "\"";
result = result + ",\"total\":" + (total);
result = result + ",\"drawn\":" + (drawn);
result = result + ",\"lastTick\":" + (lastTick);
result = result + ",\"beforeInitRejected\":" + (beforeInitRejected);
result = result + ",\"duplicateInitRejected\":" + (duplicateInitRejected);
result = result + ",\"afterDisposeRejected\":" + (afterDisposeRejected);
result = result + ",\"stopReturned\":" + (stopReturned);
result = result + ",\"stopEvents\":\"" + (stopEvents) + "\"";
result = result + ",\"overflowRejected\":" + overflowRejected;
result = result + ",\"hugeAlpha\":" + hugeAlpha;
result = result + ",\"tinyBounded\":" + tinyBounded;
result = result + ",\"finiteContract\":" + finiteContract;
result = result + ",\"sparseStarted\":" + sparseStarted;
result = result + ",\"sparseDraws\":" + sparseDraws;
result = result + ",\"updateStopEvents\":\"" + updateStopEvents + "\"";
result = result + ",\"invalidDriverRejected\":" + invalidDriverRejected;
result = result + ",\"invalidCalls\":" + invalidCalls;
console.log("BLOOM_FIXED_STEP_RESULT:" + result + "}");
