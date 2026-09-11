# Game loop and cleanup

Call `initWindow()` and create resources before starting the game. Use
`runGame(updateAndDraw, cleanup)` for a source entry that works on native and
web. The cleanup callback is optional; existing callback-only games keep working.

The engine begins drawing, calls `updateAndDraw(dt)` and ends drawing once per
frame. Do not add another `beginDrawing()`/`endDrawing()` pair inside this
callback. Native drives a blocking platform loop. Web registers the Perry
closure and returns immediately, then drives it through animation frames.

Release owned textures, models, audio and physics objects in `cleanup`, not in
code following `runGame()`. Native calls it after the normal loop exits. Web
calls it once after a stop request and after any active frame finishes. Calling
`closeWindow()` again from cleanup is safe. A browser frame error stops further
scheduling, reports the error and attempts cleanup. A fatal native process trap
does not guarantee cleanup.

One browser game loop can be active at a time. Starting another while the first
is active reports an error. Stop and finish cleanup before starting a new loop;
cancelled callbacks from the old loop cannot run the new game's update.

## Timing and platform events

- `dt` is variable frame time in seconds. This API does not add fixed updates,
  interpolation or a catch-up policy. Keep deterministic physics steps explicit.
- Pause is game policy. Pong uses `isKeyPressed(Key.P)` so holding P toggles once
  per press; paddle movement continues to use held-key state.
- Focus loss does not automatically pause the game. Browsers can throttle
  animation frames in background tabs, so games should choose how to handle a
  large delta when returning.
- Existing native/window and browser/canvas resize handling stays in place.
- Browser frame failures stop the scheduler. Automatic device-loss recovery and
  restoration of game resources are separate requirements.
- Browser page/process termination does not promise a final JavaScript callback.

Pong now uses this shared entry and cleanup instead of a blocking source loop.
Canonical examples use the palette's public uppercase names, such as
`Colors.WHITE`; the inventory rejects undefined names such as `Colors.White`.
The installed native gate verifies cleanup runs exactly once after its physics
and capture fixture. Scheduler tests cover the asynchronous ordering, stop,
failure, re-entry and stale-callback behavior without claiming browser rendering.
The compiled-game gate at #170 passes real browser rendering and its explicit
startup-fault control. Full starter assets/text and all canonical example runtimes
remain open under #142/#74.

## Fixed game lifecycle

`runGameLifecycle(game, options?)` provides optional `init`, `fixedUpdate`,
`update` and `cleanup` hooks plus required `draw`. The starter uses this entry.
Hooks are closures or free functions without a bound `this`; keep shared state
in their lexical scope. The engine still owns begin/end drawing around each frame.

```ts
let previous = 0;
let position = 0;
runGameLifecycle({
  init: () => initWindow(800, 450, 'Fixed simulation'),
  fixedUpdate: (dt, tick) => {
    previous = position;
    position += 60 * dt;
  },
  draw: (alpha) => {
    clearBackground(Colors.BLACK);
    drawRect(previous + (position - previous) * alpha, 40, 20, 20, Colors.WHITE);
  },
  cleanup: () => closeWindow(),
});
```

Each frame runs zero or more fixed ticks, then one variable update, then one
draw. Tick numbers start at 1. Defaults are `fixedStepSeconds: 1 / 60`,
`maxFixedSteps: 8`, and `maxFrameSeconds: 0.25`. Variable update receives the
clamped frame delta. Fixed update receives the constant step; use it for physics
instead of feeding variable delta into a deterministic simulation. Draw receives
the remaining fraction of a tick, `alpha` in `[0, 1)`, for interpolation between
previous and current simulation state. Interpolation intentionally trails the
latest simulation by up to one tick.

The clock clamps long frame deltas and drops whole ticks left after the catch-up
limit, retaining only the fractional remainder. A background tab therefore cannot
trigger an unbounded catch-up loop. This policy slows simulation relative to
wall time during overload; it is not a networking or lockstep guarantee. Pause
and focus policy stay with the game. Resize behavior is unchanged, and resource
restoration after device loss is still separate.

`FixedStepClock` exposes `steps`, cumulative scheduled `ticks`, `alpha`, accepted
`deltaSeconds` and cumulative `droppedSeconds` for custom loops or diagnostics.
Its `advance(dt)` returns false for negative/nonfinite deltas or overflowing
accumulation, preserving accumulated time and interpolation. Per-frame steps
and accepted delta become zero on rejection. Settings require positive finite
times and an integer step limit from 1 through 1000. A `1e-9` relative tick
tolerance handles ordinary floating-point partition rounding; determinism here
means a fixed simulation step, not arbitrary cross-machine bitwise arithmetic.

Invalid settings make `runGameLifecycle` report an error and return false before
init. A second active lifecycle is also rejected. A valid start returns true;
native returns after shutdown, while web returns after scheduling. Calling
`closeWindow()` from init prevents frame scheduling. Calling it during fixed
update or variable update suppresses later hooks in that frame. Cleanup runs
once after normal shutdown; no update runs after disposal. As with `runGame`,
fatal process termination cannot promise cleanup. General Perry 0.5.1220 throw
propagation remains limited; the browser gate's explicit FFI error control does
not establish a general language exception guarantee.
