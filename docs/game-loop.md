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
Actual compiled-game browser startup, a complete init/update/fixed-update/draw
lifecycle and the one-command starter remain open under #142/#74.
