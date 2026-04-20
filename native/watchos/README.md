# bloom-watchos (planned)

Placeholder for the watchOS native backend. Not yet implemented because
Perry does not currently support a `watchos` / `watchos-simulator` target.

## Prerequisites (on the Perry side)

Tracked in the Perry repo: https://github.com/PerryTS/perry/issues (see
the "watchOS target" issue filed from this workspace).

Perry needs to grow:

- `--target watchos` and `--target watchos-simulator` (arm64 / arm64_32 / x86_64 sim)
- A WatchKit app shell (Info.plist, entitlements, `WKExtensionDelegate`)
- Static-linking against WatchKit, SwiftUI, and GameController

## What this crate will need to implement

Once Perry supports the target, this crate should mirror `native/tvos/` with
these watchOS-specific adaptations:

1. Render loop driven by `WKInterfaceController` / SwiftUI rather than `CADisplayLink`.
2. wgpu on the Metal surface owned by the watchOS interface.
3. `bloom_get_platform()` returns `8.0` (the `Platform.WATCHOS` constant already
   reserved in `src/core/index.ts`).
4. **Digital Crown binding** — in the WatchKit controller, observe
   `WKCrownSequencer` and forward rotation deltas via
   `engine().input.accumulate_crown_rotation(delta)`. Games read them with
   `getCrownRotation()` (already plumbed through the FFI manifest and every
   existing native backend, returning 0 where no crown exists).
5. Touch input: reuse `set_touch(...)` like iOS — a single-finger tap is the
   primary "button" UI affordance on the watch.
6. Asset resolution: the watchOS app bundle layout is similar to iOS; paths
   should resolve to `[[NSBundle mainBundle] resourcePath]`.

## Out of scope for the first cut

- Complications, widgets, and background refresh.
- Always-On / low-power rendering paths.
- HealthKit / sensor integration.

These can be bolted on once the baseline game-loop path works.
