# watchOS Target

Bloom games can run on Apple Watch. Unlike every other Bloom platform, watchOS
has **no Metal/wgpu** available to third-party apps, so the watch target does not
use the wgpu renderer at all. Instead the engine emits a **draw list** that a
SwiftUI `Canvas` rasterizes — the game's imperative draw calls work unchanged.

## Architecture

```
Game.ts ─(perry --target watchos --features watchos-swift-app)─┐
                                                               │ FFI
                                                               ▼
                          bloom_draw_*() calls ──> draw-command list (Rust, native/watchos)
                                                               │
                                                               │ snapshot per frame
                                                               ▼
                          BloomWatchApp.swift  ── SwiftUI Canvas rasterizes the list
                                                               │
                                                               ▼
                          Apple Watch: Canvas + Digital Crown + taps
```

`native/watchos/` is a small crate (no wgpu, no Jolt) that turns `bloom_draw_rect`,
`bloom_draw_texture`, text, etc. into a flat draw-command buffer. `BloomWatchApp.swift`
owns the `@main struct App: App`, spawns the game on a background thread, and on
each frame copies the latest draw list and replays it into a SwiftUI `Canvas`.

## Building

watchOS builds go through Perry. The engine's watch crate (`native/watchos`) and
Perry's runtime are tier-3 Rust targets built with nightly `-Z build-std`. See
the Perry [watchOS platform docs](../../perry/perry/docs/src/platforms/watchos.md)
for the full toolchain setup; the engine-specific parts are:

- Compile the game with **`--features watchos-swift-app`** so the engine's
  `@main` SwiftUI shell is the process entry (not Perry's default UI-tree shell).
- Three architectures: `watchos-simulator` (arm64 sim), `watchos` (arm64,
  Series 9+), and arm64_32 (Series 4–8/SE, via `PERRY_WATCHOS_ARM64_32=1`). A fat
  `lipo` of the latter two covers every watch in one App Store build.

```bash
# engine watch crate for the simulator
cargo +nightly build -Z build-std=std,panic_abort --release \
  --target aarch64-apple-watchos-sim   # (native/watchos)

PERRY_RUNTIME_DIR=<perry>/target/aarch64-apple-watchos-sim/release \
  perry compile main.ts -o game --target watchos-simulator --features watchos-swift-app
```

> **Deployment-target floor: watchOS 10.0.** `BloomWatchApp.swift` uses
> SwiftUI's `onChange(of:initial:)`, which is watchOS 10+. Builds below 10.0
> fail to compile the Swift shell, so `PERRY_WATCHOS_MIN` cannot go lower than
> `10.0`.

## Game Loop

The watch shell drives frames; the game's `runGame()` callback works as on every
other platform:

```typescript
runGame((dt) => {
  clearBackground(Colors.SKYBLUE);
  drawRect(playerX, playerY, 16, 16, Colors.RED);
});
```

The blocking `while (!windowShouldClose())` loop is not used on watchOS — the
SwiftUI shell owns the run loop and calls into the game thread.

## Input

watchOS has no keyboard or pointer. Two input sources are bridged:

```typescript
const turn = getCrownRotation();   // Digital Crown delta (radians) since last call
const touches = getTouchCount();   // taps on the watch face
```

- **Digital Crown** — Swift's `.digitalCrownRotation` reports a delta each frame
  via `bloom_watchos_crown_delta`; read it with `getCrownRotation()`. Reading
  consumes the accumulator.
- **Taps** — surfaced through the same touch API as iOS (`getTouchCount()` /
  `getTouchX/Y()`), so `isWatch()` branches can treat any tap as e.g. "jump".

Use `isWatch()` (or `getPlatform() === Platform.WATCH`) to gate watch input.

## 2D Camera

`beginMode2D()` / `endMode2D()` are supported: the engine emits `BEGIN_2D` /
`END_2D` marker commands carrying the camera offset, target, and zoom, and the
SwiftUI Canvas applies the matching `CGAffineTransform` while replaying the draw
list between the markers. This lets a side-scroller frame the world correctly on
the small screen — set the zoom so a sensible number of tiles fit the watch
width.

## Assets

Asset files ship inside the `.app` bundle; the native layer resolves relative
paths against the bundle resource path. Textures, sounds, fonts, and level/text
files via `readFile` all work. Audio uses a watchOS-native mixer
(`BloomWatchAudio.swift`).

## Localization

The user's language is reported from Swift at launch (`Locale.preferredLanguages`)
through `bloom_watchos_set_language`, so `getLanguage()` returns the real device
language and the game's i18n works as on other platforms. (Before engine #63 this
was hardcoded to English.)

## Limitations

- **No Metal post-processing** — the `bloom_postfx.metal` chromatic-aberration /
  film-grain / sun-shaft pass is unavailable (SCNTechnique/SCNRenderer absent
  from the watchOS SDK). Games run fine without it.
- **No 3D** — the Canvas rasterizer is 2D only; the wgpu/Jolt paths are not built.
- **Small screen & RAM** — design for 40–49 mm faces and keep memory modest.
- **Simulator can't run device-arch builds** — the sim is arm64; an arm64_32
  build only runs on real pre-S9 hardware.

## How It Works

The watch crate (`native/watchos/src/lib.rs`) keeps a `WatchState` with the
draw-command buffer, input accumulators, screen size, and the reported language.
`bloom_draw_*` append commands; `bloom_watchos_copy_draw_list` snapshots them for
Swift each frame. Strings cross the FFI boundary using Perry's 20-byte
`StringHeader` layout (both incoming args and returned strings — `read_file`
returns paths/level data this way). Because the renderer is just a draw-list
replayer, the same game code that targets desktop and mobile runs on the watch
unchanged.
