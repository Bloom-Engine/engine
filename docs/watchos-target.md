# watchOS Target

Bloom games can run on Apple Watch. Unlike every other Bloom platform, watchOS
has **no wgpu and no direct Metal surface** for third-party apps, so the watch
target does not use the wgpu renderer at all. Instead the engine emits a
**draw list** whose 2D commands a SwiftUI `Canvas` rasterizes and whose 3D
commands drive a SceneKit `SceneView` — the game's imperative draw calls work
unchanged.

## Architecture

```
Game.ts ─(perry --target watchos --features watchos-swift-app)─┐
                                                               │ FFI
                                                               ▼
                          bloom_draw_*() calls ──> draw-command list (Rust, native/watchos)
                                                               │
                                                               │ snapshot per frame
                                                               ▼
                          BloomWatchApp.swift ── ZStack of:
                            • BloomSceneView (SceneKit) — 3D commands (kinds 20-29)
                            • SwiftUI Canvas — 2D commands, drawn on top
                                                               │
                                                               ▼
                          Apple Watch: SceneKit + Canvas + Digital Crown + taps
```

`native/watchos/` is a crate (no wgpu, no Jolt) that turns `bloom_draw_rect`,
`bloom_draw_texture`, text, etc. into a flat draw-command buffer.
`BloomWatchApp.swift` owns the `@main struct App: App`, spawns the game on a
background thread, and on each frame copies the latest draw list. 2D commands
replay into a SwiftUI `Canvas`; 3D commands (kinds 20-29) are filtered out of
the Canvas and handled by a SceneKit `SceneView` layered underneath.

## 3D

3D works through SceneKit (Metal-backed under the hood, no wgpu involved):

- **Immediate mode** — `drawCube`, `drawSphere` (+ wire variants) map to
  `bloom_draw_cube` / `bloom_draw_sphere` etc.
- **Retained scene graph** — the `bloom_scene_*` FFI surface delta-syncs
  scene nodes to `SCNNode`s (`contentRoot`/`retainedRoot`/`lightsRoot` +
  a camera node in `BloomWatchApp.swift`).
- **Models** — a hand-rolled `.glb` loader (`native/watchos/src/models.rs`),
  validated against DamagedHelmet / Buggy / Fox.
- `bloom_watchos_has_3d` reports availability.

What is *not* built on watchOS is the wgpu renderer and Jolt physics — see
Limitations.

## Building

watchOS builds go through Perry. The engine's watch crate (`native/watchos`) and
Perry's runtime are tier-3 Rust targets built with nightly `-Z build-std`. See
the Perry [watchOS platform docs](../../../perry/perry/docs/src/platforms/watchos.md)
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
> SwiftUI's two-parameter `onChange(of:) { old, new in }` overload, which is
> watchOS 10+. Builds below 10.0 fail to compile the Swift shell, so
> `PERRY_WATCHOS_MIN` cannot go lower than `10.0`.

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

- **No Metal post-processing yet** — the `bloom_postfx.metal` chromatic-
  aberration / film-grain / sun-shaft pass is staged but not visible:
  `SCNTechnique` attaches fine (and loads from `default.metallib`), but
  `SCNRenderer` is absent from the watchOS SDK, so per-frame uniforms can't
  be pushed. Games run fine without it.
- **No wgpu renderer / no Jolt physics** — 3D goes through SceneKit (see the
  3D section above); the deferred-MRT pipeline, Lumen GI, and physics are
  not available on the watch.
- **Small screen & RAM** — design for 40–49 mm faces and keep memory modest.
- **Simulator can't run device-arch builds** — the sim is arm64; an arm64_32
  build only runs on real pre-S9 hardware.

## How It Works

The watch crate (`native/watchos/src/lib.rs`) keeps a `WatchState` with the
draw-command buffer, input accumulators, screen size, and the reported language.
`bloom_draw_*` append commands; `bloom_watchos_copy_draw_list` snapshots them for
Swift each frame. Strings cross the FFI boundary using Perry's 20-byte
`StringHeader` layout (both incoming args and returned strings — `read_file`
returns paths/level data this way). Because the renderer is a draw-list
replayer (Canvas for 2D, SceneKit for 3D), the same game code that targets
desktop and mobile runs on the watch unchanged.
