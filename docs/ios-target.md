# iOS target

Metal via wgpu, UIKit for the window and touch, CoreAudio for sound, Jolt for
physics. Shipped: [Bloom Jump](https://apps.apple.com/us/app/bloom-jump/id6761447092)
is a Bloom game on the App Store.

The engine ships a **static library, not an app**. There is no Xcode project
here — Perry owns the app shell (`Info.plist`, `UIApplicationMain`, the bundle,
signing). `native/ios/` is `libbloom_ios.a` and nothing else.

## Building a game

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

perry setup ios                # once: App Store Connect API key + team id
perry setup ios --development  # once per device: registers the UDID, mints a profile

perry compile src/main.ts -o build/Game --target ios --features ios-game-loop
xcrun devicectl device install app --device <UDID> build/Game.app
xcrun devicectl device process launch --console --device <UDID> <bundle-id>
```

`--features ios-game-loop` is **mandatory** for a game and is the single most
common way to get a black screen. UIKit requires `UIApplicationMain()` to own
the main thread forever, but a Bloom game loop (`while (!windowShouldClose())`)
wants it too. The feature makes Perry emit `_perry_user_main` — run on a spawned
game thread — and hands the main thread to `UIApplicationMain`. Without it you
get a plain `main()`, the app links, and no window ever appears.

The handshake between the two, both implemented in `native/ios/src/lib.rs`:

- `perry_register_native_classes()` — registers `BloomMetalView` (a `UIView`
  whose `+layerClass` is `CAMetalLayer`) before `UIApplicationMain` starts.
- `perry_scene_will_connect()` — creates the `UIWindow`, the view, the wgpu
  surface and the engine, on the main thread, from `didFinishLaunching`.
- `bloom_init_window()` — runs on the *game* thread and waits for the engine
  the scene delegate is building. It panics after 10s if that never happened,
  which in practice means `ios-game-loop` was omitted.

Perry omits `UIApplicationSceneManifest` from the plist in game-loop mode; older
build notes told you to strip it by hand afterwards, and that is no longer
needed.

## Assets

Perry copies `assets/` (and `logo/`, `resources/`, `images/`) from the project
root — located by walking up to the nearest `package.json` — into the `.app`.

Relative paths do **not** resolve against the working directory on iOS: the CWD
is not the app bundle. Every asset-loading FFI therefore routes its path through
the platform's `resolve_path()` hook, which prepends
`NSBundle.mainBundle.resourcePath`. If you add a new FFI that opens a file for
*reading*, it must resolve too:

```rust
let path = $crate::string_header::str_from_header(path_ptr);
let path: &str = &bloom_resolve_asset_path(path);
```

Skipping that is silent on desktop and fails only on device. `loadMaterial` /
`compileMaterialFromFile` was missing it until 2026-07, and the symptom was a
scene that rendered with every from-file material dropped — water invisible,
terrain on a fallback shader — with only a `canonicalize … No such file` line on
the console to say so. FFIs that *write* a file (`takeScreenshot`,
`dumpShadowMap`) deliberately do not resolve — the bundle is read-only.

## Input

Touch is real UIKit multitouch: up to 10 points, `setMultipleTouchEnabled: YES`,
coordinates scaled from points to pixels so they share a space with
`getScreenWidth()` / `getScreenHeight()`.

Touch 0 is also synthesised into mouse button 0, so mouse-driven games work
unmodified. That is a trap for anything multi-touch: `isMouseButtonDown(0)` is
true whenever *any* first finger is down, so an FPS that reads it as "fire" will
shoot while you are steering with the movement stick. Read the touch API
directly instead.

**Touch slots are not a dense list.** `getTouchCount()` is the number of live
fingers, but touch points are addressed by *slot*, and slots go sparse the moment
a finger lifts out of order — hold two fingers, lift the first, and the live
finger is at slot 1 while the count is 1. Iterating `0..getTouchCount()` then
reads slot 0 — released, but still holding its last coordinates — as if it were
live, which presents as a finger frozen where it left the glass. Scan
`0..getMaxTouchPoints()` and skip slots that `isTouchActive(i)` rejects.

Gamepad is **not** implemented on iOS (`GCController` is never polled), despite
the framework being linked. `isGamepadAvailable()` returns false. tvOS has the
code to copy if this is ever needed.

## Renderer notes

- Backend is pinned to `wgpu::Backends::METAL`; present mode is `Fifo`.
- Device creation retries: some real devices (iPhone 16 Pro / A18) advertise
  `EXPERIMENTAL_RAY_QUERY` on the adapter and then reject it at device creation,
  which would abort on launch. The fallback re-requests with `adapter.limits()`,
  because iOS GPUs also cap individual limits (e.g.
  `max_inter_stage_shader_variables` at 15, under wgpu's default 16) below the
  desktop defaults.
- Atmosphere LUTs are tiered down on iOS (`renderer/atmosphere_lut.rs`).
- Deployment target is pinned to iOS 17.0 by Perry.

## Known gaps

- **No CI build.** No workflow compiles `native/ios/`; the only iOS gate is
  `tools/validate-ffi.js`, which parses `lib.rs` for symbol names and proves
  nothing about whether the crate compiles or runs.
- **EN-024** — iOS reports pixels where macOS reports points, so `getScreenWidth()`
  and 2D HUD coordinates do not carry across Apple targets. Games currently
  compensate themselves (scale the 2D pass through a `beginMode2D` zoom).
