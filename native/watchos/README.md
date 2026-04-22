# bloom-watchos

Bloom's watchOS backend — 2D + 3D game-engine runtime running on Apple Watch.

The crate is the watchOS counterpart of `native/tvos/` but takes a very
different approach: instead of owning a Metal surface and running wgpu, it
forwards bloom's rendering calls into SwiftUI + SceneKit via the SwiftUI
`Canvas` (2D) and `SceneView` (3D) hosted by a `@main struct App`. The Swift
side is compiled by Perry's `watchos-swift-app` feature (PerryTS/perry#118)
and links into the same `.app` as the Rust staticlib.

## Build

```bash
perry compile --target watchos-simulator --features watchos-swift-app src/main.ts -o MyApp
xcrun simctl install booted MyApp.app
xcrun simctl launch booted com.perry.MyApp
```

Apple Watch Series 10 sim (watchOS 26.4) is the validated platform.

## What works

| Layer | Coverage |
|---|---|
| **2D** | `drawRect` / `drawCircle` / `drawLine` / `drawTriangle` / `drawTexturePro` (PNG via `TextureCache`), `drawTextRgba` with system font, parallax scrolling |
| **3D immediate-mode** | `drawCube` / `drawSphere` / `drawCylinder` / `drawPlane` / `drawGrid` (+ wire variants), `beginMode3D` camera — each frame's 3D draws rebuild an SCNNode tree under `contentRoot` |
| **3D retained scene graph** | `bloom_scene_create_node` / `_destroy_node` / `_set_transform` / `_set_material_color` / `_set_material_pbr` / `_set_material_texture` / `_update_geometry` all delta-synced via dirty-flag protocol (no full rebuild per frame) |
| **glTF loader** | Hand-rolled `.glb` parser: multi-mesh × multi-primitive, scene hierarchy, 5-slot PBR textures (baseColor, normal, metalRoughness, emissive, occlusion), JOINTS_0 / WEIGHTS_0 + skins + inverseBindMatrices, keyframe animations (TRS channels with linear + slerp interpolation). Validated against DamagedHelmet, Buggy (205 nodes, 34 multi-primitive meshes), Fox (skinned + animated) |
| **Input** | Digital Crown → horizontal axis via `getCrownRotation()`, tap → `getTouchCount/X/Y`, layout size pushed via `getScreenWidth/Height` |
| **Audio** | `AVAudioPlayer`-backed sounds + looping music via Swift bridge (`BloomWatchAudio.swift`) |
| **Assets** | Perry's `watchos` bundle step (PerryTS/perry#123) auto-copies `assets/`, `logo/`, `resources/`, `images/` into `<app>.app/`. Runtime paths resolve via `[[NSBundle mainBundle] resourcePath]` |
| **Post-fx (pure SwiftUI)** | Vignette (radial-gradient overlay) + exposure (`.brightness`) |

## What's partial

### Metal-shader post-fx (chromatic aberration / film grain / sun shafts)

`shaders/bloom_postfx.metal` + Perry's `metal_sources` (PerryTS/perry#124)
compile and bundle `<app>.app/default.metallib` correctly. An `SCNTechnique`
with a single DRAW_QUAD pass attaches to `SceneView` via the `technique:`
parameter; SceneKit runs the fragment shader, and the `COLOR` input/output
routing works (verified by a debug shader that returned constant red and
filled the entire scene viewport).

**Blocker**: pushing per-draw uniforms to the technique. SCNTechnique's
`handleBindingOfSymbol:usingBlock:` takes an `SCNRenderer` parameter, and
`SCNRenderer.h` is absent from the watchOS SDK — the method is uncallable
from Swift. `setObject(NSData, forKeyedSubscript:)` works for SCNProgram
per-material shaders but doesn't deliver values to SCNTechnique's Metal
`[[buffer(N)]]` slots. Apple also didn't ship
`.colorEffect(Shader)` / `.layerEffect(Shader)` on watchOS (iOS 17+ only).

So `bloom_set_chromatic_aberration` / `_film_grain` / `_sun_shafts` store
their values but produce no visible effect today. The math is staged and the
pipeline attaches — flipping this on is ~20 lines once Apple adds either
API. Tracking: Bloom-Engine#16, Bloom-Engine#18.

## Architecture

```
BloomWatchApp.swift (@main struct App)
 ├─ init(): spawns game thread → _perry_user_main (renamed via watchos-swift-app)
 └─ body: WindowGroup {
      BloomRootView
       ├─ ZStack {
       │    BloomSceneView (3D)                    ← SceneView wrapping SCNScene
       │     ├─ contentRoot (immediate-mode 3D)
       │     ├─ retainedRoot (bloom_scene_* nodes) ← delta-sync from scene::drain_dirty
       │     ├─ lightsRoot
       │     └─ cameraNode
       │    Canvas (2D)                            ← drains draw_list each tick
       │  }
       └─ .digitalCrownRotation + .onTapGesture → bloom_watchos_* inbound FFI
    }

native/watchos/src/
 ├─ lib.rs                  ~275 FFI entry points. Perry string decoding, game loop,
 │                          bloom_scene_attach_model traverses glTF hierarchy.
 ├─ draw_list.rs            2D draw commands + 3D immediate-mode primitives + camera.
 │                          Mutex-guarded building/ready lists; Swift bulk-copies each
 │                          frame via bloom_watchos_copy_draw_list.
 ├─ scene.rs                Retained scene graph. Nodes keyed by 1-based handle with
 │                          dirty-flag protocol (DIRTY_TRANSFORM / _MATERIAL / _GEOMETRY
 │                          / _VISIBILITY / _PARENT / _SKIN). Destroyed nodes drained
 │                          separately. Static scenes sync in O(0) per frame.
 ├─ models.rs               .glb parser — meshes[]/primitives[]/nodes[]/scenes[]/skins[]/
 │                          animations[]. Hand-rolled JSON value parser (no serde).
 │                          Indexless primitives and u8/u16 joint indices supported.
 ├─ textures.rs             PNG + JPEG byte registry with CGImage caching on the
 │                          Swift side. glb-embedded textures written to the app's
 │                          tmp dir and handle-registered for SceneKit consumption.
 ├─ audio.rs                Thin shim forwarding to BloomWatchAudio.swift's
 │                          AVAudioPlayer-backed registry.
 ├─ postfx.rs               Atomic storage for CA / vignette / grain / exposure /
 │                          sun-shafts params. Read-back via bloom_watchos_postfx_state.
 ├─ ffi_stubs.rs            Auto-generated (`gen_stubs.js`) no-op stubs for every
 │                          bloom_* symbol not overridden above.
 └─ shaders/bloom_postfx.metal  Three-effect combined fragment (CA + grain + sun
                                shafts). Compiled to default.metallib by Perry,
                                attached via BloomPostFXTechnique. See "What's
                                partial" above for the current uniform-push blocker.
```

## Out of scope

- Complications, widgets, background refresh
- Always-On / low-power rendering
- HealthKit / sensor integration
- Morph-target animations (glTF `weights` path)
- `SCNProgram` per-material custom shaders
- Spatial audio
