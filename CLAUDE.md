# CLAUDE.md

This file provides guidance to Claude Code when working with the Bloom Engine codebase.

## Project Overview

Bloom is a native TypeScript game engine compiled by [Perry](../../perry/perry) (a TypeScript AOT compiler). It provides a simple, function-based API for 2D/3D games that compiles to Metal, DirectX 12, Vulkan, OpenGL, and WebGPU.

## Build Commands

```bash
# Native (macOS)
cd native/macos && cargo build --release

# Native (Windows — set INTERPROCEDURAL_OPTIMIZATION=OFF so the Jolt
# cmake build doesn't trip over LTO with the MSVC toolchain)
INTERPROCEDURAL_OPTIMIZATION=OFF cargo build --release --manifest-path native/windows/Cargo.toml

# Tests: unit + golden-image suite (run before any renderer PR)
cd native/shared && cargo test --release
BLOOM_UPDATE_GOLDEN=1 cargo test --release   # regenerate goldens — commit ONLY the ones your change intentionally moved (mean tol 2, outliers 1%)

# Web/WASM
cd native/web && cargo check --target wasm32-unknown-unknown
./native/web/build.sh [game.ts]              # Full web build pipeline

# Check shared code compiles for all targets
cd native/shared && cargo check                                          # native (default features)
cd native/shared && cargo check --target wasm32-unknown-unknown --no-default-features --features web  # WASM
```

After an engine-only rebuild, a consuming game does NOT relink by
itself: `perry compile` skips the link if the game's `main.ts` is
untouched — touch it first. After ANY change to `package.json`'s
native-function manifest, also delete the game's `.perry-cache/`.
Games should link with `perry compile … --debug-symbols` so `main.pdb`
lands next to the exe (see `docs/crash-triage-windows.md`).

## Architecture

```
src/                  TypeScript API (compiled by Perry)
  core/               Window, input, game loop, runGame()
  shapes/             2D shapes + collision
  textures/           Image loading, sprites
  text/               Font rendering
  audio/              Sound + music
  models/             3D models, skeletal animation
  math/               Vectors, matrices, easing
  scene/              Scene graph, frame callbacks, lighting
  physics/            Jolt-backed rigid + soft bodies, character, vehicles

native/               Rust implementations (one crate per platform)
  shared/             Cross-platform core
                      - renderer/: wgpu 29 renderer — deferred MRT
                        (hdr/material/velocity/albedo), TSR upscaling,
                        cascaded shadows, SSR, Lumen-class GI (screen
                        probes; HW ray-query / SDF-clipmap / Hi-Z tiers,
                        mesh cards, WSRC), material system with a
                        5-bind-group ABI (shaders/material_abi.wgsl),
                        transient texture pool, 2D draw layer.
                        WGSL lives as strings in renderer/shaders/*.rs
                      - profiler.rs: CPU+GPU pass profiler (enabling it
                        inserts a blocking per-frame GPU sync — never
                        benchmark with it on)
                      - audio/: control/render split over a lock-free
                        SPSC ring (mod/render/stream/decode)
                      - text_renderer.rs: fontdue text, rasterized at
                        physical resolution
                      - string_header.rs: Perry string ABI (read its
                        header comment before touching FFI strings)
                      - ffi_core/: define_core_ffi! macro — the shared
                        FFI surface each platform crate instantiates
                      - physics_jolt.rs: JoltPhysics wrapper (native only)
                      - jolt_sys.rs: C ABI bindings to bloom_jolt shim
                      - textures.rs, models.rs, scene.rs, etc.
  third_party/
    JoltPhysics/      Jolt 5.5.0 submodule (built via cmake crate)
    bloom_jolt/       C++ shim exposing Jolt behind a C ABI
                      - include/bloom_jolt.h, src/bloom_jolt.cpp
  macos/              Metal + AppKit + Core Audio
  ios/                Metal + UIKit + Core Audio
  tvos/               Metal + UIKit + GCController
  windows/            DirectX 12 + Win32 + WASAPI
  linux/              Vulkan/OpenGL + X11 + PulseAudio
  android/            Vulkan/OpenGL ES + NativeActivity + AAudio
  web/                WebGPU/WebGL + Canvas + Web Audio API (WASM via wasm-pack)
  visionos/           Metal + UIKit-style shell (wgpu; iOS/tvOS-family port)
  watchos/            SwiftUI Canvas (2D) + SceneKit (3D) — no wgpu/Jolt on
                      the watch; own .glb loader; built via Perry's
                      watchos-swift-app feature (see docs/watchos-target.md)
```

## FFI Pattern

Each platform implements ~470 `bloom_*` FFI functions declared in `package.json` under `perry.nativeLibrary.functions`. Native platforms use `#[no_mangle] extern "C"`, web uses `#[wasm_bindgen]`.

String parameters are `i64` on native (Perry StringHeader pointers) and NaN-boxed string IDs on web (converted by JS glue layer).

### Hard-won FFI rules (violating these produced real shipped bugs)

- **Every new native MUST be declared in `package.json`'s manifest.**
  Perry silently no-ops undeclared functions — no error, the call just
  does nothing. Consumers must clear `.perry-cache/` after manifest
  changes.
- **Never return packed text for per-frame parsing (EN-020).** Perry
  0.5.x `split()`/`parseFloat()` read past their own slice allocations;
  parsing a delimited FFI string every frame is a crash lottery the
  engine cannot pad away. Cross numbers as `f64` FFIs; strings cross
  whole and get drawn, not parsed (see the `bloom_profiler_row_*`
  numeric ABI and `docs/tickets.md` § EN-020).
- Engine→Perry strings go through `string_header::alloc_perry_string`
  (tail-padded); Perry→engine through `str_from_header` (validated,
  never UB). Don't hand-roll headers.
- **Perry 0.5.1171 i64-array regression:** passing a TS `number[]` to
  an `i64` param is broken — use the all-f64 scratch pattern
  (`bloom_mesh_scratch_*`) like createMesh does.
- Engine TS in `src/` is compiled by Perry too, so Perry codegen quirks
  apply here as well (no reachable `throw`, explicit object keys in
  returns — the shooter's `docs/perry-quirks.md` is the reference list).

### Runtime/debug behavior worth knowing

- `panic = "abort"` on all native targets: a Rust panic prints to
  stderr then fast-fails (0xC0000409). An AV instead triggers the
  Windows crate's crash filter, which prints `main.exe+RVA` and writes
  a minidump to the game's `tools/.testout/dumps/` —
  `docs/crash-triage-windows.md` is the triage runbook.
- **WGSL compiles at runtime** (`create_shader_module`), so
  `cargo build` does NOT validate shaders — boot a game (or the golden
  suite) after any WGSL edit.
- `begin_frame` resets per-frame lighting state; renderer FFI calls
  before `initWindow` panic with "Engine not initialized".

Physics FFI (121 of the ~470) is generated by the `define_physics_ffi!` macro in `native/shared/src/physics_jolt.rs`; each platform crate invokes it once to re-export the full surface. On web the same surface is wasm_bindgen wrappers forwarding to `native/web/jolt_bridge.js`, which drives JoltPhysics.js.

## Web/WASM Target

The web target uses a two-module WASM architecture:
- **Perry WASM** (game logic) imports bloom_* functions under the `"ffi"` namespace
- **bloom_web.wasm** (rendering engine) compiled from `native/web/` via wasm-pack
- **JS glue** (`bloom_glue.js`) bridges both modules, handles DOM events, asset fetching, Web Audio, and the rAF game loop. For game builds, `build.sh` runs `splice_game.py` to inject this bootstrap into Perry's self-contained WASM HTML (which carries the `rt` runtime + NaN-boxing) and gate the game's `bootPerryWasm()` on engine readiness. Perry's runtime already decodes FFI args to plain JS values (`wrapFfiForI64`), so the bridge passes plain values straight through — no manual NaN-boxing.

Key features flags in `native/shared/Cargo.toml`:
- `default = ["mp3", "jolt"]` — includes minimp3 (C dep, not WASM-compatible) + Jolt physics
- `jolt` — compiles the C++ Jolt shim via cmake on native; no-op on wasm32 (web uses JoltPhysics.js)
- `web` — uses web-time instead of std::time::Instant

The web crate exposes `_str` variants (accepting `&str`) and `_bytes` variants (accepting `&[u8]`) for functions that take strings or file data. Perry's runtime decodes NaN-boxed FFI args to plain JS values (`wrapFfiForI64`) before the glue sees them; the glue routes string/asset params to the `_str`/`_bytes` variants and fetches assets via sync XHR.

## Key Files

| File | Purpose |
|------|---------|
| `package.json` | FFI function manifest + per-platform build config |
| `src/core/index.ts` | Core API: window, input, drawing, `runGame()` |
| `src/physics/index.ts` | Physics API (see `docs/physics.md` for architecture) |
| `native/shared/src/renderer/mod.rs` | Renderer root: frame orchestration, lighting, GI plumbing |
| `native/shared/src/renderer/material_system.rs` | Material draws, per-slot UBOs, prev-frame model history (EN-022) |
| `native/shared/src/renderer/shaders/` | All WGSL (core scene, ssgi/SSR, gi, post) as Rust strings |
| `native/shared/shaders/material_abi.wgsl` | The 5-bind-group material ABI games compile against |
| `native/shared/src/string_header.rs` | Perry string ABI both directions (padded alloc + validated read) |
| `native/shared/src/engine.rs` | EngineState with timing, frame callbacks |
| `native/shared/src/physics_jolt.rs` | Jolt handle registries + `define_physics_ffi!` macro |
| `native/third_party/bloom_jolt/` | C++ shim wrapping JoltPhysics behind a C ABI |
| `native/web/src/lib.rs` | Web platform: all FFI functions via wasm-bindgen |
| `native/web/jolt_bridge.js` | Web physics: JoltPhysics.js implementation of FFI |
| `native/web/bloom_glue.js` | Engine bootstrap + FFI bridge: loads bloom_web, builds `__ffiImports`, input, asset loading, Web Audio, rAF game loop |
| `native/web/index.html` | Engine-only standalone page (no game); creates `__bloomReady` + loads bloom_glue.js |
| `native/web/splice_game.py` | Splices the Bloom bootstrap into Perry's self-contained WASM HTML, gating `bootPerryWasm()` on `__bloomReady` |
| `native/web/build.sh` | Build script: wasm-pack + wasm-opt + Perry compile + splice/assembly |

## Where the truth lives

- `bloom-renderer-spec-v2.md` — the (only) renderer plan; its header
  carries the as-built status vs. plan.
- `docs/tickets.md` — EN-xxx work items with current status.
- `docs/perf/` — per-feature design docs + `lumen-roadmap.md` for GI.
- `docs/crash-triage-windows.md` — native-fault runbook (the engine
  self-reports crashes since 2026-07).
- The Bloom Shooter (`../shooter`) is the flagship consumer; its
  `CLAUDE.md` + `docs/perry-quirks.md` document the Perry-side rules
  games (and engine `src/` TS) must follow.
