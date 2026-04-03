# CLAUDE.md

This file provides guidance to Claude Code when working with the Bloom Engine codebase.

## Project Overview

Bloom is a native TypeScript game engine compiled by [Perry](../../perry/perry) (a TypeScript AOT compiler). It provides a simple, function-based API for 2D/3D games that compiles to Metal, DirectX 12, Vulkan, OpenGL, and WebGPU.

## Build Commands

```bash
# Native (macOS example)
cd native/macos && cargo build --release

# Web/WASM
cd native/web && cargo check --target wasm32-unknown-unknown
./native/web/build.sh [game.ts]              # Full web build pipeline

# Check shared code compiles for all targets
cd native/shared && cargo check                                          # native (default features)
cd native/shared && cargo check --target wasm32-unknown-unknown --no-default-features --features web  # WASM
```

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

native/               Rust implementations (one crate per platform)
  shared/             Cross-platform core (~7000 lines)
                      - renderer.rs: wgpu + WGSL shaders (2D/3D, shadows, post-FX)
                      - audio.rs: platform-agnostic mixer
                      - text_renderer.rs: fontdue-based text
                      - textures.rs, models.rs, scene.rs, etc.
  macos/              Metal + AppKit + Core Audio
  ios/                Metal + UIKit + Core Audio
  tvos/               Metal + UIKit + GCController
  windows/            DirectX 12 + Win32 + WASAPI
  linux/              Vulkan/OpenGL + X11 + PulseAudio
  android/            Vulkan/OpenGL ES + NativeActivity + AAudio
  web/                WebGPU/WebGL + Canvas + Web Audio API (WASM via wasm-pack)
```

## FFI Pattern

Each platform implements ~130 `bloom_*` FFI functions declared in `package.json` under `perry.nativeLibrary.functions`. Native platforms use `#[no_mangle] extern "C"`, web uses `#[wasm_bindgen]`.

String parameters are `i64` on native (Perry StringHeader pointers) and NaN-boxed string IDs on web (converted by JS glue layer).

## Web/WASM Target

The web target uses a two-module WASM architecture:
- **Perry WASM** (game logic) imports bloom_* functions under the `"ffi"` namespace
- **bloom_web.wasm** (rendering engine) compiled from `native/web/` via wasm-pack
- **JS glue** (`index.html`) bridges both modules, handles DOM events, string conversion, asset fetching, and Web Audio

Key features flags in `native/shared/Cargo.toml`:
- `default = ["mp3"]` — includes minimp3 (C dep, not WASM-compatible)
- `web` — uses web-time instead of std::time::Instant

The web crate exposes `_str` variants (accepting `&str`) and `_bytes` variants (accepting `&[u8]`) for functions that take strings or file data. The JS glue converts Perry NaN-boxed values via `__perryToJsValue` and fetches assets via sync XHR.

## Key Files

| File | Purpose |
|------|---------|
| `package.json` | FFI function manifest + per-platform build config |
| `src/core/index.ts` | Core API: window, input, drawing, `runGame()` |
| `native/shared/src/renderer.rs` | wgpu renderer (2D/3D, ~2600 lines) |
| `native/shared/src/engine.rs` | EngineState with timing, frame callbacks |
| `native/web/src/lib.rs` | Web platform: all FFI functions via wasm-bindgen |
| `native/web/index.html` | JS glue: FFI bridge, input, asset loading, Web Audio |
| `native/web/build.sh` | Build script: wasm-pack + wasm-opt + assembly |
