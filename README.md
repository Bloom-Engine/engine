# Bloom Engine

**Native games from TypeScript.**

Write TypeScript. Ship native games — and now the web too.
Bloom compiles your game to Metal, DirectX 12, Vulkan, OpenGL, and WebGPU — one codebase for every platform.

> **Inspired by [raylib](https://github.com/raysan5/raylib).** Bloom models its public
> API on raylib's — in our view one of the best API designs in gamedev. Bloom is an
> independent implementation, not a port — [how Bloom relates to raylib »](#how-bloom-relates-to-raylib)

## Install

```bash
npm install @bloomengine/engine
```

Or with your preferred package manager:

```bash
bun add @bloomengine/engine
pnpm add @bloomengine/engine
yarn add @bloomengine/engine
```

The npm package ships the TypeScript API alongside the engine's Rust sources and the bundled [JoltPhysics](https://github.com/jrouwe/JoltPhysics) C++ shim, so a single `install` is enough — there's no separate native download step.

You'll also need:

- **Perry** — the TypeScript AOT compiler that turns your game into a native binary or WASM module. It also drives the engine's native build.
- **Rust toolchain** ([rustup.rs](https://rustup.rs)) — Perry invokes Cargo to compile the engine's platform crate the first time you build for each target.
- For web builds only: [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (`cargo install wasm-pack`).

## Quick Start

```typescript
import { initWindow, windowShouldClose, beginDrawing,
         endDrawing, clearBackground, drawText, Colors } from "@bloomengine/engine";

initWindow(800, 450, "My Game");

while (!windowShouldClose()) {
  beginDrawing();
  clearBackground(Colors.SNOW);
  drawText("Hello, Bloom!", 190, 200, 20, Colors.DARKGRAY);
  endDrawing();
}
```

### Web-Compatible Pattern

Use `runGame()` for code that works on both native and web:

```typescript
import { initWindow, runGame, clearBackground, drawText, Colors } from "@bloomengine/engine";

initWindow(800, 450, "My Game");

runGame((dt) => {
  clearBackground(Colors.SNOW);
  drawText("Hello, Bloom!", 190, 200, 20, Colors.DARKGRAY);
});
```

Build for web:

```bash
./native/web/build.sh main.ts
cd dist/web && python3 -m http.server 8080
```

## Features

- **Simple API** — Functions, not classes. The entire API fits on a cheatsheet. ([design rationale](docs/design-api.md))
- **True native** — Compiles to Metal, DirectX 12, Vulkan, OpenGL, and WebGPU via wgpu.
- **Ship everywhere** — macOS, Windows, Linux, iOS, tvOS, Android, and Web from one codebase.
- **Unified 2D/3D** — Shapes, textures, text, 3D models, and audio in one engine.
- **Zero magic** — Explicit game loops, no hidden framework overhead.

## How Bloom relates to raylib

Bloom's public API is heavily inspired by [raylib](https://github.com/raysan5/raylib).
raylib's API is, in our opinion, one of the best in the gamedev space — a flat library
of plain functions, no classes, small enough to learn from a cheatsheet — so we model
ours on it. You'll recognize the shape immediately: `initWindow`, `beginDrawing`,
`clearBackground`, `drawText`, and modules named core / shapes / textures / text /
audio / models.

That's where the relationship ends. **Bloom's implementation is entirely independent —
it does not link against, embed, or call raylib.** Bloom compiles TypeScript directly to
native code via Perry, our LLVM-based AOT compiler, and renders through wgpu (Metal,
DirectX 12, Vulkan, OpenGL, WebGPU). It is not a port or a binding — just an engine that
admires raylib's API design. Thanks to
[Ramon Santamaria (@raysan5)](https://github.com/raysan5) and the raylib community for
setting the bar. ([full design rationale](docs/design-api.md))

## Modules

| Module | Import | Description |
|--------|--------|-------------|
| **Core** | `@bloomengine/engine/core` | Window, game loop, input, timing |
| **Shapes** | `@bloomengine/engine/shapes` | 2D drawing + collision detection |
| **Textures** | `@bloomengine/engine/textures` | Image loading, sprite batching |
| **Text** | `@bloomengine/engine/text` | TTF/OTF font loading and rendering |
| **Audio** | `@bloomengine/engine/audio` | Sound effects + music streaming |
| **Models** | `@bloomengine/engine/models` | 3D model loading (glTF, OBJ), skeletal animation |
| **Math** | `@bloomengine/engine/math` | Vectors, matrices, quaternions, easing |
| **Scene** | `@bloomengine/engine/scene` | Retained scene graph, frame callbacks, lighting |
| **Physics** | `@bloomengine/engine/physics` | Jolt-backed rigid + soft bodies, character, vehicles ([docs](docs/physics.md)) |
| **VFX** | `@bloomengine/engine/vfx` | GPU particle systems + decals |
| **World** | `@bloomengine/engine/world` | `.world.json` loading, validation, instantiation ([docs](docs/world-format.md)) |
| **Mobile** | `@bloomengine/engine/mobile` | Virtual joystick/buttons, touch-input helpers |

## Platforms

| Platform | Graphics API | Input |
|----------|-------------|-------|
| macOS | Metal | Keyboard + mouse |
| Windows | DirectX 12 | Keyboard + mouse |
| Linux | Vulkan / OpenGL | Keyboard + mouse |
| iOS | Metal | Touch + gamepad |
| tvOS | Metal | Siri Remote + gamepad |
| watchOS | SwiftUI Canvas (2D) + SceneKit (3D) | Digital Crown + taps ([docs](docs/watchos-target.md)) |
| Android | Vulkan / OpenGL ES | Touch + gamepad |
| **Web** | **WebGPU / WebGL** | **Keyboard + mouse + touch + gamepad** |

## Architecture

```
src/                  TypeScript API
  core/               Window, input, game loop
  shapes/             2D shapes + collision
  textures/           Image loading, sprites
  text/               Font rendering
  audio/              Sound + music
  models/             3D models
  math/               Vectors, matrices, easing

native/               Rust implementations
  shared/             Cross-platform core (wgpu, fontdue, gltf)
  macos/              Metal + AppKit + Core Audio
  ios/                Metal + UIKit + Core Audio
  tvos/               Metal + UIKit + GCController
  windows/            DirectX 12 + Win32 + WASAPI
  linux/              Vulkan/OpenGL + X11 + ALSA
  android/            Vulkan/OpenGL ES + NativeActivity + AAudio
  web/                WebGPU/WebGL + Canvas + Web Audio (WASM)

examples/
  pong/               Complete working example (~170 lines)
```

## Types

Plain interfaces, no classes:

```typescript
interface Vec2 { x: number; y: number }
interface Vec3 { x: number; y: number; z: number }
interface Color { r: number; g: number; b: number; a: number }
interface Rect { x: number; y: number; width: number; height: number }
interface Camera2D { offset: Vec2; target: Vec2; rotation: number; zoom: number }
interface Camera3D { position: Vec3; target: Vec3; up: Vec3; fovy: number; projection: number }
interface Texture { handle: number; width: number; height: number }
interface Sound { handle: number }
interface Model { handle: number }
```

## Fullscreen

Launch your game in fullscreen by passing `true` as the fourth argument to `initWindow`:

```typescript
initWindow(800, 450, "My Game", true);   // launches fullscreen
initWindow(800, 450, "My Game");         // windowed (default)
```

Toggle fullscreen at runtime:

```typescript
if (isKeyPressed(Key.F11)) {
  toggleFullscreen();
}
```

Fullscreen is supported on macOS (native AppKit fullscreen), Windows (borderless fullscreen), and Linux (EWMH/X11). The width and height you pass are used as the windowed dimensions when exiting fullscreen.

## Skeletal Animation

Bloom supports GPU-accelerated skeletal animation via glTF/GLB models. The pipeline uses 4-bone linear blend skinning with a 128-joint uniform buffer, running entirely on the GPU.

```typescript
import { loadModel, loadModelAnimation, updateModelAnimation, drawModel,
         getTime, Colors } from "@bloomengine/engine";

const character = loadModel("assets/models/character.glb");
const anim = loadModelAnimation("assets/models/character.glb");

// In your game loop:
updateModelAnimation(anim, 0, getTime(), 1.0, 0, 0, 0);
drawModel(character, { x: 0, y: 0, z: 0 }, 1.0, Colors.WHITE);
```

Key functions:
- `loadModel(path)` -- loads GLB with skin data (JOINTS_0, WEIGHTS_0)
- `loadModelAnimation(path)` -- loads skeleton + animation channels from GLB
- `updateModelAnimation(handle, animIndex, time, scale, px, py, pz)` -- samples animation, computes joint matrices
- `drawModel(model, position, scale, tint)` -- renders with GPU skinning

For the full pipeline (Blender export, pitfalls, architecture), see [docs/skeletal-animation.md](docs/skeletal-animation.md).

## Made with Bloom

**[Bloom Jump](https://apps.apple.com/us/app/bloom-jump/id6761447092)** — our first shipped game and a proof point for the engine. A free retro pixel platformer with five hand-crafted levels, 60 FPS, and an original chiptune soundtrack, built entirely with Bloom from one TypeScript codebase running natively on every target.

- [App Store](https://apps.apple.com/us/app/bloom-jump/id6761447092) — iOS, macOS, tvOS, and visionOS
- [App Store (watchOS)](https://apps.apple.com/us/app/bloom-jump-watch/id6779528549) — Apple Watch
- [Google Play](https://play.google.com/store/apps/details?id=com.bloom.jump) — Android

## Links

- [bloomengine.dev](https://bloomengine.dev)
- [Docs](https://bloomengine.dev/docs)
- [Showcase](https://bloomengine.dev/showcase)
- [Brand Guidelines](https://github.com/Bloom-Engine/brand)
