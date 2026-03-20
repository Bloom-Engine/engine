# Bloom Engine

**Native games from TypeScript.**

Write TypeScript. Ship native games. No browser, no C++.
Bloom compiles your game to Metal, DirectX 12, Vulkan, and OpenGL — one codebase for every platform.

## Quick Start

```typescript
import { initWindow, windowShouldClose, beginDrawing,
         endDrawing, clearBackground, drawText, Colors } from "bloom";

initWindow(800, 450, "My Game");

while (!windowShouldClose()) {
  beginDrawing();
  clearBackground(Colors.RAYWHITE);
  drawText("Hello, Bloom!", 190, 200, 20, Colors.DARKGRAY);
  endDrawing();
}
```

## Features

- **Simple API** — Functions, not classes. The entire API fits on a cheatsheet.
- **True native** — Compiles to Metal, DirectX 12, Vulkan, and OpenGL via wgpu. No browser, no runtime.
- **Ship everywhere** — macOS, Windows, Linux, iOS, Android from one codebase.
- **Unified 2D/3D** — Shapes, textures, text, 3D models, and audio in one engine.
- **Zero magic** — Explicit game loops, no hidden framework overhead.

## Modules

| Module | Import | Description |
|--------|--------|-------------|
| **Core** | `bloom/core` | Window, game loop, input, timing |
| **Shapes** | `bloom/shapes` | 2D drawing + collision detection |
| **Textures** | `bloom/textures` | Image loading, sprite batching |
| **Text** | `bloom/text` | TTF/OTF font loading and rendering |
| **Audio** | `bloom/audio` | Sound effects + music streaming |
| **Models** | `bloom/models` | 3D model loading (glTF, OBJ), skeletal animation |
| **Math** | `bloom/math` | Vectors, matrices, quaternions, easing |

## Platforms

| Platform | Graphics API |
|----------|-------------|
| macOS | Metal |
| Windows | DirectX 12 |
| Linux | Vulkan / OpenGL |
| iOS | Metal |
| Android | Vulkan / OpenGL ES |

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
  windows/            DirectX 12 + Win32 + XAudio2
  linux/              Vulkan/OpenGL + X11/Wayland + PulseAudio
  android/            Vulkan/OpenGL ES + NativeActivity + AAudio

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

## Skeletal Animation

Bloom supports GPU-accelerated skeletal animation via glTF/GLB models. The pipeline uses 4-bone linear blend skinning with a 128-joint uniform buffer, running entirely on the GPU.

```typescript
import { loadModel, loadModelAnimation, updateModelAnimation, drawModel,
         getTime, Colors } from "bloom";

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

## Links

- [bloomengine.dev](https://bloomengine.dev)
- [Docs](https://bloomengine.dev/docs)
- [Showcase](https://bloomengine.dev/showcase)
- [Brand Guidelines](https://github.com/Bloom-Engine/brand)
