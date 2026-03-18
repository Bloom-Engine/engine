# SURREAL

**A Native TypeScript Game Engine — Powered by Perry**

Technical Specification v0.1 · Skelpo GmbH · March 2026 · DRAFT — INTERNAL

---

## 1. Vision & Philosophy

**Surreal** is a native TypeScript game engine compiled by Perry. It follows the raylib philosophy: a simple, opinionated library of functions — not a visual editor, not a framework with inheritance hierarchies, not an IDE. Just functions you call from your game loop.

The design principles, in order of priority:

- **Simplicity first.** If a feature can't be explained in one sentence, it's too complex. The entire API should fit on a single cheatsheet.
- **Native by default.** Perry compiles TypeScript to native machine code via Cranelift. No browser, no V8, no Electron, no WebGL. Actual GPU calls, actual native windows.
- **2D and 3D unified.** One coordinate system, one camera system, one draw pipeline. 2D is just 3D with an orthographic camera.
- **Zero magic.** No hidden update loops, no component lifecycle hooks, no dependency injection. You write a while-loop. You call draw functions. That's it.
- **Ship everywhere.** macOS, Windows, Linux, iOS, Android. One codebase. Perry Publish handles signing/notarization/distribution.

**The pitch:** *"Write TypeScript. Ship native games to Steam, the App Store, and the Play Store. No browser. No C++. No engine license fees."*

### 1.1 Reference Model: Why Raylib

Raylib is the gold standard for simple-but-powerful game libraries. It has six modules (core, shapes, textures, text, models, audio), fits its entire API on a cheatsheet, and has been ported to 70+ languages. Surreal follows the same modular architecture but is designed from the ground up for TypeScript idioms: strong typing, async/await for asset loading, and object literals for configuration.

### 1.2 Target Games (What Surreal Is For)

- **2D:** platformers, roguelikes, puzzle games, visual novels, tower defense, top-down RPGs
- **Lightweight 3D:** voxel games (Minecraft-style), low-poly adventures, isometric RPGs, racing games
- **Simulation:** farming sims, city builders, idle games, card games
- **Educational & creative:** interactive art, generative visuals, music visualizers, game jam entries

### 1.3 What Surreal Is NOT

- Not an Unreal/Unity competitor — no AAA photorealism, no ray tracing, no massive open worlds
- Not a visual editor — no drag-and-drop scene builder (but Hone could become one later)
- Not a framework — no mandatory game object hierarchy, no ECS baked in, no forced architecture

---

## 2. Architecture

Surreal is organized into seven modules, mirroring raylib's proven structure but adapted for TypeScript. Each module is independently importable.

| Module | Responsibility | Raylib Equivalent |
|---|---|---|
| `surreal/core` | Window creation, game loop, input (keyboard, mouse, gamepad, touch), timing, file I/O | rcore |
| `surreal/shapes` | 2D shape drawing (line, rect, circle, polygon, bezier), 2D collision detection | rshapes |
| `surreal/textures` | Image loading/manipulation (CPU), texture loading/management (GPU), sprite batching | rtextures |
| `surreal/text` | Font loading (TTF/OTF/bitmap), text rendering, text measurement, SDF fonts | rtext |
| `surreal/models` | 3D model loading (glTF, OBJ), mesh generation, skeletal animation, materials, PBR | rmodels |
| `surreal/audio` | Audio device management, sound loading (WAV/OGG/MP3), music streaming, spatial audio | raudio |
| `surreal/math` | Vec2, Vec3, Vec4, Mat4, Quaternion, Ray, BoundingBox, easing functions, RNG | raymath |

### 2.1 The Native Layer (Perry Bindings)

Underneath the TypeScript API, Surreal calls native platform APIs through Perry's FFI system. These are not WebGL calls — they are actual native GPU and OS calls compiled into the binary.

| Platform | Graphics | Windowing | Audio | Input |
|---|---|---|---|---|
| macOS | Metal | AppKit / NSWindow | Core Audio | AppKit Events |
| iOS | Metal | UIKit / UIWindow | Core Audio | UIKit Touch |
| Windows | DirectX 12 / Vulkan | Win32 / HWND | WASAPI / XAudio2 | Win32 Messages / XInput |
| Linux | Vulkan / OpenGL | X11 / Wayland | PulseAudio / ALSA | X11 Events / libinput |
| Android | Vulkan / OpenGL ES | NativeActivity | AAudio / OpenSL ES | MotionEvent |

Perry's compiler emits the correct native calls per target at compile time. The developer writes one TypeScript API; the compiler resolves the platform-specific implementation. This is the key architectural advantage over browser-based engines.

### 2.2 Graphics Abstraction: surreal/gpu

Internally (not exposed to most users), Surreal has a thin GPU abstraction layer that normalizes Metal/DirectX/Vulkan/OpenGL into a common command buffer API, similar to raylib's rlgl module. Power users can access this layer directly for custom shaders and render pipelines.

### 2.3 No Hidden Runtime

There is no garbage collector, no JIT compiler, no event loop. Perry compiles TypeScript to native code with deterministic memory management. Surreal's game loop is a plain while-loop that the developer controls. Frame timing, input polling, and buffer swapping are explicit function calls.

---

## 3. Core API Design

The API is designed to be learnable from a single cheatsheet plus examples. Function names are verbs. Configuration uses object literals. No class hierarchies.

### 3.1 Hello World — Minimal Window

```typescript
import { initWindow, closeWindow, windowShouldClose,
         beginDrawing, endDrawing, clearBackground,
         drawText, Color } from "surreal";

initWindow(800, 450, "My First Surreal Game");

while (!windowShouldClose()) {
  beginDrawing();
  clearBackground(Color.RayWhite);
  drawText("Hello, Surreal!", 190, 200, 20, Color.DarkGray);
  endDrawing();
}

closeWindow();
```

That's a complete, compilable program. `perry build` produces a native binary. `perry publish` packages it for the App Store or Steam.

### 3.2 Hello 3D — Spinning Cube

```typescript
import { initWindow, closeWindow, windowShouldClose,
         beginDrawing, endDrawing, clearBackground,
         beginMode3D, endMode3D, drawCube, drawGrid,
         getDeltaTime, getFPS, drawText,
         Camera3D, Vec3, Color } from "surreal";

initWindow(800, 450, "3D Cube");

const camera: Camera3D = {
  position: Vec3(10, 10, 10),
  target: Vec3(0, 0, 0),
  up: Vec3(0, 1, 0),
  fovy: 45,
  projection: "perspective"
};

let rotation = 0;

while (!windowShouldClose()) {
  rotation += getDeltaTime() * 45;
  beginDrawing();
  clearBackground(Color.White);
  beginMode3D(camera);
    drawCube(Vec3(0, 1, 0), 2, 2, 2, Color.Red, { rotationY: rotation });
    drawGrid(10, 1);
  endMode3D();
  drawText(`FPS: ${getFPS()}`, 10, 10, 20, Color.Gray);
  endDrawing();
}

closeWindow();
```

---

### 3.3 Module API Reference (Summary)

Each module's key functions. The full cheatsheet will be a standalone document, like raylib's.

#### surreal/core — Window, Loop, Input

| Function | Description |
|---|---|
| `initWindow(w, h, title)` | Create native window and GPU context |
| `closeWindow()` | Close window and release resources |
| `windowShouldClose(): boolean` | Check if close requested (X button, ESC, etc.) |
| `setTargetFPS(fps)` | Set target frame rate (0 = unlimited) |
| `getDeltaTime(): number` | Get time since last frame in seconds |
| `getFPS(): number` | Get current frames per second |
| `getScreenWidth/Height(): number` | Get current window dimensions |
| `isKeyPressed/Down/Released(key)` | Check keyboard input state |
| `isMouseButtonPressed/Down/Released(btn)` | Check mouse button state |
| `getMousePosition(): Vec2` | Get mouse position in screen coordinates |
| `isGamepadAvailable(id): boolean` | Check if gamepad is connected |
| `getGamepadAxisValue(id, axis): number` | Get gamepad axis value (-1 to 1) |
| `getTouchPosition(index): Vec2` | Get touch point position (mobile) |
| `getTouchPointCount(): number` | Get number of active touch points |
| `beginDrawing() / endDrawing()` | Begin/end frame drawing (manages buffer swap) |
| `beginMode2D(camera) / endMode2D()` | Begin/end 2D camera mode |
| `beginMode3D(camera) / endMode3D()` | Begin/end 3D camera mode |
| `clearBackground(color)` | Clear screen with specified color |
| `setWindowTitle(title)` | Change window title at runtime |
| `toggleFullscreen()` | Toggle between fullscreen and windowed |
| `setWindowIcon(image)` | Set window icon from Image |

#### surreal/shapes — 2D Drawing & Collision

| Function | Description |
|---|---|
| `drawLine(start, end, color)` | Draw line between two Vec2 points |
| `drawRect(x, y, w, h, color)` | Draw filled rectangle |
| `drawRectLines(x, y, w, h, color)` | Draw rectangle outline |
| `drawCircle(center, radius, color)` | Draw filled circle |
| `drawTriangle(v1, v2, v3, color)` | Draw filled triangle |
| `drawPoly(center, sides, radius, rotation, color)` | Draw regular polygon |
| `drawBezier(start, cp1, cp2, end, color)` | Draw cubic bezier curve |
| `checkCollisionRecs(r1, r2): boolean` | Check collision between two rectangles |
| `checkCollisionCircles(c1, r1, c2, r2): boolean` | Check collision between two circles |
| `checkCollisionPointRec(point, rect): boolean` | Check if point is inside rectangle |
| `getCollisionRec(r1, r2): Rect` | Get overlap rectangle of two colliding rects |

#### surreal/textures — Images & Sprites

| Function | Description |
|---|---|
| `loadImage(path): Image` | Load image from file (CPU memory) |
| `loadTexture(path): Texture` | Load texture from file (GPU memory) |
| `loadTextureFromImage(image): Texture` | Upload image to GPU as texture |
| `unloadTexture(texture)` | Free GPU memory for texture |
| `drawTexture(tex, x, y, tint)` | Draw texture at position |
| `drawTextureRec(tex, source, dest, tint)` | Draw portion of texture (sprite sheets) |
| `drawTexturePro(tex, source, dest, origin, rotation, tint)` | Draw with full transform |
| `imageResize(image, w, h)` | Resize image in CPU memory |
| `imageCrop(image, rect)` | Crop image in CPU memory |
| `imageFlipH/V(image)` | Flip image horizontally/vertically |
| `genTextureMipmaps(texture)` | Generate mipmaps for texture |

#### surreal/text — Fonts & Text

| Function | Description |
|---|---|
| `loadFont(path): Font` | Load TTF/OTF font |
| `loadFontEx(path, size, codepoints): Font` | Load font with specific size and character set |
| `unloadFont(font)` | Free font resources |
| `drawText(text, x, y, size, color)` | Draw text with default font |
| `drawTextEx(font, text, pos, size, spacing, color)` | Draw text with custom font |
| `measureText(text, size): number` | Get text width in pixels |
| `measureTextEx(font, text, size, spacing): Vec2` | Get text dimensions with custom font |

#### surreal/models — 3D Geometry & Models

| Function | Description |
|---|---|
| `drawCube(pos, w, h, d, color, opts?)` | Draw 3D cube |
| `drawSphere(pos, radius, color)` | Draw 3D sphere |
| `drawCylinder(pos, rTop, rBot, h, slices, color)` | Draw 3D cylinder |
| `drawPlane(pos, size, color)` | Draw flat plane |
| `drawGrid(slices, spacing)` | Draw reference grid |
| `drawRay(ray, color)` | Draw 3D ray for debugging |
| `loadModel(path): Model` | Load 3D model (glTF, OBJ) |
| `unloadModel(model)` | Free model resources |
| `drawModel(model, pos, scale, tint)` | Draw 3D model |
| `loadModelAnimation(path): Animation[]` | Load model animations |
| `updateModelAnimation(model, anim, frame)` | Apply animation frame to model |
| `checkCollisionBoxes(b1, b2): boolean` | Check collision between bounding boxes |
| `checkCollisionSpheres(c1, r1, c2, r2): boolean` | Check collision between spheres |
| `getRayCollisionMesh(ray, model): RayHit` | Raycast against model mesh |
| `genMeshCube(w, h, d): Mesh` | Generate cube mesh procedurally |
| `genMeshHeightmap(image, size): Mesh` | Generate terrain mesh from heightmap |

#### surreal/audio — Sound & Music

| Function | Description |
|---|---|
| `initAudioDevice()` | Initialize audio device |
| `closeAudioDevice()` | Close audio device |
| `loadSound(path): Sound` | Load sound effect (fully loads into memory) |
| `playSound(sound)` | Play sound effect |
| `stopSound(sound)` | Stop playing sound |
| `setSoundVolume(sound, vol)` | Set sound volume (0.0 to 1.0) |
| `loadMusic(path): Music` | Load music stream (streams from disk) |
| `playMusic(music)` | Start music stream playback |
| `updateMusic(music)` | Update music stream buffer (call each frame) |
| `setMusicVolume(music, vol)` | Set music volume (0.0 to 1.0) |
| `setMasterVolume(vol)` | Set global audio volume (0.0 to 1.0) |

#### surreal/math — Vectors, Matrices, Utilities

| Function | Description |
|---|---|
| `Vec2(x, y) / Vec3(x, y, z) / Vec4(...)` | Create vector (value types, not classes) |
| `vec2Add, vec2Sub, vec2Scale, vec2Length, ...` | Vector operations (non-mutating) |
| `mat4Identity, mat4Multiply, mat4Rotate, ...` | 4×4 matrix operations |
| `quatFromEuler, quatToMat4, quatSlerp, ...` | Quaternion operations |
| `lerp(a, b, t): number` | Linear interpolation |
| `clamp(val, min, max): number` | Clamp value to range |
| `remap(val, inMin, inMax, outMin, outMax)` | Remap value between ranges |
| `randomInt(min, max): number` | Random integer in range (inclusive) |
| `randomFloat(min, max): number` | Random float in range |
| `easeInOut / easeElastic / easeBounce ...` | Easing functions for animation |

---

## 4. Core Types

Surreal uses TypeScript's type system heavily. All core types are plain interfaces / object literals — no classes with methods, no prototypes.

```typescript
// All vectors are simple objects, not class instances
interface Vec2 { x: number; y: number }
interface Vec3 { x: number; y: number; z: number }
interface Vec4 { x: number; y: number; z: number; w: number }

interface Color { r: number; g: number; b: number; a: number }

interface Rect { x: number; y: number; width: number; height: number }

interface Camera2D {
  offset: Vec2;     // Camera displacement in screen space
  target: Vec2;     // Camera target (what it looks at)
  rotation: number; // Camera rotation in degrees
  zoom: number;     // Camera zoom (scaling factor)
}

interface Camera3D {
  position: Vec3;  // Camera position
  target: Vec3;    // Camera look-at point
  up: Vec3;        // Camera up vector
  fovy: number;    // Field of view (degrees)
  projection: "perspective" | "orthographic";
}

interface Texture {
  readonly id: number;     // GPU texture handle
  readonly width: number;  // Texture width
  readonly height: number; // Texture height
}

interface Model {
  readonly meshCount: number;
  readonly materialCount: number;
  transform: Mat4;
}
```

---

## 5. Build & Distribution Pipeline

Surreal games are built and shipped entirely through Perry's toolchain. No webpack, no bundler, no npm scripts.

### 5.1 Project Structure

```
my-game/
  perry.toml          # Perry project config
  src/
    main.ts           # Entry point
    player.ts         # Game logic
    enemies.ts
  assets/
    sprites/          # PNG, JPG textures
    models/           # glTF, OBJ models
    sounds/           # WAV, OGG, MP3
    fonts/            # TTF, OTF
    levels/           # JSON level data
  build/              # Output (gitignored)
```

### 5.2 perry.toml Configuration

```toml
[package]
name = "my-game"
version = "1.0.0"
entry = "src/main.ts"

[dependencies]
surreal = "0.1"

[assets]
bundle = "assets/"  # Embedded in binary

[build.macos]
bundle_id = "com.mycompany.mygame"
icon = "assets/icon.icns"

[build.windows]
icon = "assets/icon.ico"

[build.ios]
bundle_id = "com.mycompany.mygame"
orientation = "landscape"
```

### 5.3 Build Commands

| Command | Output |
|---|---|
| `perry build` | Native binary for current platform |
| `perry build --target macos-arm64` | macOS Apple Silicon binary |
| `perry build --target windows-x64` | Windows .exe |
| `perry build --target ios` | iOS .app bundle |
| `perry build --release` | Optimized release build |
| `perry run` | Build + run immediately |
| `perry publish` | Package for distribution (App Store, Steam, etc.) |

### 5.4 Asset Bundling

Assets in the configured bundle directory are compiled into the binary at build time. At runtime, `loadTexture("sprites/player.png")` resolves from the embedded bundle. In debug mode, assets are loaded from disk for fast iteration. In release mode, they're embedded for single-binary distribution.

---

## 6. Showcase Starter Games

Each game ships as a complete, well-commented project in the `surreal/examples/` directory. They serve as both learning material and proof that the engine works. Ordered by complexity.

### 6.1 🎾 Pong — "Surreal Pong"

*The "Hello World" of game engines. If you can't make Pong, the engine isn't ready.*

| Aspect | Details |
|---|---|
| **Complexity** | ⭐ Beginner — ~150 lines |
| **Modules used** | core, shapes, text, audio |
| **Teaches** | Game loop, input, collision, score, sound effects |
| **Gameplay** | Classic 2-player pong. Keyboard (W/S + Up/Down) or gamepad. |
| **Key concepts** | `getDeltaTime()` for frame-independent movement, `checkCollisionRecs()` for paddle/ball collision, `drawRect()` for rendering, `playSound()` for hit/score events |

### 6.2 👾 Space Blaster — "Surreal Invaders"

*Classic top-down shooter. Introduces sprites, textures, and entity management.*

| Aspect | Details |
|---|---|
| **Complexity** | ⭐⭐ Intermediate — ~400 lines |
| **Modules used** | core, textures, text, audio, math |
| **Teaches** | Texture loading, sprite drawing, arrays as entity pools, particle effects, scrolling backgrounds |
| **Gameplay** | Ship moves left/right, shoots upward. Waves of enemies descend. Power-ups drop. |
| **Key concepts** | `loadTexture()` + `drawTexturePro()` for sprites, array-based entity management (no ECS needed), basic particle system using `randomFloat()` and easing functions |

### 6.3 🎲 Dungeon Crawl — "Surreal Depths"

*Tile-based roguelike. Introduces tilemaps, camera control, and procedural generation.*

| Aspect | Details |
|---|---|
| **Complexity** | ⭐⭐ Intermediate — ~600 lines |
| **Modules used** | core, textures, text, audio, math |
| **Teaches** | Tilemap rendering from sprite sheets, Camera2D scrolling/zoom, procedural level generation, turn-based logic, JSON level loading |
| **Gameplay** | Top-down dungeon crawler. Move with WASD or tap. Explore rooms, fight monsters, find loot. |
| **Key concepts** | `Camera2D` with target tracking and smooth follow, `drawTextureRec()` for tilemap sprites, `randomInt()` seeded dungeon generation, JSON asset loading for level templates |

### 6.4 🏎️ Kart Racer — "Surreal Karts"

*3D racing game. The first 3D showcase — proves the engine handles real-time 3D.*

| Aspect | Details |
|---|---|
| **Complexity** | ⭐⭐⭐ Advanced — ~800 lines |
| **Modules used** | core, models, textures, text, audio, math |
| **Teaches** | 3D camera, model loading, basic physics (velocity/acceleration), heightmap terrain, 3D collision |
| **Gameplay** | Low-poly kart racing. Single track, 3 AI opponents, lap timer. Gamepad recommended. |
| **Key concepts** | `Camera3D` following player with smooth lerp, `loadModel()` for kart + track glTF assets, `genMeshHeightmap()` for terrain, `getGamepadAxisValue()` for analog steering, simple AI using waypoints |

### 6.5 ⛏️ Voxel Sandbox — "Surreal Craft"

*Minecraft-style voxel world. The ultimate stress test — chunk generation, meshing, infinite terrain.*

| Aspect | Details |
|---|---|
| **Complexity** | ⭐⭐⭐⭐ Expert — ~1500 lines |
| **Modules used** | All seven modules |
| **Teaches** | Chunk-based world, greedy meshing, frustum culling, first-person camera, block placement/destruction, procedural terrain with noise |
| **Gameplay** | First-person voxel sandbox. Place/destroy blocks. Infinite procedurally generated terrain. Day/night cycle. |
| **Key concepts** | Chunk class with greedy mesh generation, `Camera3D` in FPS mode with mouse look, `getRayCollisionMesh()` for block picking, custom shader for block face lighting, music streaming for ambient soundtrack |

### 6.6 🏰 Isometric RPG — "Surreal Quest"

*Diablo-lite isometric RPG. Showcases the full 2D/3D hybrid pipeline.*

| Aspect | Details |
|---|---|
| **Complexity** | ⭐⭐⭐⭐ Expert — ~2000 lines |
| **Modules used** | All seven modules |
| **Teaches** | Isometric projection with Camera3D (orthographic), animated sprites, dialogue system, inventory UI, save/load with JSON, NPC AI state machines |
| **Gameplay** | Click-to-move isometric RPG. One village, one dungeon, three quests, boss fight. |
| **Key concepts** | `Camera3D` with orthographic projection for isometric view, `loadModelAnimation()` for character animations, `drawTextEx()` for dialogue boxes, file I/O for save games, full game state management across multiple screens (title/game/inventory/dialogue) |

---

## 7. Naming & Branding

### 7.1 Name: Surreal

The name "Surreal" works on multiple levels:

- **Wordplay on "Unreal"** — instantly positions it in the game engine space while being clearly distinct
- **Art movement connotation** — surrealism = creative, dreamlike, unexpected. Fits the indie/creative game space perfectly
- **"Sur" prefix** means "above/beyond" in French — beyond the real, beyond what's expected from TypeScript
- **Short, memorable, easy to type** — works as a CLI command: `surreal init`, `surreal build`

**Note:** "Surreal Software" was a defunct game studio (closed 2010, absorbed by WB/Monolith). "SurrealEngine" is a small open-source Unreal 1 reimplementation project. Neither conflict is blocking — both are inactive/niche — but we should verify trademark availability before committing.

### 7.2 Alternative Names (For Consideration)

| Name | Pros | Cons |
|---|---|---|
| **Surreal** | Perfect Unreal wordplay, art movement, memorable | Minor historical overlap with defunct studio |
| **Ethereal** | Evocative, no conflicts, "beyond physical" | Harder to spell, less punchy |
| **Prism** | Short, visual, light/color connotation | Very generic, many products use it |
| **Forge** | Implies creation, strength | Overused in dev tools (Electron Forge, etc.) |
| **Tempest** | Dynamic, powerful, unique | No game engine connotation |
| **Mirage** | Visual illusion, game-appropriate | Connotes "fake/unreliable" |

### 7.3 Import Ergonomics

The package name should be short and work well in import statements:

```typescript
import { initWindow, drawCube } from "surreal";           // ✓ clean
import { initWindow } from "surreal/core";                // ✓ modular
import { Vec3, Mat4 } from "surreal/math";                // ✓ specific
```

---

## 8. Implementation Roadmap

The engine is built in phases, each producing a usable milestone. Each phase unlocks one or more showcase games.

| Phase | Milestone | Unlocks | Est. Effort |
|---|---|---|---|
| **Phase 0: Perry FFI** | Perry can call native C functions (Metal/DX/Vulkan wrappers) from TypeScript | Nothing yet — but this is the prerequisite for everything | Already in progress |
| **Phase 1: Window + 2D** | initWindow, game loop, keyboard/mouse input, 2D shape drawing, basic text, sound effects | Pong, simple puzzle games, game jam entries | 4–6 weeks |
| **Phase 2: Textures + Sprites** | Image loading, texture management, sprite batching, Camera2D, drawTexturePro() | Space Blaster, Dungeon Crawl, any 2D sprite game | 3–4 weeks |
| **Phase 3: 3D Basics** | Camera3D, primitive 3D shapes, grid, basic lighting, model loading (glTF) | Kart Racer, simple 3D games | 6–8 weeks |
| **Phase 4: 3D Advanced** | Skeletal animation, PBR materials, heightmaps, custom shaders, frustum culling | Voxel Sandbox, Isometric RPG | 8–12 weeks |
| **Phase 5: Mobile + Polish** | Touch input, iOS/Android support, gamepad, spatial audio, SDF text, performance profiling | Full cross-platform shipping | 6–8 weeks |
| **Phase 6: Community** | Documentation site, example gallery, starter templates, npm-style package registry for Surreal plugins | Community ecosystem growth | Ongoing |

**Total estimated time to Phase 3 (playable 3D games):** ~4–5 months from Perry FFI readiness.

**Total to Phase 5 (shippable cross-platform):** ~8–12 months.

### 8.1 What Perry Needs First

Surreal is only possible once Perry's compiler supports these features:

- FFI to native C functions (for GPU and OS API calls)
- Static compilation with asset embedding
- Basic struct/interface memory layout (for Vec2/Vec3/Color to be zero-cost)
- Numeric types mapping to native floats/ints (f32/f64/i32)
- while-loop compilation to native code (the game loop)
- Array/buffer handling for vertex data and audio buffers

---

## 9. Competitive Positioning

| Engine | Language | Native? | 3D? | Simplicity | Target |
|---|---|---|---|---|---|
| **Surreal** | TypeScript | ✓ Yes (Perry) | ✓ Yes | ⭐⭐⭐⭐⭐ | Indie / Mid-tier |
| Raylib | C (70+ bindings) | ✓ Yes | ✓ Yes | ⭐⭐⭐⭐⭐ | Education / Indie |
| Unity | C# | ✓ Yes (IL2CPP) | ✓ Yes | ⭐⭐ | Indie → AAA |
| Godot | GDScript / C# | ✓ Yes | ✓ Yes | ⭐⭐⭐ | Indie / Mid-tier |
| Unreal | C++ / Blueprints | ✓ Yes | ✓ Yes | ⭐ | AAA |
| Babylon.js | TypeScript | ✗ Browser only | ✓ Yes | ⭐⭐⭐ | Web apps / games |
| Phaser | JS/TS | ✗ Browser only | ✗ 2D only | ⭐⭐⭐⭐ | Web / casual |
| Excalibur | TypeScript | ✗ Browser only | ✗ 2D only | ⭐⭐⭐⭐ | Web / hobby |
| LÖVE | Lua | ✓ Yes | ✗ 2D only | ⭐⭐⭐⭐⭐ | Indie / jams |
| libGDX | Java/Kotlin | ✓ Yes (JVM) | ✓ Yes | ⭐⭐⭐ | Indie / mobile |

**Surreal's unique position:** The only native game engine for TypeScript. Combines the simplicity of raylib/LÖVE with the developer familiarity of TypeScript and the native performance of compiled code. No other engine occupies this exact space.

### 9.1 Strategic Value for Perry

- The most demanding proof that Perry's compiler produces real, performant native code
- Taps into the largest developer community in the world (TypeScript/JavaScript developers)
- Creates a self-reinforcing ecosystem: developers use Surreal → learn Perry → build other Perry apps
- Every Surreal game shipped to Steam/App Store is a Perry success story
- Game jams become viral marketing events for Perry ("Built with Surreal/Perry" splash screens)

---

## 10. Open Questions

- **Shader language:** Custom TypeScript-like shading language? Or support GLSL/HLSL/MSL pass-through? Or a simplified subset?
- **Physics:** Built-in simple physics (like raylib) or optional integration with Rapier/Box2D via Perry FFI?
- **Networking:** Out of scope for v1? Or include basic TCP/UDP for multiplayer prototyping?
- **ECS:** Provide an optional ECS module (surreal/ecs) or leave it to community packages?
- **UI system:** Immediate-mode UI for game menus (like raygui) or leave to community?
- **Editor integration:** Should Hone get a Surreal scene preview panel? Live-reload during development?
- **License:** MIT? Apache 2.0? zlib (like raylib)? No revenue share, ever.
- **Name finalization:** Surreal vs alternatives. Trademark search needed.
