# Web/WASM Target

Bloom games can run in the browser via WebAssembly. The web target uses WebGPU (with WebGL fallback) for rendering and Web Audio API for sound.

## Architecture

```
Game.ts ─(perry --target wasm)──> game WASM  (game logic, base64-embedded
                                      │       in Perry's self-contained HTML)
                                      │ FFI imports ("ffi" namespace)
                                      ▼
                               bloom_glue.js  (bridges both WASM modules)
                                      │
                                      │ wasm-bindgen calls
                                      ▼
                               bloom_web.wasm  (Bloom rendering in WASM)
                                      │
                                      ▼
                              Browser: <canvas> + WebGPU + Web Audio + DOM Events
```

Both game logic and rendering run in WebAssembly. A thin JS glue layer (`native/web/bloom_glue.js`, spliced into Perry's self-contained HTML by `splice_game.py`) bridges the two modules, handles DOM events, asset fetching, and audio output.

## Building

### Prerequisites

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/): `cargo install wasm-pack`
- [Perry compiler](../../perry/perry): built from source
- wasm-opt (optional): `cargo install wasm-opt`

### Quick Build

```bash
./native/web/build.sh path/to/game/main.ts
```

This runs:
1. `wasm-pack build` to compile `native/web/` → `pkg/bloom_web_bg.wasm` + `pkg/bloom_web.js` bindings
2. `wasm-opt -Oz` for binary size optimization (if installed)
3. `perry main.ts --target wasm` to compile game TypeScript → WASM
4. Assembles output directory at `dist/web/`

### Serve Locally

```bash
cd dist/web
python3 -m http.server 8080
# Open http://localhost:8080
```

## Game Loop

Browsers cannot run blocking `while` loops. Use `runGame()` instead:

```typescript
import { initWindow, runGame, clearBackground, drawRect, Colors } from "@bloomengine/engine";

initWindow(800, 600, "My Game");

runGame((dt) => {
  clearBackground(Colors.BLACK);
  drawRect(100, 100, 50, 50, Colors.RED);
});
```

On native, `runGame()` enters a blocking loop. On web, it passes the callback to the JS runtime which drives it via `requestAnimationFrame`.

The traditional `while (!windowShouldClose())` pattern still works on native but is not supported on web.

## Asset Loading

Assets are loaded via synchronous HTTP requests from the game's served directory:

```typescript
const tex = loadTexture("assets/player.png");   // sync fetch from server
const snd = loadSound("assets/jump.wav");        // WAV or OGG
const model = loadModel("assets/scene.glb");     // glTF/GLB
const font = loadFont("assets/font.ttf", 20);   // TTF/OTF
```

Place asset files in your game's `assets/` directory. The build script copies them to the output.

Supported formats:
- **Images**: PNG, JPEG, BMP, TGA
- **Audio**: WAV, OGG (MP3 not supported on web)
- **Models**: glTF, GLB
- **Fonts**: TTF, OTF

## Audio

Audio uses the Web Audio API with the shared Rust AudioMixer:

```typescript
initAudio();
const sound = loadSound("assets/click.wav");
playSound(sound);
```

The JS glue creates an `AudioContext` with a `ScriptProcessorNode` that calls `bloom_audio_mix()` each audio frame. The Rust AudioMixer handles mixing, volume, and spatial audio identically to native.

## File I/O

`writeFile` / `readFile` / `fileExists` use `localStorage` on web (prefixed with `bloom_fs:`):

```typescript
writeFile("save.json", JSON.stringify(gameState));
if (fileExists("save.json")) {
  const data = readFile("save.json");
}
```

## Platform Detection

```typescript
import { getPlatform, Platform } from "@bloomengine/engine";

if (getPlatform() === Platform.WEB) {
  // web-specific code
}
```

## Browser Support

- **Chrome 113+**: WebGPU (best performance)
- **Firefox 141+**: WebGPU
- **Safari**: WebGPU in Technology Preview; WebGL fallback available
- **Edge 113+**: WebGPU

The wgpu backend supports both WebGPU and WebGL. WebGL is used automatically as a fallback on browsers without WebGPU support.

## How It Works

### String Handling

Perry WASM uses NaN-boxed values internally, but Perry's runtime wraps the entire `ffi` namespace with `wrapFfiForI64`, which decodes each NaN-boxed argument to a plain JS value before the glue is called. The glue therefore receives ordinary JS strings and simply routes them to Bloom's `_str` variants via wasm-bindgen — there is no manual NaN-boxing or decoding in the glue.

### Two-Module WASM

Perry compiles game TypeScript to one WASM module. Bloom's Rust backend compiles to a second WASM module via wasm-pack. The JS glue:
1. Loads bloom_web.wasm and extracts all `bloom_*` exports
2. Wraps every export as an FFI import, passing values straight through (Perry's `wrapFfiForI64` has already decoded them)
3. Overrides string- and asset-param functions to route them to their `_str`/`_bytes` variants
4. Boots Perry WASM with these imports under the `"ffi"` namespace

### Shared Code

About two-thirds of Bloom's Rust code is in `native/shared/` — the renderer, audio mixer, text renderer, model loader, scene graph. This code compiles identically for native and WASM. Only the platform layer (~3300 lines across `native/web/src/`: `lib.rs`, `input_ffi.rs`, `material_ffi.rs`, `physics_ffi.rs`, `render_settings.rs`) is web-specific.
